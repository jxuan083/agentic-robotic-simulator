"""Shared EE target with constrained downward-grasp IK, used by keyboard and mouse input.

Pure numpy: no Isaac imports, so the behaviour (limits, rejection reasons, boundary sliding, reset) is testable
without the simulator. Every input source only proposes target changes; this class decides whether the new target
is accepted and produces the commanded arm joint positions.

Rejection reasons are kept apart so the operator (and the log) can tell what blocked a move:
- workspace_limit : the target hit the configured workspace box (clamped; not an error, e.g. the grasp-height floor)
- joint_limit     : IK ended with a joint pinned at its limit and the pose out of tolerance
- ik_no_solution  : IK could not reach the pose (out of reach / orientation) with no joint at a limit
- joint_jump      : IK found a solution but a joint would move more than max_joint_step in one update
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field

import numpy as np

from agentic_sim.config import v
from agentic_sim.kinematics import ARM_JOINTS, So101Chain

STATUS_OK = "ok"
STATUS_LIMIT = "workspace_limit"
STATUS_JOINT_LIMIT = "joint_limit"
STATUS_IK = "ik_no_solution"
STATUS_JUMP = "joint_jump"
REJECT_STATUSES = (STATUS_JOINT_LIMIT, STATUS_IK, STATUS_JUMP)
REASONS = (STATUS_LIMIT,) + REJECT_STATUSES

AT_LIMIT_TOL = 1e-3      # rad: a joint this close to its limit counts as pinned
BISECT_ITERS = 4         # boundary search resolution: 1/16 of the step


@dataclass
class TeleopParams:
    home_ee: np.ndarray
    ws_lo: np.ndarray
    ws_hi: np.ndarray
    gripper_open: float
    gripper_closed: float
    gripper_speed: float
    linear_speed: float
    yaw_speed: float
    wheel_step: float
    fine_wheel_step: float
    fine_drag_scale: float
    max_drag_step: float
    max_joint_step: float
    joint_rate_limit: float
    yaw_offset_limit: float
    dt_max: float
    max_tilt_deg: float = 4.0
    # finger opening direction at zero operator offset, relative to the heading from the pan axis:
    # pi/2 = tangential (lab_current_minimal, wrist roll about 93 deg); 0 = radial (lab scenes: wrist roll about 0
    # as the lab operators hold it, wrist camera above the gripper and its image upright)
    finger_heading: float = math.pi / 2
    start_q: np.ndarray | None = None   # arm joints at reset (lab rest pose); None = start at home
    homing_rate: float = 0.8            # rad/s from start_q to home (slower than the teleop rate limit)

    @classmethod
    def from_cfg(cls, c: dict) -> "TeleopParams":
        ws, m = c["workspace"], c["mouse"]
        return cls(
            home_ee=np.array(c["home_ee"], float),
            ws_lo=np.array([ws["x"][0], ws["y"][0], ws["z"][0]], float),
            ws_hi=np.array([ws["x"][1], ws["y"][1], ws["z"][1]], float),
            gripper_open=float(c["gripper"]["open"]),
            gripper_closed=float(c["gripper"]["closed"]),
            gripper_speed=float(c["gripper"]["speed"]),
            linear_speed=float(c["keyboard"]["linear_speed"]),
            yaw_speed=float(c["keyboard"]["yaw_speed"]),
            wheel_step=float(m["wheel_step"]),
            fine_wheel_step=float(m["fine_wheel_step"]),
            fine_drag_scale=float(m["fine_drag_scale"]),
            max_drag_step=float(m["max_drag_step"]),
            max_joint_step=float(c["ik"]["max_joint_step"]),
            joint_rate_limit=float(c["ik"]["joint_rate_limit"]),
            yaw_offset_limit=float(c["ik"]["yaw_offset_limit"]),
            dt_max=float(c["dt_max"]),
            max_tilt_deg=float(c["ik"].get("max_tilt_deg", 4.0)),
        )


@dataclass
class TargetState:
    pos: np.ndarray
    yaw_offset: float
    q_goal: np.ndarray          # IK solution for the accepted target
    q_cmd: np.ndarray           # rate-limited arm command sent to the robot
    gripper: float
    status: str = STATUS_OK
    message: str = ""
    source: str = "none"
    rejected: int = 0           # rejected update attempts this session
    reject_counts: dict = field(default_factory=lambda: {r: 0 for r in REASONS})
    yaw_goal: float | None = None      # set by the Yaw +/- buttons, approached at yaw_speed
    gripper_goal: float | None = None  # set by Open / Close, approached at gripper_speed
    session: int = 0
    info: dict = field(default_factory=dict)


@dataclass
class Evaluation:
    pos: np.ndarray
    yaw_offset: float
    q: np.ndarray
    ok: bool
    status: str
    message: str
    info: dict


class EETargetController:
    def __init__(self, chain: So101Chain, params: TeleopParams):
        self.chain = chain
        self.p = params
        self.pan = chain.pan_origin_mount()
        self.state: TargetState | None = None
        self.q_home = self._solve_home()
        self.reset()

    # ---- helpers -------------------------------------------------------------
    def yaw_for(self, pos, yaw_offset) -> float:
        """Finger opening direction: the scene's neutral heading relative to the pan circle (tangential by default,
        radial in lab scenes), plus the operator's offset."""
        return float(np.arctan2(pos[1] - self.pan[1], pos[0] - self.pan[0]) + self.p.finger_heading + yaw_offset)

    def _solve_home(self) -> np.ndarray:
        pos = self.p.home_ee
        a = np.arctan2(pos[1] - self.pan[1], pos[0] - self.pan[0]) - np.pi / 2
        for seed in ([a, 0.0, 0.0, 1.5, 0.0], [a, -1.0, 1.0, 1.0, 0.0], [a, 0.5, -0.5, 1.5, 0.0]):
            q, ok, info = self.chain.solve_ik(pos, self.yaw_for(pos, 0.0), np.array(seed), iters=300)
            if ok:
                return q
        raise RuntimeError(f"home EE {pos.tolist()} is not reachable: {info}")

    def ee_pos(self, q) -> np.ndarray:
        return self.chain.ee_pose_mount(q)[:3, 3]

    # ---- lifecycle -----------------------------------------------------------
    def reset(self):
        session = 0 if self.state is None else self.state.session + 1
        q = self.q_home.copy()
        # with a start pose (lab rest) the arm starts there and the command moves to home at homing_rate
        q_cmd = q.copy() if self.p.start_q is None else np.array(self.p.start_q, float)
        self.state = TargetState(
            pos=self.ee_pos(q), yaw_offset=0.0, q_goal=q.copy(), q_cmd=q_cmd, gripper=self.p.gripper_open,
            session=session,
        )
        self.homing = self.p.start_q is not None

    def home_joint_space(self, source: str):
        """Return to the home joint angles in joint space at homing_rate, without IK: the way out of a pose where
        every nearby EE target is rejected (the IK seeded there finds no solution)."""
        s = self.state
        s.q_goal, s.pos, s.yaw_offset = self.q_home.copy(), self.ee_pos(self.q_home), 0.0
        s.status, s.message, s.source = STATUS_OK, "", source
        self.homing = True

    # ---- evaluation (no state change) ------------------------------------------------
    def evaluate(self, pos, yaw_offset: float) -> Evaluation:
        """Clamp to the workspace, solve IK from the current solution and classify the outcome."""
        s = self.state
        pos = np.asarray(pos, float)
        cpos = np.clip(pos, self.p.ws_lo, self.p.ws_hi)
        clamped = bool(np.any(np.abs(cpos - pos) > 1e-9))
        yaw_offset = float(np.clip(yaw_offset, -self.p.yaw_offset_limit, self.p.yaw_offset_limit))
        q, ok, info = self.chain.solve_ik_tilt(cpos, self.yaw_for(cpos, yaw_offset), s.q_goal, self.p.max_tilt_deg)
        if ok:
            dq = np.abs(q - s.q_goal)
            j = int(np.argmax(dq))
            if dq[j] > self.p.max_joint_step:
                info = dict(info, joint_jump=float(dq[j]))
                return Evaluation(cpos, yaw_offset, q, False, STATUS_JUMP,
                                  f"joint jump {ARM_JOINTS[j]} {dq[j]:.2f} rad", info)
            if clamped:
                axes = [f"{n} {'min' if pos[i] < cpos[i] else 'max'}" for i, n in enumerate("xyz")
                        if abs(cpos[i] - pos[i]) > 1e-9]
                return Evaluation(cpos, yaw_offset, q, True, STATUS_LIMIT, "workspace " + ", ".join(axes), info)
            return Evaluation(cpos, yaw_offset, q, True, STATUS_OK, "", info)
        pinned = [ARM_JOINTS[i] for i in range(len(q))
                  if q[i] - self.chain.lower[i] < AT_LIMIT_TOL or self.chain.upper[i] - q[i] < AT_LIMIT_TOL]
        if pinned:
            return Evaluation(cpos, yaw_offset, q, False, STATUS_JOINT_LIMIT, "joint limit " + ",".join(pinned), info)
        return Evaluation(cpos, yaw_offset, q, False, STATUS_IK,
                          f"no IK solution (pos {info['pos_err'] * 1000:.1f} mm, tilt {info['tilt_deg']:.1f} deg)",
                          info)

    def _apply(self, e: Evaluation, source: str):
        s = self.state
        s.pos, s.yaw_offset, s.q_goal, s.source = e.pos, e.yaw_offset, e.q, source
        s.status, s.message, s.info = e.status, e.message, e.info
        if e.status == STATUS_LIMIT:
            s.reject_counts[STATUS_LIMIT] += 1

    def _reject(self, e: Evaluation, source: str):
        s = self.state
        s.status, s.message, s.source, s.info = e.status, e.message, source, e.info
        s.rejected += 1
        s.reject_counts[e.status] += 1

    @property
    def blocked(self) -> bool:
        return self.state.status in REJECT_STATUSES

    # ---- target updates ------------------------------------------------------
    def propose(self, pos, yaw_offset: float, source: str) -> bool:
        """Accept the new target whole or reject it whole (keyboard behaviour). Rejection keeps the last target."""
        s = self.state
        pos = np.asarray(pos, float)
        cpos = np.clip(pos, self.p.ws_lo, self.p.ws_hi)
        if np.allclose(cpos, s.pos) and abs(yaw_offset - s.yaw_offset) < 1e-9:
            if np.any(np.abs(cpos - pos) > 1e-9):  # pushing further into the workspace box: nothing to solve
                s.status, s.source = STATUS_LIMIT, source
                s.message = "workspace " + ", ".join(f"{n} {'min' if pos[i] < cpos[i] else 'max'}"
                                                     for i, n in enumerate("xyz") if abs(cpos[i] - pos[i]) > 1e-9)
            return False
        e = self.evaluate(pos, yaw_offset)
        if e.ok:
            self._apply(e, source)
            return True
        self._reject(e, source)
        return False

    def move_by(self, delta, source: str) -> float:
        """Relative move (mouse drag / wheel). Returns the fraction of the requested step that was applied.

        If the full step is rejected, the farthest reachable point along the step is taken; for horizontal steps each
        axis is also tried alone, so the target slides along the reach boundary instead of sticking. The part of the
        step that could not be applied is dropped (never accumulated), so moving back responds immediately.
        """
        s = self.state
        d = np.asarray(delta, float)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return 0.0
        if n > self.p.max_drag_step:  # one event can never become a large jump (e.g. a grazing view ray)
            d = d / n * self.p.max_drag_step
            n = self.p.max_drag_step
        cd = np.clip(s.pos + d, self.p.ws_lo, self.p.ws_hi) - s.pos
        if np.linalg.norm(cd) < 1e-9:  # pushing into the workspace box only
            self.propose(s.pos + d, s.yaw_offset, source)
            return 0.0
        full = self.evaluate(s.pos + d, s.yaw_offset)
        if full.ok:
            self._apply(full, source)
            return float(np.dot(cd, d)) / (n * n)
        best_e, best_gain = None, 0.0

        def gain_of(e):  # progress along the requested direction
            return float(np.dot(e.pos - s.pos, d)) / (n * n)

        # 1) horizontal step: try each axis alone (sliding along the boundary usually succeeds in one solve)
        if abs(d[2]) < 1e-12 and abs(d[0]) > 1e-12 and abs(d[1]) > 1e-12:
            for c in (np.array([d[0], 0.0, 0.0]), np.array([0.0, d[1], 0.0])):
                e = self.evaluate(s.pos + c, s.yaw_offset)
                if e.ok and gain_of(e) > best_gain + 1e-9:
                    best_e, best_gain = e, gain_of(e)
        # 2) otherwise approach the boundary along the requested direction
        if best_e is None:
            lo, hi = 0.0, 1.0
            for _ in range(BISECT_ITERS):
                mid = 0.5 * (lo + hi)
                e = self.evaluate(s.pos + d * mid, s.yaw_offset)
                if e.ok:
                    lo, best_e = mid, e
                else:
                    hi = mid
            best_gain = gain_of(best_e) if best_e is not None else 0.0
        self._reject(full, source)  # the requested step was blocked: count it, keep the reason visible
        if best_e is not None:
            self._apply(best_e, source)
            s.status, s.message, s.info = full.status, full.message, full.info
        return best_gain

    def step_velocity(self, lin_dir, yaw_dir: float, dt: float, source: str) -> bool:
        """Keyboard: lin_dir / yaw_dir are -1, 0, +1 per axis; increments scale with elapsed time, not frames."""
        dt = float(min(max(dt, 0.0), self.p.dt_max))
        lin_dir = np.asarray(lin_dir, float)
        if dt == 0.0 or (not np.any(lin_dir) and yaw_dir == 0):
            return False
        s = self.state
        return self.propose(s.pos + lin_dir * self.p.linear_speed * dt, s.yaw_offset + yaw_dir * self.p.yaw_speed * dt,
                            source)

    def wheel(self, notches: float, fine: bool = False) -> float:
        step = self.p.fine_wheel_step if fine else self.p.wheel_step
        return self.move_by([0.0, 0.0, notches * step], "mouse")

    def drag(self, delta_xy, fine: bool = False) -> float:
        k = self.p.fine_drag_scale if fine else 1.0
        return self.move_by([delta_xy[0] * k, delta_xy[1] * k, 0.0], "mouse")

    def nudge_yaw(self, delta: float, source: str) -> None:
        """Button yaw step: set a goal that step_goals approaches at yaw_speed (no single large IK jump)."""
        s = self.state
        base = s.yaw_offset if s.yaw_goal is None else s.yaw_goal
        s.yaw_goal = float(np.clip(base + delta, -self.p.yaw_offset_limit, self.p.yaw_offset_limit))
        s.source = source

    def clear_yaw_goal(self):
        self.state.yaw_goal = None

    def step_goals(self, dt: float) -> None:
        """Advance the button-driven yaw goal at yaw_speed; drop the goal if the next step is rejected."""
        s = self.state
        if s.yaw_goal is None:
            return
        dt = float(min(max(dt, 0.0), self.p.dt_max))
        d_yaw = s.yaw_goal - s.yaw_offset
        if abs(d_yaw) < 1e-6:
            s.yaw_goal = None
            return
        step = float(np.clip(d_yaw, -self.p.yaw_speed * dt, self.p.yaw_speed * dt))
        if not self.propose(s.pos, s.yaw_offset + step, "mouse") and self.blocked:
            s.yaw_goal = None

    # ---- gripper ---------------------------------------------------------------
    def step_gripper(self, direction: int, dt: float, source: str):
        if direction == 0:
            return
        dt = float(min(max(dt, 0.0), self.p.dt_max))
        g = self.state.gripper + direction * self.p.gripper_speed * dt
        self.state.gripper = float(np.clip(g, self.p.gripper_closed, self.p.gripper_open))
        self.state.gripper_goal = None  # manual O / C overrides a pending button goal
        self.state.source = source

    def set_gripper(self, open_: bool, source: str):
        """Open / Close button: the jaw command moves to the end stop at gripper_speed (no step, no slam)."""
        self.state.gripper_goal = self.p.gripper_open if open_ else self.p.gripper_closed
        self.state.source = source

    def clear_gripper_goal(self):
        self.state.gripper_goal = None

    # ---- command -----------------------------------------------------------------
    def joint_command(self, dt: float) -> np.ndarray:
        """Rate-limited arm joint command toward the accepted IK solution (no jumps even after large changes)."""
        s = self.state
        dt = float(min(max(dt, 0.0), self.p.dt_max))
        rate = min(self.p.joint_rate_limit, self.p.homing_rate) if self.homing else self.p.joint_rate_limit
        lim = rate * dt
        s.q_cmd = s.q_cmd + np.clip(s.q_goal - s.q_cmd, -lim, lim)
        if self.homing and float(np.max(np.abs(s.q_goal - s.q_cmd))) < 1e-9:
            self.homing = False
        if s.gripper_goal is not None:
            g_lim = self.p.gripper_speed * dt
            s.gripper = float(s.gripper + np.clip(s.gripper_goal - s.gripper, -g_lim, g_lim))
            if abs(s.gripper_goal - s.gripper) < 1e-9:
                s.gripper_goal = None
        return s.q_cmd.copy()


def load_controller(chain: So101Chain, teleop_cfg: dict, base_pos=(0.0, 0.0, 0.0), scene: dict | None = None
                    ) -> EETargetController:
    """base_pos: the robot base link origin in the mount frame (scene robot.base_pos).
    scene: lab scenes set robot.grasp_heading_deg (neutral finger direction) and robot.start_pose_lerobot (arm
    joints at reset, LeRobot readings); without them the defaults (tangential fingers, start at home) apply."""
    chain.ee_offset = np.array(v(teleop_cfg["ee_offset_in_gripper_link"]), float)
    chain.base_pos = np.array(base_pos, float)
    p = TeleopParams.from_cfg(teleop_cfg)
    rob = (scene or {}).get("robot") or {}
    if "grasp_heading_deg" in rob:
        p.finger_heading = math.radians(float(v(rob["grasp_heading_deg"])))
    if "start_pose_lerobot" in rob:
        from agentic_sim.real_joints import lerobot_to_shared
        arm, _ = lerobot_to_shared(v(rob["start_pose_lerobot"]))
        p.start_q = np.clip(arm, chain.lower, chain.upper)
    return EETargetController(chain, p)
