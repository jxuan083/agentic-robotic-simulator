"""Cancellable pick / place skills driven through the shared EE controller (pure Python + numpy).

A skill is a phased state machine stepped once per control frame with the latest observation. It only proposes EE
targets and gripper goals to the EETargetController, so IK, joint limits and rate limits are the same as for the
operator. Grasping is physical contact only.

Cancel (executor.cancel -> skill.cancel(reason)) is taken at the next frame and handled by phase:
  pick   rise / approach / descend / close   open the gripper, go straight up to the safe height -> cancelled, empty
         lift                                stop where it is                                    -> cancelled, holding
  place  rise / transport                    stop where it is                                    -> cancelled, holding
         lower                               go straight back up to the safe height              -> cancelled, holding
         release / retreat / verify          not cancellable (the block has been let go)
  hold   any                                 stops at once
  home   any                                 stops at once
The abort motion itself is not cancellable. A skill counts as finished only once the arm has arrived (target reached
and joints settled), not when the command was sent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from agentic_sim.scene_layout import block_in_tray


@dataclass
class Obs:
    """What the executor sees in one frame (simulation truth in this issue)."""
    t: float                    # simulated time, s
    ee: np.ndarray              # actual EE position (FK of the measured joints)
    qd_max: float               # largest measured arm joint speed, rad/s
    jaw: float                  # measured jaw angle, rad
    blocks: list                # block centre positions
    block_yaw: list             # block yaw about +Z, rad
    block_speed: list           # block linear speed, m/s


@dataclass
class World:
    """Static scene facts the skills need."""
    trays: dict                 # 'storage' / 'placement' -> tray interior (scene_layout.trays entry)
    edge: float
    colors: list                # colour name per block
    init_pos: list              # initial block centres
    home: np.ndarray
    slots: list                 # placement slot xy


@dataclass
class SkillResult:
    skill: str
    params: dict
    status: str                 # succeeded / failed / uncertain / cancelled
    failure_reason: str = ""
    duration: float = 0.0
    phases: dict = field(default_factory=dict)   # phase -> seconds
    evidence: dict = field(default_factory=dict)
    holding: int | None = None  # block in the gripper afterwards

    def to_dict(self) -> dict:
        return {"skill": self.skill, "params": self.params, "status": self.status,
                "failure_reason": self.failure_reason, "duration": round(self.duration, 3),
                "phases": {k: round(v, 3) for k, v in self.phases.items()}, "evidence": self.evidence,
                "holding": self.holding}


class _Cancel(Exception):
    pass


class _Fail(Exception):
    pass


def wrap_quarter(a: float) -> float:
    """Angle wrapped to [-pi/4, pi/4) (a cube looks the same every quarter turn)."""
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


class Motion:
    """Moves the controller's EE target along straight segments at a set speed, one step per frame."""

    def __init__(self, ctrl, cfg: dict, dt: float):
        self.ctrl, self.dt = ctrl, dt
        m = cfg["motion"]
        self.speed, self.yaw_speed = float(m["linear_speed"]), float(m["yaw_speed"])
        self.settle_qd = float(m["settle_qd"])
        self.blocked_limit = int(m["blocked_frames"])
        self.blocked = 0

    def step_toward(self, goal, yaw_off: float) -> bool:
        """One target step toward (goal, yaw_off); True once the target is there."""
        s = self.ctrl.state
        goal = np.asarray(goal, float)
        d = goal - s.pos
        n = float(np.linalg.norm(d))
        dy = yaw_off - s.yaw_offset
        if n < 1e-6 and abs(dy) < 1e-6:
            self.blocked = 0
            return True
        step = d if n <= self.speed * self.dt else d / n * self.speed * self.dt
        ystep = float(np.clip(dy, -self.yaw_speed * self.dt, self.yaw_speed * self.dt))
        if self.ctrl.propose(s.pos + step, s.yaw_offset + ystep, "task") or not self.ctrl.blocked:
            self.blocked = 0
        else:
            self.blocked += 1
            if self.blocked >= self.blocked_limit:
                raise _Fail(f"unreachable: {s.status} {s.message}")
        return False

    def arrived(self, obs: Obs) -> bool:
        """The rate-limited joint command has reached the IK solution and the arm has settled. The measured EE is
        not compared with the target: the engines' gains leave a different static sag (Isaac several mm)."""
        s = self.ctrl.state
        return float(np.max(np.abs(s.q_cmd - s.q_goal))) < 1e-4 and obs.qd_max < self.settle_qd


class Skill:
    name = "skill"
    cancel_phases: tuple = ()   # phases in which a cancel is taken (others finish first)

    def __init__(self, ctx, **params):
        """ctx: the executor (ctrl, world, cfg, motion, emit)."""
        self.ctx, self.params = ctx, params
        self.ctrl, self.world, self.cfg, self.motion = ctx.ctrl, ctx.world, ctx.cfg, ctx.motion
        self.status = "running"
        self.phase = ""
        self.cancel_reason: str | None = None
        self.revision = ctx.ledger.revision
        self.result: SkillResult | None = None
        self.holding: int | None = params.get("holding")
        self.obs: Obs | None = None
        self.t0 = self.phase_t = None
        self.phase_time: dict = {}
        self.evidence: dict = {}
        self._gen = None

    # ---- driving --------------------------------------------------------------------------
    def step(self, obs: Obs):
        self.obs = obs
        if self.t0 is None:
            self.t0 = obs.t
            self._gen = self._run()
        if self.status != "running":
            return
        try:
            next(self._gen)
        except StopIteration:
            pass

    def cancel(self, reason: str, revision: int) -> bool:
        """Request a cancel; False if the current phase cannot be cancelled (the skill finishes as planned)."""
        if self.status != "running" or self.phase not in self.cancel_phases:
            return False
        self.cancel_reason = reason
        self.revision = revision      # the abort motion belongs to the new plan
        return True

    @property
    def cancellable(self) -> bool:
        return self.status == "running" and self.phase in self.cancel_phases

    def _run(self):
        try:
            yield from self.body()
            if self.status == "running":
                self._finish("succeeded")
        except _Cancel:
            self.ctx.emit("cancel_taken", skill=self.name, phase=self.phase, reason=self.cancel_reason,
                          block=self.params.get("block"))
            yield from self.on_cancel()
            self._finish("cancelled", self.cancel_reason or "")
        except _Fail as f:
            self._finish("failed", str(f))

    def _finish(self, status: str, reason: str = ""):
        if self.phase:
            self.phase_time[self.phase] = self.phase_time.get(self.phase, 0.0) + (self.obs.t - self.phase_t)
        self.status = status
        self.result = SkillResult(self.name, dict(self.params), status, reason, self.obs.t - self.t0,
                                  dict(self.phase_time), dict(self.evidence), self.holding)

    def set_phase(self, phase: str):
        if self.phase:
            self.phase_time[self.phase] = self.phase_time.get(self.phase, 0.0) + (self.obs.t - self.phase_t)
        self.phase, self.phase_t = phase, self.obs.t
        self.ctx.emit("phase", skill=self.name, phase=phase, block=self.params.get("block"),
                      color=self.params.get("color"), nth=self.params.get("nth"))

    # ---- building blocks (generators; one yield = one frame) --------------------------------
    def _check_cancel(self):
        if self.cancel_reason is not None and self.phase in self.cancel_phases:
            raise _Cancel()

    def _check_drop(self):
        b = self.holding
        if b is None:
            return
        if float(np.linalg.norm(self.obs.blocks[b] - self.obs.ee)) > float(self.cfg["grasp_check"]["max_block_ee"]):
            self.evidence["jaw_at_drop"] = round(float(self.obs.jaw), 4)
            self.ctx.emit("dropped", block=b, phase=self.phase)
            self.holding = None
            raise _Fail("dropped")

    def move(self, goal, yaw_off: float | None = None, cancellable: bool = True):
        yaw_off = self.ctrl.state.yaw_offset if yaw_off is None else yaw_off
        t_end = self.obs.t + float(self.cfg["motion"]["move_timeout_s"])
        while True:
            if cancellable:
                self._check_cancel()
            self._check_drop()
            if self.motion.step_toward(goal, yaw_off) and self.motion.arrived(self.obs):
                return
            if self.obs.t > t_end:
                raise _Fail(f"move timeout to {np.round(goal, 3).tolist()}")
            yield

    def wait(self, seconds: float, cancellable: bool = True):
        t_end = self.obs.t + seconds
        while self.obs.t < t_end:
            if cancellable:
                self._check_cancel()
            self._check_drop()
            yield

    def gripper(self, open_: bool, cancellable: bool = True):
        self.ctrl.set_gripper(open_, "task")
        p = self.ctrl.p
        goal = p.gripper_open if open_ else p.gripper_closed
        while abs(self.ctrl.state.gripper - goal) > 1e-6:
            if cancellable:
                self._check_cancel()
            if self.ctrl.state.gripper_goal is None:   # cleared by a pause: ask again
                self.ctrl.set_gripper(open_, "task")
            yield
        yield from self.wait(float(self.cfg["gripper"]["open_wait_s" if open_ else "close_wait_s"]), cancellable)

    def up_to_safe(self, cancellable: bool):
        s = self.ctrl.state
        safe = float(self.cfg["motion"]["safe_z"])
        if s.pos[2] < safe - 1e-4:
            yield from self.move([s.pos[0], s.pos[1], safe], cancellable=cancellable)

    def hold_here(self):
        """Stop: keep the current target (the rate-limited command settles on it)."""
        s = self.ctrl.state
        s.yaw_goal = None
        while not self.motion.arrived(self.obs):
            self._check_drop()
            yield

    # ---- grasp geometry -----------------------------------------------------------------------
    def yaw_offset_for(self, xy, block_yaw: float) -> float:
        """Operator yaw offset that lines the fingers up with the block faces at xy (smallest turn)."""
        base = self.ctrl.yaw_for(np.array([xy[0], xy[1], 0.0]), 0.0)
        return float(np.clip(wrap_quarter(block_yaw - base), -self.ctrl.p.yaw_offset_limit,
                             self.ctrl.p.yaw_offset_limit))

    def grasp_xy(self, block_xy, yaw_off: float) -> np.ndarray:
        """Grasp point: the block centre shifted toward the fixed finger by the configured gap."""
        y = self.ctrl.yaw_for(np.array([block_xy[0], block_xy[1], 0.0]), yaw_off)
        u = np.array([math.cos(y), math.sin(y)])
        return np.asarray(block_xy[:2], float) - float(self.cfg["motion"]["fixed_finger_gap"]) * u

    def body(self):
        raise NotImplementedError
        yield  # noqa: unreachable  (makes body a generator in subclasses' eyes)

    def on_cancel(self):
        yield from self.hold_here()


class Pick(Skill):
    """Pick block `block` from wherever it lies: approach at the safe height, descend, close, lift.

    With a camera (executor.camera) and an observed `target` pose ({xy, yaw}) the approach goes over the observed
    pose and the wrist aligns the grasp before the descent (agentic_sim.task.camera.align_target); otherwise the
    block's simulated pose is used. The lift check reads the simulated block (sim_truth) in both cases."""
    name = "pick"
    cancel_phases = ("rise", "approach", "align", "descend", "close", "lift")

    def body(self):
        b = self.params["block"]
        m = self.cfg["motion"]
        safe = float(m["safe_z"])
        self.set_phase("rise")
        yield from self.up_to_safe(cancellable=True)
        target = self.params.get("target")
        camera = target is not None and getattr(self.ctx, "camera", None) is not None
        if camera:
            floor = self.world.trays["storage"]["floor_top"]
            bp = np.array([target["xy"][0], target["xy"][1], floor + self.world.edge / 2])
            byaw = float(target["yaw"])
        else:
            bp, byaw = self.obs.blocks[b].copy(), float(self.obs.block_yaw[b])
        off = self.yaw_offset_for(bp, byaw)
        g = self.grasp_xy(bp, off)
        self.evidence["block_start"] = [round(float(x), 5) for x in bp]
        self.evidence["target_source"] = target.get("source") if camera else "sim_truth"
        self.set_phase("approach")
        yield from self.move([g[0], g[1], safe], off)
        if camera:
            from agentic_sim.task.camera import align_target
            self.set_phase("align")
            pose = yield from align_target(self, target, self.params.get("color"))
            bp[:2] = pose["xy"]
            off = self.yaw_offset_for(bp, float(pose["yaw"]))
            g = self.grasp_xy(bp, off)
            self.evidence["grasp_pose"] = {"xy": [round(float(x), 5) for x in bp[:2]],
                                           "yaw_deg_mod_90": round(math.degrees(float(pose["yaw"])) % 90, 2)}
        self.set_phase("descend")
        yield from self.move([g[0], g[1], bp[2] + float(m["grasp_dz"])], off)
        self.set_phase("close")
        yield from self.gripper(False)
        self.set_phase("lift")
        z0 = float(self.obs.blocks[b][2]) if camera else float(bp[2])
        self.holding = b          # from here a cancel keeps the block in the gripper
        self.ctx.on_grasp(b)
        yield from self.move([g[0], g[1], safe], off)
        bz = float(self.obs.blocks[b][2])
        near = float(np.linalg.norm(self.obs.blocks[b] - self.obs.ee))
        self.evidence.update({"block_lift_dz": round(bz - z0, 4), "block_ee_dist": round(near, 4),
                              "jaw": round(float(self.obs.jaw), 4), "observation": "sim_truth"})
        if bz - z0 < float(self.cfg["grasp_check"]["lifted_dz"]) or near > float(self.cfg["grasp_check"]["max_block_ee"]):
            self.holding = None
            self.ctx.on_release(b)
            yield from self.gripper(True, cancellable=False)
            raise _Fail("grasp_missed")

    def on_cancel(self):
        if self.phase == "lift" and self.holding is not None:
            yield from self.hold_here()
            return
        if self.holding is not None:   # closing had started: let go before going up
            self.ctx.on_release(self.holding)
            self.holding = None
        if self.ctrl.state.gripper < self.ctrl.p.gripper_open - 1e-6:
            yield from self.gripper(True, cancellable=False)
        yield from self.up_to_safe(cancellable=False)


class Place(Skill):
    """Carry the held block to `xy` and set it down in `tray` ('placement' / 'storage' / None = table)."""
    name = "place"
    cancel_phases = ("rise", "transport", "lower")

    def body(self):
        b = self.params["block"]
        xy = np.asarray(self.params["xy"], float)
        m, v = self.cfg["motion"], self.cfg["verify"]
        safe = float(m["safe_z"])
        self.holding = b
        self.set_phase("rise")
        yield from self.up_to_safe(cancellable=True)
        off = self.yaw_offset_for(xy, 0.0)        # set the cube down square to the tray
        g = self.grasp_xy(xy, off)
        self.set_phase("transport")
        yield from self.move([g[0], g[1], safe], off)
        self.set_phase("lower")
        tray = self.params.get("tray")
        floor = self.world.trays[tray]["floor_top"] if tray else 0.0
        z = floor + self.world.edge / 2 + float(m["place_clearance"]) + float(m["grasp_dz"])
        yield from self.move([g[0], g[1], z], off)
        self.set_phase("release")
        self.holding = None
        self.ctx.on_release(b)
        yield from self.gripper(True, cancellable=False)
        self.set_phase("retreat")
        yield from self.up_to_safe(cancellable=False)
        self.set_phase("verify")
        t_end = self.obs.t + float(v["settle_s"])
        while True:
            p = self.obs.blocks[b]
            rest = float(self.obs.block_speed[b]) < float(v["rest_speed"])
            inside = block_in_tray(p, self.world.trays[tray], self.world.edge) if tray else True
            clear = float(self.obs.ee[2] - p[2]) > float(v["ee_clear"])
            jaw_open = self.ctrl.state.gripper >= self.ctrl.p.gripper_open - 1e-6
            if rest and inside and clear and jaw_open:
                self.evidence.update({"block_final": [round(float(x), 5) for x in p], "in_tray": tray,
                                      "observation": "sim_truth"})
                return
            if self.obs.t > t_end:
                self.evidence.update({"block_final": [round(float(x), 5) for x in p], "in_tray": bool(inside),
                                      "at_rest": rest, "ee_clear": clear, "observation": "sim_truth"})
                self._finish("uncertain", "not_verified")
                return
            yield

    def on_cancel(self):
        if self.phase == "lower":
            yield from self.up_to_safe(cancellable=False)
        else:
            yield from self.hold_here()


class Hold(Skill):
    """Keep the held block in the gripper at a safe height until the executor ends the skill (waiting for an answer)."""
    name = "hold"
    cancel_phases = ("hold",)

    def body(self):
        self.holding = self.params.get("block")
        self.set_phase("rise")
        yield from self.up_to_safe(cancellable=False)
        self.set_phase("hold")
        while True:
            self._check_cancel()
            self._check_drop()
            yield

    def cancel(self, reason: str, revision: int) -> bool:
        """Always accepted (an answer may arrive while it is still rising); taken once it holds."""
        if self.status != "running":
            return False
        self.cancel_reason, self.revision = reason, revision
        return True

    def on_cancel(self):
        return
        yield


class Home(Skill):
    """Open-gripper return to the home EE position (end of the task)."""
    name = "home"
    cancel_phases = ("rise", "home")

    def body(self):
        self.set_phase("rise")
        yield from self.up_to_safe(cancellable=True)
        self.set_phase("home")
        yield from self.move(self.world.home, 0.0)

    def on_cancel(self):
        return
        yield


class PickAt(Pick):
    """Pick the block at a perceived pose (`pos` centre, `yaw`) rather than at block `block`'s true pose. Whether it
    holds the block is judged from the robot's own sensors, as on the real arm: the measured jaw angle (a closed
    jaw stops at the block width) and the EE height from the measured joints. `block` names the block expected there,
    only to log the simulation truth beside that verdict. `yaw_offset` (optional) sets the finger orientation."""
    name = "pick_at"

    def body(self):
        b = self.params["block"]
        m = self.cfg["motion"]
        safe = float(m["safe_z"])
        self.set_phase("rise")
        yield from self.up_to_safe(cancellable=True)
        bp, byaw = np.asarray(self.params["pos"], float), float(self.params["yaw"])
        off = self.params.get("yaw_offset")
        off = self.yaw_offset_for(bp, byaw) if off is None else float(off)
        g = self.grasp_xy(bp, off)
        self.evidence["perceived_pos"] = [round(float(x), 5) for x in bp]
        self.set_phase("approach")
        yield from self.move([g[0], g[1], safe], off)
        self.set_phase("descend")
        yield from self.move([g[0], g[1], bp[2] + float(m["grasp_dz"])], off)
        self.set_phase("close")
        yield from self.gripper(False)
        self.evidence["jaw_closed"] = round(float(self.obs.jaw), 4)   # measured, as a servo would report it
        # where the block sits relative to the EE (FK of the measured joints): the held block's place in the wrist view
        self.evidence["block_from_ee_m"] = [round(float(x), 5) for x in bp - self.obs.ee]
        self.set_phase("lift")
        z0, ee_z0 = float(self.obs.blocks[b][2]), float(self.obs.ee[2])
        self.holding = b
        self.ctx.on_grasp(b)
        yield from self.move([g[0], g[1], safe], off)
        gc = self.cfg["grasp_check"]
        bz = float(self.obs.blocks[b][2])
        near = float(np.linalg.norm(self.obs.blocks[b] - self.obs.ee))
        # the robot's own verdict: the jaw stopped on something and the EE (forward kinematics of the measured
        # joints) went up; simulation truth is kept only to measure how often the two disagree
        held = self.jaw_holding() and float(self.obs.ee[2]) - ee_z0 >= float(gc["lifted_dz"])
        truth_held = bz - z0 >= float(gc["lifted_dz"]) and near <= float(gc["max_block_ee"])
        self.evidence.update({"jaw_lifted": round(float(self.obs.jaw), 4),
                              "ee_lift_dz": round(float(self.obs.ee[2]) - ee_z0, 4), "observation": "jaw+ee",
                              "block_lift_dz": round(bz - z0, 4), "block_ee_dist": round(near, 4),
                              "truth_held": bool(truth_held)})
        if not held:
            self.holding = None
            self.ctx.on_release(b)
            yield from self.gripper(True, cancellable=False)
            yield from self.up_to_safe(cancellable=False)
            raise _Fail("grasp_missed")


    def jaw_holding(self) -> bool:
        return float(self.obs.jaw) >= float(self.cfg["grasp_check"]["jaw_hold_min_rad"])

    def _check_drop(self):
        """Dropped = the measured jaw closed past the block width (no simulation truth)."""
        b = self.holding
        if b is None or self.phase != "lift" or self.jaw_holding():
            return
        self.evidence["jaw_at_drop"] = round(float(self.obs.jaw), 4)
        self.ctx.emit("dropped", block=b, phase=self.phase)
        self.holding = None
        raise _Fail("dropped")


class Push(Skill):
    """Push a block along a straight line with the closed gripper: down behind it at `start`, across to `end`
    (xy, mount frame) at height `z`, then straight up. For blocks the gripper cannot pick (tilted, crowded)."""
    name = "push"
    cancel_phases = ("rise", "approach", "descend")

    def body(self):
        start, end = np.asarray(self.params["start"], float), np.asarray(self.params["end"], float)
        z = float(self.params["z"])
        safe = float(self.cfg["motion"]["safe_z"])
        d = end - start
        off = self.yaw_offset_for(start, math.atan2(d[1], d[0]))   # finger faces across the push direction
        self.set_phase("rise")
        yield from self.up_to_safe(cancellable=True)
        yield from self.gripper(False)
        self.set_phase("approach")
        yield from self.move([start[0], start[1], safe], off)
        self.set_phase("descend")
        yield from self.move([start[0], start[1], z], off)
        self.set_phase("push")
        yield from self.move([end[0], end[1], z], off, cancellable=False)
        self.set_phase("retreat")
        yield from self.up_to_safe(cancellable=False)
        yield from self.gripper(True, cancellable=False)

    def on_cancel(self):
        yield from self.up_to_safe(cancellable=False)
        yield from self.gripper(True, cancellable=False)
