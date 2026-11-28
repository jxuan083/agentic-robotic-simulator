#!/usr/bin/env python3
"""Can the SO-101 gripper pick the finals plush? MuJoCo study on the plush scene (headless).

  MENAGERIE_DIR=... uv run tools/sim_plush_grasp.py [--scene configs/scenes/lab_20261128_plush.yaml] [--json out.json]

1. Jaw opening: the Menagerie jaw swings on a hinge, so the gap between the fingers depends on the angle and the
   moving finger rises as it opens. Printed per angle: moving-tip distance from the fixed finger's inner face
   (horizontal, in the gripper frame) and the moving tip's height above the fixed tip.
2. Grasps, each from the same start (arm at the scene's standby, plush at rest): open, go above, descend, close,
   lift 5 cm, hold 1 s. The fixed finger comes down just outside one side of the part; the moving jaw closes the
   part onto it. Scripted with the shared EE controller (IK, joint rate limit) as the teleop and task skills use it.
     tram_width   across the tram body (60 mm), centred along the tram: the palm must stay above the dog's head,
                  so only the top of the tram is between the fingers
     tram_end     across the tram body at one end (beside the head), fingers lower on the tram
     head         across the dog's head (about 52 mm), fingers in the head's middle height
   Success: the plush rose at least 45 mm and moved less than 5 mm relative to the gripper during the 1 s hold.
   --yaws repeats each grasp with other plush yaws (in the pink tray corner every yaw clears the walls; the scene's
   block sits at plush.block_xy[0], clear of the plush).

Result (2026-11-28, rigid model, yaws 0-150): head 6/6 (lift 50-52 mm, jaw closes at 0.39-0.42 rad, plush hangs
13 deg tilted); tram_width 0/6 (only the tram top is between the fingers and the open jaw's inclined face squeezes
it down: slides about 14 mm until the head jams on the moving jaw); tram_end 0/6 (palm / wrist camera mount
lands on the head, the fingers end on the rounded corners, dropped or slid).
Pink tray corner (2026-11-28 afternoon, scene plush in storage_pink, textured visual shell, same collision shapes):
head 6/6 at yaws 0-150 (lift 52 mm, slip < 1 mm, jaw closes at about 30 %); with the four blocks at plush.block_xy
the grasp succeeds at yaws 0 / 45 / 90 / 135 and no block moves.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from agentic_sim import plush as P  # noqa: E402
from agentic_sim.config import load, v  # noqa: E402
from agentic_sim.kinematics import So101Chain  # noqa: E402
from agentic_sim.mujoco_backend.model import ROBOT_CONFIG, build  # noqa: E402
from agentic_sim.real_joints import shared_to_lerobot  # noqa: E402
from agentic_sim.teleop.ee_target import load_controller  # noqa: E402

SCENE = "configs/scenes/lab_20261128_plush.yaml"
FIXED_FACE_X = -0.0079   # fixed finger inner face in the gripper link frame (configs/teleop.yaml ee offset note)
CLEARANCE = 0.004        # fixed finger this far outside the part on the way down
OPEN = 1.0               # rad: moving tip 86 mm from the fixed face, 25 mm above the fixed tip
LIFT = 0.05
HOLD_S = 1.0


def jaw_geometry(m: mujoco.MjModel, d: mujoco.MjData, jaw_qpos: int, angle: float) -> dict:
    """Moving tip relative to the fixed finger at a jaw angle (gripper link frame, mm)."""
    d.qpos[jaw_qpos] = angle
    mujoco.mj_kinematics(m, d)
    g = m.body("gripper").id
    R, p = d.xmat[g].reshape(3, 3), d.xpos[g]
    tip = R.T @ (d.geom_xpos[m.geom("moving_jaw_sph_tip1").id] - p)
    ftip = R.T @ (d.geom_xpos[m.geom("fixed_jaw_sph_tip1").id] - p)
    r = m.geom_size[m.geom("moving_jaw_sph_tip1").id][0]
    return {"angle_rad": round(angle, 4), "angle_deg": round(math.degrees(angle), 1),
            "lerobot_pct": round(shared_to_lerobot(np.zeros(5), angle)[5], 1),
            "tip_from_fixed_face_mm": round((tip[0] - r - FIXED_FACE_X) * 1000, 1),
            "tip_above_fixed_tip_mm": round((tip[2] - ftip[2]) * 1000, 1)}


class Rig:
    """Plush scene + the shared EE controller, stepped at the control rate (as the MuJoCo session does)."""

    def __init__(self, scene_path: str = SCENE, yaw_deg: float | None = None):
        self.scene = load(scene_path)
        if yaw_deg is not None:
            self.scene["plush"] = dict(self.scene["plush"], yaw_deg=yaw_deg)
        self.sc = build(self.scene)
        self.m, self.d = self.sc.model, mujoco.MjData(self.sc.model)
        self.substeps = max(1, round(float(load(ROBOT_CONFIG)["physics"]["control_dt"]) / self.m.opt.timestep))
        self.dt = self.substeps * self.m.opt.timestep
        chain = So101Chain.from_dict(load("configs/so101_chain.yaml"))
        self.ctrl = load_controller(chain, load("configs/teleop.yaml"), base_pos=v(self.scene["robot"]["base_pos"]),
                                    scene=self.scene)
        s, sc = self.ctrl.state, self.sc
        self.ctrl.homing, s.q_cmd = False, self.ctrl.q_home.copy()   # start at standby, not in the folded park pose
        for adr, q, off in zip(sc.arm_qpos, s.q_cmd, sc.arm_offset):
            self.d.qpos[adr] = q + off
        self.d.qpos[sc.jaw_qpos] = s.gripper
        self.gripper = self.m.body("gripper").id
        self.jaw_body = self.m.body("moving_jaw_so101_v1").id
        self.hold_steps(0.5)

    def frame(self):
        s, sc = self.ctrl.state, self.sc
        self.ctrl.joint_command(self.dt)
        for act, q, off in zip(sc.arm_act, s.q_cmd, sc.arm_offset):
            self.d.ctrl[act] = q + off
        self.d.ctrl[sc.jaw_act] = s.gripper
        mujoco.mj_step(self.m, self.d, nstep=self.substeps)

    def hold_steps(self, seconds: float):
        for _ in range(round(seconds / self.dt)):
            self.frame()

    def move_to(self, pos, yaw_offset: float, speed: float = 0.06):
        """Straight EE line at speed (m/s); every step must be accepted by the controller."""
        start, goal = self.ctrl.state.pos.copy(), np.asarray(pos, float)
        y0 = self.ctrl.state.yaw_offset
        n = max(1, math.ceil(np.linalg.norm(goal - start) / (speed * self.dt)), math.ceil(abs(yaw_offset - y0) / 0.02))
        for i in range(1, n + 1):
            if not self.ctrl.propose(start + (goal - start) * i / n, y0 + (yaw_offset - y0) * i / n, "script") \
                    and self.ctrl.blocked:
                raise RuntimeError(f"target {np.round(goal, 3).tolist()} rejected: {self.ctrl.state.message}")
            self.frame()
        self.hold_steps(0.4)   # let the arm catch up with the rate-limited command

    def jaw_to(self, angle: float):
        self.ctrl.state.gripper_goal = angle
        while self.ctrl.state.gripper_goal is not None:
            self.frame()
        self.hold_steps(0.5)

    def ee(self) -> np.ndarray:
        return self.ctrl.ee_pos(np.array([self.d.qpos[a] for a in self.sc.arm_qpos]) - self.sc.arm_offset)

    def plush_pose(self) -> tuple:
        b = self.sc.plush_body
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def plush_in_gripper(self) -> np.ndarray:
        R, p = self.d.xmat[self.gripper].reshape(3, 3), self.d.xpos[self.gripper]
        return R.T @ (self.d.xpos[self.sc.plush_body] - p)

    def finger_contacts(self) -> dict:
        """Contacts of the plush with the fixed (gripper body) and the moving jaw, by plush geom name."""
        out = {"fixed": set(), "moving": set()}
        plush = set(self.sc.plush_geoms)
        for c in self.d.contact[:self.d.ncon]:
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in plush:
                    body = self.m.geom_bodyid[b]
                    side = "fixed" if body == self.gripper else "moving" if body == self.jaw_body else None
                    if side:
                        out[side].add(self.m.geom(a).name.removeprefix(f"{P.ID}_"))
        return {k: sorted(x) for k, x in out.items()}


def grasp_plan(rig: Rig, kind: str) -> dict:
    """EE grasp point, opening direction and controller yaw offset for a grasp kind."""
    pos, R = rig.plush_pose()
    ax_len, ax_w = R[:2, 0] / np.linalg.norm(R[:2, 0]), R[:2, 1] / np.linalg.norm(R[:2, 1])
    if kind == "head":
        width, centre, z = 2 * P.HEAD_RADII[1], pos[:2] + R[:2, :2] @ np.array(P.HEAD_POS[:2]), pos[2] + P.HEAD_POS[2]
    elif kind == "tram_width":
        # the palm (fixed_jaw_box1, about 48 mm above the EE) must clear the head: EE no lower than 55 mm
        width, centre, z = P.WIDTH, pos[:2], pos[2] + 0.055
    elif kind == "tram_end":   # the far end (near the base the wrist camera mount hits the shoulder)
        ends = [pos[:2] + s * ax_len * (P.LENGTH / 2 - 0.013) for s in (1, -1)]
        width, z = P.WIDTH, pos[2] + 0.033
        centre = max(ends, key=lambda c: np.linalg.norm(c - rig.ctrl.pan[:2]))
    else:
        raise ValueError(kind)
    ctrl = rig.ctrl
    radial = math.atan2(centre[1] - ctrl.pan[1], centre[0] - ctrl.pan[0]) + ctrl.p.finger_heading
    across = ax_w if kind != "head" else np.array([math.cos(radial), math.sin(radial)])
    # opening direction (fixed finger -> moving jaw) along `across`, the sign that needs the smaller wrist turn
    heading = math.atan2(across[1], across[0])
    off = (heading - radial + math.pi / 2) % math.pi - math.pi / 2
    u = np.array([math.cos(radial + off), math.sin(radial + off)])
    ee_x = float(v(load("configs/teleop.yaml")["ee_offset_in_gripper_link"])[0])
    ee_xy = centre - u * (width / 2 + CLEARANCE - (ee_x - FIXED_FACE_X))
    return {"ee": np.array([*ee_xy, z]), "yaw_offset": off, "width_mm": round(width * 1000, 1),
            "opening_dir": u.round(3).tolist()}


def run_grasp(kind: str, scene_path: str = SCENE, yaw_deg: float | None = None) -> dict:
    rig = Rig(scene_path, yaw_deg)
    plan = grasp_plan(rig, kind)
    p0, R0 = rig.plush_pose()
    ee = plan["ee"]
    above = np.array([ee[0], ee[1], max(ee[2] + 0.05, P.TOTAL_HEIGHT + 0.045)])
    out = {"kind": kind, "plush_yaw_deg": round(math.degrees(math.atan2(R0[1, 0], R0[0, 0])), 1),
           "grasp_ee_mm": (ee * 1000).round(1).tolist(), "yaw_offset_deg": round(math.degrees(plan["yaw_offset"]), 1),
           "part_width_mm": plan["width_mm"]}
    try:
        rig.jaw_to(OPEN)
        rig.move_to(above, plan["yaw_offset"])
        rig.move_to(ee, plan["yaw_offset"], speed=0.03)
        out["plush_shift_on_descent_mm"] = round(float(np.linalg.norm(rig.plush_pose()[0][:2] - p0[:2])) * 1000, 1)
        out["ee_error_mm"] = round(float(np.linalg.norm(rig.ee() - ee)) * 1000, 1)   # arm blocked on the way down
        rig.jaw_to(rig.ctrl.p.gripper_closed)
        jaw = float(rig.d.qpos[rig.sc.jaw_qpos])
        out["jaw_closed"] = jaw_geometry(rig.m, mujoco.MjData(rig.m), rig.sc.jaw_qpos, jaw)
        out["contacts_closed"] = rig.finger_contacts()
        rig.move_to(ee + [0, 0, LIFT], plan["yaw_offset"], speed=0.03)
        rel0 = rig.plush_in_gripper()
        z_lift = rig.plush_pose()[0][2] - p0[2]
        rig.hold_steps(HOLD_S)
        p1, R1 = rig.plush_pose()
        slip = float(np.linalg.norm(rig.plush_in_gripper() - rel0))
        out.update({"lift_mm": round(float(p1[2] - p0[2]) * 1000, 1),
                    "lift_at_top_mm": round(float(z_lift) * 1000, 1),
                    "slip_in_hold_mm": round(slip * 1000, 2),
                    "tilt_deg": round(math.degrees(math.acos(min(1.0, R1[2, 2]))), 1),
                    "contacts_held": rig.finger_contacts(),
                    "jaw_held_rad": round(float(rig.d.qpos[rig.sc.jaw_qpos]), 4)})
        out["success"] = out["lift_mm"] >= 45 and slip < 0.005 and bool(out["contacts_held"]["moving"])
    except RuntimeError as e:   # unreachable target
        out.update({"success": False, "error": str(e)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--kinds", nargs="+", default=["tram_width", "tram_end", "head"])
    ap.add_argument("--yaws", nargs="+", type=float, help="plush yaw(s) in degrees instead of the scene's")
    ap.add_argument("--json", help="write the results here")
    a = ap.parse_args()
    rig = Rig(a.scene)
    lo, hi = rig.m.jnt_range[rig.m.joint("gripper").id]
    d = mujoco.MjData(rig.m)
    table = [jaw_geometry(rig.m, d, rig.sc.jaw_qpos, x) for x in
             (lo, 0.0, 0.18, 0.3, rig.ctrl.p.gripper_open, 0.7, 0.85, OPEN, 1.2, 1.4, 1.6, hi)]
    print("jaw angle        LeRobot %   tip from fixed face   tip above fixed tip")
    for r in table:
        print(f"{r['angle_rad']:6.3f} rad {r['angle_deg']:6.1f} deg {r['lerobot_pct']:6.1f}"
              f"   {r['tip_from_fixed_face_mm']:8.1f} mm         {r['tip_above_fixed_tip_mm']:6.1f} mm")
    results = [run_grasp(k, a.scene, y) for k in a.kinds for y in (a.yaws or [None])]
    for r in results:
        print(json.dumps(r))
    if a.json:
        Path(a.json).write_text(json.dumps({"jaw": table, "grasps": results}, indent=1))


if __name__ == "__main__":
    main()
