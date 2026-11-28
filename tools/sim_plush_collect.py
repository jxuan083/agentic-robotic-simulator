#!/usr/bin/env python3
"""Scripted plush grasps in MuJoCo, recorded at 30 fps straight into a LeRobot dataset (docs/plush.md).

  MENAGERIE_DIR=... MUJOCO_GL=egl .venv-train/bin/python tools/sim_plush_collect.py \
      --out outputs/plush/datasets/sim-plush-v1 --episodes 120 --seed 1000

Each episode (agentic_sim.plush_sim_data.sample_layout): random plush pose and 0-3 distractor blocks, the arm at
the scene's standby +-2 deg per joint. The script grasps the dog's head from straight above as
tools/sim_plush_grasp.py does (jaw opened to 0.75-0.95 rad, an approach point above the head with a random offset,
descend, close, lift 6 cm, settle), the real ACT segment start pose -> stable lift. Random pauses and speeds keep
the trajectories apart. Only grasps that pass the study's test (risen >= 45 mm, < 5 mm slip in a 1 s hold, the
moving jaw on the plush) are kept; the hold itself is not recorded. Every attempt is listed in
meta/sim_sources.jsonl (seed, layout, result, kept); kept attempts are LeRobot episodes in that order.

Observation / action layout: agentic_sim.plush_sim_data (the real ACT dataset's features). `--video` opens a
window (DISPLAY) that shows the global and wrist frames as they are recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import sim_plush_grasp as G  # noqa: E402
from agentic_sim import plush_sim_data as D  # noqa: E402
from agentic_sim.config import load, v  # noqa: E402
from agentic_sim.kinematics import So101Chain  # noqa: E402
from agentic_sim.mujoco_backend.model import build  # noqa: E402
from agentic_sim.teleop.ee_target import load_controller  # noqa: E402


LIFT_M = 0.06   # 1 cm above the study's 5 cm: the plush hangs a few mm, so the 45 mm test keeps a margin


class SimRig(G.Rig):
    """sim_plush_grasp.Rig on a given scene dict and layout, stepped at 30 Hz; records each frame when asked."""

    def __init__(self, scene: dict, layout: dict, fps: int = D.FPS):
        self.scene, self.layout = scene, layout
        sc_cfg, colors = D.scene_for(scene, layout)
        self.sc = build(sc_cfg, colors=colors)
        self.m, self.d = self.sc.model, mujoco.MjData(self.sc.model)
        self.substeps = max(1, round(1.0 / fps / self.m.opt.timestep))
        self.dt = self.substeps * self.m.opt.timestep
        chain = So101Chain.from_dict(load("configs/so101_chain.yaml"))
        self.ctrl = load_controller(chain, load("configs/teleop.yaml"), base_pos=v(scene["robot"]["base_pos"]),
                                    scene=sc_cfg)
        self.ctrl.p.dt_max = max(self.ctrl.p.dt_max, self.dt)
        s, sc = self.ctrl.state, self.sc
        q0 = self.ctrl.q_home + np.radians(layout["start_noise_deg"])
        self.ctrl.homing, s.q_cmd, s.q_goal, s.pos = False, q0.copy(), q0.copy(), self.ctrl.ee_pos(q0)
        for adr, q, off in zip(sc.arm_qpos, s.q_cmd, sc.arm_offset):
            self.d.qpos[adr] = q + off
        self.d.qpos[sc.jaw_qpos] = s.gripper
        D.set_block_yaws(sc, self.d, layout)
        mujoco.mj_forward(self.m, self.d)
        self.gripper = self.m.body("gripper").id
        self.jaw_body = self.m.body("moving_jaw_so101_v1").id
        self.recording, self.frames, self.cams, self.marks, self.cond = False, [], None, None, None
        self.hold_steps(0.5)          # settle: plush and blocks come to rest, the arm holds the start pose

    def start_recording(self, cams):
        self.cams = cams
        self.cond, info = D.plush_condition(self.sc, self.d)
        self.target = info
        self.marks = {"target": cams.project(self.d, info["top"])}
        self.recording = True

    def frame(self):
        s, sc = self.ctrl.state, self.sc
        obs = None
        if self.recording:
            obs = {"state": D.joints_lerobot(sc, self.d), "images": self.cams.render(self.d, self.marks)}
        self.ctrl.joint_command(self.dt)
        for act, q, off in zip(sc.arm_act, s.q_cmd, sc.arm_offset):
            self.d.ctrl[act] = q + off
        self.d.ctrl[sc.jaw_act] = s.gripper
        if obs is not None:
            obs["action"] = D.command_lerobot(s.q_cmd, s.gripper)
            self.frames.append(obs)
        mujoco.mj_step(self.m, self.d, nstep=self.substeps)

    def wait(self, seconds):
        self.hold_steps(seconds)


def demonstrate(rig: SimRig, rng: np.random.Generator) -> dict:
    """The scripted grasp with random pauses, speeds and approach offset; returns the test result."""
    plan = G.grasp_plan(rig, "head")
    p0 = rig.plush_pose()[0].copy()
    ee, yaw = plan["ee"], plan["yaw_offset"]
    above_z = max(ee[2] + rng.uniform(0.04, 0.07), G.P.TOTAL_HEIGHT + 0.045)
    via = np.array([ee[0] + rng.uniform(-0.010, 0.010), ee[1] + rng.uniform(-0.010, 0.010), above_z])
    rig.wait(rng.uniform(0.1, 0.4))                                    # still start (the real segment begins ~5 frames early)
    rig.ctrl.state.gripper_goal = float(rng.uniform(0.75, 0.95))       # opens while moving
    rig.move_to(via, yaw, speed=float(rng.uniform(0.05, 0.08)))
    rig.move_to([ee[0], ee[1], above_z], yaw, speed=0.04)
    while rig.ctrl.state.gripper_goal is not None:
        rig.frame()
    rig.move_to(ee, yaw, speed=float(rng.uniform(0.02, 0.035)))
    rig.ctrl.state.gripper_goal = rig.ctrl.p.gripper_closed
    while rig.ctrl.state.gripper_goal is not None:
        rig.frame()
    rig.wait(rng.uniform(0.2, 0.4))
    rig.move_to(ee + [0, 0, LIFT_M], yaw, speed=float(rng.uniform(0.025, 0.035)))
    rig.wait(rng.uniform(0.0, 0.2))                                   # move_to already settles 0.4 s
    rig.recording = False
    rel0 = rig.plush_in_gripper()
    rig.hold_steps(G.HOLD_S)
    lift = float(rig.plush_pose()[0][2] - p0[2])
    slip = float(np.linalg.norm(rig.plush_in_gripper() - rel0))
    held = rig.finger_contacts()
    ok = lift >= D.LIFT_OK_M and slip < D.SLIP_OK_M and bool(held["moving"])
    return {"success": ok, "lift_mm": round(lift * 1000, 1), "slip_mm": round(slip * 1000, 2),
            "contacts": held, "grasp_ee_mm": (ee * 1000).round(1).tolist(), "via_mm": (via * 1000).round(1).tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=G.SCENE)
    ap.add_argument("--out", required=True, help="new LeRobot dataset directory")
    ap.add_argument("--repo-id", default="local/sim-plush")
    ap.add_argument("--episodes", type=int, default=1, help="kept episodes to record")
    ap.add_argument("--seed", type=int, default=1000, help="first attempt seed (attempt i uses seed + i)")
    ap.add_argument("--max-attempts", type=int, default=0, help="0: 2 x episodes")
    ap.add_argument("--video", action="store_true", help="show the recorded frames in a window")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        ap.error(f"{out} exists; choose a new --out")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    scene = load(a.scene)
    ds = LeRobotDataset.create(a.repo_id, fps=D.FPS, features=D.features(), root=out, robot_type="so101_follower",
                               use_videos=True)
    sources, kept, attempt = [], 0, 0
    limit = a.max_attempts or 2 * a.episodes
    viewer = None
    while kept < a.episodes and attempt < limit:
        seed = a.seed + attempt
        attempt += 1
        rng = np.random.default_rng(seed)
        layout = D.sample_layout(scene, rng)
        t0 = time.monotonic()
        rig = SimRig(scene, layout)
        cams = D.Cameras(rig.sc, D.scene_for(scene, layout)[0])
        rig.start_recording(cams)
        if a.video:
            import cv2
            viewer = viewer or "sim plush collect"
            orig = rig.frame

            def frame_and_show(orig=orig, rig=rig, viewer=viewer):
                orig()
                if rig.frames and rig.recording:
                    im = rig.frames[-1]["images"]
                    cv2.imshow(viewer, np.hstack([im["global"], im["wrist"]])[..., ::-1])
                    cv2.waitKey(1)
            rig.frame = frame_and_show
        try:
            result = demonstrate(rig, rng)
        except RuntimeError as e:           # a target the controller refused
            result = {"success": False, "error": str(e)}
        cams.close()
        rec = {"attempt_seed": seed, "layout": layout, "target": rig.target, "condition": rig.cond,
               "marker_pixel": rig.marks["target"], "frames": len(rig.frames), "result": result,
               "kept": bool(result["success"]), "seconds": round(time.monotonic() - t0, 1)}
        if result["success"]:
            for f in rig.frames:
                ds.add_frame({"observation.state": np.asarray(f["state"], np.float32),
                              "action": np.asarray(f["action"], np.float32),
                              "task_is_policy": np.zeros(1, np.int64), "is_handoff": np.zeros(1, np.int64),
                              "observation.environment_state": np.asarray(rig.cond, np.float32),
                              "observation.images.global": f["images"]["global"],
                              "observation.images.wrist": f["images"]["wrist"], "task": D.TASK})
            ds.save_episode()
            rec["lerobot_episode"] = kept
            kept += 1
        sources.append(rec)
        print(f"[{'ok' if rec['kept'] else 'drop'}] seed {seed}: {len(rig.frames)} frames, "
              f"plush ({layout['plush_xy'][0] * 1000:.0f}, {layout['plush_xy'][1] * 1000:.0f}) mm "
              f"yaw {layout['plush_yaw_deg']:.0f}, {len(layout['blocks'])} blocks, "
              f"{json.dumps({k: result.get(k) for k in ('lift_mm', 'slip_mm', 'error') if k in result})}, "
              f"{rec['seconds']} s  [{kept}/{a.episodes}]", flush=True)
    ds.finalize()
    import importlib.metadata as md
    import subprocess
    with open(out / "meta" / "sim_sources.jsonl", "w", encoding="utf-8") as f:
        for r in sources:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            check=False).stdout
    (out / "meta" / "sim_collection.json").write_text(json.dumps({
        "options": vars(a), "fps": D.FPS, "scene": scene.get("name"), "commit": commit.strip(),
        "kept": kept, "attempts": attempt, "task": D.TASK, "condition_names": D.CONDITION_NAMES,
        "condition": "plush footprint centre mount mm, sin / cos (2 x tram yaw)",
        "global_camera": {"size": D.GLOBAL_SIZE, "f_px": round(D.GLOBAL_SCALE * float(
            v(scene["sensor_cameras"]["global"]["intrinsics"])["fy"]), 2), "source": "estimated (0.6 x 1280x720 fy)"},
        "versions": {"mujoco": mujoco.__version__, "lerobot": md.version("lerobot")},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}: {kept} episodes from {attempt} attempts")
    return 0 if kept == a.episodes else 1


if __name__ == "__main__":
    sys.exit(main())
