#!/usr/bin/env python3
"""Closed-loop evaluation of a plush ACT checkpoint in MuJoCo (docs/plush.md).

  MENAGERIE_DIR=... MUJOCO_GL=egl .venv-train/bin/python tools/sim_plush_eval.py \
      outputs/plush/train/<run>/checkpoints/last/pretrained_model --trials 50 --seed 900000 --json out.json

Trial i draws its layout with seed + i (agentic_sim.plush_sim_data.sample_layout; pick a seed range the training
set did not use) and starts like a demonstration. Every 1/30 s: observe (joint readings, global image with the
target circle, wrist image, plush condition), run the policy (agentic_sim.real_policy.ActPolicy, the real runtime
wrapper) and send its action through agentic_sim.real_policy.Watch, as real_policy.run does with the plush's
open_pct: jumps over 6 deg ramped, over 15 deg stop; done when the gripper opened, closed, the grasp centre is 40 mm
above its lowest point and the commands are still 0.3 s; or 20 s. Then the arm holds 1 s.

Success, as tools/sim_plush_grasp.py: the plush rose >= 45 mm, moved < 5 mm in the gripper during the hold, the
moving jaw touches it. Failures, first match: step_limit (an action out of bounds), collision (gripper side touched
a block or a tray wall, or a block moved > 5 mm), short_lift (held without slipping but raised 20-45 mm),
slipped (the plush rose >= 20 mm at some point but is not held),
body_grasp (the jaws touched the tram but never the head), timeout (20 s without the done test), missed (no grasp).

The checkpoint plays its chunks as trained (act_right.yaml: chunk 100, all 100 played, so a new chunk starts every
3.3 s); --n-action-steps or --temporal-ensemble change that at inference only, to compare execution modes.
--cond-noise adds a detector error to the condition and the drawn target (the layout and the plush stay put).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from agentic_sim import plush_sim_data as D  # noqa: E402
from agentic_sim.config import load  # noqa: E402
from agentic_sim.real_control import JOINTS  # noqa: E402
from agentic_sim.real_policy import BODY, PLUSH_OPEN_PCT, TIMEOUT_S, Watch  # noqa: E402
from sim_plush_collect import SimRig  # noqa: E402
import sim_plush_grasp as G  # noqa: E402

ROSE_M = 0.020
BLOCK_MOVED_M = 0.005


def subtree(m, root: str) -> set:
    r = m.body(root).id
    out = set()
    for b in range(m.nbody):
        x = b
        while x > 0 and x != r:
            x = m.body_parentid[x]
        if x == r:
            out.add(b)
    return out


def set_execution(policy, n_action_steps=None, temporal_ensemble=None):
    """Change how an ActPolicy plays its chunks (inference only, the weights are untouched): run the first
    n_action_steps of each chunk, or predict every step and blend the overlapping chunks (ACT's temporal
    ensembling, weight exp(-coeff * age))."""
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    p = policy.policy
    if temporal_ensemble is not None:
        p.config.temporal_ensemble_coeff, p.config.n_action_steps = temporal_ensemble, 1
        p.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble, p.config.chunk_size)
    elif n_action_steps is not None:
        p.config.n_action_steps = n_action_steps
    p.reset()


def perturb(rig, cams, rng, mm, deg) -> dict:
    """Detector error: shift the condition and the drawn target by N(0, mm) in x / y and the yaw by N(0, deg)."""
    dx, dy = rng.normal(0.0, mm, 2) if mm else (0.0, 0.0)
    dyaw = rng.normal(0.0, deg) if deg else 0.0
    x, y = rig.target["x_mm"] + dx, rig.target["y_mm"] + dy
    yaw = math.radians(rig.target["yaw_deg"] + dyaw)
    rig.cond = [x, y, math.sin(2 * yaw), math.cos(2 * yaw)]
    rig.marks = {"target": cams.project(rig.d, np.array(rig.target["top"]) + [dx / 1000, dy / 1000, 0.0])}
    return {"dx_mm": round(float(dx), 1), "dy_mm": round(float(dy), 1), "dyaw_deg": round(float(dyaw), 1)}


def trial(policy, scene, seed, video=None, noise=(0.0, 0.0)) -> dict:
    rng = np.random.default_rng(seed)
    layout = D.sample_layout(scene, rng)
    rig = SimRig(scene, layout)
    m, d, sc = rig.m, rig.d, rig.sc
    cams = D.Cameras(sc, D.scene_for(scene, layout)[0])
    rig.start_recording(cams)
    rig.recording = False
    error = perturb(rig, cams, np.random.default_rng([seed, 1]), *noise) if any(noise) else None
    arm_side = subtree(m, "wrist")
    p0 = rig.plush_pose()[0].copy()
    blocks0 = [d.xpos[b].copy() for b in sc.block_bodies[:len(layout["blocks"])]]
    policy.reset()
    s = rig.ctrl.state
    w = Watch(dict(zip(JOINTS, D.command_lerobot(s.q_cmd, s.gripper))), PLUSH_OPEN_PCT)
    steps, why, t = 0, "timeout", 0.0
    seen = {"head": False, "body": False, "blocks": False, "tray": False}
    max_rise = 0.0
    while True:
        state = D.joints_lerobot(sc, d)
        images = cams.render(d, rig.marks)
        if video is not None:
            video.write(np.hstack([images["global"], images["wrist"]])[..., ::-1])
        action = policy(state, images, rig.cond)
        cmd = w.command(dict(zip(JOINTS, action)))
        if cmd is None:
            j = max(BODY, key=lambda n: abs(action[JOINTS.index(n)] - w.last[n]))
            why = f"step {w.jump:.1f} {j} at {steps / D.FPS:.2f} s"
            break
        action = [cmd[n] for n in JOINTS]
        arm, jaw = D.command_shared(action)
        for act, q, off in zip(sc.arm_act, arm, sc.arm_offset):
            d.ctrl[act] = q + off
        d.ctrl[sc.jaw_act] = jaw
        mujoco.mj_step(m, d, nstep=rig.substeps)
        steps += 1
        t = steps * rig.dt
        c = D.contacts(m, d, sc, arm_side)
        seen["head"] |= "head" in c["plush_parts"]
        seen["body"] |= any(p != "head" for p in c["plush_parts"])
        seen["blocks"] |= c["blocks"]
        seen["tray"] |= c["tray"]
        max_rise = max(max_rise, float(rig.plush_pose()[0][2] - p0[2]))
        if w.done(t, float(rig.ctrl.ee_pos(arm)[2])):
            why = "done"
            break
        if t > TIMEOUT_S:
            break
    rel0 = rig.plush_in_gripper()
    for _ in range(round(G.HOLD_S / rig.dt)):        # hold the last command
        mujoco.mj_step(m, d, nstep=rig.substeps)
        max_rise = max(max_rise, float(rig.plush_pose()[0][2] - p0[2]))
        if video is not None:
            im = cams.render(d, rig.marks)
            video.write(np.hstack([im["global"], im["wrist"]])[..., ::-1])
    lift = float(rig.plush_pose()[0][2] - p0[2])
    slip = float(np.linalg.norm(rig.plush_in_gripper() - rel0))
    held = rig.finger_contacts()
    moved = max([float(np.linalg.norm(d.xpos[b][:2] - p[:2])) for b, p in zip(sc.block_bodies, blocks0)] or [0.0])
    ok = lift >= D.LIFT_OK_M and slip < D.SLIP_OK_M and bool(held["moving"])
    if ok:
        kind = "success"
    elif why.startswith("step"):
        kind = "step_limit"
    elif seen["blocks"] or seen["tray"] or moved > BLOCK_MOVED_M:
        kind = "collision"
    elif lift >= ROSE_M and slip < D.SLIP_OK_M and held["moving"]:
        kind = "short_lift"
    elif max_rise >= ROSE_M:
        kind = "slipped"
    elif seen["body"] and not seen["head"]:
        kind = "body_grasp"
    elif why == "timeout":
        kind = "timeout"
    else:
        kind = "missed"
    cams.close()
    return {"seed": seed, "success": ok, "class": kind, "why": why, "steps": steps, "seconds": round(t, 2), "ramped": w.ramped,
            "lift_mm": round(lift * 1000, 1), "max_rise_mm": round(max_rise * 1000, 1), "slip_mm": round(slip * 1000, 2),
            "touched": seen, "block_moved_mm": round(moved * 1000, 1), "layout": layout, "condition": rig.cond,
            "detector_error": error}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--scene", default=G.SCENE)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--json", help="write the per-trial results and the summary here")
    ap.add_argument("--video-dir", help="save the first --videos trials as MP4 (global | wrist)")
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--n-action-steps", type=int, help="play only this many actions of each chunk (default: the "
                    "checkpoint's n_action_steps)")
    ap.add_argument("--cond-noise", type=float, nargs=2, default=[0.0, 0.0], metavar=("MM", "DEG"),
                    help="detector error on the condition and the target circle (sigma, x / y mm and yaw deg)")
    ap.add_argument("--temporal-ensemble", type=float, metavar="COEFF",
                    help="predict every step and blend chunks (ACT temporal ensembling, e.g. 0.01)")
    a = ap.parse_args(argv)
    from agentic_sim.real_policy import ActPolicy

    import cv2
    policy = ActPolicy(a.checkpoint)
    set_execution(policy, a.n_action_steps, a.temporal_ensemble)
    cfg = policy.policy.config
    scene = load(a.scene)
    results = []
    for i in range(a.trials):
        video = None
        if a.video_dir and i < a.videos:
            Path(a.video_dir).mkdir(parents=True, exist_ok=True)
            w, h = D.GLOBAL_SIZE
            video = cv2.VideoWriter(str(Path(a.video_dir) / f"trial_{a.seed + i}.mp4"),
                                    cv2.VideoWriter_fourcc(*"mp4v"), D.FPS, (2 * w, h))
        t0 = time.monotonic()
        r = trial(policy, scene, a.seed + i, video, tuple(a.cond_noise))
        if video is not None:
            video.release()
        results.append(r)
        print(f"[{r['class']}] seed {r['seed']}: {r['why']}, {r['seconds']} s, lift {r['lift_mm']} mm, "
              f"slip {r['slip_mm']} mm, {len(r['layout']['blocks'])} blocks  ({time.monotonic() - t0:.0f} s)",
              flush=True)
    counts = Counter(r["class"] for r in results)
    summary = {"checkpoint": a.checkpoint, "trials": len(results), "seed": a.seed,
               "chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps,
               "temporal_ensemble": cfg.temporal_ensemble_coeff, "cond_noise": a.cond_noise,
               "success": counts.get("success", 0), "success_rate": round(counts.get("success", 0) / len(results), 3),
               "classes": dict(counts)}
    print(json.dumps(summary, ensure_ascii=False))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"summary": summary, "trials": results}, ensure_ascii=False, indent=1),
                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
