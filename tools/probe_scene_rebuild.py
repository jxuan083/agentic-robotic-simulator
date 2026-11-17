"""Measure tearing down and rebuilding the scene inside one Kit process (issue #3, P2 risk check).

Each round: build the scene + SimulationContext, step 120 frames with rendering, check that physics works
(a joint target is reached, the block rests on the tray floor), then tear everything down. Prints one JSON line
per round (build / teardown seconds, process RSS, GPU memory used, prim count) and a summary.

  docker/isaac/run.sh /isaac-sim/python.sh tools/probe_scene_rebuild.py --rounds 10 [--headless]
"""

import argparse
import json
import os
import subprocess
import sys
import time

T_PROC = time.monotonic()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--scene", default="configs/scenes/lab_current_minimal.yaml")
parser.add_argument("--mitigate", default="none", choices=["none", "undo", "undo+close"])
parser.add_argument("--out", default="outputs/probe/scene_rebuild.jsonl")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", kit_args="--/app/vsync=false --/app/runLoops/main/rateLimitEnabled=false")
args = parser.parse_args()
app = AppLauncher(args).app
T_APP = time.monotonic()

import faulthandler  # noqa: E402
import gc  # noqa: E402

faulthandler.dump_traceback_later(90, repeat=True)  # print where it is if a step stalls

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from pxr import Usd  # noqa: E402

from agentic_sim.config import load, v  # noqa: E402
from agentic_sim.isaac.scene import build_scene  # noqa: E402

PHYSICS_DT, SUBSTEPS = 1.0 / 240.0, 4


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def gpu_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        return float(out[0])
    except Exception:
        return -1.0


def build(scene_cfg):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=SUBSTEPS, device=args.device))
    cam = scene_cfg["camera"]
    sim.set_camera_view(cam["eye"], cam["target"])
    h = build_scene(scene_cfg)
    sim.reset()
    return sim, h


def exercise(sim, h, scene_cfg, frames=120):
    """Drive the shoulder pitch to an offset and let the block settle; return what was reached."""
    robot, block = h.robot, h.block
    q = robot.data.default_joint_pos.clone()
    q0 = float(q[0, 1])
    q[0, 1] = q0 - 0.3
    robot.set_joint_position_target(q)
    for _ in range(frames):
        for _ in range(SUBSTEPS):
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(PHYSICS_DT)
            block.update(PHYSICS_DT)
        sim.render()
    reached = float(robot.data.joint_pos[0, 1])
    bz = float(block.data.root_pos_w[0, 2])
    floor = v(scene_cfg["trays"]["storage_pink"]["floor_thickness"]) + v(scene_cfg["block"]["edge"]) / 2
    return {"joint_err": round(abs(reached - (q0 - 0.3)), 4), "block_z_err": round(abs(bz - floor), 4)}


def teardown(holder):
    """holder = [sim, handles]; emptied here so no reference to the old assets survives the call."""
    sim, h = holder
    holder.clear()
    # In GUI mode Isaac Lab's STOP handler keeps rendering until the timeline plays again (meant for a user
    # pressing Stop); a programmatic teardown must switch it off first or stop() never returns.
    sim._disable_app_control_on_stop_handle = True
    if sim.is_playing():
        sim.stop()
    h.robot = h.block = None  # drop the asset objects so their timeline / physics callbacks unsubscribe
    del h
    gc.collect()
    sim.clear_all_callbacks()
    sim.clear_instance()
    del sim
    if args.mitigate in ("undo", "undo+close"):
        import omni.kit.undo
        omni.kit.undo.clear_stack()  # undo history keeps commands (and the prims / layers they touched) alive
    if args.mitigate == "undo+close":
        sim_utils.close_stage()
    sim_utils.create_new_stage()
    app.update()


def main():
    scene_cfg = load(args.scene)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    print(f"[probe] app start {T_APP - T_PROC:.1f}s", flush=True)
    with open(args.out, "w") as f:
        for i in range(args.rounds):
            t0 = time.monotonic()
            print(f"[probe] round {i}: build", flush=True)
            sim, h = build(scene_cfg)
            t1 = time.monotonic()
            print(f"[probe] round {i}: exercise ({t1 - t0:.1f}s build)", flush=True)
            ok = exercise(sim, h, scene_cfg)
            t2 = time.monotonic()
            print(f"[probe] round {i}: teardown ({t2 - t1:.1f}s exercise)", flush=True)
            n_prims = sum(1 for _ in Usd.PrimRange(sim.stage.GetPseudoRoot()))
            holder = [sim, h]
            del sim, h
            teardown(holder)
            gc.collect()
            t3 = time.monotonic()
            row = {"round": i, "build_s": round(t1 - t0, 3), "exercise_s": round(t2 - t1, 3),
                   "teardown_s": round(t3 - t2, 3), "rss_mb": round(rss_mb(), 1), "gpu_mb": gpu_mb(),
                   "prims": n_prims, **ok}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print("[probe] " + json.dumps(row), flush=True)
    b = [r["build_s"] for r in rows]
    summary = {"mitigate": args.mitigate, "app_start_s": round(T_APP - T_PROC, 2), "first_build_s": b[0],
               "later_build_s_max": max(b[1:]) if len(b) > 1 else None,
               "rss_growth_mb": round(rows[-1]["rss_mb"] - rows[0]["rss_mb"], 1),
               "gpu_growth_mb": rows[-1]["gpu_mb"] - rows[0]["gpu_mb"],
               "all_physics_ok": all(r["joint_err"] < 0.02 and r["block_z_err"] < 0.003 for r in rows)}
    print("[probe] summary " + json.dumps(summary), flush=True)


main()
sys.stdout.flush()
os._exit(0)
