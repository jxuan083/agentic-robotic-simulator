#!/usr/bin/env python3
"""Reachable region of the teleop IK: vertical-only grasp vs the tilted grasp (configs/teleop.yaml ik.max_tilt_deg).

  uv run tools/reach_map.py [--out outputs/reach_map.json] [--tilts 4,60] [--step 0.01]

Pure numpy (no simulator). Two maps per max-tilt value:
  side: distance from the pan axis (r) x height (z), in the vertical plane through the pan axis (+X direction)
  top:  x, y at grasp height (z = workspace z min)
A cell is reachable when solve_ik_tilt converges from any of a few seeds (joint limits and the tilt limit
respected); the value stored is the smallest tilt found (deg), or null. Self-collision and table contact are not
modelled (physics handles them in the simulator). Plot with tools/plot_reach.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agentic_sim.config import load  # noqa: E402
from agentic_sim.kinematics import So101Chain  # noqa: E402
from agentic_sim.teleop.ee_target import load_controller  # noqa: E402

# planar seeds (pitch, elbow, wrist pitch); pan comes from the target direction, roll from home
SEEDS = ((0.0, 0.0, 1.5), (1.0, -1.0, 1.5), (0.5, -1.2, 1.6), (-0.5, 0.3, 1.6), (1.3, -1.4, 1.2), (0.8, -0.3, 0.6))


def reachable(chain, ctrl, p, max_tilt):
    yaw = ctrl.yaw_for(p, 0.0)
    # the pan joint turns opposite to the mount-frame heading (measured: target heading +0.56 rad -> q0 -0.51)
    pan = float(np.clip(-np.arctan2(p[1] - ctrl.pan[1], p[0] - ctrl.pan[0]), chain.lower[0], chain.upper[0]))
    best = None
    for s in SEEDS:
        q0 = np.array([pan, *s, ctrl.q_home[4]], float)
        q, ok, info = chain.solve_ik_tilt(p, yaw, q0, max_tilt, iters=150)
        if ok and (best is None or info["tilt_deg"] < best):
            best = round(info["tilt_deg"], 1)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reach_map.json")
    ap.add_argument("--tilts", default="4,60")
    ap.add_argument("--step", type=float, default=0.01)
    a = ap.parse_args()
    chain = So101Chain.from_dict(load("configs/so101_chain.yaml"))
    cfg = load("configs/teleop.yaml")
    ctrl = load_controller(chain, cfg)
    ws = cfg["workspace"]
    z0 = float(ws["z"][0])
    rs = np.round(np.arange(0.0, 0.46 + 1e-9, a.step), 4)
    zs = np.round(np.arange(0.0, 0.40 + 1e-9, a.step), 4)
    xs = np.round(np.arange(-0.10, 0.46 + 1e-9, a.step), 4)
    ys = np.round(np.arange(-0.46, 0.46 + 1e-9, a.step), 4)
    out = {"pan_origin": ctrl.pan.tolist(), "workspace": ws, "grasp_z": z0, "r": rs.tolist(), "z": zs.tolist(),
           "x": xs.tolist(), "y": ys.tolist(), "maps": {},
           "note": "IK reachability only (joint limits, tilt limit); no self-collision / table contact model"}
    for t in [float(x) for x in a.tilts.split(",")]:
        side = [[reachable(chain, ctrl, np.array([ctrl.pan[0] + r, ctrl.pan[1], z]), t) for r in rs] for z in zs]
        top = [[reachable(chain, ctrl, np.array([x, y, z0]), t) for y in ys] for x in xs]
        out["maps"][str(int(t))] = {"side": side, "top": top}
        far = max((r for row in side for r, v in zip(rs, row) if v is not None), default=None)
        print(f"max tilt {t:g} deg: side {sum(v is not None for row in side for v in row)} cells (farthest r {far}), "
              f"top {sum(v is not None for row in top for v in row)} cells", flush=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
