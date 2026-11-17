"""Transient IK rejections near the reach boundary: hold one key with jittered frame times and count how often the
target is rejected and then moves on more than 5 mm (a rejection that was not the real reach limit).

  uv run tools/ik_boundary_check.py [--runs 30] [--rev HEAD] [--out outputs/ik_boundary_check.json]

--rev also runs the same test with solve_ik_tilt from that git revision (before / after comparison).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from agentic_sim import kinematics as K  # noqa: E402
from agentic_sim.config import load  # noqa: E402
from agentic_sim.teleop.ee_target import load_controller  # noqa: E402

HEIGHTS = (0.03, 0.08, 0.12, 0.20)
DIRS = {"+x": [1, 0, 0], "+x+y": [1, 1, 0], "+x-y": [1, -1, 0], "+y": [0, 1, 0], "+z": [0, 0, 1]}


def run(solver, runs: int, seed: int) -> dict:
    K.So101Chain.solve_ik_tilt = solver
    chain = K.So101Chain.from_dict(load("configs/so101_chain.yaml"))
    ctrl = load_controller(chain, load("configs/teleop.yaml"),
                           base_pos=load("configs/scenes/lab_current_minimal.yaml")["robot"]["base_pos"]["value"])
    rng = np.random.default_rng(seed)
    out = {}
    for z in HEIGHTS:
        for name, d in DIRS.items():
            events = []
            for _ in range(runs):
                ctrl.reset()
                while abs(ctrl.state.pos[2] - z) > 0.002:
                    ctrl.step_velocity([0, 0, np.sign(z - ctrl.state.pos[2])], 0, 1 / 60, "kb")
                first, n = None, 0
                for _ in range(1500):
                    ok = ctrl.step_velocity(d, 0, rng.uniform(1 / 75, 1 / 45), "kb")
                    if not ok and ctrl.blocked and first is None:
                        first = (ctrl.state.pos.copy(), ctrl.state.status, ctrl.state.message)
                    if first is not None:
                        n += 1
                        if np.linalg.norm(ctrl.state.pos - first[0]) > 0.005:
                            events.append({"pos": first[0].round(4).tolist(), "status": first[1], "msg": first[2]})
                            break
                        if n > 60:
                            break
                    if ctrl.state.status == "workspace_limit":
                        break
            out[f"z={z} {name}"] = {"runs": runs, "transient": len(events), "events": events}
    return out


def solver_from_rev(rev: str):
    src = subprocess.run(["git", "show", f"{rev}:src/agentic_sim/kinematics.py"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    mod = types.ModuleType("kinematics_rev")
    sys.modules["kinematics_rev"] = mod  # dataclasses look their module up here
    exec(compile(src, f"{rev}:kinematics.py", "exec"), mod.__dict__)
    return mod.So101Chain.solve_ik_tilt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rev", default=None)
    ap.add_argument("--out", default="outputs/ik_boundary_check.json")
    a = ap.parse_args()
    current = K.So101Chain.solve_ik_tilt
    report = {"runs_per_case": a.runs, "frame_dt": [1 / 75, 1 / 45], "current": run(current, a.runs, a.seed)}
    if a.rev:
        report["rev"] = a.rev
        report["before"] = run(solver_from_rev(a.rev), a.runs, a.seed)
    for k in ("before", "current"):
        if k in report:
            tot = sum(v["transient"] for v in report[k].values())
            n = sum(v["runs"] for v in report[k].values())
            kinds = sorted({e["msg"].split(" ")[0] + " " + e["msg"].split(" ")[1] for v in report[k].values()
                            for e in v["events"]})
            report[f"{k}_total"] = {"transient": tot, "runs": n, "kinds": kinds}
            print(f"{k}: {tot}/{n} runs with a transient rejection {kinds}")
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
