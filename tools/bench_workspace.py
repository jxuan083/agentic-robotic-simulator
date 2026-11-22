#!/usr/bin/env python3
"""Operator workspace performance: the same simulated-time motion, per engine and perception setting.

  uv run tools/bench_workspace.py run [--engines isaac,mujoco] [--out outputs/bench]   # runs the GUI, one by one
  uv run --with matplotlib tools/bench_workspace.py report [--out outputs/bench]        # summary + figure

Every run replays configs/trajectories/lab_pick_place.yaml (driven by simulated time, so every run commands the
same motion whatever its speed) in lab_20261120 with the blocks red,yellow,blue, the scene's wrist lens and the
operator workspace (--focus), recording off, and writes one metrics line per host second (--metrics_log): UI FPS,
RTF (simulated / host time), perception result age and estimate latency per camera.

Settings: raw (cameras only, no perception), capture (perception captures, no estimate), estimate_hidden (capture
and estimate, annotations not shown), estimate_shown (capture, estimate and the annotated snapshots shown).
Isaac also runs the recording coexistence: the scripted keyboard capture (configs/capture/test_keyboard_yellow.yaml,
simulated-time waits) with --record, without and with perception.
Warm-up: samples before 3 s of simulated time (the trajectory's settle segment) are dropped.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = ["--scene", "configs/scenes/lab_20261120.yaml", "--block_colors", "red,yellow,blue"]
TRAJ = ["--trajectory", "configs/trajectories/lab_pick_place.yaml", "--exit_after_script", "--focus"]
SETTINGS = {
    "raw": [],
    "capture": ["--live-perception", "--perception-stage", "capture", "--annotations", "off"],
    "estimate_hidden": ["--live-perception", "--annotations", "off"],
    "estimate_shown": ["--live-perception"],
}
ENGINE_CMD = {
    "isaac": ["docker/isaac/run.sh", "/isaac-sim/python.sh", "scripts/isaac_ee_teleop.py"],
    "mujoco": ["uv", "run", "--with", "opencv-python-headless>=4.9", "scripts/mujoco_ee_teleop.py"],
}
WARMUP_SIM_S = 3.0


def runs(engines):
    out = []
    for eng in engines:
        for name, extra in SETTINGS.items():
            out.append((f"{eng}_{name}", ENGINE_CMD[eng] + SCENE + TRAJ + extra))
    if "isaac" in engines:   # recording coexistence (Isaac records; MuJoCo has no recorder)
        for name in ("raw", "estimate_shown"):
            out.append((f"isaac_record_{name}", ENGINE_CMD["isaac"] + SCENE + [
                "--target_color", "yellow", "--script", "configs/capture/test_keyboard_yellow.yaml",
                "--exit_after_script", "--focus", "--record", f"outputs/bench/record_{name}"] + SETTINGS[name]))
    return out


def run(args):
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    for key, cmd in runs(args.engines.split(",")):
        metrics = os.path.join(args.out, f"{key}.jsonl")
        if os.path.exists(os.path.join(ROOT, metrics)):
            if not args.again:
                print(f"skip {key} (exists)")
                continue
            os.remove(os.path.join(ROOT, metrics))
        cmd = cmd + ["--metrics_log", metrics]
        print("$ " + " ".join(cmd), flush=True)
        with open(os.path.join(ROOT, args.out, f"{key}.log"), "w") as log:
            try:
                rc = subprocess.call(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.timeout)
            except subprocess.TimeoutExpired:
                rc = "timeout"
                if cmd[0] == "docker/isaac/run.sh":   # the docker client is gone; the container is not
                    subprocess.call(["docker", "stop", os.environ.get("ISAAC_CONTAINER_NAME", "ars-isaac")],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"{key}: exit {rc}", flush=True)


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows if r["t_sim"] >= WARMUP_SIM_S and not r.get("paused")]
    out = {"samples": len(rows),
           "ui_fps": {"median": pct([r["ui_fps"] for r in rows], .5), "p5": pct([r["ui_fps"] for r in rows], .05)},
           "rtf": {"median": pct([r["rtf"] for r in rows], .5), "p5": pct([r["rtf"] for r in rows], .05)}}
    cams = sorted({c for r in rows for c in (r.get("perception") or {})})
    for c in cams:
        lat = [(r["perception"].get(c) or {}).get("latency_s") for r in rows if r.get("perception")]
        age = [(r["perception"].get(c) or {}).get("age_s") for r in rows if r.get("perception")]
        out[f"{c}_latency_s"] = {"median": pct(lat, .5), "p95": pct(lat, .95)}
        out[f"{c}_age_s"] = {"median": pct(age, .5), "p95": pct(age, .95),
                             "no_result_samples": sum(a is None for a in age)}
    return out


def report(args):
    base = os.path.join(ROOT, args.out)
    summary = {}
    for path in sorted(glob.glob(os.path.join(base, "*.jsonl"))):
        key = os.path.basename(path)[:-6]
        summary[key] = summarize(path)
        rec = sorted(glob.glob(os.path.join(base, f"record_{key.replace('isaac_record_', '')}", "*", "manifest.json")))
        if key.startswith("isaac_record_") and rec:
            m = json.load(open(rec[-1]))
            summary[key]["episode"] = {"path": os.path.relpath(os.path.dirname(rec[-1]), ROOT),
                                       "success": (m.get("result") or {}).get("success"),
                                       "frames": m.get("control_frames"), "operator": m.get("operator")}
    with open(os.path.join(base, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    f2 = lambda x, s=".2f": "-" if x is None else format(x, s)  # noqa: E731
    lines = ["| run | samples | UI FPS median (p5) | RTF median (p5) | global latency median / p95 s | "
             "global age median / p95 s | wrist latency median / p95 s | wrist age median / p95 s |",
             "|---|---|---|---|---|---|---|---|"]
    for k, v in summary.items():
        cell = lambda c, m: (f"{f2(v[f'{c}_{m}_s']['median'])} / {f2(v[f'{c}_{m}_s']['p95'])}"  # noqa: E731
                             if f"{c}_{m}_s" in v else "-")
        lines.append(f"| {k} | {v['samples']} | {f2(v['ui_fps']['median'], '.1f')} ({f2(v['ui_fps']['p5'], '.1f')}) | "
                     f"{f2(v['rtf']['median'])} ({f2(v['rtf']['p5'])}) | {cell('global', 'latency')} | "
                     f"{cell('global', 'age')} | {cell('wrist', 'latency')} | {cell('wrist', 'age')} |")
    open(os.path.join(base, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    figure(summary, base)


def figure(summary, base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: no figure)")
        return
    settings = list(SETTINGS)
    engines = [e for e in ENGINE_CMD if any(k.startswith(e + "_") for k in summary)]
    colors = {"isaac": "#2a6fdb", "mujoco": "#e07b39"}     # fixed per engine, never by rank
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = (("ui_fps", "UI FPS (median, bar; p5, tick)", "p5"), ("rtf", "RTF (median; p5)", "p5"),
              ("wrist_latency_s", "wrist estimate latency s (median; p95)", "p95"))
    w = 0.38
    for ax, (metric, title, tail) in zip(axes, panels):
        for i, eng in enumerate(engines):
            xs, med, tl = [], [], []
            for j, st in enumerate(settings):
                v = summary.get(f"{eng}_{st}", {}).get(metric)
                if not v or v.get("median") is None:
                    continue
                xs.append(j + (i - (len(engines) - 1) / 2) * w)
                med.append(v["median"])
                tl.append(v.get(tail))
            ax.bar(xs, med, width=w * 0.9, color=colors[eng], label=eng)
            ax.scatter(xs, [t if t is not None else float("nan") for t in tl], marker="_", s=200, color="#222222",
                       zorder=3)
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels([s.replace("_", "\n") for s in settings], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Operator workspace, lab_20261120, same simulated-time pick and place (warm-up dropped)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(base, "bench.png"), dpi=110)
    print(f"figure: {os.path.join(base, 'bench.png')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "report"])
    ap.add_argument("--engines", default="isaac,mujoco")
    ap.add_argument("--out", default="outputs/bench")
    ap.add_argument("--again", action="store_true", help="rerun runs whose metrics already exist")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args(argv)
    (run if a.action == "run" else report)(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
