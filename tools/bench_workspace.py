#!/usr/bin/env python3
"""Operator workspace performance: the same simulated-time motion, per engine and perception setting.

  uv run tools/bench_workspace.py run [--engines isaac,mujoco] [--runs KEY,...] [--out outputs/bench]  # GUI, one by one
  uv run --with matplotlib tools/bench_workspace.py report [--out outputs/bench]                       # summary + figure

Every run replays configs/trajectories/lab_pick_place.yaml (driven by simulated time, so every run commands the
same motion whatever its speed) in lab_20261120 with the blocks red,yellow,blue, the scene's wrist lens and the
operator workspace (--focus), recording off, and writes one metrics line per host second (--metrics_log): UI FPS,
RTF (simulated / host time), perception result age and estimate latency per camera.

Settings: raw (cameras only, no perception), capture (perception captures, no estimate), estimate_hidden (capture
and estimate, annotations not shown), estimate_shown (capture, estimate and the annotated snapshots shown).
Isaac also runs the recording coexistence: the scripted keyboard capture (configs/capture/test_keyboard_yellow.yaml,
simulated-time waits and key motion) with --record, without and with perception.
Warm-up: samples before 3 s of simulated time (the trajectory's settle segment) are dropped.

Every run gets its own run id, written in <out>/<key>.run.json and passed to the engine: every metrics line carries
it (--metrics_run_id), and the recording runs write it into their recorder events and manifest (--record_run_id).
Recordings go under the same --out (<out>/record_<setting>/). The report reads only what carries the run id of the
latest run of each key: metrics lines of an older run in the same file, or an older episode in the same record
directory, are never counted. An episode counts only when its `saved` event carries the run id and its manifest
exists, reads and has the same capture_run_id; any other is listed as rejected with its path and reason, and never
counted as a success. A run without such an episode says so ("本次無可驗證 episode").
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import uuid

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
NO_EPISODE = "本次無可驗證 episode"


def record_dir(out, name):
    return os.path.join(out, f"record_{name}")


def runs(engines, out="outputs/bench"):
    res = []
    for eng in engines:
        for name, extra in SETTINGS.items():
            res.append((f"{eng}_{name}", ENGINE_CMD[eng] + SCENE + TRAJ + extra))
    if "isaac" in engines:   # recording coexistence (Isaac records; MuJoCo has no recorder)
        for name in ("raw", "estimate_shown"):
            res.append((f"isaac_record_{name}", ENGINE_CMD["isaac"] + SCENE + [
                "--target_color", "yellow", "--script", "configs/capture/test_keyboard_yellow.yaml",
                "--exit_after_script", "--focus", "--record", record_dir(out, name)] + SETTINGS[name]))
    return res


def run_command(cmd, out, key, run_id):
    """The engine command of one run, bound to its run id (metrics; recorder events, manifest and result)."""
    cmd = cmd + ["--metrics_log", os.path.join(out, f"{key}.jsonl"), "--metrics_run_id", run_id]
    if "--record" in cmd:
        cmd += ["--record_run_id", run_id, "--result", os.path.join(out, f"{key}.result.json")]
    return cmd


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def run(args):
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    only = set(args.runs.split(",")) if args.runs else None
    for key, cmd in runs(args.engines.split(","), args.out):
        if only is not None and key not in only:
            continue
        metrics = os.path.join(ROOT, args.out, f"{key}.jsonl")
        if os.path.exists(metrics) and not args.again:
            print(f"skip {key} (exists)")
            continue
        for stale in (metrics, os.path.join(ROOT, args.out, f"{key}.result.json")):
            if os.path.exists(stale):
                os.remove(stale)
        run_id = uuid.uuid4().hex[:12]
        cmd = run_command(cmd, args.out, key, run_id)
        meta_path = os.path.join(ROOT, args.out, f"{key}.run.json")
        meta = {"key": key, "run_id": run_id, "cmd": cmd, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "exit": None}
        _write_json(meta_path, meta)
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
        meta.update(exit=rc, ended=time.strftime("%Y-%m-%dT%H:%M:%S"))
        _write_json(meta_path, meta)
        print(f"{key}: exit {rc}", flush=True)


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(path, run_id=None):
    """Metrics of one run: only lines carrying its run id (when given), warm-up and paused samples dropped."""
    rows = [json.loads(line) for line in open(path) if line.strip()] if os.path.exists(path) else []
    if run_id is not None:
        rows = [r for r in rows if r.get("run_id") == run_id]
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


def run_episodes(rec_dir, run_id):
    """Episodes this run saved, checked: (valid, rejected). A `saved` event carrying the run id counts only when its
    episode manifest exists, reads and has the same capture_run_id; anything else is rejected with the reason (its
    success is never counted). Episodes of other runs in the same directory are never looked at."""
    events = os.path.join(rec_dir, "recorder_events.jsonl")
    valid, rejected = [], []
    if not os.path.exists(events):
        return valid, rejected
    for line in open(events):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("run_id") != run_id or ev.get("event") != "saved":
            continue
        path = ev.get("episode")
        manifest = os.path.join(ROOT, str(path), "manifest.json")
        reason, m = None, None
        if not path:
            reason = "saved event without an episode path"
        elif not os.path.exists(manifest):
            reason = "manifest missing"
        else:
            try:
                m = json.load(open(manifest))
            except (OSError, ValueError) as e:
                reason = f"manifest unreadable ({type(e).__name__})"
        if reason is None and m.get("capture_run_id") != run_id:
            reason = f"run id mismatch (manifest {m.get('capture_run_id')!r}, run {run_id!r})"
        if reason is not None:
            rejected.append({"path": path, "reason": reason})
            continue
        frames = m.get("control_frames")
        valid.append({"path": path, "success": bool((m.get("result") or {}).get("success")),
                      "reason": (m.get("result") or {}).get("reason"), "frames": frames,
                      "operator": m.get("operator"), "capture_run_id": m.get("capture_run_id")})
    return valid, rejected


def script_result(path):
    """Checks and the scripted grasp (the script's `grasp` mark: where the gripper starts closing) of a run."""
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    g = (r.get("marks") or {}).get("grasp") or {}
    return {"checks_passed": sum(c["ok"] for c in r.get("checks", [])), "checks": len(r.get("checks", [])),
            "grasp_target": g.get("target"), "grasp_ee": g.get("ee"), "grasp_t_sim": g.get("t_sim")}


def report(args):
    base = os.path.join(ROOT, args.out)
    summary = {}
    for meta_path in sorted(glob.glob(os.path.join(base, "*.run.json"))):
        meta = json.load(open(meta_path))
        key, run_id = meta["key"], meta["run_id"]
        summary[key] = summarize(os.path.join(base, f"{key}.jsonl"), run_id)
        summary[key].update(run_id=run_id, exit=meta.get("exit"), started=meta.get("started"))
        if key.startswith("isaac_record_"):
            eps, rejected = run_episodes(record_dir(base, key[len("isaac_record_"):]), run_id)
            summary[key]["episodes"] = eps                       # verified only
            summary[key]["rejected_episodes"] = rejected
            summary[key]["successes"] = sum(e["success"] for e in eps)
            summary[key]["episode"] = eps[-1] if eps else NO_EPISODE
            summary[key]["script"] = script_result(os.path.join(base, f"{key}.result.json"))
    with open(os.path.join(base, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
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
    rec = [(k, v) for k, v in summary.items() if "episode" in v]
    if rec:
        lines += ["", "| run | run id | exit | episode (verified) | success | successes | frames | checks | "
                  "grasp target (m) |", "|---|---|---|---|---|---|---|---|---|"]
        for k, v in rec:
            e, sc = v["episode"], v.get("script") or {}
            gt = sc.get("grasp_target")
            gt = "-" if not gt else "(" + ", ".join(f"{x:.4f}" for x in gt) + ")"
            checks = f"{sc['checks_passed']}/{sc['checks']}" if sc else "-"
            if isinstance(e, dict):
                lines.append(f"| {k} | {v['run_id']} | {v['exit']} | {e['path']} | {e['success']} | "
                             f"{v['successes']} | {e['frames']} | {checks} | {gt} |")
            else:
                lines.append(f"| {k} | {v['run_id']} | {v['exit']} | {e} | - | 0 | - | {checks} | {gt} |")
        rej = [(k, r) for k, v in rec for r in v.get("rejected_episodes", [])]
        if rej:
            lines += ["", "| run | rejected episode (not counted) | reason |", "|---|---|---|"]
            lines += [f"| {k} | {r['path']} | {r['reason']} |" for k, r in rej]
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
    ap.add_argument("--runs", default=None, help="only these run keys, comma-separated (e.g. isaac_record_raw)")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args(argv)
    (run if a.action == "run" else report)(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
