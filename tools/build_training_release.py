#!/usr/bin/env python3
"""Build the training release: check every recorded episode, then convert the accepted ones into one LeRobot
dataset for ACT and one for a VLA (docs/real-training.md). The raw episodes are never moved or copied.

  ~/miniconda3/envs/lerobot/bin/python tools/build_training_release.py \
      --outputs ../agentic-robotic-simulator-real-teleop/outputs --sessions 261127-park 261127-auto \
      --release ../training_dataset/v1

Checks per episode, on top of the converter's own (readback, review marks, D455 target, operator result, control
gaps, camera pairing):
  - duplicates: the same recording (start time + first control row) seen twice is taken once;
  - grasp vs target: the commanded grasp centre at the close (FK of the calibration-file joint map, minus the sag
    correction) must lie within GRASP_TARGET_MM of the D455 target, else another block was taken (11/27 P0006);
  - source: human (P..., demonstration), takeover (H..., a scripted miss demonstrated with the leader), script (S...).
Processing (converter options): still stretches keep 0.3 s, re-grasps kept and flagged, same-colour targets get a
side in the instruction ("the left blue block").

Writes into the release directory:
  act_grasp/     LeRobot dataset: start pose -> lift, condition = target (x, y mm, sin/cos 4*yaw), target circle drawn
                 on the global images (the same draw_markers call must run at inference)
  vla_task/      LeRobot dataset: start pose -> rise over the slot (grasp + programmed place), language instruction
  episodes.json  every examined episode: source, checks, inclusion per dataset and why not
  splits.json    LeRobot episode indices per source (human / takeover / script) for the A/B/C comparison
  coverage.png   targets on the D455 reference (human green, takeover blue, script orange, excluded red)
  README.md      counts, coverage, how it was built, how to train
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from agentic_sim.real_dataset import find_episodes, plan_episode  # noqa: E402

GRASP_TARGET_MM = 30.0
EDGE_M = 0.0295
MODES = ("demonstration", "script")


def source_of(ep: Path, manifest: dict, review: dict) -> str:
    if manifest.get("mode") == "script":
        return "script"
    if (review or {}).get("takeover") or ep.name.startswith("H"):
        return "takeover"
    return "human"


def grasp_offset_mm(ep: Path, review: dict, model, corr, cam, arm="right"):
    """Horizontal distance (mm) from the commanded grasp centre at the close to the D455 target, or None."""
    frames = (review or {}).get("frames") or {}
    if "close" not in frames:
        return None
    script = (review or {}).get("script") or {}
    if script.get("target_mount_m"):
        t = np.asarray(script["target_mount_m"][:2], float)
    else:
        d = ((review or {}).get("target") or {}).get("d455") or {}
        if "pixel" not in d:
            return None
        t = cam.on_plane(np.asarray([d["pixel"]], float), EDGE_M)[0][:2]
    rows = (ep / "control.jsonl").read_text().splitlines()
    k = frames["close"]["frame"]
    cmd = json.loads(rows[min(k, len(rows) - 1)])["arms"][arm]["command"]
    p = model.ee(cmd)
    p = p - corr(p)
    return float(np.hypot(*(p[:2] - t)) * 1000)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=ROOT.parent / "agentic-robotic-simulator-real-teleop/outputs")
    ap.add_argument("--sessions", nargs="+", default=["261127-park", "261127-auto"])
    ap.add_argument("--release", type=Path, required=True, help="new directory (must not exist)")
    ap.add_argument("--pose", default=None, help="D455 pose json (default <outputs>/d455/pose_261125.json)")
    ap.add_argument("--corr", default="21,-1.3,20")
    ap.add_argument("--follower-id", default="my_awesome_follower_arm")
    ap.add_argument("--max-still-s", type=float, default=0.3)
    ap.add_argument("--skip-convert", action="store_true", help="checks, lists and figures only")
    a = ap.parse_args(argv)
    if a.release.exists():
        sys.exit(f"{a.release} exists: releases are never overwritten (pick a new version)")
    from agentic_sim.real_kinematics import ArmModel, find_calibration
    from agentic_sim.real_place import camera_from_pose, make_corr
    model = ArmModel.from_file(find_calibration(a.follower_id), ROOT)
    corr = make_corr(a.corr)
    cam = camera_from_pose(a.pose or a.outputs / "d455/pose_261125.json")

    seen, rows, accepted = {}, [], {"act": [], "vla": []}
    for name in a.sessions:
        for ep in find_episodes([a.outputs / name / "episodes"]):
            m = json.loads((ep / "manifest.json").read_text())
            rv = json.loads((ep / "review.json").read_text()) if (ep / "review.json").exists() else {}
            row = {"episode": f"{name}/{ep.name}", "path": str(ep.resolve()), "mode": m.get("mode"),
                   "source": source_of(ep, m, rv), "color": (m.get("task_card") or {}).get("color"),
                   "target_mm": [((rv.get("target") or {}).get("d455") or {}).get(k) for k in ("x_mm", "y_mm", "yaw_deg")],
                   "round": rv.get("round"), "commit": (m.get("provenance") or {}).get("git_commit", "")[:7]}
            ctl = (ep / "control.jsonl").read_text().splitlines()
            key = (m.get("started_unix_ns"), ctl[0][:80] if ctl else "")
            if key in seen:
                row.update(act=False, vla=False, why=f"duplicate of {seen[key]}")
                rows.append(row)
                continue
            seen[key] = row["episode"]
            off = grasp_offset_mm(ep, rv, model, corr, cam)
            row["grasp_vs_target_mm"] = None if off is None else round(off, 1)
            plans = {"act": plan_episode(ep, segment="grasp", condition_kind="target", modes=MODES,
                                         max_still_s=a.max_still_s),
                     "vla": plan_episode(ep, segment="task", condition_kind="none", modes=MODES,
                                         max_still_s=a.max_still_s)}
            for kind, plan in plans.items():
                ok, why = plan["included"], plan["reason"]
                if ok and off is not None and off > GRASP_TARGET_MM:
                    ok, why = False, f"grasp {off:.0f} mm from the D455 target (another block?)"
                row[kind] = ok
                row[f"{kind}_why"] = why
                if ok:
                    accepted[kind].append(str(ep))
            p = plans["act"] if plans["act"]["included"] else plans["vla"]
            row.update(regrasp=bool(p.get("regrasp")), trimmed_s=round(p["stats"].get("trimmed_rows", 0) / 30, 2),
                       instruction=p.get("instruction"), act_rows=plans["act"]["stats"].get("rows"),
                       vla_rows=plans["vla"]["stats"].get("rows"))
            rows.append(row)
            print(f"{row['episode']:22s} {row['source']:8s} ACT {'ok ' if row['act'] else '-- '} "
                  f"VLA {'ok ' if row['vla'] else '-- '} "
                  + (f"grasp-target {off:4.1f} mm " if off is not None else "")
                  + ("" if row["act"] else f"[ACT: {row['act_why']}] ")
                  + ("" if row["vla"] else f"[VLA: {row['vla_why']}]"), flush=True)

    a.release.mkdir(parents=True)
    (a.release / "episodes.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    coverage(rows, cam, a.outputs, a.release / "coverage.png")
    py = sys.executable
    conv = ROOT / "tools/real_to_lerobot.py"
    common = ["--modes", ",".join(MODES), "--max-still-s", str(a.max_still_s), "--language", "en"]
    jobs = {"act": ["--segment", "grasp", "--condition", "target", "--markers", "--repo-id",
                    "local/so101-right-act-grasp"],
            "vla": ["--segment", "task", "--condition", "none", "--repo-id", "local/so101-right-vla-task"]}
    built = {}
    if not a.skip_convert:
        for kind, extra in jobs.items():
            out = a.release / {"act": "act_grasp", "vla": "vla_task"}[kind]
            t0 = time.time()
            subprocess.run([py, str(conv), *accepted[kind], "--out", str(out), *extra, *common], check=True)
            built[kind] = {"dir": out.name, "seconds": round(time.time() - t0)}
    splits = {}
    for kind, b in built.items():
        src = [json.loads(l) for l in (a.release / b["dir"] / "meta/trc_sources.jsonl").read_text().splitlines()]
        by_path = {r["path"]: r for r in rows}
        sp = {"human": [], "takeover": [], "script": []}
        for s in src:
            r = by_path[str(Path(s["source_episode"]).resolve())]
            sp[r["source"]].append(s["lerobot_episode"])
        splits[kind] = sp
    (a.release / "splits.json").write_text(json.dumps(splits, indent=1))
    (a.release / "build.json").write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "outputs": str(a.outputs.resolve()), "sessions": a.sessions,
        "converter": built, "options": {"max_still_s": a.max_still_s, "grasp_target_mm": GRASP_TARGET_MM,
                                         "modes": MODES, "corr": a.corr},
        "git": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()}, indent=1))
    n = {k: sum(r.get(k) is True for r in rows) for k in ("act", "vla")}
    write_readme(a, rows, built, splits)
    print(f"\n{len(rows)} episodes examined; ACT {n['act']}, VLA {n['vla']} -> {a.release}")
    return 0


def write_readme(a, rows, built, splits):
    def count(kind, src):
        return sum(1 for r in rows if r.get(kind) and r["source"] == src)
    srcs = ("human", "takeover", "script")
    excluded = [r for r in rows if not r.get("act") or not r.get("vla")]
    tgt = [r["target_mm"] for r in rows if r.get("act") and r["target_mm"][0] is not None]
    xs, ys = [t[0] for t in tgt], [t[1] for t in tgt]
    colours = {}
    for r in rows:
        if r.get("act"):
            colours[r["color"]] = colours.get(r["color"], 0) + 1
    lines = [
        f"# Training release {a.release.name}", "",
        f"Built {time.strftime('%Y-%m-%d %H:%M')} by `tools/build_training_release.py` (branch feat/real-lerobot-training) "
        f"from `{a.outputs}` sessions {', '.join(a.sessions)}. Raw episodes stay where they were recorded; this "
        "directory only holds the converted datasets and the lists.", "",
        "| Dataset | Segment | Condition / language | human | takeover | script | total |", "|---|---|---|---|---|---|---|",
        f"| `act_grasp/` | start pose -> lift | target (x, y mm, sin/cos 4*yaw) as observation.environment_state, "
        f"target circle on the global images | {count('act', 'human')} | {count('act', 'takeover')} | "
        f"{count('act', 'script')} | {sum(count('act', s) for s in srcs)} |",
        f"| `vla_task/` | start pose -> rise over the slot (grasp + programmed place) | English instruction "
        f"(zh in meta/trc_sources.jsonl); same-colour targets named by side | {count('vla', 'human')} | "
        f"{count('vla', 'takeover')} | {count('vla', 'script')} | {sum(count('vla', s) for s in srcs)} |", "",
        "Sources: **human** = operator demonstration with the leader (start pose -> lift, then the programmed place); "
        "**takeover** = a scripted grasp missed and the operator demonstrated the same block; **script** = the "
        "operator-like scripted grasp (tools/auto_collect.py). `splits.json` lists the LeRobot episode indices per "
        "source for the A/B/C comparison (human only / script only / all).", "",
        "Processing: still stretches keep 0.3 s (`--max-still-s`), re-grasps kept and flagged in `episodes.json`, "
        "episodes whose grasp lies more than 30 mm from the D455 target excluded (another block was taken), "
        "duplicates taken once, failed / discarded episodes never included.", "",
        f"Colours (ACT): {colours}. Targets (tray-local mm): x {min(xs, default=0):.0f}-{max(xs, default=0):.0f}, "
        f"y {min(ys, default=0):.0f}-{max(ys, default=0):.0f} (see `coverage.png`).", "",
        "## Excluded or partly excluded", "", "| Episode | Source | ACT | VLA |", "|---|---|---|---|"]
    for r in excluded:
        lines.append(f"| {r['episode']} | {r['source']} | {'ok' if r.get('act') else r.get('act_why', r.get('why', ''))} | "
                     f"{'ok' if r.get('vla') else r.get('vla_why', r.get('why', ''))} |")
    lines += ["", "## Use", "", "```bash",
              "# ACT (LeRobot 0.4.x): the condition and the target circle must be produced the same way at inference",
              f"lerobot-train --policy.type=act --dataset.repo_id=local/so101-right-act-grasp "
              f"--dataset.root={a.release / 'act_grasp'} ...",
              "# VLA (GR00T N1.x / pi0.5): the LeRobot dataset with its task strings", "```", "",
              f"Converter runs: {json.dumps(built)}"]
    (a.release / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def coverage(rows, cam, outputs, path):
    import cv2
    ref = (outputs / "current-reference.txt").read_text().strip() if (outputs / "current-reference.txt").exists() else ""
    ref_path = (outputs.parent / ref) if ref and not Path(ref).is_absolute() else Path(ref)
    img = cv2.imread(str(ref_path)) if ref else None
    if img is None:
        img = np.full((480, 640, 3), 255, np.uint8)
    colours = {"human": (0, 170, 0), "takeover": (255, 120, 0), "script": (0, 140, 255)}
    for r in rows:
        ep = Path(r["path"])
        try:
            rv = json.loads((ep / "review.json").read_text())
        except (OSError, ValueError):
            continue
        d = ((rv.get("target") or {}).get("d455") or {})
        if "pixel" not in d:
            continue
        u, v = (int(x) for x in d["pixel"])
        if r.get("act"):
            cv2.circle(img, (u, v), 4, colours[r["source"]], -1)
        else:
            cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
    lines = [f"{k}: {sum(1 for r in rows if r['source'] == k and r.get('act'))}" for k in colours]
    cv2.rectangle(img, (0, img.shape[0] - 26), (420, img.shape[0]), (255, 255, 255), -1)
    cv2.putText(img, "ACT targets  " + "  ".join(lines) + "  (red x: excluded)", (6, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    sys.exit(main())
