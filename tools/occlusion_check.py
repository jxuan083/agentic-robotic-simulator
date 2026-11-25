#!/usr/bin/env python3
"""Which follower poses hide the trays from the global D455.

  python tools/occlusion_check.py --video outputs/261125-camera/occlusion-1900 \
      --reference outputs/261125-camera/trays-1830/rgb.png --runs outputs/routes/<route>-runs/<run> ...

Tray masks come from the reference frame (pink by colour, beige as the light low-saturation tray region). A tray
pixel counts as hidden when its Lab colour differs from the reference by more than --delta. Each video frame is
matched to the nearest follower readback (control.jsonl, aligned by `clock_origin_monotonic_ns`). Writes
occlusion.csv, summary.json and worst-*.png (frame with the hidden pixels outlined) into the video folder.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def largest(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < 2:
        raise ValueError("tray not found in the reference")
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((labels == keep).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 1, -1)
    return cv2.erode(out, np.ones((7, 7), np.uint8)).astype(bool)  # stay inside the rim


def tray_masks(reference):
    hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    pink = ((h >= 150) | (h <= 8)) & (s >= 45) & (v >= 120)
    beige = (h >= 8) & (h <= 32) & (s >= 16) & (s <= 90) & (v >= 95) & ~pink
    return {"pink": largest(pink), "beige": largest(beige)}


def readbacks(runs):
    rows = []
    for run in runs:
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        origin = status.get("clock_origin_monotonic_ns")
        if origin is None:
            raise ValueError(f"{run}: no clock_origin_monotonic_ns (recorded before the stamp was added)")
        with (run / "control.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                rows.append((origin + row["readback_ns"], row["readback"], row["phase"], run.name))
    rows.sort(key=lambda r: r[0])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--delta", type=float, default=22.0, help="Lab distance counted as hidden")
    parser.add_argument("--max-gap-ms", type=float, default=50.0)
    args = parser.parse_args(argv)

    reference = cv2.imread(str(args.reference))
    masks = tray_masks(reference)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    stamps = [json.loads(line) for line in (args.video / "frames.jsonl").read_text().splitlines()]
    poses = readbacks(args.runs)
    times = np.array([p[0] for p in poses], dtype=np.int64)
    video = cv2.VideoCapture(str(args.video / "video.mp4"))
    rows, worst = [], {}
    for stamp in stamps:
        ok, frame = video.read()
        if not ok:
            break
        diff = np.linalg.norm(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32) - ref_lab, axis=2)
        hidden = diff > args.delta
        share = {tray: float(hidden[m].mean()) for tray, m in masks.items()}
        i = int(np.searchsorted(times, stamp["host_monotonic_ns"]))
        near = min((j for j in (i - 1, i) if 0 <= j < len(times)),
                   key=lambda j: abs(times[j] - stamp["host_monotonic_ns"]), default=None)
        gap_ms = abs(times[near] - stamp["host_monotonic_ns"]) / 1e6 if near is not None else None
        pose = poses[near][1] if gap_ms is not None and gap_ms <= args.max_gap_ms else None
        rows.append({"frame": stamp["frame"], "host_monotonic_ns": stamp["host_monotonic_ns"],
                     "pink_hidden": round(share["pink"], 4), "beige_hidden": round(share["beige"], 4),
                     "phase": poses[near][2] if pose else "", "pose_gap_ms": round(gap_ms, 1) if gap_ms else "",
                     **({j: round(pose[j], 1) for j in JOINTS} if pose else dict.fromkeys(JOINTS, ""))})
        for tray in masks:
            if share[tray] > worst.get(tray, (-1,))[0]:
                shown = frame.copy()
                outline = (hidden & masks[tray]).astype(np.uint8)
                contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(shown, contours, -1, (0, 0, 255), 2)
                worst[tray] = (share[tray], stamp["frame"], shown)
    with (args.video / "occlusion.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for tray, (share, frame, shown) in worst.items():
        cv2.imwrite(str(args.video / f"worst-{tray}.png"), shown)
    posed = [r for r in rows if r["shoulder_pan"] != ""]
    summary = {"frames": len(rows), "frames_with_pose": len(posed), "delta_lab": args.delta,
               "mask_pixels": {t: int(m.sum()) for t, m in masks.items()},
               "pink_hidden_over_2pct": sum(r["pink_hidden"] > .02 for r in posed),
               "worst": {t: {"share": round(v[0], 4), "frame": v[1]} for t, v in worst.items()}}
    (args.video / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
