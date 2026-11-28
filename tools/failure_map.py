#!/usr/bin/env python3
"""Where the scripted grasps fail: outputs/d455/auto_failures.jsonl (tools/auto_collect.py) and the successful
scripted / human grasps of the auto sessions, drawn on the D455 reference frame with the safe grasp zone.

  python tools/failure_map.py [--sessions outputs/*-auto] [--out outputs/d455/failure_map.png]

Red X: a scripted miss (colour letter beside it); green dot: a scripted grasp that held; blue ring: a human takeover.
The terminal lists the misses with their radius, colour and sampled offsets.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


def main(argv=None) -> int:
    import cv2
    from agentic_sim.real_place import camera_from_pose
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(ROOT / "outputs/d455/auto_failures.jsonl"))
    ap.add_argument("--sessions", nargs="*", default=sorted(glob.glob(str(ROOT / "outputs/*-auto"))))
    ap.add_argument("--pose", default=str(ROOT / "outputs/d455/pose_261125.json"))
    ap.add_argument("--reference", default=None, help="D455 frame to draw on (default: outputs/current-reference.txt)")
    ap.add_argument("--out", default=str(ROOT / "outputs/d455/failure_map.png"))
    a = ap.parse_args(argv)
    cam = camera_from_pose(a.pose)
    ref = a.reference or (ROOT / "outputs/current-reference.txt").read_text().strip()
    img = cv2.imread(str(ROOT / ref) if not Path(ref).is_absolute() else ref)
    if Path(ref).parent.joinpath("overlay-zone.png").exists():
        img = cv2.imread(str(Path(ref).parent / "overlay-zone.png"))
    px = lambda x, y: tuple(int(v) for v in cam.project([[x, y, 0.0295]])[0])  # noqa: E731
    ok, human = [], []
    for sess in a.sessions:
        for e in sorted(Path(sess, "episodes").glob("*")):
            try:
                rv = json.loads((e / "review.json").read_text())
            except (OSError, ValueError):
                continue
            t = (rv.get("script") or {}).get("target_mount_m")
            if t:
                ok.append(t)
            elif rv.get("takeover"):
                d = (rv.get("target") or {}).get("d455") or {}
                if "pixel" in d:
                    human.append(d["pixel"])
    for t in ok:
        cv2.circle(img, px(t[0], t[1]), 4, (0, 200, 0), -1)
    for p in human:
        cv2.circle(img, (int(p[0]), int(p[1])), 9, (255, 120, 0), 2)
    fails = [json.loads(l) for l in Path(a.log).read_text().splitlines()] if Path(a.log).exists() else []
    for f in fails:
        u, v = px(f["x"], f["y"])
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(img, f["color"][0].upper(), (u + 7, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    label = f"scripted held {len(ok)}  missed {len(fails)}  human takeovers {len(human)}"
    cv2.rectangle(img, (0, img.shape[0] - 26), (12 + 9 * len(label), img.shape[0]), (255, 255, 255), -1)
    cv2.putText(img, label, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(a.out, img)
    for f in fails:
        st = f.get("style") or {}
        print(f"{f['time']} {Path(f['session']).name}/{f['episode']} {f['color']:7s} r {f['r_mm']:.0f} mm "
              f"x {f['x'] * 1000:.0f} y {f['y'] * 1000:.0f} yaw {f['yaw_deg']:.0f}  {f['why']}  "
              f"offset {np.round(st.get('offset_mm', []), 1).tolist()} yaw_err {st.get('yaw_err_deg') and round(st['yaw_err_deg'])}")
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
