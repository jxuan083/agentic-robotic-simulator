#!/usr/bin/env python3
"""Put the global D455 back where it was (new tripod spot, practice room, finals venue): live height and tilt from
the table plane in depth, the reference frame's edges over the live image for the in-plane position.

  tools/lab_day.sh d455-restore                 # live; target = the fitted pose ($POSE, else 11/25), ghost = $REFERENCE
  tools/lab_day.sh d455-restore --once          # no window: 15 frames -> report + outputs/d455/restore-*.json

Order on site:
  1. Tripod roughly in place, camera roughly top-down, robot base at the bottom edge as in the reference.
  2. Height: raise / lower until "H" is green (+-2 mm).
  3. Tilts: turn the head until "top-bottom" and "left-right" are green (+-0.2 deg = +-2 mm across the frame).
     Height and tilt interact a little: repeat 2-3 once.
  4. In-plane: slide / turn the tripod until the magenta edges sit on the robot base and arm (and the trays, if they
     are where the reference had them); the cross is the base origin under the target pose. Lock, re-check 2-3.
  5. Then the exact pose for coordinates: tools/lab_day.sh d455-capture / d455-fit, and a few aim checks (reach k)
     if the fit moved the pose by more than a few mm.

Keys (window): e edges / blend / live,  s save snapshot,  t save this plane as a new target,  q quit.
Keep hands and the arm out of the middle of the view while reading: they are ignored, but less table means more noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from agentic_sim.real_camera_restore import (TOL_HEIGHT_MM, TOL_TILT_DEG, advice, compare,  # noqa: E402
                                             table_plane, target_from_pose)

DEFAULT_POSE = ROOT / "outputs/d455/pose_261125.json"
SMOOTH = 8      # frames in the running median


def load_target(path):
    """A fitted pose (mount_from_color) or a target saved with t ({"table": {normal, height_m}})."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "mount_from_color" in data:
        return target_from_pose(data), data.get("intrinsics_640x480"), data
    table = data["table"]
    return {"normal": np.asarray(table["normal"], float), "height_m": float(table["height_m"])}, data.get(
        "intrinsics"), data


def median_compare(history):
    keys = ("height_mm", "forward_deg", "side_deg")
    out = {k: float(np.median([h[k] for h in history])) for k in keys}
    out["depth_mm"] = {k: float(np.median([h["depth_mm"][k] for h in history])) for k in history[-1]["depth_mm"]}
    out["share"] = float(np.median([h["share"] for h in history]))
    out["ok"] = abs(out["height_mm"]) <= TOL_HEIGHT_MM and abs(out["forward_deg"]) <= TOL_TILT_DEG \
        and abs(out["side_deg"]) <= TOL_TILT_DEG
    return out


def record(c, plane, args, K):
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "target": str(args.target), "intrinsics": K,
            "compare": c, "table": {"normal": [float(v) for v in plane["normal"]], "height_m": plane["height_m"]},
            "tolerance": {"height_mm": TOL_HEIGHT_MM, "tilt_deg": TOL_TILT_DEG}}


def ghost_edges(reference_path, shape):
    import cv2
    ref = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if ref is None or ref.shape[:2] != shape[:2]:
        return None, None
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 110)
    return ref, edges > 0


def annotate(bgr, c, view, ref, edges, pose):
    import cv2
    out = bgr.copy()
    if view == "blend" and ref is not None:
        out = cv2.addWeighted(ref, 0.5, out, 0.5, 0)
    elif view == "edges" and edges is not None:
        out[edges] = (255, 0, 255)
    if pose is not None:      # base origin and +X / +Y (5 cm) under the target pose
        from agentic_sim.real_d455 import Camera
        cam = Camera(pose["intrinsics_640x480"], pose["mount_from_color"]["R"], pose["mount_from_color"]["t"])
        o, x, y = cam.project([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0]]).astype(int)
        cv2.arrowedLine(out, tuple(o), tuple(x), (0, 0, 255), 2)
        cv2.arrowedLine(out, tuple(o), tuple(y), (0, 255, 0), 2)
        cv2.drawMarker(out, tuple(o), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
    if c is None:
        cv2.putText(out, "no table plane", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return out
    d = c["depth_mm"]
    rows = [(f"H {c['height_mm']:+.1f} mm", abs(c["height_mm"]) <= TOL_HEIGHT_MM),
            (f"top-bottom {d['top'] - d['bottom']:+.1f} mm ({c['forward_deg']:+.2f} deg)",
             abs(c["forward_deg"]) <= TOL_TILT_DEG),
            (f"left-right {d['left'] - d['right']:+.1f} mm ({c['side_deg']:+.2f} deg)",
             abs(c["side_deg"]) <= TOL_TILT_DEG),
            (f"table {c['share'] * 100:.0f}% of view  [{view}]  e s t q", True)]
    cv2.rectangle(out, (0, 0), (360, 22 * len(rows) + 8), (0, 0, 0), -1)
    for i, (text, good) in enumerate(rows):
        colour = (255, 255, 255) if i == 3 else ((0, 220, 0) if good else (0, 0, 255))
        cv2.putText(out, text, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1 if i == 3 else 2)
    return out


def save(out_dir, stem, payload, bgr=None, depth=None, image=None):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    if image is not None:
        cv2.imwrite(str(out_dir / f"{stem}.png"), image)
    if bgr is not None:
        np.savez_compressed(out_dir / f"{stem}.npz", bgr=bgr, depth=depth)
    return path


def frames(args):
    if args.npz:
        z = np.load(args.npz)
        while True:
            yield z["bgr"], z["depth"]
    from agentic_sim.real_d455 import open_aligned
    read, _, stop = open_aligned(args.serial)
    try:
        for _ in range(10):      # auto exposure / temporal filter settle
            read()
        while True:
            yield read()
    finally:
        stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--serial", help="global D455 serial")
    src.add_argument("--npz", type=Path, help="saved frame (bgr + aligned depth), e.g. outputs/d455/locate-*.npz")
    ap.add_argument("--target", type=Path, default=DEFAULT_POSE,
                    help="fitted pose json (outputs/d455/pose_*.json) or a target saved with t")
    ap.add_argument("--reference", type=Path, help="reference RGB frame for the edge ghost (same camera position)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/d455")
    ap.add_argument("--once", action="store_true", help="no window: average 15 frames, print, save, exit")
    args = ap.parse_args(argv)

    target, K, target_data = load_target(args.target)
    pose = target_data if "mount_from_color" in target_data else None
    if K is None:
        sys.exit("target has no intrinsics")
    history = deque(maxlen=SMOOTH if not args.once else 15)
    stream = frames(args)

    if args.once:
        plane = None
        for bgr, depth in stream:
            plane = table_plane(depth, K)
            if plane is not None:
                history.append(compare(plane, target, K))
            if len(history) == history.maxlen or args.npz:
                break
        if not history:
            sys.exit("no table plane in view")
        c = median_compare(history)
        print("\n".join(advice(c)))
        path = save(args.out, f"restore-{time.strftime('%H%M%S')}", record(c, plane, args, K))
        print(f"{'OK：高度與傾角已還原' if c['ok'] else '尚未還原'} → {path}")
        return 0 if c["ok"] else 1

    import cv2
    ref, edges = ghost_edges(args.reference, (480, 640)) if args.reference else (None, None)
    if args.reference and ref is None:
        print(f"reference {args.reference} unreadable or not 640x480: no ghost")
    views = ["edges", "blend", "live"]
    view, last_print, plane = views[0], 0.0, None
    window = "D455 restore | e view  s save  t new target  q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    for bgr, depth in stream:
        plane = table_plane(depth, K)
        if plane is not None:
            history.append(compare(plane, target, K))
        c = median_compare(history) if history else None
        image = annotate(bgr, c, view, ref, edges, pose)
        cv2.imshow(window, image)
        if c is not None and time.monotonic() - last_print > 2.0:
            print(" | ".join(advice(c)), flush=True)
            last_print = time.monotonic()
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("e"):
            view = views[(views.index(view) + 1) % len(views)]
        if key in (ord("s"), ord("t")) and c is not None and plane is not None:
            stem = f"restore-{time.strftime('%H%M%S')}" if key == ord("s") else f"restore-target-{time.strftime('%y%m%d-%H%M')}"
            path = save(args.out, stem, record(c, plane, args, K), bgr, depth, image)
            print(f"saved {path}" + ("  (use it: --target " + str(path) + ")" if key == ord("t") else ""))
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
