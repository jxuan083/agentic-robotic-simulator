#!/usr/bin/env python3
"""Live D455 placement guide: tray-local grid, live block coordinates, optional target marker.

  python tools/real_place_guide.py --serial 038122250316
  python tools/real_place_guide.py --serial 038122250316 --target pink 120 80 30 red

Grid lines every 50 mm (tray-local: origin = inner corner nearest the base on its right, +x forward = up the
image, +y left). Each red/blue/yellow block shows x,y mm and yaw. With --target the target is drawn as a cross
and the nearest block of that colour reports its distance and yaw difference. S saves rgb.png + blocks.json
under --out; Q quits. Coordinates are the coarse D455 estimate (block-top parallax not corrected).
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from agentic_sim.real_locate import TRAYS, WALL_MM, color_masks, corners, locate, tray_homography  # noqa: E402,F401

COLOR_BGR = {"red": (40, 40, 230), "blue": (230, 120, 20), "yellow": (0, 200, 240), "any": (255, 0, 255)}
# Reach exploration: near to far along each tray's centre line plus the pink sides, yaw 0 (no wrist turn).
REACH_PLAN = [("pink", 60, 110), ("pink", 120, 110), ("pink", 180, 110), ("pink", 240, 110), ("pink", 150, 40),
              ("pink", 150, 180), ("beige", 60, 125), ("beige", 150, 125), ("beige", 240, 125), ("beige", 310, 125)]


def to_image(H, points):
    inv = np.linalg.inv(H)
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), inv).reshape(-1, 2)


def draw_plan(shown, homographies, plan, current):
    for i, (tray, x, y) in enumerate(plan):
        if tray not in homographies:
            continue
        c = to_image(homographies[tray], [[x, y]])[0].astype(int)
        active = i == current
        cv2.circle(shown, tuple(c), 9 if active else 6, (255, 0, 255) if active else (160, 60, 160), 2 if active else 1)
        cv2.putText(shown, str(i + 1), (c[0] - 4, c[1] - 11), cv2.FONT_HERSHEY_SIMPLEX, .45 if active else .38,
                    (255, 0, 255) if active else (160, 60, 160), 2 if active else 1, cv2.LINE_AA)


def draw_guide(image, result, homographies, target, plan=None, current=0):
    shown = image.copy()
    if plan:
        draw_plan(shown, homographies, plan, current)
    for tray, H in homographies.items():
        lx, ly = (v - 2 * WALL_MM for v in TRAYS[tray]["outer_mm"])
        for x in np.arange(0, lx + 1, 50):
            a, b = to_image(H, [[x, 0], [x, ly]])
            cv2.line(shown, tuple(a.astype(int)), tuple(b.astype(int)), (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(shown, f"{x:.0f}", tuple((a + [3, -3]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .32,
                        (60, 60, 60), 1, cv2.LINE_AA)
        for y in np.arange(0, ly + 1, 50):
            a, b = to_image(H, [[0, y], [lx, y]])
            cv2.line(shown, tuple(a.astype(int)), tuple(b.astype(int)), (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(shown, f"{y:.0f}", tuple((a + [-14, 12]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .32,
                        (60, 60, 60), 1, cv2.LINE_AA)
        origin, ex, ey = to_image(H, [[0, 0], [40, 0], [0, 40]])
        cv2.arrowedLine(shown, tuple(origin.astype(int)), tuple(ex.astype(int)), (0, 0, 255), 2, tipLength=.3)
        cv2.arrowedLine(shown, tuple(origin.astype(int)), tuple(ey.astype(int)), (0, 160, 0), 2, tipLength=.3)
        cv2.putText(shown, tray, tuple((to_image(H, [[lx, ly / 2]])[0] + [-15, -8]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 40, 40), 1, cv2.LINE_AA)
    for block in result["blocks"]:
        px = tuple(int(v) for v in block["pixel"])
        cv2.circle(shown, px, 4, (255, 255, 255), -1)
        cv2.putText(shown, f"{block['x_mm']:.0f},{block['y_mm']:.0f} {block['yaw_deg']:.0f}d", (px[0] + 6, px[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (10, 10, 10), 1, cv2.LINE_AA)
    lines = []
    if target:
        tray, x, y, yaw, color = target
        if tray in homographies:
            centre = to_image(homographies[tray], [[x, y]])[0].astype(int)
            cv2.drawMarker(shown, tuple(centre), COLOR_BGR.get(color, (255, 0, 255)), cv2.MARKER_CROSS, 26, 2)
            ux, uy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
            corners_mm = [[x + dx * 12.5 * ux - dy * 12.5 * uy, y + dx * 12.5 * uy + dy * 12.5 * ux]
                          for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            cv2.polylines(shown, [to_image(homographies[tray], corners_mm).astype(np.int32)], True,
                          COLOR_BGR.get(color, (255, 0, 255)), 1, cv2.LINE_AA)
            same = [b for b in result["blocks"] if b["tray"] == tray and color in ("any", b["color"])]
            if same:
                b = min(same, key=lambda b: (b["x_mm"] - x) ** 2 + (b["y_mm"] - y) ** 2)
                dyaw = (b["yaw_deg"] - yaw + 45) % 90 - 45
                lines.append(f"target {tray} {b['color']} ({x:.0f},{y:.0f},{yaw:.0f}d): off {b['x_mm'] - x:+.0f} x "
                             f"{b['y_mm'] - y:+.0f} y mm, yaw {dyaw:+.0f}d")
            else:
                lines.append(f"target {tray} {color} ({x:.0f},{y:.0f},{yaw:.0f}d): no {color} block in {tray}")
        else:
            lines.append(f"target tray {tray} not found")
    lines += [f"WARNING {w}" for w in result["warnings"]]
    for i, text in enumerate(lines):
        cv2.putText(shown, text, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(shown, text, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
    return shown


def homographies_for(image, result):
    return {tray: tray_homography(np.array(info["quad_px"], np.float32), tray) for tray, info in result["trays"].items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--target", nargs=5, metavar=("TRAY", "X", "Y", "YAW", "COLOR"))
    parser.add_argument("--trays", type=Path, help="fixed tray corners JSON (hands over a tray cannot move the grid)")
    parser.add_argument("--reach-plan", action="store_true",
                        help="show the 10 reach-exploration points; n / p switch the current one")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / f"{time.strftime('%y%m%d')}-camera" / "guide")
    args = parser.parse_args(argv)
    target = None
    plan = REACH_PLAN if args.reach_plan else None
    current = 0
    if args.target:
        tray, x, y, yaw, color = args.target
        target = (tray, float(x), float(y), float(yaw), color)

    import pyrealsense2 as rs

    cfg = rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    pipeline.start(cfg)
    window = "D455 placement guide | S save | Q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    quads = json.loads(args.trays.read_text(encoding="utf-8"))["quads"] if args.trays else None
    frozen = None  # without fixed trays: re-fit the outlines every second so a moved tray is picked up
    try:
        while True:
            image = np.asanyarray(pipeline.wait_for_frames(2000).get_color_frame().get_data()).copy()
            result = locate(image, quads)
            if quads or frozen is None or time.monotonic() - frozen[0] > 1.0:
                frozen = (time.monotonic(), homographies_for(image, result))
            if plan:
                tray, x, y = plan[current]
                target = (tray, float(x), float(y), 0.0, "any")
            shown = draw_guide(image, result, frozen[1], target, plan, current)
            if plan:
                cv2.putText(shown, f"point {current + 1}/{len(plan)}  (n next, p previous, s save)", (8, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if plan and key in (ord("n"), ord("p")):
                current = (current + (1 if key == ord("n") else -1)) % len(plan)
            if key == ord("s"):
                folder = args.out / (time.strftime("%H%M%S") + (f"-point{current + 1:02}" if plan else ""))
                folder.mkdir(parents=True, exist_ok=False)
                cv2.imwrite(str(folder / "rgb.png"), image)
                cv2.imwrite(str(folder / "guide.png"), shown)
                (folder / "blocks.json").write_text(json.dumps(result | {"target": target}, ensure_ascii=False,
                                                               indent=2) + "\n", encoding="utf-8")
                print(f"saved {folder}", flush=True)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
