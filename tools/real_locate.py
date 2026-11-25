#!/usr/bin/env python3
"""Coarse block positions (tray-local mm) from the global D455, or from a saved colour frame.

  python tools/real_locate.py --serial 038122250316 --out outputs/261125-camera/locate-1910
  python tools/real_locate.py --image outputs/<session>/episodes/<id>/global/<frame>.png --out <dir>

Writes rgb.png (live capture only), blocks.json and annotated.png (tray outline, origin dot, x,y mm and yaw).
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

from agentic_sim.real_locate import draw, locate  # noqa: E402


def capture(serial, width=640, height=480, fps=30, warmup=30):
    import pyrealsense2 as rs

    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline = rs.pipeline()
    pipeline.start(cfg)
    try:
        for _ in range(warmup):  # let auto exposure settle
            pipeline.wait_for_frames(2000)
        return np.asanyarray(pipeline.wait_for_frames(2000).get_color_frame().get_data()).copy()
    finally:
        pipeline.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--serial")
    source.add_argument("--image", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"cannot read {args.image}")
    else:
        image = capture(args.serial)
        cv2.imwrite(str(args.out / "rgb.png"), image)
    result = locate(image)
    result |= {"source": str(args.image) if args.image else f"D455 {args.serial} live", "captured_unix": time.time()}
    (args.out / "blocks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cv2.imwrite(str(args.out / "annotated.png"), draw(image, result))
    for warning in result["warnings"]:
        print("WARNING:", warning)
    for b in result["blocks"]:
        print(f"{b['tray']:5s} {b['color']:6s} x {b['x_mm']:6.1f} mm  y {b['y_mm']:6.1f} mm  yaw {b['yaw_deg']:4.0f} deg"
              + ("  (tray clipped)" if b["tray_clipped"] else ""))
    print(f"saved {args.out}/blocks.json and annotated.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
