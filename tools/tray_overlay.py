"""Compare a reference global RGB frame with a still image or a live D455 color stream."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def load_color(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read RGB image: {path}")
    return image


def blend(reference, current, percent):
    if reference.shape != current.shape:
        raise ValueError(f"Use the same native RGB resolution: reference {reference.shape}, current {current.shape}")
    alpha = percent / 100
    return cv2.addWeighted(reference, alpha, current, 1 - alpha, 0)


def save(path, image):
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Could not save {path}")
    path.write_bytes(data.tobytes())
    print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Uncropped, unmirrored global RGB frame from the reference run")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Global RGB frame to compare offline")
    source.add_argument("--serial", help="RealSense serial number of the global D455")
    parser.add_argument("--fps", type=int, default=15, help="Live D455 color stream rate (default: 15)")
    parser.add_argument("--output", type=Path, help="Output overlay PNG; with --image, save without opening a window")
    args = parser.parse_args()

    reference = load_color(args.reference)
    current = load_color(args.image) if args.image else None
    if args.image and args.output:
        save(args.output, blend(reference, current, 50))
        return

    pipeline = None
    if args.serial:

        import pyrealsense2 as rs

        cfg = rs.config()
        cfg.enable_device(args.serial)
        height, width = reference.shape[:2]
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, args.fps)
        pipeline = rs.pipeline()
        pipeline.start(cfg)

    window = "Global RGB | Ref 0=current, 100=reference | S=save, Q=quit"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Reference %", window, 50, 100, lambda _: None)
        while True:
            if pipeline is not None:
                color = pipeline.wait_for_frames().get_color_frame()
                if not color:
                    continue
                current = np.asanyarray(color.get_data())
            overlay = blend(reference, current, cv2.getTrackbarPos("Reference %", window))
            cv2.imshow(window, overlay)
            key = cv2.waitKey(30) & 0xff
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                target = args.output or Path("tray_overlay.png")
                save(target.with_name(target.stem + "_current.png"), current)
                save(target, overlay)
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
