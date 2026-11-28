#!/usr/bin/env python3
"""Record the global D455 colour stream to MP4 with per-frame host monotonic stamps (no robot access).

  python tools/record_global.py --serial 038122250316 --out outputs/261125-camera/occlusion-1900 --seconds 300

Stops after --seconds or on Ctrl-C / SIGTERM. Writes video.mp4, frames.jsonl
({frame, host_monotonic_ns, device_ms}) and meta.json. Host stamps use CLOCK_MONOTONIC, the same clock the
route tool logs as `clock_origin_monotonic_ns`, so frames can be matched to follower readbacks afterwards.
"""

import argparse
import json
from pathlib import Path
import signal
import time

import cv2
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    import pyrealsense2 as rs

    args.out.mkdir(parents=True, exist_ok=False)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append("term"))
    signal.signal(signal.SIGINT, lambda *_: stop.append("int"))
    cfg = rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    writer = cv2.VideoWriter(str(args.out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (args.width, args.height))
    meta = {"camera": "RealSense D455", "serial": args.serial, "width": args.width, "height": args.height,
            "fps_set": args.fps, "stream": "color bgr8", "clock": "host CLOCK_MONOTONIC ns",
            "intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                           "model": str(intr.model), "coeffs": list(intr.coeffs)},
            "started_unix": time.time()}
    count = 0
    end = time.monotonic() + args.seconds
    try:
        with (args.out / "frames.jsonl").open("x", buffering=1) as stamps:
            print(f"recording {args.out} (Ctrl-C stops)", flush=True)
            while not stop and time.monotonic() < end:
                frames = pipeline.wait_for_frames(2000)
                host = time.monotonic_ns()
                color = frames.get_color_frame()
                if not color:
                    continue
                writer.write(np.asanyarray(color.get_data()))
                stamps.write(json.dumps({"frame": count, "host_monotonic_ns": host,
                                         "device_ms": color.get_timestamp()}) + "\n")
                count += 1
    finally:
        writer.release()
        pipeline.stop()
        meta |= {"frames": count, "stopped_unix": time.time()}
        (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"saved {count} frames to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
