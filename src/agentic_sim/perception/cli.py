"""Headless image and video inference command line."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from .annotations import export_yolo_candidates, reset_yolo_export


def _write_frame(frame, output: Path, image_rgb: np.ndarray, cv2, jsonl: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for det in frame.detections:
        mask_path = output / f"frame_{frame.frame_id}_det_{det.detection_id}_mask.png"
        if not cv2.imwrite(str(mask_path), det.mask.astype(np.uint8) * 255):
            raise OSError(f"failed to write mask image: {mask_path}")
        paths[det.detection_id] = str(mask_path)
    overlay = image_rgb.copy()
    colors = ((255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 200, 40), (190, 80, 255))
    for det in frame.detections:
        color = np.asarray(colors[det.detection_id % len(colors)], dtype=np.uint8)
        overlay[det.mask] = (overlay[det.mask].astype(np.float32) * 0.80 + color * 0.20).astype(np.uint8)
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox_xyxy)
        label = f"det {det.detection_id} track {det.track_id} {det.color} {det.confidence:.2f}"
        text_y = max(15, y1 - 5)
        cv2.rectangle(overlay, (x1, text_y - 13), (x1 + len(label) * 7, text_y + 3), (30, 30, 30), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(v) for v in color), 2)
        cv2.putText(overlay, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    if not cv2.imwrite(str(output / f"frame_{frame.frame_id}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write overlay image for frame {frame.frame_id}")
    record = frame.to_dict(paths)
    line = json.dumps(record, ensure_ascii=False)
    if jsonl:
        with (output / "detections.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    else:
        (output / f"frame_{frame.frame_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Segment and estimate colors of blocks in RGB images or video")
    parser.add_argument("source", help="image/video path")
    parser.add_argument("--backend", choices=("yolo", "classical"), default="yolo")
    parser.add_argument("--config", default=None, help="backend configuration YAML")
    parser.add_argument("--output", default="outputs/perception")
    parser.add_argument("--track", action="store_true", help="associate IDs across video frames")
    parser.add_argument("--export-yolo", default=None,
                        help="write conservative single-polygon candidate labels and needs_review manifest")
    parser.add_argument("--jsonl", action="store_true", help="write JSONL instead of one JSON per image/frame")
    parser.add_argument("--preview", action="store_true", help="show optional OpenCV preview (requires display)")
    parser.add_argument("--fps", type=float, default=None,
                        help="explicit source frame rate when video metadata does not provide a valid FPS")
    args = parser.parse_args(argv)
    try:
        import cv2
    except ImportError:
        print("OpenCV missing; install with `pip install 'ultralytics>=8.3' 'opencv-python-headless>=4.9'`", file=sys.stderr)
        return 2
    source = Path(args.source)
    if not source.is_file():
        print(f"could not read image or open video: {source} (file does not exist)", file=sys.stderr)
        return 2
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        print("--fps must be a finite positive number", file=sys.stderr)
        return 2
    try:
        if args.backend == "classical":
            from .classical import ClassicalBlockPerception, ClassicalConfig
            config_path = args.config or "configs/perception-classical.yaml"
            detector = ClassicalBlockPerception(ClassicalConfig.from_yaml(config_path))
        else:
            from .model import BlockPerception, PerceptionConfig
            config_path = args.config or "configs/perception.yaml"
            detector = BlockPerception(PerceptionConfig.from_yaml(config_path))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"perception setup error: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output)
    cap = None
    try:
        if args.export_yolo:
            reset_yolo_export(args.export_yolo, protect_paths=(source,))
        if args.jsonl:
            output.mkdir(parents=True, exist_ok=True)
            (output / "detections.jsonl").write_text("", encoding="utf-8")
        image_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image_bgr is not None:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            frame = detector.predict(rgb, frame_id=0, timestamp_s=0.0,
                                     timestamp_source="image", track=args.track)
            _write_frame(frame, output, rgb, cv2, args.jsonl)
            if args.export_yolo:
                export_yolo_candidates(frame, cv2, args.export_yolo, source.name, rgb)
            if args.preview:
                cv2.imshow("block perception", image_bgr)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return 0
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            print(f"could not read image or open video: {source}", file=sys.stderr)
            return 2
        frame_id = 0
        metadata_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = args.fps if args.fps is not None else metadata_fps
        last_media_ms = None
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            media_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if np.isfinite(fps) and fps > 0:
                timestamp = frame_id / fps
                timestamp_source = "frame_rate"
            elif (np.isfinite(media_ms) and media_ms >= 0
                  and (last_media_ms is None or media_ms > last_media_ms)):
                timestamp = media_ms / 1000.0
                timestamp_source = "media_position"
                last_media_ms = media_ms
            else:
                timestamp, timestamp_source = None, None
            frame = detector.predict(rgb, frame_id=frame_id, timestamp_s=timestamp,
                                     timestamp_source=timestamp_source, track=args.track)
            _write_frame(frame, output, rgb, cv2, args.jsonl)
            if args.export_yolo:
                export_yolo_candidates(frame, cv2, args.export_yolo, source.name, rgb,
                                       frame_tag=f"_frame_{frame_id:06d}")
            if args.preview:
                cv2.imshow("block perception", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_id += 1
        if frame_id == 0:
            print(f"video contains no readable frames: {source}", file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"perception inference/output error: {exc}", file=sys.stderr)
        return 2
    finally:
        if cap is not None:
            cap.release()
        if args.preview:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
