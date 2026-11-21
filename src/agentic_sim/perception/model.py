"""Ultralytics-backed RGB block instance segmentation with reusable model state."""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .color import ColorConfig, estimate_color
from .types import Detection, FrameDetections


class PerceptionError(RuntimeError):
    """Actionable perception setup or inference error."""


@dataclass(slots=True)
class PerceptionConfig:
    model_path: str = "weights/block-seg.pt"
    device: str = ""
    confidence: float = 0.35
    iou: float = 0.7
    image_size: int = 640
    target_class_id: int = 0
    class_name: str = "block"
    tracker: str = "bytetrack.yaml"
    color: ColorConfig = field(default_factory=ColorConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PerceptionConfig":
        try:
            with open(path, encoding="utf-8") as stream:
                data = yaml.safe_load(stream) or {}
            if not isinstance(data, dict):
                raise TypeError("top-level YAML value must be a mapping")
            color = ColorConfig(**data.pop("color", {}))
            if color.hue_ranges:
                color.hue_ranges = {name: tuple(tuple(pair) for pair in ranges)
                                    for name, ranges in color.hue_ranges.items()}
            return cls(color=color, **data)
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            raise PerceptionError(f"invalid perception config {path}: {exc}") from exc


class BlockPerception:
    """Load a segmentation model once, then call ``predict`` for RGB frames.

    Omitted ``frame_id`` values increment per instance and reset to zero only
    after ``reset_tracker()``. Detection IDs are frame-local. Track IDs only come
    from the configured tracker and are always None when ``track=False``.
    """

    def __init__(self, config: PerceptionConfig, model: Any | None = None):
        self.config = config
        self._model = model
        self._next_frame_id = 0
        if model is None:
            path = Path(config.model_path)
            if not path.is_file():
                raise PerceptionError(f"segmentation checkpoint not found: {path}. Train a block-segmentation model; generic COCO weights do not recognize project blocks.")
            _disable_ultralytics_autoinstall(required=True)
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(path))
            except ImportError as exc:
                raise PerceptionError("Ultralytics is missing; install the isolated environment from `requirements-perception.txt`.") from exc
            except Exception as exc:
                raise PerceptionError(f"could not load model {path}: {exc}") from exc
        if getattr(self._model, "task", None) != "segment":
            raise PerceptionError("checkpoint is not an Ultralytics segmentation model (expected task='segment')")

    def reset_tracker(self) -> None:
        """Reset temporal association while preserving Ultralytics predictor contracts."""
        predictor = getattr(self._model, "predictor", None)
        if predictor is not None:
            if isinstance(getattr(predictor, "trackers", None), list):
                for tracker in predictor.trackers:
                    if hasattr(tracker, "reset"):
                        tracker.reset()
            # Ultralytics stores one video path per batch element; retain its list shape.
            if isinstance(getattr(predictor, "vid_path", None), list):
                predictor.vid_path = [None] * len(predictor.vid_path)
        self._next_frame_id = 0

    def predict(self, rgb: np.ndarray, frame_id: int | str | None = None,
                timestamp_s: float | None = None, timestamp_source: str | None = None,
                track: bool = False) -> FrameDetections:
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("input must be an HxWx3 uint8 RGB numpy array")
        if frame_id is None:
            frame_id = self._next_frame_id
        if isinstance(frame_id, int):
            self._next_frame_id = max(self._next_frame_id, frame_id + 1)
        # Ultralytics accepts OpenCV-style BGR arrays. Keep the public contract RGB.
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        kwargs = dict(conf=self.config.confidence, iou=self.config.iou, imgsz=self.config.image_size,
                      device=self.config.device or None, classes=[self.config.target_class_id], retina_masks=True, verbose=False)
        try:
            if track:
                _require_tracker_dependency()
                result = self._model.track(bgr, tracker=self.config.tracker, persist=True, **kwargs)[0]
            else:
                with _without_tracker_callbacks(self._model):
                    result = self._model.predict(bgr, **kwargs)[0]
        except PerceptionError:
            raise
        except Exception as exc:
            raise PerceptionError(f"Ultralytics inference failed: {exc}") from exc
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return FrameDetections(frame_id, _timestamp(timestamp_s), rgb.shape[:2], [],
                                   timestamp_source or ("caller" if timestamp_s is not None else None))
        xyxy = _numpy(boxes.xyxy)
        confs = _numpy(boxes.conf)
        classes = _numpy(boxes.cls).astype(int)
        result_masks = getattr(result, "masks", None)
        if len(xyxy) == 0 and result_masks is None:
            return FrameDetections(frame_id, _timestamp(timestamp_s), rgb.shape[:2], [],
                                   timestamp_source or ("caller" if timestamp_s is not None else None))
        if result_masks is None:
            raise PerceptionError("segmentation model returned boxes without instance masks")
        masks = _numpy(result_masks.data)
        h, w = rgb.shape[:2]
        if masks.ndim != 3 or masks.shape[0] != len(xyxy) or len(confs) != len(xyxy) or len(classes) != len(xyxy):
            raise PerceptionError("segmentation result has inconsistent box, confidence, class, and mask counts")
        if masks.shape[1:] != (h, w):
            raise PerceptionError(
                f"Ultralytics returned masks at {masks.shape[1:]} instead of original image size {(h, w)}; "
                "retina_masks=True is required and masks are not resized because that could undo letterboxing incorrectly"
            )
        track_ids = None
        ids = getattr(boxes, "id", None)
        if ids is not None:
            track_ids = _numpy(ids).astype(int)
        detections = []
        for idx, (box, conf, cls_id, raw_mask) in enumerate(zip(xyxy, confs, classes, masks)):
            if cls_id != self.config.target_class_id:
                continue
            mask = np.asarray(raw_mask > 0.5, dtype=bool)
            yy, xx = np.nonzero(mask)
            if not len(xx):
                continue
            detections.append(Detection(
                detection_id=len(detections), track_id=int(track_ids[idx]) if track_ids is not None and idx < len(track_ids) else None,
                class_name=self.config.class_name, confidence=float(conf), color=estimate_color(rgb, mask, self.config.color),
                bbox_xyxy=tuple(float(v) for v in box),
                centroid_uv=(float(xx.mean()), float(yy.mean())), mask=mask,
            ))
        # IDs are deliberately absent in prediction-only mode, even if callbacks or a
        # supplied model happen to attach stale tracker IDs to result boxes.
        if not track:
            for detection in detections:
                detection.track_id = None
        return FrameDetections(frame_id, _timestamp(timestamp_s), rgb.shape[:2], detections,
                               timestamp_source or ("caller" if timestamp_s is not None else None))


def _timestamp(value: float | None) -> float | None:
    return None if value is None else float(value)


def _disable_ultralytics_autoinstall(*, required: bool) -> None:
    """Disable Ultralytics' runtime package installation even if already imported."""
    os.environ["YOLO_AUTOINSTALL"] = "False"
    if not required and "ultralytics" not in __import__("sys").modules:
        return
    try:
        utils = importlib.import_module("ultralytics.utils")
        checks = importlib.import_module("ultralytics.utils.checks")
    except ImportError as exc:
        if required:
            raise PerceptionError(
                "Ultralytics is missing; install the isolated environment from `requirements-perception.txt`."
            ) from exc
        return
    # checks.py imports this constant by value; set both module copies.
    utils.AUTOINSTALL = False
    checks.AUTOINSTALL = False


def _require_tracker_dependency() -> None:
    try:
        importlib.import_module("lap")
    except ImportError as exc:
        raise PerceptionError(
            "ByteTrack requires the `lap` package; install it in this Python environment "
            "with `python -m pip install 'lap>=0.5.12'` (no automatic installation is attempted)"
        ) from exc


def _is_tracker_callback(callback: Any) -> bool:
    while hasattr(callback, "func"):
        callback = callback.func
    return (
        getattr(callback, "__module__", "").startswith("ultralytics.trackers")
        and getattr(callback, "__name__", "") in {"on_predict_start", "on_predict_postprocess_end"}
    )


@contextmanager
def _without_tracker_callbacks(model: Any):
    """Temporarily remove Ultralytics tracker callbacks from model and live predictor.

    Ultralytics 8.3 registers callbacks that may filter Results by matched tracks,
    without checking ``predictor.args.mode``. Suspend them for plain prediction,
    then restore the exact callback lists so a later ``track=True`` still works.
    """
    saved: list[tuple[list, list]] = []
    seen: set[int] = set()
    for target in (model, getattr(model, "predictor", None)):
        callbacks = getattr(target, "callbacks", None)
        if not isinstance(callbacks, dict):
            continue
        for values in callbacks.values():
            if not isinstance(values, list) or id(values) in seen:
                continue
            seen.add(id(values))
            original = list(values)
            filtered = [callback for callback in values if not _is_tracker_callback(callback)]
            if len(filtered) != len(values):
                values[:] = filtered
                saved.append((values, original))
    try:
        yield
    finally:
        for values, original in reversed(saved):
            values[:] = original


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


