"""Stable in-memory and JSON contracts for perception results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Detection:
    detection_id: int
    track_id: int | None
    class_name: str
    confidence: float
    color: str
    bbox_xyxy: tuple[float, float, float, float]
    centroid_uv: tuple[float, float]
    mask: np.ndarray


@dataclass(slots=True)
class FrameDetections:
    frame_id: int | str
    timestamp_s: float | None
    image_size: tuple[int, int]  # (height, width)
    detections: list[Detection]
    timestamp_source: str | None = None

    def to_dict(self, mask_paths: dict[int, str] | None = None) -> dict[str, Any]:
        """Serialize metadata; masks are represented by paths, never huge arrays."""
        rows = []
        for detection in self.detections:
            rows.append({
                "detection_id": detection.detection_id,
                "track_id": detection.track_id,
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "color": detection.color,
                "bbox_xyxy": list(detection.bbox_xyxy),
                "centroid_uv": list(detection.centroid_uv),
                "mask_path": (mask_paths or {}).get(detection.detection_id),
            })
        return {
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "timestamp_source": self.timestamp_source,
            "image_size": {"height": self.image_size[0], "width": self.image_size[1]},
            "detections": rows,
        }
