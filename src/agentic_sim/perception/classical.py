"""OpenCV classical block candidates and lightweight color/position association."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .color import GREEN_VALUE_SPLIT, ColorConfig, estimate_color
from .types import Detection, FrameDetections


@dataclass(slots=True)
class ClassicalConfig:
    min_saturation: int = 48
    min_value: int = 35
    max_value: int = 252
    min_area_px: int = 120
    max_area_fraction: float = 0.25
    min_side_px: int = 5
    max_aspect_ratio: float = 4.0
    min_solidity: float = 0.42
    close_kernel: int = 3
    open_kernel: int = 3
    # Split the foreground by hue band before connected components, so touching blocks of different colours (common
    # in close wrist views) become separate candidates. Bands come from color.hue_ranges (shared bands merged).
    split_by_hue: bool = False
    split_green_by_value: bool = False   # with split_by_hue: also separate light / dark green by value
    roi_xyxy_normalized: tuple[float, float, float, float] | None = None
    track_max_distance_px: float = 90.0
    track_max_missed_frames: int = 3
    color: ColorConfig = field(default_factory=ColorConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClassicalConfig":
        try:
            with open(path, encoding="utf-8") as stream:
                data = yaml.safe_load(stream) or {}
            if not isinstance(data, dict):
                raise TypeError("top-level YAML value must be a mapping")
            color_data = data.pop("color", {})
            if color_data:
                color = ColorConfig(**color_data)
                if color.hue_ranges:
                    color.hue_ranges = {name: tuple(tuple(pair) for pair in ranges)
                                        for name, ranges in color.hue_ranges.items()}
            else:
                color = ColorConfig()
            if data.get("roi_xyxy_normalized") is not None:
                data["roi_xyxy_normalized"] = tuple(data["roi_xyxy_normalized"])
            return cls(color=color, **data)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"invalid classical perception config {path}: {exc}") from exc

    def validate(self) -> None:
        if not 0 <= self.min_saturation <= 255 or not 0 <= self.min_value < self.max_value <= 255:
            raise ValueError("classical HSV saturation/value limits must be in [0,255]")
        if self.min_area_px < 1 or not 0 < self.max_area_fraction <= 1 or self.min_side_px < 1:
            raise ValueError("classical area and side limits must be positive")
        if self.max_aspect_ratio < 1 or not 0 <= self.min_solidity <= 1:
            raise ValueError("classical shape limits are invalid")
        if self.close_kernel < 1 or self.open_kernel < 1:
            raise ValueError("morphology kernels must be positive")
        if self.track_max_distance_px <= 0 or self.track_max_missed_frames < 0:
            raise ValueError("classical tracking limits are invalid")
        if self.roi_xyxy_normalized is not None:
            x1, y1, x2, y2 = self.roi_xyxy_normalized
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("roi_xyxy_normalized must be an increasing rectangle in [0,1]")


class ClassicalBlockPerception:
    """RGB-only saturation foreground segmentation; confidence is heuristic."""

    class_name = "block"

    def __init__(self, config: ClassicalConfig | None = None, cv2_module: Any | None = None):
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as exc:
                raise RuntimeError("OpenCV missing; install requirements-perception-classical.txt") from exc
        self.cv2 = cv2_module
        self.config = config or ClassicalConfig()
        self.config.validate()
        self._next_frame_id = 0
        self._next_track_id = 0
        self._tracks: dict[int, dict[str, Any]] = {}

    def reset_tracker(self) -> None:
        """Forget active associations; future track IDs remain globally increasing."""
        self._tracks.clear()
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
        height, width = rgb.shape[:2]
        hsv = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2HSV)
        saturation, value = hsv[..., 1], hsv[..., 2]
        foreground = ((saturation >= self.config.min_saturation)
                      & (value >= self.config.min_value) & (value <= self.config.max_value))
        if self.config.roi_xyxy_normalized is not None:
            rx1, ry1, rx2, ry2 = self.config.roi_xyxy_normalized
            x1, x2 = round(rx1 * width), round(rx2 * width)
            y1, y2 = round(ry1 * height), round(ry2 * height)
            roi = np.zeros_like(foreground)
            roi[y1:y2, x1:x2] = True
            foreground &= roi
        binary = foreground.astype(np.uint8) * 255
        if self.config.close_kernel > 1:
            k = self._odd_kernel(self.config.close_kernel)
            binary = self.cv2.morphologyEx(binary, self.cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        if self.config.open_kernel > 1:
            k = self._odd_kernel(self.config.open_kernel)
            binary = self.cv2.morphologyEx(binary, self.cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        if self.config.roi_xyxy_normalized is not None:
            # Closing can grow foreground across the crop boundary; enforce ROI last.
            binary[~roi] = 0
        detections = []
        max_area = int(height * width * self.config.max_area_fraction)
        for label, x, y, w, h, area, component in self._components(binary, hsv):
            if area < self.config.min_area_px or area > max_area or min(w, h) < self.config.min_side_px:
                continue
            if max(w / h, h / w) > self.config.max_aspect_ratio:
                continue
            contour = self._largest_contour(component)
            hull_area = float(self.cv2.contourArea(self.cv2.convexHull(contour))) if len(contour) >= 3 else 0.0
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < self.config.min_solidity:
                continue
            # Fill enclosed shading/holes in a single component to preserve one block mask.
            mask_u8 = np.zeros((height, width), np.uint8)
            self.cv2.drawContours(mask_u8, [contour], -1, 255, thickness=-1)
            mask = mask_u8 != 0
            yy, xx = np.nonzero(mask)
            if not len(xx):
                continue
            color = estimate_color(rgb, mask, self.config.color)
            # Candidate-ranking heuristic only; this is not a calibrated probability.
            sat_quality = min(1.0, float(np.median(saturation[mask])) / 255.0)
            confidence = float(np.clip(0.65 * sat_quality + 0.35 * min(1.0, solidity), 0, 1))
            detections.append(Detection(
                detection_id=0, track_id=None, class_name=self.class_name,
                confidence=confidence, color=color,
                bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                centroid_uv=(float(xx.mean()), float(yy.mean())), mask=mask,
            ))
        detections.sort(key=lambda det: (det.centroid_uv[1], det.centroid_uv[0]))
        for detection_id, detection in enumerate(detections):
            detection.detection_id = detection_id
        if track:
            self._associate(detections)
        return FrameDetections(frame_id, None if timestamp_s is None else float(timestamp_s),
                               (height, width), detections,
                               timestamp_source or ("caller" if timestamp_s is not None else None))

    def _hue_bands(self) -> list:
        """(hue ranges, value range) per colour band; with split_green_by_value the green band shared by light / dark
        green is split by value."""
        bands = []
        for name, ranges in self.config.color.hue_ranges.items():
            r = tuple(tuple(float(x) for x in pair) for pair in ranges)
            vr = (0, 256)
            if self.config.split_green_by_value:
                vr = {"light_green": (GREEN_VALUE_SPLIT, 256), "dark_green": (0, GREEN_VALUE_SPLIT)}.get(name, vr)
            if (r, vr) not in bands:
                bands.append((r, vr))
        return bands

    def _components(self, binary: np.ndarray, hsv: np.ndarray):
        """(label, x, y, w, h, area, mask) of each foreground component, per colour band when split_by_hue."""
        if self.config.split_by_hue:
            hue = hsv[..., 0].astype(np.float32)
            value = hsv[..., 2]
            layers = []
            for ranges, (vlo, vhi) in self._hue_bands():
                band = np.zeros(hue.shape, bool)
                for lo, hi in ranges:
                    band |= (hue >= lo) & (hue < hi)
                band &= (value >= vlo) & (value < vhi)
                layer = (binary != 0) & band
                if self.config.open_kernel > 1:   # drop thin hue fringes along other blocks' edges
                    k = self._odd_kernel(self.config.open_kernel)
                    layer = self.cv2.morphologyEx(layer.astype(np.uint8), self.cv2.MORPH_OPEN,
                                                  np.ones((k, k), np.uint8)) != 0
                layers.append(layer.astype(np.uint8) * 255)
        else:
            layers = [binary]
        for layer in layers:
            count, labels, stats, _ = self.cv2.connectedComponentsWithStats(layer, connectivity=8)
            for label in range(1, count):
                x, y, w, h, area = (int(v) for v in stats[label])
                yield label, x, y, w, h, area, labels == label

    @staticmethod
    def _odd_kernel(size: int) -> int:
        return size if size % 2 else size + 1

    def _largest_contour(self, component: np.ndarray):
        contours, _ = self.cv2.findContours(component.astype(np.uint8), self.cv2.RETR_EXTERNAL,
                                             self.cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=self.cv2.contourArea) if contours else np.empty((0, 1, 2), np.int32)

    def _associate(self, detections: list[Detection]) -> None:
        pairs = []
        for tid, old in self._tracks.items():
            for index, det in enumerate(detections):
                if old["color"] != det.color:
                    continue
                distance = float(np.hypot(old["uv"][0] - det.centroid_uv[0],
                                          old["uv"][1] - det.centroid_uv[1]))
                if distance <= self.config.track_max_distance_px:
                    pairs.append((distance, tid, index))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, tid, index in sorted(pairs):
            if tid in used_tracks or index in used_detections:
                continue
            used_tracks.add(tid)
            used_detections.add(index)
            detection = detections[index]
            detection.track_id = tid
            self._tracks[tid] = {"uv": detection.centroid_uv, "color": detection.color, "missed": 0}
        for tid in list(self._tracks):
            if tid not in used_tracks:
                self._tracks[tid]["missed"] += 1
                if self._tracks[tid]["missed"] > self.config.track_max_missed_frames:
                    del self._tracks[tid]
        for index, detection in enumerate(detections):
            if index not in used_detections:
                tid = self._next_track_id
                self._next_track_id += 1
                detection.track_id = tid
                self._tracks[tid] = {"uv": detection.centroid_uv, "color": detection.color, "missed": 0}
