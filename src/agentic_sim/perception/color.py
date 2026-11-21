"""Robust, mask-restricted HSV color estimates, separate from segmentation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

COLOR_NAMES = ("red", "orange", "yellow", "light_green", "dark_green", "blue", "cyan", "purple")
DEFAULT_HUE_RANGES = {
    # Half-open intervals use floating hue in OpenCV's [0, 180) scale, so
    # adjacent hues have neither gaps nor double-counted boundaries.
    "red": ((0.0, 10.5), (170.0, 180.0)), "orange": ((10.5, 24.5),),
    "yellow": ((24.5, 36.5),), "light_green": ((36.5, 85.0,),),
    "dark_green": ((36.5, 85.0,),), "cyan": ((85.0, 99.5),),
    "blue": ((99.5, 132.5),), "purple": ((132.5, 170.0),),
}


@dataclass(slots=True)
class ColorConfig:
    min_saturation: int = 55
    min_value: int = 35
    max_value: int = 245
    center_crop_fraction: float = 0.72
    min_valid_fraction: float = 0.15
    min_dominance: float = 0.28
    hue_ranges: dict[str, tuple[tuple[int, int], ...]] = field(default_factory=lambda: DEFAULT_HUE_RANGES.copy())


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenCV-compatible HSV scale: hue [0,179], saturation/value [0,255]."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    hi, lo = np.max(x, axis=-1), np.min(x, axis=-1)
    delta = hi - lo
    h = np.zeros_like(hi)
    nz = delta > 1e-6
    mr = nz & (hi == r); mg = nz & (hi == g); mb = nz & (hi == b)
    h[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6
    h[mg] = (b[mg] - r[mg]) / delta[mg] + 2
    h[mb] = (r[mb] - g[mb]) / delta[mb] + 4
    h = h * 30.0  # 360 degrees scaled to OpenCV's [0,179]
    s = np.divide(delta, hi, out=np.zeros_like(delta), where=hi > 1e-6) * 255
    return h, s, hi * 255


def estimate_color(rgb: np.ndarray, mask: np.ndarray, config: ColorConfig | None = None) -> str:
    """Estimate color from central, non-dark, non-highlight pixels inside mask."""
    config = config or ColorConfig()
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("rgb must be HxWx3 and mask must match its HxW dimensions")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return "unknown"
    # Crop relative to this object's extent, not the camera frame. Blocks near
    # the image border therefore receive the same interior sampling as centered blocks.
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    fraction = float(np.clip(config.center_crop_fraction, 0.0, 1.0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = width * fraction / 2, height * fraction / 2
    fx = (xs >= cx - half_w) & (xs < cx + half_w)
    fy = (ys >= cy - half_h) & (ys < cy + half_h)
    keep = fx & fy
    # A poorly centered/fragmented mask still gets a majority vote instead of no estimate.
    if keep.sum() < max(1, int(len(xs) * config.min_valid_fraction)):
        keep = np.ones(len(xs), dtype=bool)
    pixels = rgb[ys[keep], xs[keep]]
    hue, sat, val = _rgb_to_hsv(pixels)
    valid = (sat >= config.min_saturation) & (val >= config.min_value) & (val <= config.max_value)
    if valid.mean() < config.min_valid_fraction:
        return "unknown"
    hue, valid_value = hue[valid], val[valid]
    votes: dict[str, int] = {}
    for name, ranges in config.hue_ranges.items():
        matched = np.zeros(hue.shape, dtype=bool)
        for lo, hi in ranges:
            matched |= (hue >= lo) & (hue < hi)
        votes[name] = int(matched.sum())
    # light_green and dark_green share a hue band; brightness separates them.
    green_count = max(votes.get("light_green", 0), votes.get("dark_green", 0))
    votes["light_green"] = votes["dark_green"] = green_count
    name, count = max(votes.items(), key=lambda item: item[1], default=("unknown", 0))
    if count / len(hue) < config.min_dominance:
        return "unknown"
    # Green is split by brightness to distinguish the two palette classes.
    if name in ("light_green", "dark_green"):
        median_v = float(np.median(valid_value))
        name = "light_green" if median_v >= 145 else "dark_green"
    return name if count else "unknown"
