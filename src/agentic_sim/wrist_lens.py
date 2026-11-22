"""Wrist camera lens: make an ideal pinhole render look like the raw (barrel-distorted) wrist camera.

The real wrist camera (Innomaker U20CAM-1080p-S1) shows clear barrel distortion. A renderer only draws ideal pinhole
images, so the "real" lens mode renders a wider pinhole image and resamples it onto the raw sensor:
for each raw pixel, undistort it (OpenCV radial-tangential model, iterative like cv2.undistortPoints) to a viewing
ray, then look the ray up in the wide render. Pure numpy; maps are built once per (lens, render size).

  lens = WristLens.from_scene(scene)               # None when the scene has no lens model
  w, h, fovy = lens.render_size()                  # wide pinhole render that covers every raw pixel's ray
  maps = lens.maps(w, h, fovy)                     # (iy, ix) index arrays, raw (480 x 640) -> render
  raw = lens.apply(render_rgb, maps)               # 480 x 640 x 3
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from agentic_sim.config import v

MODES = ("real", "pinhole")


@dataclass
class WristLens:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple            # k1, k2, p1, p2, k3 (OpenCV order)

    @classmethod
    def from_scene(cls, scene: dict):
        wr = (scene.get("sensor_cameras") or {}).get("wrist") or {}
        lens = wr.get("lens")
        if not lens:
            return None
        w, h = (int(x) for x in v(wr["resolution"]))
        k = v(lens["intrinsics"])
        d = list(v(lens["distortion"])) + [0.0] * 5
        return cls(w, h, float(k["fx"]), float(k["fy"]), float(k["cx"]), float(k["cy"]), tuple(float(x) for x in d[:5]))

    def _fold_radius(self) -> float:
        """Ideal radius where the radial model stops increasing (d r_d / d r = 0); beyond it the model folds back.
        Raw pixels whose distorted radius exceeds the model's maximum (extreme corners) map to this radius."""
        k1, k2, _, _, k3 = self.dist
        rs = np.linspace(0, 4, 4001)[1:]
        d = 1 + 3 * k1 * rs ** 2 + 5 * k2 * rs ** 4 + 7 * k3 * rs ** 6
        bad = np.nonzero(d <= 0)[0]
        return float(rs[bad[0]]) if len(bad) else math.inf

    def undistort(self, u, v_):
        """Raw pixel coordinates -> ideal normalized image coordinates (x, y) = (X/Z, Y/Z).
        Radial part by bisection on the radius (up to the fold radius), tangential part by fixed-point steps."""
        k1, k2, p1, p2, k3 = self.dist
        xd = (np.asarray(u, float) - self.cx) / self.fx
        yd = (np.asarray(v_, float) - self.cy) / self.fy
        rmax = self._fold_radius()
        x, y = xd.copy(), yd.copy()
        for _ in range(8 if (p1 or p2) else 1):
            r2 = x * x + y * y
            tx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            ty = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            ax, ay = xd - tx, yd - ty                      # radially distorted point without the tangential term
            rd = np.hypot(ax, ay)
            # the radial map is increasing on [0, rmax]: bisection is robust up to the fold
            hi_r = min(rmax, 4.0)
            lo, hi = np.zeros_like(rd), np.full_like(rd, hi_r)
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                over = mid * (1 + k1 * mid ** 2 + k2 * mid ** 4 + k3 * mid ** 6) > rd
                hi, lo = np.where(over, mid, hi), np.where(over, lo, mid)
            r = 0.5 * (lo + hi)
            scale = np.where(rd > 1e-12, r / np.maximum(rd, 1e-12), 1.0)
            x, y = ax * scale, ay * scale
        return x, y

    def distort(self, x, y):
        """Ideal normalized coordinates -> raw pixel coordinates (forward model; tests)."""
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd * self.fx + self.cx, yd * self.fy + self.cy

    def render_size(self, scale: float = 1.5, margin: float = 1.03) -> tuple:
        """(width, height, vertical fov deg) of a centred pinhole render whose view covers every raw pixel's ray.
        The render keeps the raw aspect ratio; scale sets its resolution relative to the raw image.
        Cached per lens: undistorting every raw pixel takes about 0.8 s, and live perception asks on every capture."""
        return _render_size(self.width, self.height, self.fx, self.fy, self.cx, self.cy, tuple(self.dist), scale, margin)

    def _render_size(self, scale: float, margin: float) -> tuple:
        uu, vv = np.meshgrid(np.arange(self.width), np.arange(self.height))
        x, y = self.undistort(uu, vv)
        aspect = self.width / self.height
        half_y = max(float(np.abs(y).max()), float(np.abs(x).max()) / aspect) * margin
        w, h = int(round(self.width * scale)), int(round(self.height * scale))
        return w, h, math.degrees(2 * math.atan(half_y))

    def maps(self, w: int, h: int, fovy_deg: float, out_w: int | None = None, out_h: int | None = None) -> tuple:
        """Nearest-pixel lookup (iy, ix) into a centred w x h pinhole render with vertical fov, for an output image
        of out_w x out_h (default: the raw size) that covers the raw sensor (a scaled raw image, e.g. a GUI inset)."""
        ow, oh = out_w or self.width, out_h or self.height
        f = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
        jj, ii = np.meshgrid(np.arange(ow), np.arange(oh))
        u = (jj + 0.5) * self.width / ow - 0.5
        v_ = (ii + 0.5) * self.height / oh - 0.5
        x, y = self.undistort(u, v_)
        ix = np.clip(np.rint(x * f + (w - 1) / 2), 0, w - 1).astype(np.intp)
        iy = np.clip(np.rint(y * f + (h - 1) / 2), 0, h - 1).astype(np.intp)
        return iy, ix

    @staticmethod
    def apply(render: np.ndarray, maps: tuple) -> np.ndarray:
        iy, ix = maps
        return render[iy, ix]


@functools.lru_cache(maxsize=16)
def _render_size(width, height, fx, fy, cx, cy, dist, scale, margin) -> tuple:
    return WristLens(width, height, fx, fy, cx, cy, dist)._render_size(scale, margin)


def lens_mode(scene: dict, override: str | None = None) -> str:
    """'real' (raw-looking wrist image) or 'pinhole'. Scene default sensor_cameras.wrist.lens_mode; an explicit
    override (run option / CLI) wins. Scenes without a lens model are always 'pinhole'."""
    if WristLens.from_scene(scene) is None:
        return "pinhole"
    mode = override or str((scene["sensor_cameras"]["wrist"].get("lens_mode") or "real"))
    if mode not in MODES:
        raise ValueError(f"wrist lens mode {mode!r}; expected one of {MODES}")
    return mode


def apply_wrist_lens(scene: dict, override: str | None) -> dict:
    """The scene with its wrist lens mode resolved (override or scene default); returns the same dict."""
    wr = (scene.get("sensor_cameras") or {}).get("wrist")
    if wr is not None:
        wr["lens_mode"] = lens_mode(scene, override)
    return scene
