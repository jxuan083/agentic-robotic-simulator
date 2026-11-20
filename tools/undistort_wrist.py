#!/usr/bin/env python3
"""Straighten raw wrist camera frames with the scene's lens model (the opposite direction of the simulated "real"
wrist image). No simulator, no OpenCV: numpy + ffmpeg only.

  uv run tools/undistort_wrist.py <raw.png> [more.png ...] [--out-dir DIR] [--scene configs/scenes/lab_20261120.yaml]

Input: original 640 x 480 wrist frames (any image ffmpeg reads); other sizes are refused with a message (cropped or
scaled frames do not fit the intrinsics). Output: <name>_flat.png next to the input (or in --out-dir); the input is
never overwritten. The output keeps the raw intrinsics (fx = fy = 400, centre 320, 240): the centre keeps its scale,
straight edges become straight, and the stretched periphery that falls outside 640 x 480 is cropped (no black
borders, since barrel distortion maps every output pixel inside the raw frame). Bilinear sampling.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402

from agentic_sim.config import load  # noqa: E402
from agentic_sim.wrist_lens import WristLens  # noqa: E402
from render_sensor_cameras import read_png, write_png  # noqa: E402


def flat_maps(lens: WristLens):
    """For each output pixel (ideal pinhole with the raw intrinsics): its source position in the raw frame."""
    uu, vv = np.meshgrid(np.arange(lens.width, dtype=float), np.arange(lens.height, dtype=float))
    x, y = (uu - lens.cx) / lens.fx, (vv - lens.cy) / lens.fy
    return lens.distort(x, y)


def undistort(img: np.ndarray, maps) -> np.ndarray:
    su, sv = maps
    h, w = img.shape[:2]
    u0 = np.clip(np.floor(su).astype(int), 0, w - 2)
    v0 = np.clip(np.floor(sv).astype(int), 0, h - 2)
    fu = np.clip(su - u0, 0.0, 1.0)[..., None]
    fv = np.clip(sv - v0, 0.0, 1.0)[..., None]
    f = img.astype(float)
    top = f[v0, u0] * (1 - fu) + f[v0, u0 + 1] * fu
    bot = f[v0 + 1, u0] * (1 - fu) + f[v0 + 1, u0 + 1] * fu
    return np.clip(top * (1 - fv) + bot * fv + 0.5, 0, 255).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--scene", default=os.path.join(ROOT, "configs/scenes/lab_20261120.yaml"))
    a = ap.parse_args(argv)
    lens = WristLens.from_scene(load(a.scene))
    if lens is None:
        print(f"{a.scene}: no wrist lens model (sensor_cameras.wrist.lens)", file=sys.stderr)
        return 2
    maps = flat_maps(lens)
    rc = 0
    for path in a.images:
        img = read_png(path)
        if img.shape[:2] != (lens.height, lens.width):
            print(f"skip {path}: {img.shape[1]}x{img.shape[0]}, expected the original {lens.width}x{lens.height} "
                  "wrist frame", file=sys.stderr)
            rc = 1
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(a.out_dir or os.path.dirname(os.path.abspath(path)), f"{stem}_flat.png")
        if os.path.abspath(out) == os.path.abspath(path):
            print(f"skip {path}: output would overwrite the input", file=sys.stderr)
            rc = 1
            continue
        if a.out_dir:
            os.makedirs(a.out_dir, exist_ok=True)
        write_png(out, undistort(img, maps))
        print(f"{path} -> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
