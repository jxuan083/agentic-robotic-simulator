#!/usr/bin/env python3
"""Straighten raw wrist camera frames with the scene's lens model (the opposite direction of the simulated "real"
wrist image). No simulator, no OpenCV: numpy + ffmpeg only.

  uv run tools/undistort_wrist.py <raw.png> [more.png ...] [--out-dir DIR] [--scene configs/scenes/lab_20261120.yaml]

Input: original 640 x 480 wrist frames (any image ffmpeg reads). Output: <name>_flat.png next to the input (or in
--out-dir). The whole batch is checked before anything is written; on any problem the tool names it, writes nothing
and exits 1: a frame of another size (cropped or scaled frames do not fit the intrinsics), an unreadable input, an
output that is one of the inputs (e.g. frame.png and frame_flat.png together), or two outputs with the same path
(e.g. a/frame.png and b/frame.png with one --out-dir). Inputs are never written. The output keeps the raw intrinsics (fx = fy = 400, centre 320, 240): the centre keeps its scale,
straight edges become straight, and the stretched periphery that falls outside 640 x 480 is cropped (no black
borders, since barrel distortion maps every output pixel inside the raw frame). Bilinear sampling.

Needs only numpy, PyYAML and ffmpeg / ffprobe on PATH (no MuJoCo, no OpenGL).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from agentic_sim.config import load  # noqa: E402
from agentic_sim.image_io import image_size, read_png, write_png  # noqa: E402
from agentic_sim.wrist_lens import WristLens  # noqa: E402


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


def _same(a: str, b: str) -> bool:
    if os.path.realpath(a) == os.path.realpath(b):
        return True
    try:
        return os.path.samefile(a, b)                   # hard links
    except OSError:
        return False


def plan(images: list, out_dir, width: int, height: int):
    """(input, output) pairs and the problems that stop the batch before anything is written."""
    jobs = [(p, os.path.join(out_dir or os.path.dirname(os.path.abspath(p)),
                             f"{os.path.splitext(os.path.basename(p))[0]}_flat.png")) for p in images]
    problems = []
    for i, (p, out) in enumerate(jobs):
        hit = next((q for q in images if _same(out, q)), None)
        if hit is not None:
            problems.append(f"{p}: its output {out} is the input {hit}")
        other = next((q for q, o in jobs[:i] if _same(out, o)), None)
        if other is not None:
            problems.append(f"{p}: its output {out} is also the output of {other}")
        try:
            w, h = image_size(p)
        except (OSError, ValueError, subprocess.CalledProcessError):
            problems.append(f"{p}: cannot read the image")
            continue
        if (w, h) != (width, height):
            problems.append(f"{p}: {w}x{h}, expected the original {width}x{height} wrist frame")
    return jobs, problems


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
    jobs, problems = plan(a.images, a.out_dir, lens.width, lens.height)
    if problems:
        for msg in problems:
            print(msg, file=sys.stderr)
        print("nothing written", file=sys.stderr)
        return 1
    maps = flat_maps(lens)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    for path, out in jobs:
        write_png(out, undistort(read_png(path), maps))
        print(f"{path} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
