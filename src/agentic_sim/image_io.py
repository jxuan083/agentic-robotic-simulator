"""Read and write images with ffmpeg / ffprobe and numpy only (no simulator, no OpenGL, no OpenCV). Shared by the
offline tools (tools/undistort_wrist.py) and tools/render_sensor_cameras.py."""

from __future__ import annotations

import subprocess

import numpy as np


def image_size(path: str) -> tuple[int, int]:
    """(width, height) of the first video stream of any image ffprobe reads."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
    w, h = (int(x) for x in out.strip().split(",")[:2])
    return w, h


def read_png(path: str) -> np.ndarray:
    """H x W x 3 uint8 RGB."""
    w, h = image_size(path)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def write_png(path: str, img: np.ndarray) -> None:
    h, w, _ = img.shape
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-i", "-", path], input=np.ascontiguousarray(img).tobytes(), check=True)
