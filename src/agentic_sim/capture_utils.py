"""Small dependency-free helpers for deterministic local simulator captures."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable


def validate_capture_options(frames: int, fps: float, width: int, height: int) -> None:
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
        raise ValueError("frames must be a positive integer")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (width, height)):
        raise ValueError("width and height must be positive integers")


def capture_output_paths(output: str | Path) -> tuple[Path, Path, Path]:
    """Create output directory and return only this tool's three fixed filenames.

    Existing unrelated files are deliberately untouched; the image/video/metadata
    with these names are the outputs of this capture command and may be replaced.
    """
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    return (directory / "block_scene.png", directory / "block_scene.mp4",
            directory / "capture_info.txt")


def write_video_frames(cv2, path: str | Path, fps: float, width: int, height: int,
                       frames: Iterable[object]) -> int:
    """Write BGR frames and release VideoWriter even if setup or writing fails."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not open video writer: {path}")
        count = 0
        for frame in frames:
            writer.write(frame)
            count += 1
        if count == 0:
            raise RuntimeError("cannot write a video with zero frames")
        return count
    finally:
        writer.release()
