"""RGB block instance segmentation, color estimation, and optional tracking."""

from .color import ColorConfig, estimate_color
from .model import BlockPerception, PerceptionConfig, PerceptionError
from .types import Detection, FrameDetections

__all__ = ["BlockPerception", "ColorConfig", "Detection", "FrameDetections", "PerceptionConfig", "PerceptionError", "estimate_color"]
