"""Two wrist views for pre-grasp alignment, shared across simulator adapters."""
from collections.abc import Callable
from typing import TypeVar

Pose = TypeVar("Pose")


def align_from_hover(
    coarse: Pose,
    observe: Callable[[Pose], Pose | None],
    reposition: Callable[[Pose | None], None],
) -> tuple[Pose | None, Pose | None, Pose]:
    """Return both accepted observations (None on rejection) and the grasp estimate.

    Start at the coarse hover. The adapter captures a fresh wrist image for each
    observation and completes the view adjustment before the second capture.
    For B, reposition uses the full first correction, holding if it is missing.
    """
    first = observe(coarse)
    anchor = coarse if first is None else first
    reposition(first)
    second = observe(anchor)
    return first, second, anchor if second is None else second
