"""Which block a spoken position means ("左上角那顆紅色"): pure Python, shared by the chat (early answer) and the
executor (the choice it acts on).

Positions are in the mount frame (x forward from the base, y to the robot's left, metres). The words map onto it
through `frame` (configs/task.yaml pick_ref): `up` is the mount-frame direction a person calls 上／遠, `left` the one
called 左. The default is the view from behind the arm looking forward (上 = +x, 左 = +y); a different viewpoint (the
operator across the table, the overhead camera image) only changes these two vectors.

  best, tie = resolve(blocks, "top_left", frame, pan_xy)   # blocks: [{"block", "xy", ...}] of one colour

`tie`: the runner-up scores within frame["tie_m"] of the best, so the words do not single out one block.
"""

from __future__ import annotations

import numpy as np

REFS = ("top_left", "top_right", "bottom_left", "bottom_right", "leftmost", "rightmost", "farthest", "nearest")
REF_ZH = {"top_left": "左上角", "top_right": "右上角", "bottom_left": "左下角", "bottom_right": "右下角",
          "leftmost": "最左邊", "rightmost": "最右邊", "farthest": "最遠", "nearest": "最近"}
DEFAULT_FRAME = {"up": [1.0, 0.0], "left": [0.0, 1.0], "tie_m": 0.015}

# (weight on up, weight on left) for the direction refs; farthest / nearest use the distance from the pan axis
_DIR = {"top_left": (1, 1), "top_right": (1, -1), "bottom_left": (-1, 1), "bottom_right": (-1, -1),
        "leftmost": (0, 1), "rightmost": (0, -1)}


def score(xy, ref: str, frame: dict, pan_xy=(0.0, 0.0), centre=(0.0, 0.0)) -> float:
    """Larger = more like `ref`. Direction refs are measured from `centre` (the candidates' mean), so a corner means
    the corner of where the blocks lie, not of the table."""
    p = np.asarray(xy, float)
    if ref in ("farthest", "nearest"):
        d = float(np.linalg.norm(p - np.asarray(pan_xy, float)))
        return d if ref == "farthest" else -d
    wu, wl = _DIR[ref]
    up, left = np.asarray(frame["up"], float), np.asarray(frame["left"], float)
    r = p - np.asarray(centre, float)
    return float(wu * r @ up + wl * r @ left)


def resolve(blocks: list, ref: str, frame: dict | None = None, pan_xy=(0.0, 0.0)):
    """The block `ref` means among `blocks` (one colour), and whether that is a tie. (None, False) when empty."""
    if ref not in REFS:
        raise ValueError(f"ref must be one of {list(REFS)}")
    frame = {**DEFAULT_FRAME, **(frame or {})}
    if not blocks:
        return None, False
    centre = np.mean([np.asarray(b["xy"], float) for b in blocks], axis=0)
    ranked = sorted(blocks, key=lambda b: -score(b["xy"], ref, frame, pan_xy, centre))
    if len(ranked) == 1:
        return ranked[0], False
    gap = score(ranked[0]["xy"], ref, frame, pan_xy, centre) - score(ranked[1]["xy"], ref, frame, pan_xy, centre)
    return ranked[0], gap < float(frame["tie_m"])
