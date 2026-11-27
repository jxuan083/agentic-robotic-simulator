"""Stable block IDs over D455 scans.

The task executor indexes blocks by a fixed ID with a fixed colour (World.colors). A real scan only gives colours and
positions, so each scan is matched to the known blocks: same colour within MATCH_M of the last position; a block
the program moved (place) is moved here first; an unmatched detection takes a same-colour ID that is no longer seen
(a block restocked at a new spot) or a new ID. Blocks not seen in a scan keep their last position but belong to no
tray (taken away, or hidden).

The plush toy (colour "plush", agentic_sim.real_plush) stands on the table outside the trays: there it counts as in
storage (available to order); in the beige tray it is placed like a block.
"""

from __future__ import annotations

import math

import numpy as np

MATCH_M = 0.04
EDGE_M = 0.0295
TRAYS = {"pink": "storage", "beige": "placement"}
PLUSH = "plush"


def tray_of(d: dict) -> str | None:
    if d.get("color") == PLUSH and d.get("tray") is None:
        return "storage"          # on the table, where it is put out to be ordered
    return TRAYS.get(d.get("tray"))


class Tracker:
    def __init__(self, match_m: float = MATCH_M):
        self.match_m = match_m
        self.colors: list = []      # shared with World.colors (appended in place)
        self.pos: list = []         # block centre, mount frame (m)
        self.yaw: list = []
        self.tray: list = []        # "storage" / "placement" / None (not seen in the last scan)

    def __len__(self):
        return len(self.colors)

    def _centre(self, d: dict) -> np.ndarray:
        return np.array([d["x"], d["y"], d["z_top"] - EDGE_M / 2], float)

    def _new(self, d: dict) -> int:
        self.colors.append(d["color"])
        self.pos.append(self._centre(d))
        self.yaw.append(float(d.get("yaw", 0.0)))
        self.tray.append(tray_of(d))
        return len(self.colors) - 1

    def _set(self, b: int, d: dict):
        self.pos[b] = self._centre(d)
        self.yaw[b] = float(d.get("yaw", 0.0))
        self.tray[b] = tray_of(d)

    def update(self, detections: list, exclude=()) -> list:
        """Match one scan (dicts with color, tray, x, y, z_top, yaw; mount m). exclude: IDs not on the table (held).
        Returns [(id, detection)]; detections outside both trays (except the plush) are ignored."""
        dets = [d for d in detections if tray_of(d) is not None]
        pairs = sorted((math.hypot(d["x"] - self.pos[b][0], d["y"] - self.pos[b][1]), i, b)
                       for i, d in enumerate(dets) for b in range(len(self))
                       if b not in exclude and self.colors[b] == d["color"])
        used_d, used_b, out = set(), set(), []
        for dist, i, b in pairs:
            if dist > self.match_m or i in used_d or b in used_b:
                continue
            used_d.add(i)
            used_b.add(b)
            self._set(b, dets[i])
            out.append((b, dets[i]))
        for i, d in enumerate(dets):
            if i in used_d:
                continue
            spare = [b for b in range(len(self)) if b not in used_b and b not in exclude
                     and self.colors[b] == d["color"] and self.tray[b] is None]
            if spare:
                b = spare[0]
                self._set(b, d)
            else:
                b = self._new(d)
            used_b.add(b)
            out.append((b, d))
        for b in range(len(self)):
            if b not in used_b and b not in exclude:
                self.tray[b] = None
        return out

    def move(self, b: int, xy, tray: str | None):
        self.pos[b] = np.array([xy[0], xy[1], self.pos[b][2]], float)
        self.tray[b] = tray

    def remove(self, b: int):
        self.tray[b] = None

    def counts(self, tray: str) -> dict:
        out = {}
        for c, t in zip(self.colors, self.tray):
            if t == tray:
                out[c] = out.get(c, 0) + 1
        return out
