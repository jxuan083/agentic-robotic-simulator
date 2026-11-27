"""What the real-arm skills need from the arm, and a simulated table to test the whole chat -> executor -> arm flow
without hardware (tools/real_task_server.py --fake).

Arm (duck-typed; agentic_sim.real_task.real_arm.RealArm on the follower):
  scan()                       -> (blocks, clear, reason): D455 at park; blocks = [{color, tray, x, y, z_top, yaw}] (m)
  pick(block, phase)           -> (held: bool, info): park -> start -> grasp -> lift; a miss ends back at park
  place(xy, phase)             -> error str or None: holding -> slot -> open -> rise -> start -> park
  home(open_gripper=True)      -> to park (gripper opened when the hand is empty)
  at_park() / ee() / gripper_open() / reachable(xy) / home_xyz / pan_xy / slots / plush_slot
The plush toy is a block of colour "plush" on the table (scan: tray None): pick takes it with its ACT policy, place
puts it at plush_slot (beige P0).
  pause() / resume()           operator pause from the chat page (the arm holds)
phase(name) reports progress inside a routine.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

EDGE_M = 0.0295
# 11/25 rig, mount frame (m): pink tray and the beige slots P1-P4 (agentic_sim.real_place.beige_slots, 11/26)
FAKE_PINK = ((0.15, 0.06), (0.45, 0.27))
FAKE_BEIGE = ((0.14, -0.27), (0.48, -0.01))
FAKE_SLOTS = [(0.210, -0.065), (0.210, -0.120), (0.265, -0.065), (0.265, -0.120)]
FAKE_PLUSH_SLOT = (0.215, -0.197)
FAKE_PLUSH_AT = (0.09, 0.20)     # on the table beside the pink tray
FAKE_HOME = (0.08, 0.0, 0.20)
REACH_M = 0.32


def _inside(box, x, y):
    (x0, y0), (x1, y1) = box
    return x0 <= x <= x1 and y0 <= y <= y1


class FakeArm:
    """A table with blocks; motions take `seconds` each. pick() misses blocks listed in `misses` (colour, once)."""

    def __init__(self, blocks, seconds: float = 0.05, misses=(), slots=FAKE_SLOTS):
        self.blocks = [dict(b) for b in blocks]     # {color, x, y, yaw}; tray from the position
        self.seconds = seconds
        self.misses = list(misses)
        self.slots = [tuple(s) for s in slots]
        self.plush_slot = FAKE_PLUSH_SLOT
        self.home_xyz = np.array(FAKE_HOME, float)
        self.pan_xy = (0.0, 0.0)
        self.held = None
        self.where = "park"
        self.gripper = "open"
        self.log: list = []
        self._paused = threading.Event()
        self._lock = threading.Lock()

    # ---- operator side (tests, tools/real_task_server.py fake_scene) -----------------------------
    def remove(self, color: str, tray: str = "pink"):
        with self._lock:
            for b in self.blocks:
                if b["color"] == color and self._tray(b) == tray:
                    self.blocks.remove(b)
                    return b
        return None

    def add(self, color: str, xy, yaw: float = 0.0):
        with self._lock:
            self.blocks.append({"color": color, "x": float(xy[0]), "y": float(xy[1]), "yaw": yaw})

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    # ---- arm interface -------------------------------------------------------------------------------
    def _tray(self, b):
        return "pink" if _inside(FAKE_PINK, b["x"], b["y"]) else "beige" if _inside(FAKE_BEIGE, b["x"], b["y"]) else None

    def _move(self, where, phase=None, label=None):
        if phase and label:
            phase(label)
        while self._paused.is_set():
            time.sleep(0.01)
        time.sleep(self.seconds)
        self.where = where
        self.log.append(where)

    def scan(self):
        if self.where != "park":
            raise RuntimeError("scan away from park")
        with self._lock:
            out = [{"color": b["color"], "tray": self._tray(b), "x": b["x"], "y": b["y"], "z_top": EDGE_M,
                    "yaw": b.get("yaw", 0.0)} for b in self.blocks]
        return out, True, ""

    def pick(self, block, phase=None):
        self._move("start", phase, "approach")
        with self._lock:
            near = [b for b in self.blocks if b["color"] == block["color"]
                    and (self._tray(b) == "pink" or b["color"] == "plush" and self._tray(b) is None)
                    and math.hypot(b["x"] - block["x"], b["y"] - block["y"]) < 0.02]
        self._move("grasp", phase, "descend")
        self.gripper = "closed"
        if not near or block["color"] in self.misses:
            if block["color"] in self.misses:
                self.misses.remove(block["color"])
            self.gripper = "open"
            self._move("park", phase, "retreat")
            return False, {"reason": "missed"}
        with self._lock:
            self.blocks.remove(near[0])
        self.held = near[0]
        self._move("lift", phase, "lift")
        return True, {"gripper_pct": 17.0}

    def place(self, xy, phase=None):
        if self.held is None:
            return "nothing in the gripper"
        self._move("slot", phase, "transport")
        with self._lock:
            self.blocks.append(dict(self.held, x=float(xy[0]) + 0.003, y=float(xy[1]) - 0.002))
        self.held = None
        self.gripper = "open"
        self._move("start", phase, "retreat")
        self._move("park", phase, "home")
        return None

    def home(self, open_gripper=True, phase=None):
        if open_gripper:
            self.held = None
            self.gripper = "open"
        self._move("park", phase, "home")

    def at_park(self) -> bool:
        return self.where == "park"

    def ee(self):
        return self.home_xyz.copy() if self.where == "park" else np.array([0.25, 0.0, 0.10])

    def gripper_open(self) -> bool:
        return self.gripper == "open"

    def reachable(self, xy) -> bool:
        return math.hypot(xy[0] - self.pan_xy[0], xy[1] - self.pan_xy[1]) <= REACH_M

    def close(self):
        pass
