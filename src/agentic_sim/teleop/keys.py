"""Teleop key map and held-key state, shared by every backend (pure Python).

Each backend feeds its window's raw key events into KeyState.on_key(name, down) (Isaac: carb keyboard events,
MuJoCo: GLFW key callbacks) with the key names below. Held keys start only on a fresh press (auto-repeat never
re-arms a key) and are dropped on release, focus loss, pause and reset, so stale input is never replayed.
"""

from __future__ import annotations

MOVE_KEYS = {
    "W": (0, +1), "S": (0, -1),   # +X forward / -X
    "A": (1, +1), "D": (1, -1),   # +Y left / -Y
    "E": (2, +1), "Q": (2, -1),   # +Z up / -Z
}
YAW_KEYS = {"Z": -1, "X": +1}
GRIPPER_KEYS = {"O": +1, "C": -1}  # open / close while held
ACTION_KEYS = {"P": "pause", "R": "reset", "M": "focus",
               "KEY_1": "annotate_global", "KEY_2": "annotate_wrist"}   # camera cell: raw / annotated snapshot
MJ_ACTION_KEYS = {"P": "pause", "R": "reset", "M": "focus", "KEY_1": "annotate_global", "KEY_2": "annotate_wrist"}
SHIFT_KEYS = ("LEFT_SHIFT", "RIGHT_SHIFT")  # modifier only: fine mouse adjustment while held

HELP = ("W/S: +X/-X  A/D: +Y/-Y  E/Q: up/down  Z/X: yaw  O/C: open/close  P: pause  R: reset  M: focus view  "
        "1/2: global/wrist raw or annotated")
HOLD_KEYS = set(MOVE_KEYS) | set(YAW_KEYS) | set(GRIPPER_KEYS)


class KeyState:
    def __init__(self, actions: dict | None = None):
        self.action_keys = dict(ACTION_KEYS if actions is None else actions)
        self.held: set[str] = set()
        self.mods: set[str] = set()
        self.actions: list[str] = []
        self.events = 0

    def on_key(self, name: str, down: bool, repeat: bool = False):
        """One raw key event. Repeats (auto-repeat while held) are ignored."""
        if repeat:
            return
        if down:
            self.events += 1
            if name in SHIFT_KEYS:
                self.mods.add(name)
            elif name in self.action_keys:
                self.actions.append(self.action_keys[name])
            elif name in HOLD_KEYS:
                self.held.add(name)
        else:
            self.held.discard(name)
            self.mods.discard(name)

    @property
    def shift(self) -> bool:
        return bool(self.mods)

    def clear(self):
        self.held.clear()
        self.mods.clear()

    def pop_actions(self) -> list[str]:
        a, self.actions = self.actions, []
        return a

    def directions(self):
        """(linear [x, y, z] in -1/0/+1, yaw -1/0/+1, gripper -1/0/+1) from the held keys."""
        lin = [0, 0, 0]
        for k in self.held:
            if k in MOVE_KEYS:
                axis, sgn = MOVE_KEYS[k]
                lin[axis] += sgn
        yaw = sum(YAW_KEYS[k] for k in self.held if k in YAW_KEYS)
        grip = sum(GRIPPER_KEYS[k] for k in self.held if k in GRIPPER_KEYS)
        return lin, yaw, max(-1, min(1, grip))
