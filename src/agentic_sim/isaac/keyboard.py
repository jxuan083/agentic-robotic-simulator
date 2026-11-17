"""Keyboard input for EE teleop via carb input on the Kit app window.

Held keys start only on a fresh KEY_PRESS (auto-repeat never re-arms a key), and are dropped on release, on focus
loss, on pause and on reset, so stale input is never replayed.
"""

from __future__ import annotations

import carb
import carb.input
import omni.appwindow

MOVE_KEYS = {
    "W": (0, +1), "S": (0, -1),   # +X forward / -X
    "A": (1, +1), "D": (1, -1),   # +Y left / -Y
    "E": (2, +1), "Q": (2, -1),   # +Z up / -Z
}
YAW_KEYS = {"Z": -1, "X": +1}
GRIPPER_KEYS = {"O": +1, "C": -1}  # open / close while held
ACTION_KEYS = {"P": "pause", "R": "reset", "M": "focus"}
SHIFT_KEYS = ("LEFT_SHIFT", "RIGHT_SHIFT")  # modifier only: fine mouse adjustment while held

HELP = "W/S: +X/-X  A/D: +Y/-Y  E/Q: up/down  Z/X: yaw  O/C: open/close  P: pause  R: reset  M: focus view"


class KeyboardInput:
    def __init__(self):
        self._app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._app_window.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)
        self.held: set[str] = set()
        self.mods: set[str] = set()
        self.actions: list[str] = []
        self.events = 0

    def _on_event(self, event, *args, **kwargs):
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.events += 1
            if name in SHIFT_KEYS:
                self.mods.add(name)
            elif name in ACTION_KEYS:
                self.actions.append(ACTION_KEYS[name])
            elif name in MOVE_KEYS or name in YAW_KEYS or name in GRIPPER_KEYS:
                self.held.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.held.discard(name)
            self.mods.discard(name)
        return False  # let Kit keep its own handling (camera fly needs the right mouse button anyway)

    def focused(self) -> bool:
        try:
            return bool(self._app_window.is_focused())
        except Exception:
            return True

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
        lin = [0, 0, 0]
        for k in self.held:
            if k in MOVE_KEYS:
                axis, sgn = MOVE_KEYS[k]
                lin[axis] += sgn
        yaw = sum(YAW_KEYS[k] for k in self.held if k in YAW_KEYS)
        grip = sum(GRIPPER_KEYS[k] for k in self.held if k in GRIPPER_KEYS)
        return lin, yaw, max(-1, min(1, grip))

    def close(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None
