"""Keyboard input for EE teleop via carb input on the Kit app window.

The key map and held-key rules live in agentic_sim.teleop.keys (shared with the MuJoCo backend); this class only
feeds carb keyboard events into it.
"""

from __future__ import annotations

import carb
import carb.input
import omni.appwindow

from agentic_sim.teleop.keys import ACTION_KEYS, GRIPPER_KEYS, HELP, MOVE_KEYS, SHIFT_KEYS, YAW_KEYS, KeyState  # noqa: F401


class KeyboardInput(KeyState):
    def __init__(self):
        super().__init__()
        self._app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._app_window.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.on_key(event.input.name, True)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.on_key(event.input.name, False)
        return False  # let Kit keep its own handling (camera fly needs the right mouse button anyway)

    def focused(self) -> bool:
        try:
            return bool(self._app_window.is_focused())
        except Exception:
            return True

    def close(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None
