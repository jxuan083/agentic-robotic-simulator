"""Drive the Isaac teleop app through real Kit input events (carb input provider), for repeatable acceptance runs.

Events go through the same path as a human: keyboard events reach KeyboardInput through carb, mouse events reach the
omni.ui panel (buttons, top view) and the viewport handle through the app window. The steps themselves (keys, waits,
checks, clicks, drags, wheel, camera, resize) are shared with the other backends (agentic_sim.teleop.script_core,
MouseScript); this class only turns them into carb events and finds the panel buttons. Isaac only:
  - {set: {blocks: 4, color_dr: true, color_dr_seed: 5}}   edit the panel's scene-settings fields (applied by Reset)
"""

from __future__ import annotations

import carb
import carb.input
import omni.appwindow
import omni.ui as ui
from carb.input import KeyboardEventType, MouseEventType

from agentic_sim.teleop.script_core import MouseScript


class ScriptedInput(MouseScript):

    def __init__(self, steps: list, panel, log, handle=None, set_camera=None):
        self.panel = panel
        self.handle = handle
        self.set_camera = set_camera
        self.provider = carb.input.acquire_input_provider()
        self.app_window = omni.appwindow.get_default_app_window()
        self.keyboard = self.app_window.get_keyboard()
        self.mouse = self.app_window.get_mouse()
        super().__init__(steps, log)

    def rebind(self, panel, handle, set_camera):
        """Continue this script in a rebuilt session (block count changed): point it at the new UI."""
        self.panel, self.handle, self.set_camera = panel, handle, set_camera
        self.log(f"script: continues in the rebuilt scene at step {self.i}")

    # ---- raw events --------------------------------------------------------------
    def _key(self, name, down):
        et = KeyboardEventType.KEY_PRESS if down else KeyboardEventType.KEY_RELEASE
        self.provider.buffer_keyboard_key_event(self.keyboard, et, getattr(carb.input.KeyboardInput, name), 0)

    def _mouse(self, et, x, y):
        dpi = ui.Workspace.get_dpi_scale()
        px, py = x * dpi, y * dpi
        w, h = ui.Workspace.get_main_window_width(), ui.Workspace.get_main_window_height()
        self.provider.buffer_mouse_event(self.mouse, et, (px / w, py / h), 0, (px, py))

    def _set(self, settings: dict):
        self.panel.set_settings(settings)

    def _mouse_move(self, x, y):
        self._mouse(MouseEventType.MOVE, x, y)

    def _mouse_button(self, down, x, y):
        self._mouse(MouseEventType.LEFT_BUTTON_DOWN if down else MouseEventType.LEFT_BUTTON_UP, x, y)

    def _mouse_scroll(self, dy):
        # like omni.kit.ui_test.emulate_mouse_scroll: the scroll delta goes in both coordinate slots
        d = (0.0, float(dy))
        self.provider.buffer_mouse_event(self.mouse, MouseEventType.SCROLL, d, 0, d)

    def _button_center(self, name):
        btn = self.panel.buttons[name]
        return btn.screen_position_x + btn.computed_width / 2, btn.screen_position_y + btn.computed_height / 2

    def _fine_scale(self):
        return self.handle._fine_scale_fn()

    def _resize(self, w, h):
        self.app_window.resize(int(w), int(h))
