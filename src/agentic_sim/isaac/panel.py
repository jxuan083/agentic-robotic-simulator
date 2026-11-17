"""Compact control panel (omni.ui): status, legend, gripper / yaw / pause / reset, optional top view.

The main mouse input is the handle in the 3D viewport (viewport_handle.py). The top view is an auxiliary, collapsible
(default collapsed) view of the same state; dragging on it moves the target relatively, exactly like the handle.
Colours are omni.ui ABGR (0xAABBGGRR).
"""

from __future__ import annotations

import math
import time

import numpy as np
import omni.ui as ui

MAP_X = (-0.04, 0.46)   # mount-frame x range shown (bottom..top); covers both trays
MAP_Y = (-0.36, 0.36)   # mount-frame y range shown (right..left)
SCALE = 480.0           # px per metre

KIND_COLOR = {"ok": 0xFF40D040, "limit": 0xFF20A0FF, "blocked": 0xFF3030FF}
EE_COLOR = 0xFFFF4D1A    # matches the blue EE sphere in the viewport (0.1, 0.3, 1.0)
REASON_TEXT = {
    "ok": "OK",
    "workspace_limit": "WORKSPACE LIMIT",
    "joint_limit": "JOINT LIMIT",
    "ik_no_solution": "NO IK SOLUTION",
    "joint_jump": "JOINT JUMP (step too large)",
}


def kind_of(status: str) -> str:
    if status == "ok":
        return "ok"
    return "limit" if status == "workspace_limit" else "blocked"


class TeleopPanel:
    def __init__(self, trays: dict, help_text: str, tray_colors: dict, fine_scale_fn, shift_fn):
        self._fine_scale_fn = fine_scale_fn
        self._shift_fn = shift_fn
        self.actions: list[str] = []
        self.dragging = False
        self._last = None
        self._delta = np.zeros(2)
        self._wheel = {"coarse": 0.0, "fine": 0.0}
        self._event_times = []
        self._circles = {}
        self._w = int((MAP_Y[1] - MAP_Y[0]) * SCALE)
        self._h = int((MAP_X[1] - MAP_X[0]) * SCALE)
        win_w = self._w + 24
        main_w = ui.Workspace.get_main_window_width()
        self.window = ui.Window("EE Teleop", width=win_w, height=self._h + 420,
                                position_x=max(0, main_w - win_w - 6), position_y=30)
        with self.window.frame:
            with ui.VStack(spacing=3):
                self.mode_label = ui.Label("", height=20, style={"font_size": 18})
                self.status_label = ui.Label("", height=20, style={"font_size": 16})
                self.reason_label = ui.Label("", height=18, word_wrap=True)
                with ui.HStack(height=18, spacing=6):
                    ui.Label("Legend:", width=50)
                    ui.Circle(width=12, height=12, style={"background_color": 0x00000000,
                                                         "border_color": KIND_COLOR["ok"], "border_width": 2})
                    ui.Label("target (drag this ring in the 3D view)", width=0)
                with ui.HStack(height=18, spacing=6):
                    ui.Spacer(width=50)
                    ui.Circle(width=10, height=10, style={"background_color": EE_COLOR, "border_width": 0})
                    ui.Label("actual EE (where the gripper centre is)", width=0)
                ui.Label("Ring colour: green OK, amber workspace limit, red blocked (reason above)", height=18,
                         word_wrap=True, style={"color": 0xFFB0B0B0})
                self.target_label = ui.Label("", height=18)
                self.ee_label = ui.Label("", height=18)
                self.follow_label = ui.Label("", height=18)
                self.block_label = ui.Label("", height=18)
                self.colors_label = ui.Label("", height=18, style={"color": 0xFFB0B0B0})
                ui.Label("Mouse: drag the ring = move on the horizontal plane, wheel on the ring = height, "
                         "hold Shift = fine. Alt / right button = camera.", height=36, word_wrap=True)
                ui.Label(help_text + "  (keyboard unchanged)", height=36, word_wrap=True)
                self.buttons = {}
                rows = ((("open", "Open gripper", "open"), ("close", "Close gripper", "close"),
                         ("pause", "Pause", "pause")),
                        (("yaw_minus", "Yaw -15", "yaw-"), ("yaw_plus", "Yaw +15", "yaw+"), ("reset", "Reset", "reset")),
                        (("focus", "Focus view (M)", "focus"),))
                for row in rows:
                    with ui.HStack(height=26, spacing=4):
                        for key, text, action in row:
                            self.buttons[key] = ui.Button(text, clicked_fn=lambda a=action: self.actions.append(a))
                self.map_frame = ui.CollapsableFrame("Top view (auxiliary)", collapsed=True)
                with self.map_frame:
                    self.map = ui.ZStack(width=self._w, height=self._h)
                    with self.map:
                        ui.Rectangle(style={"background_color": 0xFF303030})
                        for name, tray in trays.items():
                            self._rect_for_tray(tray, tray_colors[name])
                        self.block_mark = self._marker(0xFF2020D0, 12, border=0xFFFFFFFF)
                        self.ee_mark = self._marker(EE_COLOR, 8)
                        self.target_mark = self._marker(0x00000000, 14, border=KIND_COLOR["ok"])
                        self.base_mark = self._marker(0xFFAAAAAA, 14)
                    self.map.set_mouse_pressed_fn(self._on_press)
                    self.map.set_mouse_moved_fn(self._on_move)
                    self.map.set_mouse_released_fn(self._on_release)
                    self.map.set_mouse_wheel_fn(self._on_wheel)
        self._place(self.base_mark, 0.0, 0.0, 14)

    # ---- coordinate helpers --------------------------------------------------
    def world_to_local(self, x, y):
        return (MAP_Y[1] - y) * SCALE, (MAP_X[1] - x) * SCALE

    def local_to_world(self, u, v):
        return MAP_X[1] - v / SCALE, MAP_Y[1] - u / SCALE

    def world_to_screen(self, x, y):
        u, v = self.world_to_local(x, y)
        return self.map.screen_position_x + u, self.map.screen_position_y + v

    def _screen_to_world(self, sx, sy):
        return np.array(self.local_to_world(sx - self.map.screen_position_x, sy - self.map.screen_position_y))

    # ---- drawing ---------------------------------------------------------------
    def _marker(self, color, size, border=None):
        placer = ui.Placer(offset_x=0, offset_y=0)
        style = {"background_color": color, "border_width": 0}
        if border is not None:
            style.update({"border_color": border, "border_width": 2})
        with placer:
            circle = ui.Circle(width=size, height=size, style=style)
        self._circles[id(placer)] = circle
        return placer

    def _place(self, placer, x, y, size):
        u, v = self.world_to_local(x, y)
        placer.offset_x = ui.Pixel(u - size / 2)
        placer.offset_y = ui.Pixel(v - size / 2)

    def _rect_for_tray(self, tray, color):
        cx, cy = tray["center"]
        hx, hy = tray["inner_half"]
        # axis-aligned approximation (trays are not rotated in the provisional scene)
        u0, v0 = self.world_to_local(cx + hx, cy + hy)
        with ui.Placer(offset_x=ui.Pixel(u0), offset_y=ui.Pixel(v0)):
            ui.Rectangle(width=2 * hy * SCALE, height=2 * hx * SCALE,
                         style={"background_color": color, "border_color": 0xFFFFFFFF, "border_width": 1})

    # ---- top-view mouse (relative, like the viewport handle) ---------------------------------
    def _on_press(self, x, y, button, mod):
        if button == 0:
            self.dragging = True
            self._last = self._screen_to_world(x, y)
        elif button == 1:
            self.actions.append("toggle_gripper")

    def _on_move(self, x, y, mod, pressed):
        if self.dragging and self._last is not None:
            now = self._screen_to_world(x, y)
            k = self._fine_scale_fn() if self._shift_fn() else 1.0
            self._delta += (now - self._last) * k
            self._last = now
            self._event_times.append(time.monotonic())

    def _on_release(self, x, y, button, mod):
        if button == 0:
            self.dragging = False
            self._last = None

    def _on_wheel(self, x, y, mod):
        # y is the wheel delta (positive = away from the user = up)
        self._wheel["fine" if self._shift_fn() else "coarse"] += y
        self._event_times.append(time.monotonic())

    # ---- per-frame -------------------------------------------------------------
    def cancel(self):
        """Stop any drag and drop accumulated input (focus loss, pause, reset)."""
        self.dragging = False
        self._last = None
        self._delta = np.zeros(2)
        self._wheel = {"coarse": 0.0, "fine": 0.0}
        self._event_times = []

    def set_colors(self, text: str):
        self.colors_label.text = text

    def pop_actions(self):
        a, self.actions = self.actions, []
        return a

    def pop_drag(self):
        d, self._delta = self._delta.copy(), np.zeros(2)
        times, self._event_times = self._event_times, []
        return d, times

    def pop_wheel(self):
        w, self._wheel = self._wheel, {"coarse": 0.0, "fine": 0.0}
        return w

    def update(self, mode, paused, fine, status, message, target, yaw_offset, ee, block, gripper, block_color):
        kind = kind_of(status)
        fine_txt = f"   [FINE x{self._fine_scale_fn():.2f}]" if fine else ""
        self.mode_label.text = f"Mode: {mode}{'   [PAUSED]' if paused else ''}{fine_txt}"
        self.status_label.text = f"Target: {REASON_TEXT.get(status, status.upper())}"
        self.status_label.style = {"color": KIND_COLOR[kind], "font_size": 16}
        self.reason_label.text = message if kind != "ok" else ""
        self.target_label.text = (f"Target    x={target[0]:+.3f} y={target[1]:+.3f} z={target[2]:+.3f} m  "
                                  f"yaw {math.degrees(yaw_offset):+.0f} deg  grip {gripper:+.2f}")
        self.ee_label.text = f"Actual EE x={ee[0]:+.3f} y={ee[1]:+.3f} z={ee[2]:+.3f} m"
        err = float(np.linalg.norm(np.asarray(target) - np.asarray(ee)))
        self.follow_label.text = f"Follow error (target - actual): {err * 1000:.1f} mm"
        self.block_label.text = f"Block     x={block[0]:+.3f} y={block[1]:+.3f} z={block[2]:+.3f} m"
        if not self.map_frame.collapsed:
            self._circles[id(self.target_mark)].style = {"background_color": 0x00000000, "border_color": KIND_COLOR[kind],
                                             "border_width": 2}
            self._circles[id(self.block_mark)].style = {"background_color": block_color, "border_color": 0xFFFFFFFF,
                                            "border_width": 2}
            self._place(self.target_mark, target[0], target[1], 14)
            self._place(self.ee_mark, ee[0], ee[1], 8)
            self._place(self.block_mark, block[0], block[1], 12)
