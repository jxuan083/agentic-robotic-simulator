"""EE Teleop panel (omni.ui): status, controls, scene settings (blocks, colour DR), help, optional top view.

The main mouse input is the handle in the 3D viewport (viewport_handle.py). The top view is an auxiliary, collapsible
(default collapsed) view of the same state; dragging on it moves the target relatively, exactly like the handle.

Scene settings (block count, colour randomization on / off, seed) are edited here and take effect on the next Reset
(button or R key): colours change in place, a different block count rebuilds the scene (about 2 s).
Colours are omni.ui ABGR (0xAABBGGRR).
"""

from __future__ import annotations

import math
import random
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
    "ik_no_solution": "OUT OF REACH",
    "joint_jump": "JOINT JUMP (step too large)",
}

# palette (ABGR)
BG_CARD = 0xFF2B2B2B
BG_PILL = {"ok": 0xFF2E5A2E, "limit": 0xFF1F4F7A, "blocked": 0xFF2A2A8A}
TEXT = 0xFFE6E6E6
TEXT_DIM = 0xFF9A9A9A
TEXT_SECTION = 0xFFB8B8B8
PENDING = 0xFF30B0F0     # amber (ABGR of #F0B030)

STYLE = {
    "Label": {"color": TEXT, "font_size": 14},
    "Label::dim": {"color": TEXT_DIM, "font_size": 13},
    "Label::title": {"color": TEXT, "font_size": 19},
    "Label::section": {"color": TEXT_SECTION, "font_size": 13},
    "Label::value": {"color": TEXT, "font_size": 14},
    "Label::pending": {"color": PENDING, "font_size": 13},
    "Rectangle::card": {"background_color": BG_CARD, "border_radius": 6},
    "Button": {"background_color": 0xFF3A3A3A, "border_radius": 4, "margin": 1, "padding": 4},
    "Button:hovered": {"background_color": 0xFF4A4A4A},
    "Button:pressed": {"background_color": 0xFF5A5A5A},
    "Button.Label": {"color": TEXT, "font_size": 14},
    "Button::reset": {"background_color": 0xFF8C5A2A},           # blue
    "Button::reset:hovered": {"background_color": 0xFFA06C34},
    "Button::reset_pending": {"background_color": 0xFF1E86D0},   # amber: settings wait for this Reset
    "Button::reset_pending:hovered": {"background_color": 0xFF2A96E0},
    "Button::small": {"background_color": 0xFF3A3A3A, "border_radius": 4, "padding": 2},
    "Button::seg": {"background_color": 0xFF353535, "border_radius": 3, "padding": 1, "margin": 0},
    "Button::seg:hovered": {"background_color": 0xFF454545},
    "Button::seg_on": {"background_color": 0xFF8C5A2A, "border_radius": 3, "padding": 1, "margin": 0},
    "Button::seg_pending": {"background_color": 0xFF1E86D0, "border_radius": 3, "padding": 1, "margin": 0},
    "CollapsableFrame": {"background_color": BG_CARD, "secondary_color": 0xFF333333, "border_radius": 6,
                         "padding": 6, "margin": 0},
}


def kind_of(status: str) -> str:
    if status == "ok":
        return "ok"
    return "limit" if status == "workspace_limit" else "blocked"


def _card(build_fn, spacing=4):
    """A rounded card around the widgets that build_fn creates."""
    with ui.ZStack(height=0):
        ui.Rectangle(name="card")
        with ui.HStack(height=0):
            ui.Spacer(width=8)
            with ui.VStack(spacing=spacing, height=0):
                ui.Spacer(height=4)
                build_fn()
                ui.Spacer(height=4)
            ui.Spacer(width=8)


def _row(label: str, width=88):
    """Label column + value label; returns the value label."""
    with ui.HStack(height=20):
        ui.Label(label, name="dim", width=width)
        return ui.Label("", name="value")


class TeleopPanel:
    def __init__(self, trays: dict, help_text: str, tray_colors: dict, fine_scale_fn, shift_fn,
                 settings: dict | None = None, max_blocks: int = 8):
        self._fine_scale_fn = fine_scale_fn
        self._shift_fn = shift_fn
        self.actions: list[str] = []
        self.dragging = False
        self._last = None
        self._delta = np.zeros(2)
        self._wheel = {"coarse": 0.0, "fine": 0.0}
        self._event_times = []
        self._circles = {}
        self._block_marks = []
        self._w = int((MAP_Y[1] - MAP_Y[0]) * SCALE)
        self._h = int((MAP_X[1] - MAP_X[0]) * SCALE)
        self.applied = dict(settings or {"blocks": 1, "color_dr": False, "color_dr_seed": 0})
        self._max_blocks = max_blocks
        win_w = self._w + 28
        main_w = ui.Workspace.get_main_window_width()
        self.window = ui.Window("EE Teleop", width=win_w, height=760,
                                position_x=max(0, main_w - win_w - 6), position_y=30)
        self.window.frame.set_style(STYLE)
        with self.window.frame:
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                                   vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
                with ui.VStack(spacing=8, height=0):
                    self._build_header()
                    _card(self._build_status)
                    _card(self._build_controls, spacing=3)
                    _card(self._build_scene, spacing=5)
                    self._build_help(help_text)
                    self._build_map(trays, tray_colors)
        self._place(self.base_mark, 0.0, 0.0, 14)
        self._refresh_pending()

    # ---- layout ----------------------------------------------------------------------
    def _build_header(self):
        with ui.HStack(height=28, spacing=6):
            ui.Label("SO-101  EE Teleop", name="title")
            self.mode_label = ui.Label("", name="dim", alignment=ui.Alignment.RIGHT_CENTER)

    def _build_status(self):
        with ui.HStack(height=26, spacing=8):
            with ui.ZStack(width=180):
                self.pill = ui.Rectangle(style={"background_color": BG_PILL["ok"], "border_radius": 12})
                self.status_label = ui.Label("", alignment=ui.Alignment.CENTER,
                                             style={"color": KIND_COLOR["ok"], "font_size": 15})
            self.fine_label = ui.Label("", name="dim")
        self.reason_label = ui.Label("", height=0, word_wrap=True, name="dim")
        self.target_label = _row("Target")
        self.ee_label = _row("Actual EE")
        self.follow_label = _row("Follow error")
        self.tilt_label = _row("Gripper")
        self.block_label = _row("Blocks")
        with ui.HStack(height=16, spacing=6):
            ui.Circle(width=12, height=12, style={"background_color": 0x00000000,
                                                 "border_color": KIND_COLOR["ok"], "border_width": 2})
            ui.Label("target ring (drag it in the 3D view)", name="dim", width=0)
            ui.Spacer(width=10)
            ui.Circle(width=10, height=10, style={"background_color": EE_COLOR, "border_width": 0})
            ui.Label("actual EE", name="dim", width=0)

    def _build_controls(self):
        ui.Label("CONTROLS", name="section", height=16)
        self.buttons = {}
        rows = ((("open", "Open gripper", "open", None), ("close", "Close gripper", "close", None)),
                (("yaw_minus", "Yaw -15", "yaw-", None), ("yaw_plus", "Yaw +15", "yaw+", None)),
                (("pause", "Pause", "pause", None), ("reset", "Reset", "reset", "reset"),
                 ("focus", "Focus view (M)", "focus", None)))
        for row in rows:
            with ui.HStack(height=30, spacing=4):
                for key, text, action, name in row:
                    kw = {"name": name} if name else {}
                    self.buttons[key] = ui.Button(text, clicked_fn=lambda a=action: self.actions.append(a), **kw)

    def _build_scene(self):
        ui.Label("SCENE  (applied on Reset)", name="section", height=16)
        self._blocks_model = ui.SimpleIntModel(int(self.applied["blocks"]))
        self._dr_model = ui.SimpleBoolModel(bool(self.applied["color_dr"]))
        self._seed_model = ui.SimpleIntModel(int(self.applied["color_dr_seed"] or 0))
        with ui.HStack(height=24, spacing=6):
            ui.Label("Blocks", name="dim", width=88)
            with ui.HStack(spacing=3):
                self._block_buttons = [ui.Button(str(i), name="seg", width=ui.Fraction(1),
                                                 clicked_fn=lambda i=i: self._blocks_model.set_value(i))
                                       for i in range(1, self._max_blocks + 1)]
        with ui.HStack(height=22, spacing=6):
            ui.Label("Colour DR", name="dim", width=88)
            self.dr_check = ui.CheckBox(self._dr_model, width=20)
            ui.Label("new block and arm colours on every reset", name="dim")
        with ui.HStack(height=22, spacing=6):
            ui.Label("Seed", name="dim", width=88)
            self.seed_field = ui.IntField(self._seed_model, width=90)
            ui.Button("New seed", name="small", width=80, clicked_fn=self._new_seed)
            ui.Spacer()
        with ui.HStack(height=18):
            ui.Label("Colours", name="dim", width=88)
            self.colors_label = ui.Label("", name="value", word_wrap=True)
        self.pending_label = ui.Label("", name="pending", height=0, word_wrap=True)
        for m in (self._blocks_model, self._dr_model, self._seed_model):
            m.add_value_changed_fn(lambda _m: self._refresh_pending())

    def _build_help(self, help_text):
        frame = ui.CollapsableFrame("Help: mouse and keyboard", collapsed=True, height=0)
        with frame:
            with ui.VStack(spacing=4, height=0):
                ui.Label("Ring colour: green OK, amber workspace limit, red blocked (reason shown under the "
                         "status). Far out or high up the gripper tilts outward to keep reaching.",
                         word_wrap=True, name="dim", height=0)
                ui.Label("Mouse: drag the ring = move on the horizontal plane, wheel on the ring = height, "
                         "hold Shift = fine. Alt / right button = camera.", word_wrap=True, height=0)
                ui.Label("Keys: " + help_text, word_wrap=True, height=0)

    def _build_map(self, trays, tray_colors):
        self.map_frame = ui.CollapsableFrame("Top view (auxiliary)", collapsed=True, height=0)
        with self.map_frame:
            self.map = ui.ZStack(width=self._w, height=self._h)
            with self.map:
                ui.Rectangle(style={"background_color": 0xFF303030})
                for name, tray in trays.items():
                    self._rect_for_tray(tray, tray_colors[name])
                self._block_layer = ui.ZStack()
                self.ee_mark = self._marker(EE_COLOR, 8)
                self.target_mark = self._marker(0x00000000, 14, border=KIND_COLOR["ok"])
                self.base_mark = self._marker(0xFFAAAAAA, 14)
            self.map.set_mouse_pressed_fn(self._on_press)
            self.map.set_mouse_moved_fn(self._on_move)
            self.map.set_mouse_released_fn(self._on_release)
            self.map.set_mouse_wheel_fn(self._on_wheel)

    # ---- scene settings ----------------------------------------------------------------
    def _new_seed(self):
        self._seed_model.set_value(random.randint(0, 99999))
        if not self._dr_model.get_value_as_bool():
            self._dr_model.set_value(True)

    def settings(self) -> dict:
        """Settings as edited in the panel (what the next Reset applies)."""
        return {"blocks": int(min(max(self._blocks_model.get_value_as_int(), 1), self._max_blocks)),
                "color_dr": bool(self._dr_model.get_value_as_bool()),
                "color_dr_seed": int(self._seed_model.get_value_as_int())}

    def set_settings(self, settings: dict):
        """Show these values in the editors (e.g. options set by a client), without marking them applied."""
        if "blocks" in settings:
            self._blocks_model.set_value(int(settings["blocks"]))
        if "color_dr" in settings:
            self._dr_model.set_value(bool(settings["color_dr"]))
        if settings.get("color_dr_seed") is not None:
            self._seed_model.set_value(int(settings["color_dr_seed"]))

    def pending(self) -> dict:
        """Settings that differ from what the scene currently shows."""
        cur, out = self.settings(), {}
        for k, val in cur.items():
            if k == "color_dr_seed" and not cur["color_dr"]:
                continue  # the seed only matters with randomization on
            if val != self.applied.get(k):
                out[k] = val
        return out

    def mark_applied(self, settings: dict):
        self.applied = dict(settings)
        self._refresh_pending()

    def _refresh_pending(self):
        p = self.pending()
        want, shown = self.settings()["blocks"], int(self.applied.get("blocks", 1))
        for i, b in enumerate(self._block_buttons, start=1):
            b.name = "seg_pending" if (i == want and want != shown) else ("seg_on" if i == shown else "seg")
        if not p:
            self.pending_label.text = ""
            self.buttons["reset"].name = "reset"
            return
        parts = []
        if "blocks" in p:
            parts.append(f"{p['blocks']} blocks (rebuilds the scene, about 2 s)")
        if "color_dr" in p:
            parts.append("colour DR " + ("on" if p["color_dr"] else "off"))
        if "color_dr_seed" in p:
            parts.append(f"seed {p['color_dr_seed']}")
        self.pending_label.text = "Press Reset to apply: " + ", ".join(parts)
        self.buttons["reset"].name = "reset_pending"

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

    def _ensure_block_marks(self, n):
        if len(self._block_marks) == n:
            return
        self._block_layer.clear()
        self._block_marks = []
        with self._block_layer:
            for _ in range(n):
                self._block_marks.append(self._marker(0xFF2020D0, 12, border=0xFFFFFFFF))

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

    def update(self, mode, paused, fine, status, message, target, yaw_offset, ee, blocks, gripper, block_colors,
               tilt_deg=0.0, in_placement=0):
        kind = kind_of(status)
        self.mode_label.text = f"input: {'ready' if mode == 'none' else mode}{'   PAUSED' if paused else ''}"
        self.fine_label.text = f"fine x{self._fine_scale_fn():.2f}" if fine else ""
        self.status_label.text = REASON_TEXT.get(status, status.upper())
        self.status_label.style = {"color": KIND_COLOR[kind], "font_size": 15}
        self.pill.style = {"background_color": BG_PILL[kind], "border_radius": 12}
        self.reason_label.text = message if kind != "ok" else ""
        self.target_label.text = f"x {target[0]:+.3f}  y {target[1]:+.3f}  z {target[2]:+.3f} m"
        self.ee_label.text = f"x {ee[0]:+.3f}  y {ee[1]:+.3f}  z {ee[2]:+.3f} m"
        err = float(np.linalg.norm(np.asarray(target) - np.asarray(ee)))
        self.follow_label.text = f"{err * 1000:.1f} mm"
        self.tilt_label.text = (f"yaw {math.degrees(yaw_offset):+.0f} deg   tilt {tilt_deg:.0f} deg   "
                                f"jaw cmd {gripper:+.2f} rad")
        n = len(blocks)
        b0 = blocks[0]
        if n == 1:
            self.block_label.text = f"x {b0[0]:+.3f}  y {b0[1]:+.3f}  z {b0[2]:+.3f} m"
        else:
            self.block_label.text = f"{n} blocks, {in_placement} in the beige tray"
        self.buttons["pause"].text = "Resume" if paused else "Pause"
        if not self.map_frame.collapsed:
            self._circles[id(self.target_mark)].style = {"background_color": 0x00000000,
                                                         "border_color": KIND_COLOR[kind], "border_width": 2}
            self._place(self.target_mark, target[0], target[1], 14)
            self._place(self.ee_mark, ee[0], ee[1], 8)
            self._ensure_block_marks(n)
            for mark, pos, col in zip(self._block_marks, blocks, block_colors):
                self._circles[id(mark)].style = {"background_color": col, "border_color": 0xFFFFFFFF,
                                                 "border_width": 2}
                self._place(mark, pos[0], pos[1], 12)
