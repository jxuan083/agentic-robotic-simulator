"""Operation panel of the MuJoCo window: same content and behaviour as the Isaac panel (agentic_sim.isaac.panel).

  status card     status pill + reason, target, actual EE, follow error, gripper yaw / tilt / jaw, blocks
  TASK            only while the task executor runs: ledger per colour, holding, skill / phase, open question
  CONTROLS        Open / Close gripper, Yaw -15 / +15, Pause (Resume) / Reset / Reset view
  SCENE           blocks 1-8, colour DR on / off, seed (- / + / type digits / New seed); applied on Reset
  Help            collapsible
  Top view        collapsible (open by default): the scene seen from above; drag = move the target (relative,
                  like the 3D handle), wheel = height, right click = open / close the gripper

The state and logic (settings, pending, actions, drag / wheel accumulators, map mapping) need no OpenGL, so a headless
session can own a panel; draw() lays the widgets out and draws them on a draw2d.Canvas each frame (immediate mode),
remembering their rectangles for the mouse handlers.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np

MAP_X = (-0.04, 0.46)   # mount-frame x range shown (bottom..top), as in the Isaac top view
MAP_Y = (-0.36, 0.36)   # mount-frame y range shown (right..left)
WIDTH = 400
PAD = 10

REASON_TEXT = {"ok": "OK", "workspace_limit": "WORKSPACE LIMIT", "joint_limit": "JOINT LIMIT",
               "ik_no_solution": "OUT OF REACH", "joint_jump": "JOINT JUMP (step too large)"}
KIND_RGB = {"ok": (0.25, 0.82, 0.25), "limit": (1.0, 0.63, 0.13), "blocked": (1.0, 0.25, 0.25)}
PILL_BG = {"ok": (0.18, 0.35, 0.18, 1), "limit": (0.48, 0.31, 0.12, 1), "blocked": (0.54, 0.16, 0.16, 1)}
BG_PANEL = (0.12, 0.12, 0.12, 1)
BG_CARD = (0.17, 0.17, 0.17, 1)
BTN = (0.24, 0.24, 0.24, 1)
BTN_HOVER = (0.31, 0.31, 0.31, 1)
BLUE = (0.16, 0.35, 0.55, 1)
BLUE_HOVER = (0.2, 0.42, 0.63, 1)
AMBER = (0.82, 0.53, 0.12, 1)
AMBER_HOVER = (0.88, 0.59, 0.16, 1)
TEXT = (0.9, 0.9, 0.9)
DIM = (0.6, 0.6, 0.6)
SECTION = (0.72, 0.72, 0.72)
PENDING_RGB = (0.94, 0.69, 0.19)
EE_RGB = (0.1, 0.3, 1.0)


def kind_of(status: str) -> str:
    if status == "ok":
        return "ok"
    return "limit" if status == "workspace_limit" else "blocked"


def _inside(r, x, y) -> bool:
    return r is not None and r[0] <= x < r[0] + r[2] and r[1] <= y < r[1] + r[3]


class MjPanel:
    def __init__(self, settings: dict, max_blocks: int, fine_scale_fn, shift_fn, help_text: str):
        self._fine_scale_fn = fine_scale_fn
        self._shift_fn = shift_fn
        self.help_text = help_text
        self.max_blocks = int(max_blocks)
        self.applied = dict(settings)
        self.edit = dict(settings)
        self.seed_text = None            # digits being typed into the seed field (None = not editing)
        self.actions: list[str] = []
        self.dragging = False
        self._last = None
        self._delta = np.zeros(2)
        self._wheel = {"coarse": 0.0, "fine": 0.0}
        self._event_times: list = []
        self.help_open = False
        self.map_open = True
        self.scroll = 0.0
        self.cursor = (-1.0, -1.0)
        self.colors_text = ""
        self.view = {}
        self.buttons: dict = {}          # name -> (x, y, w, h), window coordinates (scripts click these)
        self._hits: list = []            # (rect, callback) of the last layout
        self.rect = None                 # the panel area
        self.map_rect = None             # top-view area (rendered by the viewer)
        self._content_h = 0.0
        self._seed_rect = None
        self.task_lines = None           # task executor lines (text, kind) while a task runs

    # ---- scene settings (same API as the Isaac panel) ------------------------------------
    def settings(self) -> dict:
        return {"blocks": int(min(max(int(self.edit["blocks"]), 1), self.max_blocks)),
                "color_dr": bool(self.edit["color_dr"]), "color_dr_seed": int(self.edit["color_dr_seed"] or 0)}

    def set_settings(self, settings: dict):
        for k in ("blocks", "color_dr", "color_dr_seed"):
            if settings.get(k) is not None:
                self.edit[k] = settings[k]
        self.seed_text = None

    def pending(self) -> dict:
        cur, out = self.settings(), {}
        for k, val in cur.items():
            if k == "color_dr_seed" and not cur["color_dr"]:
                continue  # the seed only matters with randomization on
            if val != self.applied.get(k):
                out[k] = val
        return out

    def mark_applied(self, settings: dict):
        self.applied = dict(settings)

    def set_task(self, lines):
        """Task card content: [(text, kind)] with kind text / dim / ask; None hides the card."""
        self.task_lines = lines

    def set_colors(self, text: str):
        self.colors_text = text

    def _new_seed(self):
        self.edit["color_dr_seed"] = random.randint(0, 99999)
        self.edit["color_dr"] = True
        self.seed_text = None

    def _commit_seed(self):
        if self.seed_text is not None:
            if self.seed_text:
                self.edit["color_dr_seed"] = int(self.seed_text)
            self.seed_text = None

    # ---- input accumulators (same API as the Isaac panel) --------------------------------
    def cancel(self):
        self.dragging = False
        self._last = None
        self._delta = np.zeros(2)
        self._wheel = {"coarse": 0.0, "fine": 0.0}
        self._event_times = []

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
        self.view = {"mode": mode, "paused": paused, "fine": fine, "status": status, "message": message,
                     "target": np.asarray(target, float).copy(), "yaw": yaw_offset, "ee": np.asarray(ee, float).copy(),
                     "blocks": [np.asarray(b, float) for b in blocks], "gripper": gripper,
                     "block_colors": block_colors, "tilt": tilt_deg, "in_placement": in_placement}

    # ---- top-view mapping (window coordinates) --------------------------------------------
    def map_scale(self) -> float:
        return self.map_rect[3] / (MAP_X[1] - MAP_X[0])  # px per metre

    def world_to_screen(self, x, y):
        mx, my, mw, mh = self.map_rect
        s = self.map_scale()
        return mx + mw / 2 - (y - 0.5 * (MAP_Y[0] + MAP_Y[1])) * s, my + (MAP_X[1] - x) * s

    def screen_to_world(self, sx, sy):
        mx, my, mw, mh = self.map_rect
        s = self.map_scale()
        return np.array([MAP_X[1] - (sy - my) / s, 0.5 * (MAP_Y[0] + MAP_Y[1]) - (sx - (mx + mw / 2)) / s])

    # ---- mouse / keys from the viewer (window coordinates) ---------------------------------
    def contains(self, x, y) -> bool:
        return _inside(self.rect, x, y)

    def over_map(self, x, y) -> bool:
        return self.map_open and _inside(self.map_rect, x, y)

    def press(self, x, y, button: str) -> bool:
        """button: left | right. Returns True when the panel takes the press."""
        if not self.contains(x, y):
            return False
        if self.over_map(x, y):
            if button == "left":
                self.dragging = True
                self._last = self.screen_to_world(x, y)
            elif button == "right":
                self.actions.append("toggle_gripper")
            return True
        if button != "left":
            return True
        editing_before = self.seed_text is not None
        for r, fn in reversed(self._hits):
            if _inside(r, x, y):
                fn()
                break
        if editing_before and self.seed_text is not None and not _inside(self._seed_rect, x, y):
            self._commit_seed()
        return True

    def move(self, x, y):
        self.cursor = (x, y)
        if self.dragging and self._last is not None and self.map_rect is not None:
            now = self.screen_to_world(x, y)
            k = self._fine_scale_fn() if self._shift_fn() else 1.0
            self._delta += (now - self._last) * k
            self._last = now
            self._event_times.append(time.monotonic())

    def release(self, button: str):
        if button == "left":
            self.dragging = False
            self._last = None

    def scroll_at(self, x, y, dy) -> bool:
        if self.over_map(x, y):
            self._wheel["fine" if self._shift_fn() else "coarse"] += dy
            self._event_times.append(time.monotonic())
            return True
        if self.contains(x, y):
            self.scroll = float(np.clip(self.scroll - dy * 30.0, 0.0, max(0.0, self._content_h - self.rect[3])))
            return True
        return False

    @property
    def editing(self) -> bool:
        return self.seed_text is not None

    def char(self, c: str):
        if self.seed_text is not None and c.isdigit() and len(self.seed_text) < 9:
            self.seed_text += c

    def edit_key(self, name: str) -> bool:
        """Keys while the seed field is being edited; returns True if consumed."""
        if self.seed_text is None:
            return False
        if name == "BACKSPACE":
            self.seed_text = self.seed_text[:-1]
        elif name in ("ENTER", "KP_ENTER", "TAB"):
            self._commit_seed()
        elif name == "ESCAPE":
            self.seed_text = None
        else:
            return name.isdigit() or name.startswith("KP_")
        return True

    # ---- layout + drawing --------------------------------------------------------------------
    def draw(self, cv, x0: float, y0: float, height: float):
        """Lay out and draw the panel at (x0, y0) with the given height; sets map_rect for the viewer."""
        self.rect = (x0, y0, WIDTH, height)
        self._hits = []
        self.buttons = {}
        cv.rect(x0, y0, WIDTH, height, BG_PANEL)
        lh = cv.line_h
        y = y0 + PAD - self.scroll
        x, w = x0 + PAD, WIDTH - 2 * PAD
        v = self.view
        cx, cy = self.cursor

        def button(name, bx, by, bw, bh, text, action=None, color=None, fn=None):
            hover = _inside((bx, by, bw, bh), cx, cy)
            base = color or BTN
            bg = (BLUE_HOVER if base == BLUE else AMBER_HOVER if base == AMBER else BTN_HOVER) if hover else base
            cv.label(bx, by, bw, bh, text, bg)
            if name:
                self.buttons[name] = (bx, by, bw, bh)
            self._hits.append(((bx, by, bw, bh), fn or (lambda a=action: self.actions.append(a))))

        def row(label, value, yy, color=TEXT):
            cv.text(x + 8, yy, label, DIM)
            cv.text(x + 112, yy, value, color)
            return yy + lh

        # header
        cv.text(x, y, "SO-101  EE Teleop", TEXT, big=True)
        y += cv.con.charHeightBig / cv.sy + 6
        mode_txt = f"input: {v.get('mode', 'ready')}" + ("  PAUSED" if v.get("paused") else "")

        # status card
        status = v.get("status", "ok")
        kind = kind_of(status)
        lines = cv.wrap(v.get("message", ""), w - 16) if kind != "ok" else []
        card_h = 8 + 30 + 4 + lh * len(lines) + lh * 5 + lh + 10
        cv.rect(x, y, w, card_h, BG_CARD)
        yy = y + 8
        cv.label(x + 8, yy, 200, 28, REASON_TEXT.get(status, status.upper()), PILL_BG[kind], KIND_RGB[kind])
        cv.text(x + 216, yy - 4, mode_txt, PENDING_RGB if v.get("paused") or "lost" in mode_txt else DIM)
        if v.get("fine"):
            cv.text(x + 216, yy + 12, f"fine x{self._fine_scale_fn():.2f}", DIM)
        yy += 34
        for ln in lines:
            cv.text(x + 8, yy, ln, DIM)
            yy += lh
        if v:
            t, e = v["target"], v["ee"]
            yy = row("Target", f"x {t[0]:+.3f}  y {t[1]:+.3f}  z {t[2]:+.3f}", yy)
            yy = row("Actual EE", f"x {e[0]:+.3f}  y {e[1]:+.3f}  z {e[2]:+.3f}", yy)
            yy = row("Follow error", f"{np.linalg.norm(t - e) * 1000:.1f} mm", yy)
            yy = row("Gripper", f"yaw {math.degrees(v['yaw']):+.0f}  tilt {v['tilt']:.0f} deg  jaw {v['gripper']:+.2f}",
                     yy)
            b = v["blocks"]
            yy = row("Blocks", f"x {b[0][0]:+.3f}  y {b[0][1]:+.3f}  z {b[0][2]:+.3f}" if len(b) == 1 else
                     f"{len(b)} blocks, {v['in_placement']} in the beige tray", yy)
        cv.frame(x + 8, yy + 4, 12, 12, (*KIND_RGB[kind], 1), 2)
        cv.text(x + 26, yy, "target ring (drag it)", DIM)
        cv.rect(x + 210, yy + 6, 9, 9, (*EE_RGB, 1))
        cv.text(x + 226, yy, "actual EE", DIM)
        y += card_h + 8

        # task executor (issue #5): ledger, holding, skill / phase, open question
        if self.task_lines:
            wrapped = [(ln, k) for t, k in self.task_lines for ln in (cv.wrap(t, w - 16) or [""])]
            task_h = 6 + lh + lh * len(wrapped) + 8
            cv.rect(x, y, w, task_h, BG_CARD)
            yy = y + 6
            cv.text(x + 8, yy, "TASK  (executor has the arm)", SECTION)
            yy += lh
            for ln, k in wrapped:
                cv.text(x + 8, yy, ln, PENDING_RGB if k == "ask" else (TEXT if k == "text" else DIM))
                yy += lh
            y += task_h + 8

        # controls
        bh = 30
        ctrl_h = 6 + lh + 3 * (bh + 4) + 6
        cv.rect(x, y, w, ctrl_h, BG_CARD)
        yy = y + 6
        cv.text(x + 8, yy, "CONTROLS", SECTION)
        yy += lh
        half = (w - 16 - 4) / 2
        button("open", x + 8, yy, half, bh, "Open gripper", "open")
        button("close", x + 12 + half, yy, half, bh, "Close gripper", "close")
        yy += bh + 4
        button("yaw_minus", x + 8, yy, half, bh, "Yaw -15", "yaw-")
        button("yaw_plus", x + 12 + half, yy, half, bh, "Yaw +15", "yaw+")
        yy += bh + 4
        third = (w - 16 - 8) / 3
        pend = self.pending()
        button("pause", x + 8, yy, third, bh, "Resume" if v.get("paused") else "Pause", "pause")
        button("reset", x + 12 + third, yy, third, bh, "Reset", "reset", color=AMBER if pend else BLUE)
        button("view", x + 16 + 2 * third, yy, third, bh, "Reset view", "reset_view")
        y += ctrl_h + 8

        # scene settings
        cur = self.settings()
        p_lines = []
        if pend:
            parts = []
            if "blocks" in pend:
                parts.append(f"{pend['blocks']} blocks (rebuilds the scene)")
            if "color_dr" in pend:
                parts.append("colour DR " + ("on" if pend["color_dr"] else "off"))
            if "color_dr_seed" in pend:
                parts.append(f"seed {pend['color_dr_seed']}")
            p_lines = cv.wrap("Press Reset to apply: " + ", ".join(parts), w - 16)
        c_lines = cv.wrap(self.colors_text, w - 120) or [""]
        scene_h = 6 + lh + 3 * 30 + lh * len(c_lines) + lh * len(p_lines) + 10
        cv.rect(x, y, w, scene_h, BG_CARD)
        yy = y + 6
        cv.text(x + 8, yy, "SCENE  (applied on Reset)", SECTION)
        yy += lh
        cv.text(x + 8, yy + 2, "Blocks", DIM)
        sx, sw = x + 112, (w - 120 - 3 * (self.max_blocks - 1)) / self.max_blocks
        shown = int(self.applied.get("blocks", 1))
        for i in range(1, self.max_blocks + 1):
            col = AMBER if (i == cur["blocks"] and i != shown) else (BLUE if i == shown else BTN)
            button(f"blocks_{i}", sx + (i - 1) * (sw + 3), yy, sw, 26, str(i), color=col,
                   fn=lambda i=i: self.edit.__setitem__("blocks", i))
        yy += 30
        cv.text(x + 8, yy + 2, "Colour DR", DIM)
        box = (x + 112, yy + 4, 18, 18)
        cv.frame(*box, TEXT + (1,), 1.5)
        if cur["color_dr"]:
            cv.rect(box[0] + 4, box[1] + 4, 10, 10, BLUE[:3] + (1,))
        self._hits.append(((x + 112, yy, w - 120, 26), lambda: self.edit.__setitem__("color_dr", not self.edit["color_dr"])))
        self.buttons["color_dr"] = (x + 112, yy, 26, 26)
        cv.text(x + 140, yy + 2, "new colours on every reset", DIM)
        yy += 30
        cv.text(x + 8, yy + 2, "Seed", DIM)
        button("seed_minus", x + 112, yy, 26, 26, "-",
               fn=lambda: self.edit.__setitem__("color_dr_seed", max(0, int(self.edit["color_dr_seed"] or 0) - 1)))
        self._seed_rect = (x + 142, yy, 86, 26)
        editing = self.seed_text is not None
        cv.rect(*self._seed_rect, (0.08, 0.08, 0.08, 1))
        cv.frame(*self._seed_rect, (*PENDING_RGB, 1) if editing else (0.35, 0.35, 0.35, 1), 1.5)
        cv.text(x + 148, yy + 1, (self.seed_text + "_") if editing else str(cur["color_dr_seed"]), TEXT)
        self._hits.append((self._seed_rect, lambda: setattr(self, "seed_text", "")))
        self.buttons["seed_field"] = self._seed_rect
        button("seed_plus", x + 232, yy, 26, 26, "+",
               fn=lambda: self.edit.__setitem__("color_dr_seed", int(self.edit["color_dr_seed"] or 0) + 1))
        button("new_seed", x + 264, yy, w - 272, 26, "New seed", fn=self._new_seed)
        yy += 30
        cv.text(x + 8, yy, "Colours", DIM)
        for ln in c_lines:
            cv.text(x + 112, yy, ln, TEXT)
            yy += lh
        for ln in p_lines:
            cv.text(x + 8, yy, ln, PENDING_RGB)
            yy += lh
        y += scene_h + 8

        # help (collapsible)
        def header(name, text, is_open, toggle):
            hover = _inside((x, y, w, 26), cx, cy)
            cv.rect(x, y, w, 26, BTN_HOVER if hover else BG_CARD)
            cv.text(x + 8, y + 1, ("[-] " if is_open else "[+] ") + text, TEXT)
            self._hits.append(((x, y, w, 26), toggle))
            self.buttons[name] = (x, y, w, 26)

        header("help", "Help: mouse and keyboard", self.help_open,
               lambda: setattr(self, "help_open", not self.help_open))
        y += 28
        if self.help_open:
            for para in ("Ring colour: green OK, amber workspace limit, red blocked (reason under the status). "
                         "Far out or high up the gripper tilts outward to keep reaching.",
                         "Mouse: drag the ring = move on the horizontal plane, wheel on the ring = height, hold "
                         "Shift = fine. Elsewhere: left drag rotates the view, right drag pans, wheel zooms; Alt + "
                         "drag on the ring also moves the view.",
                         "Top view: drag = move, wheel = height, right click = open / close.",
                         "Keys: " + self.help_text):
                for ln in cv.wrap(para, w - 8):
                    cv.text(x + 4, y, ln, DIM)
                    y += lh
                y += 4
        # top view (collapsible); the viewer renders the scene into map_rect
        header("map", "Top view (drag = move, wheel = height)", self.map_open,
               lambda: setattr(self, "map_open", not self.map_open))
        y += 28
        if self.map_open:
            mh = w * (MAP_X[1] - MAP_X[0]) / (MAP_Y[1] - MAP_Y[0])
            self.map_rect = (x, y, w, mh)
            y += mh + 4
        else:
            self.map_rect = None
        self._content_h = y + self.scroll - y0 + PAD

    def draw_map_overlay(self, cv, trays: dict | None = None):
        """Tray outlines and axis labels on top of the rendered top view (pale trays on a white table are hard
        to see from above)."""
        if self.map_rect is None:
            return
        mx, my, mw, mh = self.map_rect
        for name, t in (trays or {}).items():
            (cx, cy), (hx, hy) = t["center"], t["inner_half"]   # axis-aligned (trays are not rotated here)
            u0, v0 = self.world_to_screen(cx + hx, cy + hy)
            u1, v1 = self.world_to_screen(cx - hx, cy - hy)
            col = (0.85, 0.35, 0.55, 1) if "pink" in name else (0.62, 0.5, 0.25, 1)
            cv.frame(u0, v0, u1 - u0, v1 - v0, col, 2)
        cv.frame(mx, my, mw, mh, (0.35, 0.35, 0.35, 1), 1)
        for text, ty in (("+X forward (up)", my + 3), ("+Y left", my + mh - cv.line_h - 3)):
            cv.rect(mx + 3, ty, cv.text_width(text) + 8, cv.line_h, (0.12, 0.12, 0.12, 0.75))
            cv.text(mx + 7, ty, text, (0.92, 0.92, 0.92))
