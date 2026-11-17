"""Drive the teleop app through real Kit input events (carb input provider), for repeatable acceptance runs.

Events go through the same path as a human: keyboard events reach KeyboardInput through carb, mouse events reach the
omni.ui panel (map drag / wheel / buttons) through the app window. Conditions (``until``) only decide when the
"operator" releases a key, like a person watching the screen; they never touch the robot, the target or the block.

Script = YAML list of steps:
  - {note: text}                                   log a phase label
  - {wait: seconds}
  - {hold: [W, A], until: "target[0] >= 0.17", timeout: 8}
  - {tap: P}                                       press + release on the next frame
  - {drag_to: [x, y], tol: 0.002, timeout: 10}     left-drag on the map from the target marker to (x, y)
  - {drag_hold: [x, y], seconds: 2}                drag to (x, y), keep holding (for unreachable tests), release
  - {wheel: -1, until: "target[2] <= 0.03", timeout: 8}   one notch per frame until the condition holds
  - {click: close}                                 click a panel button by name (open/close/yaw_minus/yaw_plus/pause/reset)
  - {focus_loss: 1.0, hold: [W]}                   hold keys, then report the window unfocused for N seconds (simulated)
  - {press: W} / {release: W}                      key down / up that stays held across steps
  - {mark: name}                                   snapshot the frame state; checks read it as m['name']['target']
  - {check: "expr", label: name}                   record a pass/fail result
Viewport handle (primary mouse path; the cursor is driven like an operator watching the screen):
  - {vdrag_to: [x, y], tol: 0.002, timeout: 10}    press on the handle, move the cursor to bring the target to (x, y)
  - {vdrag_push: [dx, dy], seconds: 2}             press on the handle and keep pushing the cursor in world
                                                   direction (dx, dy) (e.g. into the reach boundary), then release
  - {vwheel: -1, until: "target[2] <= 0.03"}       scroll over the handle, one notch per frame
  - {camera: {eye: [..], target: [..]}}            change the viewport camera
  - {resize: [w, h]}                               resize the main application window
Each viewport drag stores m['drag'] = {press_jump, max_step} (target motion at press, largest per-frame step).
The focus_loss step snapshots m['focus_lost'] on the frame the window is reported unfocused.
"""

from __future__ import annotations

import copy
import time

import carb
import carb.input
import omni.appwindow
import omni.ui as ui
from carb.input import KeyboardEventType, MouseEventType


class ScriptedInput:
    def __init__(self, steps: list, panel, log, handle=None, set_camera=None):
        self.steps = list(steps)
        self.i = -1
        self.panel = panel
        self.handle = handle
        self.set_camera = set_camera
        self.log = log
        self._cursor = None
        self.provider = carb.input.acquire_input_provider()
        self.app_window = omni.appwindow.get_default_app_window()
        self.keyboard = self.app_window.get_keyboard()
        self.mouse = self.app_window.get_mouse()
        self.results = []
        self.focus_override = None  # None = real focus; False = simulated focus loss
        self._phase = None
        self._t0 = 0.0
        self._held = []
        self._sticky = []
        self.marks = {}
        self._drag = None
        self.done = False
        self._next()

    # ---- raw events --------------------------------------------------------------
    def _key(self, name, down):
        et = KeyboardEventType.KEY_PRESS if down else KeyboardEventType.KEY_RELEASE
        self.provider.buffer_keyboard_key_event(self.keyboard, et, getattr(carb.input.KeyboardInput, name), 0)

    def _mouse(self, et, x, y):
        dpi = ui.Workspace.get_dpi_scale()
        px, py = x * dpi, y * dpi
        w, h = ui.Workspace.get_main_window_width(), ui.Workspace.get_main_window_height()
        self.provider.buffer_mouse_event(self.mouse, et, (px / w, py / h), 0, (px, py))

    def _release_all(self):
        for k in self._held:
            self._key(k, False)
        self._held = []

    # ---- sequencing ------------------------------------------------------------
    def _next(self):
        self._release_all()
        self.i += 1
        self._phase = None
        self._drag = None
        self._t0 = time.monotonic()
        if self.i >= len(self.steps):
            self.done = True
            self.log("script: done")
            return
        self.log(f"script step {self.i}: {self.steps[self.i]}")

    def _cond(self, expr, ns):
        env = {"__builtins__": {}, "abs": abs, "min": min, "max": max, "len": len, "sorted": sorted, "zip": zip, "all": all, "any": any, "round": round,
               "m": self.marks}
        return bool(eval(expr, env, ns))

    @staticmethod
    def _snap(ns):
        return {k: copy.deepcopy(v) for k, v in ns.items() if k != "stats"}

    def finish(self):
        """Release every key the script still holds (call before exit)."""
        self._release_all()
        for k in self._sticky:
            self._key(k, False)
        self._sticky = []

    def step(self, ns: dict):
        """Call once per frame after the frame's state is known; ns holds target/ee/block/status/..."""
        if self.done:
            return
        s = self.steps[self.i]
        el = time.monotonic() - self._t0
        timeout = s.get("timeout", 15.0)
        if "note" in s:
            self._next()
        elif "wait" in s:
            if el >= s["wait"]:
                self._next()
        elif "press" in s:
            self._key(s["press"], True)
            self._sticky.append(s["press"])
            self._next()
        elif "release" in s:
            self._key(s["release"], False)
            self._sticky.remove(s["release"])
            self._next()
        elif "mark" in s:
            self.marks[s["mark"]] = self._snap(ns)
            self._next()
        elif "check" in s:
            ok = self._cond(s["check"], ns)
            self.results.append({"label": s.get("label", s["check"]), "ok": ok, "expr": s["check"]})
            self.log(f"CHECK {'PASS' if ok else 'FAIL'}: {s.get('label', s['check'])}")
            self._next()
        elif "tap" in s:
            if self._phase is None:
                self._key(s["tap"], True)
                self._phase = "down"
            else:
                self._key(s["tap"], False)
                self._next()
        elif "focus_loss" in s:
            if self._phase is None:
                for k in s.get("hold", []):
                    self._key(k, True)
                self._held = list(s.get("hold", []))
                self._phase = "holding"
                self._t0 = time.monotonic()
            elif self._phase == "holding" and el >= 0.5:
                self.marks["focus_lost"] = self._snap(ns)
                self.focus_override = False
                self._phase = "unfocused"
                self._t0 = time.monotonic()
            elif self._phase == "unfocused" and el >= s["focus_loss"]:
                self.focus_override = None
                self._phase = "refocused"
                self._t0 = time.monotonic()
            elif self._phase == "refocused" and el >= 0.5:
                self._next()
        elif "hold" in s:
            if self._phase is None:
                for k in s["hold"]:
                    self._key(k, True)
                self._held = list(s["hold"])
                self._phase = "holding"
            elif self._cond(s["until"], ns) or el >= timeout:
                if el >= timeout:
                    self.log(f"script: hold timeout {s}")
                self._next()
        elif "wheel" in s:
            if self._cond(s["until"], ns) or el >= timeout:
                self._next()
            else:
                x, y = self.panel.world_to_screen(*ns["target"][:2])
                if self._phase is None:
                    self._mouse(MouseEventType.MOVE, x, y)
                    self._phase = "scroll"
                else:
                    # like omni.kit.ui_test.emulate_mouse_scroll: the scroll delta goes in both coordinate slots
                    d = (0.0, float(s["wheel"]))
                    self.provider.buffer_mouse_event(self.mouse, MouseEventType.SCROLL, d, 0, d)
        elif "drag_to" in s or "drag_hold" in s:
            goal = s.get("drag_to") or s.get("drag_hold")
            if self._phase is None:
                self._drag = self.panel.world_to_screen(*ns["target"][:2])
                self._mouse(MouseEventType.MOVE, *self._drag)
                self._phase = "move"
            elif self._phase == "move":
                self._mouse(MouseEventType.LEFT_BUTTON_DOWN, *self._drag)
                self._phase = "drag"
                self._t0 = time.monotonic()
            elif self._phase == "drag":
                gx, gy = self.panel.world_to_screen(*goal)
                sx, sy = self._drag
                k = min(1.0, (time.monotonic() - self._t0) / 0.5)  # cursor reaches the goal over 0.5 s
                self._mouse(MouseEventType.MOVE, sx + (gx - sx) * k, sy + (gy - sy) * k)
                t = ns["target"]
                reached = abs(t[0] - goal[0]) < s.get("tol", 0.002) and abs(t[1] - goal[1]) < s.get("tol", 0.002)
                if "drag_hold" in s:
                    reached = el >= s.get("seconds", 2.0)
                if reached or el >= timeout:
                    if el >= timeout and "drag_to" in s:
                        self.log(f"script: drag timeout {s}")
                    self._mouse(MouseEventType.LEFT_BUTTON_UP, sx + (gx - sx) * k, sy + (gy - sy) * k)
                    self._phase = "released"
            else:
                self._next()
        elif "click" in s:
            btn = self.panel.buttons[s["click"]]
            x = btn.screen_position_x + btn.computed_width / 2
            y = btn.screen_position_y + btn.computed_height / 2
            if self._phase is None:
                self._mouse(MouseEventType.MOVE, x, y)
                self._phase = "move"
            elif self._phase == "move":
                self._mouse(MouseEventType.LEFT_BUTTON_DOWN, x, y)
                self._phase = "down"
            else:
                self._mouse(MouseEventType.LEFT_BUTTON_UP, x, y)
                self._next()
        elif "vdrag_to" in s or "vdrag_push" in s:
            self._vdrag(s, ns, el, timeout)
        elif "vwheel" in s:
            if self._cond(s["until"], ns) or el >= timeout:
                if el >= timeout:
                    self.log(f"script: vwheel timeout {s}")
                self._next()
            else:
                x, y = self.handle.world_to_screen(ns["target"])
                if self._phase is None:
                    self._mouse(MouseEventType.MOVE, x, y)
                    self._phase = "hover"
                elif self._phase == "hover":  # one frame for the hover gesture to register
                    self._phase = "scroll"
                else:
                    d = (0.0, float(s["vwheel"]))
                    self.provider.buffer_mouse_event(self.mouse, MouseEventType.SCROLL, d, 0, d)
        elif "camera" in s:
            self.set_camera(s["camera"]["eye"], s["camera"]["target"])
            self._next()
        elif "resize" in s:
            w, h = s["resize"]
            try:
                self.app_window.resize(int(w), int(h))
                self.log(f"script: resized main window to {w}x{h}")
            except Exception as e:  # noqa: BLE001
                self.log(f"script: resize failed: {e}")
            self._next()
        else:
            raise ValueError(f"unknown script step {s}")

    # ---- viewport handle drag ------------------------------------------------------
    PX_PER_FRAME = 6.0      # cursor speed cap of the scripted "operator" (pixels per frame)
    WORLD_PER_FRAME = 0.005  # and at most 5 mm of target motion per frame

    def _vdrag(self, s, ns, el, timeout):
        import numpy as np
        h = self.handle
        tgt = np.asarray(ns["target"], float)
        if self._phase is None:
            self._cursor = np.array(h.world_to_screen(tgt), float)
            self._mouse(MouseEventType.MOVE, *self._cursor)
            self._phase = "hover"
        elif self._phase == "hover":
            self._mouse(MouseEventType.LEFT_BUTTON_DOWN, *self._cursor)
            self._press_target = tgt.copy()
            self._max_step = 0.0
            self._frames = 0
            self._phase = "pressed"
        elif self._phase == "pressed":  # hold still for a few frames: the target must not move on press
            self._frames += 1
            if self._frames >= 3:
                self.marks["drag"] = {"press_jump": float(np.linalg.norm(tgt - self._press_target)), "max_step": 0.0}
                self._press_cursor = self._cursor.copy()
                self._settle = 0
                self._phase = "drag"
                self._t0 = time.monotonic()
        elif self._phase == "drag":
            self._max_step = max(self._max_step, float(ns.get("target_step", 0.0)))
            self.marks["drag"]["max_step"] = self._max_step
            k = h._fine_scale_fn() if ns.get("fine") else 1.0
            if "vdrag_to" in s:
                # the mapping is 1:1 on the plane: aim the cursor at the goal's screen point (scaled for Shift) and
                # hold it there until the target, which follows one frame later, arrives
                goal = np.array([s["vdrag_to"][0], s["vdrag_to"][1], self._press_target[2]])
                aim = self._press_cursor + (np.array(h.world_to_screen(goal))
                                            - np.array(h.world_to_screen(self._press_target))) / k
                at_aim = float(np.linalg.norm(aim - self._cursor)) < 0.5
                arrived = np.max(np.abs(goal[:2] - tgt[:2])) < s.get("tol", 0.002)
                if at_aim:
                    self._settle += 1
                done = at_aim and (arrived or self._settle > 20)
                want = aim - self._cursor
            else:
                dx, dy = s["vdrag_push"]
                done = el >= s.get("seconds", 2.0)
                want = (np.array(h.world_to_screen(tgt + np.array([dx, dy, 0.0]) * 0.05))
                        - np.array(h.world_to_screen(tgt)))
            if done or el >= timeout:
                if "vdrag_to" in s and not (np.max(np.abs(goal[:2] - tgt[:2])) < s.get("tol", 0.002)):
                    self.log(f"script: vdrag did not reach {s} target={tgt.round(4).tolist()}")
                self._mouse(MouseEventType.LEFT_BUTTON_UP, *self._cursor)
                self._phase = "released"
                return
            n = float(np.linalg.norm(want))
            if n > 1e-6:
                # hand speed: at most PX_PER_FRAME pixels and at most WORLD_PER_FRAME on the plane (scaled for Shift),
                # so the operator moves alike in any view / window size
                px_per_m = 0.5 * sum(float(np.linalg.norm(np.array(h.world_to_screen(tgt + e)) -
                                                          np.array(h.world_to_screen(tgt)))) / 0.01
                                     for e in (np.array([0.01, 0, 0]), np.array([0, 0.01, 0])))
                cap = min(self.PX_PER_FRAME, self.WORLD_PER_FRAME * px_per_m / k)
                step = want * min(1.0, cap / n)
                self._cursor = self._cursor + step
                self._mouse(MouseEventType.MOVE, *self._cursor)
        else:
            self._next()
