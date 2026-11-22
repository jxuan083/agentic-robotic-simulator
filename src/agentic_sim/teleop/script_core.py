"""Backend-independent part of the scripted acceptance "operator" (pure Python).

A script is a YAML list of steps. This class runs the keyboard steps, waits, marks and checks; a backend subclass
sends the raw key events through its own input path (`_key`) and may add steps (Isaac: mouse / viewport steps).
Conditions (`until`, `check`) only read the frame state the session passes in; they never touch the robot, the target
or the block.

Steps handled here:
  - {note: text}                                   log a phase label
  - {wait: seconds}                                 host time (monotonic): UI waits, e.g. while paused
  - {settle: seconds}                               simulated time (ns["t_sim"]): let the physics act, e.g. after
                                                   closing the gripper; holds however slowly the simulation runs
  - {hold: [W, A], until: "target[0] >= 0.17", timeout: 8}   keys down until the condition holds (or timeout);
    injected keys move by simulated time (input_dt): one control step per frame, whatever the frame rate, so a
    hold with keys counts its timeout in simulated seconds too (the same motion at any speed); a hold with no
    keys ({hold: [], until: ...}, waiting on the UI or perception) counts host seconds. sim_timeout: N is always
    simulated seconds
  - {tap: P}                                       press + release on the next frame
  - {press: W} / {release: W}                      key down / up that stays held across steps
  - {focus_loss: 1.0, hold: [W]}                   hold keys, then report the window unfocused for N seconds
  - {mark: name}                                   snapshot the frame state; checks read it as m['name']['target']
  - {set: {blocks: 4, color_dr: true, ...}}        scene settings (backend hook `_set`)
  - {check: "expr", label: name}                   record a pass/fail result
The focus_loss step snapshots m['focus_lost'] on the frame the window is reported unfocused.
"""

from __future__ import annotations

import copy
import time

import numpy as np

CORE_STEPS = ("note", "wait", "settle", "press", "release", "set", "mark", "check", "tap", "focus_loss", "hold")
EVAL_BUILTINS = {"abs": abs, "min": min, "max": max, "len": len, "sorted": sorted, "zip": zip, "all": all, "any": any,
                 "round": round, "range": range, "set": set, "tuple": tuple}


def input_dt(host_dt: float, control_dt: float, scripted: bool) -> float:
    """Increment for held keys this frame. A person's keys move by host time (the speed they feel at any frame
    rate); keys injected by a script move by one control step of simulated time, so a script commands the same
    motion whatever the frame rate (at 13 fps a host-time step overshoots an `until` threshold by ~6 mm).
    Paused frames advance nothing either way (the caller does not step input while paused)."""
    return control_dt if scripted else host_dt


MARK_KEYS = ("t_sim", "target", "ee", "block", "blocks", "gripper", "jaw", "yaw_offset", "status")


def marks_summary(marks: dict) -> dict:
    """The script's marks for the result JSON: positions and gripper of each marked frame (JSON-safe)."""
    def plain(v):
        if hasattr(v, "tolist"):
            return plain(v.tolist())
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return round(v, 5) if isinstance(v, float) else v
    return {name: {k: plain(m[k]) for k in MARK_KEYS if k in m} for name, m in marks.items()}


STEP_MODIFIERS = ("until", "timeout", "sim_timeout", "label", "tol", "seconds")


def step_kind(s: dict) -> str | None:
    """The step's kind: its first key that is not a modifier (`hold` is a modifier of focus_loss)."""
    if "focus_loss" in s:
        return "focus_loss"
    return next((k for k in s if k not in STEP_MODIFIERS), None)


class ScriptCore:
    EXTRA_STEPS: tuple = ()  # step kinds a backend adds

    def __init__(self, steps: list, log):
        self.steps = list(steps)
        bad = sorted({str(step_kind(s)) for s in self.steps if step_kind(s) not in CORE_STEPS + self.EXTRA_STEPS})
        if bad:
            raise ValueError(f"{type(self).__name__}: unsupported script steps {bad} on this backend")
        self.i = -1
        self.log = log
        self.results = []
        self.focus_override = None  # None = real focus; False = simulated focus loss
        self._phase = None
        self._t0 = 0.0
        self._held = []
        self._sticky = []
        self.marks = {}
        self.done = False
        self._next()

    # ---- backend hooks --------------------------------------------------------------
    def _key(self, name: str, down: bool):
        raise NotImplementedError

    def _set(self, settings: dict):
        raise ValueError("this backend has no scene-settings step")

    def _step_extra(self, s: dict, ns: dict, el: float, timeout: float) -> bool:
        """Run a backend step; return False if the step kind is unknown."""
        return False

    def _on_next(self):
        """Per-step state a subclass resets."""

    # ---- sequencing ------------------------------------------------------------
    def _release_all(self):
        for k in self._held:
            self._key(k, False)
        self._held = []

    def _next(self):
        self._release_all()
        self.i += 1
        self._phase = None
        self._on_next()
        self._t0 = time.monotonic()
        self._t0_sim = None          # simulated time at the step's first frame (set by step())
        if self.i >= len(self.steps):
            self.done = True
            self.log("script: done")
            return
        self.log(f"script step {self.i}: {self.steps[self.i]}")

    def _cond(self, expr, ns):
        env = {"__builtins__": {}, **EVAL_BUILTINS, "m": self.marks}
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
        if self._t0_sim is None and "t_sim" in ns:
            self._t0_sim = float(ns["t_sim"])
        el_sim = None if self._t0_sim is None else float(ns["t_sim"]) - self._t0_sim
        if ("settle" in s or "sim_timeout" in s) and el_sim is None:
            raise ValueError(f"{s}: simulated-time steps need t_sim in the frame state")
        if "note" in s:
            self._next()
        elif "settle" in s:
            if el_sim >= s["settle"]:
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
        elif "set" in s:
            self._set(s["set"])
            self._next()
        elif "mark" in s:
            self.marks[s["mark"]] = self._snap(ns)
            self._next()
        elif "check" in s:
            try:
                ok, err = self._cond(s["check"], ns), None
            except Exception as e:  # e.g. a value that an earlier failed step never set: a FAIL, not a crash
                ok, err = False, f"{type(e).__name__}: {e}"
            res = {"label": s.get("label", s["check"]), "ok": ok, "expr": s["check"]}
            if err:
                res["error"] = err
            self.results.append(res)
            self.log(f"CHECK {'PASS' if ok else 'FAIL'}: {s.get('label', s['check'])}" + (f" ({err})" if err else ""))
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
            else:
                if "sim_timeout" in s:
                    out = el_sim >= s["sim_timeout"]
                elif s["hold"] and el_sim is not None:   # keys move by simulated time: so does their timeout
                    out = el_sim >= timeout
                else:
                    out = el >= timeout
                if self._cond(s["until"], ns) or out:
                    if out:
                        self.log(f"script: hold timeout {s}")
                    self._next()
        elif not self._step_extra(s, ns, el, timeout):
            raise ValueError(f"unknown script step {s}")


MOUSE_STEPS = ("wheel", "drag_to", "drag_hold", "click", "vdrag_to", "vdrag_push", "vwheel", "camera", "resize")


class MouseScript(ScriptCore):
    """Adds the mouse steps (shared by every backend that has a panel, a top view and a 3D target handle):
      - {click: close}                                 click a panel button by name (open/close/yaw_minus/yaw_plus/
                                                       pause/reset)
      - {drag_to: [x, y], tol: 0.002, timeout: 10}     left-drag on the top view from the target marker to (x, y)
      - {drag_hold: [x, y], seconds: 2}                drag to (x, y), keep holding (unreachable tests), release
      - {wheel: -1, until: "target[2] <= 0.03"}        wheel over the top view, one notch per frame
      - {vdrag_to: [x, y], tol: 0.002, timeout: 10}    press on the 3D handle, move the cursor to bring the target
                                                       to (x, y)
      - {vdrag_push: [dx, dy], seconds: 2}             press on the handle and keep pushing the cursor in world
                                                       direction (dx, dy) (e.g. into the reach boundary), release
      - {vwheel: -1, until: "target[2] <= 0.03"}       wheel over the handle, one notch per frame
      - {camera: {eye: [..], target: [..]}}            change the 3D view camera
      - {resize: [w, h]}                               resize the application window
    Each viewport drag stores m['drag'] = {press_jump, max_step} (target motion at press, largest per-frame step).
    The cursor moves like an operator watching the screen (speed-capped). A backend provides the raw events and
    the geometry: _mouse_move, _mouse_button, _mouse_scroll, _button_center, _fine_scale, _resize, and the
    attributes panel (world_to_screen(x, y) for the top view), handle (world_to_screen(p)) and set_camera.
    """
    EXTRA_STEPS = MOUSE_STEPS

    def __init__(self, steps: list, log):
        self._cursor = None
        self._drag = None
        super().__init__(steps, log)

    # ---- backend hooks --------------------------------------------------------------
    def _mouse_move(self, x, y):
        raise NotImplementedError

    def _mouse_button(self, down: bool, x, y):
        raise NotImplementedError

    def _mouse_scroll(self, dy: float):
        raise NotImplementedError

    def _button_center(self, name: str) -> tuple:
        raise NotImplementedError

    def _fine_scale(self) -> float:
        raise NotImplementedError

    def _resize(self, w: int, h: int):
        raise NotImplementedError

    def _on_next(self):
        self._drag = None

    def _step_extra(self, s, ns, el, timeout) -> bool:
        if "wheel" in s:
            if self._cond(s["until"], ns) or el >= timeout:
                self._next()
            else:
                x, y = self.panel.world_to_screen(*ns["target"][:2])
                if self._phase is None:
                    self._mouse_move(x, y)
                    self._phase = "scroll"
                else:
                    self._mouse_scroll(float(s["wheel"]))
        elif "drag_to" in s or "drag_hold" in s:
            goal = s.get("drag_to") or s.get("drag_hold")
            if self._phase is None:
                self._drag = self.panel.world_to_screen(*ns["target"][:2])
                self._mouse_move(*self._drag)
                self._phase = "move"
            elif self._phase == "move":
                self._mouse_button(True, *self._drag)
                self._phase = "drag"
                self._t0 = time.monotonic()
            elif self._phase == "drag":
                gx, gy = self.panel.world_to_screen(*goal)
                sx, sy = self._drag
                k = min(1.0, (time.monotonic() - self._t0) / 0.5)  # cursor reaches the goal over 0.5 s
                self._mouse_move(sx + (gx - sx) * k, sy + (gy - sy) * k)
                t = ns["target"]
                reached = abs(t[0] - goal[0]) < s.get("tol", 0.002) and abs(t[1] - goal[1]) < s.get("tol", 0.002)
                if "drag_hold" in s:
                    reached = el >= s.get("seconds", 2.0)
                if reached or el >= timeout:
                    if el >= timeout and "drag_to" in s:
                        self.log(f"script: drag timeout {s}")
                    self._mouse_button(False, sx + (gx - sx) * k, sy + (gy - sy) * k)
                    self._phase = "released"
            else:
                self._next()
        elif "click" in s:
            x, y = self._button_center(s["click"])
            if self._phase is None:
                self._mouse_move(x, y)
                self._phase = "move"
            elif self._phase == "move":
                self._mouse_button(True, x, y)
                self._phase = "down"
            else:
                self._mouse_button(False, x, y)
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
                    self._mouse_move(x, y)
                    self._phase = "hover"
                elif self._phase == "hover":  # one frame for the hover gesture to register
                    self._phase = "scroll"
                else:
                    self._mouse_scroll(float(s["vwheel"]))
        elif "camera" in s:
            self.set_camera(s["camera"]["eye"], s["camera"]["target"])
            self._next()
        elif "resize" in s:
            w, h = s["resize"]
            try:
                self._resize(int(w), int(h))
                self.log(f"script: resized the window to {w}x{h}")
            except Exception as e:  # noqa: BLE001
                self.log(f"script: resize failed: {e}")
            self._next()
        else:
            return False
        return True

    # ---- viewport handle drag ------------------------------------------------------
    PX_PER_FRAME = 6.0      # cursor speed cap of the scripted "operator" (pixels per frame)
    WORLD_PER_FRAME = 0.005  # and at most 5 mm of target motion per frame

    def _vdrag(self, s, ns, el, timeout):
        h = self.handle
        tgt = np.asarray(ns["target"], float)
        if self._phase is None:
            self._cursor = np.array(h.world_to_screen(tgt), float)
            self._mouse_move(*self._cursor)
            self._phase = "hover"
        elif self._phase == "hover":
            self._mouse_button(True, *self._cursor)
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
            k = self._fine_scale() if ns.get("fine") else 1.0
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
                self._mouse_button(False, *self._cursor)
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
                self._mouse_move(*self._cursor)
        else:
            self._next()
