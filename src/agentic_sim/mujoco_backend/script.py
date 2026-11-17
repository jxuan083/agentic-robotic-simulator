"""Scripted acceptance operator for the MuJoCo backend: the shared script steps (agentic_sim.teleop.script_core),
sent through the same input path as a person.

Keys go into KeyState.on_key and mouse events into Viewer.inject_move / inject_button / inject_scroll, the functions
the GLFW callbacks call, so the session sees them exactly like real input (only the OS -> GLFW hop is skipped).
Mouse steps need the window; a headless run accepts keyboard-only scripts.
"""

from __future__ import annotations

from agentic_sim.teleop.keys import KeyState
from agentic_sim.teleop.script_core import MOUSE_STEPS, MouseScript, step_kind


class MjScript(MouseScript):
    def __init__(self, steps: list, keys: KeyState, log):
        self.keys = keys
        self.session = self.viewer = self.panel = self.handle = None
        self.set_camera = None
        super().__init__(steps, log)

    def rebind(self, session, viewer):
        """Point the script at this session (first one, or rebuilt for another block count) and the window."""
        uses_mouse = any(step_kind(s) in MOUSE_STEPS for s in self.steps)
        if uses_mouse and viewer is None:
            raise ValueError("this script has mouse steps: it needs the MuJoCo window (not --headless)")
        if self.session is not None:
            self.log(f"script: continues in the rebuilt scene at step {self.i}")
        self.session, self.viewer = session, viewer
        self.panel, self.handle = session.panel, session.handle
        self.set_camera = viewer.set_camera if viewer is not None else None

    def step(self, ns: dict):
        # a just-built session (start, or rebuilt for another block count) has no panel layout until its first
        # frame is drawn: wait for it, as an operator waits for the window to show
        if self.viewer is not None and not self.panel.buttons:
            return
        super().step(ns)

    def _key(self, name: str, down: bool):
        self.keys.on_key(name, down)

    def _set(self, settings: dict):
        self.panel.set_settings(settings)

    def _mouse_move(self, x, y):
        self.viewer.inject_move(x, y)

    def _mouse_button(self, down: bool, x, y):
        self.viewer.inject_move(x, y)
        self.viewer.inject_button("left", down)

    def _mouse_scroll(self, dy: float):
        self.viewer.inject_scroll(dy)

    def _button_center(self, name: str):
        x, y, w, h = self.panel.buttons[name]
        return x + w / 2, y + h / 2

    def _fine_scale(self) -> float:
        return float(self.session.ctrl.p.fine_drag_scale)

    def _resize(self, w: int, h: int):
        self.viewer.resize(w, h)
