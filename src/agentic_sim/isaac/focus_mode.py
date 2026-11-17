"""Focus (operation) layout: the 3D viewport takes the main area, the compact EE Teleop panel stays on the right.

Toggle with the M key or the panel's "Focus view" button. Entering hides the editor windows (Stage, Property,
Content, ...) and docks the panel to the right of the viewport; leaving restores exactly the windows that were
visible before and puts the panel back where it was. Nothing in the scene or the teleop state changes.
"""

from __future__ import annotations

import omni.ui as ui

EDITOR_WINDOWS = ("Stage", "Layer", "Render Settings", "Property", "Semantics Schema Editor", "Content", "Console")
PANEL_RATIO = 0.28  # share of the viewport area given to the docked panel


class FocusMode:
    def __init__(self, panel_window: ui.Window, log=print):
        self._panel = panel_window
        self._log = log
        self.active = False
        self._saved_visible: dict[str, bool] = {}
        self._saved_panel = None
        self._pending_front = []
        self._pending_frames = 0

    @staticmethod
    def _front(w) -> bool:
        try:
            return bool(w.is_selected_in_dock())
        except Exception:  # noqa: BLE001
            return False

    def _describe(self):
        out = {}
        for title in EDITOR_WINDOWS:
            w = ui.Workspace.get_window(title)
            if w is not None:
                out[title] = (int(bool(w.visible)), int(bool(getattr(w, "docked", False))), int(self._front(w)))
        return out  # title: (visible, docked, front tab)

    def front_tabs(self) -> list[str]:
        return [t for t in EDITOR_WINDOWS if ui.Workspace.get_window(t) is not None
                and ui.Workspace.get_window(t).visible and self._front(ui.Workspace.get_window(t))]

    def editor_windows_visible(self) -> list[str]:
        out = []
        for title in EDITOR_WINDOWS:
            w = ui.Workspace.get_window(title)
            if w is not None and w.visible:
                out.append(title)
        return out

    def update(self):
        """Call once per frame (finishes the deferred part of leaving focus mode)."""
        if not self._pending_front:
            return
        self._pending_frames -= 1
        if self._pending_frames > 0:
            return
        for title in self._pending_front + ["Viewport"]:  # Viewport last: keyboard focus back on the scene
            w = ui.Workspace.get_window(title)
            if w is not None:
                try:
                    w.focus()
                except Exception:  # noqa: BLE001
                    pass
        self._pending_front = []
        self._log(f"focus mode: front tabs restored, now {self._describe()}")

    def toggle(self):
        if self.active:
            self.exit()
        else:
            self.enter()

    def enter(self):
        if self.active:
            return
        # ui.Workspace.restore_workspace() is not used: it rebuilds the Viewport and breaks its rendering.
        self._saved_visible = {}
        self._saved_front = []
        for title in EDITOR_WINDOWS:
            w = ui.Workspace.get_window(title)
            if w is not None:
                self._saved_visible[title] = bool(w.visible)
                if self._front(w):
                    self._saved_front.append(title)
        self._log(f"focus mode: before {self._describe()}")
        for title in self._saved_visible:
            ui.Workspace.show_window(title, False)  # through the extension's own show / hide callback
        p = self._panel
        self._saved_panel = (p.position_x, p.position_y, p.width, p.height)
        try:
            p.dock_in(ui.Workspace.get_window("Viewport"), ui.DockPosition.RIGHT, PANEL_RATIO)
        except Exception as e:  # noqa: BLE001
            self._log(f"focus mode: could not dock the panel ({e}); it stays floating")
        self.active = True
        self._log(f"focus mode: on (saved editor visibility {self._saved_visible})")

    def exit(self):
        if not self.active:
            return
        for title, vis in self._saved_visible.items():
            if vis:
                ui.Workspace.show_window(title, True)
        # re-shown windows are recreated and docked over the next frames: bring the old front tabs forward then
        self._pending_front = list(self._saved_front)
        self._pending_frames = 3
        p = self._panel
        try:
            p.undock()
        except Exception as e:  # noqa: BLE001
            self._log(f"focus mode: could not undock the panel ({e})")
        if self._saved_panel is not None:
            p.position_x, p.position_y, p.width, p.height = self._saved_panel
        self.active = False
        self._log(f"focus mode: off, after {self._describe()}")
