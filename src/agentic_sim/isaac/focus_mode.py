"""Focus (operation) layout, the operator workspace: true fullscreen, the 3D viewport in the middle for EE drags,
the status column (agentic_sim.isaac.workspace) above the EE Teleop panel on the right, global / wrist side by side
below (agentic_sim.isaac.sensor_views), each with its raw / annotated switch.

Toggle with the M key or the panel's "Focus view" button. Entering makes the app window fullscreen, hides the main
menu bar, the status bar, the editor windows (Stage, Layer, Console, ...), the dock tab bars (window titles), the
viewport's own menu bar and axis, and the floating perception previews; leaving restores exactly what was there.
A rebuilt scene (new session, new panel and viewports) is brought into the same layout over the next frames.
Nothing in the scene or the teleop state changes.
"""

from __future__ import annotations

import omni.ui as ui

EDITOR_WINDOWS = ("Stage", "Layer", "Render Settings", "Property", "Semantics Schema Editor", "Content", "Console",
                  "Main ToolBar", "Status Bar")
PANEL_RATIO = 0.28  # share of the viewport area given to the docked panel
STATUS_RATIO = 0.42  # share of the right column given to the status window (above the panel)
VIEWPORT_LAYERS = (("Menubar", "menubar"), ("Axis", "guide"))


class FocusMode:
    def __init__(self, panel_window: ui.Window, log=print):
        self._panel = panel_window
        self._log = log
        self.active = False
        self._saved_visible: dict[str, bool] = {}
        self._saved_panel = None
        self._pending_front = []
        self._pending_frames = 0
        self._dock_in_frames = -1
        self._saved_app = None
        self._status = None
        self._cells = None
        self.extra_windows = lambda: []            # session hook: windows to keep tab-less (sensor views)
        self.perception_windows = lambda v: None   # session hook: show / hide the floating perception previews
        self._enforce_in = -1
        self._focus_viewport_in = -1
        self._dock_status_in = -1
        self._next_enforce = 0.0

    # ---- operator workspace parts (created on first use; persist across sessions) -------------------------------
    @property
    def status(self):
        if self._status is None:
            from agentic_sim.isaac.workspace import StatusWindow
            self._status = StatusWindow()
        return self._status

    def cells(self, annotate_default=True):
        if self._cells is None:
            from agentic_sim.isaac.workspace import CameraCells
            self._cells = CameraCells(annotate_default)
        return self._cells

    def frame(self, lines, perception, windows, now):
        """Per frame from the session: status text, camera cells, and the layout of a rebuilt scene."""
        cells = self.cells()
        cells.bind(windows)
        cells.update(perception, now)
        if self.active:
            if self._dock_status_in >= 0:
                self._dock_status_in -= 1
                if self._dock_status_in < 0:
                    try:
                        self.status.window.dock_in(self._panel, ui.DockPosition.TOP, STATUS_RATIO)
                    except Exception as e:  # noqa: BLE001
                        self._log(f"focus mode: could not dock the status column ({e})")
            elif (not getattr(self.status.window, "docked", True) and self._panel is not None
                  and self._dock_in_frames < 0 and self.status.window.visible):
                self._dock_status_in = 2      # not docked (yet): try again shortly, never float over the 3D view
            self.status.show(lines, now)
            if perception is not None:
                self.perception_windows(False)
            if self._enforce_in >= 0:
                self._enforce_in -= 1
                if self._enforce_in < 0:
                    self._tab_bars(False)
            elif now >= self._next_enforce:   # windows created later (rebuilt scene) get the same treatment
                self._next_enforce = now + 0.5
                self._tab_bars(False)

    def _tab_bars(self, enabled: bool):
        wins = [ui.Workspace.get_window("Viewport"), self._panel, self.status.window] + list(self.extra_windows())
        for w in wins:
            if w is None:
                continue
            try:
                w.dock_tab_bar_enabled = enabled
            except Exception:  # noqa: BLE001
                pass
        for vp in [ui.Workspace.get_window("Viewport")] + list(self.extra_windows()):
            for name, cat in VIEWPORT_LAYERS:   # the viewports' own menu bar and axis guide
                try:
                    layer = vp._find_viewport_layer(name, cat)   # noqa: SLF001  (the call Kit's hide-UI toggle makes)
                    if layer is not None:
                        layer.visible = enabled
                except Exception:  # noqa: BLE001
                    pass

    def _app(self, operator: bool):
        """Fullscreen app window without the main menu bar and status bar (operator), or what was there before."""
        import omni.appwindow
        app = omni.appwindow.get_default_app_window()
        try:
            import omni.kit.mainwindow
            mw = omni.kit.mainwindow.get_main_window()
        except Exception:  # noqa: BLE001
            mw = None
        bars = [] if mw is None else [mw.get_main_menu_bar(), mw.get_status_bar_frame()]
        if operator:
            self._saved_app = (bool(app.is_fullscreen()), [bool(b.visible) for b in bars])
            for b in bars:
                b.visible = False
            app.set_fullscreen(True)
        elif self._saved_app is not None:
            full, vis = self._saved_app
            app.set_fullscreen(full)
            for b, v in zip(bars, vis):
                b.visible = v
            self._saved_app = None

    def set_panel(self, panel_window):
        """Point the layout at another panel window (persistent server: a new session brings a new panel).
        While the layout is active the new panel is docked a few frames later (a brand-new window cannot be
        docked before it has been laid out)."""
        if panel_window is None and self._status is not None and self.active:
            # the panel goes away with its session: take the status column out of its dock node first, or it fills
            # the node and the next session's panel docks behind it
            try:
                self.status.window.undock()
            except Exception:  # noqa: BLE001
                pass
            self.status.window.visible = False
            self._dock_status_in = -1
        self._panel = panel_window
        self._dock_in_frames = 10 if (panel_window is not None and self.active) else -1

    def _dock_panel(self):
        try:
            self._panel.dock_in(ui.Workspace.get_window("Viewport"), ui.DockPosition.RIGHT, PANEL_RATIO)
            self.status.window.visible = True
            self._dock_status_in = 3          # into the panel once the panel's own docking is laid out
        except Exception as e:  # noqa: BLE001
            self._log(f"focus mode: could not dock the panel ({e}); it stays floating")

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
        """Call once per frame (finishes the deferred parts: front tabs after leaving, docking a new panel)."""
        if self._dock_in_frames >= 0:
            self._dock_in_frames -= 1
            if self._dock_in_frames < 0 and self._panel is not None and self.active:
                self._dock_panel()
                self._enforce_in = 3          # the rebuilt scene's windows: tab-less like the rest
                self._focus_viewport_in = 4
                self._log("focus mode: new panel docked")
        if self._focus_viewport_in >= 0:
            self._focus_viewport_in -= 1
            if self._focus_viewport_in < 0:
                vp = ui.Workspace.get_window("Viewport")
                if vp is not None:
                    try:
                        vp.focus()
                    except Exception:  # noqa: BLE001
                        pass
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
            w = ui.Workspace.get_window(title)
            if w is not None and w.visible:         # no show / hide callback registered (e.g. "Status Bar")
                w.visible = False
        p = self._panel
        self._saved_panel = (p.position_x, p.position_y, p.width, p.height)
        self._app(True)
        self._dock_panel()
        self.perception_windows(False)
        self._enforce_in = 3                 # dock tab bars go after the new docking is laid out
        self._focus_viewport_in = 6          # keyboard focus back on the viewport once fullscreen is laid out
        self.active = True
        self._log(f"focus mode: on (saved editor visibility {self._saved_visible})")

    def exit(self):
        if not self.active:
            return
        self._tab_bars(True)
        self._enforce_in = -1
        try:
            self.status.window.undock()
        except Exception:  # noqa: BLE001
            pass
        self.status.window.visible = False
        self._app(False)
        self.perception_windows(True)
        for title, vis in self._saved_visible.items():
            if vis:
                ui.Workspace.show_window(title, True)
                w = ui.Workspace.get_window(title)
                if w is not None and not w.visible:
                    w.visible = True
        # re-shown windows are recreated and docked over the next frames: bring the old front tabs forward then
        self._pending_front = list(self._saved_front)
        self._pending_frames = 3
        p = self._panel
        self._dock_in_frames = -1
        if p is not None:  # None: the panel was already destroyed (session closed)
            try:
                p.undock()
            except Exception as e:  # noqa: BLE001
                self._log(f"focus mode: could not undock the panel ({e})")
            if self._saved_panel is not None:
                p.position_x, p.position_y, p.width, p.height = self._saved_panel
        self.active = False
        self._log(f"focus mode: off, after {self._describe()}")
