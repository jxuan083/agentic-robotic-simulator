"""Sensor camera viewports for the current Isaac stage.

Destroy these windows during unbind: a viewport render product can remain attached to a cleared old stage after
SimulationContext rebuild. The next session creates new viewports for its camera prims. Wrist lens distortion is
rendered by the camera prim in real-lens mode.
"""

from __future__ import annotations

import omni.ui as ui

LABELS = {"global": "Camera: D455 global", "wrist": "Camera: wrist"}
ORDER = ("global", "wrist")
BOTTOM_RATIO = 0.34     # share of the viewport area given to the camera strip
DOCK_AFTER_FRAMES = 10  # a brand-new window cannot be docked before it has been laid out

_instance = None


def sensor_views(log=print) -> "SensorViews":
    """The app's resident views (created on first use)."""
    global _instance
    if _instance is None:
        _instance = SensorViews(log)
    return _instance


class SensorViews:
    def __init__(self, log=print):
        self.log = log
        self.windows = {}          # key -> viewport window
        self._dock_in = -1
        self._docked = False
        self.bound = {}

    def bind(self, cameras: dict):
        """cameras: key -> (prim path, width, height) of this session's sensor cameras."""
        self.bound = dict(cameras)
        if not cameras:
            for w in self.windows.values():
                w.visible = False
            return
        from omni.kit.viewport.utility import create_viewport_window
        for key in ORDER:
            if key not in cameras:
                if key in self.windows:
                    self.windows[key].visible = False
                continue
            path, w, h = cameras[key]
            win = self.windows.get(key)
            try:
                if win is None:
                    win = create_viewport_window(LABELS[key], width=int(w * 0.3), height=int(h * 0.3) + 52,
                                                 camera_path=path)
                    win.flags = win.flags | ui.WINDOW_FLAGS_NO_CLOSE
                    self.windows[key] = win
                    self._dock_in = DOCK_AFTER_FRAMES
                else:
                    win.viewport_api.camera_path = path
                    win.visible = True
                # The widget otherwise overwrites API resolution when docked or
                # resized. Sensor intrinsics refer to native pixels, not UI size.
                win.viewport_widget.fill_frame = False
                win.viewport_widget.expand_viewport = False
                win.viewport_api.fill_frame = False
                win.viewport_api.resolution_scale = 1.0
                win.viewport_widget.set_resolution((int(w), int(h)))
                self.log(f"sensor view {key}: {path} {w}x{h}")
            except Exception as e:  # noqa: BLE001  (the teleop session must not fail on an extra view)
                self.log(f"sensor view {key} not available: {e}")

    def update(self):
        """Per frame: dock newly created windows once they are laid out."""
        if self._dock_in < 0:
            return
        self._dock_in -= 1
        if self._dock_in > 0:
            return
        self._dock_in = -1
        vp = ui.Workspace.get_window("Viewport")
        g, w = self.windows.get("global"), self.windows.get("wrist")
        try:
            if g is not None and vp is not None:
                g.dock_in(vp, ui.DockPosition.BOTTOM, BOTTOM_RATIO)
                if w is not None:   # wrist to the right of global; widths follow the 16:9 / 4:3 aspect ratios
                    w.dock_in(g, ui.DockPosition.RIGHT, 0.43)
            elif w is not None and vp is not None:
                w.dock_in(vp, ui.DockPosition.BOTTOM, BOTTOM_RATIO)
            self._docked = True
        except Exception as e:  # noqa: BLE001
            self.log(f"sensor views: could not dock ({e}); they stay floating")

    def unbind(self):
        """Destroy viewports before this session's stage is cleared/replaced."""
        self.bound = {}
        for window in self.windows.values():
            try:
                window.visible = False
                window.destroy()
            except Exception as exc:  # noqa: BLE001
                self.log(f"sensor view teardown warning: {exc}")
        self.windows.clear()
        self._dock_in = -1
        self._docked = False
