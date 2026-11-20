"""Floating viewport windows that show the scene's sensor cameras (global D455, wrist camera) in the Isaac app.

One extra viewport per camera, bound to the camera prim (agentic_sim.isaac.scene). The windows take no teleop input;
they are destroyed with the session (the next session's stage brings new camera prims).
"""

from __future__ import annotations

LABELS = {"global": "Camera: D455 global", "wrist": "Camera: wrist"}
HEIGHT = {"global": 216, "wrist": 216}   # image height in the window (pixels); width follows the camera's aspect
TITLE_BAR = 52                           # window title + viewport menu bar


class SensorViews:
    def __init__(self, cameras: dict, log=print, x0: int = 8, y0: int = 64):
        """cameras: key -> (prim path, width, height). Windows stack down the left edge of the app window."""
        self.windows = []
        if not cameras:
            return
        from omni.kit.viewport.utility import create_viewport_window
        y = y0
        for key in ("global", "wrist"):
            if key not in cameras:
                continue
            path, w, h = cameras[key]
            ih = HEIGHT[key]
            iw = int(round(ih * w / h))
            try:
                win = create_viewport_window(LABELS[key], width=iw, height=ih + TITLE_BAR, position_x=x0,
                                             position_y=y, camera_path=path)
                api = win.viewport_api
                api.resolution = (int(w), int(h))   # render at the sensor resolution, shown scaled down
                self.windows.append(win)
                log(f"sensor view {key}: {path} {w}x{h}")
            except Exception as e:  # noqa: BLE001  (the teleop session must not fail on an extra view)
                log(f"sensor view {key} not created: {e}")
            y += ih + TITLE_BAR + 8

    def destroy(self):
        for win in self.windows:
            try:
                win.visible = False
                win.destroy()
            except Exception:  # noqa: BLE001
                pass
        self.windows = []
