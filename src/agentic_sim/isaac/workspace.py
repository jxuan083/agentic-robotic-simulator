"""Operator workspace parts of the Isaac window (focus mode, M key): the status column and the camera cells.

Status column: target colour, current task, gripper, recording state, UI FPS / RTF and the perception result age
(agentic_sim.workspace_status). Docked above the EE Teleop panel on the right; shown only in the workspace.

Camera cells: an overlay on each docked sensor viewport (global, wrist). Every cell has its own "原圖 / 標註"
button. "標註" shows the annotated snapshot of the image the estimate was computed from (boxes are never drawn on
a newer frame), dated with its age; a missing or stale result leaves the live raw camera with the reason. The
recorder is unaffected: it reads its own render products at the native resolution, without any of this UI.
"""

from __future__ import annotations

import time

import numpy as np
import omni.ui as ui

from agentic_sim.workspace_status import cell_view, perception_item, status_lines

STATUS_TITLE = "Operator status"
CELL_FRAME = "ars_camera_cell"


class StatusWindow:
    def __init__(self):
        self.window = ui.Window(STATUS_TITLE, width=360, height=300, visible=False,
                                flags=ui.WINDOW_FLAGS_NO_CLOSE | ui.WINDOW_FLAGS_NO_SCROLLBAR)
        with self.window.frame:
            with ui.VStack(spacing=2, style={"margin": 6}):
                self.label = ui.Label("", alignment=ui.Alignment.LEFT_TOP, word_wrap=True,
                                      style={"font_size": 17})
        self._next = 0.0

    def show(self, lines: list, now: float):
        if now < self._next:          # 5 updates per second are enough to read; label layout is not free
            return
        self._next = now + 0.2
        self.label.text = "\n".join(lines)


class CameraCells:
    """Overlays on the sensor viewports; rebuilt whenever the viewports are (new session)."""

    def __init__(self, annotate_default: bool = True):
        self.annotate_default = annotate_default
        self.annotate = {}
        self.cells = {}
        self._windows = {}

    def bind(self, windows: dict):
        """windows: key -> viewport window (agentic_sim.isaac.sensor_views)."""
        if {k: id(w) for k, w in windows.items()} == self._windows:
            return
        self._windows = {k: id(w) for k, w in windows.items()}
        self.cells = {}
        for key, win in windows.items():
            self.annotate.setdefault(key, self.annotate_default)
            frame = win.get_frame(CELL_FRAME)
            provider = ui.ByteImageProvider()
            with frame:
                with ui.ZStack():
                    image = ui.ImageWithProvider(provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                                                 visible=False)
                    with ui.VStack():
                        ui.Spacer()
                        with ui.HStack(height=26, spacing=6):   # bottom row: clear of the viewport's own menu bar
                            button = ui.Button("", width=150, clicked_fn=lambda k=key: self.toggle(k),
                                               tooltip=f"key {1 if key == 'global' else 2}")
                            with ui.ZStack():
                                ui.Rectangle(style={"background_color": 0xB0000000})
                                label = ui.Label("", style={"color": 0xFFFFFFFF, "font_size": 16})
            self.cells[key] = {"provider": provider, "image": image, "button": button, "label": label,
                               "shown": None, "text": None}

    def toggle(self, key: str):
        self.annotate[key] = not self.annotate.get(key, self.annotate_default)

    def update(self, perception, now: float):
        for key, c in self.cells.items():
            if perception is None:
                mode, text = "raw", "raw (perception off)"
            else:
                st = perception.state()
                st = st["cameras"].get(key) if "cameras" in st else (st if st.get("camera") == key else None)
                mode, text = cell_view(self.annotate.get(key, True), perception_item(st, now))
                p = perception.cameras().get(key)
                if p is not None:
                    p.display = self.annotate.get(key, True)
            want = f"{key}: annotated" if self.annotate.get(key, True) else f"{key}: raw"
            if c["button"].text != want:
                c["button"].text = want
            if c["text"] != text:
                c["label"].text = " " + text
                c["text"] = text
            if mode == "annotated":
                snap = perception.snapshot(key)
                if snap is not None and c["shown"] is not snap[0]:   # upload each snapshot once
                    rgb = snap[0]
                    rgba = np.concatenate([rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)], axis=2)
                    c["provider"].set_data_array(rgba.reshape(-1), [rgb.shape[1], rgb.shape[0]])
                    c["shown"] = snap[0]
            c["image"].visible = mode == "annotated"

    def state(self) -> dict:
        return {k: {"annotate": self.annotate.get(k), "text": c["text"], "image_visible": bool(c["image"].visible)}
                for k, c in self.cells.items()}


def status_info(session, now: float) -> dict:
    """What the status column shows, from a TeleopSession."""
    s = session.ctrl.state
    p = session.ctrl.p
    mid = 0.5 * (p.gripper_open + p.gripper_closed)
    rec = None
    if session.recorder is not None:
        r = session.recorder
        if r.w is None:
            rec = {"state": "saved", "episode": r.episodes[-1]["path"] if r.episodes else None}
        else:
            rec = {"state": "waiting_cameras" if r.step0 is None else "recording", "frames": r.w.n,
                   "episode": r.w.root}
    perc = None
    if session.live_perception is not None:
        st = session.live_perception.state()
        cams = st["cameras"] if "cameras" in st else {st["camera"]: st}
        perc = {k: perception_item(v, now) for k, v in cams.items()}
    task = None
    if session.task is not None:
        lines = session.task.ex.panel_lines() or []
        task = lines[0] if lines else "任務執行器"
    return {"clocks": session.clocks, "target_color": session.cfg.target_color or (
                session.block_color_names[0] if session.recorder is not None and session.block_color_names else None),
            "task": task, "jaw": session.last_frame.get("jaw"), "gripper_cmd": s.gripper,
            "gripper_open": s.gripper > mid, "recording": rec, "paused": session.paused, "perception": perc,
            "screen_video": session.screen_video.text(),
            "perception_stage": getattr(session.live_perception, "stage", "estimate")}


def lines_for(session) -> list:
    return status_lines(status_info(session, time.monotonic()))
