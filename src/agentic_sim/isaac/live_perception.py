"""Nonblocking viewport capture and a read-only eight-corner preview."""
from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import queue
import time

import numpy as np
import omni.ui as ui
from omni.kit.widget.viewport.capture import ByteCapture

from agentic_sim.config import v
from agentic_sim.scene_layout import block_support_top
from agentic_sim.perception.live import LiveEstimator, rendered_camera
from agentic_sim.perception.vertices import CameraGeometry


class _Capture(ByteCapture):
    def __init__(self, inbox, generation):
        super().__init__()
        self.inbox = inbox
        self.generation = generation

    def capture(self, aov_map, frame_info, hydra_texture, result_handle):
        self.rendered_at = time.monotonic()
        return super().capture(aov_map, frame_info, hydra_texture, result_handle)

    def on_capture_completed(self, buffer, buffer_size, width, height, byte_format):
        # Renderer-owned memory is valid only within this callback. Copy before
        # returning; all USD/UI work and worker submission happen on the main tick.
        try:
            if int(buffer_size) != int(width)*int(height)*4:
                raise ValueError(f'expected RGBA8 capture, got {byte_format}, {buffer_size} bytes')
            ptr = ctypes.pythonapi.PyCapsule_GetPointer
            ptr.restype = ctypes.c_void_p
            ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
            address = ptr(buffer, None)
            data = np.ctypeslib.as_array((ctypes.c_uint8 * int(buffer_size)).from_address(address))
            rgb = data.reshape(int(height), int(width), 4)[..., :3].copy()
            self.inbox.put((self.generation, rgb, np.array(self.view), np.array(self.projection),
                            self.frame_number, self.rendered_at, None))
        except Exception as exc:
            self.inbox.put((self.generation, None, None, None, None, time.monotonic(), str(exc)))


class LivePerception:
    def __init__(self, scene, views, camera='wrist', hz=2.0, log=print, json_path=None, stage='estimate',
                 delay_s=0.0):
        self.stage = stage
        if camera == 'both':
            self.children = {name: LivePerception(scene, views, name, hz, log, stage=stage, delay_s=delay_s)
                             for name in ('global', 'wrist')}
            self.json_path = None if json_path is None else Path(json_path)
            self._write_json()
            return
        self.children = None
        self.json_path = None if json_path is None else Path(json_path)
        if not math.isfinite(hz) or not 0 < hz <= 30:
            raise ValueError('perception_hz must be finite and in (0, 30]')
        if camera not in views.bound or camera not in views.windows:
            raise ValueError(f'live perception needs the {camera} sensor camera')
        self.api = views.windows[camera].viewport_api
        self.base = CameraGeometry.from_scene(scene, camera, [0.]*5 if camera == 'wrist' else None)
        self.estimator = LiveEstimator(float(v(scene['block']['edge'])),
                                       block_support_top(scene))
        self.work = self.estimator.worker(stage, delay_s)
        self.snapshot_rgb = None          # the annotated image of `latest` (boxes drawn on the image they came from)
        self.display = False              # someone shows the annotated image (workspace cell or this window)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='block-perception')
        self.inbox = queue.SimpleQueue()
        self.period = 1/hz
        self.next_at = 0.
        self.pending_at = None
        self.future = None
        self.closed = False
        self.generation = 0
        self.latest = None
        self.error = None
        self._capture = None
        self.provider = ui.ByteImageProvider()
        self.window = ui.Window(f'Block perception — {camera}', width=720, height=700,
                                position_x=24 if camera == 'global' else 760, position_y=80)
        with self.window.frame:
            with ui.VStack(spacing=4):
                self.status = ui.Label('Waiting for camera...', height=24)
                ui.ImageWithProvider(self.provider, height=360, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                with ui.ScrollingFrame():
                    self.coordinates = ui.Label('Mount XYZ in metres; corners inferred; IDs belong to this frame.',
                                                alignment=ui.Alignment.LEFT_TOP, word_wrap=True)
        log(f'live perception: {camera}, up to {hz:g} Hz; upright cubes on the scene support '
            f'(z {self.estimator.floor * 1000:.1f} mm)')
        # A newly constructed session supersedes any previous scene in the same
        # output file, even before its first viewport capture completes.
        self._write_json()

    def invalidate(self):
        """Reset: discard results and callbacks for the previous scene state."""
        if self.children is not None:
            for child in self.children.values():
                child.invalidate()
            self._write_json()
            return
        self.generation += 1
        self.latest = None
        self.snapshot_rgb = None
        self.pending_at = None
        self.error = None
        self.coordinates.text = 'Reset: waiting for a new image.'
        self.provider.set_data_array(np.array([0, 0, 0, 255], dtype=np.uint8), [1, 1])
        self._write_json()

    def update(self):
        if self.children is not None:
            for child in self.children.values():
                child.update()
            self._write_json()
            return
        if self.closed:
            return
        now = time.monotonic()
        if self.future is not None and self.future.done():
            generation, future = self.future_generation, self.future
            self.future = None
            try:
                rgb, record = future.result()
                if generation == self.generation:
                    record['completed_at_monotonic_s'] = now
                    self.latest = record
                    self.snapshot_rgb = rgb
                    self.error = None
                    self._write_json()
                    if self.window.visible:       # the preview window costs an upload; skipped while hidden
                        rgba = np.concatenate([rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)], axis=2)
                        self.provider.set_data_array(rgba.reshape(-1), [rgb.shape[1], rgb.shape[0]])
                    lines = ['Mount XYZ (m). Corners inferred; IDs are per frame.']
                    for d in record['detections']:
                        g = d['geometry']
                        prefix = f"#{d['detection_id']} {d['color']}"
                        if g['status'] != 'estimated':
                            lines.append(f"{prefix}: rejected ({g.get('reason')})")
                            continue
                        x, y, z = g['center_xyz_m']
                        lines.append(f"{prefix}: center ({x:.4f}, {y:.4f}, {z:.4f}), IoU {g['fit_iou']:.3f}")
                        for vertex in g['vertices']:
                            x, y, z = vertex['xyz_m']
                            lines.append(f"  v{vertex['id']}: ({x:.4f}, {y:.4f}, {z:.4f})")
                    self.coordinates.text = '\n'.join(lines) if record['detections'] else 'No block candidates in this frame.'
            except Exception as exc:
                if generation == self.generation:
                    self.error = str(exc)
        while not self.inbox.empty():
            generation, rgb, view, projection, frame, captured_at, error = self.inbox.get()
            if generation != self.generation:
                continue
            self.pending_at = None
            if error:
                self.error = error
                continue
            try:
                if self.future is None:
                    cam = rendered_camera(self.base, view, projection, rgb.shape[1], rgb.shape[0])
                    self.future_generation = generation
                    self.future = self.pool.submit(self.work, rgb, cam, frame, captured_at)
            except Exception as exc:
                self.error = str(exc)
        if self.pending_at is not None and now-self.pending_at > 5:
            # Do not pile up capture requests if the renderer has stopped.
            self.error = 'Camera capture delayed; waiting for renderer'
        if self.future is None and self.pending_at is None and now >= self.next_at:
            self.next_at = now+self.period
            self.pending_at = now
            try:
                self._capture = _Capture(self.inbox, self.generation)
                self.api.schedule_capture(self._capture)
            except Exception as exc:
                self.pending_at = None
                self.error = str(exc)
        age = now-self.latest['captured_at_monotonic_s'] if self.latest else None
        text = 'Waiting for first estimate' if age is None else f'Snapshot age {age:.1f}s' + (' [STALE]' if age > 2 else '')
        if self.future is not None:
            text += ' | processing'
        if self.error:
            text += f' | ERROR: {self.error}'
        self.status.text = text

    def cameras(self) -> dict:
        """camera key -> the LivePerception of that camera."""
        return dict(self.children) if self.children is not None else {self.base.name: self}

    def set_windows_visible(self, visible: bool):
        """The floating preview windows (hidden in the operator workspace: the camera cells show the snapshots)."""
        for p in self.cameras().values():
            p.window.visible = bool(visible)

    def snapshot(self, key):
        """(annotated RGB, latest record) of a camera, or None before its first result."""
        p = self.cameras().get(key)
        if p is None or p.latest is None or p.snapshot_rgb is None:
            return None
        return p.snapshot_rgb, p.latest

    def state(self):
        if self.children is not None:
            cameras = {name: child.state() for name, child in self.children.items()}
            return {"enabled": True, "camera": "both", "cameras": cameras,
                    "stale": any(item["stale"] for item in cameras.values())}
        age = None if self.latest is None else time.monotonic()-self.latest["captured_at_monotonic_s"]
        return {"enabled": True, "camera": self.base.name, "error": self.error,
                "age_s": age, "stale": age is None or age > 2 or self.error is not None,
                "processing": self.future is not None, "latest": self.latest}

    def _write_json(self):
        if self.json_path is None:
            return
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.json_path.with_suffix(self.json_path.suffix + '.tmp')
        temp.write_text(json.dumps(self.state(), indent=2), encoding='utf-8')
        temp.replace(self.json_path)

    def close(self):
        if self.children is not None:
            for child in self.children.values():
                child.close()
            return
        self.closed = True
        self.generation += 1
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.window.visible = False
        self.window.destroy()
        self._capture = None
