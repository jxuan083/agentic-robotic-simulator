"""Asynchronous native-resolution image-only global/wrist pose estimates for MuJoCo."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path

import numpy as np

from agentic_sim.perception.live import LiveEstimator
from agentic_sim.perception.vertices import CameraGeometry
from agentic_sim.config import v
from agentic_sim.scene_layout import block_support_top, sensor_cameras


def camera_selection(camera):
    if camera not in ("both", "global", "wrist"):
        raise ValueError("perception camera must be both, global or wrist")
    return ("global", "wrist") if camera == "both" else (camera,)


class MjLivePerception:
    def __init__(self, scene, model, hz=2.0, log=print, camera="both", json_path=None):
        if not np.isfinite(hz) or not 0 < hz <= 30:
            raise ValueError("perception_hz must be finite and in (0, 30]")
        self.scene_cfg, self.model = scene, model
        self.specs = sensor_cameras(scene)
        if set(self.specs) != {"global", "wrist"}:
            raise ValueError("MuJoCo live perception requires both configured sensor cameras")
        self.selected = camera_selection(camera)
        self.json_path = None if json_path is None else Path(json_path)
        self.cam_ids = {key: int(model.camera(name).id) for key, name in
                        (("global", "global_cam"), ("wrist", "wrist_cam"))}
        from agentic_sim.wrist_lens import WristLens
        min_w = max(int(spec["width"]) for spec in self.specs.values())
        min_h = max(int(spec["height"]) for spec in self.specs.values())
        wrist_lens = WristLens.from_scene(scene)
        if wrist_lens is not None:
            rw, rh, _ = wrist_lens.render_size()
            min_w, min_h = max(min_w, rw), max(min_h, rh)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), min_w)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), min_h)
        self.edge = float(v(scene["block"]["edge"]))
        self.floor = block_support_top(scene)
        self.hz, self.period = hz, 1.0 / hz
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mujoco-perception")
        self.estimators = {key: LiveEstimator(self.edge, self.floor) for key in self.specs}
        self.renderers = {}
        self.latest, self.futures, self.last_capture = {}, {}, {}
        self.generation = {key: 0 for key in self.specs}
        self.future_generation = {}
        self.frame_numbers = {key: 0 for key in self.specs}
        self.maps = {}
        self.errors = {}
        self.log = log
        self.closed = False
        log(f"MuJoCo image-only perception enabled for global+wrist at up to {hz:g} Hz")

    def _geometry(self, key, data):
        spec = self.specs[key]
        cid = self.cam_ids[key]
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(data.cam_xmat[cid]).reshape(3, 3) @ np.diag([1., -1., -1.])
        pose[:3, 3] = data.cam_xpos[cid]
        lens = spec.get("lens") if key == "wrist" else None
        if lens is not None:
            k = np.array([[lens.fx, 0, lens.cx], [0, lens.fy, lens.cy], [0, 0, 1.]])
            dist = np.asarray(lens.dist, float)
        else:
            # MuJoCo renders a centered pinhole from vertical fovy. Scene calibration's
            # measured off-center K is not represented by this renderer, so use its
            # actual centered principal point and focal length instead.
            fovy = float(spec.get("render_fovy_deg", spec["fovy_deg"]))
            focal = spec["height"] / (2 * np.tan(np.deg2rad(fovy) / 2))
            k = np.array([[focal, 0, spec["width"] / 2],
                          [0, focal, spec["height"] / 2], [0, 0, 1.]])
            dist = np.zeros(5)
        return CameraGeometry(spec["width"], spec["height"], k, dist, pose, key)

    def _capture(self, key, data, now):
        import mujoco
        spec = self.specs[key]
        lens = spec.get("lens") if key == "wrist" else None
        if lens is not None:
            rw, rh, _ = lens.render_size()
            render_key = (key, rw, rh)
        else:
            rw, rh = spec["width"], spec["height"]
            render_key = (key, rw, rh)
        renderer = self.renderers.get(render_key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=rh, width=rw)
            self.renderers[render_key] = renderer
        renderer.update_scene(data, camera=self.cam_ids[key])
        rgb = renderer.render()
        if lens is not None:
            maps = self.maps.get(key)
            if maps is None:
                maps = lens.maps(rw, rh, spec["render_fovy_deg"])
                self.maps[key] = maps
            rgb = lens.apply(rgb, maps)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        geometry = self._geometry(key, data)
        number = self.frame_numbers[key]
        self.frame_numbers[key] += 1
        return self.pool.submit(self.estimators[key].predict, rgb, geometry, number, now), rgb

    def update(self, data):
        if self.closed:
            return
        now = time.monotonic()
        for key, future in list(self.futures.items()):
            if future.done():
                generation = self.future_generation.pop(key, None)
                try:
                    overlay, record = future.result()
                    record["completed_at_monotonic_s"] = now
                    if generation == self.generation[key]:
                        self.latest[key] = {"record": record, "overlay_rgb": overlay}
                        self.errors.pop(key, None)
                        self._write_json()
                except Exception as exc:
                    if generation == self.generation[key]:
                        self.errors[key] = str(exc)
                del self.futures[key]
        for key in self.selected:
            if key in self.futures or now - self.last_capture.get(key, 0) < self.period:
                continue
            self.last_capture[key] = now
            try:
                self.futures[key], _ = self._capture(key, data, now)
                self.future_generation[key] = self.generation[key]
            except Exception as exc:
                self.errors[key] = str(exc)

    def overlay(self, key):
        item = self.latest.get(key)
        return None if item is None else item["overlay_rgb"]

    def state(self):
        now = time.monotonic()
        cameras = {}
        for key in self.selected:
            item = self.latest.get(key)
            age = None if item is None else now - item["record"]["captured_at_monotonic_s"]
            cameras[key] = {"enabled": key in self.selected, "camera": key, "error": self.errors.get(key),
                            "age_s": age, "stale": age is None or age > 2 or key in self.errors,
                            "processing": key in self.futures,
                            "latest": None if item is None else item["record"]}
        return {"enabled": True, "camera": self.selected[0] if len(self.selected) == 1 else "both",
                "cameras": cameras,
                "stale": any(cameras[key]["stale"] for key in self.selected)}

    def _write_json(self):
        if self.json_path is None:
            return
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.json_path.with_suffix(self.json_path.suffix + ".tmp")
        temp.write_text(json.dumps(self.state(), indent=2), encoding="utf-8")
        temp.replace(self.json_path)

    def invalidate(self):
        """Drop old-frame predictions on reset without reusing detection IDs as tracks."""
        for key in self.specs:
            self.generation[key] += 1
            self.latest.pop(key, None)
            self.errors.pop(key, None)
            self.last_capture.pop(key, None)
        self._write_json()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
