"""Cameras for the task executor's perception skills (Acquire) in MuJoCo: on-demand global RGB-D and wrist captures.

  cams = MjTaskCameras(session, depth_noise_1m_m=0.005, seed=0)
  rec = cams.global_record()          # global detections + fits (perception.localize.global_candidates input)
  rgb, depth, cam = cams.global_rgbd()  # global colour + depth (z along the optical axis, D455-like noise)
  dets = cams.segment_global(rgb)     # hue-split colour segments (depth height check input)
  rec, rgb = cams.wrist()             # one wrist capture with detections and fits

Every capture is rendered and estimated within the calling frame (simulated time does not advance). Predictions use
images and camera calibration only. block_id_near() is the one simulation-truth call: it names the block nearest a
perceived pose so the executor can keep its per-block bookkeeping and verify deliveries (replaced by camera checks
later); no decision about where to move or grasp uses it.
"""
from __future__ import annotations

import time

import numpy as np


class MjTaskCameras:
    def __init__(self, session, depth_noise_1m_m: float = 0.005, seed: int = 0, log=lambda _: None):
        from agentic_sim.mujoco_backend.live_perception import MjLivePerception
        self.session = session
        manager = getattr(session, "live_perception", None)
        if manager is None:
            manager = MjLivePerception(session.scene_cfg, session.m, hz=2, log=log, camera="both")
            manager.last_capture["global"] = manager.last_capture["wrist"] = float("inf")   # captures on request
            session.live_perception = manager
        self.manager = manager
        self.scene = session.scene_cfg
        self.edge, self.floor = manager.edge, manager.floor
        self.depth_noise_1m_m = float(depth_noise_1m_m)
        self.rng = np.random.default_rng(seed)
        self._renderer = None

    def _estimate(self, key: str):
        future, rgb = self.manager._capture(key, self.session.d, time.monotonic())
        overlay, record = future.result(timeout=45)
        self.manager.latest[key] = {"record": record, "overlay_rgb": overlay}
        return record, rgb

    def global_record(self) -> dict:
        return self._estimate("global")[0]

    def wrist(self):
        return self._estimate("wrist")

    def global_rgbd(self):
        """Global colour and depth of one frame. MuJoCo depth is exact; stereo error growing with z² (sigma
        depth_noise_1m_m at 1 m) and mixed values at depth edges (flying pixels) are added."""
        import cv2
        import mujoco
        m, d = self.session.m, self.session.d
        cam = self.manager._geometry("global", d)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(m, height=cam.height, width=cam.width)
        r, cid = self._renderer, self.manager.cam_ids["global"]
        r.update_scene(d, camera=cid)
        rgb = np.ascontiguousarray(r.render(), np.uint8)
        r.enable_depth_rendering()
        r.update_scene(d, camera=cid)
        depth = r.render().astype(float)
        r.disable_depth_rendering()
        if self.depth_noise_1m_m > 0:
            depth = depth + self.rng.normal(0, 1, depth.shape) * self.depth_noise_1m_m * depth ** 2
            lo = cv2.erode(depth.astype(np.float32), np.ones((3, 3), np.uint8))
            hi = cv2.dilate(depth.astype(np.float32), np.ones((3, 3), np.uint8))
            mix = self.rng.uniform(0, 1, depth.shape)
            depth = np.where((hi - lo) > 0.005, lo + mix * (hi - lo), depth)
        return rgb, depth, cam

    def segment_global(self, rgb) -> list:
        """Colour segments split by hue (and green by value), so touching blocks of different colours separate."""
        return list(self.manager.estimators["global"].split_detectors[1].predict(rgb).detections)

    def show(self, key: str, overlay):
        """Put an annotated image on the simulator panel for `key` (global / wrist)."""
        self.manager.latest[key] = {"record": None, "overlay_rgb": overlay}

    def block_id_near(self, xyz) -> int:
        """Simulation truth, bookkeeping only: the block whose centre is nearest `xyz` (3D: stacks share xy)."""
        blocks = self.session.block_positions()
        return int(np.argmin([np.linalg.norm(np.asarray(b) - np.asarray(xyz, float)) for b in blocks]))

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
