"""Camera observations for the task executor: the storage-tray inventory and the pre-grasp wrist alignment.

  cam = TaskCamera(scene_cfg, capture, storage_tray, edge, ctrl)   # capture(key) -> (rgb, CameraGeometry)
  ex.camera = cam                                                 # the executor then looks with the cameras

Inventory (Observe skill): one global image; global fits inside the storage tray are candidates. Colours the global
camera is known to confuse (configs/target_localization.yaml `global_lookalikes`: dark green seen as light green,
light green seen as yellow) and silhouettes inside the tray that could not be fitted (merged, occluded) are looked
at with the wrist camera, which decides colour and pose. When a colour the order still needs is not in the global
image, the wrist sweeps the tray before the executor accepts that it is missing. A silhouette in the tray that stays
unexplained makes the observation unclear. While waiting for restock, only the fixed global camera is used.

Pick alignment: approach over the observed pose, wrist image, move over the wrist estimate, second wrist image (the
#19 two-view rule); when the first view does not confirm, the camera is re-centred over the estimate once.

Ground truth is never read here. The executor admits only observations it can match to a simulator block ID
into its inventory (agentic_sim.task.executor, `associate`); grasp targets use the camera pose.
"""

from __future__ import annotations

import math

import numpy as np

from agentic_sim.config import load
from agentic_sim.perception.localize import _record_camera, plan_wrist_view, refine_with_wrist
from agentic_sim.perception.vertices import cube_vertices
from agentic_sim.scene_layout import block_in_tray
from agentic_sim.task.skills import Skill, _Fail

LOCALIZATION = "configs/target_localization.yaml"


class TaskCamera:
    def __init__(self, scene_cfg: dict, capture, storage_tray: dict, edge: float, ctrl, estimators: dict):
        """capture(key) -> (rgb, geometry) of the current frame; estimators[key].fit_detections / predict."""
        self.scene_cfg, self.capture, self.tray, self.edge, self.ctrl = scene_cfg, capture, storage_tray, edge, ctrl
        self.estimators = estimators
        self.cfg = load(LOCALIZATION)
        self.lookalike_of: dict = {}        # global colour -> colours it may really be
        for true, seen in (self.cfg.get("global_lookalikes") or {}).items():
            for c in seen:
                self.lookalike_of.setdefault(c, set()).add(true)
        self.frames = 0
        self.log: list = []                 # per image: what was seen (for the evidence files)

    # ---- images -----------------------------------------------------------------------------------
    def record(self, key: str) -> tuple:
        rgb, geo = self.capture(key)
        overlay, rec = self.estimators[key].predict(rgb, geo, self.frames, 0.0)
        self.frames += 1
        return rgb, rec, overlay

    def tray_fits(self, rec: dict) -> list:
        """Accepted fits inside the storage tray: {color, xy, yaw, fit, source}."""
        out = []
        for d in rec["detections"]:
            g = d["geometry"]
            if g["status"] != "estimated" or not block_in_tray(g["center_xyz_m"], self.tray, self.edge):
                continue
            out.append({"color": d["color"], "xy": [float(x) for x in g["center_xyz_m"][:2]],
                        "yaw": float(g["yaw_rad_mod_pi_over_2"]), "fit": float(g["fit_iou"]),
                        "source": rec["camera"]})
        return out

    def tray_unfitted(self, rec: dict) -> list:
        """Silhouettes whose box centre lies on the storage tray floor but that could not be fitted."""
        cam = _record_camera(rec)
        out = []
        z = self.tray["floor_top"] + self.edge / 2
        for d in rec["detections"]:
            if d["geometry"]["status"] == "estimated":
                continue
            x0, y0, x1, y1 = d["bbox_xyxy"]
            try:
                p = cam.intersect_plane(((x0 + x1) / 2, (y0 + y1) / 2), z)
            except ValueError:
                continue
            if block_in_tray([p[0], p[1], z], self.tray, self.edge * 0.5):
                out.append({"color": d["color"], "xy": [float(p[0]), float(p[1])], "area_px": d["area_px"],
                            "reason": d["geometry"].get("reason")})
        return out

    def ambiguous(self, color: str) -> set:
        """Colours a global detection of `color` may really be (empty: trusted)."""
        return self.lookalike_of.get(color, set())

    def candidate(self, item: dict) -> dict:
        z = self.tray["floor_top"]
        return {"center_xyz_m": [item["xy"][0], item["xy"][1], z + self.edge / 2],
                "vertices_xyz_m": cube_vertices(item["xy"], item.get("yaw", 0.0), self.edge, z).tolist()}

    def wrist_verdict(self, rec: dict, rgb, item: dict, colors) -> dict:
        """The wrist's judgement of one item: confirmed colour and pose, or not seen."""
        cand = self.candidate(item)
        for c in colors:
            v = refine_with_wrist(rec, cand, c, self.cfg, rgb)
            if v["status"] == "confirmed":
                return {"status": "confirmed", "color": c, "xy": [float(x) for x in v["center_xyz_m"][:2]],
                        "yaw": float(v["yaw_rad_mod_pi_over_2"]), "fit": float(v["fit_iou"]), "source": "wrist"}
        near = [f for f in self.tray_fits(rec) if np.linalg.norm(np.subtract(f["xy"], item["xy"])) <=
                float(self.cfg["association_gate_m"])]
        if near:
            f = min(near, key=lambda f: np.linalg.norm(np.subtract(f["xy"], item["xy"])))
            return {"status": "confirmed", "color": f["color"], "xy": f["xy"], "yaw": f["yaw"], "fit": f["fit"],
                    "source": "wrist"}
        return {"status": "not_seen"}

    def view_plan(self, xy, z_ee: float):
        target = [xy[0], xy[1], self.tray["floor_top"] + self.edge / 2]
        return plan_wrist_view(self.ctrl.chain, self.ctrl.p, self.scene_cfg, target, z_ee, self.ctrl.state.q_goal)

    def sweep_points(self) -> list:
        """Tray points the wrist looks at when a needed colour is not in the global image (#19 source search)."""
        c, s = math.cos(self.tray["yaw"]), math.sin(self.tray["yaw"])
        rot = np.array([[c, -s], [s, c]])
        return [(np.asarray(self.tray["center"]) + rot @ (np.array([u, v]) * np.asarray(self.tray["inner_half"])))
                .tolist() for u, v in ((-0.5, 0), (0.4, 0), (-0.5, -0.55), (-0.5, 0.55))]


def merge(items: list, item: dict, edge: float) -> bool:
    """Add an observed block; one closer than one edge to a known one is the same block (a wrist estimate replaces a
    global one). Returns True when it was new."""
    for k in items:
        if np.linalg.norm(np.subtract(k["xy"], item["xy"])) < edge:
            if item["source"] == "wrist" or k["source"] != "wrist":
                k.update({x: item[x] for x in ("color", "xy", "yaw", "fit", "source")})
            return False
    items.append(dict(item))
    return True


class Observe(Skill):
    """Look at the storage tray: global image, wrist checks where the global camera is unsure, a wrist sweep when a
    needed colour is not seen (standby / select). Restock polls take one stationary global image."""
    name = "observe"
    cancel_phases = ()

    def body(self):
        cam: TaskCamera = self.ctx.camera
        purpose = self.params.get("purpose", "select")
        needed = set(self.params.get("needed") or [])
        z_view = float(cam.cfg["view_ee_z"][0])
        self.set_phase("global")
        yield
        rgb, rec, _ = cam.record("global")
        items, unresolved, views = [], [], []
        for f in cam.tray_fits(rec):
            f["global_color"] = f["color"]
            merge(items, f, cam.edge)
        # standby: the question names what is left, so every unsure colour is checked; otherwise only blocks that
        # may be a colour the order still needs (the arm stays at standby while it waits for a restock)
        relevant = None if purpose == "standby" else needed
        to_check = [k for k in items if cam.ambiguous(k["color"]) and
                    (relevant is None or ({k["color"]} | cam.ambiguous(k["color"])) & relevant)]
        unresolved = cam.tray_unfitted(rec)
        views.append({"camera": "global", "fits": len(items), "unfitted": unresolved, "frame": rec["frame_number"]})
        if purpose == "restock_wait":
            # Someone may be reaching into the tray. An unclear frame must wait for the next global poll.
            for item in to_check:
                item["uncertain"] = True
            reason = ""
            if unresolved:
                reason = f"{len(unresolved)} unresolved silhouette(s) in the global restock view"
            elif to_check:
                reason = "needed colour is ambiguous in the global restock view"
            self.evidence.update(items=items, unresolved=unresolved, views=views, clear=not reason,
                                 reason_unclear=reason, observation="camera")
            return
        # wrist checks: look-alike colours, then silhouettes the global camera could not fit
        for k in to_check:
            colors = [k["color"], *sorted(cam.ambiguous(k["color"]))]
            v = yield from self._wrist_look(cam, k["xy"], z_view, views, k, colors)
            k["checked"] = v["status"]
            if v["status"] == "confirmed":
                k.update({x: v[x] for x in ("color", "xy", "yaw", "fit", "source")})
            else:
                k["uncertain"] = True
        still = []
        for u in unresolved:
            if any(np.linalg.norm(np.subtract(k["xy"], u["xy"])) < cam.edge for k in items if k["source"] == "wrist"):
                continue
            seen = yield from self._wrist_area(cam, u["xy"], z_view, views, items)
            if not seen:
                still.append(u)
        missing = [c for c in needed if not any(k["color"] == c and not k.get("uncertain") for k in items)]
        if missing and purpose in ("standby", "select"):
            for xy in cam.sweep_points():
                yield from self._wrist_area(cam, xy, z_view + 0.025, views, items)
        for k in items:          # unchecked look-alikes that cannot matter here keep their global colour
            if k not in to_check and cam.ambiguous(k["color"]):
                k["unverified_lookalike"] = True
        clear = not still and not any(k.get("uncertain") for k in items)
        reason = ""
        if still:
            reason = f"{len(still)} silhouette(s) in the tray could not be identified"
        elif any(k.get("uncertain") for k in items):
            reason = "a block's colour could not be confirmed by the wrist"
        self.evidence.update({"items": items, "unresolved": still, "views": views, "clear": clear,
                              "reason_unclear": reason, "observation": "camera"})
        if purpose == "standby" and len(views) > 1:
            self.set_phase("return")
            yield from self.move(self.world.home, 0.0, cancellable=False)

    def _wrist_look(self, cam, xy, z, views, item, colors):
        plan = cam.view_plan(xy, z)
        if plan is None:
            views.append({"camera": "wrist", "at": xy, "outcome": "no_view_plan"})
            return {"status": "not_seen"}
        self.set_phase("wrist")
        yield from self.move(plan["ee_xyz_m"], 0.0, cancellable=False)
        yield from self.wait(float(cam.cfg.get("settle_frames", 30)) * self.ctx.dt, cancellable=False)
        rgb, rec, _ = cam.record("wrist")
        v = cam.wrist_verdict(rec, rgb, item, colors)
        views.append({"camera": "wrist", "at": xy, "ee": plan["ee_xyz_m"], "frame": rec["frame_number"],
                      "outcome": v["status"], "color": v.get("color")})
        return v

    def _wrist_area(self, cam, xy, z, views, items):
        """Wrist image centred on a tray point; every accepted fit there joins the inventory. True if anything fit."""
        plan = cam.view_plan(xy, z)
        if plan is None:
            views.append({"camera": "wrist", "at": xy, "outcome": "no_view_plan"})
            return False
        self.set_phase("wrist")
        yield from self.move(plan["ee_xyz_m"], 0.0, cancellable=False)
        yield from self.wait(float(cam.cfg.get("settle_frames", 30)) * self.ctx.dt, cancellable=False)
        rgb, rec, _ = cam.record("wrist")
        fits = cam.tray_fits(rec)
        near = [f for f in fits if np.linalg.norm(np.subtract(f["xy"], xy)) < 2 * cam.edge]
        for f in fits:
            merge(items, f, cam.edge)
        views.append({"camera": "wrist", "at": xy, "ee": plan["ee_xyz_m"], "frame": rec["frame_number"],
                      "fits": [{"color": f["color"], "xy": f["xy"]} for f in fits]})
        return bool(near)


def align_target(skill: Skill, target: dict, color: str):
    """Wrist alignment over an observed block before the descent (generator; returns the grasp pose or raises).

    First wrist image at the approach hover, move over the wrist estimate, second image (#19 B). If the first image
    does not confirm the block, the camera is re-centred over the estimate once."""
    cam: TaskCamera = skill.ctx.camera
    pose = dict(target)
    views = []

    def look():
        rgb, rec, _ = cam.record("wrist")
        v = cam.wrist_verdict(rec, rgb, pose, [color])
        views.append({"frame": rec["frame_number"], "outcome": v["status"], "color": v.get("color"),
                      "xy": v.get("xy")})
        return v if v["status"] == "confirmed" and v["color"] == color else None

    settle = float(cam.cfg.get("settle_frames", 30)) * skill.ctx.dt
    yield from skill.wait(settle)
    first = look()
    if first is None:
        plan = cam.view_plan(pose["xy"], float(cam.cfg["view_ee_z"][0]))
        if plan is not None:
            yield from skill.move(plan["ee_xyz_m"], 0.0)
            yield from skill.wait(settle)
            first = look()
    if first is None:
        skill.evidence["wrist_views"] = views
        raise _Fail("wrist_unconfirmed")
    pose.update(xy=first["xy"], yaw=first["yaw"])
    off = skill.yaw_offset_for(pose["xy"], pose["yaw"])
    g = skill.grasp_xy(pose["xy"], off)
    yield from skill.move([g[0], g[1], float(skill.cfg["motion"]["safe_z"])], off)
    yield from skill.wait(settle)
    second = look()
    if second is not None:
        pose.update(xy=second["xy"], yaw=second["yaw"])
    skill.evidence["wrist_views"] = views
    skill.evidence["pose_source"] = "wrist" if second is not None else "wrist_first"
    return pose
