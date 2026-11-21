"""Coarse-to-fine target localization: global camera candidates, wrist close-up confirmation.

  cands = global_candidates(global_record, "red", cfg)          # ranked, image-derived only
  plan = plan_wrist_view(chain, ctrl_params, scene, cand["center_xyz_m"], ee_z, q_seed)
  ... move the arm to plan["ee_xyz_m"], capture one wrist record ...
  verdict = refine_with_wrist(wrist_record, cand, "red", cfg)  # confirmed / wrong_color / not_seen

The global camera sees the whole storage tray but confuses a few palette colours (dark green -> light green, light
green -> yellow in the rendered evaluations), so its candidates for a requested colour include those look-alikes.
The wrist camera, centred over one candidate from above, decides the colour and gives the final pose. Nothing here
reads simulator block poses; the inputs are perception records (agentic_sim.perception.live) and calibration.
"""
from __future__ import annotations

import math

import numpy as np

from agentic_sim.perception.vertices import CameraGeometry


def _estimated(detection: dict) -> bool:
    return (detection.get("geometry") or {}).get("status") == "estimated"


def _center(detection: dict) -> np.ndarray:
    return np.asarray(detection["geometry"]["center_xyz_m"], float)


def global_candidates(record: dict, target_color: str, cfg: dict) -> list[dict]:
    """Global-camera blocks that may be `target_color`: exact colour first, then configured look-alikes.

    Within each group, better silhouette fits come first. Rejected fits are left out (no reliable centre)."""
    alike = list((cfg.get("global_lookalikes") or {}).get(target_color, []))
    out = []
    for d in record.get("detections", []):
        if not _estimated(d):
            continue
        c = d.get("color")
        if c == target_color:
            rank = 0
        elif c in alike:
            rank = 1
        else:
            continue
        g = d["geometry"]
        out.append({"rank": rank, "global_color": c, "global_detection_id": d.get("detection_id"),
                    "center_xyz_m": [float(x) for x in g["center_xyz_m"]],
                    "yaw_rad_mod_pi_over_2": g.get("yaw_rad_mod_pi_over_2"),
                    "vertices_xyz_m": [vx["xyz_m"] for vx in g.get("vertices", [])],
                    "fit_iou": g.get("fit_iou")})
    out.sort(key=lambda c: (c["rank"], -float(c["fit_iou"] or 0.0)))
    return out[:int(cfg.get("max_candidates", 4))]


def _optical_hit(camera: CameraGeometry, plane_z: float):
    """Where the wrist camera's optical axis meets the horizontal plane z = plane_z (None if it looks away)."""
    o = camera.mount_from_cv[:3, 3]
    axis = camera.mount_from_cv[:3, 2]
    if axis[2] > -1e-3:
        return None
    t = (plane_z - o[2]) / axis[2]
    return o + t * axis if t > 0 else None


def _view(chain, params, scene, p, q_seed, target):
    """IK for EE position p (neutral finger heading), the wrist camera there and where the target lands in its image:
    (q, camera, optical-axis hit, normalized image offset of the target from the centre) or None (no IK)."""
    pan = chain.pan_origin_mount()
    yaw = float(np.arctan2(p[1] - pan[1], p[0] - pan[0]) + params.finger_heading)
    q, ok, _ = chain.solve_ik_tilt(p, yaw, q_seed, params.max_tilt_deg)
    if not ok:
        return None
    cam = CameraGeometry.from_scene(scene, "wrist", joints_rad=q)
    hit = _optical_hit(cam, float(target[2]))
    if hit is None:
        return None
    try:
        uv = cam.project(target[None, :])[0]
    except ValueError:      # behind the camera
        return None
    off = max(abs(uv[0] - cam.width / 2) / (cam.width / 2), abs(uv[1] - cam.height / 2) / (cam.height / 2))
    return q, cam, hit, float(off)


def plan_wrist_view(chain, params, scene: dict, target_xyz, ee_z: float, q_seed, iters: int = 15,
                    tol_m: float = 0.002, max_center_offset: float = 0.5) -> dict | None:
    """EE target (at height ee_z) that brings target_xyz to the middle of the wrist image.

    Fixed-point iteration on the EE xy: solve the controller's IK (finger heading as for a neutral grasp), place the
    camera by FK + the scene's camera mount, intersect its optical axis with the target's height and shift the EE by
    the miss. Where the arm cannot reach the centring pose (far side of the tray) the step is halved and the best
    reachable view is kept. Accepted when the target projects within max_center_offset of the image centre
    (fraction of the half width / height); None otherwise."""
    target = np.asarray(target_xyz, float)
    lo, hi = np.asarray(params.ws_lo, float), np.asarray(params.ws_hi, float)
    p = np.clip(np.array([target[0], target[1], ee_z], float), lo, hi)
    q = np.asarray(q_seed, float)
    cur = _view(chain, params, scene, p, q, target)
    if cur is None:    # start nearer the arm: the target's own xy may be out of reach at this height
        pan = chain.pan_origin_mount()
        d = target[:2] - pan[:2]
        p[:2] = pan[:2] + d * 0.8
        cur = _view(chain, params, scene, p, q, target)
        if cur is None:
            return None
    best = (cur[3], p.copy(), cur)
    step = 1.0
    for _ in range(iters):
        q, cam, hit, off = cur
        miss = target[:2] - hit[:2]
        if float(np.linalg.norm(miss)) <= tol_m:
            break
        nxt = np.clip(p + step * np.array([miss[0], miss[1], 0.0]), lo, hi)
        v = _view(chain, params, scene, nxt, q, target)
        if v is None:
            step *= 0.5
            if step < 0.1:
                break
            continue
        p, cur = nxt, v
        if v[3] < best[0]:
            best = (v[3], p.copy(), v)
    off, p, (q, cam, hit, _) = best
    if off > max_center_offset:
        return None
    return {"ee_xyz_m": p.tolist(), "q": q.tolist(), "center_offset": off,
            "axis_miss_mm": float(np.linalg.norm(np.asarray(target[:2]) - hit[:2]) * 1000),
            "camera_height_m": float(cam.mount_from_cv[2, 3] - target[2])}


def _record_camera(record: dict) -> CameraGeometry | None:
    size = record.get("image_size")
    if not size:
        return None
    return CameraGeometry(int(size[0]), int(size[1]), np.asarray(record["intrinsics"], float),
                          np.asarray(record["distortion"], float), np.asarray(record["mount_from_cv"], float),
                          record.get("camera", "wrist"))


def _covering_rejected(record: dict, center_xyz) -> dict | None:
    """The rejected wrist silhouette whose box contains the candidate centre's projection (smallest box first)."""
    cam = _record_camera(record)
    if cam is None:
        return None
    try:
        u, v = cam.project(np.asarray(center_xyz, float)[None, :])[0]
    except ValueError:
        return None
    hits = []
    for d in record.get("detections", []):
        box = d.get("bbox_xyxy")
        if _estimated(d) or not box:
            continue
        if box[0] <= u <= box[2] and box[1] <= v <= box[3]:
            hits.append(((box[2] - box[0]) * (box[3] - box[1]), d))
    return min(hits, key=lambda h: h[0])[1] if hits else None


def refine_with_wrist(record: dict, candidate: dict, target_color: str, cfg: dict) -> dict:
    """Judge one candidate from a wrist record taken over it.

    confirmed     an accepted wrist fit of the target colour within the gate of the candidate centre
    color_only    no accepted fit there, but the rejected wrist silhouette covering the candidate is the target
                  colour (e.g. two same-colour blocks touching in the image): colour confirmed, no wrist pose
    wrong_color   an accepted wrist fit of another colour is there (the global colour was a look-alike)
    not_seen      nothing usable near the candidate (clipped, occluded); worth another view"""
    gate = float(cfg.get("association_gate_m", 0.03))
    c0 = np.asarray(candidate["center_xyz_m"], float)
    near = []
    for d in record.get("detections", []):
        if not _estimated(d):
            continue
        dist = float(np.linalg.norm(_center(d)[:2] - c0[:2]))
        if dist <= gate:
            near.append((dist, d))
    if not near:
        rejected = [((d.get("geometry") or {}).get("reason") or "rejected") for d in record.get("detections", [])
                    if not _estimated(d)]
        cover = _covering_rejected(record, c0)
        if cover is not None and cover.get("color") == target_color:
            return {"status": "color_only", "wrist_color": cover["color"],
                    "wrist_rejected_reason": cover["geometry"].get("reason"), "wrist_rejected_reasons": rejected}
        # a rejected silhouette of another colour may be several blocks merged: not proof, try another view
        return {"status": "not_seen", "wrist_rejected_reasons": rejected,
                "covering_color": None if cover is None else cover.get("color")}
    near.sort(key=lambda x: x[0])
    same = [(dist, d) for dist, d in near if d.get("color") == target_color]
    if not same:
        dist, d = near[0]
        return {"status": "wrong_color", "wrist_color": d.get("color"), "offset_from_global_mm": dist * 1000}
    dist, d = same[0]
    g = d["geometry"]
    return {"status": "confirmed", "wrist_color": d.get("color"), "wrist_detection_id": d.get("detection_id"),
            "offset_from_global_mm": dist * 1000,
            "center_xyz_m": [float(x) for x in g["center_xyz_m"]],
            "yaw_rad_mod_pi_over_2": g.get("yaw_rad_mod_pi_over_2"),
            "vertices_xyz_m": [vx["xyz_m"] for vx in g.get("vertices", [])],
            "fit_iou": g.get("fit_iou")}


def target_pose(found: dict, candidate: dict, wrist_record: dict, global_record: dict) -> dict:
    """The localization output (mount frame, m / rad) with its provenance. The pose is the wrist fit when the wrist
    confirmed one (found["status"] == "confirmed"), else the global fit of the candidate whose colour the wrist
    confirmed (source.pose == "global")."""
    if found["status"] != "confirmed":
        found = dict(found, center_xyz_m=candidate["center_xyz_m"], vertices_xyz_m=candidate["vertices_xyz_m"],
                     yaw_rad_mod_pi_over_2=candidate.get("yaw_rad_mod_pi_over_2"), fit_iou=candidate.get("fit_iou"))
    yaw = found.get("yaw_rad_mod_pi_over_2")
    return {"status": "found", "coordinate_frame": "mount", "units": "m, rad",
            "center_xyz_m": found["center_xyz_m"],
            "yaw_rad_mod_pi_over_2": yaw, "yaw_deg_mod_90": None if yaw is None else math.degrees(yaw),
            "vertices_xyz_m": found["vertices_xyz_m"], "fit_iou": found.get("fit_iou"),
            "source": {"pose": "wrist" if found["status"] == "confirmed" else "global",
                       "color_check": "wrist", "wrist_frame_number": wrist_record.get("frame_number"),
                       "wrist_captured_at_monotonic_s": wrist_record.get("captured_at_monotonic_s"),
                       "wrist_mount_from_cv": wrist_record.get("mount_from_cv"),
                       "global_frame_number": global_record.get("frame_number"),
                       "global_color": candidate["global_color"],
                       "global_center_xyz_m": candidate["center_xyz_m"],
                       "global_yaw_rad_mod_pi_over_2": candidate.get("yaw_rad_mod_pi_over_2"),
                       "global_vertices_xyz_m": candidate.get("vertices_xyz_m"),
                       "global_to_wrist_offset_mm": found.get("offset_from_global_mm")}}
