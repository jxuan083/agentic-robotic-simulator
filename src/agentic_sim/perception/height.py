"""Depth check of each colour segment: an upright block on the tray floor, the top of a stack, or a problem.

Input is the global D455 depth aligned to the colour image (same intrinsics as the RGB frame, z along the optical
axis in metres, 0 = no reading) and one colour mask per segment. The segment's dominant plane is its visible top face:
its tilt tells an upright block from a tilted one, its height in block edges tells the floor layer from a stack, and
its area tells one whole top from touching blocks or a partly hidden one. Nothing here assumes the RGB fit succeeded.
"""
from __future__ import annotations

import math

import numpy as np

STATUSES = ("upright", "stacked", "tilted", "merged", "partial", "off_layer", "side_face", "no_depth")
PROBLEMS = ("stacked", "tilted", "merged", "partial", "off_layer")   # side_face / no_depth: not evidence either way


def backproject(depth, mask, camera, erode_px: int = 2) -> np.ndarray:
    """Mount-frame points of the valid depth pixels inside `mask`, eroded so edge (flying) pixels are dropped."""
    import cv2
    m = np.asarray(mask, np.uint8)
    if erode_px > 0:
        m = cv2.erode(m, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    z = np.asarray(depth, float)
    vv, uu = np.nonzero((m > 0) & np.isfinite(z) & (z > 0))
    if len(uu) == 0:
        return np.zeros((0, 3))
    zz = z[vv, uu]
    pix = np.stack([uu, vv, np.ones_like(uu)], 0).astype(float)
    if np.any(camera.dist):
        pts = cv2.undistortPoints(np.stack([uu, vv], 1).astype(np.float64)[:, None], camera.k, camera.dist)[:, 0]
        rays = np.c_[pts, np.ones(len(pts))].T
    else:
        rays = np.linalg.inv(camera.k) @ pix
    cam = rays * zz
    return (camera.mount_from_cv[:3, :3] @ cam).T + camera.mount_from_cv[:3, 3]


def dominant_plane(points, tol_m: float, iters: int = 200, seed: int = 0):
    """RANSAC plane with the most inliers, refit by least squares. Returns (unit normal with +z, offset, inlier mask)."""
    rng = np.random.default_rng(seed)
    n_pts = len(points)
    best = None
    for _ in range(iters):
        a, b, c = points[rng.choice(n_pts, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        inl = np.abs((points - a) @ normal) < tol_m
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        return None
    inliers = best
    for _ in range(3):   # refit; widen the band to the measured noise so a noisy face is not cut to a tilted slab
        sel = points[inliers]
        centroid = sel.mean(0)
        normal = np.linalg.svd(sel - centroid)[2][2]
        normal = normal if normal[2] >= 0 else -normal
        res = (points - centroid) @ normal
        sigma = 1.4826 * float(np.median(np.abs(res[inliers])))
        inliers = np.abs(res) < max(tol_m, 3 * sigma)
        if inliers.sum() < 3:
            return None
    return normal, float(-normal @ centroid), inliers


def _hull(xy):
    import cv2
    return cv2.convexHull(np.asarray(xy, np.float32)).reshape(-1, 2).astype(float)


def height_check(points, floor_z_m: float, edge_m: float, cfg: dict, seed: int = 0) -> dict:
    """Classify one segment from its mount-frame depth points.

    upright    horizontal top, one block edge above the floor, about one face of area
    stacked    horizontal top, two or more edges up: something is under this block
    tilted     the top face is tilted beyond tilt_max_deg
    merged     horizontal one-layer top much larger than a face: touching blocks of one colour
    partial    horizontal top much smaller than a face: partly hidden, or a lower block showing past a stack
    off_layer  horizontal top between layers (resting on something uneven)
    side_face  nearly vertical plane: a block's shaded side split off by hue (its top is another segment)
    no_depth   too few depth points
    PROBLEMS are the statuses the localizer must deal with; side_face and no_depth say nothing."""
    out = {"status": "no_depth", "points_n": int(len(points))}
    if len(points) < cfg.get("min_points", 30):
        return out
    fit = dominant_plane(points, cfg.get("plane_tol_m", 0.0015), seed=seed)
    if fit is None:
        return out
    normal, _, inliers = fit
    top = points[inliers]
    tilt = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    top_z = float(np.median(top[:, 2]))
    layers = (top_z - floor_z_m) / edge_m
    hull = _hull(top[:, :2] / max(abs(float(normal[2])), 1e-6) ** 0.5)   # tilted face: undo the xy foreshortening
    import cv2
    area = float(cv2.contourArea(hull.astype(np.float32))) / edge_m ** 2
    out.update(top_z_m=round(top_z, 4), layers=round(layers, 2), tilt_deg=round(tilt, 1),
               top_area_faces=round(area, 2), top_center_xyz_m=top.mean(0).round(4).tolist(),
               top_hull_xy_m=_hull(top[:, :2]).round(4).tolist(), inliers_n=int(inliers.sum()))
    if tilt > cfg.get("side_face_min_deg", 70.0):
        out["status"] = "side_face"
    elif tilt > cfg.get("tilt_max_deg", 5.0):
        out["status"] = "tilted"
    elif abs(layers - round(layers)) > cfg.get("layer_tol", 0.15) or round(layers) < 1:
        out["status"] = "off_layer"
    elif area > cfg.get("merged_area_faces", 1.6):
        out["status"] = "merged"
    elif area < cfg.get("min_area_faces", 0.5):
        out["status"] = "partial"
    else:
        out["status"] = "upright" if round(layers) == 1 else "stacked"
    return out


def _segment_distance(p, a, b):
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip((p - a) @ ab / (ab @ ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def polygon_gap(a, b) -> float:
    """Smallest distance between two convex xy polygons (0 when they overlap)."""
    import cv2
    a, b = np.asarray(a, float), np.asarray(b, float)
    area, _ = cv2.intersectConvexConvex((a * 1000).astype(np.float32), (b * 1000).astype(np.float32))   # mm: OpenCV
    # returns 0 for overlaps of polygons this small in metres
    if area > 0:
        return 0.0
    gaps = [_segment_distance(p, q[i], q[(i + 1) % len(q)]) for p, q in ((a, b), (b, a))
            for p in p for i in range(len(q))]
    return min(gaps)


def close_pairs(checks: list, gap_m: float, hull_margin_m: float = 0.0) -> list:
    """Index pairs (i, j, gap_m) of segments whose top faces are closer than `gap_m` in xy (gripper clearance).
    The depth top face misses its rim (mask erosion, plane band), so each top's gap is reduced by `hull_margin_m`."""
    hulls = [c.get("top_hull_xy_m") if c.get("status") not in ("side_face", "no_depth") else None for c in checks]
    pairs = []
    for i in range(len(hulls)):
        for j in range(i + 1, len(hulls)):
            if hulls[i] is None or hulls[j] is None:
                continue
            g = max(0.0, polygon_gap(hulls[i], hulls[j]) - 2 * hull_margin_m)
            if g < gap_m:
                pairs.append((i, j, round(g, 4)))
    return pairs
