"""Re-mounting the global D455 (practice room, finals venue): camera height and tilt from the table plane in depth.

The plane gives 3 of the 6 pose numbers - the height above the table and the two tilts - and these are the ones the
eye judges worst on a near top-down camera (1 cm of height is a 1.7 % scale change). On 68 saved 11/26 frames of the
unmoved camera the height reads 0.3-1.0 mm and the tilts within 0.22 deg of the fitted 11/25 pose (live 11/27: +0.9 mm,
0.1 deg). The in-plane position (x, y, yaw about the
vertical) is not in the plane; the restore tool shows it as the reference frame's edges over the live image, and the
exact pose for coordinates still comes from tools/finals_calib.py (d455-capture / d455-fit) afterwards.

Camera frame: colour optical frame (x right, y down, z forward), metres. The table normal n points up (towards the
camera, so n_z < 0); a table point p satisfies n.p = -h with h the camera's height above the table.
"""

from __future__ import annotations

import math

import numpy as np

TABLE_Z = -0.0016          # table top in the mount frame (finals_calib: the base stands 1.6 mm into the fitted plane)
EDGE_PX = 40               # the four edge probes sit this far inside the 640x480 frame
TOL_HEIGHT_MM = 2.0        # "restored" when the height is within this
TOL_TILT_DEG = 0.2         # and each tilt within this (0.2 deg = 2 mm of depth across the frame)
NEAR_M = 0.015             # parallel planes searched within this of the dominant one (tray floors sit ~6 mm higher)
MIN_SHARE = 0.08           # a parallel plane counts as a surface with at least this share of the points


def _points(depth, K, step):
    vv, uu = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    z = depth[vv, uu]
    ok = (z > 0.2) & (z < 2.0)
    return np.stack([(uu[ok] - K["cx"]) / K["fx"] * z[ok], (vv[ok] - K["cy"]) / K["fy"] * z[ok], z[ok]], 1)


def _refine(P, n, h, band):
    """Least-squares plane through the points within band of n.p = -h."""
    on = P[np.abs(P @ n + h) < band]
    c = on.mean(0)
    n = np.linalg.svd(on - c, full_matrices=False)[2][2]
    n = n if n[2] < 0 else -n
    return n, float(-(n @ c)), len(on)


def table_plane(depth, K, step=6, seed=0):
    """Table plane from an aligned depth frame (m): {"normal", "height_m", "share"} or None.

    RANSAC for the dominant plane, then the lowest well-populated plane parallel to it (within NEAR_M): the table, not
    a tray floor or a box lid that may cover more of the view."""
    P = _points(depth, K, step)
    if len(P) < 500:
        return None
    rng = np.random.default_rng(seed)
    a, b, c = (P[rng.integers(0, len(P), 400)] for _ in range(3))
    N = np.cross(b - a, c - a)
    ok = np.linalg.norm(N, axis=1) > 1e-9
    N = N[ok] / np.linalg.norm(N[ok], axis=1, keepdims=True)
    N[N[:, 2] > 0] *= -1
    H = -(N * a[ok]).sum(1)
    sample = P[rng.integers(0, len(P), 3000)]
    best = int(np.argmax((np.abs(sample @ N.T + H) < 0.004).sum(0)))
    n, h, _ = _refine(P, N[best], H[best], 0.004)
    # heights of all points along n; the table is the farthest surface among the parallel ones
    s = -(P @ n)
    near = s[np.abs(s - h) < NEAR_M]
    edges = np.arange(h - NEAR_M, h + NEAR_M + 1e-4, 0.0005)
    counts = np.histogram(near, edges)[0]
    window = np.convolve(counts, np.ones(9), "same")        # +-2 mm
    centres = (edges[:-1] + edges[1:]) / 2
    surfaces = np.nonzero(window >= MIN_SHARE * len(P))[0]
    if len(surfaces):
        far = centres[surfaces].max()
        pick = np.nonzero(np.abs(centres - far) < 0.004)[0]
        h = float(centres[pick[np.argmax(window[pick])]])
    for band in (0.004, 0.0025, 0.0025) + (0.0015,) * 6:   # narrow last: a tray floor 6 mm up must not pull it
        n, h, count = _refine(P, n, h, band)
    return {"normal": n, "height_m": h, "share": count / len(P)}


def target_from_pose(pose):
    """Table plane the camera saw at a fitted pose (outputs/d455/pose_*.json, mount_from_color)."""
    R = np.asarray(pose["mount_from_color"]["R"], float)
    t = np.asarray(pose["mount_from_color"]["t"], float)
    return {"normal": R[2].copy(), "height_m": float(t[2] - TABLE_Z)}


def tilts_deg(n):
    """(forward, side): the camera's rotation about its image x axis (top / bottom edge) and image y axis."""
    return math.degrees(math.atan2(n[1], -n[2])), math.degrees(math.atan2(n[0], -n[2]))


def depth_at(plane, K, u, v):
    """Depth (z, m) the camera reads on the table at pixel (u, v)."""
    r = np.array([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], 1.0])
    return -plane["height_m"] / float(plane["normal"] @ r)


def compare(plane, target, K, width=640, height=480):
    """Current table plane vs the target: height, tilts and the table depth at the centre and the four edge probes."""
    probes = {"centre": (width / 2, height / 2), "top": (width / 2, EDGE_PX), "bottom": (width / 2, height - EDGE_PX),
              "left": (EDGE_PX, height / 2), "right": (width - EDGE_PX, height / 2)}
    d_mm = {k: (depth_at(plane, K, *uv) - depth_at(target, K, *uv)) * 1000 for k, uv in probes.items()}
    f, s = tilts_deg(plane["normal"])
    ft, st = tilts_deg(target["normal"])
    out = {"height_mm": (plane["height_m"] - target["height_m"]) * 1000, "forward_deg": f - ft, "side_deg": s - st,
           "depth_mm": d_mm, "share": plane.get("share")}
    out["ok"] = abs(out["height_mm"]) <= TOL_HEIGHT_MM and abs(out["forward_deg"]) <= TOL_TILT_DEG \
        and abs(out["side_deg"]) <= TOL_TILT_DEG
    return out


def advice(c):
    """Operator lines (zh-TW) for compare(): what to turn, and which way."""
    lines = []
    h = c["height_mm"]
    lines.append(f"高度 {h:+.1f} mm：" + ("OK" if abs(h) <= TOL_HEIGHT_MM else ("相機太高，往下降" if h > 0 else "相機太低，往上升")))
    d = c["depth_mm"]
    tb = d["top"] - d["bottom"]
    lines.append(f"上下傾 {c['forward_deg']:+.2f}°（上緣−下緣 {tb:+.1f} mm）：" + (
        "OK" if abs(c["forward_deg"]) <= TOL_TILT_DEG else ("畫面上緣看得太遠，鏡頭朝畫面下緣那側轉" if tb > 0 else "畫面下緣看得太遠，鏡頭朝畫面上緣那側轉")))
    lr = d["left"] - d["right"]
    lines.append(f"左右傾 {c['side_deg']:+.2f}°（左緣−右緣 {lr:+.1f} mm）：" + (
        "OK" if abs(c["side_deg"]) <= TOL_TILT_DEG else ("畫面左緣看得太遠，鏡頭朝畫面右緣那側轉" if lr > 0 else "畫面右緣看得太遠，鏡頭朝畫面左緣那側轉")))
    return lines
