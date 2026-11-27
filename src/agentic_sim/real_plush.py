"""The finals plush toy in D455 depth: a Shiba dog sitting in an Enoden 305 tram (notes/261128_玩偶規格與決賽用法.md).

The block detector (agentic_sim.real_d455.detect) only keeps tops 26-45 mm above the table, so the plush (tram roof
~50 mm, the dog's head ~100 mm) is not a block; but its rounded tram edges and paws pass through that band and its
colours (dark green, pale yellow, light blue, tan) are block colours. So the plush is found first, from heights:

  mount-frame points (aligned depth, or the native depth PNG of a recorded episode) -> top-down height raster
  (max z per GRID_M cell, table height taken off) -> cells higher than MIN_Z -> connected parts. A part is the plush
  when its highest point and its footprint fit the toy; taller parts are the arm, a hand or the other base.

Each plush: centre and yaw of the footprint (the tram's long axis, mod 180 deg), its height, the head position
(the highest cells: which end the dog sits at). The block detections inside the footprint are dropped
(`drop_blocks_on`), and its image area is left out of the tray-occlusion check (`image_mask`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

CONFIG = Path(__file__).resolve().parents[2] / "configs/real_plush.json"

NAME = "plush"
NAME_ZH = "柴犬玩偶"
LENGTH_M, WIDTH_M, BODY_M, HEIGHT_M = 0.090, 0.060, 0.050, 0.100   # tram length / width / height; total height
MASS_G = 28.6                                                        # without the ball chain and tag (11/28)
GRID_M = 0.004
MIN_Z = 0.040                  # above a block on a tray floor (~35 mm) and the tray rims (~23 mm)
TOP_M = (0.050, 0.135)         # its highest point: 50 lying on its side, ~100 upright; the arm and hands are taller
LONG_M = (0.065, 0.130)        # footprint sides (depth edges smooth a few mm off, the head can overhang)
SHORT_M = (0.040, 0.110)
UPRIGHT_M = 0.080              # top above this: sitting on the tram floor (as photographed); lower: fallen over
HEAD_BAND = 0.025
AREA = ((-0.05, 0.55), (-0.35, 0.40))   # mount x / y range searched (the table in front of both arms)


def mount_points(depth_m, cam, step=2):
    """Mount-frame points of an aligned depth image (metres per colour pixel; agentic_sim.real_d455.Camera)."""
    d = depth_m[::step, ::step]
    vv, uu = np.mgrid[0:depth_m.shape[0]:step, 0:depth_m.shape[1]:step]
    ok = d > 0.15
    return cam.points(np.stack([uu[ok], vv[ok]], 1), d[ok])


def native_points(depth_u16, units_m, intrinsics, depth_to_color, cam, step=2):
    """Mount-frame points of a recorded native depth image (manifest cameras.global.depth: intrinsics,
    depth_to_color rotation_column_major / translation_m) through the colour camera pose."""
    d = depth_u16[::step, ::step].astype(np.float32) * float(units_m)
    vv, uu = np.mgrid[0:depth_u16.shape[0]:step, 0:depth_u16.shape[1]:step]
    ok = d > 0.15
    z = d[ok]
    p = np.stack([(uu[ok] - intrinsics["cx"]) / intrinsics["fx"] * z,
                  (vv[ok] - intrinsics["cy"]) / intrinsics["fy"] * z, z], 1)
    R = np.asarray(depth_to_color["rotation_column_major"], float).reshape(3, 3, order="F")
    p = p @ R.T + np.asarray(depth_to_color["translation_m"], float)
    return p @ cam.R.T + cam.t


def detect(points, area=AREA):
    """Plush toys among mount-frame points. Returns (plushes, rejected, info); x / y / z_top in m, yaw in rad
    mod 180 deg (the tram's long axis), head = [x, y] of the dog's head."""
    (x0, x1), (y0, y1) = area
    p = points[(points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)]
    flat = p[np.abs(p[:, 2]) < 0.02, 2]
    table_z = float(np.median(flat)) if len(flat) > 500 else 0.0
    nx, ny = int(math.ceil((x1 - x0) / GRID_M)), int(math.ceil((y1 - y0) / GRID_M))
    ix = np.clip(((p[:, 0] - x0) / GRID_M).astype(int), 0, nx - 1)
    iy = np.clip(((p[:, 1] - y0) / GRID_M).astype(int), 0, ny - 1)
    h = np.full((nx, ny), -1.0)
    np.maximum.at(h, (ix, iy), p[:, 2] - table_z)
    occ = (h > MIN_Z).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # cells without a depth sample
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    plushes, rejected = [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] * GRID_M ** 2 < 0.4 * SHORT_M[0] * LONG_M[0]:
            continue
        ci, cj = np.nonzero(labels == i)
        xy = np.stack([x0 + (ci + 0.5) * GRID_M, y0 + (cj + 0.5) * GRID_M], 1).astype(np.float32)
        hs = h[ci, cj]
        top = float(np.percentile(hs[hs > 0], 95)) if (hs > 0).any() else 0.0
        (cx, cy), (w, l), ang = cv2.minAreaRect(xy)
        long_, short = max(w, l) + GRID_M, min(w, l) + GRID_M
        why = None
        if top > TOP_M[1]:
            why = f"too tall ({top * 1000:.0f} mm): arm / hand"
        elif top < TOP_M[0]:
            why = f"too low ({top * 1000:.0f} mm)"
        elif not (LONG_M[0] <= long_ <= LONG_M[1] and SHORT_M[0] <= short <= SHORT_M[1]):
            why = f"footprint {long_ * 1000:.0f}x{short * 1000:.0f} mm"
        if why:
            rejected.append({"x": float(cx), "y": float(cy), "why": why})
            continue
        a = math.radians(ang) if w >= l else math.radians(ang) + math.pi / 2
        head = xy[hs > top - HEAD_BAND].mean(0)
        plushes.append({"color": NAME, "kind": NAME, "x": float(cx), "y": float(cy), "z_top": top + table_z,
                        "yaw": a % math.pi, "height_mm": round(top * 1000, 1),
                        "size_mm": [round(long_ * 1000, 1), round(short * 1000, 1)],
                        "head": [float(head[0]), float(head[1])], "upright": top > UPRIGHT_M, "source": "depth"})
    return plushes, rejected, {"table_z_mm": round(table_z * 1000, 1)}


def footprint(plush, margin=0.0):
    """Corners (4 x 2, m) of the plush footprint rectangle grown by margin."""
    c, s = math.cos(plush["yaw"]), math.sin(plush["yaw"])
    a, b = plush["size_mm"][0] / 2000 + margin, plush["size_mm"][1] / 2000 + margin
    return np.array([[plush["x"] + c * u - s * v, plush["y"] + s * u + c * v]
                     for u, v in ((a, b), (-a, b), (-a, -b), (a, -b))])


def inside(plush, x, y, margin=0.0) -> bool:
    c, s = math.cos(plush["yaw"]), math.sin(plush["yaw"])
    dx, dy = x - plush["x"], y - plush["y"]
    return abs(c * dx + s * dy) <= plush["size_mm"][0] / 2000 + margin and \
        abs(-s * dx + c * dy) <= plush["size_mm"][1] / 2000 + margin


def drop_blocks_on(blocks, plushes, margin=0.012):
    """Block detections that are really parts of a plush (tram edges, paws) removed."""
    return [b for b in blocks if not any(inside(p, b["x"], b["y"], margin) for p in plushes)]


def image_mask(shape, cam, plushes, margin=0.015):
    """Colour-image pixels covered by the plushes (footprint from the table up to their top): left out of the block
    detector's tray-occlusion check."""
    mask = np.zeros(shape[:2], np.uint8)
    for p in plushes:
        fp = footprint(p, margin)
        pts = [[x, y, z] for x, y in fp for z in (0.0, p["z_top"])]
        hull = cv2.convexHull(cam.project(pts).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def tray_of(plush, cam, quads):
    """Which tray quad (image polygons) the plush centre falls in, or None (on the table)."""
    u, v = cam.project([[plush["x"], plush["y"], 0.0]])[0]
    for tray, quad in (quads or {}).items():
        if cv2.pointPolygonTest(np.asarray(quad, np.float32), (float(u), float(v)), False) >= 0:
            return tray
    return None


def draw(img, cam, plushes):
    """Outline, head dot and a label for each plush on a colour image (in place)."""
    for p in plushes:
        base = cam.project([[x, y, 0.0] for x, y in footprint(p)]).astype(np.int32)
        roof = cam.project([[x, y, min(p["z_top"], BODY_M)] for x, y in footprint(p)]).astype(np.int32)
        cv2.polylines(img, [base], True, (60, 110, 30), 1, cv2.LINE_AA)
        cv2.polylines(img, [roof], True, (80, 200, 255), 2, cv2.LINE_AA)
        hu, hv = cam.project([[*p["head"], p["z_top"]]])[0]
        cv2.circle(img, (int(hu), int(hv)), 5, (90, 150, 200), -1, cv2.LINE_AA)
        c = roof.mean(0).astype(int)
        label = f"plush x{p['x'] * 1000:.0f} y{p['y'] * 1000:.0f} yaw{math.degrees(p['yaw']):.0f}" + \
            ("" if p["upright"] else " FALLEN")
        cv2.putText(img, label, (int(c[0]) - 60, int(c[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
        cv2.putText(img, label, (int(c[0]) - 60, int(c[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 200, 255), 1)
    return img


def scene(bgr, depth_m, cam, quads=None):
    """Blocks and plushes in one aligned frame: the plush first (heights), then the block detector without the
    plush's pixels. Returns (blocks, plushes, rejected, info) like agentic_sim.real_d455.detect, plus info["plush"]."""
    from agentic_sim.real_d455 import detect as detect_blocks
    plushes, plush_rejected, plush_info = detect(mount_points(depth_m, cam))
    for p in plushes:
        p["tray"] = tray_of(p, cam, quads)
    mask = image_mask(bgr.shape, cam, plushes) if plushes else None
    blocks, rejected, info = detect_blocks(bgr, depth_m, cam, quads, ignore=mask)
    return drop_blocks_on(blocks, plushes), plushes, rejected, info | {"plush": plush_info | {
        "rejected": plush_rejected}}


TRAY_FLOOR_M = 0.005           # beige tray floor above the table
RELEASE_M = 0.010              # bottom above the floor when the gripper opens
CLEAR_M = 0.083                # carried: bottom >= 6 cm over the beige rim (~23 mm)


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def slot(cam, quads):
    """The plush's place in the beige tray (P0, configs/real_plush.json): ((x, y) mount m, tray x-axis yaw)."""
    from agentic_sim.real_place import beige_slots
    return beige_slots(cam, quads, [config()["slot_beige_mm"]])[0]


def place_heights(grasp_z: float) -> dict:
    """plan_place heights for a plush taken with the grasp centre grasp_z (m) above the table: released with its
    bottom RELEASE_M over the tray floor, carried with its bottom CLEAR_M over the table."""
    from agentic_sim.real_place import CARRY_Z
    return {"place_z": grasp_z + TRAY_FLOOR_M + RELEASE_M, "carry_z": max(CARRY_Z, grasp_z + CLEAR_M)}


def delivered(plushes, slot_xy, verify_m=None) -> dict | None:
    """The plush seen in the beige tray within verify_mm of P0 (closest), or None."""
    verify_m = verify_m if verify_m is not None else config()["verify_mm"] / 1000
    near = [p for p in plushes if p.get("tray") == "beige"
            and math.hypot(p["x"] - slot_xy[0], p["y"] - slot_xy[1]) <= verify_m]
    return min(near, key=lambda p: math.hypot(p["x"] - slot_xy[0], p["y"] - slot_xy[1])) if near else None
