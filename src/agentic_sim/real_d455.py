"""D455 block localisation in the arm's mount frame (finals): aligned depth finds the blocks, colour names them.

Mount frame of the right follower: +X forward, +Y left, +Z up, origin on the table below the base, metres.
The camera pose (mount_from_color: R, t with columns = camera x / y / z axes) comes from
tools/finals_calib.py (arm depth vs SO-101 meshes + table plane), so a re-mounted camera only needs a new pose.

Per frame: the bare table height (median of the table pixels outside the trays) is measured and taken off, so a
few mm of depth bias do not matter. Pixels higher than TOP_MIN above the table (above both tray rims, below a block
on the table) form the candidates; each connected part keeps only its top face (within TOP_BAND of its highest part),
which is cast onto the top plane; a minimum-area rectangle gives the centre and the yaw (mod 90 deg). A part whose
rectangle is not block-sized is rejected (tray rim, a hand, the arm). The colour is the median hue / saturation /
value of the top face, named by the nearest entry of COLOR_REF (the finals palette: 8 colours); far from every entry
it is "unknown". A block without depth (holes over its top) is not reported.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

EDGE_M = 0.0295
TOP_MIN = 0.026              # above the beige rim (~23 mm), below a block lying on the table (29.5 mm)
TOP_MAX = 0.075
TOP_BAND = 0.008
SIDE_M = (0.020, 0.042)      # accepted rectangle sides
MIN_PX = 40
# OpenCV HSV (H 0-180) of the block tops seen by the D455; red / blue / yellow from the 11/25 lab frames, the
# others first guesses from the palette (tools/finals_reach.py preview prints the measured values to refine them)
COLOR_REF = {
    "red": (176, 200, 170), "orange": (10, 210, 220), "yellow": (25, 200, 220), "light_green": (38, 170, 190),
    "dark_green": (62, 170, 90), "cyan": (88, 170, 170), "blue": (102, 240, 170), "purple": (135, 140, 120),
}
COLOR_TOL = 1.0
LABEL = {"red": (40, 40, 230), "orange": (20, 130, 255), "yellow": (30, 210, 230), "light_green": (60, 220, 160),
         "dark_green": (40, 120, 30), "cyan": (200, 200, 40), "blue": (230, 120, 30), "purple": (180, 60, 140),
         "unknown": (128, 128, 128)}


def classify(hsv):
    """(name, distance) of the nearest reference colour; hue is circular, low-saturation pixels weigh hue less."""
    h, s, v = (float(x) for x in hsv)
    best = ("unknown", 1e9)
    for name, (rh, rs, rv) in COLOR_REF.items():
        dh = min(abs(h - rh), 180 - abs(h - rh)) / 12.0
        d = math.sqrt((dh * min(1.0, s / 90.0)) ** 2 + ((s - rs) / 90.0) ** 2 + ((v - rv) / 110.0) ** 2)
        if d < best[1]:
            best = (name, d)
    return best if best[1] <= COLOR_TOL else ("unknown", best[1])


class Camera:
    def __init__(self, intrinsics: dict, R, t):
        self.fx, self.fy, self.cx, self.cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
        self.R = np.asarray(R, float)
        self.t = np.asarray(t, float)

    def rays(self, uv):
        uv = np.asarray(uv, float).reshape(-1, 2)
        d = np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy, np.ones(len(uv))], 1)
        return d @ self.R.T

    def on_plane(self, uv, z):
        d = self.rays(uv)
        s = (z - self.t[2]) / d[:, 2]
        return self.t + s[:, None] * d

    def points(self, uv, depth_m):
        uv = np.asarray(uv, float).reshape(-1, 2)
        z = np.asarray(depth_m, float).reshape(-1)
        pc = np.stack([(uv[:, 0] - self.cx) / self.fx * z, (uv[:, 1] - self.cy) / self.fy * z, z], 1)
        return pc @ self.R.T + self.t

    def project(self, xyz):
        c = (np.asarray(xyz, float).reshape(-1, 3) - self.t) @ self.R
        return np.stack([self.fx * c[:, 0] / c[:, 2] + self.cx, self.fy * c[:, 1] / c[:, 2] + self.cy], 1)

    def height_map(self, depth_m):
        """Mount-frame z of every colour pixel (nan without depth)."""
        h, w = depth_m.shape
        if not hasattr(self, "_grid") or self._grid[0].shape != (h, w):
            vv, uu = np.mgrid[0:h, 0:w]
            self._grid = ((uu - self.cx) / self.fx, (vv - self.cy) / self.fy)
        gx, gy = self._grid
        z = depth_m
        Z = self.R[2, 0] * gx * z + self.R[2, 1] * gy * z + self.R[2, 2] * z + self.t[2]
        return np.where(depth_m > 0.15, Z, np.nan)


def edge_yaw(bgr_f, cam, centre_xyz, ref_hsv, radius_px=16):
    """Block yaw (rad, mod 90 deg) from the colour edges around the block: gradients of the distance to the block's
    own colour, angles folded by four (a square repeats every 90 deg), weighted by gradient strength, then turned
    from image angle to the mount frame on the top plane. None when the edges are too weak."""
    u0, v0 = cam.project([centre_xyz])[0]
    h, w = bgr_f.shape[:2]
    x0, x1 = int(max(u0 - radius_px, 1)), int(min(u0 + radius_px, w - 2))
    y0, y1 = int(max(v0 - radius_px, 1)), int(min(v0 + radius_px, h - 2))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    patch = bgr_f[y0 - 1:y1 + 1, x0 - 1:x1 + 1]
    hsv = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    dh = np.abs(hsv[..., 0] - ref_hsv[0])
    dh = np.minimum(dh, 180 - dh) / 12.0
    dist = np.sqrt(dh ** 2 + ((hsv[..., 1] - ref_hsv[1]) / 90.0) ** 2 + ((hsv[..., 2] - ref_hsv[2]) / 110.0) ** 2)
    dist = cv2.GaussianBlur(np.minimum(dist, 3.0).astype(np.float32), (3, 3), 0)
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    wgt = (gx ** 2 + gy ** 2) * (np.hypot(xx - u0, yy - v0) <= radius_px)
    if wgt.sum() < 1e-3:
        return None
    z = (wgt * np.exp(4j * np.arctan2(gy, gx))).sum()
    if abs(z) / wgt.sum() < 0.15:      # no dominant edge direction
        return None
    a = np.angle(z) / 4                 # image angle of an edge normal (mod 90 deg)
    p = cam.on_plane([[u0, v0], [u0 + 5 * math.cos(a), v0 + 5 * math.sin(a)]], centre_xyz[2])
    return math.atan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0]) % (math.pi / 2)


def _regions(shape, quads):
    out = {}
    for tray, quad in (quads or {}).items():
        r = np.zeros(shape, np.uint8)
        cv2.fillPoly(r, [np.asarray(quad, np.int32)], 1)
        out[tray] = r.astype(bool)
    return out


def _split_kmeans(xy, emit, max_k=4):
    """Touching same-colour blocks not in a row (offset, L, corner to corner): k-means the top-face points into
    k = 2..max_k clusters and accept the smallest k whose clusters are all block-sized. Returns True if split."""
    pts = np.ascontiguousarray(xy, np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-5)
    for k in range(2, max_k + 1):
        if len(pts) < 8 * k:
            break
        _, lab, _ = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        lab = lab.ravel()
        ok = True
        for j in range(k):
            sel = pts[lab == j]
            if len(sel) < 6:
                ok = False
                break
            (_, _), (w, h), _ = cv2.minAreaRect(sel)
            if not (SIDE_M[0] - 0.004 <= min(w, h) and max(w, h) <= SIDE_M[1]):
                ok = False
                break
        if ok:
            for j in range(k):
                emit(lab == j)
            return True
    return False


def detect(bgr, depth_m, cam: Camera, quads=None):
    """Blocks in one aligned colour / depth frame (depth_m: metres per colour pixel, 0 = none).
    Returns (blocks, rejected, info); block x / y / z_top in metres (mount frame), yaw in rad mod 90 deg."""
    H = cam.height_map(depth_m)
    regions = _regions(bgr.shape[:2], quads)
    trays = np.zeros(bgr.shape[:2], bool)
    for r in regions.values():
        trays |= r
    table = H[(~trays) & (np.abs(H) < 0.02)]
    table_z = float(np.median(table)) if table.size > 500 else 0.0
    rel = H - table_z
    cand = np.nan_to_num(rel, nan=0.0)
    cand = ((cand > TOP_MIN) & (cand < TOP_MAX)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bgr_f = bgr.astype(np.float32)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand)
    names = list(COLOR_REF)
    refs = np.array([COLOR_REF[c] for c in names], float)
    blocks, rejected = [], []

    def add(us, vs, top, name, dist, med, split=True):
        uv = np.stack([us, vs], 1)
        xy = cam.on_plane(uv, top + table_z)[:, :2].astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(xy)
        long_, short = max(w, h), min(w, h)
        k = int(round(long_ / EDGE_M)) if SIDE_M[0] <= short <= SIDE_M[1] else 1
        if k >= 2 and abs(long_ - k * EDGE_M) < 0.012:      # same-colour blocks touching in a row: split evenly
            axis = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))]) if w >= h else \
                np.array([-math.sin(math.radians(ang)), math.cos(math.radians(ang))])
            proj = (xy - [cx, cy]) @ axis
            edges = np.linspace(proj.min(), proj.max() + 1e-9, k + 1)
            for j in range(k):
                sel = (proj >= edges[j]) & (proj < edges[j + 1])
                if sel.sum() >= 6:
                    add(us[sel], vs[sel], top, name, dist, med, split=False)
            return
        if not (SIDE_M[0] <= short and long_ <= SIDE_M[1]):
            if short < 0.012:          # rim slivers and edges: not worth listing
                return
            if split and _split_kmeans(xy, lambda sel: add(us[sel], vs[sel], top, name, dist, med, split=False)):
                return                 # blocks touching at an offset / corner: one cluster per block
            rejected.append({"color": name, "pixel": [float(np.mean(us)), float(np.mean(vs))],
                             "why": f"size {w * 1000:.0f}x{h * 1000:.0f} mm"})
            return
        u0, v0 = cam.project([[cx, cy, top + table_z]])[0]
        yaw_e = edge_yaw(bgr_f, cam, [cx, cy, top + table_z], med)
        yaw = yaw_e if yaw_e is not None else math.radians(ang) % (math.pi / 2)
        tray = next((t for t, r in regions.items() if r[min(max(int(v0), 0), r.shape[0] - 1),
                                                        min(max(int(u0), 0), r.shape[1] - 1)]), None)
        blocks.append({"color": name, "color_distance": round(dist, 2), "hsv": [int(x) for x in med],
                       "x": float(cx), "y": float(cy), "z_top": top + table_z, "height_mm": round(top * 1000, 1),
                       "yaw": yaw, "yaw_source": "edges" if yaw_e is not None else "rectangle", "size_mm": [round(w * 1000, 1), round(h * 1000, 1)],
                       "pixel": [float(u0), float(v0)], "tray": tray, "source": "depth"})

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_PX:
            continue
        vs, us = np.nonzero(labels == i)
        z = rel[vs, us]
        top = float(np.nanpercentile(z, 90))
        keep = z > top - TOP_BAND
        us, vs = us[keep], vs[keep]
        if len(us) < 8:
            rejected.append({"color": "?", "pixel": [float(c) for c in cents[i]], "why": "few top pixels"})
            continue
        px = hsv[vs, us].astype(float)
        dh = np.abs(px[:, None, 0] - refs[None, :, 0])
        dh = np.minimum(dh, 180 - dh) / 12.0 * np.minimum(1.0, px[:, None, 1] / 90.0)
        dd = dh ** 2 + ((px[:, None, 1] - refs[None, :, 1]) / 90.0) ** 2 + ((px[:, None, 2] - refs[None, :, 2]) / 110.0) ** 2
        cls = dd.argmin(1)
        for c in np.unique(cls):            # touching blocks of different colours: one part per colour
            sel = cls == c
            if sel.sum() < 12:
                continue
            part = np.zeros(hsv.shape[:2], np.uint8)
            part[vs[sel], us[sel]] = 1
            part = cv2.morphologyEx(part, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            m, lab, st, _ = cv2.connectedComponentsWithStats(part)
            for j in range(1, m):
                if st[j, cv2.CC_STAT_AREA] < 12:
                    continue
                pv, pu = np.nonzero(lab == j)
                med = np.median(hsv[pv, pu], 0)
                name, dist = classify(med)
                add(pu, pv, top, name, dist, med)
    blocks.sort(key=lambda b: (b["tray"] or "~", -b["x"], -b["y"]))
    return blocks, rejected, {"table_z_mm": round(table_z * 1000, 1)}


def corners(block):
    c, s = math.cos(block["yaw"]), math.sin(block["yaw"])
    h = EDGE_M / 2
    return np.array([[block["x"] + c * a - s * b, block["y"] + s * a + c * b, block["z_top"]]
                     for a, b in ((h, h), (-h, h), (-h, -h), (h, -h))])


def draw(bgr, cam: Camera, blocks, *, selected=None, quads=None, rejected=(), arm=None, lines=()):
    img = bgr.copy()
    for quad in (quads or {}).values():
        cv2.polylines(img, [np.asarray(quad, np.int32)], True, (90, 200, 90), 1)
    o = cam.project([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
    cv2.arrowedLine(img, tuple(o[0].astype(int)), tuple(o[1].astype(int)), (0, 0, 255), 2, tipLength=0.15)
    cv2.arrowedLine(img, tuple(o[0].astype(int)), tuple(o[2].astype(int)), (0, 255, 0), 2, tipLength=0.15)
    cv2.putText(img, "+X", tuple(o[1].astype(int) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(img, "+Y", tuple(o[2].astype(int) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    for r in rejected:
        cv2.drawMarker(img, tuple(int(c) for c in r["pixel"]), (128, 128, 128), cv2.MARKER_TILTED_CROSS, 8, 1)
    rows = []
    for k, b in enumerate(blocks, 1):
        poly = cam.project(corners(b)).astype(np.int32)
        col = LABEL.get(b["color"], (128, 128, 128))
        cv2.polylines(img, [poly], True, (255, 255, 255) if k == selected else col, 3 if k == selected else 2)
        c = poly.mean(0).astype(int)
        cv2.putText(img, str(k), (int(c[0]) - 4, int(c[1]) - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, str(k), (int(c[0]) - 4, int(c[1]) - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        turn = f" wrist{math.degrees(b['turn']):+.0f}" if "turn" in b else ""
        rows.append((f"{k} {b['color']}{'?' if b.get('color_distance', 0) > 0.6 else ''}  x{b['x'] * 1000:.0f} "
                     f"y{b['y'] * 1000:.0f}  yaw{math.degrees(b['yaw']):.0f}{turn}", col))
    lines = list(lines)
    y0 = 18 + 16 * len(lines)
    if rows:
        cv2.rectangle(img, (4, y0 - 13), (250, y0 + 16 * len(rows) - 10), (0, 0, 0), -1)
    for j, (s, col) in enumerate(rows):
        cv2.putText(img, s, (8, y0 + 16 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    if arm is not None:
        p = cam.project([arm])[0]
        cv2.drawMarker(img, tuple(p.astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
    for j, s in enumerate(lines):
        cv2.putText(img, s, (8, 18 + 16 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, s, (8, 18 + 16 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def open_aligned(serial: str, width=640, height=480, fps=30):
    """Context-free D455 opener: returns (read, intrinsics, stop). read() -> (bgr, depth_m aligned to colour)."""
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(str(serial))
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    prof = pipe.start(cfg)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)
    spatial, temporal, holes = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()

    def read():
        fs = align.process(pipe.wait_for_frames(1000))
        c, d = fs.get_color_frame(), fs.get_depth_frame()
        d = holes.process(temporal.process(spatial.process(d)))
        return np.asanyarray(c.get_data()).copy(), np.asanyarray(d.get_data()).astype(np.float32) * scale

    return read, {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy, "width": intr.width,
                  "height": intr.height}, pipe.stop
