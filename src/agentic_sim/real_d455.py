"""D455 block localisation in the arm's mount frame (finals): colour + aligned depth + the camera pose.

Mount frame of the right follower: +X forward, +Y left, +Z up, origin on the table below the base, metres.
The camera pose (mount_from_color: R, t with columns = camera x / y / z axes) comes from
tools/finals_reach.py calib (arm depth vs SO-101 meshes + table plane), so a re-mounted camera only needs a new pose.

Per block: red / blue / yellow pixels (agentic_sim.real_locate colour masks). Their aligned depth gives the block top
height (only pixels within TOP_BAND of the highest part count as the top face, so side faces do not shift the
centre); those top pixels are cast onto the plane of the top face and a minimum-area rectangle gives the centre and the
yaw (mod 90 deg). A blob whose top is not at block height or whose rectangle is not block-sized is rejected (an orange
arm, a hand, a tray rim).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from agentic_sim.real_locate import COLORS, color_masks

EDGE_M = 0.0295
TOP_Z = (0.020, 0.075)       # block top height above the table (tray floor 4-6 mm + edge; a block on the table: 29.5)
TOP_BAND = 0.008
SIDE_M = (0.020, 0.042)      # accepted rectangle sides
MIN_PX = 40
LABEL = {"red": (40, 40, 230), "blue": (230, 120, 30), "yellow": (30, 210, 230)}


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


def detect(bgr, depth_m, cam: Camera, quads=None):
    """Blocks in one aligned colour / depth frame. depth_m: metres per colour pixel (0 = no depth)."""
    masks = color_masks(bgr)
    regions = {}
    for tray, quad in (quads or {}).items():
        r = np.zeros(bgr.shape[:2], np.uint8)
        cv2.fillPoly(r, [np.asarray(quad, np.int32)], 1)
        regions[tray] = r.astype(bool)
    blocks, rejected = [], []
    for color in COLORS:
        mask = cv2.morphologyEx(masks[color].astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_PX:
                continue
            vs, us = np.nonzero(labels == i)
            d = depth_m[vs, us]
            ok = d > 0.15
            if ok.sum() >= 10:
                p = cam.points(np.stack([us[ok], vs[ok]], 1), d[ok])
                top = float(np.percentile(p[:, 2], 90))
                keep = np.zeros(len(us), bool)
                keep[np.nonzero(ok)[0][p[:, 2] > top - TOP_BAND]] = True
                source = "depth"
            else:
                top, keep, source = 0.0335, np.ones(len(us), bool), "plane"
            why = None
            if not TOP_Z[0] <= top <= TOP_Z[1]:
                why = f"top {top * 1000:.0f} mm"
            uv = np.stack([us[keep], vs[keep]], 1)
            if why is None and len(uv) < 8:
                why = "few top pixels"
            if why is None:
                xy = cam.on_plane(uv, top)[:, :2].astype(np.float32)
                (cx, cy), (w, h), ang = cv2.minAreaRect(xy)
                if not (SIDE_M[0] <= min(w, h) and max(w, h) <= SIDE_M[1]):
                    why = f"size {w * 1000:.0f}x{h * 1000:.0f} mm"
            if why is not None:
                rejected.append({"color": color, "pixel": [float(c) for c in cents[i]], "why": why})
                continue
            yaw = math.radians(ang) % (math.pi / 2)
            tray = next((t for t, r in regions.items() if r[int(cents[i][1]), int(cents[i][0])]), None)
            blocks.append({"color": color, "x": float(cx), "y": float(cy), "z_top": top, "yaw": yaw,
                           "size_mm": [round(w * 1000, 1), round(h * 1000, 1)], "pixel": [float(c) for c in cents[i]],
                           "tray": tray, "source": source})
    blocks.sort(key=lambda b: (b["tray"] or "~", -b["x"], -b["y"]))
    return blocks, rejected


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
    for k, b in enumerate(blocks, 1):
        poly = cam.project(corners(b)).astype(np.int32)
        col = LABEL[b["color"]]
        cv2.polylines(img, [poly], True, (255, 255, 255) if k == selected else col, 3 if k == selected else 2)
        u, v = poly.max(0)[0] + 4, poly.min(0)[1]
        text = [f"{k} {b['color']}", f"x{b['x'] * 1000:.0f} y{b['y'] * 1000:.0f}", f"yaw {math.degrees(b['yaw']):.0f}"]
        if "turn" in b:
            text.append(f"wrist {math.degrees(b['turn']):+.0f}")
        for j, s in enumerate(text):
            cv2.putText(img, s, (int(u), int(v) + 13 * j + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(img, s, (int(u), int(v) + 13 * j + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
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
