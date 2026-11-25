"""Coarse block positions from one global D455 colour frame, in tray-local mm.

Tray-local frame (same as the task cards): origin at the inner tray corner nearest the arm base on its right,
+x forward (away from the base), +y left. In the global image the bases are at the bottom, so +x points up the
image and +y to the left; the origin is the tray's bottom-right inner corner.

Each tray is found by colour, its outline reduced to four corners and mapped to the measured outer size with a
homography (outer = inner + 10 mm). Blocks are red / blue / yellow blobs inside a tray. The centroid is the block
top seen from the camera, so a ~25 mm tall block reads a few mm off its footprint (parallax, not corrected): use
these values as the coarse position the policy refines, not as ground truth.
"""

import cv2
import numpy as np

TRAYS = {"pink": {"outer_mm": (305.0, 230.0)}, "beige": {"outer_mm": (365.0, 265.0)}}  # (x forward, y left)
WALL_MM = 5.0
COLORS = ("red", "blue", "yellow")
MIN_BLOCK_PX = 60


def _hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)


def color_masks(image):
    h, s, v = _hsv(image)
    pink = ((h >= 150) | (h <= 8)) & (s >= 45) & (s < 150) & (v >= 120)
    return {
        "pink": pink,
        "beige": (h >= 8) & (h <= 32) & (s >= 16) & (s <= 90) & (v >= 95) & ~pink,
        "red": ((h <= 8) | (h >= 165)) & (s >= 150) & (v >= 60),
        "blue": (h >= 90) & (h <= 130) & (s >= 120) & (v >= 60),
        "yellow": (h >= 15) & (h <= 35) & (s >= 120) & (v >= 100),
    }


def _largest(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < 2:
        return None
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def tray_region(mask, blocks):
    """Filled tray (floor, rim and the blocks lying on it) and its convex hull."""
    core = _largest(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool))
    if core is None:
        return None, None
    closed = cv2.morphologyEx((core | blocks).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, [hull], -1, 1, -1)
    return filled.astype(bool), hull


def _extreme_corners(points):
    s, d = points.sum(1), points[:, 0] - points[:, 1]
    return np.array([points[np.argmax(s)], points[np.argmax(d)], points[np.argmin(s)], points[np.argmin(d)]],
                    np.float32)  # bottom-right, top-right, top-left, bottom-left


def corners(region, near_px=6.0, end_share=0.2):
    """Four tray corners in image px ordered origin (bottom-right), +x end (top-right), top-left, +y end
    (bottom-left). Each side is a line fitted to the outline points lying on it, away from the rounded corners;
    points pulled in by an occluding arm or out by a thin leak are dropped, so one bad spot cannot drag a corner."""
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    rough = _extreme_corners(cv2.convexHull(outline).reshape(-1, 2).astype(np.float32))
    lines = []
    for a, b in zip(rough, np.roll(rough, -1, axis=0)):  # right, top, left, bottom sides
        direction = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        normal = np.array([-direction[1], direction[0]])
        rel = outline - a
        along = rel @ direction / max(np.linalg.norm(b - a), 1e-6)
        on_side = (np.abs(rel @ normal) <= near_px) & (along > end_share) & (along < 1 - end_share)
        if on_side.sum() < 10:
            return rough
        vx, vy, x0, y0 = cv2.fitLine(outline[on_side], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    result = []
    for (p1, d1), (p2, d2) in zip(np.roll(np.array(lines, dtype=object), 1, axis=0), lines):
        # corner i joins side i-1 and side i (corner 0 = bottom-right = bottom side x right side)
        m = np.array([d1, -d2], dtype=float).T
        if abs(np.linalg.det(m)) < 1e-6:
            return rough
        t = np.linalg.solve(m, (p2 - p1).astype(float))
        result.append(p1 + t[0] * d1)
    return np.array(result, np.float32)


def tray_homography(quad, tray):
    lx, ly = TRAYS[tray]["outer_mm"]
    outer = np.array([[0, 0], [lx, 0], [lx, ly], [0, ly]], np.float32) - WALL_MM  # tray-local (inner origin)
    return cv2.getPerspectiveTransform(quad, outer)


def _to_tray(H, points):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def locate(image, quads=None):
    """Trays and blocks in one BGR frame. Returns {"trays": {...}, "blocks": [...], "warnings": [...]}.

    quads: fixed tray corners {tray: [[x, y] * 4]} from a clean reference frame (trays do not move during a
    session). With them the tray outlines are not re-detected, so a hand or the arm over a tray cannot shift the
    tray-local frame; only the blocks are found in this frame."""
    masks = color_masks(image)
    any_block = masks["red"] | masks["blue"] | masks["yellow"]
    height, width = image.shape[:2]
    trays, warnings = {}, []
    if quads:
        for tray, quad in quads.items():
            quad = np.array(quad, np.float32)
            region = np.zeros((height, width), np.uint8)
            cv2.fillPoly(region, [quad.astype(np.int32)], 1)
            clipped = bool((quad[:, 0] <= 1).any() or (quad[:, 1] <= 1).any() or (quad[:, 0] >= width - 2).any()
                           or (quad[:, 1] >= height - 2).any())
            trays[tray] = {"region": region.astype(bool), "quad_px": quad, "H": tray_homography(quad, tray),
                           "clipped": clipped}
    taken = np.zeros(masks["pink"].shape, bool)
    for tray in [] if quads else TRAYS:  # pink first: its rim shades into beige, so beige excludes the pink tray
        region, hull = tray_region(masks[tray] & ~taken, any_block)
        if region is None or region.sum() < 2000:
            warnings.append(f"{tray} tray not found")
            continue
        quad = corners(region)
        x0, y0, w, h = cv2.boundingRect(hull)
        clipped = x0 <= 1 or y0 <= 1 or x0 + w >= width - 1 or y0 + h >= height - 1
        if clipped:
            warnings.append(f"{tray} tray touches the image border; its corners and positions are unreliable")
        trays[tray] = {"region": region, "quad_px": quad, "H": tray_homography(quad, tray), "clipped": clipped}
        taken |= cv2.dilate(region.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    blocks = []
    for color in COLORS:
        mask = cv2.morphologyEx(masks[color].astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_BLOCK_PX:
                continue
            blob = labels == i
            inside = [t for t, info in trays.items() if info["region"][blob].mean() > .5]
            if not inside:
                continue
            tray = inside[0]
            H = trays[tray]["H"]
            ys, xs = np.nonzero(blob)
            rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            box = _to_tray(H, cv2.boxPoints(rect))
            edge = box[1] - box[0]
            yaw = float(np.degrees(np.arctan2(edge[1], edge[0])) % 90.0)
            x, y = _to_tray(H, [centroids[i]])[0]
            blocks.append({"color": color, "tray": tray, "x_mm": round(float(x), 1), "y_mm": round(float(y), 1),
                           "yaw_deg": round(yaw, 1), "pixel": [round(float(c), 1) for c in centroids[i]],
                           "area_px": int(stats[i, cv2.CC_STAT_AREA]),
                           "tray_clipped": trays[tray]["clipped"]})
    public = {t: {"quad_px": info["quad_px"].round(1).tolist(), "clipped": info["clipped"]} for t, info in trays.items()}
    return {"trays": public, "blocks": blocks, "warnings": warnings,
            "frame": "tray-local mm: origin inner corner nearest the base on its right, +x forward, +y left",
            "method": "colour segmentation + tray-outline homography (block-top parallax uncorrected)",
            "tray_source": "fixed reference quads" if quads else "detected in this frame"}


def reference_quads(image, require=tuple(TRAYS)):
    """Tray corners from a clean frame (no hands or arm over the trays), to reuse for the whole session."""
    result = locate(image)
    missing = [t for t in require if t not in result["trays"]]
    if missing or any(result["trays"][t]["clipped"] for t in require):
        raise ValueError("; ".join(result["warnings"]) or f"trays missing: {missing}")
    return {tray: info["quad_px"] for tray, info in result["trays"].items()}


def draw(image, result):
    shown = image.copy()
    for tray, info in result["trays"].items():
        quad = np.array(info["quad_px"], np.int32)
        cv2.polylines(shown, [quad], True, (0, 200, 0) if not info["clipped"] else (0, 0, 255), 2)
        cv2.circle(shown, tuple(quad[0]), 5, (0, 0, 255), -1)  # tray-local origin
    for block in result["blocks"]:
        px = tuple(int(v) for v in block["pixel"])
        cv2.circle(shown, px, 4, (255, 255, 255), -1)
        cv2.putText(shown, f"{block['x_mm']:.0f},{block['y_mm']:.0f} {block['yaw_deg']:.0f}d", (px[0] + 6, px[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 20, 20), 1, cv2.LINE_AA)
    return shown


def match_target(result, card):
    """The detected block for a task card: same colour and source tray, nearest to the card pose."""
    pose = card.get("actual_pose") or card["pose"]
    same = [b for b in result["blocks"] if b["color"] == card["color"] and b["tray"] == card["source_tray"]]
    if not same:
        return None
    best = min(same, key=lambda b: (b["x_mm"] - pose[0]) ** 2 + (b["y_mm"] - pose[1]) ** 2)
    return best | {"card_offset_mm": round(float(np.hypot(best["x_mm"] - pose[0], best["y_mm"] - pose[1])), 1),
                   "same_colour_in_tray": len(same)}
