"""Decluttering before localization: which block to move so the target can be localized and grasped.

Input is the global camera's view of the storage tray: one item per colour segment with its depth height check
(agentic_sim.perception.height). Rules, first match wins (target = the requested colour, then its global look-alikes,
nearest the arm first):

  target upright, colour not yet confirmed by the wrist  -> localize it (wrist look; no move spent on a look-alike)
  target upright and confirmed, nothing too close        -> localize it (final wrist refine, then ACT)
  target confirmed but crowded (or merged with its colour) -> move the target itself to a free spot (its wrist pose;
                                                            one move) if the fingers have room, else move a neighbour
  target the wrist could not confirm while crowded       -> move a neighbour, then look again
  target on top of a stack                              -> move the target itself down to a free spot
  target tilted                                         -> move the block it leans on (it then falls flat); none: move it
  target only partly visible under a stacked block      -> move the block on top
  target on an uneven support (off_layer)               -> move the target to a free spot
  no usable target, but a block that may hide one       -> move it (stacked first, then tilted / off_layer), nearest first
  nothing left                                          -> not_found

Every move is for the current target only; the scene is looked at again after each move. "Move" means pick and place
on a free spot in the storage tray, or a push when the pick fails (decided by the caller). Items that already failed
twice are skipped; a target the wrist could not confirm is excluded and the rules run again.
"""
from __future__ import annotations

import math

import numpy as np

from agentic_sim.perception.height import polygon_gap

IGNORED = ("side_face", "no_depth")
HIDERS = ("stacked", "tilted", "off_layer")


def items_from(colors: list, checks: list) -> list:
    """One item per usable segment: colour, height status and a pose read from the depth top face."""
    import cv2
    items = []
    for i, (color, c) in enumerate(zip(colors, checks)):
        if c["status"] in IGNORED or not c.get("top_hull_xy_m"):
            continue
        hull = np.asarray(c["top_hull_xy_m"], np.float32)
        (cx, cy), _, ang = cv2.minAreaRect(hull * 1000)          # mm: minAreaRect is unstable on metre values
        if c.get("pose_xy_m"):                                    # the tilted face inside a merged segment
            cx, cy = (1000 * x for x in c["pose_xy_m"])
        items.append({"segment": i, "color": color, "status": c["status"], "xy": [cx / 1000, cy / 1000],
                      "top_z_m": c["top_z_m"], "layers": c["layers"], "tilt_deg": c["tilt_deg"],
                      "yaw_rad": math.radians(ang) % (math.pi / 2), "hull": c["top_hull_xy_m"]})
    return items


def item_gap(a: dict, b: dict, margin_m: float) -> float:
    return max(0.0, polygon_gap(a["hull"], b["hull"]) - 2 * margin_m)


def item_key(item):
    """Identity of an item across looks: its xy rounded to 1 cm (it does not move unless moved)."""
    return (round(item["xy"][0], 2), round(item["xy"][1], 2))


def face_room(items: list, i: int, edge: float) -> float:
    """Most finger room beside item i over its two face pairs (fingers open along a face normal)."""
    y = float(items[i]["yaw_rad"])
    return max(finger_clearance(items, i, y + k * math.pi / 2, edge) for k in (0, 1))


def plan(items: list, target_color: str, cfg: dict, base_xy=(0.0, 0.0), failed: dict | None = None,
         excluded: set | None = None, confirmed: set | None = None, unconfirmed_crowded: set | None = None) -> dict:
    """Next step: {"action": "localize" | "move" | "not_found", "item": index or None, "target": index or None,
    "reason": text}. cfg: the declutter section plus close_gap_m / hull_margin_m / global_lookalikes.
    failed: item key -> failed moves; excluded: keys of targets the wrist could not confirm (with nothing close);
    confirmed: keys of targets the wrist confirmed (tried first); unconfirmed_crowded: keys of targets the wrist could
    not confirm while blocks crowded them (cleared, then looked at again). An upright target is confirmed by the wrist
    before any block is moved for it: the global camera mixes up some colours, and a look costs no move."""
    failed, excluded, confirmed = failed or {}, excluded or set(), confirmed or set()
    crowded_seen = unconfirmed_crowded or set()
    edge = float(cfg.get("edge_m", 0.0295))
    room_m = float(cfg.get("finger_room_m", 0.018))
    gap_m, margin = float(cfg["close_gap_m"]), float(cfg.get("hull_margin_m", 0.0))
    touch_m = float(cfg.get("support_gap_m", 0.005))
    looks = list(cfg.get("global_lookalikes", {}).get(target_color, []))
    base = np.asarray(base_xy, float)

    def dist(i):
        return float(np.linalg.norm(np.asarray(items[i]["xy"]) - base))

    def movable(i):
        return failed.get(item_key(items[i]), 0) < int(cfg.get("max_failures_per_item", 2))

    def move(i, reason, target=None):
        return {"action": "move", "item": i, "target": target, "reason": reason}

    targets = sorted((i for i, it in enumerate(items) if (it["color"] == target_color or it["color"] in looks)
                      and item_key(it) not in excluded),
                     key=lambda i: (item_key(items[i]) not in confirmed | crowded_seen, items[i]["color"] != target_color,
                                    dist(i)))
    for t in targets:
        it = items[t]
        others = [o for o in range(len(items)) if o != t]
        near = sorted(((item_gap(it, items[o], margin), o) for o in others), key=lambda g: g[0])
        st = it["status"]
        if st in ("upright", "merged"):
            key = item_key(it)
            crowded = any(g < gap_m for g, _ in near) or st == "merged"
            if key not in confirmed and key not in crowded_seen:
                return {"action": "localize", "item": None, "target": t, "reason": "check the colour with the wrist"}
            if not crowded:
                return {"action": "localize", "item": None, "target": t,
                        "reason": "target confirmed and clear" if key in confirmed else "cleared: look again"}
            if key in confirmed and movable(t) and (st == "merged" or face_room(items, t, edge) >= room_m):
                return move(t, "relocate the confirmed target to a free spot", t)   # one move, its wrist pose
            blockers = [o for g, o in near if g < gap_m and movable(o)]
            if blockers:
                return move(blockers[0], "neighbour too close to the target", t)
            continue                                       # crowded by blocks that cannot be moved: try another
        if st == "stacked" and movable(t):
            return move(t, "target on top of a stack: set it down", t)
        if st == "tilted":
            support = [o for g, o in near if g < touch_m and movable(o)]
            if support:
                return move(support[0], "target leans on this block", t)
            if movable(t):
                return move(t, "target tilted", t)
        if st == "partial":
            above = [o for g, o in near if g < touch_m and items[o]["layers"] > it["layers"] + 0.5 and movable(o)]
            if above:
                return move(above[0], "block on top of the target", t)
        if st == "off_layer" and movable(t):
            return move(t, "target on an uneven support", t)
    hiders = sorted((i for i, it in enumerate(items) if it["status"] in HIDERS and movable(i)),
                    key=lambda i: (HIDERS.index(items[i]["status"]), dist(i)))
    if hiders:
        return move(hiders[0], f"may hide the target ({items[hiders[0]]['status']})")
    return {"action": "not_found", "item": None, "target": None,
            "reason": "no usable block of the colour and nothing left that could hide one"}


def free_spot(items: list, tray: dict, edge: float, reachable, base_xy=(0.0, 0.0), avoid_xy=(),
              clearance_m: float = 0.025, avoid_m: float = 0.06, step_m: float = 0.01, ignore=(),
              reach_margin_m: float = 0.015, inset_m: float = 0.0):
    """A tray-floor spot for a block set down square to the mount frame: inside the tray interior (inset_m further
    from the walls), at least
    clearance_m from every other item's top, avoid_m from avoid_xy (the target), and graspable there and
    reach_margin_m further out (a set-down block lands a few mm off). Spots with twice the clearance first, then the
    one nearest the arm. None if there is none."""
    from agentic_sim.scene_layout import block_in_tray
    cx, cy = tray["center"]
    hx, hy = tray["inner_half"]
    c, s = math.cos(tray["yaw"]), math.sin(tray["yaw"])
    z = tray["floor_top"] + edge / 2
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * edge / 2
    others = [it["hull"] for k, it in enumerate(items) if k not in ignore]
    cands = []
    for lx in np.arange(-hx, hx + 1e-9, step_m):
        for ly in np.arange(-hy, hy + 1e-9, step_m):
            xy = np.array([cx + c * lx - s * ly, cy + s * lx + c * ly])
            if not block_in_tray([xy[0], xy[1], z], tray, edge) or \
                    max(abs(lx) - hx, abs(ly) - hy) > -edge / 2 - inset_m:
                continue
            if any(float(np.linalg.norm(xy - np.asarray(a[:2]))) < avoid_m for a in avoid_xy):
                continue
            gap = min((polygon_gap(sq + xy, h) for h in others), default=1.0)
            if gap < clearance_m:
                continue
            cands.append((gap < 2 * clearance_m, float(np.linalg.norm(xy - np.asarray(base_xy))), xy))
    for _, r, xy in sorted(cands, key=lambda c: (c[0], c[1])):
        out = xy + reach_margin_m * (xy - np.asarray(base_xy)) / max(r, 1e-9)
        if reachable([xy[0], xy[1], z], 0.0) and reachable([out[0], out[1], z], 0.0):
            return [float(xy[0]), float(xy[1])]
    return None


def push_path(items: list, i: int, tray: dict, edge: float, target: int | None = None, distance_m: float = 0.04,
              lead_m: float = 0.02, feasible=lambda start, end: True):
    """Straight push of item i: start behind it, end `distance_m` past its centre. Direction: away from the
    target (or from its nearest neighbour), turned in 45 degree steps until the end stays inside the tray and
    feasible(start_xy, end_xy) (e.g. IK along the path) holds."""
    from agentic_sim.scene_layout import block_in_tray
    p = np.asarray(items[i]["xy"], float)
    ref = None
    if target is not None and target != i:
        ref = np.asarray(items[target]["xy"], float)
    else:
        rest = [np.asarray(it["xy"], float) for k, it in enumerate(items) if k != i]
        if rest:
            ref = min(rest, key=lambda q: float(np.linalg.norm(q - p)))
    d = p - ref if ref is not None and np.linalg.norm(p - ref) > 1e-6 else np.array([-1.0, 0.0])
    a0 = math.atan2(d[1], d[0])
    z = tray["floor_top"] + edge / 2
    for k in (0, 1, -1, 2, -2, 3, -3, 4):
        a = a0 + k * math.pi / 4
        u = np.array([math.cos(a), math.sin(a)])
        end = p + distance_m * u
        start = p - (edge / 2 + lead_m) * u
        if block_in_tray([end[0], end[1], z], tray, edge) and feasible(start, end):
            return [float(x) for x in start], [float(x) for x in end]
    return None


def localize_candidate(item: dict, target_color: str, edge: float, floor_z: float, base_xy=(0.0, 0.0)) -> dict:
    """The rules' target as a localization candidate (fields of perception.localize.global_candidates): an upright
    cube on the tray floor at the item's depth pose. The wrist then confirms the colour and refines the pose."""
    from agentic_sim.perception.vertices import cube_vertices
    xy = np.asarray(item["xy"], float)
    return {"rank": 0 if item["color"] == target_color else 1, "global_color": item["color"],
            "global_detection_id": item["segment"], "source": "depth_height_check",
            "base_distance_m": float(np.linalg.norm(xy - np.asarray(base_xy, float))),
            "center_xyz_m": [float(xy[0]), float(xy[1]), floor_z + edge / 2],
            "yaw_rad_mod_pi_over_2": float(item["yaw_rad"]),
            "vertices_xyz_m": cube_vertices(xy, item["yaw_rad"], edge, floor_z).tolist(), "fit_iou": None}


def finger_clearance(items: list, i: int, finger_dir_rad: float, edge: float, corridor_m: float = 0.012) -> float:
    """Free distance beside item i along the finger opening line (both sides), where the open fingers come down:
    the smallest gap from the block's face to another top inside the finger corridor (about a finger wide)."""
    p = np.asarray(items[i]["xy"], float)
    u = np.array([math.cos(finger_dir_rad), math.sin(finger_dir_rad)])
    w = np.array([-u[1], u[0]])
    free = 1.0
    for k, it in enumerate(items):
        if k == i:
            continue
        h = np.asarray(it["hull"], float) - p
        along, across = h @ u, h @ w
        if across.min() > corridor_m or across.max() < -corridor_m:
            continue                                         # outside the finger corridor
        for side in (1, -1):
            a = side * along
            if a.max() > 0:
                free = min(free, max(0.0, float(a[a > 0].min() if (a > 0).any() else 0.0) - edge / 2))
    return free


def grasp_yaw_offset(items: list, i: int, base_offset: float, finger_dir, limit: float, edge: float,
                     feasible=lambda off: True, need_m: float = 0.018) -> float | None:
    """Finger orientation for picking item i: the smallest turn that lines the fingers up with the block faces
    (base_offset, within +-45 deg) unless its finger corridor has less than need_m of room and the other face pair
    (+-90 deg, within the yaw offset limit) has it. Large offsets tilt the gripper and drop blocks more often, so
    they are used only for room. Only orientations with feasible(offset) (IK at approach and grasp) are kept; None
    if there is none."""
    alts = [base_offset + s * math.pi / 2 for s in (1, -1) if abs(base_offset + s * math.pi / 2) <= limit]
    opts = [o for o in [base_offset] + alts if feasible(o)]
    if not opts:
        return None
    room = {o: finger_clearance(items, i, finger_dir(o), edge) for o in opts}
    if room[opts[0]] >= need_m or max(room.values()) < need_m:
        return opts[0]
    return max(opts[1:], key=lambda o: room[o]) if opts[0] == base_offset else opts[0]
