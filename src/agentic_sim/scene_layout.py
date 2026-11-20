"""Scene geometry from a SceneSpec (configs/scenes/*.yaml), engine independent (pure Python).

Every backend builds the same table, trays and blocks from these numbers, so the scene is defined once:
  table_box(scene)          static table box (size, centre), top face at z = 0
  tray_parts(spec)          five boxes per open-top tray (floor + four walls) and the tray's interior
  block_layout(scene, n)    initial positions and colours of n blocks (or named colours, one per block)
  block_in_tray(pos, tray)  block resting inside a tray (acceptance checks, panel counts)
  sensor_cameras(scene)     global (fixed) and wrist (gripper) sensor cameras of lab scenes
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agentic_sim.config import v

TRAYS = ("storage_pink", "placement_beige")


def yaw_quat(deg: float) -> tuple:
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def table_box(scene: dict) -> dict:
    tb = scene["table"]
    sx, sy, sz = v(tb["size"])
    cx, cy = v(tb["center_xy"])
    return {"size": (sx, sy, sz), "pos": (cx, cy, -sz / 2), "color": tuple(tb["color"]), "friction": v(tb["friction"])}


def tray_parts(spec: dict) -> tuple:
    """(parts, info). parts: name -> (size xyz, centre xyz) in the mount frame, all rotated by info['yaw'] about Z.
    The tray is five boxes so the interior stays hollow (no convex hull over the opening)."""
    L, W = v(spec["outer_size"])
    d, t, f = v(spec["depth"]), v(spec["wall_thickness"]), v(spec["floor_thickness"])
    cx, cy = v(spec["center_xy"])
    yaw = math.radians(v(spec["yaw_deg"]))
    h = d + f
    c, s = math.cos(yaw), math.sin(yaw)

    def place(lx, ly, z):
        return (cx + c * lx - s * ly, cy + s * lx + c * ly, z)

    local = {
        "floor": ((L, W, f), (0.0, 0.0), f / 2),
        "wall_front": ((t, W, h), (L / 2 - t / 2, 0.0), h / 2),
        "wall_back": ((t, W, h), (-L / 2 + t / 2, 0.0), h / 2),
        "wall_left": ((L - 2 * t, t, h), (0.0, W / 2 - t / 2), h / 2),
        "wall_right": ((L - 2 * t, t, h), (0.0, -W / 2 + t / 2), h / 2),
    }
    parts = {name: (size, place(lx, ly, z)) for name, (size, (lx, ly), z) in local.items()}
    info = {"center": (cx, cy), "inner_half": ((L - 2 * t) / 2, (W - 2 * t) / 2), "yaw": yaw, "floor_top": f,
            "yaw_deg": v(spec["yaw_deg"]), "color": tuple(spec["color"])}
    return parts, info


def trays(scene: dict) -> dict:
    """Tray interiors (center, inner_half, yaw, floor_top) by name."""
    return {name: tray_parts(scene["trays"][name])[1] for name in TRAYS}


def block_count_range(scene: dict) -> tuple:
    return 1, int(scene["block"].get("max_count", 1))


@dataclass
class BlockLayout:
    edge: float
    mass: float
    friction: tuple
    init_pos: list          # (x, y, z) per block, resting on the storage tray floor
    init_rot: tuple         # wxyz, shared
    display_rgb: list       # sRGB-ish design colour per block (panels, MuJoCo rgba)
    linear_rgb: list        # linear colour per block (renderer materials that expect linear values)


def block_layout(scene: dict, n_blocks: int | None = None, colors: list | None = None) -> BlockLayout:
    """Block 0 at block.init_xy with the scene colour; blocks 1..n-1 at block.layout_xy[i] with palette colours.

    colors: palette colour names, one per block (task scenes, e.g. [red, red, blue]); sets the block count and
    replaces the default colours. Positions are the same layout slots."""
    bl = scene["block"]
    lo, hi = block_count_range(scene)
    n = int(bl.get("count", 1) if n_blocks is None else n_blocks)
    if colors:
        n = len(colors)
    if not lo <= n <= hi:
        raise ValueError(f"block count {n} outside {lo}..{hi}")
    e = v(bl["edge"])
    floor_top = tray_parts(scene["trays"]["storage_pink"])[1]["floor_top"]
    layout = [tuple(v(bl["init_xy"]))] + [tuple(xy) for xy in (v(bl["layout_xy"])[1:] if "layout_xy" in bl else [])]
    if len(layout) < n:
        raise ValueError(f"scene block.layout_xy has {len(layout)} positions, {n} blocks requested")
    linear, display = [tuple(bl["color"])], [tuple(bl["color"])]
    if colors:
        from agentic_sim.color_dr import srgb_to_linear
        by_name = palette_by_name(bl["extra_palette"])
        unknown = [c for c in colors if c not in by_name]
        if unknown:
            raise ValueError(f"unknown block colour(s) {unknown}; palette has {sorted(by_name)}")
        display = [tuple(by_name[c]) for c in colors]
        linear = [tuple(srgb_to_linear(rgb)) for rgb in display]
    elif n > 1:
        from agentic_sim.color_dr import Palette, srgb_to_linear
        pal = Palette.load(bl["extra_palette"])
        for i in range(1, n):
            rgb = pal.colors[i % len(pal.colors)].rgb
            linear.append(tuple(srgb_to_linear(rgb)))
            display.append(tuple(rgb))
    inits = [(x, y, floor_top + e / 2 + 0.0005) for x, y in layout[:n]]
    return BlockLayout(e, v(bl["mass"]), tuple(v(bl["friction"])), inits, yaw_quat(v(bl["init_yaw_deg"])),
                       display, linear)


def palette_by_name(path: str) -> dict:
    """Palette colour name -> display rgb."""
    from agentic_sim.color_dr import Palette
    return {c.name: tuple(c.rgb) for c in Palette.load(path).colors}


def block_in_tray(pos, tray: dict, edge: float) -> bool:
    """Block centre inside the tray interior (with half an edge margin) and resting near the floor."""
    cx, cy = tray["center"]
    c, s = math.cos(-tray["yaw"]), math.sin(-tray["yaw"])
    lx = c * (pos[0] - cx) - s * (pos[1] - cy)
    ly = s * (pos[0] - cx) + c * (pos[1] - cy)
    hx, hy = tray["inner_half"]
    return abs(lx) <= hx - edge / 2 and abs(ly) <= hy - edge / 2 and pos[2] < tray["floor_top"] + edge


def wrist_camera_mount(scene: dict) -> bool:
    """The arm carries the SO-101 wrist camera mount (lab scenes); lab_current_minimal keeps the no-camera arm."""
    return bool(scene.get("robot", {}).get("wrist_camera_mount", False))


def global_camera_quat(cam: dict) -> tuple:
    """wxyz of the global camera in the USD / MuJoCo camera convention (-Z forward, +Y up) from its OpenCV axes."""
    from agentic_sim.kinematics import mat_to_quat_wxyz
    import numpy as np
    m = np.array(cam["rot_cv"], float) @ np.diag([1.0, -1.0, -1.0])
    return tuple(float(x) for x in mat_to_quat_wxyz(m))


def sensor_cameras(scene: dict) -> dict:
    """Sensor cameras of the scene (scene['sensor_cameras']), engine independent.

    global: fixed in the mount frame. pos, rot_cv (3x3, columns = OpenCV camera x right / y down / z forward in the
            mount frame), width, height, fx, fy, cx, cy, fovy_deg (vertical, from fy).
    wrist:  attached to the gripper link through the camera mount; width, height, fovy_deg, fx, fy, cx, cy,
            pos_in_gripper / quat_in_gripper (wxyz; camera looks along -Z with +Y up, USD and MuJoCo convention).
    Scenes without the section have no sensor cameras."""
    out = {}
    sc = scene.get("sensor_cameras") or {}
    g = sc.get("global")
    if g:
        w, h = (int(x) for x in v(g["resolution"]))
        k = v(g["intrinsics"])
        out["global"] = {"pos": tuple(float(x) for x in v(g["pos"])),
                         "rot_cv": [[float(x) for x in row] for row in v(g["rot"])],
                         "width": w, "height": h, "fx": float(k["fx"]), "fy": float(k["fy"]),
                         "cx": float(k["cx"]), "cy": float(k["cy"]),
                         "fovy_deg": math.degrees(2 * math.atan(h / 2 / float(k["fy"])))}
    wr = sc.get("wrist")
    if wr:
        w, h = (int(x) for x in v(wr["resolution"]))
        fovy = float(v(wr["fovy_deg"]))
        f = h / 2 / math.tan(math.radians(fovy) / 2)
        pose = wr.get("pose_in_gripper") or {}
        out["wrist"] = {"width": w, "height": h, "fovy_deg": fovy, "fx": f, "fy": f, "cx": w / 2, "cy": h / 2,
                        "pos_in_gripper": tuple(float(x) for x in v(pose["pos"])) if pose else None,
                        "quat_in_gripper": tuple(float(x) for x in v(pose["quat_wxyz"])) if pose else None}
    return out
