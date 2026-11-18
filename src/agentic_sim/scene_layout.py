"""Scene geometry from a SceneSpec (configs/scenes/*.yaml), engine independent (pure Python).

Every backend builds the same table, trays and blocks from these numbers, so the scene is defined once:
  table_box(scene)          static table box (size, centre), top face at z = 0
  tray_parts(spec)          five boxes per open-top tray (floor + four walls) and the tray's interior
  block_layout(scene, n)    initial positions and colours of n blocks (or named colours, one per block)
  block_in_tray(pos, tray)  block resting inside a tray (acceptance checks, panel counts)
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
