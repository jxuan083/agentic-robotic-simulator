"""Build the minimal lab_current scene: table, two trays (open-top colliders), 1-8 blocks, SO-101, lights, axes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg

from agentic_sim.config import v
from agentic_sim.isaac.so101 import so101_cfg

# Mount frame (+X forward) = robot base link rotated +90 deg about Z (see agentic_sim.kinematics.MOUNT_FROM_BASE).
ROBOT_ROT_WXYZ = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))


def _yaw_quat(deg: float):
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def _box(path, size, pos, color, friction=None, collide=True, rot=(1.0, 0.0, 0.0, 0.0)):
    cfg = sim_utils.CuboidCfg(
        size=tuple(size),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0) if collide else None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.6),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=friction[0], dynamic_friction=friction[1])
        if (collide and friction) else None,
    )
    cfg.func(path, cfg, translation=tuple(pos), orientation=rot)


def _tray(root, spec, friction):
    """Static open-top tray from five boxes so the interior stays hollow (no convex hull over the opening)."""
    L, W = v(spec["outer_size"])
    d, t, f = v(spec["depth"]), v(spec["wall_thickness"]), v(spec["floor_thickness"])
    cx, cy = v(spec["center_xy"])
    yaw = math.radians(v(spec["yaw_deg"]))
    h = d + f
    color = spec["color"]
    c, s = math.cos(yaw), math.sin(yaw)

    def place(lx, ly):
        return (cx + c * lx - s * ly, cy + s * lx + c * ly)

    rot = _yaw_quat(v(spec["yaw_deg"]))
    parts = {
        "floor": ((L, W, f), (0.0, 0.0), f / 2),
        "wall_front": ((t, W, h), (L / 2 - t / 2, 0.0), h / 2),
        "wall_back": ((t, W, h), (-L / 2 + t / 2, 0.0), h / 2),
        "wall_left": ((L - 2 * t, t, h), (0.0, W / 2 - t / 2), h / 2),
        "wall_right": ((L - 2 * t, t, h), (0.0, -W / 2 + t / 2), h / 2),
    }
    for name, (size, (lx, ly), z) in parts.items():
        x, y = place(lx, ly)
        _box(f"{root}/{name}", size, (x, y, z), color, friction, rot=rot)
    return {"center": (cx, cy), "inner_half": ((L - 2 * t) / 2, (W - 2 * t) / 2), "yaw": yaw, "floor_top": f}


@dataclass
class SceneHandles:
    robot: Articulation
    blocks: list            # RigidObject per block; blocks[0] is /World/Block
    block_prims: list
    block_init_pos: list    # (x, y, z) per block
    block_init_rot: tuple
    block_colors: list      # baseline display colour (sRGB-ish) per block, for the panel
    trays: dict

    @property
    def block(self) -> RigidObject:
        return self.blocks[0]


def block_prim(i: int) -> str:
    return "/World/Block" if i == 0 else f"/World/Block_{i}"


def block_count_range(scene: dict) -> tuple:
    bl = scene["block"]
    return 1, int(bl.get("max_count", 1))


def build_scene(scene: dict, n_blocks: int | None = None) -> SceneHandles:
    tb = scene["table"]
    tsx, tsy, tsz = v(tb["size"])
    tcx, tcy = v(tb["center_xy"])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg(), translation=(0.0, 0.0, -tsz))
    _box("/World/Table", (tsx, tsy, tsz), (tcx, tcy, -tsz / 2), tb["color"], v(tb["friction"]))

    trays = {}
    for name in ("storage_pink", "placement_beige"):
        trays[name] = _tray(f"/World/Trays/{name}", scene["trays"][name], v(scene["trays"]["friction"]))

    bl = scene["block"]
    lo, hi = block_count_range(scene)
    n = int(bl.get("count", 1) if n_blocks is None else n_blocks)
    if not lo <= n <= hi:
        raise ValueError(f"block count {n} outside {lo}..{hi}")
    e = v(bl["edge"])
    floor_top = trays["storage_pink"]["floor_top"]
    layout = [tuple(v(bl["init_xy"]))] + [tuple(xy) for xy in (v(bl["layout_xy"])[1:] if "layout_xy" in bl else [])]
    if len(layout) < n:
        raise ValueError(f"scene block.layout_xy has {len(layout)} positions, {n} blocks requested")
    # material colours (linear, written to the shader) and display colours (panel) per block; block 0 keeps the
    # scene colour as before, the others take palette colours (sRGB design values -> linear, like colour DR)
    colors, display = [tuple(bl["color"])], [tuple(bl["color"])]
    if n > 1:
        from agentic_sim.color_dr import Palette, srgb_to_linear
        pal = Palette.load(bl["extra_palette"])
        for i in range(1, n):
            rgb = pal.colors[i % len(pal.colors)].rgb
            colors.append(tuple(srgb_to_linear(rgb)))
            display.append(tuple(rgb))
    init_rot = _yaw_quat(v(bl["init_yaw_deg"]))
    blocks, prims, inits = [], [], []
    for i in range(n):
        bx, by = layout[i]
        init_pos = (bx, by, floor_top + e / 2 + 0.0005)
        blocks.append(RigidObject(
            RigidObjectCfg(
                prim_path=block_prim(i),
                spawn=sim_utils.CuboidCfg(
                    size=(e, e, e),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                        max_depenetration_velocity=1.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=v(bl["mass"])),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=v(bl["friction"])[0], dynamic_friction=v(bl["friction"])[1],
                        friction_combine_mode="max",
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(colors[i]), roughness=0.5),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
            )
        ))
        prims.append(block_prim(i))
        inits.append(init_pos)

    robot = Articulation(so101_cfg("/World/Robot", pos=tuple(v(scene["robot"]["base_pos"])), rot=ROBOT_ROT_WXYZ))

    sim_utils.DomeLightCfg(intensity=600.0).func("/World/Light/dome", sim_utils.DomeLightCfg(intensity=600.0))
    sim_utils.DistantLightCfg(intensity=900.0, angle=0.5).func(
        "/World/Light/key", sim_utils.DistantLightCfg(intensity=900.0, angle=0.5),
        orientation=(0.9239, 0.0, 0.3827, 0.0))

    # Mount-frame axes (visual only): red = +X forward, green = +Y left, blue = +Z up. Drawn beside the base.
    o, L, t = (-0.12, 0.20, 0.0), 0.08, 0.004
    _box("/World/Axes/x", (L, t, t), (o[0] + L / 2, o[1], t / 2), (0.9, 0.1, 0.1), collide=False)
    _box("/World/Axes/y", (t, L, t), (o[0], o[1] + L / 2, t / 2), (0.1, 0.8, 0.1), collide=False)
    _box("/World/Axes/z", (t, t, L), (o[0], o[1], L / 2), (0.1, 0.2, 0.9), collide=False)

    return SceneHandles(robot, blocks, prims, inits, init_rot, display, trays)


def block_in_tray(pos, tray: dict, edge: float) -> bool:
    """Block centre inside the tray interior (with half an edge margin) and resting near the floor."""
    cx, cy = tray["center"]
    c, s = math.cos(-tray["yaw"]), math.sin(-tray["yaw"])
    lx = c * (pos[0] - cx) - s * (pos[1] - cy)
    ly = s * (pos[0] - cx) + c * (pos[1] - cy)
    hx, hy = tray["inner_half"]
    return abs(lx) <= hx - edge / 2 and abs(ly) <= hy - edge / 2 and pos[2] < tray["floor_top"] + edge
