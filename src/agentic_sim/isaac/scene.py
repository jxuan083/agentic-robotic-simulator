"""Build the minimal lab_current scene: table, two trays (open-top colliders), 1-8 blocks, SO-101, lights, axes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg

from agentic_sim.config import v
from agentic_sim.isaac.so101 import so101_cfg
from agentic_sim.scene_layout import TRAYS, block_count_range, block_in_tray, block_layout, table_box, tray_parts  # noqa: F401
from agentic_sim.scene_layout import global_camera_quat, sensor_cameras, wrist_camera_mount
from agentic_sim.scene_layout import yaw_quat as _yaw_quat

# Mount frame (+X forward) = robot base link rotated +90 deg about Z (see agentic_sim.kinematics.MOUNT_FROM_BASE).
ROBOT_ROT_WXYZ = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))


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
    """Static open-top tray from five boxes (geometry: agentic_sim.scene_layout.tray_parts)."""
    parts, info = tray_parts(spec)
    rot = _yaw_quat(info["yaw_deg"])
    for name, (size, pos) in parts.items():
        _box(f"{root}/{name}", size, pos, info["color"], friction, rot=rot)
    return {k: info[k] for k in ("center", "inner_half", "yaw", "floor_top")}


@dataclass
class SceneHandles:
    robot: Articulation
    blocks: list            # RigidObject per block; blocks[0] is /World/Block
    block_prims: list
    block_init_pos: list    # (x, y, z) per block
    block_init_rot: tuple
    block_colors: list      # baseline display colour (sRGB-ish) per block, for the panel
    trays: dict
    sensor_cameras: dict    # key -> (camera prim path, width, height)

    @property
    def block(self) -> RigidObject:
        return self.blocks[0]


def block_prim(i: int) -> str:
    return "/World/Block" if i == 0 else f"/World/Block_{i}"


def build_scene(scene: dict, n_blocks: int | None = None, colors: list | None = None) -> SceneHandles:
    tb = table_box(scene)
    tsz = tb["size"][2]
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg(), translation=(0.0, 0.0, -tsz))
    _box("/World/Table", tb["size"], tb["pos"], tb["color"], tb["friction"])

    trays = {}
    for name in TRAYS:
        trays[name] = _tray(f"/World/Trays/{name}", scene["trays"][name], v(scene["trays"]["friction"]))

    lay = block_layout(scene, n_blocks, colors)
    e, n, colors, display, init_rot = lay.edge, len(lay.init_pos), lay.linear_rgb, lay.display_rgb, lay.init_rot
    blocks, prims, inits = [], [], []
    for i in range(n):
        init_pos = lay.init_pos[i]
        blocks.append(RigidObject(
            RigidObjectCfg(
                prim_path=block_prim(i),
                spawn=sim_utils.CuboidCfg(
                    size=(e, e, e),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                        max_depenetration_velocity=1.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=lay.mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=lay.friction[0], dynamic_friction=lay.friction[1],
                        friction_combine_mode="max",
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(colors[i]), roughness=0.5),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
            )
        ))
        prims.append(block_prim(i))
        inits.append(init_pos)

    robot = Articulation(so101_cfg("/World/Robot", pos=tuple(v(scene["robot"]["base_pos"])), rot=ROBOT_ROT_WXYZ,
                                   camera_mount=wrist_camera_mount(scene)))
    cams = _sensor_cameras(scene)

    sim_utils.DomeLightCfg(intensity=600.0).func("/World/Light/dome", sim_utils.DomeLightCfg(intensity=600.0))
    sim_utils.DistantLightCfg(intensity=900.0, angle=0.5).func(
        "/World/Light/key", sim_utils.DistantLightCfg(intensity=900.0, angle=0.5),
        orientation=(0.9239, 0.0, 0.3827, 0.0))

    # Mount-frame axes (visual only): red = +X forward, green = +Y left, blue = +Z up. Drawn beside the base.
    o, L, t = (-0.12, 0.20, 0.0), 0.08, 0.004
    _box("/World/Axes/x", (L, t, t), (o[0] + L / 2, o[1], t / 2), (0.9, 0.1, 0.1), collide=False)
    _box("/World/Axes/y", (t, L, t), (o[0], o[1] + L / 2, t / 2), (0.1, 0.8, 0.1), collide=False)
    _box("/World/Axes/z", (t, t, L), (o[0], o[1], L / 2), (0.1, 0.2, 0.9), collide=False)

    return SceneHandles(robot, blocks, prims, inits, init_rot, display, trays, cams)


GLOBAL_CAMERA_PRIM = "/World/Cameras/global"
WRIST_CAMERA_PRIM = "/World/Robot/gripper/wrist_cam"   # child of the gripper link: follows the arm


def _pinhole(cam: dict):
    k = [cam["fx"], 0.0, cam["cx"], 0.0, cam["fy"], cam["cy"], 0.0, 0.0, 1.0]
    return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(k, cam["width"], cam["height"], clipping_range=(0.01, 20.0))


def _sensor_cameras(scene: dict) -> dict:
    """USD camera prims of the scene's sensor cameras (pinhole from the intrinsics; lens distortion not modelled).
    The wrist camera is spawned under the gripper link after the robot (the link prim must exist)."""
    out = {}
    specs = sensor_cameras(scene)
    g = specs.get("global")
    if g:
        cfg = _pinhole(g)
        cfg.func(GLOBAL_CAMERA_PRIM, cfg, translation=g["pos"], orientation=global_camera_quat(g))
        out["global"] = (GLOBAL_CAMERA_PRIM, g["width"], g["height"])
    w = specs.get("wrist")
    if w and w["pos_in_gripper"] is not None and wrist_camera_mount(scene):
        cfg = _pinhole(w)
        cfg.func(WRIST_CAMERA_PRIM, cfg, translation=w["pos_in_gripper"], orientation=w["quat_in_gripper"])
        out["wrist"] = (WRIST_CAMERA_PRIM, w["width"], w["height"])
    return out
