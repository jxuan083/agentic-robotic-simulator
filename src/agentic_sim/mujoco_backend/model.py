"""Build the MuJoCo model of a SceneSpec: menagerie SO-101 + table, trays and blocks from agentic_sim.scene_layout.

The arm is the menagerie MJCF with its own options, gains and contact settings unchanged; only its base is moved so
its frames coincide with the shared kinematic chain (configs/mujoco_so101.yaml, alignment). Scene numbers come from
the same scene YAML as the Isaac scene.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

from agentic_sim import plush as P
from agentic_sim.config import load, v
from agentic_sim.kinematics import ARM_JOINTS, GRIPPER_JOINT
from agentic_sim.scene_layout import TRAYS, block_layout, global_camera_quat, sensor_cameras, table_box, tray_parts, wrist_camera_mount, yaw_quat

ROBOT_CONFIG = "configs/mujoco_so101.yaml"
SHELL_EXCLUDE = ("sts3215",)  # motor materials keep their colour (as in the Isaac colour DR)


def menagerie_dir() -> str:
    return os.environ.get("MENAGERIE_DIR", os.path.expanduser("~/ref/mujoco_menagerie"))


def model_path(robot_cfg: dict) -> str:
    p = os.path.join(menagerie_dir(), robot_cfg["model"]["path"])
    if not os.path.isfile(p):
        raise FileNotFoundError(f"SO-101 MJCF not found at {p}; run tools/fetch_mujoco_assets.sh")
    return p


@dataclass
class MjScene:
    model: mujoco.MjModel
    robot_cfg: dict
    arm_qpos: list           # qpos address per ARM_JOINTS entry
    arm_qvel: list
    arm_act: list            # actuator id per ARM_JOINTS entry
    arm_offset: np.ndarray   # q_mujoco = q_shared + offset
    jaw_qpos: int
    jaw_act: int
    block_bodies: list       # body id per block
    block_geoms: list        # geom id per block
    block_qpos: list         # free-joint qpos address per block (7: pos + quat wxyz)
    block_qvel: list
    block_init_pos: list
    block_init_rot: tuple
    block_init_rots: list
    block_colors: list       # display rgb per block (baseline)
    shell_materials: list    # material ids recoloured by colour DR
    edge: float
    trays: dict
    plush_body: int | None = None    # finals plush (scene `plush:` section); None without it
    plush_geoms: list = field(default_factory=list)   # its collision geoms
    plush_qpos: int | None = None    # free-joint qpos address (7: pos + quat wxyz)
    plush_qvel: int | None = None
    plush_init_pos: tuple | None = None
    plush_init_quat: tuple | None = None


def _box(parent, name, size, pos, rgba, friction=None, quat=(1, 0, 0, 0), collide=True):
    g = parent.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [s / 2 for s in size]
    g.pos = list(pos)
    g.quat = list(quat)
    g.rgba = list(rgba)
    if friction is not None:
        g.friction = [friction, 0.005, 0.0001]
    if not collide:
        g.contype = 0
        g.conaffinity = 0
    return g


def build(scene_cfg: dict, n_blocks: int | None = None, robot_cfg: dict | None = None,
          colors: list | None = None) -> MjScene:
    rc = robot_cfg or load(ROBOT_CONFIG)
    spec = mujoco.MjSpec.from_file(model_path(rc))
    phys = rc["physics"]
    spec.option.timestep = float(phys["timestep"])
    keep = {"camera_mount"} if wrist_camera_mount(scene_cfg) else set()
    for name in rc["model"].get("remove_bodies", []):
        if name not in keep:
            spec.delete(spec.body(name))
    al = rc["alignment"]
    base = spec.body("base")
    # MJCF base origin = the scene's robot base position (shared base link frame) + the MJCF / USD base offset
    base.pos = (np.array(v(scene_cfg["robot"]["base_pos"]), float) + np.array(v(al["base_pos_in_mount"]), float)).tolist()
    fr = 0 if phys.get("friction_from", "static") == "static" else 1

    wb = spec.worldbody
    # environment: lights, ground under the table, table, trays
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    sky = spec.add_texture()
    sky.name, sky.type, sky.builtin = "sky", mujoco.mjtTexture.mjTEXTURE_SKYBOX, mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1, sky.rgb2, sky.width, sky.height = [0.86, 0.89, 0.93], [0.42, 0.47, 0.55], 512, 3072
    for pos, d in (((0.3, -0.6, 1.5), (-0.2, 0.4, -1.0)), ((-0.4, 0.5, 1.2), (0.4, -0.4, -1.0))):
        lt = wb.add_light()
        lt.pos, lt.dir = list(pos), list(d)
        lt.diffuse = [0.45, 0.45, 0.45]
        lt.castshadow = True
    tb = table_box(scene_cfg)
    ground = wb.add_geom()
    ground.name = "ground"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [3, 3, 0.05]
    ground.pos = [0, 0, -tb["size"][2]]
    ground.rgba = [0.55, 0.57, 0.6, 1]
    _box(wb, "table", tb["size"], tb["pos"], (*tb["color"], 1), friction=tb["friction"][fr])
    trays = {}
    tray_fr = v(scene_cfg["trays"]["friction"])[fr]
    for name in TRAYS:
        parts, info = tray_parts(scene_cfg["trays"][name])
        q = yaw_quat(info["yaw_deg"])
        for pname, (size, pos) in parts.items():
            _box(wb, f"{name}_{pname}", size, pos, (*info["color"], 1), friction=tray_fr, quat=q)
        trays[name] = {k: info[k] for k in ("center", "inner_half", "yaw", "floor_top")}
    # mount-frame axes beside the base (visual only): red +X, green +Y, blue +Z, as in the Isaac scene
    o, L, t = (-0.12, 0.20, 0.0), 0.08, 0.004
    _box(wb, "axis_x", (L, t, t), (o[0] + L / 2, o[1], t / 2), (0.9, 0.1, 0.1, 1), collide=False)
    _box(wb, "axis_y", (t, L, t), (o[0], o[1] + L / 2, t / 2), (0.1, 0.8, 0.1, 1), collide=False)
    _box(wb, "axis_z", (t, t, L), (o[0], o[1], L / 2), (0.1, 0.2, 0.9, 1), collide=False)

    # top view of the panel: orthographic, looking straight down, screen up = +X, screen right = -Y (same as the
    # Isaac top view); fovy is the shown height in metres for an orthographic camera
    from agentic_sim.mujoco_backend.panel import MAP_X, MAP_Y
    top = wb.add_camera()
    top.name = "topview"
    top.pos = [0.5 * (MAP_X[0] + MAP_X[1]), 0.5 * (MAP_Y[0] + MAP_Y[1]), 1.5]
    top.quat = [math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4)]
    top.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
    top.fovy = MAP_X[1] - MAP_X[0]

    add_sensor_cameras(spec, scene_cfg)

    lay = block_layout(scene_cfg, n_blocks, colors)
    for i, (pos, rgb) in enumerate(zip(lay.init_pos, lay.display_rgb)):
        b = wb.add_body()
        b.name = f"block_{i}"
        b.pos = list(pos)
        b.quat = list(lay.init_rots[i])
        b.add_freejoint().name = f"block_{i}_free"
        g = _box(b, f"block_{i}_geom", (lay.edge,) * 3, (0, 0, 0), (*rgb, 1), friction=lay.friction[fr])
        g.mass = lay.mass

    plush = add_plush(wb, scene_cfg["plush"]) if "plush" in scene_cfg else None

    m = spec.compile()
    names = al["joints"]
    offsets = {k: v(e) for k, e in (al.get("joint_offset") or {}).items()}
    jid = [m.joint(names[n]).id for n in ARM_JOINTS]
    shell = [i for i in range(m.nmat) if m.material(i).name.endswith("_material")
             and not any(x in m.material(i).name for x in SHELL_EXCLUDE)]
    return MjScene(
        model=m, robot_cfg=rc,
        arm_qpos=[int(m.jnt_qposadr[j]) for j in jid], arm_qvel=[int(m.jnt_dofadr[j]) for j in jid],
        arm_act=[m.actuator(names[n]).id for n in ARM_JOINTS],
        arm_offset=np.array([float(offsets.get(n, 0.0)) for n in ARM_JOINTS]),
        jaw_qpos=int(m.jnt_qposadr[m.joint(names[GRIPPER_JOINT]).id]), jaw_act=m.actuator(names[GRIPPER_JOINT]).id,
        block_bodies=[m.body(f"block_{i}").id for i in range(len(lay.init_pos))],
        block_geoms=[m.geom(f"block_{i}_geom").id for i in range(len(lay.init_pos))],
        block_qpos=[int(m.jnt_qposadr[m.joint(f"block_{i}_free").id]) for i in range(len(lay.init_pos))],
        block_qvel=[int(m.jnt_dofadr[m.joint(f"block_{i}_free").id]) for i in range(len(lay.init_pos))],
        block_init_pos=list(lay.init_pos), block_init_rot=lay.init_rot, block_init_rots=list(lay.init_rots),
        block_colors=list(lay.display_rgb),
        shell_materials=shell, edge=lay.edge, trays=trays,
        **(_plush_ids(m, *plush) if plush else {}))


PLUSH_BODY = P.ID


def add_plush(wb, cfg: dict):
    """Finals plush as one free body of rigid primitives, bottom face on the table top at cfg pos / yaw_deg.
    Collision: tram (two crossed boxes + four corner cylinders = a box with rounded vertical edges) and the head
    ellipsoid; band, windows, underside, muzzle and paws are visual only. Returns (init pos, init quat)."""
    x, y = v(cfg["pos"])
    pos, quat = (float(x), float(y), 0.0005), yaw_quat(float(v(cfg.get("yaw_deg", 0.0))))
    b = wb.add_body()
    b.name = PLUSH_BODY
    b.pos, b.quat = list(pos), list(quat)
    b.add_freejoint().name = f"{PLUSH_BODY}_free"
    com, diag = P.inertia()
    b.explicitinertial = True
    b.mass, b.ipos, b.inertia = P.MASS, list(com), list(diag)
    L, W, H, r = P.LENGTH, P.WIDTH, P.BODY_HEIGHT, P.CORNER_RADIUS
    green, dog = (*P.rgb(P.COLOR_BODY), 1), (*P.rgb(P.COLOR_DOG), 1)
    for name, size in (("body_x", (L, W - 2 * r, H)), ("body_y", (L - 2 * r, W, H))):
        _box(b, f"{PLUSH_BODY}_{name}", size, (0, 0, H / 2), green, friction=P.FRICTION).mass = 0
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        g = b.add_geom()
        g.name, g.type = f"{PLUSH_BODY}_corner_{i}", mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size, g.pos = [r, H / 2, 0], [sx * (L / 2 - r), sy * (W / 2 - r), H / 2]
        g.rgba, g.friction, g.mass = list(green), [P.FRICTION, 0.005, 0.0001], 0
    g = b.add_geom()
    g.name, g.type = f"{PLUSH_BODY}_head", mujoco.mjtGeom.mjGEOM_ELLIPSOID
    g.size, g.pos, g.rgba = list(P.HEAD_RADII), list(P.HEAD_POS), list(dog)
    g.friction, g.mass = [P.FRICTION, 0.005, 0.0001], 0

    def deco(name, size, pos, hex_color):
        _box(b, f"{PLUSH_BODY}_{name}", size, pos, (*P.rgb(hex_color), 1), collide=False).mass = 0

    e = 0.0006   # decorations stand this far out of the faces they cover
    for band, (z0, z1), color in (("under", P.UNDER_BAND, P.COLOR_UNDER), ("waist", P.WAIST_BAND, P.COLOR_BAND)):
        zc, hz = (z0 + z1) / 2, z1 - z0
        for s in (1, -1):
            deco(f"{band}_y{s:+d}", (L - 2 * r, 2 * e, hz), (0, s * W / 2, zc), color)
            deco(f"{band}_x{s:+d}", (2 * e, W - 2 * r, hz), (s * L / 2, 0, zc), color)
    zc, hz = (P.WINDOW_BAND[0] + P.WINDOW_BAND[1]) / 2, P.WINDOW_BAND[1] - P.WINDOW_BAND[0]
    for s in (1, -1):
        for i, wx in enumerate((-0.025, -0.0085, 0.0085, 0.025)):   # four side windows
            deco(f"window_y{s:+d}_{i}", (0.013, 4 * e, hz), (wx, s * W / 2, zc), P.COLOR_WINDOW)
        for i, wy in enumerate((-0.011, 0.011)):                    # two front / rear windows
            deco(f"window_x{s:+d}_{i}", (4 * e, 0.016, hz), (s * L / 2, wy, zc), P.COLOR_WINDOW)
    hx, hy, hz = P.HEAD_POS
    m = b.add_geom()
    m.name, m.type = f"{PLUSH_BODY}_muzzle", mujoco.mjtGeom.mjGEOM_ELLIPSOID
    m.size, m.pos = list(P.MUZZLE_RADII), [hx, hy + P.HEAD_RADII[1] - 0.004, hz - 0.006]
    m.rgba, m.contype, m.conaffinity, m.mass = [*P.rgb(P.COLOR_FACE), 1], 0, 0, 0
    for i, px in enumerate((-0.013, 0.013)):
        p = b.add_geom()
        p.name, p.type = f"{PLUSH_BODY}_paw_{i}", mujoco.mjtGeom.mjGEOM_SPHERE
        p.size, p.pos = [P.PAW_RADIUS, 0, 0], [px, W / 2 - P.PAW_RADIUS, H + 0.3 * P.PAW_RADIUS]
        p.rgba, p.contype, p.conaffinity, p.mass = list(dog), 0, 0, 0
    return pos, quat


def _plush_ids(m: mujoco.MjModel, pos, quat) -> dict:
    body, jnt = m.body(PLUSH_BODY).id, m.joint(f"{PLUSH_BODY}_free").id
    return {"plush_body": body,
            "plush_geoms": [i for i in range(m.ngeom) if m.geom_bodyid[i] == body and m.geom_contype[i]],
            "plush_qpos": int(m.jnt_qposadr[jnt]), "plush_qvel": int(m.jnt_dofadr[jnt]),
            "plush_init_pos": pos, "plush_init_quat": quat}


WRIST_CAMERA = "wrist_cam"     # Menagerie camera inside the camera_mount body
GLOBAL_CAMERA = "global_cam"


def add_sensor_cameras(spec, scene_cfg: dict):
    """Global camera fixed in the world (scene pose, vertical FOV from fy) and the wrist camera's FOV."""
    cams = sensor_cameras(scene_cfg)
    g = cams.get("global")
    if g:
        c = spec.worldbody.add_camera()
        c.name = GLOBAL_CAMERA
        c.pos = list(g["pos"])
        c.quat = list(global_camera_quat(g))
        c.fovy = g["fovy_deg"]
    w = cams.get("wrist")
    if w and wrist_camera_mount(scene_cfg):
        c = spec.camera(WRIST_CAMERA)
        c.fovy = w["render_fovy_deg"]   # "real" lens mode: a wider pinhole, resampled to the raw sensor by the viewer
        if w["pos_in_gripper"] is not None:  # the camera_mount body sits at the gripper body origin
            c.pos, c.quat = list(w["pos_in_gripper"]), list(w["quat_in_gripper"])


def sensor_camera_names(m: mujoco.MjModel) -> dict:
    """Scene camera name -> MuJoCo camera id, for the sensor cameras present in the model."""
    out = {}
    for key, name in (("global", GLOBAL_CAMERA), ("wrist", WRIST_CAMERA)):
        try:
            out[key] = m.camera(name).id
        except KeyError:
            pass
    return out


def free_camera(scene_cfg: dict) -> dict:
    """mjvCamera parameters (lookat, distance, azimuth, elevation) for the scene's viewer camera eye / target."""
    eye, tgt = np.array(scene_cfg["camera"]["eye"], float), np.array(scene_cfg["camera"]["target"], float)
    f = tgt - eye
    d = float(np.linalg.norm(f))
    f /= d
    return {"lookat": tgt.tolist(), "distance": d, "azimuth": math.degrees(math.atan2(f[1], f[0])),
            "elevation": math.degrees(math.asin(f[2]))}
