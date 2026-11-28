"""Plush grasp episodes in MuJoCo for ACT (tools/sim_plush_collect.py, tools/sim_plush_eval.py; docs/plush.md).

One episode: the plush scene with a random plush pose (x +-15 mm, y 0..+20 mm about the scene's plush.pos, yaw
0-180 deg) and 0-3 distractor blocks on plush.block_xy, the arm at the scene's standby (+-2 deg per joint). The
observation is laid out like the real ACT dataset (tools/real_to_lerobot.py, agentic_sim.real_policy):

  observation.state                arm + jaw angles as LeRobot values (agentic_sim.real_joints.shared_to_lerobot)
  observation.images.global        640 x 480, D455 pose of the scene; target circle drawn at the plush top
                                   (agentic_sim.real_dataset.draw_markers, as at runtime)
  observation.images.wrist         640 x 480, the scene's wrist lens (barrel distortion in "real" mode)
  observation.environment_state    [x mm, y mm, sin 2yaw, cos 2yaw]: footprint centre in the mount frame, tram yaw
                                   (yaw_fold 2), the real plush condition (agentic_sim.real_dataset.condition)
  action                           the joint command sent this frame (LeRobot values)

Global camera at 640 x 480: the scene calibrates the D455 colour stream at 1280 x 720. The 640 x 480 stream is the
4:3 part of the 1280 x 800 sensor scaled by 0.6, so f = 0.6 fx (estimated, not measured at 640 x 480). MuJoCo
renders a centred pinhole; the marker pixel is projected with the same pinhole.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from agentic_sim import plush as P
from agentic_sim.config import v
from agentic_sim.real_control import JOINTS
from agentic_sim.real_joints import lerobot_to_shared, shared_to_lerobot

FPS = 30
GLOBAL_SIZE = (640, 480)
GLOBAL_SCALE = 0.6               # 1280 x 720 intrinsics -> 640 x 480 stream (estimated, see the module doc)
POS_RANGE_X = (-0.015, 0.015)    # docs/plush.md: demonstration spread about plush.pos
POS_RANGE_Y = (0.0, 0.020)
START_NOISE_DEG = 2.0
BLOCK_COLORS = ("blue", "red", "yellow")
BLOCK_GAP_M = 0.030              # block edge to tram edge, at least (docs/plush.md asks for ~5 cm centre clearance)
PARK_XY = (0.58, 0.42)           # "no distractor": the one block the scene must hold, far off on the table
CONDITION_NAMES = ["target_x_mm", "target_y_mm", "target_sin4yaw", "target_cos4yaw"]   # same names as the real set
TASK = "Put the Shiba plush in the beige tray."
LIFT_OK_M = 0.045                # success: risen at least this much ...
SLIP_OK_M = 0.005                # ... and moved less than this in the gripper during the 1 s hold


def sample_layout(scene: dict, rng: np.random.Generator) -> dict:
    """Random plush pose and distractor blocks for one episode (positions in the mount frame, m / deg)."""
    x0, y0 = (float(a) for a in v(scene["plush"]["pos"]))
    px, py = x0 + rng.uniform(*POS_RANGE_X), y0 + rng.uniform(*POS_RANGE_Y)
    yaw = float(rng.uniform(0.0, 180.0))
    slots = [tuple(float(a) for a in xy) for xy in v(scene["plush"]["block_xy"])]
    want = int(rng.integers(0, 4))
    blocks = []
    for i in rng.permutation(len(slots)):
        if len(blocks) == want:
            break
        x, y = slots[i]
        if gap_to_tram(x, y, px, py, math.radians(yaw)) >= BLOCK_GAP_M:
            blocks.append({"xy": [x, y], "color": str(rng.choice(BLOCK_COLORS)),
                           "yaw_deg": float(rng.uniform(0.0, 90.0))})
    noise = rng.uniform(-START_NOISE_DEG, START_NOISE_DEG, 5)
    return {"plush_xy": [px, py], "plush_yaw_deg": yaw, "blocks": blocks, "start_noise_deg": noise.tolist()}


def gap_to_tram(x, y, px, py, yaw, edge=0.0295) -> float:
    """Distance from a block's edge (axis-free: half edge as a radius) to the tram footprint rectangle."""
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * (x - px) + s * (y - py), -s * (x - px) + c * (y - py)
    dx, dy = max(abs(lx) - P.LENGTH / 2, 0.0), max(abs(ly) - P.WIDTH / 2, 0.0)
    return math.hypot(dx, dy) - edge / 2


def scene_for(scene: dict, layout: dict) -> tuple:
    """(scene copy with the layout's plush pose and blocks, block colours for model.build)."""
    sc = copy.deepcopy(scene)
    sc["plush"] = dict(sc["plush"], pos=list(layout["plush_xy"]), yaw_deg=layout["plush_yaw_deg"])
    blocks = layout["blocks"] or [{"xy": list(PARK_XY), "color": "red", "yaw_deg": 0.0}]
    sc["plush"]["block_xy"] = [b["xy"] for b in blocks]
    return sc, [b["color"] for b in blocks]


def set_block_yaws(sc, d, layout):
    """Turn each distractor about Z (model.build places them axis-aligned)."""
    for i, b in enumerate(layout["blocks"]):
        a = math.radians(b["yaw_deg"]) / 2
        d.qpos[sc.block_qpos[i] + 3:sc.block_qpos[i] + 7] = [math.cos(a), 0.0, 0.0, math.sin(a)]


def joints_lerobot(sc, d) -> list:
    """Measured arm + jaw as LeRobot values (JOINTS order)."""
    q = np.array([d.qpos[a] for a in sc.arm_qpos]) - sc.arm_offset
    return shared_to_lerobot(q, float(d.qpos[sc.jaw_qpos]))


def command_lerobot(q_cmd, jaw_cmd) -> list:
    return shared_to_lerobot(np.asarray(q_cmd, float), float(jaw_cmd))


def command_shared(action) -> tuple:
    """LeRobot action -> (arm rad, jaw rad)."""
    return lerobot_to_shared(action)


def plush_condition(sc, d) -> tuple:
    """(condition vector, plush dict): footprint centre (mount mm), tram yaw mod 180, top point for the marker."""
    b = sc.plush_body
    p, R = d.xpos[b], d.xmat[b].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0]) % math.pi
    cond = [p[0] * 1000.0, p[1] * 1000.0, math.sin(2 * yaw), math.cos(2 * yaw)]
    top = p + R @ np.array([0.0, 0.0, P.TOTAL_HEIGHT])
    return cond, {"x_mm": round(p[0] * 1000, 1), "y_mm": round(p[1] * 1000, 1),
                  "yaw_deg": round(math.degrees(yaw), 1), "yaw_fold": 2, "top": top.tolist()}


class Cameras:
    """Global (640 x 480 pinhole, D455 pose) and wrist (scene lens) RGB renders of one model."""

    def __init__(self, sc, scene: dict):
        import mujoco

        from agentic_sim.mujoco_backend.model import GLOBAL_CAMERA, WRIST_CAMERA
        from agentic_sim.scene_layout import sensor_cameras

        m = sc.model
        cams = sensor_cameras(scene)
        g = cams["global"]
        self.f = GLOBAL_SCALE * g["fy"]
        self.gw, self.gh = GLOBAL_SIZE
        self.g_id = m.camera(GLOBAL_CAMERA).id
        m.cam_fovy[self.g_id] = math.degrees(2 * math.atan(self.gh / 2 / self.f))
        self.g_render = mujoco.Renderer(m, height=self.gh, width=self.gw)
        w = cams["wrist"]
        self.w_id = m.camera(WRIST_CAMERA).id
        self.lens = w.get("lens")
        rw, rh = w["render_size"]
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), rw, self.gw)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), rh, self.gh)
        self.w_render = mujoco.Renderer(m, height=rh, width=rw)
        self.maps = self.lens.maps(rw, rh, w["render_fovy_deg"]) if self.lens is not None else None

    def project(self, d, xyz) -> list:
        """Global pixel (u, v) of a mount-frame point, same pinhole as the render."""
        R = d.cam_xmat[self.g_id].reshape(3, 3)
        c = R.T @ (np.asarray(xyz, float) - d.cam_xpos[self.g_id])   # MuJoCo camera: x right, y up, looks along -z
        return [float(self.gw / 2 + self.f * c[0] / -c[2]), float(self.gh / 2 - self.f * c[1] / -c[2])]

    def render(self, d, marks=None) -> dict:
        from PIL import Image

        from agentic_sim.real_dataset import draw_markers

        self.g_render.update_scene(d, camera=self.g_id)
        g = self.g_render.render()
        if marks:
            g = np.asarray(draw_markers(Image.fromarray(g), marks))
        self.w_render.update_scene(d, camera=self.w_id)
        w = self.w_render.render()
        if self.maps is not None:
            w = self.lens.apply(w, self.maps)
        return {"global": np.ascontiguousarray(g, np.uint8), "wrist": np.ascontiguousarray(w, np.uint8)}

    def close(self):
        self.g_render.close()
        self.w_render.close()


def features() -> dict:
    """LeRobot features, the real ACT layout (act_grasp: state, action, flags, two cameras, condition)."""
    w, h = GLOBAL_SIZE
    f = {"observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
         "action": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
         "task_is_policy": {"dtype": "int64", "shape": (1,), "names": None},
         "is_handoff": {"dtype": "int64", "shape": (1,), "names": None},
         "observation.environment_state": {"dtype": "float32", "shape": (4,), "names": CONDITION_NAMES}}
    for cam in ("global", "wrist"):
        f[f"observation.images.{cam}"] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return f


def contacts(m, d, sc, gripper_bodies) -> dict:
    """Current contacts: plush geoms touched by the gripper / jaw, and gripper-side bodies touching blocks or trays."""
    plush = set(sc.plush_geoms)
    blocks = set(sc.block_geoms)
    out = {"plush_parts": set(), "blocks": False, "tray": False}
    for c in d.contact[:d.ncon]:
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            body_b = m.geom_bodyid[b]
            if a in plush and body_b in gripper_bodies:
                out["plush_parts"].add(m.geom(a).name.removeprefix(f"{P.ID}_"))
            if body_b in gripper_bodies and a in blocks:
                out["blocks"] = True
            if body_b in gripper_bodies and "_wall_" in m.geom(a).name:
                out["tray"] = True
    return out
