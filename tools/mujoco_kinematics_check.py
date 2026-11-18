"""FK alignment of the MuJoCo SO-101 with the shared kinematic chain (configs/so101_chain.yaml, from the Workshop USD).

  uv run tools/mujoco_kinematics_check.py [--n 2000] [--out outputs/mujoco/kinematics_check.json]

For random joint vectors inside the shared limits, compares the EE (grasp centre, configs/teleop.yaml offset) and the
gripper-link orientation from the shared chain with MuJoCo's own FK of the built scene (configs/mujoco_so101.yaml
alignment applied). Also re-derives the alignment from the q = 0 pose, so a changed asset shows up as a mismatch.
Exit code 1 if the EE difference reaches 1 mm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from agentic_sim.config import load, v  # noqa: E402
from agentic_sim.kinematics import So101Chain, _inv  # noqa: E402
from agentic_sim.mujoco_backend.model import build  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="outputs/mujoco/kinematics_check.json")
a = ap.parse_args()

chain = So101Chain.from_dict(load("configs/so101_chain.yaml"))
chain.ee_offset = np.array(v(load("configs/teleop.yaml")["ee_offset_in_gripper_link"]), float)
scene = load("configs/scenes/lab_current_minimal.yaml")
chain.base_pos = np.array(v(scene["robot"]["base_pos"]), float)   # both engines place the base from the scene
sc = build(scene, 1)
m = sc.model
d = mujoco.MjData(m)
gid = m.body("gripper").id
links = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]


def mj_pose(q_shared, body):
    d.qpos[:] = m.qpos0
    for adr, qi, off in zip(sc.arm_qpos, q_shared, sc.arm_offset):
        d.qpos[adr] = qi + off
    mujoco.mj_kinematics(m, d)
    t = np.eye(4)
    t[:3, :3] = d.xmat[body].reshape(3, 3)
    t[:3, 3] = d.xpos[body]
    return t


def rot_deg(r):
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


# per-link frame differences at q = 0 (alignment as configured)
q0 = np.zeros(5)
from agentic_sim.kinematics import MOUNT_FROM_BASE  # noqa: E402
ws = {k: chain.mount_from_base @ t for k, t in chain.link_poses_base(q0).items()}
per_link = {}
for k in links:
    tm = mj_pose(q0, m.body(k).id)
    per_link[k] = {"pos_mm": round(float(np.linalg.norm(tm[:3, 3] - ws[k][:3, 3])) * 1000, 4),
                   "rot_deg": round(rot_deg(ws[k][:3, :3].T @ tm[:3, :3]), 4)}
# alignment re-derived: the extra base translation that would make the shoulder frames coincide (should be ~0)
d.qpos[:] = m.qpos0
mujoco.mj_kinematics(m, d)
sh = m.body("shoulder").id
t_mj = np.eye(4)
t_mj[:3, :3], t_mj[:3, 3] = d.xmat[sh].reshape(3, 3), d.xpos[sh]
residual = ws["shoulder"] @ _inv(t_mj)
base_pos_cfg = np.array(v(sc.robot_cfg["alignment"]["base_pos_in_mount"]))  # relative to the USD base link

rng = np.random.default_rng(a.seed)
pos_err, rot_err = [], []
for _ in range(a.n):
    q = rng.uniform(chain.lower, chain.upper)
    ee_s = chain.ee_pose_mount(q)
    tm = mj_pose(q, gid)
    ee_m = tm[:3, :3] @ chain.ee_offset + tm[:3, 3]
    pos_err.append(float(np.linalg.norm(ee_m - ee_s[:3, 3])) * 1000)
    g_s = chain.mount_from_base @ chain.link_poses_base(q)["gripper"]
    rot_err.append(rot_deg(g_s[:3, :3].T @ tm[:3, :3]))
pos_err, rot_err = np.array(pos_err), np.array(rot_err)
out = {
    "samples": a.n, "seed": a.seed,
    "ee_pos_err_mm": {"max": round(float(pos_err.max()), 4), "p95": round(float(np.percentile(pos_err, 95)), 4),
                      "mean": round(float(pos_err.mean()), 4)},
    "gripper_rot_err_deg": {"max": round(float(rot_err.max()), 4), "mean": round(float(rot_err.mean()), 4)},
    "per_link_at_zero": per_link,
    "base_pos_in_mount": {"configured": base_pos_cfg.round(7).tolist(),
                          "derived": (base_pos_cfg + residual[:3, 3]).round(7).tolist(),
                          "rotation_dev": float(np.abs(residual[:3, :3] - np.eye(3)).max())},
    "joint_offset": dict(zip(["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"], sc.arm_offset.tolist())),
    "limits": {"shared_lower": chain.lower.round(5).tolist(), "shared_upper": chain.upper.round(5).tolist(),
               "mujoco_range": [[round(float(x), 5) for x in m.jnt_range[m.joint(n).id]] for n in
                                ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]]},
    "pass": bool(pos_err.max() < 1.0),
}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: out[k] for k in ("ee_pos_err_mm", "gripper_rot_err_deg", "per_link_at_zero", "base_pos_in_mount",
                                      "pass")}, indent=1))
print(f"wrote {a.out}")
sys.exit(0 if out["pass"] else 1)
