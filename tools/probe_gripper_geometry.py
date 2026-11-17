"""Measure gripper/jaw mesh extents in the `gripper` link frame to locate the grasp center.

Run: docker/isaac/run.sh /isaac-sim/python.sh tools/probe_gripper_geometry.py
Writes outputs/probe_gripper_geometry.txt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from agentic_sim.isaac.so101 import so101_cfg  # noqa: E402

os.makedirs("outputs", exist_ok=True)
sys.stdout = open("outputs/probe_gripper_geometry.txt", "w", buffering=1)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
robot = Articulation(so101_cfg("/World/Robot"))
sim.reset()
stage = sim.stage


def mesh_points_world(root_path, coll_only):
    pts = []
    it = iter(Usd.PrimRange(stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()))
    for prim in it:
        p = str(prim.GetPath())
        if coll_only != ("collision" in p.lower()):
            continue
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for v in UsdGeom.Mesh(prim).GetPointsAttr().Get() or []:
                w = m.Transform(Gf.Vec3d(v))
                pts.append((w[0], w[1], w[2]))
    return np.array(pts)


def to_link(points_w, link):
    m = UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/Robot/{link}")).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    inv = m.GetInverse()
    return np.array([list(inv.Transform(Gf.Vec3d(*p))) for p in points_w])


for jaw in (-0.1745, 0.0, 0.5, 1.0):
    q = robot.data.default_joint_pos.clone()
    q[0, :] = 0.0
    q[0, 5] = jaw
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    sim.step(render=False)
    sim.render()
    robot.update(sim.get_physics_dt())
    for coll in (False, True):
        for link in ("gripper", "jaw"):
            pw = mesh_points_world(f"/World/Robot/{link}", coll)
            if len(pw) == 0:
                print(f"jaw={jaw} {link} coll={coll}: no mesh points")
                continue
            pl = to_link(pw, "gripper")
            print(f"jaw={jaw} {link} coll={coll} n={len(pl)} in-gripper-frame min={pl.min(0).round(4).tolist()} max={pl.max(0).round(4).tolist()}")
    gp = robot.data.body_pos_w[0, 5].tolist()
    jp = robot.data.body_pos_w[0, 6].tolist()
    print(f"jaw={jaw} gripper_body={np.round(gp,4).tolist()} jaw_body={np.round(jp,4).tolist()}")

# finger-region extents (authored USD state; physics poses are not written back to USD)
for link in ("gripper", "jaw"):
    pl = to_link(mesh_points_world(f"/World/Robot/{link}", True), "gripper")
    for zcut in (-0.06, -0.08, -0.095):
        sel = pl[pl[:, 2] < zcut]
        print(f"finger {link} z<{zcut}: n={len(sel)} min={sel.min(0).round(4).tolist()} max={sel.max(0).round(4).tolist()}")
    # x-profile of the finger near the tip region
    sel = pl[(pl[:, 2] < -0.07) & (pl[:, 2] > -0.10)]
    for xb in np.arange(-0.04, 0.035, 0.005):
        s2 = sel[(sel[:, 0] >= xb) & (sel[:, 0] < xb + 0.005)]
        if len(s2):
            print(f"  {link} x[{xb:.3f},{xb+0.005:.3f}) n={len(s2)} y=[{s2[:,1].min():.4f},{s2[:,1].max():.4f}] z=[{s2[:,2].min():.4f},{s2[:,2].max():.4f}]")
jp = stage.GetPrimAtPath("/World/Robot/joints/Jaw")
print("jaw joint attrs", {a.GetName(): a.Get() for a in jp.GetAttributes() if "local" in a.GetName() or "limit" in a.GetName().lower()})

# list the prim tree under gripper/jaw for reference
for link in ("gripper", "jaw"):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/World/Robot/{link}"), Usd.TraverseInstanceProxies()):
        print("prim", prim.GetPath(), prim.GetTypeName())

sys.stdout.flush()
os._exit(0)
