"""Headless inspection of the SO-101 workshop asset: joints, limits, links, collision bounds.

Run inside the Isaac container:
    docker/isaac/run.sh ./../../isaac-sim/python.sh tools/inspect_so101_asset.py
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

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from agentic_sim.isaac.so101 import so101_cfg  # noqa: E402

os.makedirs("outputs", exist_ok=True)
sys.stdout = open("outputs/inspect_so101.txt", "w", buffering=1)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
robot = Articulation(so101_cfg("/World/Robot"))
sim.reset()
for _ in range(60):
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())

print("joint_names", robot.joint_names)
print("body_names", robot.body_names)
lim = robot.data.joint_pos_limits[0]
for n, (lo, hi) in zip(robot.joint_names, lim.tolist()):
    print(f"limit {n}: [{lo:.4f}, {hi:.4f}]")
print("default_joint_pos", robot.data.default_joint_pos[0].tolist())
root = robot.data.root_pos_w[0]
for n, p, q in zip(robot.body_names, robot.data.body_pos_w[0].tolist(), robot.data.body_quat_w[0].tolist()):
    print(f"body {n}: pos={[round(v, 4) for v in p]} quat={[round(v, 4) for v in q]}")

stage = sim.stage
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot")):
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = box.GetMin(), box.GetMax()
        approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else "-"
        print(f"collider {prim.GetPath()} approx={approx} min={[round(v, 4) for v in mn]} max={[round(v, 4) for v in mx]}")
    if prim.HasAPI(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.RevoluteJoint):
        j = UsdPhysics.RevoluteJoint(prim)
        print(f"joint {prim.GetPath()} axis={j.GetAxisAttr().Get()} b0={j.GetBody0Rel().GetTargets()} b1={j.GetBody1Rel().GetTargets()}")

sys.stdout.flush()
os._exit(0)  # SimulationApp.close() can hang headless in Isaac Sim 5.1
