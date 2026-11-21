"""Keep the viewport camera synchronized with the measured physics link pose.

Fabric can move an articulation without updating its USD ancestor transforms.
Viewport camera transforms are read from USD, so an inherited camera can remain
at the authored pose. A reset transform stack and explicit link-based world pose
avoid mixing a stale image with current joint readings.
"""
from __future__ import annotations

import numpy as np
from pxr import Gf, UsdGeom

from agentic_sim.config import v
from agentic_sim.kinematics import quat_wxyz_to_mat, mat_to_quat_wxyz
from agentic_sim.isaac.scene import WRIST_CAMERA_PRIM


class WristCameraFollower:
    def __init__(self, stage, scene, robot):
        self.robot = robot
        self.active = False
        prim = stage.GetPrimAtPath(WRIST_CAMERA_PRIM)
        spec = scene.get('sensor_cameras', {}).get('wrist', {}).get('pose_in_gripper')
        if not prim.IsValid() or not spec:
            return
        self.body = robot.body_names.index('gripper')
        self.local_pos = np.array(v(spec['pos']),float)
        self.local_rot = quat_wxyz_to_mat(v(spec['quat_wxyz']))
        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        self.translate = next((op for op in ops if op.GetOpType()==UsdGeom.XformOp.TypeTranslate), None)
        self.orient = next((op for op in ops if op.GetOpType()==UsdGeom.XformOp.TypeOrient), None)
        if self.translate is None:
            self.translate = xform.AddTranslateOp()
        if self.orient is None:
            self.orient = xform.AddOrientOp()
        xform.SetXformOpOrder([self.translate,self.orient],resetXformStack=True)
        self.active = True
        self.update()

    def update(self):
        if not self.active:
            return
        data = self.robot.data
        position = data.body_link_pos_w[0,self.body].cpu().numpy()
        quaternion = data.body_link_quat_w[0,self.body].cpu().numpy()
        rotation = quat_wxyz_to_mat(quaternion)
        world_pos = position + rotation @ self.local_pos
        q = mat_to_quat_wxyz(rotation @ self.local_rot)
        vec = Gf.Vec3d if self.translate.GetPrecision()==UsdGeom.XformOp.PrecisionDouble else Gf.Vec3f
        quat = Gf.Quatd if self.orient.GetPrecision()==UsdGeom.XformOp.PrecisionDouble else Gf.Quatf
        self.translate.Set(vec(*map(float,world_pos)))
        self.orient.Set(quat(float(q[0]),Gf.Vec3d(*map(float,q[1:])) if quat==Gf.Quatd else Gf.Vec3f(*map(float,q[1:]))))
