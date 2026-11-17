"""Build the numpy SO-101 chain from the spawned robot's USD joint definitions."""

from __future__ import annotations

import numpy as np
from pxr import UsdPhysics

from agentic_sim.kinematics import ARM_JOINTS, JointDef, So101Chain


def _q(gfq) -> tuple:
    return (float(gfq.GetReal()), *[float(v) for v in gfq.GetImaginary()])


def chain_from_usd(stage, robot_root: str, ee_offset) -> So101Chain:
    joints, lower, upper = [], [], []
    for name in ARM_JOINTS:
        prim = stage.GetPrimAtPath(f"{robot_root}/joints/{name}")
        j = UsdPhysics.RevoluteJoint(prim)
        b0 = str(j.GetBody0Rel().GetTargets()[0]).rsplit("/", 1)[-1]
        b1 = str(j.GetBody1Rel().GetTargets()[0]).rsplit("/", 1)[-1]
        joints.append(
            JointDef(
                name=name, parent=b0, child=b1,
                pos0=tuple(float(v) for v in j.GetLocalPos0Attr().Get()), rot0=_q(j.GetLocalRot0Attr().Get()),
                pos1=tuple(float(v) for v in j.GetLocalPos1Attr().Get()), rot1=_q(j.GetLocalRot1Attr().Get()),
                axis=str(j.GetAxisAttr().Get()),
            )
        )
        lower.append(np.radians(j.GetLowerLimitAttr().Get()))
        upper.append(np.radians(j.GetUpperLimitAttr().Get()))
    return So101Chain(joints, np.array(lower), np.array(upper), np.asarray(ee_offset, float))
