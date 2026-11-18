"""SO-101 serial-chain kinematics (numpy only) and constrained downward-grasp IK.

The chain parameters are read from the robot USD joint definitions (see agentic_sim.isaac.chain_from_usd)
or from a dumped YAML, so this module runs without Isaac Sim (tests, offline checks).

Frames
------
- ``base``: the robot base link frame of the USD asset.
- ``mount``: operator / scene frame used for teleop: +X forward (toward the workspace), +Y left, +Z up, origin on the
  table top below the robot. The base link sits at ``base_pos`` in it, rotated by ``MOUNT_FROM_BASE`` (+90 deg about
  Z). base_pos comes from the scene (robot.base_pos): the base link origin of the SO-101 asset lies 30.1 mm below the
  bottom face of its base, so a base standing on the table has base_pos z = -0.0301.
- ``ee``: grasp center between the two fingers, ``ee_offset`` expressed in the ``gripper`` link frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
GRIPPER_JOINT = "Jaw"


def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _cross(a, b):
    """Cross product of 3-vectors (or rows of N x 3 arrays); np.cross is slow for tiny arrays (IK inner loop)."""
    a, b = np.asarray(a), np.asarray(b)
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def _tf(rot: np.ndarray, pos) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def _inv(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3]
    return _tf(r.T, -r.T @ t[:3, 3])


# Base link -> mount frame: the asset's arm reaches along base -Y; rotate +90 deg about Z so it reaches along +X.
MOUNT_FROM_BASE = _tf(rot_z(np.pi / 2), (0.0, 0.0, 0.0))


@dataclass
class JointDef:
    name: str
    parent: str
    child: str
    pos0: tuple
    rot0: tuple  # wxyz
    pos1: tuple
    rot1: tuple  # wxyz
    axis: str = "Z"

    def __post_init__(self):
        axis = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}[self.axis]
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        self._k, self._kk, self._axis = k, k @ k, axis
        self._j0 = _tf(quat_wxyz_to_mat(self.rot0), self.pos0)
        self._j1_inv = _inv(_tf(quat_wxyz_to_mat(self.rot1), self.pos1))

    def transform(self, q: float) -> np.ndarray:
        """child-in-parent transform at joint position q (rad)."""
        rq = np.eye(4)
        rq[:3, :3] += np.sin(q) * self._k + (1 - np.cos(q)) * self._kk
        return self._j0 @ rq @ self._j1_inv


@dataclass
class So101Chain:
    joints: list  # JointDef for ARM_JOINTS in order
    lower: np.ndarray
    upper: np.ndarray
    ee_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # base link origin in the mount frame

    @property
    def mount_from_base(self) -> np.ndarray:
        t = MOUNT_FROM_BASE.copy()
        t[:3, 3] = self.base_pos
        return t

    @classmethod
    def from_dict(cls, d: dict) -> "So101Chain":
        joints = [JointDef(**j) for j in d["joints"]]
        return cls(joints, np.array(d["lower"], float), np.array(d["upper"], float), np.array(d.get("ee_offset", [0, 0, 0]), float))

    def to_dict(self) -> dict:
        return {
            "joints": [
                dict(name=j.name, parent=j.parent, child=j.child, pos0=list(map(float, j.pos0)), rot0=list(map(float, j.rot0)),
                     pos1=list(map(float, j.pos1)), rot1=list(map(float, j.rot1)), axis=j.axis)
                for j in self.joints
            ],
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "ee_offset": self.ee_offset.tolist(),
        }

    # ---- forward kinematics -------------------------------------------------
    def link_poses_base(self, q) -> dict:
        """Poses (4x4, base link frame) of every link for arm joints q (len 5)."""
        poses = {self.joints[0].parent: np.eye(4)}
        for j, qi in zip(self.joints, q):
            poses[j.child] = poses[j.parent] @ j.transform(qi)
        return poses

    def gripper_pose_mount(self, q) -> np.ndarray:
        return self.mount_from_base @ self.link_poses_base(q)[self.joints[-1].child]

    def ee_pose_mount(self, q) -> np.ndarray:
        g = self.gripper_pose_mount(q)
        return g @ _tf(np.eye(3), self.ee_offset)

    # ---- IK -----------------------------------------------------------------
    def _fk_with_axes(self, q):
        """EE pose (mount) plus each joint's world axis and origin (mount frame)."""
        t = self.mount_from_base.copy()
        axes, origins = [], []
        for j, qi in zip(self.joints, q):
            jf = t @ j._j0  # joint frame (rotation about its axis does not move the axis)
            axes.append(jf[:3, :3] @ j._axis)
            origins.append(jf[:3, 3])
            t = t @ j.transform(qi)
        ee = t @ _tf(np.eye(3), self.ee_offset)
        return ee, np.array(axes), np.array(origins)

    @staticmethod
    def desired_rotation(yaw: float) -> np.ndarray:
        """Downward grasp: gripper +Z = mount +Z (fingers point down), gripper X (opening direction) heading = yaw."""
        x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        z = np.array([0.0, 0.0, 1.0])
        return np.column_stack([x, np.cross(z, x), z])

    def solve_ik(self, target_pos, yaw: float, q_seed, iters: int = 40, damping: float = 1e-4, w_rot: float = 0.05,
                 pos_tol: float = 1.5e-3, tilt_tol_deg: float = 4.0, yaw_tol_deg: float = 6.0):
        """Damped least squares on position + orientation error (analytic Jacobian), joint limits clamped.

        Returns (q, ok, info). ok requires the final pose to be within tolerance; q is always inside limits.
        """
        q = np.clip(np.asarray(q_seed, float).copy(), self.lower, self.upper)
        target_pos = np.asarray(target_pos, float)
        rd = self.desired_rotation(yaw)
        prev, stalled = np.inf, 0
        for _ in range(iters):
            ee, axes, origins = self._fk_with_axes(q)
            p, r = ee[:3, 3], ee[:3, :3]
            e_rot = 0.5 * sum(np.cross(r[:, i], rd[:, i]) for i in range(3))
            e = np.concatenate([target_pos - p, w_rot * e_rot])
            if np.linalg.norm(e[:3]) < pos_tol * 0.1 and np.linalg.norm(e_rot) < 1e-3:
                break
            # out-of-reach targets stop improving within a few steps: stop instead of burning all iterations
            err = float(np.linalg.norm(e))
            stalled = stalled + 1 if err > prev * 0.999 else 0
            if stalled >= 3:
                break
            prev = min(prev, err)
            jac = np.zeros((6, 5))
            for i in range(5):
                jac[:3, i] = np.cross(axes[i], p - origins[i])
                jac[3:, i] = w_rot * axes[i]
            step = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), e)
            n = np.linalg.norm(step)
            if n > 0.3:
                step *= 0.3 / n
            q = np.clip(q + step, self.lower, self.upper)
        t = self.ee_pose_mount(q)
        pos_err = float(np.linalg.norm(target_pos - t[:3, 3]))
        tilt = float(np.degrees(np.arccos(np.clip(t[2, 2], -1, 1))))
        heading = np.arctan2(t[1, 0], t[0, 0])
        yaw_err = float(np.degrees(abs((heading - yaw + np.pi) % (2 * np.pi) - np.pi)))
        ok = pos_err < pos_tol and tilt < tilt_tol_deg and yaw_err < yaw_tol_deg
        return q, ok, {"pos_err": pos_err, "tilt_deg": tilt, "yaw_err_deg": yaw_err}

    def solve_ik_tilt(self, target_pos, yaw: float, q_seed, max_tilt_deg: float, iters: int = 60,
                      pos_tol: float = 1.5e-3, roll_tol_deg: float = 6.0, step_max: float = 0.3,
                      null_damping: float = 1e-2, null_step_max: float = 0.05):
        """Downward grasp that may tilt when a vertical gripper cannot reach the position.

        Task priority: position (3) and the finger-opening angle about the gripper axis (1) are the primary task; the
        remaining degree of freedom of the 5-joint arm, the tilt of the gripper axis away from vertical, is driven
        toward zero in the null space of the primary task. Where a vertical gripper reaches, the result equals
        solve_ik; beyond that the gripper tilts only as much as needed (at most max_tilt_deg). Joints at a limit are
        locked for the step so the remaining joints keep solving the primary task.

        The finger-opening angle is defined like solve_ik's yaw for a vertical gripper; for a tilted gripper it is the
        same frame carried along by the smallest rotation that tilts mount +Z onto the gripper +Z axis.
        Returns (q, ok, info) like solve_ik, with info["tilt_deg"] the final tilt.
        """
        q = np.clip(np.asarray(q_seed, float).copy(), self.lower, self.upper)
        target_pos = np.asarray(target_pos, float)
        x_vert = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        z_up = np.array([0.0, 0.0, 1.0])

        def errors(r):
            xg, zg = r[:, 0], r[:, 2]
            # the vertical frame tilted by the smallest rotation taking +Z onto zg: desired finger direction
            axis = _cross(z_up, zg)
            s, c = np.linalg.norm(axis), float(zg[2])
            if s > 1e-9:
                k = axis / s
                x_des = x_vert * c + _cross(k, x_vert) * s + k * np.dot(k, x_vert) * (1 - c)
            else:
                x_des = x_vert
            e_roll = float(np.arctan2(np.dot(_cross(xg, x_des), zg), np.dot(xg, x_des)))
            e_vert = _cross(zg, z_up)  # rotation that brings the gripper axis back to vertical
            return e_roll, e_vert

        prev, stalled = np.inf, 0
        for _ in range(iters):
            ee, axes, origins = self._fk_with_axes(q)
            p, r = ee[:3, 3], ee[:3, :3]
            e_roll, e_vert = errors(r)
            e1 = np.concatenate([target_pos - p, [e_roll]])
            if np.linalg.norm(e1[:3]) < pos_tol * 0.1 and abs(e_roll) < 1e-3 and np.linalg.norm(e_vert) < 1e-3:
                break
            err = float(np.linalg.norm(e1[:3]) + 0.01 * abs(e_roll) + 0.001 * np.linalg.norm(e_vert))
            stalled = stalled + 1 if err > prev * 0.999 else 0
            if stalled >= 3:
                break
            prev = min(prev, err)
            jp = _cross(axes, p - origins).T
            jr = axes.T
            j1_full = np.vstack([jp, r[:, 2] @ jr])
            free = np.ones(5, bool)
            for _k in range(5):
                w = free.astype(float)
                j1 = j1_full * w
                u, sv, vt = np.linalg.svd(j1)
                rank = int(np.sum(sv > 1e-3))
                inv = np.where(sv > 1e-3, sv / (sv * sv + 1e-6), 0.0)
                dq = (vt[:len(sv)].T * inv) @ (u.T @ e1)
                n = np.linalg.norm(dq)
                if n > step_max:
                    dq *= step_max / n
                null = vt[rank:].T * w[:, None]  # null-space basis of the primary task (free joints only)
                if null.size:
                    # damped: at the edge of the vertical reach the tilt barely changes along the null space, and an
                    # undamped solve would ask for a huge null-space step there
                    a = jr @ null
                    c = np.linalg.solve(a.T @ a + null_damping * np.eye(a.shape[1]), a.T @ (e_vert - jr @ dq))
                    d2 = null @ c
                    n2 = np.linalg.norm(d2)
                    if n2 > null_step_max:
                        d2 *= null_step_max / n2
                    # the secondary step must not carry a joint past its limit: clipping it afterwards would undo
                    # part of the step the other joints already compensate for, and the position error would
                    # oscillate (seen at the wrist limit when the gripper starts to tilt). Shorten it instead.
                    alpha = 1.0
                    for i in np.nonzero(np.abs(d2) > 1e-12)[0]:
                        room = (self.upper[i] if d2[i] > 0 else self.lower[i]) - (q[i] + dq[i])
                        alpha = min(alpha, max(0.0, room / d2[i]))
                    dq = dq + alpha * d2
                # a joint whose primary step would cross its limit is locked for this step and the step is re-solved
                stuck = free & (((q + dq > self.upper + 1e-9) & (dq > 0)) | ((q + dq < self.lower - 1e-9) & (dq < 0)))
                if not stuck.any():
                    break
                free &= ~stuck
            q = np.clip(q + dq, self.lower, self.upper)
            # converged while tilted: the primary task is met and the tilt no longer changes
            if np.linalg.norm(e1[:3]) < pos_tol * 0.1 and abs(e_roll) < 1e-3 and np.linalg.norm(dq) < 1e-4:
                break
        t = self.ee_pose_mount(q)
        pos_err = float(np.linalg.norm(target_pos - t[:3, 3]))
        tilt = float(np.degrees(np.arccos(np.clip(t[2, 2], -1, 1))))
        e_roll, _ = errors(t[:3, :3])
        roll_err = float(np.degrees(abs(e_roll)))
        ok = pos_err < pos_tol and roll_err < roll_tol_deg and tilt <= max_tilt_deg
        return q, ok, {"pos_err": pos_err, "tilt_deg": tilt, "yaw_err_deg": roll_err}

    def pan_origin_mount(self) -> np.ndarray:
        """Pan (first joint) axis origin in the mount frame; used for radial yaw and reach limits."""
        return (self.mount_from_base @ self.joints[0]._j0)[:3, 3]
