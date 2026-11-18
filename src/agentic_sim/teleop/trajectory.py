"""Scripted EE target trajectory for cross-engine comparison (pure Python): the same target sequence is fed to the
shared EE controller in every engine, without any UI input, sampled at *simulated* time.

Trajectory file (configs/trajectories/<name>.yaml):
  name: pick_place
  start: {pos: [0.15, 0.0, 0.09], yaw: 0.0, gripper: 0.52}
  segments:                      # each one moves from the previous keyframe to `to` over `t` seconds
    - {t: 1.0}                                         # hold
    - {t: 1.5, pos: [0.17, 0.115, 0.09], yaw: -0.59}   # move (smooth start and stop)
    - {t: 1.0, gripper: -0.1745}                       # close the gripper
Keys left out keep their previous value. `yaw` is the operator yaw offset (rad), `gripper` the jaw command (rad).
Interpolation is a smoothstep, so the target has zero velocity at every keyframe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentic_sim.config import load


@dataclass
class Keyframe:
    t: float
    pos: np.ndarray
    yaw: float
    gripper: float
    label: str = ""


class Trajectory:
    def __init__(self, raw: dict, path: str = ""):
        self.name = str(raw.get("name", path))
        self.path = path
        s = raw["start"]
        k = Keyframe(0.0, np.array(s["pos"], float), float(s.get("yaw", 0.0)), float(s["gripper"]), "start")
        self.keys = [k]
        for seg in raw["segments"]:
            dt = float(seg["t"])
            if dt <= 0:
                raise ValueError(f"{path}: segment duration must be > 0: {seg}")
            prev = self.keys[-1]
            self.keys.append(Keyframe(prev.t + dt, np.array(seg.get("pos", prev.pos), float),
                                      float(seg.get("yaw", prev.yaw)), float(seg.get("gripper", prev.gripper)),
                                      str(seg.get("label", ""))))
        self.duration = self.keys[-1].t

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        return cls(load(path), path)

    def phase(self, t: float) -> str:
        """Label of the segment containing t (the last labelled one at or before it)."""
        lab = ""
        for k0, k1 in zip(self.keys[:-1], self.keys[1:]):
            if k1.label and k0.t <= t:
                lab = k1.label
        return lab

    def sample(self, t: float):
        """(pos, yaw, gripper) at time t (clamped to the trajectory)."""
        t = float(np.clip(t, 0.0, self.duration))
        for k0, k1 in zip(self.keys[:-1], self.keys[1:]):
            if t <= k1.t:
                u = (t - k0.t) / (k1.t - k0.t)
                s = u * u * (3 - 2 * u)
                return k0.pos + (k1.pos - k0.pos) * s, k0.yaw + (k1.yaw - k0.yaw) * s, \
                    k0.gripper + (k1.gripper - k0.gripper) * s
        k = self.keys[-1]
        return k.pos.copy(), k.yaw, k.gripper

    def done(self, t: float) -> bool:
        return t >= self.duration


class TrajectoryDriver:
    """Feeds a trajectory into an EETargetController each control frame (instead of keyboard / mouse input)."""

    def __init__(self, traj: Trajectory, ctrl):
        self.traj = traj
        self.ctrl = ctrl
        self.rejected_frames = 0

    def apply(self, t_sim: float):
        pos, yaw, grip = self.traj.sample(t_sim)
        s = self.ctrl.state
        if not np.allclose(pos, s.pos, atol=1e-9) or abs(yaw - s.yaw_offset) > 1e-12:
            if not self.ctrl.propose(pos, yaw, "trajectory"):
                self.rejected_frames += 1
        s.gripper = float(np.clip(grip, self.ctrl.p.gripper_closed, self.ctrl.p.gripper_open))
        s.gripper_goal = None

    def done(self, t_sim: float) -> bool:
        return self.traj.done(t_sim)


def make_driver(path: str, ctrl):
    """configs/trajectories/*.yaml -> TrajectoryDriver; a run's CSV -> CommandReplay."""
    if path.endswith(".csv"):
        return CommandReplay(path, ctrl)
    return TrajectoryDriver(Trajectory.load(path), ctrl)


def trajectory_result(engine: str, session, summary: dict, block_in_placement: bool) -> dict:
    """Summary of a trajectory run; both engines write the same fields (session: last_frame, stats, t_sim, frames,
    traj)."""
    f = session.last_frame
    return {"engine": engine, "trajectory": session.traj.traj.name, "file": session.traj.traj.path,
            "t_sim": round(float(session.t_sim), 4), "frames": int(session.frames),
            "rejected_frames": int(session.traj.rejected_frames),
            "block_final": [round(float(x), 5) for x in f["block"]], "jaw_final": round(float(f["jaw"]), 4),
            "block_in_placement": bool(block_in_placement),
            "block_lifted_max_z": round(float(session.stats["block_lifted_max_z"]), 5), "metrics": summary}



class CommandReplay:
    """Replays the arm / jaw commands recorded in another run's CSV (columns t_sim, tx..tz, yaw_off, grip_cmd,
    qcmd_*), frame by frame, bypassing the IK: both engines then receive exactly the same joint commands, so the
    remaining differences are physics only. Same interface as TrajectoryDriver."""

    def __init__(self, csv_path: str, ctrl):
        import csv
        from agentic_sim.kinematics import ARM_JOINTS
        rows = list(csv.DictReader(open(csv_path)))
        self.t = np.array([float(r["t_sim"]) for r in rows])
        self.q = np.array([[float(r[f"qcmd_{j}"]) for j in ARM_JOINTS] for r in rows])
        self.grip = np.array([float(r["grip_cmd"]) for r in rows])
        self.pos = np.array([[float(r[k]) for k in ("tx", "ty", "tz")] for r in rows])
        self.yaw = np.array([float(r["yaw_off"]) for r in rows])
        self.ctrl = ctrl
        self.rejected_frames = 0
        self.traj = type("Replay", (), {"name": "replay:" + csv_path, "path": csv_path})()
        self.duration = float(self.t[-1])
        self._i = 0

    def apply(self, t_sim: float):
        # row k holds the command sent in control frame k; apply() is called once per (unpaused) frame. The logged
        # t_sim is rounded, so rows are taken in order rather than looked up by time.
        i = min(self._i, len(self.t) - 1)
        self._i += 1
        s = self.ctrl.state
        s.q_goal = self.q[i].copy()
        s.q_cmd = self.q[i].copy()   # joint_command() then keeps it (goal == command)
        s.pos, s.yaw_offset = self.pos[i].copy(), float(self.yaw[i])
        s.gripper, s.gripper_goal = float(self.grip[i]), None

    def done(self, t_sim: float) -> bool:
        return self._i >= len(self.t)
