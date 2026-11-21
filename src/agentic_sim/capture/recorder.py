"""Episode lifecycle for teleop demonstrations (issue #13), engine independent (pure Python).

The session calls `frame(...)` once per control step with that step's command, state and blocks; images are read
only on recorded steps (`grab` callable). One episode starts with the session and with every Reset; it ends
  - success: the target-colour block was lifted out of the storage tray, now rests in the placement tray with the
    jaw command open, for hold_s (status "complete", result success);
  - Reset or the window closing first: status "complete" with the result evaluated at that moment (success false,
    reason "reset" / "closed").
After a success nothing is recorded until the next Reset (it starts the next episode).
"""

from __future__ import annotations

import datetime
import os
import subprocess
import time

import numpy as np

from agentic_sim.capture.episode import EpisodeWriter
from agentic_sim.real_joints import shared_to_lerobot
from agentic_sim.scene_layout import block_in_tray

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def revision() -> dict:
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=5).stdout.strip())
        if commit:
            return {"commit": commit, "working_tree_dirty": dirty}
    except Exception:  # noqa: BLE001
        pass
    # no usable git (the Isaac container: missing, or the mounted repo is owned by another user)
    return {"commit": _head_from_files(), "working_tree_dirty": None,
            "note": "git not usable here: commit read from .git, working tree state unknown"}


def _head_from_files():
    """HEAD commit without git (the Isaac container has none; the repo is mounted with its .git)."""
    try:
        git = os.path.join(ROOT, ".git")
        head = open(os.path.join(git, "HEAD")).read().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        p = os.path.join(git, ref)
        if os.path.exists(p):
            return open(p).read().strip()
        for line in open(os.path.join(git, "packed-refs")):
            if line.strip().endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


class DemoRecorder:
    def __init__(self, out_dir: str, *, scene_path: str, experiment: str | None, target_color: str,
                 block_colors: list, trays: dict, edge: float, gripper: dict, cameras: dict, control_dt: float,
                 record_hz: float, operator: str, engine: str = "isaac", lens_mode: str | None = None,
                 hold_s: float = 1.0, lift_m: float = 0.01, log=print, writer_cls=EpisodeWriter):
        self.out_dir = out_dir
        self.log = log
        self.target_color = target_color
        self.block_colors = list(block_colors)
        self.targets = [i for i, c in enumerate(self.block_colors) if c == target_color]
        if not self.targets:
            raise ValueError(f"no {target_color} block among {self.block_colors}")
        self.trays, self.edge, self.gripper = trays, edge, gripper
        self.cameras = cameras
        self.control_dt = control_dt
        self.stride = max(1, int(round(1.0 / (record_hz * control_dt))))
        self.hold_steps = int(round(hold_s / control_dt))
        self.lift_m = lift_m
        self.writer_cls = writer_cls
        self.base = dict(capture="teleop", source="sim", engine=engine, operator=operator, scene=scene_path,
                         experiment=experiment, target_color=target_color, block_colors=self.block_colors,
                         control_dt=control_dt, record_hz=1.0 / (self.stride * control_dt), record_stride=self.stride,
                         lens_mode=lens_mode, gripper=gripper,
                         trays={k: {kk: (list(vv) if isinstance(vv, (tuple, list, np.ndarray)) else vv)
                                    for kk, vv in t.items()} for k, t in trays.items()},
                         block_edge=edge, revision=revision())
        self.w = None
        self.episodes = []          # manifests of the finished episodes
        self._new()

    # ---- lifecycle ---------------------------------------------------------------------------------------------
    def _new(self):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eid = f"{stamp}_{self.target_color}"
        root = os.path.join(self.out_dir, eid)
        k = 1
        while os.path.exists(root):
            k += 1
            root = os.path.join(self.out_dir, f"{eid}_{k}")
        self.w = self.writer_cls(root, dict(self.base, episode_id=os.path.basename(root),
                                            started=datetime.datetime.now().isoformat(timespec="seconds")),
                                 self.cameras)
        self.step0 = None
        self.init_z = None
        self.lifted = {i: False for i in self.targets}
        self.ok_run = 0
        self.last_pos = None
        self.log(f"recording episode {self.w.root} (target {self.target_color}, every {self.stride} control steps)")

    def _finish(self, success: bool, reason: str, placed=None, last_step=None):
        if self.w is None:
            return None
        res = {"success": success, "reason": reason, "target_block": placed if placed is not None else self.targets[0],
               "lifted": any(self.lifted.values())}
        dur = 0.0 if self.step0 is None or last_step is None else (last_step - self.step0) * self.control_dt
        man = self.w.close("complete" if self.w.n else "empty", res, {"duration_s": round(dur, 3)})
        self.log(f"episode {man['episode_id']}: {'success' if success else 'not successful'} ({reason}), "
                 f"{man['control_frames']} frames, {dur:.1f} s -> {self.w.root}")
        self.episodes.append(dict(man, path=self.w.root))
        self.w = None
        return man

    def reset(self):
        """Reset pressed: end the current episode (if still open) and start the next one."""
        if self.w is not None:
            self._finish(False, "reset", last_step=self._last_step)
        self._new()

    def close(self):
        if self.w is not None:
            self._finish(False, "closed", last_step=self._last_step)

    @property
    def recording(self) -> bool:
        return self.w is not None

    # ---- per control step --------------------------------------------------------------------------------------
    _last_step = None

    def frame(self, step: int, t_sim: float, *, paused: bool, focused: bool, source: str, ik_status: str,
              cmd_ee: np.ndarray, cmd_yaw: float, cmd_q: np.ndarray, q: np.ndarray, qd: np.ndarray, ee: np.ndarray,
              blocks: list, grab):
        """blocks: (pos, quat wxyz) per block. grab(): camera key -> (image, capture_ns), called on recorded steps."""
        if self.w is None:
            return
        imgs = None
        if self.step0 is None:          # the episode starts at the first step every camera delivers an image
            imgs = grab()
            if any(v is None for v in imgs.values()):
                return
            self.step0 = step
            self.init_z = {i: float(blocks[i][0][2]) for i in self.targets}
        self._last_step = step
        if (step - self.step0) % self.stride == 0:
            imgs = imgs if imgs is not None else grab()
            row = {"step": int(step), "t_sim": round(float(t_sim), 6), "wall_ns": time.time_ns(),
                   "source": source, "paused": bool(paused), "focused": bool(focused), "ik_status": ik_status,
                   "command": {"ee_pos": [round(float(x), 6) for x in cmd_ee], "ee_yaw_offset": round(float(cmd_yaw), 6),
                               "gripper": round(float(cmd_q[5]), 6), "joint_pos": [round(float(x), 6) for x in cmd_q]},
                   "state": {"joint_pos": [round(float(x), 6) for x in q], "joint_vel": [round(float(x), 5) for x in qd],
                             "ee_pos": [round(float(x), 6) for x in ee]},
                   "target": [round(x, 4) for x in shared_to_lerobot(cmd_q[:5], cmd_q[5])],
                   "readback": [round(x, 4) for x in shared_to_lerobot(q[:5], q[5])],
                   "blocks": [{"color": c, "pos": [round(float(x), 6) for x in p], "quat": [round(float(x), 6) for x in r]}
                              for c, (p, r) in zip(self.block_colors, blocks)],
                   "capture_ns": {k: v[1] for k, v in imgs.items() if v is not None}}
            self.w.add(row, {k: (None if v is None else v[0]) for k, v in imgs.items()})
        # success: a lifted target block resting in the placement tray, jaw command open, for hold_s
        mid = 0.5 * (self.gripper["open"] + self.gripper["closed"])
        placed = None
        for i in self.targets:
            p = np.asarray(blocks[i][0], float)
            if p[2] > self.init_z[i] + self.lift_m:
                self.lifted[i] = True
            moving = self.last_pos is not None and np.linalg.norm(p - self.last_pos[i]) / self.control_dt > 0.02
            if self.lifted[i] and not moving and block_in_tray(p, self.trays["placement_beige"], self.edge):
                placed = i
        self.last_pos = {i: np.asarray(blocks[i][0], float) for i in self.targets}
        open_cmd = (cmd_q[5] > mid) == (self.gripper["open"] > mid)
        self.ok_run = self.ok_run + 1 if (placed is not None and open_cmd and not paused) else 0
        if self.ok_run >= self.hold_steps:
            self._finish(True, "placed", placed=placed, last_step=step)
            self.log("press R (Reset) to record the next episode")
