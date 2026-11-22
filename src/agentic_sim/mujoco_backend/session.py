"""One MuJoCo teleop session: scene model + shared EE controller, stepped one control frame at a time.

Same behaviour as the Isaac session (agentic_sim.isaac.session): the shared EETargetController (IK, limits, boundary
handling, rejection reasons, rate limits) proposes arm joint targets; keys come through agentic_sim.teleop.keys; input
is dropped on focus loss, pause and reset. The arm is driven by the menagerie position actuators.

  s = MjSession(SessionConfig(...), log, keys=KeyState(), focused=lambda: True)
  while s.step(): ...            # one control frame (real-time dt from the clock, or step(dt=...) when headless)
  s.finish()                     # CSV / result JSON
No window here: agentic_sim.mujoco_backend.viewer draws the session and feeds its keys.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from agentic_sim.color_dr import ColorSampler, Palette
from agentic_sim.config import REPO_ROOT, load, provisional_fields, v
from agentic_sim.wrist_lens import apply_wrist_lens
from agentic_sim.kinematics import ARM_JOINTS, So101Chain
from agentic_sim.mujoco_backend.model import ROBOT_CONFIG, build
from agentic_sim.scene_layout import block_count_range, block_in_tray
from agentic_sim.teleop.ee_target import REASONS, load_controller
from agentic_sim.mujoco_backend.handle import Handle
from agentic_sim.mujoco_backend.panel import MjPanel
from agentic_sim.teleop.keys import HELP, KeyState

MJ_HELP = HELP.replace("  M: focus view", "")

CSV_HEADER = ["t_wall", "t_sim", "session", "paused", "focused", "source", "status", "keys",
              "tx", "ty", "tz", "yaw_off", "ex", "ey", "ez", "bx", "by", "bz", "grip_cmd", "jaw",
              *[f"qcmd_{n}" for n in ARM_JOINTS], *[f"q_{n}" for n in ARM_JOINTS], "max_dq_cmd", "max_qd", "os_focus",
              "status_msg", "follow_err_mm", "tilt_deg", "step_ms", "input_lat_ms", "mouse_dx", "mouse_dy", "mouse_dz",
              "fine", "handle_drag"]


@dataclass
class SessionConfig:
    scene: str = "configs/scenes/lab_current_minimal.yaml"
    teleop: str = "configs/teleop.yaml"
    color_dr_config: str = "configs/color_randomization.yaml"
    color_dr: bool = False
    color_dr_seed: int | None = None
    color_dr_log: str | None = None
    blocks: int | None = None
    script: str | None = None
    log: str | None = None
    result: str | None = None
    trajectory: str | None = None      # scripted EE target sequence (cross-engine comparison); replaces all input
    block_colors: list | None = None   # colour name per block (task scenes); sets the block count
    task: str | None = None            # task scenario (configs/tasks/*.yaml): the executor drives the arm
    task_log: str | None = None        # executor event log (JSONL)
    random_blocks: bool = False
    random_seed: int | None = None
    random_region: str = "storage"
    random_config: str = "configs/random_blocks.yaml"
    random_episode: int = 0           # advanced on Reset; seed + episode reproduce the layout
    wrist_lens: str | None = None      # wrist camera image mode ("real" | "pinhole"); None = scene default
    live_perception: bool = False
    perception_camera: str = "both"
    perception_hz: float = 2.0
    perception_json: str | None = None
    task_perception: bool = False      # the executor finds blocks with the cameras (Acquire), not the sim truth


class MjColorRandomizer:
    """Colour DR on the MuJoCo model: same sampler, palette, draw order and log format as the Isaac version.

    MuJoCo's renderer takes rgba as given (no linear conversion), so the palette's sRGB values are written as they
    are. Arm shell = every menagerie material except the motors, gripper fingers included (as in Isaac)."""

    def __init__(self, cfg_path, model, block_geoms, shell_materials, enabled=None, seed=None, log_path=None,
                 log=print):
        self.cfg = load(cfg_path)
        self.m, self.block_geoms, self.shell = model, list(block_geoms), list(shell_materials)
        self._log = log
        self.palette = Palette.load(self.cfg["palette"])
        self.block_baselines = [tuple(model.geom_rgba[g][:3]) for g in self.block_geoms]
        self.shell_baselines = [tuple(model.mat_rgba[i][:3]) for i in self.shell]
        p = log_path or self.cfg.get("log")
        self.log_path = None if not p else (p if os.path.isabs(p) else os.path.join(REPO_ROOT, p))
        self._log_opened = False
        self.records, self.current, self.sampler = [], None, None
        self.enabled = False
        self.seed = int(self.cfg["seed"] if seed is None else seed)
        self.configure(bool(self.cfg["enabled"] if enabled is None else enabled), self.seed)

    def configure(self, enabled: bool, seed: int | None = None) -> bool:
        seed = self.seed if seed is None else int(seed)
        if bool(enabled) == self.enabled and (seed == self.seed or not enabled) and (self.sampler or not enabled):
            self.seed = seed
            return False
        self.enabled, self.seed = bool(enabled), seed
        if not self.enabled:
            self.sampler = self.current = None
            for g, rgb in zip(self.block_geoms, self.block_baselines):
                self.m.geom_rgba[g][:3] = rgb
            for i, rgb in zip(self.shell, self.shell_baselines):
                self.m.mat_rgba[i][:3] = rgb
            return True
        self.sampler = ColorSampler(self.palette, self.seed, bool(self.cfg["block"]), bool(self.cfg["arm_shell"]))
        if self.log_path and not self._log_opened:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            open(self.log_path, "w").close()
            self._log_opened = True
        self._log(f"colour DR: enabled, seed={self.seed}, palette={self.palette.version}, log={self.log_path}")
        return True

    def apply_next(self):
        if not self.enabled:
            return None
        rec = self.sampler.next(len(self.block_geoms))
        blocks = ([rec["block"]] + rec.get("extra_blocks", [])) if "block" in rec else []
        for g, b in zip(self.block_geoms, blocks):
            self.m.geom_rgba[g][:3] = b["rgb"]
            b["written"] = list(b["rgb"])  # MuJoCo takes the sRGB value as is (Isaac writes the linear value)
        if "arm_shell" in rec:
            for i in self.shell:
                self.m.mat_rgba[i][:3] = rec["arm_shell"]["rgb"]
            rec["arm_shell"]["written"] = list(rec["arm_shell"]["rgb"])
        rec["applied"] = {"blocks_rgb": [[round(float(x), 6) for x in self.m.geom_rgba[g][:3]] for g in self.block_geoms],
                          "arm_shell_rgb": [round(float(x), 6) for x in self.m.mat_rgba[self.shell[0]][:3]]}
        self.records.append(rec)
        self.current = rec
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        self._log(f"colour DR: session {rec['session']} blocks={[b['name'] for b in blocks]} "
                  f"arm_shell={rec.get('arm_shell', {}).get('name')}")
        return rec

    def block_rgbs(self, baselines) -> list:
        """Display colour of every block: the drawn colours, or the given baselines when off."""
        if self.current and "block" in self.current:
            return [tuple(b["rgb"]) for b in [self.current["block"]] + self.current.get("extra_blocks", [])]
        return [tuple(b) for b in baselines]

    def summary(self) -> str:
        if not self.enabled:
            return "baseline colours (DR off)"
        r = self.current or {}
        names = [x["name"] for x in ([r["block"]] + r.get("extra_blocks", []))] if "block" in r else []
        return f"draw #{r.get('session', 0)} seed {self.seed}: blocks {'/'.join(names[:3])}" + \
               ("..." if len(names) > 3 else "") + (f", arm {r['arm_shell']['name']}" if "arm_shell" in r else "")


def _pct(xs):
    if not xs:
        return None
    a = np.asarray(xs)
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}


class MjSession:
    def __init__(self, cfg: SessionConfig, log=print, keys: KeyState | None = None, focused=None, script=None,
                 viewer=None):
        """script: a running script carried over from the previous session (scene rebuilt for another block
        count); viewer: the window that scripted mouse steps go through (None: keyboard-only scripts)."""
        self.cfg = cfg
        self.log = log
        self.keys = keys or KeyState(actions={"P": "pause", "R": "reset"})
        self._focused = focused or (lambda: True)
        self._requests: list[str] = []
        self._finished = None
        self.scene_cfg = apply_wrist_lens(load(cfg.scene), cfg.wrist_lens)
        from agentic_sim.random_blocks import prepare_session
        cfg, self.scene_cfg, spawn_colors, self.random_spawn = prepare_session(cfg, self.scene_cfg, log)
        self.cfg = cfg
        teleop_cfg = load(cfg.teleop)
        from agentic_sim.task.runner import scenario_colors
        self.block_color_names = spawn_colors or cfg.block_colors or scenario_colors(cfg.task)
        self.sc = build(self.scene_cfg, cfg.blocks, colors=self.block_color_names)
        self.m = self.sc.model
        self.d = mujoco.MjData(self.m)
        rc = load(ROBOT_CONFIG)["physics"]
        self.control_dt = float(rc["control_dt"])
        self.substeps = max(1, int(round(self.control_dt / self.m.opt.timestep)))
        self.chain = So101Chain.from_dict(load("configs/so101_chain.yaml"))
        self.ctrl = load_controller(self.chain, teleop_cfg, base_pos=v(self.scene_cfg["robot"]["base_pos"]),
                                    scene=self.scene_cfg)
        self.color_dr = MjColorRandomizer(cfg.color_dr_config, self.m, self.sc.block_geoms, self.sc.shell_materials,
                                          enabled=True if cfg.color_dr else None, seed=cfg.color_dr_seed,
                                          log_path=cfg.color_dr_log, log=log)
        self.carry = {"rejected": 0, "reasons": {r: 0 for r in REASONS}}
        self.placement = self.sc.trays["placement_beige"]
        self.storage = self.sc.trays["storage_pink"]
        self.rebuild_request: dict | None = None
        self.reset_view_requested = False
        self.camera_position = lambda: np.zeros(3)   # replaced by the viewer
        fine = lambda: self.ctrl.p.fine_drag_scale  # noqa: E731
        shift = lambda: self.keys.shift  # noqa: E731
        self.panel = MjPanel(self.settings(), block_count_range(self.scene_cfg)[1], fine, shift, MJ_HELP)
        self.handle = Handle(fine, shift)
        self.script = script
        if self.script is None and cfg.script:
            from agentic_sim.mujoco_backend.script import MjScript
            self.script = MjScript(load(cfg.script), self.keys, log)
        if self.script is not None:
            self.script.rebind(self, viewer)
        self.traj = None
        if cfg.trajectory:
            from agentic_sim.teleop.trajectory import make_driver
            self.traj = make_driver(cfg.trajectory, self.ctrl)
        self.task = None
        log(f"MuJoCo {mujoco.__version__}, menagerie SO-101 @{self.sc.robot_cfg['model']['menagerie_commit'][:7]}; "
            f"physics dt {self.m.opt.timestep:.5f} s x {self.substeps} per control frame; provisional scene fields: "
            f"{', '.join(provisional_fields(self.scene_cfg))}")
        log(f"EE offset in gripper link (provisional): {self.chain.ee_offset.tolist()}; "
            f"home q={np.round(self.ctrl.q_home, 3).tolist()}")
        log(MJ_HELP)
        self.do_reset()
        self.live_perception = None
        if cfg.live_perception:
            from agentic_sim.mujoco_backend.live_perception import MjLivePerception
            self.live_perception = MjLivePerception(self.scene_cfg, self.m, cfg.perception_hz, log,
                                                    cfg.perception_camera, cfg.perception_json)

        self.log_f = self.log_w = None
        if cfg.log:
            os.makedirs(os.path.dirname(cfg.log) or ".", exist_ok=True)
            cont = script is not None and os.path.exists(cfg.log)  # rebuilt mid-script: keep the rows before
            self.log_f = open(cfg.log, "a" if cont else "w", newline="")
            self.log_w = csv.writer(self.log_f)
            if not cont:
                self.log_w.writerow(CSV_HEADER)
        self.paused = False
        self.t_start = self.last = time.monotonic()
        self.t_sim = 0.0
        self.frames = 0
        self.stats = {"max_dq_cmd_per_frame": 0.0, "max_joint_vel": 0.0, "rejected_targets": 0,
                      "block_lifted_max_z": 0.0}
        self.metrics = {"follow_err_mm": [], "follow_err_idle_mm": [], "step_ms": [], "input_lat_ms": []}
        self.last_input_t = -1e9
        self.prev_target = self.ctrl.state.pos.copy()
        self.prev_status = "ok"
        self.last_frame = {}
        if cfg.task:
            self.start_task()

    # ---- task executor (issue #5) ----------------------------------------------------------------
    def start_task(self):
        """Hand the arm to the task executor (scenario from cfg.task, or orders from the server)."""
        from agentic_sim.task.runner import TaskRunner, default_color_names
        if self.task is not None:
            self.task.close()
        colors = self.block_color_names or default_color_names(self.scene_cfg, len(self.sc.block_bodies))
        cameras = None
        if self.cfg.task_perception:
            from agentic_sim.mujoco_backend.task_cameras import MjTaskCameras
            cameras = MjTaskCameras(self, log=self.log)
        self.task = TaskRunner(self.ctrl, self.scene_cfg, self.sc.block_init_pos, colors, self.control_dt, self.log,
                               scenario=self.cfg.task, event_log=self.cfg.task_log, cameras=cameras,
                               teleop=self.cfg.teleop)
        return self.task

    def stop_task(self):
        """Give the arm back to the operator (keyboard / mouse) where it is."""
        if self.task is not None:
            self.task.close()
            self.task = None
            self.ctrl.clear_gripper_goal()
            self.log("task stopped: operator input active")

    def task_obs(self):
        from agentic_sim.task.runner import quat_yaw
        from agentic_sim.task.skills import Obs
        q = self.arm_q()
        # arm speed from the measured joint positions frame to frame (same definition as the Isaac session)
        prev, self._task_q_prev = getattr(self, "_task_q_prev", None), q.copy()
        qd = 0.0 if prev is None else float(np.max(np.abs(q - prev))) / self.control_dt
        return Obs(t=self.t_sim, ee=self.ctrl.chain.ee_pose_mount(q)[:3, 3].copy(), qd_max=qd, jaw=self.jaw(),
                   blocks=self.block_positions(), block_yaw=[quat_yaw(self.d.xquat[b]) for b in self.sc.block_bodies],
                   block_speed=[float(np.linalg.norm(self.d.qvel[a:a + 3])) for a in self.sc.block_qvel])

    # ---- settings / commands --------------------------------------------------------
    def settings(self) -> dict:
        return {"blocks": len(self.sc.block_bodies), "color_dr": bool(self.color_dr.enabled),
                "color_dr_seed": int(self.color_dr.seed)}

    def max_blocks(self) -> int:
        return block_count_range(self.scene_cfg)[1]

    def _apply_settings(self) -> bool:
        """On reset: apply the panel's settings. Returns False if the scene must be rebuilt (block count)."""
        want = self.panel.settings()
        if self.cfg.random_blocks:
            want["color_dr"] = False  # spawn colours are the task colour identities
            want.update(random_seed=self.cfg.random_seed, random_episode=self.cfg.random_episode + 1)
            self.rebuild_request = want
            self.log(f"random scene rebuild requested: {want}")
            return False
        if want["blocks"] != len(self.sc.block_bodies):
            self.rebuild_request = want
            self.log(f"rebuild requested: {want}")
            return False
        self.color_dr.configure(want["color_dr"], want["color_dr_seed"])
        return True

    def handover_script(self):
        """Detach the running script (rebuild): the next session continues it; this one writes no result."""
        script, self.script = self.script, None
        return script

    def request(self, action: str):
        if action not in ("reset", "pause", "resume"):
            raise ValueError(f"unknown action {action!r}")
        self._requests.append(action)

    def _pop_requests(self) -> list:
        out = []
        for a in self._requests:
            if a == "reset":
                out.append("reset")
            elif (a == "pause") != self.paused:
                out.append("pause")
        self._requests = []
        return out

    # ---- state access ---------------------------------------------------------------
    def arm_q(self) -> np.ndarray:
        """Measured arm joints in the shared convention (MuJoCo offsets removed)."""
        return np.array([self.d.qpos[a] for a in self.sc.arm_qpos]) - self.sc.arm_offset

    def jaw(self) -> float:
        return float(self.d.qpos[self.sc.jaw_qpos])

    def block_positions(self) -> list:
        return [self.d.xpos[b].copy() for b in self.sc.block_bodies]

    def _command(self):
        s = self.ctrl.state
        for act, q, off in zip(self.sc.arm_act, s.q_cmd, self.sc.arm_offset):
            self.d.ctrl[act] = q + off
        self.d.ctrl[self.sc.jaw_act] = s.gripper

    def do_reset(self):
        ctrl, sc, d = self.ctrl, self.sc, self.d
        self.carry["rejected"] += ctrl.state.rejected
        for r in REASONS:
            self.carry["reasons"][r] += ctrl.state.reject_counts[r]
        ctrl.reset()
        mujoco.mj_resetData(self.m, d)
        for adr, q, off in zip(sc.arm_qpos, ctrl.state.q_cmd, sc.arm_offset):
            d.qpos[adr] = q + off
        d.qpos[sc.jaw_qpos] = ctrl.state.gripper
        for qa, pos, rot in zip(sc.block_qpos, sc.block_init_pos, sc.block_init_rots):
            d.qpos[qa:qa + 3] = pos
            d.qpos[qa + 3:qa + 7] = rot
        self._command()
        mujoco.mj_forward(self.m, d)
        for _ in range(30):  # settle (blocks land on the tray floor, arm holds home)
            mujoco.mj_step(self.m, d)
        self.color_dr.apply_next()
        if getattr(self, "live_perception", None) is not None:
            self.live_perception.invalidate()
        if hasattr(self, "panel"):
            self.panel.set_colors(self.color_dr.summary())
            self.panel.mark_applied(self.settings())
            self.panel.cancel()
            self.handle.cancel()
        self.keys.clear()
        self.log(f"reset: session={ctrl.state.session}")

    @property
    def script_done(self) -> bool:
        return ((self.script is not None and self.script.done) or (self.traj is not None and self.traj.done(self.t_sim))
                or (self.task is not None and self.task.done))

    def focused(self) -> bool:
        if self.script is not None and self.script.focus_override is not None:
            return bool(self.script.focus_override)
        return bool(self._focused())

    # ---- one control frame --------------------------------------------------------------
    def step(self, dt: float | None = None) -> bool:
        """One control frame. dt: input increment (default: wall time since the last frame, as in Isaac)."""
        ctrl, keys, stats = self.ctrl, self.keys, self.stats
        now = time.monotonic()
        if dt is None:
            dt = now - self.last
        self.last = now
        panel, handle = self.panel, self.handle
        if self.rebuild_request is not None:  # waiting for the owner to rebuild the scene
            return not self.script_done
        actions = keys.pop_actions() + panel.pop_actions() + self._pop_requests()
        if "reset" in actions:
            if not self._apply_settings():
                return not self.script_done
            self.do_reset()
            self.paused = False
            dt = 0.0
            if self.task is not None:   # a reset ends the task; a scenario run starts again from its first order
                self.stop_task()
                if self.cfg.task:
                    self.start_task()
        for a in actions:
            if a == "pause":
                self.paused = not self.paused
                keys.clear()
                panel.cancel()
                handle.cancel()
                ctrl.clear_yaw_goal()
                ctrl.clear_gripper_goal()
                self.log(f"pause={self.paused}")
            elif a == "reset_view":
                self.reset_view_requested = True
            elif a in ("open", "close") and not self.paused:
                ctrl.set_gripper(a == "open", "mouse")
            elif a == "toggle_gripper" and not self.paused:
                mid = 0.5 * (ctrl.p.gripper_open + ctrl.p.gripper_closed)
                ctrl.set_gripper(ctrl.state.gripper < mid, "mouse")
            elif a in ("yaw-", "yaw+") and not self.paused:
                ctrl.nudge_yaw(math.radians(15) * (1 if a == "yaw+" else -1), "mouse")
        focused = self.focused()
        if not focused:
            keys.clear()
            panel.cancel()
            handle.cancel()
            if self.task is None:  # operator goals only: a running task keeps its own gripper / yaw goals
                ctrl.clear_yaw_goal()
                ctrl.clear_gripper_goal()
        max_dq_cmd, step_ms = 0.0, 0.0
        input_lat_ms, mouse_d = None, np.zeros(3)
        if self.paused:
            keys.clear()  # input received while paused is dropped, so nothing replays on resume
            panel.cancel()
            handle.cancel()
        else:
            t0 = time.perf_counter()
            if self.traj is not None:  # comparison run: the trajectory is the only input
                keys.clear()
                panel.cancel()
                handle.cancel()
                self.traj.apply(self.t_sim)
            if self.task is not None:  # the executor is the only control source while a task runs
                keys.clear()
                panel.cancel()
                handle.cancel()
                self.task.step(self.task_obs())
            lin, yaw, grip = keys.directions()
            ctrl.step_velocity(lin, yaw, dt, "keyboard")
            ctrl.step_gripper(grip, dt, "keyboard")
            # mouse: 3D handle (primary) + top view (auxiliary), both relative, one IK update per frame
            d1, t1 = handle.pop_drag()
            d2, t2 = panel.pop_drag()
            w1, w2 = handle.pop_wheel(), panel.pop_wheel()
            p_before = ctrl.state.pos.copy()
            drag = d1 + d2
            if np.linalg.norm(drag) > 0:
                ctrl.move_by([drag[0], drag[1], 0.0], "mouse")
            if w1["coarse"] + w2["coarse"]:
                ctrl.wheel(w1["coarse"] + w2["coarse"])
            if w1["fine"] + w2["fine"]:
                ctrl.wheel(w1["fine"] + w2["fine"], fine=True)
            ctrl.step_goals(dt)
            times = t1 + t2
            if times:
                input_lat_ms = (time.monotonic() - min(times)) * 1000.0
                self.metrics["input_lat_ms"].append(input_lat_ms)
                mouse_d = ctrl.state.pos - p_before
            q_prev = ctrl.state.q_cmd.copy()
            ctrl.joint_command(self.control_dt)
            max_dq_cmd = float(np.max(np.abs(ctrl.state.q_cmd - q_prev)))
            self._command()
            mujoco.mj_step(self.m, self.d, nstep=self.substeps)
            self.t_sim += self.control_dt
            step_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics["step_ms"].append(step_ms)

        q_meas = self.arm_q()
        ee_pose = ctrl.chain.ee_pose_mount(q_meas)
        ee = ee_pose[:3, 3].copy()
        tilt_deg = float(np.degrees(np.arccos(np.clip(ee_pose[2, 2], -1.0, 1.0))))
        all_pos = self.block_positions()
        bpos = all_pos[0]
        edge = self.sc.edge
        n_placed = sum(block_in_tray(p, self.placement, edge) for p in all_pos)
        s = ctrl.state
        follow_mm = float(np.linalg.norm(s.pos - ee)) * 1000.0
        target_step = float(np.linalg.norm(s.pos - self.prev_target))
        if target_step > 1e-9 or keys.held or handle.dragging:
            self.last_input_t = now
        self.metrics["follow_err_mm"].append(follow_mm)
        if now - self.last_input_t > 1.0 and not self.paused:
            self.metrics["follow_err_idle_mm"].append(follow_mm)
        self.prev_target = s.pos.copy()
        if s.status != self.prev_status:
            self.log(f"status {self.prev_status} -> {s.status} {s.message}")
            self.prev_status = s.status
        qd_max = float(np.max(np.abs([self.d.qvel[a] for a in self.sc.arm_qvel])))
        stats["max_dq_cmd_per_frame"] = max(stats["max_dq_cmd_per_frame"], max_dq_cmd)
        stats["max_joint_vel"] = max(stats["max_joint_vel"], qd_max)
        stats["rejected_targets"] = self.carry["rejected"] + s.rejected
        stats["block_lifted_max_z"] = max(stats["block_lifted_max_z"], float(bpos[2]))
        jaw = self.jaw()
        self.last_frame = {"target": s.pos.copy(), "ee": ee, "block": bpos, "q": q_meas, "jaw": jaw,
                           "status": s.status, "message": s.message, "focused": focused, "blocks": all_pos,
                           "n_placed": n_placed, "tilt_deg": tilt_deg, "follow_mm": follow_mm}
        if self.log_w:
            self.log_w.writerow([f"{now - self.t_start:.4f}", f"{self.t_sim:.4f}", s.session, int(self.paused),
                                 int(focused), s.source, s.status, "".join(sorted(keys.held)), *np.round(s.pos, 5),
                                 round(s.yaw_offset, 4), *np.round(ee, 5), *np.round(bpos, 5), round(s.gripper, 4),
                                 round(jaw, 4), *np.round(s.q_cmd, 5), *np.round(q_meas, 5), round(max_dq_cmd, 5),
                                 round(qd_max, 4), int(bool(self._focused())), s.message, round(follow_mm, 2),
                                 round(tilt_deg, 2), round(step_ms, 3),
                                 "" if input_lat_ms is None else round(input_lat_ms, 2), *np.round(mouse_d, 6),
                                 int(keys.shift), int(handle.dragging)])
        mode = "focus lost" if not focused else ("ready" if s.source == "none" else s.source)
        if self.task is not None:
            mode = "task executor"
        panel.update(mode, self.paused, keys.shift, s.status, s.message, s.pos, s.yaw_offset, ee, all_pos, s.gripper,
                     self.color_dr.block_rgbs(self.sc.block_colors), tilt_deg=tilt_deg, in_placement=n_placed)
        panel.set_task(self.task.ex.panel_lines() if self.task is not None else None)
        if self.script is not None:
            ns = {"target": s.pos, "ee": ee, "block": bpos, "status": s.status, "gripper": s.gripper, "jaw": jaw,
                  "paused": self.paused, "session": s.session, "focused": focused, "held": set(keys.held),
                  "rejected": s.rejected, "yaw_offset": s.yaw_offset,
                  "block_in_placement": block_in_tray(bpos, self.placement, edge),
                  "block_in_storage": block_in_tray(bpos, self.storage, edge),
                  "blocks": all_pos, "n_blocks": len(all_pos), "blocks_in_placement": n_placed,
                  "blocks_init": [np.array(p) for p in self.sc.block_init_pos], "tilt_deg": tilt_deg,
                  "settings": self.settings(), "stats": stats, "dq_cmd": max_dq_cmd, "home": ctrl.p.home_ee.copy(),
                  "block_init": np.array(self.sc.block_init_pos[0]), "target_step": target_step,
                  "follow_err_mm": follow_mm, "dr": self.color_dr.current, "dr_records": len(self.color_dr.records),
                  "blocks_mat_rgb": [[round(float(x), 6) for x in self.m.geom_rgba[g][:3]] for g in self.sc.block_geoms],
                  "block_mat_rgb": [round(float(x), 6) for x in self.m.geom_rgba[self.sc.block_geoms[0]][:3]],
                  "shell_mat_rgb": [round(float(x), 6) for x in self.m.mat_rgba[self.sc.shell_materials[0]][:3]],
                  "pending": panel.pending(), "dragging": handle.dragging, "hover": handle.hover,
                  "fine": keys.shift, "cam": np.asarray(self.camera_position(), float),
                  "selection": [],  # MuJoCo has no scene selection: a drag can never select anything
                  "reasons": {r: self.carry["reasons"][r] + s.reject_counts[r] for r in REASONS}}
            self.script.step(ns)
        self.frames += 1
        return not self.script_done

    def block_in_placement(self) -> bool:
        return bool(block_in_tray(self.last_frame["block"], self.placement, self.sc.edge))

    def _write_trajectory_result(self, summary) -> dict:
        from agentic_sim.teleop.trajectory import trajectory_result
        res = trajectory_result("mujoco", self, summary, self.block_in_placement())
        res["mujoco"] = mujoco.__version__
        os.makedirs(os.path.dirname(self.cfg.result) or ".", exist_ok=True)
        with open(self.cfg.result, "w") as f:
            json.dump(res, f, indent=2)
        self.log(f"wrote {self.cfg.result}: trajectory {res['trajectory']}, block in placement "
                 f"{res['block_in_placement']}, rejected frames {res['rejected_frames']}")
        return {"result": self.cfg.result}

    # ---- reporting ------------------------------------------------------------------------
    def state(self) -> dict:
        s = self.ctrl.state
        f = self.last_frame
        r = lambda a: [round(float(x), 5) for x in a]  # noqa: E731
        out = {"session": s.session, "paused": self.paused, "status": s.status, "message": s.message,
               "settings": self.settings(), "rebuild_pending": False,
               "target": r(s.pos), "yaw_offset": round(float(s.yaw_offset), 4), "gripper_cmd": round(float(s.gripper), 4),
               "frames": self.frames, "t_sim": round(self.t_sim, 3)}
        if self.random_spawn is not None:
            out["random_spawn"] = self.random_spawn
        if self.live_perception is not None:
            out["perception"] = self.live_perception.state()
        if f:
            out.update({"ee": r(f["ee"]), "block": r(f["block"]), "q": r(f["q"]), "jaw": round(f["jaw"], 4),
                        "focused": bool(f["focused"]), "tilt_deg": round(f["tilt_deg"], 2),
                        "blocks": [r(p) for p in f["blocks"]], "blocks_in_placement": int(f["n_placed"]),
                        "block_in_placement": bool(block_in_tray(f["block"], self.placement, self.sc.edge))})
        if self.script is not None:
            out["script"] = {"step": self.script.i, "steps": len(self.script.steps), "done": self.script.done,
                             "passed": sum(x["ok"] for x in self.script.results), "checks": len(self.script.results)}
        if self.task is not None:
            out["task"] = self.task.ex.view()
        return out

    def finish(self) -> dict:
        if self._finished is not None:
            return self._finished
        if self.script is not None:
            self.script.finish()
        if self.log_f:
            self.log_f.close()
            self.log_f = self.log_w = None
        summary = {
            "reject_reason_counts": {r: self.carry["reasons"][r] + self.ctrl.state.reject_counts[r] for r in REASONS},
            "follow_error_mm": _pct(self.metrics["follow_err_mm"]),
            "follow_error_settled_mm": _pct(self.metrics["follow_err_idle_mm"]),
            "control_step_ms": _pct(self.metrics["step_ms"]),
            "mouse_input_latency_ms": _pct(self.metrics["input_lat_ms"]),
        }
        if self.cfg.log or self.cfg.result:
            self.log("metrics: " + json.dumps(summary))
        out = {"metrics": summary}
        if self.traj is not None and self.cfg.result:
            out.update(self._write_trajectory_result(summary))
        if self.task is not None and self.task.scenario is not None and self.cfg.result:
            res = self.task.result(self.task_obs(), "mujoco", {"mujoco": mujoco.__version__, "stats": self.stats,
                                                               "metrics": summary, "blocks": self.block_color_names})
            os.makedirs(os.path.dirname(self.cfg.result) or ".", exist_ok=True)
            with open(self.cfg.result, "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            self.log(f"wrote {self.cfg.result}: {res['passed']}/{res['checks_total']} checks passed")
            out.update({"result": self.cfg.result, "passed": res["passed"], "checks": res["checks_total"]})
        if self.task is not None:
            self.task.close()
        if self.script is not None and self.cfg.result:
            os.makedirs(os.path.dirname(self.cfg.result) or ".", exist_ok=True)
            passed = sum(x["ok"] for x in self.script.results)
            with open(self.cfg.result, "w") as f:
                json.dump({"backend": "mujoco", "mujoco": mujoco.__version__, "checks": self.script.results,
                           "stats": self.stats, "metrics": summary, "script": self.cfg.script,
                           "settings": self.settings(),
                           "color_dr": {"enabled": self.color_dr.enabled, "seed": self.color_dr.seed,
                                        "records": self.color_dr.records}}, f, indent=2)
            self.log(f"wrote {self.cfg.result}: {passed}/{len(self.script.results)} checks passed")
            out.update({"result": self.cfg.result, "passed": passed, "checks": len(self.script.results)})
        self._finished = out
        if self.live_perception is not None:
            self.live_perception.close()
            self.live_perception = None
        return out


def session_config_with(cfg: SessionConfig, settings: dict) -> SessionConfig:
    """A copy of cfg with the panel's scene settings (rebuild after a block-count change)."""
    from dataclasses import replace
    n = int(settings["blocks"])
    colors = cfg.block_colors if cfg.block_colors and len(cfg.block_colors) == n else None
    return replace(cfg, blocks=n, block_colors=colors, color_dr=bool(settings["color_dr"]),
                   color_dr_seed=int(settings["color_dr_seed"]),
                   random_seed=settings.get("random_seed", cfg.random_seed),
                   random_episode=settings.get("random_episode", cfg.random_episode))
