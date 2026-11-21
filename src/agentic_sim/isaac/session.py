"""One teleop session in a running Kit app: scene + SimulationContext + teleop UI, stepped one frame at a time.

Used by scripts/isaac_ee_teleop.py (one session per process) and by the persistent server
(scripts/isaac_sim_server.py), which closes a session and builds the next one inside the same Kit process.

  s = TeleopSession(SessionConfig(scene=..., teleop=...), log)
  while app.is_running() and s.step():   # one control frame (physics + render); pace the loop outside
      ...
  summary = s.finish()                   # write the CSV / result JSON (idempotent)
  s.close()                              # remove the UI and the stage contents; the app keeps running

A Reset applies the panel's scene settings. A different block count cannot be applied in place: the session then
sets `rebuild_request` (the new settings) and stops stepping; the owner closes it and builds the next session with
those settings (session_config_with()).
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.sim as sim_utils
import omni.usd
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from agentic_sim.config import load, provisional_fields, v
from agentic_sim.wrist_lens import apply_wrist_lens
from agentic_sim.isaac.chain import chain_from_usd
from agentic_sim.isaac.color_randomizer import ColorRandomizer
from agentic_sim.isaac.focus_mode import FocusMode
from agentic_sim.isaac.keyboard import HELP, KeyboardInput
from agentic_sim.isaac.panel import TeleopPanel, kind_of
from agentic_sim.isaac.scene import block_count_range, block_in_tray, build_scene
from agentic_sim.isaac.so101 import WORKSHOP_COMMIT
from agentic_sim.isaac.viewport_handle import EEHandle
from agentic_sim.kinematics import ARM_JOINTS, GRIPPER_JOINT
from agentic_sim.teleop.ee_target import REASONS, load_controller

CONTROL_DT = 1.0 / 60.0
PHYSICS_DT = 1.0 / 240.0
SUBSTEPS = int(round(CONTROL_DT / PHYSICS_DT))

CSV_HEADER = ["t_wall", "t_sim", "session", "paused", "focused", "source", "status", "keys",
              "tx", "ty", "tz", "yaw_off", "ex", "ey", "ez", "bx", "by", "bz", "grip_cmd", "jaw",
              *[f"qcmd_{n}" for n in ARM_JOINTS], *[f"q_{n}" for n in ARM_JOINTS], "max_dq_cmd", "max_qd", "os_focus",
              "status_msg", "follow_err_mm", "input_lat_ms", "mouse_dx", "mouse_dy", "mouse_dz", "fine",
              "handle_drag"]


def abgr(rgb):
    r, g, b = (int(max(0, min(1, c)) * 255) for c in rgb)
    return (0xFF << 24) | (b << 16) | (g << 8) | r


@dataclass
class SessionConfig:
    scene: str = "configs/scenes/lab_current_minimal.yaml"
    teleop: str = "configs/teleop.yaml"
    color_dr_config: str = "configs/color_randomization.yaml"
    focus: bool = False
    color_dr: bool = False
    color_dr_seed: int | None = None
    color_dr_log: str | None = None
    blocks: int | None = None          # number of blocks (default: the scene's block.count)
    script: str | None = None          # acceptance script (injected input events)
    trajectory: str | None = None      # scripted EE target sequence (cross-engine comparison); replaces all input
    block_colors: list | None = None   # colour name per block (task scenes); sets the block count
    task: str | None = None            # task scenario (configs/tasks/*.yaml): the executor drives the arm
    task_log: str | None = None        # executor event log (JSONL)
    log: str | None = None             # per-frame CSV
    result: str | None = None          # result JSON (script runs)
    wrist_lens: str | None = None      # wrist camera image mode ("real" | "pinhole"); None = scene default
    record: str | None = None          # demonstration capture (issue #13): episodes are written under this directory
    target_color: str | None = None    # colour of the block to pick (capture); default: the first block's colour
    record_hz: float = 20.0            # recorded control rate (images and rows), like the lab captures
    profile: bool = False
    device: str = "cpu"


def _pct(xs):
    if not xs:
        return None
    a = np.asarray(xs)
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}


class TeleopSession:
    def __init__(self, cfg: SessionConfig, log=print, focus_layout: FocusMode | None = None, carry_script=None):
        self.cfg = cfg
        self.log = log
        self.closed = False
        self._finished = None
        self._requests: list[str] = []
        scene_cfg = self.scene_cfg = apply_wrist_lens(load(cfg.scene), cfg.wrist_lens)
        teleop_cfg = load(cfg.teleop)

        sim = self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=SUBSTEPS, device=cfg.device))
        self.quitting = False
        self._quit_subs = self._watch_quit()
        cam = scene_cfg["camera"]
        sim.set_camera_view(cam["eye"], cam["target"])
        from agentic_sim.task.runner import scenario_colors
        self.block_color_names = cfg.block_colors or scenario_colors(cfg.task)
        h = self.h = build_scene(scene_cfg, cfg.blocks, colors=self.block_color_names)
        self.robot, self.block, self.blocks = h.robot, h.block, h.blocks
        self.rebuild_request: dict | None = None

        # actual EE: solid blue sphere in the scene. The target is drawn only as the viewport handle ring.
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/ee",
            markers={"ee": sim_utils.SphereCfg(radius=0.005, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.3, 1.0)))},
        ))

        sim.reset()
        sim.set_camera_view(cam["eye"], cam["target"])
        from agentic_sim.isaac.sensor_views import sensor_views
        self.sensor_views = sensor_views(log)      # resident across sessions, docked below the viewport
        self.sensor_views.bind(h.sensor_cameras)
        self.recorder = self.grabber = None
        # operation UI: scene prims are not pickable in the viewport, so a drag can never select (and then show a
        # transform gizmo for) the robot or the table. The Stage panel can still select prims.
        self.usd_ctx = omni.usd.get_context()
        for root in ("/World", "/Visuals"):
            self.usd_ctx.set_pickable(root, False)
        self.color_dr = ColorRandomizer(cfg.color_dr_config, enabled=True if cfg.color_dr else None,
                                        seed=cfg.color_dr_seed, log_path=cfg.color_dr_log, log=log,
                                        block_prims=h.block_prims)
        self.chain = chain_from_usd(sim.stage, "/World/Robot", ee_offset=(0, 0, 0))
        self.ctrl = load_controller(self.chain, teleop_cfg, base_pos=v(scene_cfg["robot"]["base_pos"]),
                                    scene=scene_cfg)
        self.arm_ids = [self.robot.joint_names.index(n) for n in ARM_JOINTS]
        self.jaw_id = self.robot.joint_names.index(GRIPPER_JOINT)
        self.edge = v(scene_cfg["block"]["edge"])
        self.carry = {"rejected": 0, "reasons": {r: 0 for r in REASONS}}

        self.kb = KeyboardInput()
        tray_colors = {k: abgr(scene_cfg["trays"][k]["color"]) for k in h.trays}
        fine_scale = lambda: self.ctrl.p.fine_drag_scale  # noqa: E731
        shift = lambda: self.kb.shift  # noqa: E731
        self.panel = TeleopPanel(h.trays, HELP, tray_colors, fine_scale, shift, settings=self.settings(),
                                 max_blocks=block_count_range(scene_cfg)[1])
        self.handle = EEHandle(fine_scale, shift, log)
        # the layout state can outlive a session (server): reuse it, pointed at this session's panel
        self.focus_mode = focus_layout or FocusMode(self.panel.window, log)
        self.focus_mode.set_panel(self.panel.window)
        self.block_colors = [abgr(c) for c in h.block_colors]
        self.script = None
        set_camera = lambda eye, tgt: sim.set_camera_view(eye, tgt)  # noqa: E731
        if carry_script is not None:  # the scene was rebuilt (block count) while a script was running
            self.script = carry_script
            self.script.rebind(self.panel, self.handle, set_camera)
        elif cfg.script:
            from agentic_sim.isaac.scripted_input import ScriptedInput
            self.script = ScriptedInput(load(cfg.script), self.panel, log, handle=self.handle, set_camera=set_camera)

        self.traj = None
        if cfg.trajectory:
            from agentic_sim.teleop.trajectory import make_driver
            self.traj = make_driver(cfg.trajectory, self.ctrl)
        self.task = None
        log(f"asset: Sim-to-Real-SO-101-Workshop@{WORKSHOP_COMMIT}; provisional scene fields: "
            f"{', '.join(provisional_fields(scene_cfg))}")
        log(f"EE offset in gripper link (provisional): {self.chain.ee_offset.tolist()}; "
            f"home q={np.round(self.ctrl.q_home, 3).tolist()}")
        log(HELP)

        # dock after the UI has been laid out (docking at start-up fails)
        self.focus_at_frame = 10 if (cfg.focus and not self.focus_mode.active) else -1
        self.do_reset()
        for _ in range(30):  # settle
            self.robot.write_data_to_sim()
            sim.step(render=False)
            self.robot.update(PHYSICS_DT)
            for b in self.blocks:
                b.update(PHYSICS_DT)

        self.log_f = self.log_w = None
        if cfg.log:
            os.makedirs(os.path.dirname(cfg.log) or ".", exist_ok=True)
            cont = carry_script is not None and os.path.exists(cfg.log)  # keep the rows of the scene before
            self.log_f = open(cfg.log, "a" if cont else "w", newline="")
            self.log_w = csv.writer(self.log_f)
            if not cont:
                self.log_w.writerow(CSV_HEADER)

        self.paused = False
        self.t_start = self.last = time.monotonic()
        self.t_sim = 0.0
        self.stats = {"max_dq_cmd_per_frame": 0.0, "max_joint_vel": 0.0, "rejected_targets": 0,
                      "block_lifted_max_z": 0.0}
        self.prev_status = "ok"
        self.metrics = {"input_lat_ms": [], "follow_err_mm": [], "follow_err_idle_mm": []}
        self.last_input_t = -1e9
        self.prev_sel = []
        self.prev_target = self.ctrl.state.pos.copy()
        self.prof = {}
        self.prof_t = time.monotonic()
        self.frames = 0
        self.last_frame = {}
        if cfg.task:
            self.start_task()
        if cfg.record:
            self._start_recording()

    def _start_recording(self):
        from agentic_sim.capture.recorder import DemoRecorder
        from agentic_sim.isaac.capture import CameraGrabber
        from agentic_sim.scene_layout import sensor_cameras, trays
        cfg, h = self.cfg, self.h
        if not self.block_color_names:
            raise ValueError("demonstration capture needs the block colour names (--block_colors)")
        self.grabber = CameraGrabber(h.sensor_cameras, self.log)
        # the actual-EE sphere is an operator aid, not part of the scene: keep it out of the recorded images (the
        # target ring of the viewport handle is a UI overlay and never reaches the cameras)
        self.markers.set_visibility(False)
        specs = sensor_cameras(self.scene_cfg)
        cams = {}
        for key, (path, w, hh) in h.sensor_cameras.items():
            sp = specs[key]
            cams[key] = {"width": int(w), "height": int(hh), "prim": path, "fx": sp.get("fx"), "fy": sp.get("fy"),
                         "cx": sp.get("cx"), "cy": sp.get("cy")}
            if key == "wrist":
                cams[key]["lens_mode"] = sp.get("lens_mode")
                lens = sp.get("lens")
                if lens is not None:
                    cams[key]["distortion_opencv"] = [float(x) for x in lens.dist]
        self.recorder = DemoRecorder(
            cfg.record, scene_path=cfg.scene, experiment=None, target_color=cfg.target_color or self.block_color_names[0],
            block_colors=self.block_color_names, trays=trays(self.scene_cfg), edge=self.edge,
            gripper={"open": self.ctrl.p.gripper_open, "closed": self.ctrl.p.gripper_closed}, cameras=cams,
            control_dt=CONTROL_DT, record_hz=cfg.record_hz, operator="scripted_input" if cfg.script else "human",
            lens_mode=cams.get("wrist", {}).get("lens_mode"), log=self.log)

    def _record(self, q_meas):
        robot = self.robot
        s = self.ctrl.state
        jq = robot.data.joint_pos[0].cpu().numpy()
        jv = robot.data.joint_vel[0].cpu().numpy()
        ids = list(self.arm_ids) + [self.jaw_id]
        cmd_q = np.r_[s.q_cmd, s.gripper]
        blocks = [(b.data.root_pos_w[0].cpu().numpy(), b.data.root_quat_w[0].cpu().numpy()) for b in self.blocks]
        ee = self.ctrl.chain.ee_pose_mount(q_meas)[:3, 3]
        self.recorder.frame(int(round(self.t_sim / CONTROL_DT)), self.t_sim, paused=self.paused,
                            focused=self.last_focused, source=s.source, ik_status=s.status, cmd_ee=s.pos,
                            cmd_yaw=s.yaw_offset, cmd_q=cmd_q, q=jq[ids], qd=jv[ids], ee=ee, blocks=blocks,
                            grab=self.grabber.grab)

    # ---- task executor (issue #5) ----------------------------------------------------------------
    def start_task(self):
        """Hand the arm to the task executor (scenario from cfg.task, or orders from the server)."""
        from agentic_sim.task.runner import TaskRunner, default_color_names
        if self.task is not None:
            self.task.close()
        colors = self.block_color_names or default_color_names(self.scene_cfg, len(self.blocks))
        self.task = TaskRunner(self.ctrl, self.scene_cfg, self.h.block_init_pos, colors, CONTROL_DT, self.log,
                               scenario=self.cfg.task, event_log=self.cfg.task_log)
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
        q = self.robot.data.joint_pos[0, self.arm_ids].cpu().numpy()
        # arm speed from the measured joint positions frame to frame: PhysX reports a constant joint velocity of
        # about 0.2 rad/s for an arm at rest while the jaw squeezes a block, so joint_vel cannot tell "stopped"
        prev, self._task_q_prev = getattr(self, "_task_q_prev", None), q.copy()
        qd = 0.0 if prev is None else float(np.max(np.abs(q - prev))) / CONTROL_DT
        return Obs(t=self.t_sim, ee=self.ctrl.chain.ee_pose_mount(q)[:3, 3].copy(), qd_max=qd,
                   jaw=float(self.robot.data.joint_pos[0, self.jaw_id]),
                   blocks=[b.data.root_pos_w[0].cpu().numpy() for b in self.blocks],
                   block_yaw=[quat_yaw(b.data.root_quat_w[0].cpu().numpy()) for b in self.blocks],
                   block_speed=[float(b.data.root_lin_vel_w[0].norm()) for b in self.blocks])

    # ---- scene settings -----------------------------------------------------------------
    def settings(self) -> dict:
        """Scene settings currently shown (block count, colour DR on / off, seed)."""
        return {"blocks": len(self.h.block_prims), "color_dr": bool(self.color_dr.enabled),
                "color_dr_seed": int(self.color_dr.seed)}

    def _apply_settings(self) -> bool:
        """On reset: apply the panel's settings. Returns False if the scene must be rebuilt (block count)."""
        want = self.panel.settings()
        if want["blocks"] != len(self.h.block_prims):
            self.rebuild_request = want
            self.log(f"rebuild requested: {want}")
            return False
        self.color_dr.configure(want["color_dr"], want["color_dr_seed"])
        return True

    def handover_script(self):
        """Detach the running script (rebuild): the next session continues it; this one writes no result."""
        script, self.script = self.script, None
        return script

    # ---- commands from outside (server) -----------------------------------------------
    def request(self, action: str):
        """Queue 'reset', 'pause' or 'resume'; applied at the start of the next frame like a panel button."""
        if action not in ("reset", "pause", "resume"):
            raise ValueError(f"unknown action {action!r}")
        self._requests.append(action)

    def _pop_requests(self) -> list:
        out = []
        for a in self._requests:
            if a == "reset":
                out.append("reset")
            elif (a == "pause") != self.paused:  # pause / resume only act when they change the state
                out.append("pause")
        self._requests = []
        return out

    def full_command(self):
        q = self.robot.data.default_joint_pos.clone()
        q[0, self.arm_ids] = torch.tensor(self.ctrl.state.q_cmd, dtype=q.dtype, device=q.device)
        q[0, self.jaw_id] = self.ctrl.state.gripper
        return q

    def do_reset(self):
        ctrl, robot, block, sim, h = self.ctrl, self.robot, self.block, self.sim, self.h
        self.carry["rejected"] += ctrl.state.rejected
        for r in REASONS:
            self.carry["reasons"][r] += ctrl.state.reject_counts[r]
        ctrl.reset()
        q = self.full_command()
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        robot.reset()
        for b, init in zip(self.blocks, h.block_init_pos):
            pose = torch.tensor([[*init, *h.block_init_rot]], dtype=torch.float32, device=sim.device)
            b.write_root_pose_to_sim(pose)
            b.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
            b.reset()
        self.color_dr.apply_next()  # no-op when colour DR is off
        self.block_colors = [abgr(c) for c in self.color_dr.block_rgbs(h.block_colors)]
        self.panel.set_colors(self.color_dr.summary())
        self.panel.mark_applied(self.settings())
        self.kb.clear()
        self.panel.cancel()
        self.handle.cancel()
        self.log(f"reset: session={ctrl.state.session}")
        if getattr(self, "recorder", None) is not None:
            self.recorder.reset()

    def _tick(self, name, t0):
        t1 = time.monotonic()
        self.prof[name] = self.prof.get(name, 0.0) + (t1 - t0)
        return t1

    @property
    def script_done(self) -> bool:
        return ((self.script is not None and self.script.done) or (self.traj is not None and self.traj.done(self.t_sim))
                or (self.task is not None and self.task.done))

    # ---- one control frame --------------------------------------------------------------
    def _watch_quit(self):
        """Closing the window posts the app quit. With Replicator loaded (recording), its quit handler stops the
        timeline, and Isaac Lab's STOP handler would then keep rendering until Play: the window never closes. Run
        before it (lower order) and switch that handler off, so the main loop ends and finish() writes everything."""
        import carb.eventdispatcher
        import omni.kit.app

        def on_quit(*_):
            self.quitting = True
            if self.sim is not None:
                self.sim._disable_app_control_on_stop_handle = True

        app = omni.kit.app.get_app()
        return [app.get_shutdown_event_stream().create_subscription_to_pop_by_type(
                    omni.kit.app.POST_QUIT_EVENT_TYPE, on_quit, name="agentic_sim.session.quit", order=-100),
                carb.eventdispatcher.get_eventdispatcher().observe_event(
                    event_name=omni.kit.app.GLOBAL_EVENT_POST_QUIT, on_event=on_quit,
                    observer_name="agentic_sim.session.quit", order=-100)]

    def step(self) -> bool:
        """Run one frame. Returns False once the acceptance script has finished (nothing else ends a session)."""
        ctrl, robot, block, sim, kb, panel, handle = (self.ctrl, self.robot, self.block, self.sim, self.kb,
                                                       self.panel, self.handle)
        script, stats, metrics, carry = self.script, self.stats, self.metrics, self.carry
        now = time.monotonic()
        tp = now
        dt = now - self.last
        self.last = now

        if self.rebuild_request is not None:  # waiting for the owner to rebuild the scene
            sim.render()
            return not self.script_done
        actions = kb.pop_actions() + panel.pop_actions() + self._pop_requests()
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
            if a == "focus":  # layout only: does not touch the teleop state
                self.focus_mode.toggle()
            elif a == "pause":
                self.paused = not self.paused
                kb.clear()
                panel.cancel()
                handle.cancel()
                ctrl.clear_yaw_goal()
                ctrl.clear_gripper_goal()
                self.log(f"pause={self.paused}")
            elif a in ("open", "close") and not self.paused:
                ctrl.set_gripper(a == "open", "mouse")
            elif a == "toggle_gripper" and not self.paused:
                mid = 0.5 * (ctrl.p.gripper_open + ctrl.p.gripper_closed)
                ctrl.set_gripper(ctrl.state.gripper < mid, "mouse")
            elif a in ("yaw-", "yaw+") and not self.paused:
                ctrl.nudge_yaw(math.radians(15) * (1 if a == "yaw+" else -1), "mouse")

        focused = kb.focused() if (script is None or script.focus_override is None) else script.focus_override
        if not focused:
            kb.clear()
            panel.cancel()
            handle.cancel()
            if self.task is None:  # operator goals only: a running task keeps its own gripper / yaw goals
                ctrl.clear_yaw_goal()
                ctrl.clear_gripper_goal()

        input_lat_ms = None
        mouse_d = np.zeros(3)
        max_dq_cmd = 0.0
        if self.paused:
            # input received while paused is dropped, so nothing replays on resume
            kb.clear()
            panel.cancel()
            handle.cancel()
            sim.render()
        else:
            if self.traj is not None:  # comparison run: the trajectory is the only input
                kb.clear()
                panel.cancel()
                handle.cancel()
                self.traj.apply(self.t_sim)
            if self.task is not None:  # the executor is the only control source while a task runs
                kb.clear()
                panel.cancel()
                handle.cancel()
                self.task.step(self.task_obs())
            lin, yaw, grip = kb.directions()
            ctrl.step_velocity(lin, yaw, dt, "keyboard")
            ctrl.step_gripper(grip, dt, "keyboard")
            # mouse: viewport handle (primary) + top view (auxiliary), both relative, one IK update per frame
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
                metrics["input_lat_ms"].append(input_lat_ms)
                mouse_d = ctrl.state.pos - p_before
            tp = self._tick("input+ik", tp)
            q_prev = ctrl.state.q_cmd.copy()
            ctrl.joint_command(CONTROL_DT)
            max_dq_cmd = float(np.max(np.abs(ctrl.state.q_cmd - q_prev)))
            robot.set_joint_position_target(self.full_command())
            for _ in range(SUBSTEPS):
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(PHYSICS_DT)
                for b in self.blocks:
                    b.update(PHYSICS_DT)
            self.t_sim += CONTROL_DT
            tp = self._tick("physics", tp)
            sim.render()
            tp = self._tick("render", tp)
            if self.quitting:       # the window was closed during this render: the physics views may be gone
                return False

        q_meas = robot.data.joint_pos[0, self.arm_ids].cpu().numpy()
        ee_pose = ctrl.chain.ee_pose_mount(q_meas)
        ee = ee_pose[:3, 3].copy()
        tilt_deg = float(np.degrees(np.arccos(np.clip(ee_pose[2, 2], -1.0, 1.0))))  # actual gripper tilt
        bpos = (block.data.root_pos_w[0] - 0).cpu().numpy()
        all_pos = [bpos] + [b.data.root_pos_w[0].cpu().numpy() for b in self.blocks[1:]]
        placement = self.h.trays["placement_beige"]
        n_placed = sum(block_in_tray(p, placement, self.edge) for p in all_pos)
        s = ctrl.state
        self.markers.visualize(translations=torch.tensor(np.array([ee]), dtype=torch.float32),
                               marker_indices=torch.tensor([0]))
        handle.update(s.pos, kind_of(s.status))
        self.focus_mode.update()
        self.sensor_views.update()
        if self.focus_at_frame >= 0:
            self.focus_at_frame -= 1
            if self.focus_at_frame < 0:
                self.focus_mode.enter()
        self.last_focused = focused
        if self.recorder is not None and not self.paused:   # paused frames advance no physics: not recorded
            self._record(q_meas)
        mode = "focus lost" if not focused else ("keyboard" if s.source == "keyboard" else s.source)
        if self.task is not None:
            mode = "task executor"
        panel.set_task(self.task.ex.panel_lines() if self.task is not None else None)
        panel.update(mode, self.paused, kb.shift, s.status, s.message, s.pos, s.yaw_offset, ee, all_pos, s.gripper,
                     self.block_colors, tilt_deg=tilt_deg, in_placement=n_placed)
        follow_mm = float(np.linalg.norm(s.pos - ee)) * 1000.0
        target_step = float(np.linalg.norm(s.pos - self.prev_target))
        if target_step > 1e-9 or kb.held:
            self.last_input_t = now
        metrics["follow_err_mm"].append(follow_mm)
        if now - self.last_input_t > 1.0 and not self.paused:  # settled: no target change for 1 s
            metrics["follow_err_idle_mm"].append(follow_mm)
        self.prev_target = s.pos.copy()
        sel = list(self.usd_ctx.get_selection().get_selected_prim_paths())
        if sel != self.prev_sel:
            self.log(f"selection changed: {sel}")
            self.prev_sel = sel
        if s.status != self.prev_status:
            self.log(f"status {self.prev_status} -> {s.status} {s.message}")
            self.prev_status = s.status

        stats["max_dq_cmd_per_frame"] = max(stats["max_dq_cmd_per_frame"], max_dq_cmd)
        qd_max = float(robot.data.joint_vel[0, self.arm_ids].abs().max())
        stats["max_joint_vel"] = max(stats["max_joint_vel"], qd_max)
        stats["rejected_targets"] = carry["rejected"] + s.rejected
        stats["block_lifted_max_z"] = max(stats["block_lifted_max_z"], float(bpos[2]))
        jaw = float(robot.data.joint_pos[0, self.jaw_id])
        self.last_frame = {"target": s.pos.copy(), "ee": ee, "block": bpos, "q": q_meas, "jaw": jaw,
                           "status": s.status, "message": s.message, "focused": focused, "blocks": all_pos,
                           "n_placed": n_placed, "tilt_deg": tilt_deg}

        if self.log_w:
            self.log_w.writerow([f"{now - self.t_start:.4f}", f"{self.t_sim:.4f}", s.session, int(self.paused),
                                 int(focused), s.source, s.status, "".join(sorted(kb.held)), *np.round(s.pos, 5),
                                 round(s.yaw_offset, 4), *np.round(ee, 5), *np.round(bpos, 5), round(s.gripper, 4),
                                 round(jaw, 4), *np.round(s.q_cmd, 5), *np.round(q_meas, 5), round(max_dq_cmd, 5),
                                 round(qd_max, 4), int(kb.focused()), s.message, round(follow_mm, 2),
                                 "" if input_lat_ms is None else round(input_lat_ms, 2), *np.round(mouse_d, 6),
                                 int(kb.shift), int(handle.dragging)])

        if script is not None:
            h, usd_ctx, color_dr = self.h, self.usd_ctx, self.color_dr
            ns = {"target": s.pos, "ee": ee, "block": bpos, "status": s.status, "gripper": s.gripper,
                  "jaw": jaw, "paused": self.paused, "session": s.session,
                  "focused": focused, "held": set(kb.held), "rejected": s.rejected, "yaw_offset": s.yaw_offset,
                  "block_in_placement": block_in_tray(bpos, h.trays["placement_beige"], self.edge),
                  "block_in_storage": block_in_tray(bpos, h.trays["storage_pink"], self.edge),
                  "blocks": all_pos, "n_blocks": len(all_pos), "blocks_in_placement": n_placed,
                  "blocks_init": [np.array(p) for p in h.block_init_pos], "tilt_deg": tilt_deg,
                  "settings": self.settings(), "pending": self.panel.pending(),
                  "blocks_mat_rgb": [[round(float(x), 6) for x in usd_ctx.get_stage().GetPrimAtPath(
                      f"{bp}/geometry/material/Shader").GetAttribute("inputs:diffuseColor").Get()]
                      for bp in h.block_prims],
                  "stats": stats, "dq_cmd": max_dq_cmd, "home": ctrl.p.home_ee.copy(),
                  "block_init": np.array(h.block_init_pos[0]), "dragging": handle.dragging, "hover": handle.hover,
                  "fine": kb.shift, "target_step": target_step, "follow_err_mm": follow_mm,
                  "cam": handle.camera_position(),
                  "selection": list(usd_ctx.get_selection().get_selected_prim_paths()),
                  "focus_mode": self.focus_mode.active, "editors_visible": self.focus_mode.editor_windows_visible(),
                  "front_tabs": self.focus_mode.front_tabs(),
                  "dr": color_dr.current, "dr_records": len(color_dr.records),
                  "block_mat_rgb": [round(float(x), 6) for x in
                                    usd_ctx.get_stage().GetPrimAtPath("/World/Block/geometry/material/Shader")
                                    .GetAttribute("inputs:diffuseColor").Get()],
                  "shell_mat_rgb": [round(float(x), 6) for x in
                                    usd_ctx.get_stage().GetPrimAtPath("/World/Robot/Looks/material_a_3d_printed/Shader")
                                    .GetAttribute("inputs:diffuse_color_constant").Get()],
                  "reasons": {r: carry["reasons"][r] + s.reject_counts[r] for r in REASONS}}
            script.step(ns)

        tp = self._tick("ui+log", tp)
        self.frames += 1
        self.prof["frames"] = self.prof.get("frames", 0) + 1
        if self.cfg.profile and time.monotonic() - self.prof_t > 5.0:
            n = self.prof.pop("frames")
            self.log("profile ms/frame: " + ", ".join(f"{k}={1000 * val / n:.1f}" for k, val in self.prof.items())
                     + f"  fps={n / (time.monotonic() - self.prof_t):.1f}")
            self.prof.clear()
            self.prof_t = time.monotonic()
        return not self.script_done

    # ---- reporting ------------------------------------------------------------------------
    def state(self) -> dict:
        """JSON-friendly snapshot for clients (get_state)."""
        s = self.ctrl.state
        f = self.last_frame
        r = lambda a: [round(float(x), 5) for x in a]  # noqa: E731
        out = {"session": s.session, "paused": self.paused, "status": s.status, "message": s.message,
               "settings": self.settings(), "rebuild_pending": self.rebuild_request is not None,
               "target": r(s.pos), "yaw_offset": round(float(s.yaw_offset), 4), "gripper_cmd": round(float(s.gripper), 4),
               "frames": self.frames, "t_sim": round(self.t_sim, 3)}
        if f:
            out.update({"ee": r(f["ee"]), "block": r(f["block"]), "q": r(f["q"]), "jaw": round(f["jaw"], 4),
                        "focused": bool(f["focused"]), "tilt_deg": round(f["tilt_deg"], 2),
                        "blocks": [r(p) for p in f["blocks"]], "blocks_in_placement": int(f["n_placed"]),
                        "block_in_placement": bool(block_in_tray(f["block"], self.h.trays["placement_beige"],
                                                                 self.edge))})
        if self.script is not None:
            out["script"] = {"step": self.script.i, "steps": len(self.script.steps), "done": self.script.done,
                             "passed": sum(x["ok"] for x in self.script.results), "checks": len(self.script.results)}
        if self.task is not None:
            out["task"] = self.task.ex.view()
        return out

    def finish(self) -> dict:
        """Release scripted keys, close the CSV, write the result JSON. Idempotent; returns the summary."""
        if self._finished is not None:
            return self._finished
        if self.script is not None:
            self.script.finish()
        if self.log_f:
            self.log_f.close()
            self.log_f = self.log_w = None
        summary = {
            "reject_reason_counts": {r: self.carry["reasons"][r] + self.ctrl.state.reject_counts[r] for r in REASONS},
            "mouse_input_latency_ms": _pct(self.metrics["input_lat_ms"]),
            "follow_error_mm": _pct(self.metrics["follow_err_mm"]),
            "follow_error_settled_mm": _pct(self.metrics["follow_err_idle_mm"]),
        }
        if self.cfg.log or self.cfg.result:
            self.log("metrics: " + json.dumps(summary))
        out = {"metrics": summary}
        if self.traj is not None and self.cfg.result:
            from agentic_sim.teleop.trajectory import trajectory_result
            res = trajectory_result("isaac", self, summary,
                                    block_in_tray(self.last_frame["block"], self.h.trays["placement_beige"], self.edge))
            os.makedirs(os.path.dirname(self.cfg.result) or ".", exist_ok=True)
            with open(self.cfg.result, "w") as f:
                json.dump(res, f, indent=2)
            self.log(f"wrote {self.cfg.result}: trajectory {res['trajectory']}, block in placement "
                     f"{res['block_in_placement']}, rejected frames {res['rejected_frames']}")
            out["result"] = self.cfg.result
        if self.task is not None and self.task.scenario is not None and self.cfg.result:
            res = self.task.result(self.task_obs(), "isaac", {"stats": self.stats, "metrics": summary,
                                                              "blocks": self.block_color_names})
            os.makedirs(os.path.dirname(self.cfg.result) or ".", exist_ok=True)
            with open(self.cfg.result, "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            self.log(f"wrote {self.cfg.result}: {res['passed']}/{res['checks_total']} checks passed")
            out.update({"result": self.cfg.result, "passed": res["passed"], "checks": res["checks_total"]})
        if self.task is not None:
            self.task.close()
        if self.recorder is not None:
            self.recorder.close()
            out["episodes"] = [{"path": e["path"], "success": e["result"]["success"], "frames": e["control_frames"],
                                "duration_s": e.get("duration_s")} for e in self.recorder.episodes]
            self.grabber.close()
        if self.script is not None and self.cfg.result:
            os.makedirs(os.path.dirname(self.cfg.result) or ".", exist_ok=True)
            passed = sum(r["ok"] for r in self.script.results)
            with open(self.cfg.result, "w") as f:
                json.dump({"checks": self.script.results, "stats": self.stats, "metrics": summary,
                           "script": self.cfg.script, "settings": self.settings(),
                           "color_dr": {"enabled": self.color_dr.enabled, "seed": self.color_dr.seed,
                                        "records": self.color_dr.records}}, f, indent=2)
            self.log(f"wrote {self.cfg.result}: {passed}/{len(self.script.results)} checks passed")
            out.update({"result": self.cfg.result, "passed": passed, "checks": len(self.script.results)})
        self._finished = out
        return out

    def close(self, keep_layout: bool = True):
        """Tear the session down so another one can be built in the same Kit app.

        keep_layout: leave the focus layout as it is (the next session reuses it); False restores the editors."""
        if self.closed:
            return
        self.finish()
        self.kb.close()
        self.handle.destroy()
        self.sensor_views.unbind()
        if not keep_layout:
            self.focus_mode.exit()
        self.focus_mode.set_panel(None)
        self.panel.window.visible = False
        self.panel.window.destroy()
        sim = self.sim
        self._quit_subs = None      # unsubscribe: the next session (rebuild) watches the quit itself
        # In GUI mode Isaac Lab's STOP handler keeps rendering until the timeline plays again (meant for a user
        # pressing Stop); a programmatic teardown must switch it off first, or stop() never returns.
        sim._disable_app_control_on_stop_handle = True
        if sim.is_playing():
            sim.stop()
        # drop every reference to the assets so their timeline / physics callbacks unsubscribe
        self.h = self.robot = self.block = self.blocks = self.markers = self.script = self.ctrl = None
        gc.collect()
        sim.clear_all_callbacks()
        sim.clear_instance()
        self.sim = None
        sim_utils.create_new_stage()
        self.closed = True


def session_config_with(cfg: SessionConfig, settings: dict) -> SessionConfig:
    """A copy of cfg with the panel's scene settings (rebuild after a block-count change)."""
    from dataclasses import replace
    n = int(settings["blocks"])
    colors = cfg.block_colors if cfg.block_colors and len(cfg.block_colors) == n else None
    return replace(cfg, blocks=n, block_colors=colors, color_dr=bool(settings["color_dr"]),
                   color_dr_seed=int(settings["color_dr_seed"]))
