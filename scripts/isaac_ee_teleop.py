"""Isaac Sim minimal desk scene with keyboard / mouse EE teleop for one SO-101 (issue #1).

Run inside the pinned container (see docs/isaac-ee-teleop.md):
    docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py
Optional:
    --script configs/acceptance/keyboard_pick_place.yaml   drive the UI with injected input events
    --log outputs/run.csv                                  per-frame debug log
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scene", default="configs/scenes/lab_current_minimal.yaml")
parser.add_argument("--teleop", default="configs/teleop.yaml")
parser.add_argument("--script", default=None, help="YAML input script for automated runs")
parser.add_argument("--log", default=None, help="CSV debug log path")
parser.add_argument("--result", default=None, help="JSON summary path (script runs)")
parser.add_argument("--exit_after_script", action="store_true")
parser.add_argument("--max_time", type=float, default=0.0, help="exit after N wall seconds (0 = run until closed)")
parser.add_argument("--profile", action="store_true", help="print per-section frame timing every 5 s")
parser.add_argument("--focus", action="store_true", help="start in the focus (operation) layout")
parser.add_argument("--color_dr", action="store_true", help="enable colour domain randomization (default off)")
parser.add_argument("--color_dr_seed", type=int, default=None, help="seed (default: from the config)")
parser.add_argument("--color_dr_config", default="configs/color_randomization.yaml")
parser.add_argument("--color_dr_log", default=None, help="JSONL of the drawn colours (default: from the config)")
AppLauncher.add_app_launcher_args(parser)
# One small articulation: CPU PhysX steps faster than the GPU pipeline here, and the viewport should not wait for
# vsync / the 60 Hz main-loop limiter (measured: 16 -> 40 fps on the tested machine, see docs/isaac-ee-teleop.md).
parser.set_defaults(device="cpu", kit_args="--/app/vsync=false --/app/runLoops/main/rateLimitEnabled=false")
args = parser.parse_args()
app_launcher = AppLauncher(args)
app = app_launcher.app

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402

from agentic_sim.config import load, provisional_fields, v  # noqa: E402
from agentic_sim.isaac.chain import chain_from_usd  # noqa: E402
from agentic_sim.isaac.color_randomizer import ColorRandomizer  # noqa: E402
from agentic_sim.isaac.focus_mode import FocusMode  # noqa: E402
from agentic_sim.isaac.keyboard import HELP, KeyboardInput  # noqa: E402
from agentic_sim.isaac.panel import TeleopPanel, kind_of  # noqa: E402
from agentic_sim.isaac.viewport_handle import EEHandle  # noqa: E402
from agentic_sim.isaac.scene import block_in_tray, build_scene  # noqa: E402
from agentic_sim.isaac.so101 import WORKSHOP_COMMIT  # noqa: E402
from agentic_sim.kinematics import ARM_JOINTS, GRIPPER_JOINT  # noqa: E402
from agentic_sim.teleop.ee_target import REASONS, load_controller  # noqa: E402

CONTROL_DT = 1.0 / 60.0
PHYSICS_DT = 1.0 / 240.0
SUBSTEPS = int(round(CONTROL_DT / PHYSICS_DT))


def log(msg):
    print(f"[ee_teleop] {msg}", flush=True)


def abgr(rgb):
    r, g, b = (int(max(0, min(1, c)) * 255) for c in rgb)
    return (0xFF << 24) | (b << 16) | (g << 8) | r


def main():
    scene_cfg = load(args.scene)
    teleop_cfg = load(args.teleop)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, render_interval=SUBSTEPS, device=args.device))
    cam = scene_cfg["camera"]
    sim.set_camera_view(cam["eye"], cam["target"])
    h = build_scene(scene_cfg)
    robot, block = h.robot, h.block

    # actual EE: solid blue sphere in the scene. The target is drawn only as the viewport handle ring.
    markers = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/ee",
        markers={"ee": sim_utils.SphereCfg(radius=0.005, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.1, 0.3, 1.0)))},
    ))

    sim.reset()
    sim.set_camera_view(cam["eye"], cam["target"])
    # operation UI: scene prims are not pickable in the viewport, so a drag can never select (and then show a
    # transform gizmo for) the robot or the table. The Stage panel can still select prims.
    import omni.usd
    usd_ctx = omni.usd.get_context()
    for root in ("/World", "/Visuals"):
        usd_ctx.set_pickable(root, False)
    color_dr = ColorRandomizer(args.color_dr_config, enabled=True if args.color_dr else None, seed=args.color_dr_seed,
                               log_path=args.color_dr_log, log=log)
    chain = chain_from_usd(sim.stage, "/World/Robot", ee_offset=(0, 0, 0))
    ctrl = load_controller(chain, teleop_cfg)
    arm_ids = [robot.joint_names.index(n) for n in ARM_JOINTS]
    jaw_id = robot.joint_names.index(GRIPPER_JOINT)
    edge = v(scene_cfg["block"]["edge"])

    def full_command():
        q = robot.data.default_joint_pos.clone()
        q[0, arm_ids] = torch.tensor(ctrl.state.q_cmd, dtype=q.dtype, device=q.device)
        q[0, jaw_id] = ctrl.state.gripper
        return q

    carry = {"rejected": 0, "reasons": {r: 0 for r in REASONS}}

    def do_reset():
        carry["rejected"] += ctrl.state.rejected
        for r in REASONS:
            carry["reasons"][r] += ctrl.state.reject_counts[r]
        ctrl.reset()
        q = full_command()
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        robot.reset()
        pose = torch.tensor([[*h.block_init_pos, *h.block_init_rot]], dtype=torch.float32, device=sim.device)
        block.write_root_pose_to_sim(pose)
        block.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
        block.reset()
        color_dr.apply_next()  # no-op when colour DR is off
        block_color[0] = abgr(color_dr.block_rgb(scene_cfg["block"]["color"]))
        panel.set_colors(color_dr.summary())
        kb.clear()
        panel.cancel()
        handle.cancel()
        log(f"reset: session={ctrl.state.session}")

    kb = KeyboardInput()
    tray_colors = {k: abgr(scene_cfg["trays"][k]["color"]) for k in h.trays}
    fine_scale = lambda: ctrl.p.fine_drag_scale  # noqa: E731
    shift = lambda: kb.shift  # noqa: E731
    panel = TeleopPanel(h.trays, HELP, tray_colors, fine_scale, shift)
    handle = EEHandle(fine_scale, shift, log)
    focus_mode = FocusMode(panel.window, log)
    block_color = [abgr(scene_cfg["block"]["color"])]  # list: updated in do_reset when colour DR is on
    script = None
    if args.script:
        from agentic_sim.isaac.scripted_input import ScriptedInput
        script = ScriptedInput(load(args.script), panel, log, handle=handle,
                               set_camera=lambda eye, tgt: sim.set_camera_view(eye, tgt))

    log(f"asset: Sim-to-Real-SO-101-Workshop@{WORKSHOP_COMMIT}; provisional scene fields: "
        f"{', '.join(provisional_fields(scene_cfg))}")
    log(f"EE offset in gripper link (provisional): {chain.ee_offset.tolist()}; home q={np.round(ctrl.q_home, 3).tolist()}")
    log(HELP)

    focus_at_frame = 10 if args.focus else -1  # dock after the UI has been laid out (docking at start-up fails)
    do_reset()
    for _ in range(30):  # settle
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(PHYSICS_DT)
        block.update(PHYSICS_DT)

    log_f = log_w = None
    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        log_f = open(args.log, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["t_wall", "t_sim", "session", "paused", "focused", "source", "status", "keys",
                        "tx", "ty", "tz", "yaw_off", "ex", "ey", "ez", "bx", "by", "bz", "grip_cmd", "jaw",
                        *[f"qcmd_{n}" for n in ARM_JOINTS], *[f"q_{n}" for n in ARM_JOINTS], "max_dq_cmd", "max_qd", "os_focus",
                        "status_msg", "follow_err_mm", "input_lat_ms", "mouse_dx", "mouse_dy", "mouse_dz", "fine",
                        "handle_drag"])

    paused = False
    t_start = last = time.monotonic()
    t_sim = 0.0
    max_dq_cmd = 0.0
    stats = {"max_dq_cmd_per_frame": 0.0, "max_joint_vel": 0.0, "rejected_targets": 0, "block_lifted_max_z": 0.0}
    prev_status = "ok"
    metrics = {"input_lat_ms": [], "follow_err_mm": [], "follow_err_idle_mm": []}
    last_input_t = -1e9
    prev_sel = []
    prev_target = ctrl.state.pos.copy()
    prof = {}
    prof_t = time.monotonic()

    def tick(name, t0):
        t1 = time.monotonic()
        prof[name] = prof.get(name, 0.0) + (t1 - t0)
        return t1

    while app.is_running():
        now = time.monotonic()
        tp = now
        dt = now - last
        last = now

        actions = kb.pop_actions() + panel.pop_actions()
        if "reset" in actions:
            do_reset()
            paused = False
            dt = 0.0
        for a in actions:
            if a == "focus":  # layout only: does not touch the teleop state
                focus_mode.toggle()
            elif a == "pause":
                paused = not paused
                kb.clear()
                panel.cancel()
                handle.cancel()
                ctrl.clear_yaw_goal()
                ctrl.clear_gripper_goal()
                log(f"pause={paused}")
            elif a in ("open", "close") and not paused:
                ctrl.set_gripper(a == "open", "mouse")
            elif a == "toggle_gripper" and not paused:
                mid = 0.5 * (ctrl.p.gripper_open + ctrl.p.gripper_closed)
                ctrl.set_gripper(ctrl.state.gripper < mid, "mouse")
            elif a in ("yaw-", "yaw+") and not paused:
                ctrl.nudge_yaw(math.radians(15) * (1 if a == "yaw+" else -1), "mouse")

        focused = kb.focused() if (script is None or script.focus_override is None) else script.focus_override
        if not focused:
            kb.clear()
            panel.cancel()
            handle.cancel()
            ctrl.clear_yaw_goal()
            ctrl.clear_gripper_goal()

        input_lat_ms = None
        mouse_d = np.zeros(3)
        if paused:
            # input received while paused is dropped, so nothing replays on resume
            kb.clear()
            panel.cancel()
            handle.cancel()
            sim.render()
        else:
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
            tp = tick("input+ik", tp)
            q_prev = ctrl.state.q_cmd.copy()
            ctrl.joint_command(CONTROL_DT)
            max_dq_cmd = float(np.max(np.abs(ctrl.state.q_cmd - q_prev)))
            robot.set_joint_position_target(full_command())
            for i in range(SUBSTEPS):
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(PHYSICS_DT)
                block.update(PHYSICS_DT)
            t_sim += CONTROL_DT
            tp = tick("physics", tp)
            sim.render()
            tp = tick("render", tp)

        q_meas = robot.data.joint_pos[0, arm_ids].cpu().numpy()
        ee = ctrl.ee_pos(q_meas)
        bpos = (block.data.root_pos_w[0] - 0).cpu().numpy()
        s = ctrl.state
        markers.visualize(translations=torch.tensor(np.array([ee]), dtype=torch.float32),
                          marker_indices=torch.tensor([0]))
        handle.update(s.pos, kind_of(s.status))
        focus_mode.update()
        if focus_at_frame >= 0:
            focus_at_frame -= 1
            if focus_at_frame < 0:
                focus_mode.enter()
        mode = "focus lost" if not focused else ("keyboard" if s.source == "keyboard" else s.source)
        panel.update(mode, paused, kb.shift, s.status, s.message, s.pos, s.yaw_offset, ee, bpos, s.gripper,
                     block_color[0])
        follow_mm = float(np.linalg.norm(s.pos - ee)) * 1000.0
        target_step = float(np.linalg.norm(s.pos - prev_target))
        if target_step > 1e-9 or kb.held:
            last_input_t = now
        metrics["follow_err_mm"].append(follow_mm)
        if now - last_input_t > 1.0 and not paused:  # settled: no target change for 1 s
            metrics["follow_err_idle_mm"].append(follow_mm)
        prev_target = s.pos.copy()
        sel = list(usd_ctx.get_selection().get_selected_prim_paths())
        if sel != prev_sel:
            log(f"selection changed: {sel}")
            prev_sel = sel
        if s.status != prev_status:
            log(f"status {prev_status} -> {s.status} {s.message}")
            prev_status = s.status

        stats["max_dq_cmd_per_frame"] = max(stats["max_dq_cmd_per_frame"], max_dq_cmd)
        qd_max = float(robot.data.joint_vel[0, arm_ids].abs().max())
        stats["max_joint_vel"] = max(stats["max_joint_vel"], qd_max)
        stats["rejected_targets"] = carry["rejected"] + s.rejected
        stats["block_lifted_max_z"] = max(stats["block_lifted_max_z"], float(bpos[2]))

        if log_w:
            log_w.writerow([f"{now - t_start:.4f}", f"{t_sim:.4f}", s.session, int(paused), int(focused), s.source,
                            s.status, "".join(sorted(kb.held)), *np.round(s.pos, 5), round(s.yaw_offset, 4),
                            *np.round(ee, 5), *np.round(bpos, 5), round(s.gripper, 4),
                            round(float(robot.data.joint_pos[0, jaw_id]), 4), *np.round(s.q_cmd, 5),
                            *np.round(q_meas, 5), round(max_dq_cmd, 5), round(qd_max, 4), int(kb.focused()),
                            s.message, round(follow_mm, 2), "" if input_lat_ms is None else round(input_lat_ms, 2),
                            *np.round(mouse_d, 6), int(kb.shift), int(handle.dragging)])

        if script is not None:
            ns = {"target": s.pos, "ee": ee, "block": bpos, "status": s.status, "gripper": s.gripper,
                  "jaw": float(robot.data.joint_pos[0, jaw_id]), "paused": paused, "session": s.session,
                  "focused": focused, "held": set(kb.held), "rejected": s.rejected, "yaw_offset": s.yaw_offset,
                  "block_in_placement": block_in_tray(bpos, h.trays["placement_beige"], edge),
                  "block_in_storage": block_in_tray(bpos, h.trays["storage_pink"], edge),
                  "stats": stats, "dq_cmd": max_dq_cmd, "home": ctrl.p.home_ee.copy(),
                  "block_init": np.array(h.block_init_pos), "dragging": handle.dragging, "hover": handle.hover,
                  "fine": kb.shift, "target_step": target_step, "follow_err_mm": follow_mm,
                  "cam": handle.camera_position(),
                  "selection": list(usd_ctx.get_selection().get_selected_prim_paths()),
                  "focus_mode": focus_mode.active, "editors_visible": focus_mode.editor_windows_visible(),
                  "front_tabs": focus_mode.front_tabs(),
                  "dr": color_dr.current, "dr_records": len(color_dr.records),
                  "block_mat_rgb": [round(float(x), 6) for x in
                                    usd_ctx.get_stage().GetPrimAtPath("/World/Block/geometry/material/Shader")
                                    .GetAttribute("inputs:diffuseColor").Get()],
                  "shell_mat_rgb": [round(float(x), 6) for x in
                                    usd_ctx.get_stage().GetPrimAtPath("/World/Robot/Looks/material_a_3d_printed/Shader")
                                    .GetAttribute("inputs:diffuse_color_constant").Get()],
                  "reasons": {r: carry["reasons"][r] + s.reject_counts[r] for r in REASONS}}
            script.step(ns)
            if script.done and args.exit_after_script:
                break

        tp = tick("ui+log", tp)
        prof["frames"] = prof.get("frames", 0) + 1
        if args.profile and time.monotonic() - prof_t > 5.0:
            n = prof.pop("frames")
            log("profile ms/frame: " + ", ".join(f"{k}={1000 * v / n:.1f}" for k, v in prof.items())
                + f"  fps={n / (time.monotonic() - prof_t):.1f}")
            prof.clear()
            prof_t = time.monotonic()
        if args.max_time and now - t_start > args.max_time:
            break
        # pace to real time so physics does not run faster than the operator sees
        spare = CONTROL_DT - (time.monotonic() - now)
        if spare > 0:
            time.sleep(spare)

    if script is not None:
        script.finish()
    if log_f:
        log_f.close()

    def pct(xs):
        if not xs:
            return None
        a = np.asarray(xs)
        return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 2),
                "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}

    summary = {
        "reject_reason_counts": {r: carry["reasons"][r] + ctrl.state.reject_counts[r] for r in REASONS},
        "mouse_input_latency_ms": pct(metrics["input_lat_ms"]),
        "follow_error_mm": pct(metrics["follow_err_mm"]),
        "follow_error_settled_mm": pct(metrics["follow_err_idle_mm"]),
    }
    if args.log or args.result:
        log("metrics: " + json.dumps(summary))
    if script is not None and args.result:
        os.makedirs(os.path.dirname(args.result) or ".", exist_ok=True)
        with open(args.result, "w") as f:
            json.dump({"checks": script.results, "stats": stats, "metrics": summary, "script": args.script,
                       "color_dr": {"enabled": color_dr.enabled, "seed": color_dr.seed, "records": color_dr.records}},
                      f, indent=2)
        log(f"wrote {args.result}: {sum(r['ok'] for r in script.results)}/{len(script.results)} checks passed")
    kb.close()


main()
sys.stdout.flush()
os._exit(0)  # SimulationApp.close() can hang in Isaac Sim 5.1; all outputs are flushed above
