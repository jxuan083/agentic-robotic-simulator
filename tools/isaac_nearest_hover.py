#!/usr/bin/env python3
"""Replay nearest-colour trials of tools/benchmark_verified_overhead.py in Isaac Sim, unchanged trial().

Layouts, colours, yaws and injected coarse errors come from docs/nearest-color-5060.jsonl (by seed). An adapter gives
trial() the Benchmark interface backed by Isaac: RGB captures of the existing global / wrist sensor views with the
matrices of the captured frame, measured joints, the session's IK chain and EE control point, PD joint targets stepped
by PhysX. Control reads images and measured joints only; block poses are read for scoring and the frame overlays.

Optional, Isaac only (not part of trial()):
  --lift_comp_mm 6   one upward hover correction when the arm has settled with identity and XY confirmed and only the
                     height is low: min(80 mm - measured height, cap); IK from the measured joints, sent once. The
                     descent gate stays at 80 mm (77-83 mm, XY <= 3 mm, joint speed <= 0.15 rad/s in the same frame).
  --hold_s 1.0       log every control step of the hover for this long (height curve); the decision is unchanged.

Run in the Isaac container from the repo root (docs/wrist-alignment.md):
  docker/isaac/run.sh /isaac-sim/python.sh tools/isaac_nearest_hover.py --seeds 85000 --lift_comp_mm 6 \
      --out outputs/isaac_nearest_85000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]
os.chdir(ROOT)

import numpy as np  # noqa: E402

CASE_SUITE = {"adjacent_red_storage": "configs/benchmarks/adjacent_red_holdout.yaml",
              "adjacent_red_placement": "configs/benchmarks/adjacent_red_holdout.yaml"}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if hasattr(x, "record") and hasattr(x, "xy"):
        return x.record()
    return x


def case_config(case_name):
    """The benchmark config of a 5060 case: suite base + the case's bench overrides."""
    from agentic_sim.config import load
    suite = load(CASE_SUITE.get(case_name, "configs/benchmarks/finals_overhead_suite.yaml"))
    case = next(c for c in suite["cases"] if c["name"] == case_name)
    cfg = load(suite["base"])
    cfg.update(case.get("bench", {}))
    cfg["settle_qd"] = load("configs/task.yaml")["motion"]["settle_qd"]
    return cfg


def compensation_summary(result):
    """Heights before / after the optional hover correction of the last hover, or None when it was not tried."""
    c = (result.get("hovers") or [{}])[-1].get("compensation")
    if c is None:
        return None
    out = {"applied": c.get("applied"), "reason": c.get("reason"),
           "before_z_mm": round(c["before"]["ee_m"][2] * 1000, 2)}
    if c.get("applied"):
        out.update(comp_mm=round(c["comp_mm"], 2), command_z_mm=round(c["command_z_m"] * 1000, 2))
    if "after" in c:
        out.update(after_z_mm=round(c["after"]["ee_m"][2] * 1000, 2), passed=c["after"]["passed"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--hold_s", type=float, default=0.0,
                   help="hover curve: after the final hover joint command is reached (t0), hold the same target this "
                        "long and log every control step; the descent decision is the unchanged one at t0 + 0.15 s")
    p.add_argument("--lift_comp_mm", type=float, default=0.0,
                   help="one upward hover correction (cap, mm) when the arm has settled with identity and XY confirmed "
                        "and only the height is low; the descent gate stays at the original 80 mm")
    p.add_argument("--render_every", type=int, default=2, help="GUI: draw every Nth control step during motion")
    a = p.parse_args()
    if a.lift_comp_mm < 0 or a.hold_s < 0:
        p.error("--lift_comp_mm and --hold_s must not be negative")
    if a.lift_comp_mm and a.hold_s:
        p.error("--lift_comp_mm and --hold_s are separate experiments; pass one")
    if (a.out / "summary.json").exists():
        p.error(f"{a.out} already holds a run; choose a new --out")
    from isaaclab.app import AppLauncher
    launcher = AppLauncher({"headless": a.headless, "device": "cpu",  # noqa: F841
                             "kit_args": "--/app/vsync=false --/app/runLoops/main/rateLimitEnabled=false"})
    try:
        run(a)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def run(a):
    import cv2
    import torch
    from omni.kit.app import get_app

    import types
    if "mujoco" not in sys.modules:
        # The Isaac container has no MuJoCo. The benchmark module only needs these names at import time; every
        # method that would call MuJoCo (reset, motion, snapshot, depth) is overridden by the Isaac adapter below.
        stub = types.ModuleType("mujoco")
        stub.__version__ = "not used (Isaac adapter)"
        stub.mjtState = types.SimpleNamespace(mjSTATE_INTEGRATION=None)
        sys.modules["mujoco"] = stub
    import benchmark_finals_alignment as bfa
    import benchmark_verified_overhead as bvo
    from agentic_sim.config import load
    from agentic_sim.isaac.live_perception import _Capture
    from agentic_sim.isaac.session import CONTROL_DT, PHYSICS_DT, SUBSTEPS, SessionConfig, TeleopSession
    from agentic_sim.perception.live import LiveEstimator, rendered_camera
    from agentic_sim.perception.vertices import CameraGeometry, cube_vertices
    from agentic_sim.random_blocks import REGION_TRAY
    from agentic_sim.scene_layout import tray_parts, yaw_quat

    ref = {r["seed"]: r for r in map(json.loads, open("docs/nearest-color-5060.jsonl"))}
    rows = [ref[s] for s in a.seeds]
    colors = rows[0]["colors"]
    assert all(r["colors"] == colors for r in rows)
    scene_path = load("configs/benchmarks/finals_alignment.yaml")["scene"]
    session = TeleopSession(SessionConfig(scene=scene_path, block_colors=list(colors), focus=not a.headless,
                                          device="cpu"), log=lambda x: print(f"[isaac] {x}", flush=True))
    app = get_app()
    for _ in range(20):
        session.step()
        app.update()
    a.out.mkdir(parents=True, exist_ok=True)

    class Data:
        """Benchmark's MuJoCo `data` fields, read from Isaac when accessed (arm 0..4, jaw 5; blocks 0..3)."""
        def __init__(self):
            self.ctrl = np.zeros(6)
            self.time = 0.0

        @property
        def qpos(self):
            q = session.robot.data.joint_pos[0].cpu().numpy()
            return np.r_[q[session.arm_ids], q[session.jaw_id]].astype(float)

        @property
        def qvel(self):
            q = session.robot.data.joint_vel[0].cpu().numpy()
            return np.r_[q[session.arm_ids], q[session.jaw_id]].astype(float)

        @property
        def xpos(self):
            return np.array([b.data.root_pos_w[0].cpu().numpy() for b in session.blocks], float)

    class Scene:
        arm_qpos = arm_act = arm_qvel = np.arange(5)
        jaw_qpos = jaw_act = 5
        arm_offset = np.zeros(5)
        block_bodies = np.arange(len(colors))
        edge = float(session.edge)

    class Perception:
        def __init__(self, floor):
            self.estimators = {k: LiveEstimator(Scene.edge, floor) for k in ("global", "wrist")}
            self.last = {}

        def capture_rgb(self, key, data):
            q = data.qpos[:5]
            base = CameraGeometry.from_scene(session.scene_cfg, key, q if key == "wrist" else None)
            inbox = queue.SimpleQueue()
            session.sensor_views.windows[key].viewport_api.schedule_capture(_Capture(inbox, 0))
            start = time.monotonic()
            while time.monotonic() - start < 45:
                session.sim.render()
                app.update()
                if not inbox.empty():
                    _, rgb, view, projection, frame, at, error = inbox.get()
                    if error:
                        raise RuntimeError(f"{key} capture failed: {error}")
                    cam = rendered_camera(base, view, projection, rgb.shape[1], rgb.shape[0])
                    p_err = float(np.linalg.norm(cam.mount_from_cv[:3, 3] - base.mount_from_cv[:3, 3]))
                    self.last[key] = {"rgb": rgb, "camera": cam, "frame": frame, "fk_mismatch_mm": p_err * 1000}
                    return rgb, cam
            raise TimeoutError(f"{key} capture timed out")

        def close(self):
            pass

    class IsaacBench(bfa.Benchmark):
        def __init__(self, cfg):
            self.cfg = cfg
            self.scene_config = session.scene_cfg
            self.colors = list(colors)
            self.scene = Scene()
            self.data = Data()
            self.ctrl, self.chain = session.ctrl, session.ctrl.chain
            self.pan = session.ctrl.pan[:2]
            self.dt = CONTROL_DT
            self.region = cfg.get("source_region", "storage")
            info = tray_parts(session.scene_cfg["trays"][REGION_TRAY[self.region]])[1]
            self.source_tray = {k: info[k] for k in ("center", "inner_half", "yaw", "floor_top")}
            self.source_tray["center"] = np.asarray(self.source_tray["center"], float)
            self.source_tray["inner_half"] = np.asarray(self.source_tray["inner_half"], float)
            self.floor = self.source_tray["floor_top"]
            self.perception = Perception(self.floor)
            self.depth_renderer = None
            self.jaw_open, self.jaw_closed = session.ctrl.p.gripper_open, session.ctrl.p.gripper_closed
            self.frames = 0

        def apply(self):
            q = session.robot.data.default_joint_pos.clone()
            q[0, session.arm_ids] = torch.tensor(self.data.ctrl[:5], dtype=q.dtype)
            q[0, session.jaw_id] = float(self.data.ctrl[5])
            session.robot.set_joint_position_target(q)
            return q

        def physics(self):
            for _ in range(SUBSTEPS):
                session.robot.write_data_to_sim()
                session.sim.step(render=False)
                session.robot.update(PHYSICS_DT)
                for b in session.blocks:
                    b.update(PHYSICS_DT)
            self.data.time += CONTROL_DT
            session.wrist_camera.update()
            self.frames += 1
            if not a.headless and self.frames % a.render_every == 0:
                session.sim.render()
                app.update()

        def reset(self, episode):
            d = self.data
            q_home = np.asarray(session.ctrl.q_home, float)
            d.ctrl[:5], d.ctrl[5] = q_home, self.jaw_open
            q = self.apply()
            session.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            session.robot.reset()
            for b, pos, yaw in zip(session.blocks, episode.layout["positions_m"], episode.layout["yaw_degs"]):
                pose = torch.tensor([[*pos, *yaw_quat(yaw)]], dtype=torch.float32)
                b.write_root_pose_to_sim(pose)
                b.write_root_velocity_to_sim(torch.zeros((1, 6)))
                b.reset()
            for _ in range(15):        # MuJoCo reset steps 60 substeps before t = 0
                self.physics()
            d.time = 0.0

        def motion(self, duration, q=None, jaw=None, score=None):
            d = self.data
            arm_now, jaw_now = d.ctrl[:5].copy(), float(d.ctrl[5])
            arm_goal = arm_now if q is None else np.asarray(q, float)
            jaw_goal = jaw_now if jaw is None else jaw
            frames = math.ceil(duration / self.dt - 1e-6)
            for k in range(1, frames + 1):
                w = k / frames
                w = w * w * (3 - 2 * w)
                d.ctrl[:5] = arm_now + w * (arm_goal - arm_now)
                d.ctrl[5] = jaw_now + w * (jaw_goal - jaw_now)
                self.apply()
                self.physics()
                if score is not None:
                    score.update(self)

    # ---- evidence: three frames per trial and the scored hover ----
    def truth():
        out = []
        for i, b in enumerate(session.blocks):
            pos = b.data.root_pos_w[0].cpu().numpy().astype(float)
            w, x, y, z = b.data.root_quat_w[0].cpu().numpy().astype(float)
            out.append({"id": i, "color": colors[i], "xyz_m": pos,
                        "yaw_rad": math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))})
        return out

    def draw(sim, ep, label, estimate):
        last = sim.perception.last["wrist"]
        img, cam = last["rgb"].copy(), last["camera"]
        for t in truth():
            verts = cube_vertices(t["xyz_m"][:2], t["yaw_rad"], Scene.edge, t["xyz_m"][2] - Scene.edge / 2)
            try:
                uv = cam.project(verts)
            except ValueError:
                continue
            col = (0, 200, 255) if t["id"] == ep.target_index else (
                (255, 80, 0) if t["color"] == ep.target_color else (200, 200, 200))
            for i, j in EDGES:
                cv2.line(img, tuple(np.rint(uv[i]).astype(int)), tuple(np.rint(uv[j]).astype(int)), col, 1, cv2.LINE_AA)
            c = tuple(uv[4:].mean(axis=0).astype(int))
            cv2.putText(img, f"T{t['id']}" + ("*" if t["id"] == ep.target_index else ""), c,
                        cv2.FONT_HERSHEY_SIMPLEX, .45, col, 1, cv2.LINE_AA)
        if estimate is not None:
            try:
                uv = cam.project(np.array([[*estimate.xy, sim.floor + Scene.edge]]))[0]
                cv2.drawMarker(img, tuple(np.rint(uv).astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            except ValueError:
                pass
        q = sim.data.qpos[:5]
        ee = sim.chain.ee_pose_mount(q)[:3, 3]
        try:
            uv = cam.project(ee[None, :] * [1, 1, 0] + [0, 0, sim.floor + Scene.edge])[0]
            cv2.drawMarker(img, tuple(np.rint(uv).astype(int)), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        except ValueError:
            pass
        h = img.shape[0]
        cv2.rectangle(img, (0, h - 18), (img.shape[1], h), (0, 0, 0), -1)
        cv2.putText(img, f"Isaac {label} t_sim {sim.data.time:.3f}s | cyan*=near target orange=other red "
                    "gray=other | green+=image estimate magenta x=EE xy", (4, h - 5), cv2.FONT_HERSHEY_SIMPLEX, .36,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    summary = []
    orig_wrist, orig_hover = bfa.Benchmark.wrist, bvo.at_hover
    for row in rows:
        cfg = case_config(row["case"])
        sim = IsaacBench(cfg)
        ep = bfa.Episode({"positions_m": row["positions_m"], "yaw_degs": row["yaw_degs"]}, row["target_color"],
                         row["target_index"], np.asarray(row["injected_xy_m"]), math.radians(row["injected_yaw_deg"]))
        out = a.out / str(row["seed"])
        out.mkdir(parents=True, exist_ok=True)
        frames, hover_log = [], []
        same = [i for i, c in enumerate(colors) if c == ep.target_color and i != ep.target_index]

        def record(label, estimate, extra=None):
            q = sim.data.qpos
            ee = sim.chain.ee_pose_mount(q[:5])[:3, 3]
            t = truth()
            name = f"{len(frames):02d}_{label}.png"
            cv2.imwrite(str(out / name), cv2.cvtColor(draw(sim, ep, label, estimate), cv2.COLOR_RGB2BGR))
            frames.append(jsonable({
                "label": label, "file": name, "t_sim_s": sim.data.time, "q_measured_rad": q[:5], "jaw_rad": q[5],
                "joint_speed_max_rad_s": float(np.max(np.abs(sim.data.qvel[:5]))), "ee_xyz_m": ee,
                "image_estimate_xy_m": None if estimate is None else estimate.xy,
                "ee_to_near_xy_mm": float(np.linalg.norm(ee[:2] - t[ep.target_index]["xyz_m"][:2]) * 1000),
                "ee_to_far_xy_mm": [float(np.linalg.norm(ee[:2] - t[i]["xyz_m"][:2]) * 1000) for i in same],
                "fk_mismatch_mm": sim.perception.last["wrist"]["fk_mismatch_mm"], **(extra or {})}))

        def wrist(self, anchor, color):
            found = orig_wrist(self, anchor, color)
            record(f"wrist{sum(f['label'].startswith('wrist') for f in frames) + 1}", found,
                   {"confirmed": found is not None})
            return found

        def hover_curve(s, estimate):
            """bvo.at_hover with every control step logged and the hold extended to --hold_s (same decision)."""
            height = s.cfg["hover_z_m"]
            q = s.solve(estimate.xy, height, s.heading(estimate))
            if q is None:
                return orig_hover(s, estimate)
            s.motion(0.55, q=q)
            t0, samples = s.data.time, []
            near = ep.target_index

            def sample():
                meas = s.data.qpos[:5]
                ee = s.chain.ee_pose_mount(meas)[:3, 3]
                t = truth()
                speed = float(np.max(np.abs(s.data.qvel[:5])))
                xy = float(np.linalg.norm(ee[:2] - estimate.xy) * 1000)
                z = float(ee[2] - height) * 1000
                samples.append(jsonable({
                    "t_sim_s": s.data.time, "t_from_t0_s": s.data.time - t0, "q_target_rad": s.data.ctrl[:5].copy(),
                    "q_measured_rad": meas, "ee_xyz_m": ee, "joint_speed_max_rad_s": speed,
                    "xy_error_to_estimate_mm": xy, "z_error_mm": z,
                    "ready": xy <= 3 and abs(z) <= 3 and speed <= s.cfg["settle_qd"],
                    "truth_ee_to_near_xy_mm": float(np.linalg.norm(ee[:2] - t[near]["xyz_m"][:2]) * 1000)}))
            sample()
            decision_step = math.ceil(0.15 / s.dt - 1e-6)     # bvo.at_hover holds 0.15 s, then decides
            for _ in range(max(decision_step, math.ceil(a.hold_s / s.dt - 1e-6))):
                s.motion(s.dt)
                sample()
            d = samples[decision_step]
            ee = np.asarray(d["ee_xyz_m"])
            passed = bool(d["ready"])
            h = {"passed": passed, "status": "ready" if passed else "hover_not_settled", "ee_m": ee.tolist(),
                 "estimated_target_xy_m": estimate.xy.tolist(), "xy_error_mm": d["xy_error_to_estimate_mm"],
                 "z_error_mm": abs(d["z_error_mm"]), "joint_speed_rad_s": d["joint_speed_max_rad_s"],
                 "decision_t_from_t0_s": d["t_from_t0_s"], "curve_file": f"hover{len(hover_log) + 1}_curve.json"}
            first = next((i for i, x in enumerate(samples) if x["ready"]), None)
            h["first_ready_t_from_t0_s"] = None if first is None else samples[first]["t_from_t0_s"]
            h["ready_held_to_end"] = first is not None and all(x["ready"] for x in samples[first:])
            (out / h["curve_file"]).write_text(json.dumps({"t0_sim_s": t0, "hover_z_m": height,
                                                           "estimate_xy_m": estimate.xy, "samples": samples},
                                                          default=jsonable, indent=1))
            return h

        def joint_state(s):
            meas = s.data.qpos[:5]
            lo, hi = s.chain.lower, s.chain.upper
            return {"q_command_rad": s.data.ctrl[:5].copy(), "q_measured_rad": meas,
                    "command_margin_to_limit_rad": np.minimum(s.data.ctrl[:5] - lo, hi - s.data.ctrl[:5]),
                    "measured_margin_to_limit_rad": np.minimum(meas - lo, hi - meas)}

        def hover_comp(s, estimate):
            """bvo.at_hover, then at most one upward correction of the commanded height (Isaac experiment)."""
            h = orig_hover(s, estimate)
            if "ee_m" not in h:
                return h
            height = s.cfg["hover_z_m"]
            before = dict(h, **joint_state(s))
            h = dict(h, compensation={"before": before})
            if h["passed"]:
                h["compensation"]["applied"] = False
                h["compensation"]["reason"] = "passed_without_correction"
                return h
            z = before["ee_m"][2]
            settled = before["joint_speed_rad_s"] <= s.cfg["settle_qd"] and before["xy_error_mm"] <= 3
            if not settled or z >= height:
                h["compensation"].update(applied=False, reason="not_settled_or_not_low" if not settled else "not_low")
                return h
            # record the frame before the correction (render only)
            s.perception.capture_rgb("wrist", s.data)
            record(f"hover{len(hover_log) + 1}_before_comp", estimate, {"hover": before})
            comp = min(height - z, a.lift_comp_mm / 1000)
            q = s.solve(estimate.xy, height + comp, s.heading(estimate), seed=s.data.qpos[:5])
            h["compensation"].update(applied=q is not None, comp_mm=comp * 1000, command_z_m=height + comp)
            if q is None:
                h["compensation"]["reason"] = "ik_no_solution"
                return h
            s.motion(0.55, q=q)
            s.motion(0.15)
            meas = s.data.qpos[:5]
            ee = s.chain.ee_pose_mount(meas)[:3, 3]
            speed = float(np.max(np.abs(s.data.qvel[:5])))
            xy = float(np.linalg.norm(ee[:2] - estimate.xy) * 1000)
            zerr = abs(float(ee[2]) - height) * 1000          # gate stays at the original 80 mm
            passed = xy <= 3 and zerr <= 3 and speed <= s.cfg["settle_qd"]
            after = {"ee_m": ee.tolist(), "xy_error_mm": xy, "z_error_mm": zerr, "joint_speed_rad_s": speed,
                     "passed": passed, **joint_state(s)}
            h["compensation"]["after"] = after
            h.update(passed=passed, status="ready" if passed else "hover_not_settled", ee_m=ee.tolist(),
                     xy_error_mm=xy, z_error_mm=zerr, joint_speed_rad_s=speed)
            return h

        def at_hover(s, estimate):
            h = (hover_curve(s, estimate) if a.hold_s > 0 else
                 hover_comp(s, estimate) if a.lift_comp_mm > 0 else orig_hover(s, estimate))
            s.perception.capture_rgb("wrist", s.data)     # render only: no physics step
            record(f"hover{len(hover_log) + 1}", estimate, {"hover": h})
            hover_log.append(h)
            return h

        bfa.Benchmark.wrist, bvo.at_hover = wrist, at_hover
        print(f"[pr19] seed {row['seed']} case {row['case']} target {ep.target_index} ({ep.target_color})", flush=True)
        try:
            result = bvo.trial(sim, ep)
        finally:
            bfa.Benchmark.wrist, bvo.at_hover = orig_wrist, orig_hover
        first = result.get("first_grasp", {}).get("stable_block_indices", [])
        retry = result.get("empty_grasp_retry", {}).get("grasp", {}).get("stable_block_indices", [])
        hovers = [f for f in frames if f["label"].startswith("hover")]
        last_hover = hovers[-1] if hovers else None
        summary_row = jsonable({
            "seed": row["seed"], "case": row["case"], "target": ep.target_index, "color": ep.target_color,
            "status": result["status"], "success": result.get("success", False),
            "view_changed": result.get("retry") is not None or bool(result.get("separation_views")),
            "retry_reason": (result.get("retry") or {}).get("reason"),
            "separation_views": result.get("separation_views"),
            "descended": "first_grasp" in result, "first_lifted_ids": first, "retry_lifted_ids": retry,
            "wrong_lock": bool(set(first + retry) - {ep.target_index}),
            "final_hover": last_hover and {k: last_hover[k] for k in ("t_sim_s", "ee_xyz_m", "image_estimate_xy_m",
                                                                     "ee_to_near_xy_mm", "ee_to_far_xy_mm",
                                                                     "joint_speed_max_rad_s", "hover")},
            "compensation": compensation_summary(result),
            "mujoco_5060": {"status": row["status"], "first_lifted_ids": row["first_lifted_ids"],
                            "retry_lifted_ids": row["retry_lifted_ids"]}})
        (out / "trial.json").write_text(json.dumps(jsonable({"seed": row["seed"], **result}), indent=1))
        (out / "frames.json").write_text(json.dumps(frames, indent=1))
        (out / "summary.json").write_text(json.dumps(summary_row, indent=1))
        summary.append(summary_row)
        print("[pr19] " + json.dumps({k: summary_row[k] for k in ("seed", "status", "view_changed", "compensation",
                                                                  "descended", "first_lifted_ids", "wrong_lock")}),
              flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    session.close()


if __name__ == "__main__":
    main()
