"""Native MuJoCo finals first-lift experiment: paired RGB approach and wrist alignment."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import time

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_sim.config import load, v
from agentic_sim.kinematics import So101Chain
from agentic_sim.mujoco_backend.live_perception import MjLivePerception
from agentic_sim.mujoco_backend.model import build
from agentic_sim.perception.localize import global_candidates, plan_wrist_view, refine_with_wrist
from agentic_sim.perception.vertices import cube_vertices
from agentic_sim.random_blocks import REGION_TRAY, grasp_ik_checker, sample_scene
from agentic_sim.scene_layout import block_in_tray, yaw_quat
from agentic_sim.teleop.ee_target import load_controller
from agentic_sim.task.wrist_alignment import align_from_hover

MOD = math.pi / 2
METHODS = ("A_coarse", "B_wrist", "C_rl", "D_rgbd")
# Index zero is the hand-coded baseline action and tie-breaker for the learned policy.
VIEW_GAINS = (1.0, 0.5, 0.0)
# Extra correction in the gripper frame: closing and perpendicular directions (m).
FINAL_OFFSETS = ((0.0, 0.0), (-0.006, 0.0), (0.006, 0.0), (0.0, -0.006), (0.0, 0.006))
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def yaw_delta(a, b):
    return (a - b + MOD / 2) % MOD - MOD / 2


@dataclass
class Estimate:
    xy: np.ndarray
    yaw: float
    fit: float
    z: float | None = None

    @classmethod
    def from_record(cls, record):
        return cls(np.asarray(record["center_xyz_m"][:2]), record["yaw_rad_mod_pi_over_2"], record["fit_iou"])

    def record(self):
        return {"xy_m": self.xy.tolist(), "yaw_deg_mod_90": math.degrees(self.yaw % MOD), "fit_iou": self.fit, "center_z_m": self.z}


@dataclass
class Episode:
    layout: dict
    target_color: str
    target_index: int
    offset_xy: np.ndarray
    offset_yaw: float


class Score:
    """Read object truth for reward/scoring only, throughout the common lift and hold."""
    def __init__(self, episode, cfg, start):
        self.target = episode.target_index
        self.floor = np.asarray(episode.layout["positions_m"])[:, 2] - 0.0005
        self.held = np.zeros_like(self.floor)
        self.maximum = np.zeros_like(self.floor)
        self.height, self.hold = cfg["stable_lift_height_m"], cfg["stable_lift_hold_s"]
        self.start = start
        self.success_time = None
        self.stable_objects = set()

    def update(self, sim):
        d = sim.data
        height = d.xpos[sim.scene.block_bodies, 2] - self.floor
        self.maximum = np.maximum(self.maximum, height)
        self.held = np.where(height >= self.height, self.held + sim.dt, 0.0)
        self.stable_objects.update(np.nonzero(self.held + 1e-8 >= self.hold)[0].tolist())
        if self.target in self.stable_objects and self.success_time is None:
            self.success_time = (float(d.time), time.perf_counter() - self.start)

    def result(self):
        success = self.success_time is not None
        wrong = sorted(self.stable_objects - {self.target})
        return {
            "status": "stable_correct_lift" if success else ("stable_wrong_lift" if wrong else "no_stable_lift"),
            "success": success,
            "first_correct_lift_sim_s": self.success_time[0] if success else None,
            "first_correct_lift_wall_s": self.success_time[1] if success else None,
            "stable_block_indices": sorted(self.stable_objects),
            "max_target_rise_m": float(self.maximum[self.target]),
        }

    def reward(self):
        hit = float(self.success_time is not None)
        wrong = float(bool(self.stable_objects - {self.target}))
        rise = min(1.0, self.maximum[self.target] / self.height)
        return 2.0 * hit - 0.55 * wrong + 0.25 * rise


class Benchmark:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scene_config = load(cfg["scene"])
        self.layout_config = load(cfg["layout"])
        self.colors = self.layout_config["fixed_colors"]
        self.reachable = grasp_ik_checker(self.scene_config, cfg["teleop"])
        self.scene = build(self.scene_config, colors=self.colors)
        self.model = self.scene.model
        self.data = mujoco.MjData(self.model)
        ctrl = load_controller(So101Chain.from_dict(load("configs/so101_chain.yaml")), load(cfg["teleop"]),
                               base_pos=v(self.scene_config["robot"]["base_pos"]), scene=self.scene_config)
        self.chain, self.ctrl, self.pan = ctrl.chain, ctrl, ctrl.pan[:2]
        self.dt = self.scene.robot_cfg["physics"]["control_dt"]
        self.substeps = round(self.dt / self.model.opt.timestep)
        self.state_size = mujoco.mj_stateSize(self.model, STATE)
        self.perception = MjLivePerception(self.scene_config, self.model, log=lambda _: None)
        self.depth_renderer = None
        self.region = cfg.get("source_region", "storage")
        self.source_tray = self.scene.trays[REGION_TRAY[self.region]]
        self.floor = self.source_tray["floor_top"]
        for estimator in self.perception.estimators.values():
            estimator.floor = self.floor
        self.jaw_open, self.jaw_closed = ctrl.p.gripper_open, ctrl.p.gripper_closed

    def episode(self, seed, cycle_index):
        _, _, layout = sample_scene(self.scene_config, self.layout_config, seed=seed, region=self.region, reachable=self.reachable)
        color = self.cfg["target_cycle"][cycle_index % len(self.cfg["target_cycle"])]
        target = min((i for i, c in enumerate(self.colors) if c == color),
                     key=lambda i: np.linalg.norm(np.asarray(layout["positions_m"][i][:2]) - self.pan))
        rng = np.random.default_rng(seed + 614422)
        radius = self.cfg["coarse_xy_error_mm"] / 1000 * math.sqrt(rng.random())
        angle = rng.uniform(-math.pi, math.pi)
        offset = radius * np.array([math.cos(angle), math.sin(angle)])
        yaw = math.radians(rng.uniform(-self.cfg["coarse_yaw_error_deg"], self.cfg["coarse_yaw_error_deg"]))
        return Episode(layout, color, target, offset, yaw)

    def reset(self, episode):
        d, sc = self.data, self.scene
        mujoco.mj_resetData(self.model, d)
        home = self.ctrl.q_home + sc.arm_offset
        d.qpos[sc.arm_qpos] = d.ctrl[sc.arm_act] = home
        d.qpos[sc.jaw_qpos] = d.ctrl[sc.jaw_act] = self.jaw_open
        for address, pos, yaw in zip(sc.block_qpos, episode.layout["positions_m"], episode.layout["yaw_degs"]):
            d.qpos[address:address + 3] = pos
            d.qpos[address + 3:address + 7] = yaw_quat(yaw)
        mujoco.mj_forward(self.model, d)
        mujoco.mj_step(self.model, d, nstep=60)
        d.time = 0.0

    def snapshot(self):
        state = np.empty(self.state_size)
        mujoco.mj_getState(self.model, self.data, state, STATE)
        return state

    def restore(self, state):
        mujoco.mj_setState(self.model, self.data, state, STATE)
        mujoco.mj_forward(self.model, self.data)

    def motion(self, duration, q=None, jaw=None, score=None):
        d, sc = self.data, self.scene
        arm_now, jaw_now = d.ctrl[sc.arm_act].copy(), float(d.ctrl[sc.jaw_act])
        arm_goal = arm_now if q is None else q + sc.arm_offset
        jaw_goal = jaw_now if jaw is None else jaw
        frames = math.ceil(duration / self.dt - 1e-6)
        for k in range(1, frames + 1):
            w = k / frames
            w = w * w * (3 - 2 * w)
            d.ctrl[sc.arm_act] = arm_now + w * (arm_goal - arm_now)
            d.ctrl[sc.jaw_act] = jaw_now + w * (jaw_goal - jaw_now)
            mujoco.mj_step(self.model, d, nstep=self.substeps)
            if score is not None:
                score.update(self)

    def solve(self, xy, z, heading, seed=None):
        pos = np.array([*xy, z])
        if np.any(pos < self.ctrl.p.ws_lo) or np.any(pos > self.ctrl.p.ws_hi):
            return None
        if seed is None:
            seed = self.data.ctrl[self.scene.arm_act] - self.scene.arm_offset
        q, ok, _ = self.chain.solve_ik_tilt(pos, heading, seed, self.ctrl.p.max_tilt_deg)
        return q if ok else None

    def heading(self, pose):
        dx, dy = pose.xy - self.pan
        neutral = math.atan2(dy, dx) + self.ctrl.p.finger_heading
        return neutral + yaw_delta(pose.yaw, neutral)

    def image(self, key):
        now = time.perf_counter()
        rgb, geometry = self.perception.capture_rgb(key, self.data)
        # The detector sees RGB and camera calibration without the spawn record or object state.
        _, record = self.perception.estimators[key].predict(rgb, geometry, 0, now)
        return record

    def coarse(self, episode, use_depth=False, *, scene_candidates=None):
        now = time.perf_counter()
        rgb, geometry = self.perception.capture_rgb("global", self.data)
        fits = self.perception.estimators["global"].fit_detections(rgb, geometry, now)
        record = {"detections": [{"detection_id": i, "color": item.color, "geometry": fit}
                                 for i, (item, fit) in enumerate(fits) if fit["status"] == "estimated"
                                 and block_in_tray(fit["center_xyz_m"], self.source_tray,
                                                   self.scene.edge)]}
        candidates = global_candidates(record, episode.target_color, {}, base_xy=self.pan)
        if scene_candidates is not None:
            for detection in record["detections"]:
                pose = Estimate.from_record(detection["geometry"])
                # All candidates share the same injected camera-frame translation.
                scene_candidates.append((detection["color"], Estimate(pose.xy + episode.offset_xy,
                                                                       pose.yaw + episode.offset_yaw, pose.fit)))
        if not candidates:
            return None, None, None
        rgb_pose = Estimate.from_record(candidates[0])
        pose = rgb_pose
        if use_depth:
            mask = fits[candidates[0]["global_detection_id"]][0].mask
            pose = self.rgbd_pose(geometry, mask, rgb_pose)
            if pose is None:
                return rgb_pose, None, None
        coarse = Estimate(pose.xy + episode.offset_xy, pose.yaw + episode.offset_yaw, pose.fit, pose.z)
        return rgb_pose, coarse, pose

    def init_depth_renderer(self):
        spec = self.perception.specs["global"]
        self.depth_renderer = mujoco.Renderer(self.model, height=spec["height"], width=spec["width"])
        self.depth_renderer.enable_depth_rendering()
        self.depth_renderer.update_scene(self.data, camera=self.perception.cam_ids["global"])
        self.depth_renderer.render()

    def rgbd_pose(self, geometry, mask, rgb_pose):
        """Deproject the rendered metric optical depths of the same RGB mask into the mount frame."""
        self.depth_renderer.update_scene(self.data, camera=self.perception.cam_ids["global"])
        depth = self.depth_renderer.render()
        px_y, px_x = np.nonzero(mask)
        z = depth[px_y, px_x]
        valid = np.isfinite(z) & (z > 0)
        if np.count_nonzero(valid) < 30:
            return None
        k = geometry.k
        z = z[valid]
        rays = np.column_stack(((px_x[valid] + 0.5 - k[0, 2]) / k[0, 0],
                                (px_y[valid] + 0.5 - k[1, 2]) / k[1, 1], np.ones_like(z)))
        pose = geometry.mount_from_cv
        points = (rays * z[:, None]) @ pose[:3, :3].T + pose[:3, 3]
        top_z = np.quantile(points[:, 2], 0.85)
        if abs(top_z - (self.floor + self.scene.edge)) > 0.010:
            return None
        top = points[np.abs(points[:, 2] - top_z) < 0.002, :2]
        if len(top) < 30:
            return None
        c, s = math.cos(rgb_pose.yaw), math.sin(rgb_pose.yaw)
        axes = np.array([[c, -s], [s, c]])
        lower, upper = np.quantile(top @ axes, (0.02, 0.98), axis=0)
        if np.any(upper - lower < self.scene.edge / 2):
            return None
        center = ((lower + upper) / 2) @ axes.T
        return Estimate(center, rgb_pose.yaw, rgb_pose.fit, float(top_z - self.scene.edge / 2))

    def approach(self, pose):
        q = self.solve(pose.xy, self.cfg["hover_z_m"], self.heading(pose))
        if q is None:
            return False
        self.motion(1.50, q=q)
        return True

    def wrist(self, anchor, color):
        candidate = {"center_xyz_m": [*anchor.xy, self.floor + self.scene.edge / 2],
                     "vertices_xyz_m": cube_vertices(anchor.xy, anchor.yaw, self.scene.edge, self.floor).tolist()}
        # Simulated sensor periods are tracked independently from render and compute wall time.
        self.motion(1 / 30)
        found = refine_with_wrist(self.image("wrist"), candidate, color,
                                  {"association_gate_m": self.cfg["association_gate_m"]})
        return Estimate.from_record(found) if found["status"] == "confirmed" else None

    def next_view(self, coarse, wrist, action):
        gain = VIEW_GAINS[action]
        refined = wrist or coarse
        pose = Estimate(coarse.xy + gain * (refined.xy - coarse.xy),
                        coarse.yaw + gain * yaw_delta(refined.yaw, coarse.yaw), refined.fit)
        q = self.solve(pose.xy, self.cfg["hover_z_m"], self.heading(pose))
        self.motion(0.55, q=q)

    def recenter_wrist(self, coarse, *, height=None):
        """Move the camera over the image-derived candidate at the original hover height."""
        current_q = self.data.qpos[self.scene.arm_qpos] - self.scene.arm_offset
        target = [*coarse.xy, self.floor + self.scene.edge / 2]
        plan = plan_wrist_view(self.chain, self.ctrl.p, self.scene_config, target,
                               self.cfg["hover_z_m"] if height is None else height, current_q)
        if plan is not None:
            self.motion(0.55, q=np.asarray(plan["q"]))
        return plan

    def grasp(self, episode, pose, action, start, *, from_hover=False):
        # Common script: align over the cube, descend, close, lift and hold.
        offset = FINAL_OFFSETS[action]
        yaw = self.heading(pose)
        axis = np.array([math.cos(yaw), math.sin(yaw)])
        cross = np.array([-axis[1], axis[0]])
        xy = pose.xy + (offset[0] - self.cfg["finger_offset_m"]) * axis + offset[1] * cross
        q = (self.data.qpos[self.scene.arm_qpos] - self.scene.arm_offset if from_hover else
             self.solve(xy, self.cfg["hover_z_m"], yaw))
        stages = []
        dz = 0.0 if pose.z is None else pose.z - (self.floor + self.scene.edge / 2)
        for z in (0.053, self.cfg["grasp_z_m"] + dz, self.cfg["lift_z_m"]):
            if q is None:
                break
            stages.append(q)
            q = self.solve(xy, z, yaw, q)
        if len(stages) != 3 or q is None:
            return {"status": "grasp_ik_failed", "success": False}, -0.55
        over, mid, low = stages
        score = Score(episode, self.cfg, start)
        if not from_hover:
            self.motion(0.55, q=over)
        self.motion(0.45, q=mid)
        self.motion(0.55, q=low)
        self.motion(0.80, jaw=self.jaw_closed)
        self.motion(1.10, q=q, score=score)
        self.motion(0.50, score=score)
        return score.result(), score.reward()


def features(coarse, observed, view):
    pose = observed or coarse
    dx, dy = (pose.xy - coarse.xy) / 0.016
    x, y = (pose.xy - [0.238, 0.188]) / 0.10
    return np.array([1.0, x, y, math.hypot(x, y), math.sin(4 * pose.yaw), math.cos(4 * pose.yaw),
                     observed is not None, pose.fit - 0.85, dx, dy, yaw_delta(pose.yaw, coarse.yaw) / 0.35, view])


def fit_action_values(x, y):
    """Ridge-regularized linear Q for every discrete action, with an unpenalized intercept."""
    ridge = np.eye(x.shape[1]) * 4
    ridge[0, 0] = 0
    return np.linalg.solve(x.T @ x + ridge, x.T @ y)


def seeds(split):
    return range(split["start"], split["start"] + split["count"])


def pose_record(estimate):
    return estimate.record() if estimate is not None else None


def train(sim, split, out):
    initial_x, final_x, returns = [], [], []
    with (out / "training.jsonl").open("w", encoding="utf-8") as log:
        for i, seed in enumerate(seeds(split)):
            episode = sim.episode(seed, i)
            sim.reset(episode)
            start = time.perf_counter()
            raw, coarse, _ = sim.coarse(episode)
            profile = {"seed": seed, "layout": episode.layout, "target_color": episode.target_color,
                       "target_index": episode.target_index, "raw": pose_record(raw),
                       "coarse": pose_record(coarse), "branches": []}
            if coarse is not None and sim.approach(coarse):
                wrist0 = sim.wrist(coarse, episode.target_color)
                profile["wrist_1"] = pose_record(wrist0)
                held = sim.snapshot()
                initial_x.append(features(coarse, wrist0, 0.0))
                for a0, gain in enumerate(VIEW_GAINS):
                    sim.restore(held)
                    sim.next_view(coarse, wrist0, a0)
                    wrist1 = sim.wrist(wrist0 or coarse, episode.target_color)
                    final = wrist1 or wrist0 or coarse
                    final_x.append(features(coarse, wrist1 or wrist0, gain))
                    before_grasp = sim.snapshot()
                    reward_row = []
                    for a1 in range(len(FINAL_OFFSETS)):
                        sim.restore(before_grasp)
                        result, reward = sim.grasp(episode, final, a1, start)
                        reward_row.append(reward)
                        profile["branches"].append({"view_action": a0, "final_action": a1, "reward": reward,
                                                    "wrist_2": pose_record(wrist1), "status": result["status"]})
                    returns.append(reward_row)
            log.write(json.dumps(profile) + "\n")
            log.flush()
            if (i + 1) % 5 == 0 or i + 1 == split["count"]:
                print(f"train {i + 1}/{split['count']} seeds, {len(returns) * len(FINAL_OFFSETS)} grasp trials", flush=True)
    x1, y1 = np.asarray(final_x), np.asarray(returns)
    q1 = fit_action_values(x1, y1)
    # Final actions terminate in grasp reward. Intermediate actions use the fitted
    # downstream value at the resulting camera observation (finite-horizon Bellman target).
    y0 = np.max(x1 @ q1, axis=1).reshape(-1, len(VIEW_GAINS))
    q0 = fit_action_values(np.asarray(initial_x), y0)
    np.savez(out / "fitted_q_policy.npz", view=q0, final=q1)
    return q0, q1


def run_trial(sim, episode, method, policy):
    sim.reset(episode)
    start = time.perf_counter()
    raw, coarse, sensor_pose = sim.coarse(episode, use_depth=method == METHODS[3])
    timeline = {"global": {"wall_s": time.perf_counter() - start, "sim_s": float(sim.data.time)}}
    result = {"raw_global": pose_record(raw), "sensor_pose": pose_record(sensor_pose), "coarse": pose_record(coarse)}
    wrist0, wrist1, action0, action1 = None, None, None, 0
    if raw is None:
        outcome = {"status": "target_not_detected_global", "success": False}
    elif coarse is None:
        outcome = {"status": "target_depth_unusable", "success": False}
    elif not sim.approach(coarse):
        outcome = {"status": "coarse_approach_ik_failed", "success": False}
    else:
        timeline["approach"] = {"wall_s": time.perf_counter() - start, "sim_s": float(sim.data.time)}
        final = coarse
        if method in (METHODS[1], METHODS[2]):
            captures = []

            def observe(anchor):
                observation = sim.wrist(anchor, episode.target_color)
                captures.append({"wall_s": time.perf_counter() - start, "sim_s": float(sim.data.time)})
                return observation

            def reposition(first):
                nonlocal action0
                action0 = 0  # Fixed full correction in method B.
                if method == METHODS[2]:
                    action0 = int(np.argmax(features(coarse, first, 0.0) @ policy[0]))
                if first is None and method == "B_wrist" and sim.cfg.get("retry_wrist_view"):
                    plan = sim.recenter_wrist(coarse)
                    result["view_retry"] = {"reachable": plan is not None, "plan": plan}
                    if plan is not None:
                        return
                sim.next_view(coarse, first, action0)

            wrist0, wrist1, final = align_from_hover(coarse, observe, reposition)
            timeline["wrist_1"], timeline["wrist_2"] = captures
            if method == METHODS[2]:
                action1 = int(np.argmax(features(coarse, wrist1 or wrist0, VIEW_GAINS[action0]) @ policy[1]))
        outcome, _ = sim.grasp(episode, final, action1, start)
        result["final_estimate"] = final.record()
    result.update(outcome, method=method, view_action=action0, final_action=action1, timeline=timeline,
                  wrist_1=pose_record(wrist0), wrist_2=pose_record(wrist1),
                  total_sim_s=float(sim.data.time), total_wall_s=time.perf_counter() - start)
    return result


def evaluate(sim, split, policy, out, methods):
    episodes = []
    with (out / "trials.jsonl").open("w", encoding="utf-8") as log:
        for i, seed in enumerate(seeds(split)):
            episode = sim.episode(seed, i)
            row = {"seed": seed, "layout": episode.layout, "target_color": episode.target_color,
                   "target_index": episode.target_index, "injected_xy_m": episode.offset_xy.tolist(),
                   "injected_yaw_deg": math.degrees(episode.offset_yaw), "trials": {}}
            # Rotate run order to distribute warm-up and cache effects across methods.
            first = i % len(methods)
            for method in methods[first:] + methods[:first]:
                row["trials"][method] = run_trial(sim, episode, method, policy)
            if len({json.dumps(row["trials"][m]["coarse"], sort_keys=True) for m in methods if m != "D_rgbd"}) > 1:
                raise RuntimeError(f"Unpaired coarse perceptions at seed {seed}")
            if len({json.dumps(t["raw_global"], sort_keys=True) for t in row["trials"].values()}) != 1:
                raise RuntimeError(f"Unpaired RGB target selection at seed {seed}")
            episodes.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            status = ", ".join(f"{m[0]}={row['trials'][m]['status']}" for m in methods)
            print(f"eval {i + 1}/{split['count']} seed {seed} {episode.target_color} {status}", flush=True)
    return episodes


def summarize(episodes, cfg, methods):
    horizon = cfg["sim_horizon_s"]
    summary = {}
    for method in methods:
        trials = [row["trials"][method] for row in episodes]
        successes = [t for t in trials if t["success"]]
        stats = {"n": len(trials), "successes": len(successes), "success_rate": len(successes) / len(trials),
                 "status_counts": dict(Counter(t["status"] for t in trials)),
                 "by_target": {c: {"n": sum(row["target_color"] == c for row in episodes),
                                   "successes": sum(row["target_color"] == c and row["trials"][method]["success"]
                                                    for row in episodes)}
                               for c in sorted({row["target_color"] for row in episodes})},
                 "view_action_counts": dict(Counter(str(t["view_action"]) for t in trials)),
                 "final_action_counts": dict(Counter(t["final_action"] for t in trials)),
                 "wrist_1_valid": sum(t["wrist_1"] is not None for t in trials),
                 "wrist_2_valid": sum(t["wrist_2"] is not None for t in trials)}
        for clock, limit, cutoffs in (("sim", horizon, (3, 4, 5, 6, horizon)),
                                      ("wall", 6.0, (0.25, 0.5, 1.0, 2.0, 6.0))):
            field = f"first_correct_lift_{clock}_s"
            times = [t[field] for t in successes]
            penalized = [min(limit, t.get(field) or limit) for t in trials]
            stats[f"{clock}_time"] = {
                "success_only_median_s": float(np.median(times)) if times else None,
                "success_only_mean_s": float(np.mean(times)) if times else None,
                "restricted_mean_time_to_success_s": float(np.mean(penalized)),
                "restriction_s": limit,
                "success_by_s": {str(s): sum(t <= s for t in times) / len(trials) for s in cutoffs},
            }
        summary[method] = stats
    paired = {f"{a[0]}_vs_{b[0]}": dict(Counter(f"{int(r['trials'][a]['success'])}{int(r['trials'][b]['success'])}"
                                               for r in episodes))
              for a, b in combinations(methods, 2)}
    return {"engine": f"MuJoCo {mujoco.__version__}", "config": cfg, "methods": summary,
            "paired_success": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("B", "all"), default="all",
                        help="Run the fixed wrist policy alone (B), or the paired four-method experiment")
    parser.add_argument("--retry-wrist-view", action="store_true",
                        help="For B, re-center the camera on the coarse estimate if the first wrist fit fails")
    parser.add_argument("--config", default="configs/benchmarks/finals_alignment.yaml")
    parser.add_argument("--out", type=Path, default=Path("outputs/finals_alignment"))
    parser.add_argument("--policy", type=Path, help="Evaluate a previously fitted policy, skipping training")
    parser.add_argument("--smoke", action="store_true", help="Use separate pilot ranges with three scenes each")
    args = parser.parse_args()
    cfg = load(args.config)
    if args.retry_wrist_view:
        if args.method != "B":
            parser.error("--retry-wrist-view requires --method B")
        cfg["retry_wrist_view"] = True
    if args.smoke:
        cfg.update(train_seeds={"start": 43000, "count": 3}, eval_seeds={"start": 44000, "count": 3})
    if set(seeds(cfg["train_seeds"])) & set(seeds(cfg["eval_seeds"])):
        parser.error("training and evaluation seeds must be disjoint")
    if any((args.out / name).exists() for name in ("summary.json", "training.jsonl", "trials.jsonl")):
        parser.error("output folder already contains a benchmark run; choose a new --out")
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    sim = Benchmark(cfg)
    try:
        methods = ("B_wrist",) if args.method == "B" else METHODS
        policy = None
        if args.method == "B" and args.policy:
            parser.error("method B uses fixed corrections and takes no --policy")
        if args.policy:
            with np.load(args.policy) as weights:
                policy = weights["view"], weights["final"]
        elif args.method == "all":
            policy = train(sim, cfg["train_seeds"], args.out)
        if "D_rgbd" in methods:
            sim.init_depth_renderer()
        episodes = evaluate(sim, cfg["eval_seeds"], policy, args.out, methods)
        summary = summarize(episodes, cfg, methods)
        if policy is not None:
            summary["policy_file"] = str((args.policy or args.out / "fitted_q_policy.npz").resolve())
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for m, stats in summary["methods"].items():
            print(f"{m}: {stats['successes']}/{stats['n']} stable correct lifts", flush=True)
    finally:
        if sim.depth_renderer is not None:
            sim.depth_renderer.close()
        sim.perception.close()


if __name__ == "__main__":
    main()
