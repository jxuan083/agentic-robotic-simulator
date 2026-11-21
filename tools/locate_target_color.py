#!/usr/bin/env python3
"""Coarse-to-fine localization of one block of a requested colour (MuJoCo): global candidates, wrist close-up.

  1. arm at home: one global capture; candidates = accepted global fits of the colour (then its look-alikes)
  2. per candidate: plan an EE pose that centres it in the wrist image (low, vertical gripper), move there through
     the controller (IK, rate limits, physics), one wrist capture; the wrist confirms the colour and gives the pose
  3. output: center / yaw / eight corners in the mount frame from the wrist fit, with provenance

The search sees only images, calibration and measured joints. Simulator block poses are read after the result is
written, to score it (and the global-only baseline: the first global candidate taken as the answer).

  MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/locate_target_color.py --episodes 10 --seed 20261121
  ... --episodes 3 --seed 7 --video            # also write episode_*/video.webm (no window needed)
  tools/mujoco_remote.sh --locate            # one episode in the browser desktop
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)

import numpy as np
import yaml

import evaluate_perception_exploration as common
from agentic_sim.config import load, v
from agentic_sim.mujoco_backend.live_perception import MjLivePerception
from agentic_sim.mujoco_backend.session import MjSession, SessionConfig
from agentic_sim.perception.exploration import choose_existing_color, score_target_confirmation
from agentic_sim.perception.localize import global_candidates, plan_wrist_view, refine_with_wrist, target_pose
from agentic_sim.random_blocks import grasp_ik_checker
from agentic_sim.teleop.keys import KeyState

LOCALIZE_CONFIG = "configs/target_localization.yaml"


@dataclass(frozen=True)
class Route:
    """Transit for common._move_to: up to transit_z, across, down; 1 cm controller steps."""
    transit_z_m: float = 0.13
    max_segment_m: float = 0.01


class VideoRecorder:
    """Offscreen WebM of one episode (VP8: plays in VS Code and browsers; this OpenCV build has no H.264): scene
    view (left) with the status line, latest global and wrist captures with their fits (right). One video frame per
    control frame, so the video plays at simulated real time."""

    def __init__(self, session, manager, path: Path, size=(960, 720), panel_w=480):
        import cv2
        import mujoco
        from agentic_sim.mujoco_backend.model import free_camera
        self.cv2, self.mujoco, self.manager = cv2, mujoco, manager
        self.w, self.h, self.pw = size[0], size[1], panel_w
        m = session.m
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), self.w)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), self.h)
        self.renderer = mujoco.Renderer(m, height=self.h, width=self.w)
        self.cam = mujoco.MjvCamera()
        c = free_camera(session.scene_cfg)
        self.cam.lookat[:] = c["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = c["distance"], c["azimuth"], c["elevation"]
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), round(1.0 / session.control_dt),
                                   (self.w + self.pw, self.h))
        if not self.out.isOpened():
            raise IOError(f"cannot open video writer for {path}")

    def _panel(self, key, height, label):
        cv2 = self.cv2
        item = self.manager.latest.get(key)
        tile = np.full((height, self.pw, 3), 40, np.uint8)
        if item is not None:
            img = item["overlay_rgb"]
            s = min(self.pw / img.shape[1], (height - 24) / img.shape[0])
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            y0, x0 = 24 + (height - 24 - img.shape[0]) // 2, (self.pw - img.shape[1]) // 2
            tile[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        cv2.putText(tile, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return tile

    def __call__(self, session, extra=""):
        cv2 = self.cv2
        self.renderer.update_scene(session.d, camera=self.cam)
        scene = self.renderer.render().copy()
        text = getattr(session, "_target_search_ui", None) or ""
        cv2.rectangle(scene, (0, self.h - 34), (self.w, self.h), (0, 0, 0), -1)
        cv2.putText(scene, f"t {session.t_sim:5.2f} s  {text}"[:110], (8, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        top = self.h * 3 // 8
        right = np.vstack([self._panel("global", top, "global camera (last capture)"),
                           self._panel("wrist", self.h - top, "wrist camera (last capture)")])
        self.out.write(cv2.cvtColor(np.hstack([scene, right]), cv2.COLOR_RGB2BGR))

    def close(self):
        self.out.release()
        self.renderer.close()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_global(manager, session, out_dir: Path):
    import cv2
    future, rgb = manager._capture("global", session.d, time.monotonic())
    overlay, record = future.result(timeout=45)
    manager.latest["global"] = {"record": record, "overlay_rgb": overlay}
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    _write_json(out_dir / "view.json", {"view_id": "global_home", "record": record})
    return record


def locate(session, manager, target_color: str, cfg: dict, episode_dir: Path, reachable, viewer=None, status=None,
           first_candidates=None):
    """The image-only search. Returns (result, global_record, candidates, attempts).

    Candidates nearest the arm first. A candidate counts only once the wrist confirmed its colour; its grasp
    reachability is then judged from that pose (reachable(center, yaw_deg)). found: a confirmed, reachable block.
    unreachable: only confirmed blocks out of reach (the first one's pose as evidence). not_found: nothing
    confirmed, including a block the global camera saw but the wrist could not view. first_candidates: candidates
    (same fields as global_candidates) tried before the global camera's own."""
    status = status or (lambda text: None)
    status(f"Target {target_color} | global look")
    global_record = _capture_global(manager, session, episode_dir / "views" / "global_home")
    base_xy = [float(x) for x in v(session.scene_cfg["robot"]["base_pos"])[:2]]
    cands = global_candidates(global_record, target_color, cfg, base_xy)
    if first_candidates:   # chosen by the caller (e.g. the declutter rules): tried first, duplicates dropped
        cands = list(first_candidates) + [c for c in cands if all(
            np.hypot(c["center_xyz_m"][0] - f["center_xyz_m"][0], c["center_xyz_m"][1] - f["center_xyz_m"][1]) > 0.02
            for f in first_candidates)]
    attempts = []
    route = Route()
    ctrl = session.ctrl
    out_of_reach = None            # first wrist-confirmed block that cannot be grasped

    def judged(found, cand, record, ci):
        result = target_pose(found, cand, record, global_record)
        yaw = result["yaw_rad_mod_pi_over_2"]
        result["graspable"] = bool(reachable(result["center_xyz_m"], math.degrees(yaw or 0.0)))
        result.update(target_color=target_color, candidates_tried=ci + 1, views=len(attempts))
        return result

    for ci, cand in enumerate(cands):
        color_only = None          # (verdict, wrist record) of a view that confirmed the colour but gave no pose
        confirmed = None
        for z in cfg["view_ee_z"]:
            view_id = f"cand{ci}_z{int(round(z * 1000))}"
            row = {"view_id": view_id, "candidate": ci, "global_color": cand["global_color"],
                   "global_center_xyz_m": cand["center_xyz_m"], "base_distance_m": cand["base_distance_m"],
                   "ee_z": z}
            attempts.append(row)
            status(f"Target {target_color} | candidate {ci + 1}/{len(cands)} ({cand['global_color']}) z {z:.2f}")
            plan = plan_wrist_view(ctrl.chain, ctrl.p, session.scene_cfg, cand["center_xyz_m"], z,
                                   ctrl.state.q_goal)
            if plan is None:
                row["outcome"] = "no_view_plan"
                continue
            row["plan"] = {k: plan[k] for k in ("ee_xyz_m", "center_offset", "camera_height_m")}
            ok, why = common._move_to(session, plan["ee_xyz_m"], route, viewer)
            if not ok:
                row["outcome"] = f"unreachable view: {why}"
                continue
            common._dwell(session, viewer, int(cfg.get("settle_frames", 30)), f"settle at {view_id}")
            payload, rgb, _ = common._capture(manager, session, view_id, plan["ee_xyz_m"],
                                              episode_dir / "views" / view_id, viewer,
                                              display_extra=f"wrist close-up {view_id}")
            verdict = refine_with_wrist(payload["record"], cand, target_color, cfg, rgb)
            row["outcome"] = verdict["status"]
            row["verdict"] = {k: v for k, v in verdict.items() if k != "vertices_xyz_m"}
            if verdict["status"] == "confirmed":
                confirmed = (verdict, payload["record"])
                break
            if verdict["status"] == "color_only" and color_only is None:
                color_only = (verdict, payload["record"])
            if verdict["status"] == "wrong_color":   # an accepted fit outweighs a colour-only look
                color_only = None
                break
        hit = confirmed or color_only      # colour_only: colour confirmed by the wrist, pose from the global fit
        if hit is None:
            continue
        result = judged(hit[0], cand, hit[1], ci)
        attempts[-1]["graspable"] = result["graspable"]
        if result["graspable"]:
            return result, global_record, cands, attempts
        if out_of_reach is None:
            out_of_reach = result
    if out_of_reach is not None:
        out_of_reach.update(status="unreachable", views=len(attempts), candidates_tried=len(cands),
                            reason="wrist confirmed the colour, but no confirmed block of it is within the arm's "
                                   "grasp reach")
        return out_of_reach, global_record, cands, attempts
    if not cands:
        reason = "no global candidate of this colour"
    elif all(a["outcome"] in ("no_view_plan",) or a["outcome"].startswith("unreachable view") for a in attempts):
        reason = f"{len(cands)} global candidate(s), but the wrist camera could not be brought to view any of them"
    else:
        reason = f"{len(cands)} candidate(s), none confirmed by the wrist camera"
    return ({"status": "not_found", "target_color": target_color, "reason": reason,
             "candidates_tried": len(cands), "views": len(attempts)}, global_record, cands, attempts)


def run_episode(index, layout_seed, target_seed, random_path, output, scene_path, cfg, reachable, gui=False,
                video=False):
    viewer = None
    keys = KeyState(actions={"P": "pause", "R": "reset"})
    if gui:
        from agentic_sim.mujoco_backend.viewer import Viewer
        viewer = Viewer(keys, "SO-101 Target Localization", (1500, 940), print, visible=True)
    session = MjSession(SessionConfig(scene=scene_path, random_blocks=True, random_seed=layout_seed,
                                      random_episode=index, random_config=str(random_path)),
                        log=lambda message: print(f"[episode {index}] {message}", flush=True),
                        keys=keys, focused=(lambda: True) if not gui else viewer.focused, viewer=viewer)
    if viewer:
        viewer.attach(session, reset_camera=True)
    manager = MjLivePerception(session.scene_cfg, session.m, hz=2, log=lambda _: None, camera="both")
    session.live_perception = manager
    manager.last_capture["global"] = manager.last_capture["wrist"] = float("inf")   # captures only on request
    episode_dir = output / f"episode_{index:03d}"
    target_color = choose_existing_color(session.block_color_names, target_seed, index)   # colour names only
    recorder = None
    if video:
        recorder = session._frame_hook = VideoRecorder(session, manager, episode_dir / "video.webm")

    def status(text):
        session._target_search_ui = text

    try:
        for _ in range(120):
            common._render_tick(session, viewer, "settling")
        reference = np.asarray(session.block_positions(), float)
        session._exploration_motion_audit = {"reference_xyz_m": reference,
            "robot_bodies": common._robot_bodies(session.m), "sampled_frames": 0, "max_block_drift_mm": 0.0,
            "max_block_linear_speed_m_s": 0.0, "max_block_contact_count": 0,
            "max_robot_block_contact_count": 0, "robot_block_contact_frames": 0}
        t0 = session.t_sim
        result, global_record, cands, attempts = locate(session, manager, target_color, cfg, episode_dir,
                                                       reachable, viewer, status)
        result["sim_time_s"] = round(session.t_sim - t0, 3)
        _write_json(episode_dir / "target.json", result)     # frozen before any ground truth is read
        if result["status"] in ("found", "unreachable"):
            c = result["center_xyz_m"]
            status(f"Target {target_color} {result['status'].upper()} | center [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}] m"
                   f" | yaw {result['yaw_deg_mod_90']:.1f} deg")
        else:
            status(f"Target {target_color} NOT FOUND | {result['reason']}")
        print(f"[episode {index}] {target_color}: {result['status']} "
              f"({result['views']} wrist views) -> {episode_dir / 'target.json'}", flush=True)
        if viewer is not None or recorder is not None:
            common._dwell(session, viewer, 120, "done")    # hold the result on screen / in the video
        # ---- scoring only from here on ----
        truth, colors = common._pose_truth(session)
        found = result if result["status"] in ("found", "unreachable") else None
        score = score_target_confirmation(found, target_color, truth, colors)
        # expected outcome: found if any block of the colour is graspable (judged on its true pose), else unreachable
        target_reach = [bool(reachable(t["center_xyz_m"], math.degrees(t["yaw_rad"])))
                        for t, c in zip(truth, colors) if c == target_color]
        expected = "found" if any(target_reach) else "unreachable"
        matched = score["matched_gt_index"]
        matched_reachable = None if matched is None else bool(reachable(truth[matched]["center_xyz_m"],
                                                                        math.degrees(truth[matched]["yaw_rad"])))
        outcome_correct = (result["status"] == expected and score["correct_target_confirmation"]
                           and (expected != "found" or matched_reachable))
        global_of_found = None if found is None else score_target_confirmation(
            {"center_xyz_m": found["source"]["global_center_xyz_m"],
             "vertices_xyz_m": found["source"]["global_vertices_xyz_m"],
             "yaw_rad_mod_pi_over_2": found["source"]["global_yaw_rad_mod_pi_over_2"]}, target_color, truth, colors)
        global_only = None if not cands else score_target_confirmation(cands[0], target_color, truth, colors)
        drift = np.linalg.norm(np.asarray(session.block_positions(), float) - reference, axis=1)
        audit = session._exploration_motion_audit
        episode = {"episode": index, "layout_seed": layout_seed, "target_seed": target_seed,
                   "target_color": target_color, "result": result, "candidates": cands, "attempts": attempts,
                   "score": score, "expected_status": expected, "target_blocks_reachable": target_reach,
                   "matched_block_reachable": matched_reachable, "outcome_correct": bool(outcome_correct),
                   "score_global_pose_of_found_block": global_of_found,
                   "score_global_only_first_candidate": global_only,
                   "global_detections": [{"color": d.get("color"), "status": d["geometry"].get("status"),
                                          "reason": d["geometry"].get("reason")}
                                         for d in global_record.get("detections", [])],
                   "block_motion_audit": {"max_block_drift_mm": float(drift.max(initial=0) * 1000),
                                          "robot_block_contact_frames": audit["robot_block_contact_frames"]},
                   "ground_truth_after_search": truth, "ground_truth_colors": colors}
        _write_json(episode_dir / "evaluation.json", episode)
        return episode
    finally:
        if recorder is not None:
            session._frame_hook = None
            recorder.close()
            print(f"[episode {index}] video: {recorder.path}", flush=True)
        session.live_perception = None
        manager.close()
        session.finish()
        if viewer:
            viewer.close()


def _stats(values):
    values = [float(x) for x in values if x is not None]
    if not values:
        return None
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "max": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="layout seed (episode i = Reset i)")
    parser.add_argument("--target-seed", type=int, default=None, help="target-colour seed (default: --seed)")
    parser.add_argument("--output", default=None, help="default outputs/target-localization/<time>")
    parser.add_argument("--scene", default="configs/scenes/lab_20261120.yaml")
    parser.add_argument("--config", default=LOCALIZE_CONFIG)
    parser.add_argument("--blocks", type=int, default=8, help="blocks per episode")
    parser.add_argument("--gui", action="store_true", help="show one episode in a MuJoCo window")
    parser.add_argument("--anywhere", action="store_true",
                        help="blocks anywhere in the storage tray, also out of the arm's reach (default: reachable only)")
    parser.add_argument("--video", action="store_true",
                        help="write episode_*/video.webm (offscreen, simulated real time; no window needed)")
    args = parser.parse_args()
    if args.episodes <= 0 or (args.gui and args.episodes != 1):
        parser.error("episodes must be positive; --gui runs exactly one episode")
    output = Path(args.output or f"outputs/target-localization/{time.strftime('%Y%m%d-%H%M%S')}")
    if (output / "evaluation.json").exists():
        parser.error(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    cfg = load(args.config)
    random_cfg = load("configs/random_blocks.yaml")
    random_cfg["count_range"] = [args.blocks, args.blocks]
    random_cfg["check_ik"] = not args.anywhere
    if args.anywhere:   # the whole storage tray, out-of-reach corner included
        random_cfg["max_radius_m"] = 1.0
        random_cfg["workspace_xy"] = [[0.0, 1.0], [-1.0, 1.0]]
    reachable = grasp_ik_checker(load(args.scene), "configs/teleop.yaml")
    random_path = output / "random_blocks.yaml"
    random_path.write_text(yaml.safe_dump(random_cfg, sort_keys=False), encoding="utf-8")
    target_seed = args.target_seed if args.target_seed is not None else (
        args.seed if args.seed is not None else secrets.randbits(32))
    episodes = []
    try:
        for i in range(args.episodes):
            row = run_episode(i, args.seed, target_seed, random_path, output, args.scene, cfg, reachable,
                              args.gui, args.video)
            episodes.append(row)
            s = row["score"]
            print(f"episode {i + 1}/{args.episodes}: {row['target_color']} {row['result']['status']} "
                  f"(expected {row['expected_status']}) correct={row['outcome_correct']} "
                  f"center_mm={s['center_error_mm']}", flush=True)
    except common.ResetCancelled as exc:
        print(f"cancelled: {exc}", flush=True)
        return
    found = [e for e in episodes if e["result"]["status"] in ("found", "unreachable")]   # with a pose
    g_only = [e["score_global_only_first_candidate"] for e in episodes]
    outcomes = {}
    for e in episodes:
        key = f"expected {e['expected_status']} -> {e['result']['status']}"
        outcomes[key] = outcomes.get(key, 0) + 1
    summary = {
        "episodes_n": len(episodes), "layout_seed": args.seed, "target_seed": target_seed,
        "blocks_per_episode": args.blocks, "anywhere": args.anywhere,
        "outcomes": outcomes,
        "outcome_correct_n": sum(e["outcome_correct"] for e in episodes),
        "false_unreachable_n": sum(e["result"]["status"] == "unreachable" and e["expected_status"] == "found"
                                   for e in episodes),
        "found_n": sum(e["result"]["status"] == "found" for e in episodes),
        "unreachable_n": sum(e["result"]["status"] == "unreachable" for e in episodes),
        "correct_n": sum(e["score"]["correct_target_confirmation"] for e in episodes),
        "wrong_n": sum(not e["score"]["correct_target_confirmation"] for e in found),
        "not_found_n": len(episodes) - len(found),
        "pose_from_wrist_n": sum(e["result"]["source"]["pose"] == "wrist" for e in found),
        "pose_from_global_n": sum(e["result"]["source"]["pose"] == "global" for e in found),
        "center_error_mm": _stats(e["score"]["center_error_mm"] for e in found),
        "corner_rmse_mm": _stats(e["score"]["corner_rmse_mm"] for e in found),
        "yaw_error_deg_mod90": _stats(e["score"]["yaw_error_deg_mod90"] for e in found),
        "global_pose_of_found_center_error_mm": _stats(e["score_global_pose_of_found_block"]["center_error_mm"]
                                                       for e in found),
        "global_only_first_candidate_correct_n": sum(bool(s and s["correct_target_confirmation"]) for s in g_only),
        "global_only_first_candidate_wrong_color_n": sum(bool(s and s["matched_gt_color"] is not None
                                                              and not s["color_correct"]) for s in g_only),
        "wrist_views": _stats(e["result"]["views"] for e in episodes),
        "sim_time_s": _stats(e["result"]["sim_time_s"] for e in episodes),
        "max_block_drift_mm": max(e["block_motion_audit"]["max_block_drift_mm"] for e in episodes),
        "robot_block_contact_frames": sum(e["block_motion_audit"]["robot_block_contact_frames"] for e in episodes),
        "episodes": [{"episode": e["episode"], "target_color": e["target_color"], "status": e["result"]["status"],
                      "expected": e["expected_status"], "outcome_correct": e["outcome_correct"],
                      "views": e["result"]["views"], "correct": e["score"]["correct_target_confirmation"],
                      "pose_source": (e["result"].get("source") or {}).get("pose"),
                      "center_error_mm": e["score"]["center_error_mm"],
                      "global_only_correct": bool(e["score_global_only_first_candidate"]
                                                  and e["score_global_only_first_candidate"]["correct_target_confirmation"])}
                     for e in episodes]}
    _write_json(output / "evaluation.json", {"summary": summary, "config": cfg})
    with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary["episodes"][0]))
        w.writeheader()
        w.writerows(summary["episodes"])
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
