#!/usr/bin/env python3
"""Coarse-to-fine localization of one block of a requested colour (MuJoCo): global candidates, wrist close-up.

  1. arm at home: one global capture; candidates = accepted global fits of the colour (then its look-alikes)
  2. per candidate: plan an EE pose that centres it in the wrist image (low, vertical gripper), move there through
     the controller (IK, rate limits, physics), one wrist capture; the wrist confirms the colour and gives the pose
  3. output: center / yaw / eight corners in the mount frame from the wrist fit, with provenance

The search sees only images, calibration and measured joints. Simulator block poses are read after the result is
written, to score it (and the global-only baseline: the first global candidate taken as the answer).

  MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/locate_target_color.py --episodes 10 --seed 20261121
  tools/mujoco_remote.sh --locate            # one episode in the browser desktop
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
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
from agentic_sim.config import load
from agentic_sim.mujoco_backend.live_perception import MjLivePerception
from agentic_sim.mujoco_backend.session import MjSession, SessionConfig
from agentic_sim.perception.exploration import choose_existing_color, score_target_confirmation
from agentic_sim.perception.localize import global_candidates, plan_wrist_view, refine_with_wrist, target_pose
from agentic_sim.teleop.keys import KeyState

LOCALIZE_CONFIG = "configs/target_localization.yaml"


@dataclass(frozen=True)
class Route:
    """Transit for common._move_to: up to transit_z, across, down; 1 cm controller steps."""
    transit_z_m: float = 0.13
    max_segment_m: float = 0.01


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


def locate(session, manager, target_color: str, cfg: dict, episode_dir: Path, viewer=None, status=None):
    """The image-only search. Returns (result, global_record, candidates, attempts)."""
    status = status or (lambda text: None)
    status(f"Target {target_color} | global look")
    global_record = _capture_global(manager, session, episode_dir / "views" / "global_home")
    cands = global_candidates(global_record, target_color, cfg)
    attempts = []
    route = Route()
    ctrl = session.ctrl
    for ci, cand in enumerate(cands):
        color_only = None          # (verdict, wrist record) of a view that confirmed the colour but gave no pose
        for z in cfg["view_ee_z"]:
            view_id = f"cand{ci}_z{int(round(z * 1000))}"
            row = {"view_id": view_id, "candidate": ci, "global_color": cand["global_color"],
                   "global_center_xyz_m": cand["center_xyz_m"], "ee_z": z}
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
                row["outcome"] = f"unreachable: {why}"
                continue
            common._dwell(session, viewer, int(cfg.get("settle_frames", 30)), f"settle at {view_id}")
            payload, _, _ = common._capture(manager, session, view_id, plan["ee_xyz_m"],
                                            episode_dir / "views" / view_id, viewer,
                                            display_extra=f"wrist close-up {view_id}")
            verdict = refine_with_wrist(payload["record"], cand, target_color, cfg)
            row["outcome"] = verdict["status"]
            row["verdict"] = {k: v for k, v in verdict.items() if k != "vertices_xyz_m"}
            if verdict["status"] == "confirmed":
                result = target_pose(verdict, cand, payload["record"], global_record)
                result.update(target_color=target_color, candidates_tried=ci + 1, views=len(attempts))
                return result, global_record, cands, attempts
            if verdict["status"] == "color_only" and color_only is None:
                color_only = (verdict, payload["record"])
            if verdict["status"] == "wrong_color":   # an accepted fit outweighs a colour-only look
                color_only = None
                break
        if color_only is not None:     # colour confirmed by the wrist, pose from the global fit
            result = target_pose(color_only[0], cand, color_only[1], global_record)
            result.update(target_color=target_color, candidates_tried=ci + 1, views=len(attempts))
            return result, global_record, cands, attempts
    reason = "no global candidate of this colour" if not cands else \
        f"{len(cands)} candidate(s), none confirmed by the wrist camera"
    return ({"status": "not_found", "target_color": target_color, "reason": reason,
             "candidates_tried": len(cands), "views": len(attempts)}, global_record, cands, attempts)


def run_episode(index, layout_seed, target_seed, random_path, output, scene_path, cfg, gui=False):
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
                                                       viewer, status)
        result["sim_time_s"] = round(session.t_sim - t0, 3)
        _write_json(episode_dir / "target.json", result)     # frozen before any ground truth is read
        if result["status"] == "found":
            c = result["center_xyz_m"]
            status(f"Target {target_color} FOUND | center [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}] m | "
                   f"yaw {result['yaw_deg_mod_90']:.1f} deg")
        else:
            status(f"Target {target_color} NOT FOUND | {result['reason']}")
        print(f"[episode {index}] {target_color}: {result['status']} "
              f"({result['views']} wrist views) -> {episode_dir / 'target.json'}", flush=True)
        if viewer is not None:
            common._dwell(session, viewer, 120, "done")
        # ---- scoring only from here on ----
        truth, colors = common._pose_truth(session)
        found = result if result["status"] == "found" else None
        score = score_target_confirmation(found, target_color, truth, colors)
        global_of_found = None if found is None else score_target_confirmation(
            {"center_xyz_m": found["source"]["global_center_xyz_m"],
             "vertices_xyz_m": found["source"]["global_vertices_xyz_m"],
             "yaw_rad_mod_pi_over_2": found["source"]["global_yaw_rad_mod_pi_over_2"]}, target_color, truth, colors)
        global_only = None if not cands else score_target_confirmation(cands[0], target_color, truth, colors)
        drift = np.linalg.norm(np.asarray(session.block_positions(), float) - reference, axis=1)
        audit = session._exploration_motion_audit
        episode = {"episode": index, "layout_seed": layout_seed, "target_seed": target_seed,
                   "target_color": target_color, "result": result, "candidates": cands, "attempts": attempts,
                   "score": score, "score_global_pose_of_found_block": global_of_found,
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
    random_cfg["check_ik"] = True
    random_path = output / "random_blocks.yaml"
    random_path.write_text(yaml.safe_dump(random_cfg, sort_keys=False), encoding="utf-8")
    target_seed = args.target_seed if args.target_seed is not None else (
        args.seed if args.seed is not None else secrets.randbits(32))
    episodes = []
    try:
        for i in range(args.episodes):
            row = run_episode(i, args.seed, target_seed, random_path, output, args.scene, cfg, args.gui)
            episodes.append(row)
            s = row["score"]
            print(f"episode {i + 1}/{args.episodes}: {row['target_color']} {row['result']['status']} "
                  f"correct={s['correct_target_confirmation']} center_mm={s['center_error_mm']}", flush=True)
    except common.ResetCancelled as exc:
        print(f"cancelled: {exc}", flush=True)
        return
    found = [e for e in episodes if e["result"]["status"] == "found"]
    g_only = [e["score_global_only_first_candidate"] for e in episodes]
    summary = {
        "episodes_n": len(episodes), "layout_seed": args.seed, "target_seed": target_seed,
        "blocks_per_episode": args.blocks,
        "found_n": len(found),
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
