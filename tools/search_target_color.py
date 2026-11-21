#!/usr/bin/env python3
"""Bounded image-only search for a randomly selected existing block color."""
from __future__ import annotations

import argparse
import csv
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
from agentic_sim.mujoco_backend.session import MjSession, SessionConfig
from agentic_sim.mujoco_backend.live_perception import MjLivePerception
from agentic_sim.perception.exploration import (TargetColorSearch, choose_existing_color,
    plan_target_search_views, plan_wrist_sweep, score_target_confirmation)
from agentic_sim.teleop.keys import KeyState


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _score_after_search(confirmed, target_color, session):
    """Read settled GT only after the image-only decision has been finalized."""
    truth, colors = common._pose_truth(session)
    return score_target_confirmation(confirmed, target_color, truth, colors), truth, colors


def run_episode(index, layout_seed, target_seed, random_path, output, scene_path, gui=False):
    viewer = None
    keys = KeyState(actions={"P": "pause", "R": "reset"})
    if gui:
        from agentic_sim.mujoco_backend.viewer import Viewer
        viewer = Viewer(keys, "SO-101 Target Color Search", (1500, 940), print, visible=True)
    session = MjSession(SessionConfig(scene=scene_path, random_blocks=True, random_seed=layout_seed,
                                      random_episode=index, random_config=str(random_path)),
                        log=lambda message: print(f"[episode {index}] {message}", flush=True),
                        keys=keys, focused=(lambda: True) if not gui else viewer.focused, viewer=viewer)
    if viewer:
        viewer.attach(session, reset_camera=True)
    manager = MjLivePerception(session.scene_cfg, session.m, hz=2, log=lambda _: None, camera="wrist")
    session.live_perception = manager
    manager.next_at = time.monotonic() + 1e9
    manager.last_capture["wrist"] = float("inf")
    sweep = plan_wrist_sweep(session.scene_cfg)
    views = plan_target_search_views(session.scene_cfg)
    episode_dir = output / f"episode_{index:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    if gui:
        # Render at the initial position while the scene settles.
        pass
    colors = list(session.block_color_names)  # The sampler receives color names only.
    target_color = choose_existing_color(colors, target_seed, index)
    search = TargetColorSearch(target_color)
    session._target_search_ui = f"Target color: {target_color} | searching (0/2 views)"
    latest_path = episode_dir / "latest-target.json"
    planned = [{"view_id": "initial_baseline", "position_xyz_m": list(common.BASELINE_POSE),
                "phase": "initial"}, *[{"view_id": name, "position_xyz_m": list(pos),
                "phase": "primary" if name.startswith("sweep") else "expanded"} for name, pos in views]]
    visited, skipped = [], []

    def save_latest(status=None, reason=None):
        current = search.result()
        _write_json(latest_path, {"episode": index, "camera": "wrist", "target_color": search.target_color,
            "status": status or current["status"], "reason": reason or current["reason"],
            "visited_view_ids": list(visited), "skipped_views": list(skipped),
            "search_result": current, "confirmed_result": current["confirmed"]})

    save_latest()

    def accept_view(view_id, position, phase):
        session._target_search_ui = f"Target color: {target_color} | searching ({search.result()['observation_n']} observations)"
        output_dir = episode_dir / "views" / view_id
        payload, _, _ = common._capture(manager, session, view_id, position, output_dir, viewer,
            display_extra=f"{phase} view {view_id}; target {target_color}")
        state = search.observe(payload["record"])
        visited.append({"view_id": view_id, "phase": phase, "target_xyz_m": list(position),
                        "measured_ee_xyz_m": payload["record"]["ee_measured_xyz_m"],
                        "captured_at_monotonic_s": payload["record"]["captured_at_monotonic_s"],
                        "matching_color_n": sum(r.get("color") == target_color for r in payload["record"]["detections"]),
                        "accepted_geometry_n": sum(r.get("color") == target_color and
                            r.get("geometry", {}).get("status") == "estimated" for r in payload["record"]["detections"])})
        current = state["confirmed"]
        session._target_search_ui = (f"Target color: {target_color} | CONFIRMED" if current else
            f"Target color: {target_color} | searching ({state['observation_n']} observations)")
        save_latest()
        if current:
            chosen = current["selected_observation"]
            xyz = current["center_xyz_m"]
            print(f"target confirmed: color={target_color} estimated_center_xyz_m="
                  f"[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] "
                  f"source_frame={chosen['frame_number']} full_result={latest_path}", flush=True)
            session._target_search_ui = (f"Target {target_color} CONFIRMED | "
                f"center xyz [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}] m | "
                f"frame {chosen['frame_number']} | 8 corners in latest-target.json")
        return current

    robot_bodies = common._robot_bodies(session.m)
    try:
        for _ in range(120):
            common._render_tick(session, viewer, "Preparing target-color search")
        motion_reference = np.asarray(session.block_positions(), float)
        session._exploration_motion_audit = {"reference_xyz_m": motion_reference,
            "robot_bodies": robot_bodies, "sampled_frames": 0, "max_block_drift_mm": 0.0,
            "max_block_linear_speed_m_s": 0.0, "max_block_contact_count": 0,
            "max_robot_block_contact_count": 0, "robot_block_contact_frames": 0}
        ok, why = common._move_to(session, common.BASELINE_POSE, sweep, viewer)
        if not ok:
            skipped.append({"view_id": "initial_baseline", "reason": why})
        else:
            common._dwell(session, viewer, 45, "Settling at initial target-search view")
            if accept_view("initial_baseline", common.BASELINE_POSE, "initial"):
                pass
        for view_id, position in views:
            if search.result()["status"] == "confirmed":
                break
            phase = "primary" if view_id.startswith("sweep") else "expanded"
            session._target_search_ui = f"Target color: {target_color} | searching {view_id}"
            ok, why = common._move_to(session, position, sweep, viewer)
            if not ok:
                skipped.append({"view_id": view_id, "position_xyz_m": list(position), "reason": why})
                save_latest()
                continue
            common._dwell(session, viewer, 30, f"Settling at target-search view {view_id}")
            if accept_view(view_id, position, phase):
                break
        if search.result()["status"] != "confirmed":
            reason = (f"unreachable: no planned observation pose was reached ({len(skipped)} skipped)"
                      if not visited else f"bounded search exhausted after {len(visited)} views; {len(skipped)} unreachable")
            search.exhaust(reason)
            session._target_search_ui = f"Target color: {target_color} | {search.result()['status'].upper()}"
            save_latest()
        # Small settle period before post-run-only GT scoring.
        for _ in range(30):
            common._render_tick(session, viewer, "Finalizing target-color search")
        score, gt, gt_colors = _score_after_search(search.result()["confirmed"], target_color, session)
        if gui and search.result()["confirmed"] is not None:
            deadline = time.monotonic() + 2.0
            import glfw
            while time.monotonic() < deadline and not viewer.should_close():
                glfw.poll_events()
                viewer.render(session, extra=session._target_search_ui)
                time.sleep(0.05)
        motion = np.asarray(session.block_positions(), float)
        drift = np.linalg.norm(motion - motion_reference, axis=1)
        episode = {"episode": index, "layout_seed": layout_seed, "target_seed": target_seed,
            "target_color": target_color, "target_color_source": "random choice from current scene block color-name list",
            "search": search.result(), "search_score_post_run_only": score,
            "views_planned": planned, "views_visited": visited, "views_skipped": skipped,
            "view_count": len(visited), "status": search.result()["status"],
            "block_motion_audit": {"max_block_drift_mm": float(drift.max(initial=0)*1000),
                "robot_block_contact_frames": session._exploration_motion_audit["robot_block_contact_frames"],
                "all_motion_max_robot_block_contact_count": session._exploration_motion_audit["max_robot_block_contact_count"],
                "sampled_frames": session._exploration_motion_audit["sampled_frames"]},
            "ground_truth_after_search": gt, "ground_truth_colors_after_search": gt_colors,
            "limitations": "bounded RGB-only search; target existence does not guarantee visibility; no collision-free claim"}
        _write_json(episode_dir / "evaluation.json", episode)
        save_latest(status=search.result()["status"])
        return episode
    except common.ResetCancelled as exc:
        search.reset()
        _write_json(latest_path, {"episode": index, "status": "cancelled", "reason": str(exc),
            "target_color": None, "search_result": search.result(), "confirmed_result": None,
            "visited_view_ids": [], "map_cleared": True})
        _write_json(episode_dir / "accumulated.json", {"records": [], "status": "cancelled"})
        raise
    finally:
        if session.live_perception:
            session.live_perception = None
        manager.close()
        session.finish()
        if viewer:
            viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="optional reproducible layout seed")
    parser.add_argument("--target-seed", type=int, default=None, help="independent target-color selection seed")
    parser.add_argument("--output", default="outputs/perception-target-search/mujoco-final")
    parser.add_argument("--scene", default="configs/scenes/lab_20261120.yaml")
    parser.add_argument("--gui", action="store_true", help="show one interactive target-color search")
    args = parser.parse_args()
    if args.episodes <= 0 or (args.gui and args.episodes != 1):
        parser.error("episodes must be positive; --gui supports exactly one episode")
    output = Path(args.output)
    if (output / "evaluation.json").exists():
        parser.error(f"refusing to overwrite existing target-search run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = load("configs/random_blocks.yaml")
    config["count_range"] = [8, 8]
    config["check_ik"] = True
    random_path = output / "target_search_random_blocks.yaml"
    random_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    target_seed = args.target_seed if args.target_seed is not None else (
        args.seed if args.seed is not None else secrets.randbits(32))
    episodes = []
    try:
        for i in range(args.episodes):
            row = run_episode(i, args.seed, target_seed, random_path, output, args.scene, args.gui)
            episodes.append(row)
            score = row["search_score_post_run_only"]
            print(f"episode {i+1}/{args.episodes}: target={row['target_color']} status={row['status']} "
                  f"views={row['view_count']} correct={score['correct_target_confirmation']} "
                  f"center_mm={score['center_error_mm']}", flush=True)
    except common.ResetCancelled as exc:
        print(f"target search cancelled: {exc}; current target and map cleared", flush=True)
        return
    counts = {}
    for row in episodes:
        counts[row["target_color"]] = counts.get(row["target_color"], 0) + 1
    summary = {"episodes_n": len(episodes), "layout_seed": args.seed,
        "target_selection_seed": target_seed, "target_color_counts": counts,
        "target_color_existed_n": sum(x["search_score_post_run_only"]["target_color_exists"] for x in episodes),
        "status_counts": {name: sum(x["status"] == name for x in episodes)
                          for name in ("confirmed", "exhausted", "unreachable", "cancelled")},
        "confirmed_n": sum(x["search"]["confirmed"] is not None for x in episodes),
        "correct_target_confirmation_n": sum(x["search_score_post_run_only"]["correct_target_confirmation"] for x in episodes),
        "false_confirmation_n": sum(x["search"]["confirmed"] is not None and
            not x["search_score_post_run_only"]["correct_target_confirmation"] for x in episodes),
        "total_views": sum(x["view_count"] for x in episodes),
        "skipped_view_n": sum(len(x["views_skipped"]) for x in episodes),
        "mean_views_to_terminal": float(np.mean([x["view_count"] for x in episodes])) if episodes else None,
        "episodes": [{"episode": x["episode"], "target_color": x["target_color"], "status": x["status"],
            "views": x["view_count"], "correct_target_confirmation": x["search_score_post_run_only"]["correct_target_confirmation"],
            "center_error_mm": x["search_score_post_run_only"]["center_error_mm"],
            "corner_rmse_mm": x["search_score_post_run_only"]["corner_rmse_mm"]} for x in episodes]}
    _write_json(output / "evaluation.json", {"summary": summary, "episodes": episodes})
    with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "target_color", "status", "view_count",
            "correct_target_confirmation", "matched_gt_color", "center_error_mm", "corner_rmse_mm",
            "yaw_error_deg_mod90"])
        writer.writeheader()
        for row in episodes:
            score = row["search_score_post_run_only"]
            writer.writerow({"episode": row["episode"], "target_color": row["target_color"],
                "status": row["status"], "view_count": row["view_count"],
                "correct_target_confirmation": score["correct_target_confirmation"],
                "matched_gt_color": score["matched_gt_color"], "center_error_mm": score["center_error_mm"],
                "corner_rmse_mm": score["corner_rmse_mm"], "yaw_error_deg_mod90": score["yaw_error_deg_mod90"]})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
