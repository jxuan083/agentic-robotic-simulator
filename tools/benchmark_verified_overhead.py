"""Evaluate wrist re-localization and the measured pre-descent hover in MuJoCo."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import math

import cv2
import numpy as np

from benchmark_finals_alignment import Benchmark, Estimate, block_in_tray, global_candidates, load, pose_record, seeds

from agentic_sim.perception.vertices import CameraGeometry, cube_vertices


def at_hover(sim, estimate):
    """Control uses image estimates and measured joints exclusively."""
    height = sim.cfg["hover_z_m"]
    q = sim.solve(estimate.xy, height, sim.heading(estimate))
    if q is None:
        return {"passed": False, "status": "hover_ik_failed"}
    sim.motion(0.55, q=q)
    sim.motion(0.15)
    measured = sim.data.qpos[sim.scene.arm_qpos] - sim.scene.arm_offset
    ee = sim.chain.ee_pose_mount(measured)[:3, 3]
    speed = float(np.max(np.abs(sim.data.qvel[sim.scene.arm_qvel])))
    xy_mm = float(np.linalg.norm(ee[:2] - estimate.xy) * 1000)
    z_mm = abs(float(ee[2]) - height) * 1000
    passed = xy_mm <= 3 and z_mm <= 3 and speed <= sim.cfg["settle_qd"]
    return {"passed": passed, "status": "ready" if passed else "hover_not_settled",
            "ee_m": ee.tolist(), "estimated_target_xy_m": estimate.xy.tolist(),
            "xy_error_mm": xy_mm, "z_error_mm": z_mm, "joint_speed_rad_s": speed}


def search_source(sim, color):
    """Look across the known tray when the global image has no reliable candidate."""
    tray = sim.source_tray
    c, s = np.cos(tray["yaw"]), np.sin(tray["yaw"])
    rotation = np.array([[c, -s], [s, c]])
    seen, views = [], []
    # Two red cubes are in the benchmark order. Require both before ranking nearest.
    expected = sim.colors.count(color)
    for u, v in ((-0.5, 0), (0.4, 0), (-0.5, -0.55), (-0.5, 0.55)):
        xy = tray["center"] + rotation @ (np.array([u, v]) * tray["inner_half"])
        if sim.recenter_wrist(Estimate(xy, 0, 0), height=sim.cfg["hover_z_m"] + 0.025) is None:
            continue
        record = sim.image("wrist")
        views.append(xy.tolist())
        for found in global_candidates(record, color, {}, base_xy=sim.pan):
            if not block_in_tray(found["center_xyz_m"], tray, sim.scene.edge):
                continue
            estimate = Estimate.from_record(found)
            if all(np.linalg.norm(estimate.xy - other.xy) > sim.scene.edge / 2 for other in seen):
                seen.append(estimate)
        if len(seen) == expected:
            return sorted(seen, key=lambda item: np.linalg.norm(item.xy - sim.pan)), views
    return [], views


def separated_views(sim, target, neighbors):
    """Rank reachable wrist viewpoints by projected separation and distance to frame edges."""
    if not neighbors:
        return []
    edge, floor = sim.scene.edge, sim.floor
    vertices = [cube_vertices(p.xy, p.yaw, edge, floor) for p in [target, *neighbors]]
    current = sim.data.qpos[sim.scene.arm_qpos] - sim.scene.arm_offset
    height = sim.cfg["hover_z_m"] + 0.025
    choices = []
    for angle in np.arange(8) * math.pi / 4:
        xy = target.xy + 0.035 * np.array([math.cos(angle), math.sin(angle)])
        for roll in (0, math.pi / 2, -math.pi / 2):
            q = sim.solve(xy, height, sim.heading(target) + roll)
            if q is None:
                continue
            camera = CameraGeometry.from_scene(sim.scene_config, "wrist", joints_rad=q)
            try:
                points = [camera.project(v, distorted=False).astype(np.float32) for v in vertices]
            except ValueError:
                continue
            uv = points[0]
            margin = min(uv[:, 0].min(), uv[:, 1].min(), camera.width - uv[:, 0].max(),
                         camera.height - uv[:, 1].max())
            if margin < 5:
                continue
            hull = cv2.convexHull(uv)
            overlap = max(cv2.intersectConvexConvex(hull, cv2.convexHull(other))[0] for other in points[1:])
            gap = min(np.linalg.norm(other.mean(axis=0) - uv.mean(axis=0)) for other in points[1:])
            score = (overlap / cv2.contourArea(hull), -min(margin, 45), -gap, np.linalg.norm(q-current))
            choices.append((score, q, xy))
    choices.sort(key=lambda x: x[0])
    selected = []
    for _, q, xy in choices:
        if all(np.linalg.norm(q - other[0]) > 0.35 for other in selected):
            selected.append((q, xy))
        if len(selected) == 2:
            break
    return selected


def trial(sim, episode):
    sim.reset(episode)
    start = time.perf_counter()
    result = {"views": [], "hovers": [], "retry": None, "success": False}

    def observe(anchor):
        seen = sim.wrist(anchor, episode.target_color)
        if seen is not None and same_neighbors:
            distance = np.linalg.norm(seen.xy - locked.xy)
            other = min(np.linalg.norm(seen.xy - n.xy) for n in same_neighbors)
            # After a view change, reject a fit that could belong to the adjacent cube.
            if distance + 0.003 >= other or distance > sim.cfg["association_gate_m"]:
                result["ambiguous_views"] += 1
                seen = None
        result["views"].append(pose_record(seen))
        return seen

    def check_hover(estimate):
        hover = at_hover(sim, estimate)
        result["hovers"].append(hover)
        if "ee_m" in hover:
            # Evaluation channel only: this value is never used for control or retries.
            target = sim.data.xpos[sim.scene.block_bodies[episode.target_index]]
            hover["scored_xy_error_mm"] = float(np.linalg.norm(np.array(hover["ee_m"][:2]) - target[:2]) * 1000)
        return hover["passed"]

    observed = []
    color = episode.target_color
    _, coarse, _ = sim.coarse(episode, scene_candidates=observed)
    same = [p for c, p in observed if c == color]
    if coarse is None or len(same) != sim.colors.count(color):
        same, result["search_views_xy_m"] = search_source(sim, color)
        if not same:
            return dict(result, status="source_search_unconfirmed")
        coarse = same[0]
    same_neighbors = [p for p in same if np.linalg.norm(p.xy - coarse.xy) > 1e-8]
    neighbors = same_neighbors + [p for c, p in observed if c != color]
    neighbors = [p for p in neighbors if np.linalg.norm(p.xy - coarse.xy) <= 0.10]
    locked = coarse
    result.update(locked_xy_m=locked.xy.tolist(), same_color_neighbors_xy_m=[p.xy.tolist() for p in same_neighbors],
                  ambiguous_views=0)
    approached = sim.approach(coarse)
    first = observe(coarse) if approached else None
    anchor = coarse if first is None else first
    final = None
    if first is not None:
        sim.next_view(coarse, first, 0)
        final = observe(first)
    if final is None or not check_hover(final):
        reason = ("coarse_approach_ik_failed" if not approached else
                  "first_wrist_unconfirmed" if first is None else
                  "second_wrist_unconfirmed" if final is None else "hover_not_ready")
        anchor = anchor if final is None else final
        plan = sim.recenter_wrist(anchor)
        result["retry"] = {"reason": reason, "plan": plan}
        final = observe(anchor) if plan is not None else None
        if final is None and neighbors:
            result["separation_views"] = []
            for q, xy in separated_views(sim, anchor, neighbors):
                sim.motion(0.55, q=q)
                final = observe(anchor)
                result["separation_views"].append({"ee_xy_m": xy.tolist(), "confirmed": final is not None})
                if final is not None:
                    break
        if plan is None and final is None:
            return dict(result, status="alternate_view_unreachable")
        if final is None:
            return dict(result, status="alternate_wrist_unconfirmed")
        if not check_hover(final):
            return dict(result, status=result["hovers"][-1]["status"])
    outcome, _ = sim.grasp(episode, final, 0, start, from_hover=True)
    result["first_grasp"] = outcome
    # A closed-to-limit jaw signals an empty grasp. Reconfirm the same target at
    # the tray before opening or commanding a second descent. No scoring truth is consulted.
    jaw = float(sim.data.qpos[sim.scene.jaw_qpos])
    result["post_lift_jaw_rad"] = jaw
    if abs(jaw - sim.jaw_closed) <= 0.05:
        plan = sim.recenter_wrist(final, height=sim.cfg["hover_z_m"] + 0.025)
        recovery = {"view_reachable": plan is not None}
        result["empty_grasp_retry"] = recovery
        seen = observe(final) if plan is not None else None
        recovery["confirmed"] = seen is not None
        if seen is not None:
            sim.motion(0.35, jaw=sim.jaw_open)
            if check_hover(seen):
                outcome, _ = sim.grasp(episode, seen, 0, start, from_hover=True)
                recovery["grasp"] = outcome
    result.update(outcome)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="configs/benchmarks/finals_overhead_suite.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    suite = load(args.suite)
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    summaries = []
    with (args.out / "trials.jsonl").open("x", encoding="utf8") as stream:
        for case in suite["cases"]:
            cfg = load(suite["base"])
            cfg.update(case.get("bench", {}))
            cfg["settle_qd"] = load("configs/task.yaml")["motion"]["settle_qd"]
            sim = Benchmark(cfg)
            sim.layout_config.update(case.get("layout", {}))
            rows = []
            try:
                episode_seeds = case["seeds"] if "seeds" in case else seeds(case)
                for i, seed in enumerate(episode_seeds):
                    episode = sim.episode(seed, i)
                    row = {"case": case["name"], "seed": seed, "layout": episode.layout,
                           "target": episode.target_index, "color": episode.target_color,
                           "injected_xy_m": episode.offset_xy.tolist(), "injected_yaw_deg": float(np.degrees(episode.offset_yaw)),
                           **trial(sim, episode)}
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    rows.append(row)
            finally:
                sim.perception.close()
            hovers = [r["hovers"][-1]["scored_xy_error_mm"] for r in rows if r["hovers"] and r["hovers"][-1]["passed"]]
            summary = {"case": case["name"], "n": len(rows), "success": sum(r["success"] for r in rows),
                       "retry": sum(r["retry"] is not None for r in rows),
                       "retry_lifted": sum(r["retry"] is not None and r["success"] for r in rows),
                       "first_lifted": sum(r.get("first_grasp", {}).get("success", False) for r in rows),
                       "source_search": sum("search_views_xy_m" in r for r in rows),
                       "empty_grasp_rechecks": sum("empty_grasp_retry" in r for r in rows),
                       "verified_hover": len(hovers), "true_xy_within_3mm": sum(x <= 3 for x in hovers),
                       "xy_mm_median": float(np.median(hovers)) if hovers else None,
                       "xy_mm_max": max(hovers, default=None), "outcomes": dict(Counter(r["status"] for r in rows))}
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            (args.out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()

