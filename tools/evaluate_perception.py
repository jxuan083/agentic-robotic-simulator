#!/usr/bin/env python3
"""Rebuild randomized MuJoCo episodes; compare RGB-only estimates with settled GT.

The predictor sees only native camera RGB, calibration and synchronized camera pose.
Ground-truth block poses are read by the evaluator after inference inputs are captured.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import mujoco
import numpy as np
import yaml

from agentic_sim.config import load
from agentic_sim.mujoco_backend.session import MjSession, SessionConfig
from agentic_sim.mujoco_backend.live_perception import MjLivePerception
from agentic_sim.perception.vertices import SIGNS


def _yaw_from_wxyz(q):
    w, x, y, z = (float(a) for a in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _ground_truth_pose(center_xyz_m, quat_wxyz, edge_m):
    """Build corner GT from the simulator's settled rigid pose and canonical cube mesh."""
    from agentic_sim.kinematics import quat_wxyz_to_mat
    center = np.asarray(center_xyz_m, float)
    rotation = quat_wxyz_to_mat(np.asarray(quat_wxyz, float))
    local = SIGNS.copy()
    local[:, :2] *= float(edge_m) / 2
    local[:, 2] = (local[:, 2] - 0.5) * float(edge_m)
    corners = local @ rotation.T + center
    return {"center_xyz_m": center.tolist(), "yaw_rad": _yaw_from_wxyz(quat_wxyz) % (math.pi / 2),
            "corners_xyz_m": corners.tolist(), "quat_wxyz": np.asarray(quat_wxyz, float).tolist()}


def _assign(predicted, truth, gate_m=0.10):
    """Max-cardinality, min-distance one-to-one 3D matching (no color/GT in predictor)."""
    best = ((), float("inf"))
    n = len(predicted)
    distances = [[float(np.linalg.norm(np.asarray(p["center_xyz_m"]) - np.asarray(t["center_xyz_m"])))
                  for t in truth] for p in predicted]
    def visit(i, used, pairs, total):
        nonlocal best
        if i == n:
            candidate = (tuple(pairs), total)
            if len(candidate[0]) > len(best[0]) or (len(candidate[0]) == len(best[0]) and total < best[1]):
                best = candidate
            return
        visit(i + 1, used, pairs, total)
        for j, distance in enumerate(distances[i]):
            if j not in used and distance <= gate_m:
                visit(i + 1, used | {j}, pairs + [(i, j)], total + distance)
    visit(0, set(), [], 0.0)
    return list(best[0])


def _angle_error_deg(a, b):
    period = math.pi / 2
    return math.degrees(abs((float(a) - float(b) + period / 2) % period - period / 2))


def _corner_error(pred, truth):
    a = np.asarray([v["xyz_m"] for v in pred["vertices"]], float)
    b = np.asarray(truth["corners_xyz_m"], float)
    errors = []
    for k in range(4):
        # Rotate the local square indices by 90-degree symmetry; top and bottom share it.
        perm = [(i + k) % 4 for i in range(4)] + [4 + (i + k) % 4 for i in range(4)]
        errors.append(float(np.sqrt(np.mean(np.sum((a - b[perm]) ** 2, axis=1)))))
    return min(errors)


def _camera_metrics(pred_records, rejected, truth, colors, edge, floor):
    estimated = [row["geometry"] for row in pred_records if row["geometry"]["status"] == "estimated"]
    pairs = _assign(estimated, truth)
    matched_pred, matched_gt = {i for i, _ in pairs}, {j for _, j in pairs}
    center, yaw, corners, color_errors = [], [], [], []
    rows = []
    for i, j in pairs:
        p, t = estimated[i], truth[j]
        ce = float(np.linalg.norm(np.asarray(p["center_xyz_m"]) - np.asarray(t["center_xyz_m"])))
        ye = _angle_error_deg(p["yaw_rad_mod_pi_over_2"], t["yaw_rad"])
        ke = _corner_error(p, t)
        predicted_row = next(r for r in pred_records if r["geometry"] is p)
        color_wrong = predicted_row["color"] != colors[j]
        center.append(ce); yaw.append(ye); corners.append(ke); color_errors.append(color_wrong)
        rows.append({"gt_index": j, "detection_id": predicted_row["detection_id"], "status": "matched",
                     "center_error_mm": ce * 1000, "yaw_error_deg_mod90": ye,
                     "corner_rmse_mm": ke * 1000, "predicted_color": predicted_row["color"],
                     "ground_truth_color": colors[j], "color_error": color_wrong})
    for j, target in enumerate(truth):
        if j not in matched_gt:
            rows.append({"gt_index": j, "detection_id": None, "status": "missed",
                         "center_error_mm": None, "yaw_error_deg_mod90": None, "corner_rmse_mm": None,
                         "predicted_color": None, "ground_truth_color": colors[j], "color_error": None})
    false_positive = len(estimated) - len(matched_pred)
    for i, prediction in enumerate(estimated):
        if i not in matched_pred:
            rec = next(r for r in pred_records if r["geometry"] is prediction)
            rows.append({"gt_index": None, "detection_id": rec["detection_id"], "status": "false_positive",
                         "center_error_mm": None, "yaw_error_deg_mod90": None, "corner_rmse_mm": None,
                         "predicted_color": rec["color"], "ground_truth_color": None, "color_error": None})
    for rec in rejected:
        rows.append({"gt_index": None, "detection_id": rec["detection_id"], "status": "rejected",
                     "center_error_mm": None, "yaw_error_deg_mod90": None, "corner_rmse_mm": None,
                     "predicted_color": rec["color"], "ground_truth_color": None, "color_error": None})
    return {"ground_truth_n": len(truth), "estimated_n": len(estimated), "matched_n": len(pairs),
            "missed_n": len(truth) - len(matched_gt), "false_positive_n": false_positive,
            "rejected_n": len(rejected), "rejected_reasons": _counts(r.get("geometry", {}).get("reason") for r in rejected),
            "center_error_mm_matched": _summary([x * 1000 for x in center]), "yaw_error_deg_mod90_matched": _summary(yaw),
            "corner_rmse_mm_matched": _summary([x * 1000 for x in corners]), "color_error_n_matched": sum(color_errors),
            "center_le_5mm_n_over_all_gt": sum(x <= .005 for x in center),
            "center_le_5mm_rate_over_all_gt": sum(x <= .005 for x in center) / max(1, len(truth)),
            "yaw_le_5deg_n_over_all_gt": sum(x <= 5 for x in yaw),
            "yaw_le_5deg_rate_over_all_gt": sum(x <= 5 for x in yaw) / max(1, len(truth)),
            "corner_le_5mm_n_over_all_gt": sum(x <= .005 for x in corners),
            "corner_le_5mm_rate_over_all_gt": sum(x <= .005 for x in corners) / max(1, len(truth)),
            "per_ground_truth": rows, "assignment": pairs}


def _counts(items):
    out = {}
    for item in items:
        out[str(item)] = out.get(str(item), 0) + 1
    return out


def _summary(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(values, float)
    return {"n": int(len(arr)), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)), "max": float(arr.max())}


def _cross_camera(global_rows, wrist_rows):
    g = [r for r in global_rows if r["geometry"]["status"] == "estimated"]
    w = [r for r in wrist_rows if r["geometry"]["status"] == "estimated"]
    candidates = []
    for i, a in enumerate(g):
        for j, b in enumerate(w):
            dist = float(np.linalg.norm(np.asarray(a["geometry"]["center_xyz_m"]) -
                                        np.asarray(b["geometry"]["center_xyz_m"])))
            if dist <= .015:
                candidates.append((dist, i, j))
    used_g, used_w, pairs = set(), set(), []
    for dist, i, j in sorted(candidates):
        if i in used_g or j in used_w:
            continue
        used_g.add(i); used_w.add(j)
        ga, wa = g[i]["geometry"], w[j]["geometry"]
        pairs.append({"color": g[i]["color"], "global_detection_id": g[i]["detection_id"],
                      "wrist_detection_id": w[j]["detection_id"], "center_delta_mm": dist * 1000,
                      "colors_agree": g[i]["color"] == w[j]["color"],
                      "yaw_delta_deg_mod90": _angle_error_deg(ga["yaw_rad_mod_pi_over_2"],
                                                                 wa["yaw_rad_mod_pi_over_2"])})
    return {"association": "predicted_3d_nearest_one_to_one_gate_15mm_no_GT", "matched_n": len(pairs),
            "global_unpaired_n": len(g) - len(used_g), "wrist_unpaired_n": len(w) - len(used_w),
            "color_disagreement_n": sum(not p["colors_agree"] for p in pairs),
            "color_consistency_threshold": "exact label agreement; >=5mm predicted center difference is a separate disagreement warning",
            "center_delta_mm": _summary([p["center_delta_mm"] for p in pairs]),
            "yaw_delta_deg_mod90": _summary([p["yaw_delta_deg_mod90"] for p in pairs]), "pairs": pairs}


def run(episodes, seed, output, scene_path):
    os.environ.setdefault("MUJOCO_GL", "egl")
    output.mkdir(parents=True, exist_ok=True)
    random_cfg = load("configs/random_blocks.yaml")
    random_cfg["count_range"] = [8, 8]
    random_cfg["check_ik"] = True
    random_path = output / "random_blocks_eval.yaml"
    random_path.write_text(yaml.safe_dump(random_cfg, sort_keys=False), encoding="utf-8")
    all_episodes, csv_rows = [], []
    for index in range(episodes):
        cfg = SessionConfig(scene=scene_path, random_blocks=True, random_seed=seed,
                            random_episode=index, random_config=str(random_path))
        session = MjSession(cfg, log=lambda s: print(f"[episode {index}] {s}", flush=True))
        try:
            # Let contact settling finish while the home arm remains at its reset target.
            for _ in range(120):
                mujoco.mj_step(session.m, session.d)
            # Fixed, camera-oriented reachable wrist pose; it is independent of block GT.
            target = np.array([0.15, 0.15, 0.20], float)
            q, ok, info = session.ctrl.chain.solve_ik_tilt(
                target, session.ctrl.yaw_for(target, 0.0), session.ctrl.q_home.copy(), session.ctrl.p.max_tilt_deg)
            if not ok:
                raise RuntimeError(f"fixed wrist observation pose is unreachable: {info}")
            for address, actuator, offset, value in zip(session.sc.arm_qpos, session.sc.arm_act,
                                                        session.sc.arm_offset, q):
                command = value + offset
                session.d.qpos[address] = command
                session.d.ctrl[actuator] = command
            for _ in range(700):
                mujoco.mj_step(session.m, session.d)
            mujoco.mj_forward(session.m, session.d)
            gt, gt_colors = [], list(session.block_color_names)
            for body_id in session.sc.block_bodies:
                pos = np.asarray(session.d.xpos[body_id], float).copy()
                gt.append(_ground_truth_pose(pos, session.d.xquat[body_id], session.sc.edge))
            manager = MjLivePerception(session.scene_cfg, session.m, hz=2.0, log=lambda _: None)
            camera_data = {}
            try:
                for key in ("global", "wrist"):
                    captured_at = time.monotonic()
                    future, rgb = manager._capture(key, session.d, captured_at)
                    overlay, record = future.result(timeout=40)
                    camera_data[key] = {"rgb": rgb, "overlay": overlay, "record": record}
            finally:
                manager.close()
            edge = session.sc.edge
            floor = session.sc.trays["storage_pink"]["floor_top"]
            camera_results, episode_dir = {}, output / f"episode_{index:03d}"
            episode_dir.mkdir(exist_ok=True)
            for key, captured in camera_data.items():
                record = captured["record"]
                rows = record["detections"]
                rejected = [r for r in rows if r["geometry"]["status"] != "estimated"]
                metric = _camera_metrics(rows, rejected, gt, gt_colors, edge, floor)
                record["evaluation"] = metric
                camera_results[key] = {"record": record, "metrics": metric}
                if index < 3:
                    import cv2
                    cv2.imwrite(str(episode_dir / f"{key}_rgb.png"), cv2.cvtColor(captured["rgb"], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(episode_dir / f"{key}_overlay.png"), cv2.cvtColor(captured["overlay"], cv2.COLOR_RGB2BGR))
                for row in metric["per_ground_truth"]:
                    csv_rows.append({"episode": index, "seed": seed, "camera": key, **row})
            cross = _cross_camera(camera_data["global"]["record"]["detections"],
                                  camera_data["wrist"]["record"]["detections"])
            episode_result = {"episode": index, "seed": seed, "random_spawn": session.random_spawn,
                              "settled_ground_truth": gt, "ground_truth_colors": gt_colors,
                              "cameras": camera_results, "cross_camera": cross,
                              "camera_visibility": {key: {"captured": True,
                                  "candidate_n": len(camera_data[key]["record"]["detections"]),
                                  "estimated_n": camera_results[key]["metrics"]["estimated_n"]}
                                  for key in ("global", "wrist")}}
            all_episodes.append(episode_result)
            (episode_dir / "evaluation.json").write_text(json.dumps(episode_result, indent=2), encoding="utf-8")
            (episode_dir / "ground_truth.json").write_text(json.dumps({"episode": index, "seed": seed,
                "poses": gt, "colors": gt_colors, "source": "settled MuJoCo data after RGB captures"}, indent=2), encoding="utf-8")
            print(f"episode {index+1}/{episodes}: global {camera_results['global']['metrics']['matched_n']}/8 matched, "
                  f"wrist {camera_results['wrist']['metrics']['matched_n']}/8 matched; cross={cross['matched_n']}", flush=True)
        finally:
            session.finish()
    summary = summarize(all_episodes, seed)
    (output / "evaluation.json").write_text(json.dumps({"summary": summary, "episodes": all_episodes}, indent=2), encoding="utf-8")
    with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["episode", "seed", "camera", "gt_index", "detection_id", "status", "center_error_mm",
                  "yaw_error_deg_mod90", "corner_rmse_mm", "predicted_color", "ground_truth_color", "color_error"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(csv_rows)
    write_summary(output / "luna-summary.md", summary)
    return summary


def summarize(episodes, seed):
    out = {"seed": seed, "episodes_n": len(episodes), "ground_truth_n_per_camera": sum(
        len(ep["settled_ground_truth"]) for ep in episodes), "cameras": {}, "cross_camera": {}}
    for key in ("global", "wrist"):
        metrics = [ep["cameras"][key]["metrics"] for ep in episodes]
        gt_n = sum(m["ground_truth_n"] for m in metrics)
        matched = sum(m["matched_n"] for m in metrics)
        center_values = [ep["cameras"][key]["record"]["evaluation"]["per_ground_truth"] for ep in episodes]
        center = [row["center_error_mm"] for rows in center_values for row in rows if row["status"] == "matched"]
        yaw = [row["yaw_error_deg_mod90"] for rows in center_values for row in rows if row["status"] == "matched"]
        corner = [row["corner_rmse_mm"] for rows in center_values for row in rows if row["status"] == "matched"]
        out["cameras"][key] = {"ground_truth_n": gt_n, "matched_n": matched,
            "missed_n": sum(m["missed_n"] for m in metrics),
            "false_positive_n": sum(m["false_positive_n"] for m in metrics),
            "rejected_n": sum(m["rejected_n"] for m in metrics),
            "matched_center_error_mm": _summary(center), "matched_yaw_error_deg_mod90": _summary(yaw),
            "matched_corner_rmse_mm": _summary(corner),
            "center_le_5mm_over_all_gt": sum(v <= 5 for v in center) / max(1, gt_n),
            "yaw_le_5deg_over_all_gt": sum(v <= 5 for v in yaw) / max(1, gt_n),
            "corner_le_5mm_over_all_gt": sum(v <= 5 for v in corner) / max(1, gt_n),
            "color_error_n_matched": sum(m["color_error_n_matched"] for m in metrics),
            "center_error_denominator": "matched only for distribution; pass rate denominator is every GT, misses fail"}
    pairs = [ep["cross_camera"] for ep in episodes]
    out["cross_camera"] = {"association": "predicted 3D nearest one-to-one, <=15 mm, no GT",
        "matched_n": sum(p["matched_n"] for p in pairs),
        "global_unpaired_n": sum(p["global_unpaired_n"] for p in pairs),
        "wrist_unpaired_n": sum(p["wrist_unpaired_n"] for p in pairs),
        "color_disagreement_n": sum(p["color_disagreement_n"] for p in pairs),
        "center_disagreement_ge_5mm_n": sum(1 for ep in episodes for q in ep["cross_camera"]["pairs"]
                                             if q["center_delta_mm"] >= 5),
        "pair_center_delta_mm": _summary([q["center_delta_mm"] for ep in episodes for q in ep["cross_camera"]["pairs"]]),
        "pair_yaw_delta_deg_mod90": _summary([q["yaw_delta_deg_mod90"] for ep in episodes for q in ep["cross_camera"]["pairs"]])}
    return out


def write_summary(path, summary):
    lines = ["# Live perception multi-episode validation", "",
             f"MuJoCo episodes: {summary['episodes_n']}, seed: {summary['seed']}.",
             "Eight-block randomized layouts; each episode rebuilt the MuJoCo model and used native global and wrist RGB.",
             "Prediction receives camera RGB/calibration only. Ground truth is read separately after settling.",
             "Matching uses 3D center only (gate 100 mm), never color. Center pass threshold is 5 mm; denominator includes all GT and misses fail.", ""]
    for key, m in summary["cameras"].items():
        lines += [f"## {key}", "", f"GT {m['ground_truth_n']}; matched {m['matched_n']}; missed {m['missed_n']}; "
                  f"false positive {m['false_positive_n']}; fit rejected {m['rejected_n']}.",
                  f"Matched center error mm: {m['matched_center_error_mm']}; pass ≤5 mm over all GT: {m['center_le_5mm_over_all_gt']:.3f}.",
                  f"Yaw modulo 90°: {m['matched_yaw_error_deg_mod90']}; ≤5° over all GT: {m['yaw_le_5deg_over_all_gt']:.3f}.",
                  f"Eight-corner RMSE mm: {m['matched_corner_rmse_mm']}; ≤5 mm over all GT: {m['corner_le_5mm_over_all_gt']:.3f}.",
                  f"Matched color disagreements: {m['color_error_n_matched']}.", ""]
    cross = summary["cross_camera"]
    lines += ["## Cross-camera RGB-only association", "",
             f"Predicted 3D nearest one-to-one ≤15 mm, no GT: matched {cross['matched_n']}, "
              f"global unpaired {cross['global_unpaired_n']}, wrist unpaired {cross['wrist_unpaired_n']}; "
             f"color disagreements {cross['color_disagreement_n']}, center differences ≥5 mm {cross['center_disagreement_ge_5mm_n']}; "
              f"center delta {cross['pair_center_delta_mm']}; yaw delta {cross['pair_yaw_delta_deg_mod90']}.", "",
              "Files: `evaluation.json`, `evaluation.csv`, per-episode `evaluation.json` and `ground_truth.json`; "
              "the first three episodes include each camera's raw RGB and prediction overlay.",
              "Wrist capture uses the configured real-lens renderer-and-resampling path at native sensor resolution. "
              "This validates the simulated camera model, not a physical camera calibration."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--output", default="outputs/perception-validation")
    parser.add_argument("--scene", default="configs/scenes/lab_20261120.yaml")
    args = parser.parse_args()
    if args.episodes < 10:
        parser.error("--episodes must be at least 10")
    summary = run(args.episodes, args.seed, Path(args.output), args.scene)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
