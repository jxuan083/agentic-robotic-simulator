#!/usr/bin/env python3
"""Paired wrist-only fixed-view vs controller-driven multi-view RGB evaluation.

Predictions and view association use camera RGB/calibration only. Simulator poses are
read after prediction solely to score the saved outputs and detect block drift.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml

from agentic_sim.config import load
from agentic_sim.mujoco_backend.session import MjSession, SessionConfig
from agentic_sim.mujoco_backend.live_perception import MjLivePerception
from agentic_sim.perception.exploration import WristEstimateAccumulator, plan_wrist_sweep
from evaluate_perception import _ground_truth_pose, _angle_error_deg, _corner_error
from agentic_sim.teleop.keys import KeyState

BASELINE_POSE = (0.15, 0.15, 0.20)
ROBOT_BODY_ROOT = "base"


class ResetCancelled(RuntimeError):
    pass


def _pose_truth(session):
    rows, colors = [], list(session.block_color_names)
    for body_id in session.sc.block_bodies:
        rows.append(_ground_truth_pose(np.asarray(session.d.xpos[body_id], float).copy(),
                                       np.asarray(session.d.xquat[body_id], float).copy(), session.sc.edge))
    return rows, colors


def _robot_bodies(model):
    root = model.body(ROBOT_BODY_ROOT).id
    result = {root}
    for body_id in range(1, model.nbody):
        parent = int(model.body_parentid[body_id])
        while parent and parent not in result:
            parent = int(model.body_parentid[parent])
        if parent in result:
            result.add(body_id)
    return result


def _block_monitor(session, baseline_xyz, robot_bodies):
    pos = np.asarray(session.block_positions(), float)
    drift = np.linalg.norm(pos - baseline_xyz, axis=1)
    speeds = [float(np.linalg.norm(session.d.qvel[adr:adr+3])) for adr in session.sc.block_qvel]
    block_geom = set(session.sc.block_geoms)
    robot_contacts = 0
    block_contacts = 0
    for i in range(session.d.ncon):
        contact = session.d.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if g1 in block_geom or g2 in block_geom:
            block_contacts += 1
            other = g2 if g1 in block_geom else g1
            if int(session.m.geom_bodyid[other]) in robot_bodies:
                robot_contacts += 1
    return {"max_block_drift_mm": float(np.max(drift) * 1000),
            "max_block_linear_speed_m_s": max(speeds, default=0.0),
            "block_contact_count": block_contacts, "robot_block_contact_count": robot_contacts}


def _render_tick(session, viewer, extra):
    if viewer is not None and viewer.should_close():
        raise ResetCancelled("viewer closed; exploration canceled and map cleared")
    session.step(dt=session.control_dt)
    if session.rebuild_request is not None:
        raise ResetCancelled("operator reset requested; exploration canceled and map cleared")
    audit = getattr(session, "_exploration_motion_audit", None)
    if audit is not None:
        sample = _block_monitor(session, audit["reference_xyz_m"], audit["robot_bodies"])
        audit["sampled_frames"] += 1
        audit["max_block_drift_mm"] = max(audit["max_block_drift_mm"], sample["max_block_drift_mm"])
        audit["max_block_linear_speed_m_s"] = max(audit["max_block_linear_speed_m_s"], sample["max_block_linear_speed_m_s"])
        audit["max_block_contact_count"] = max(audit["max_block_contact_count"], sample["block_contact_count"])
        audit["max_robot_block_contact_count"] = max(audit["max_robot_block_contact_count"], sample["robot_block_contact_count"])
        audit["robot_block_contact_frames"] += int(sample["robot_block_contact_count"] > 0)
    hook = getattr(session, "_frame_hook", None)   # e.g. an offscreen video recorder
    if hook is not None:
        hook(session, extra)
    if viewer is not None:
        ticks = getattr(session, "_exploration_gui_tick", 0) + 1
        session._exploration_gui_tick = ticks
        # Poll input each control frame but redraw at 10 Hz; rendering every
        # physics-control step makes this remote Xvfb GUI unnecessarily slow.
        if ticks % 6 == 0:
            status = getattr(session, "_target_search_ui", None)
            viewer.render(session, extra=f"{status} | {extra}" if status else extra)
        else:
            import glfw
            glfw.poll_events()


def _move_segment(session, target, viewer, max_segment_m=0.01, tolerance_m=0.006):
    ctrl = session.ctrl
    start = ctrl.state.pos.copy()
    target = np.asarray(target, float)
    n = max(1, int(math.ceil(np.linalg.norm(target-start) / max_segment_m)))
    for i in range(1, n+1):
        point = start + (target-start) * (i/n)
        if not ctrl.propose(point, ctrl.state.yaw_offset, "wrist_exploration"):
            return False, f"IK/controller rejected waypoint increment: {ctrl.state.message}"
        reached = False
        for _ in range(240):
            _render_tick(session, viewer, f"Wrist exploration target {point.round(3).tolist()}")
            measured = ctrl.chain.ee_pose_mount(session.arm_q())[:3, 3]
            if np.linalg.norm(measured-point) <= tolerance_m:
                reached = True
                break
        if not reached:
            return False, f"measured EE did not reach {point.round(3).tolist()} within timeout"
    return True, "reached"


def _move_to(session, target, sweep, viewer):
    """Raise, traverse at transit height, then descend at the observation point."""
    cur = session.ctrl.chain.ee_pose_mount(session.arm_q())[:3, 3].copy()
    transit_z = max(sweep.transit_z_m, float(cur[2]), float(target[2]))
    route = [np.array([cur[0], cur[1], transit_z]),
             np.array([target[0], target[1], transit_z]), np.asarray(target, float)]
    for point in route:
        ok, reason = _move_segment(session, point, viewer, sweep.max_segment_m)
        if not ok:
            return False, reason
    measured = session.ctrl.chain.ee_pose_mount(session.arm_q())[:3, 3]
    if np.linalg.norm(measured-np.asarray(target, float)) > 0.008:
        return False, f"final measured EE error {np.linalg.norm(measured-np.asarray(target))*1000:.1f} mm"
    return True, "reached"


def _dwell(session, viewer, frames, label):
    for _ in range(frames):
        _render_tick(session, viewer, label)


def _capture(manager, session, view_id, target, output_dir, viewer, accumulator=None, display_extra=None):
    at = time.monotonic()
    future, rgb = manager._capture("wrist", session.d, at)
    overlay, record = future.result(timeout=45)
    measured = session.ctrl.chain.ee_pose_mount(session.arm_q())[:3, 3].copy()
    record["exploration_view_id"] = view_id
    record["ee_target_xyz_m"] = list(target)
    record["ee_measured_xyz_m"] = measured.tolist()
    record["captured_at_unix_s"] = time.time()
    manager.latest["wrist"] = {"record": record, "overlay_rgb": overlay}
    if viewer is not None:
        status = getattr(session, "_target_search_ui", None)
        detail = display_extra or f"Wrist view {view_id}: {len(record['detections'])} candidates"
        viewer.render(session, extra=f"{status} | {detail}" if status else detail)
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path, overlay_path = output_dir / "rgb.png", output_dir / "overlay.png"
    if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write {rgb_path}")
    if not cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write {overlay_path}")
    payload = {"view_id": view_id, "record": record, "rgb_path": str(rgb_path), "overlay_path": str(overlay_path)}
    (output_dir / "view.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if accumulator is not None:
        accumulator.add_view(view_id, float(record["captured_at_monotonic_s"]), record["detections"])
    return payload, rgb, overlay


def _metrics_for(rows, rejected, truth, colors, edge, floor):
    # Successive shortest augmenting paths: exact max-cardinality/min-distance
    # one-to-one match, without exponential search when exploration yields many
    # spatially distinct candidate clusters.
    pred = [r["geometry"] for r in rows if r["geometry"]["status"] == "estimated"]
    n, m = len(pred), len(truth)
    source, p0, t0, sink = 0, 1, 1+n, 1+n+m
    graph = [[] for _ in range(sink+1)]
    def edge(a,b,cap,cost):
        graph[a].append([b,cap,cost,len(graph[b])])
        graph[b].append([a,0,-cost,len(graph[a])-1])
    for i in range(n): edge(source,p0+i,1,0.0)
    for j in range(m): edge(t0+j,sink,1,0.0)
    distances = {}
    for i,prediction in enumerate(pred):
        for j,target in enumerate(truth):
            d=float(np.linalg.norm(np.asarray(prediction["center_xyz_m"])-np.asarray(target["center_xyz_m"])))
            distances[i,j]=d
            if d <= .10: edge(p0+i,t0+j,1,d)
    flow=0
    while True:
        dist=[float("inf")]*len(graph); prev=[None]*len(graph); dist[source]=0.0
        for _ in range(len(graph)-1):
            changed=False
            for u in range(len(graph)):
                if not math.isfinite(dist[u]): continue
                for ei,e in enumerate(graph[u]):
                    v,cap,cost,_=e
                    if cap and dist[v] > dist[u]+cost+1e-12:
                        dist[v]=dist[u]+cost;prev[v]=(u,ei);changed=True
            if not changed: break
        if prev[sink] is None: break
        v=sink
        while v!=source:
            u,ei=prev[v];e=graph[u][ei];e[1]-=1;graph[v][e[3]][1]+=1;v=u
        flow+=1
    pairs=[]
    for i in range(n):
        for e in graph[p0+i]:
            if t0 <= e[0] < t0+m and e[1]==0:
                pairs.append((i,e[0]-t0))
    usedp={i for i,j in pairs};usedt={j for i,j in pairs}
    metric_rows=[]; center=[];yaw=[];corners=[];color_errors=[]
    for i,j in pairs:
        p,t=pred[i],truth[j]
        ce=float(np.linalg.norm(np.asarray(p["center_xyz_m"])-np.asarray(t["center_xyz_m"])))
        ye=_angle_error_deg(p["yaw_rad_mod_pi_over_2"],t["yaw_rad"])
        ke=_corner_error(p,t)
        rec=next(r for r in rows if r["geometry"] is p)
        wrong=rec["color"]!=colors[j]
        center.append(ce);yaw.append(ye);corners.append(ke);color_errors.append(wrong)
        metric_rows.append({"gt_index":j,"detection_id":rec["detection_id"],"status":"matched",
            "center_error_mm":ce*1000,"yaw_error_deg_mod90":ye,"corner_rmse_mm":ke*1000,
            "predicted_color":rec["color"],"ground_truth_color":colors[j],"color_error":wrong})
    for j in range(m):
        if j not in usedt: metric_rows.append({"gt_index":j,"detection_id":None,"status":"missed",
            "center_error_mm":None,"yaw_error_deg_mod90":None,"corner_rmse_mm":None,
            "predicted_color":None,"ground_truth_color":colors[j],"color_error":None})
    for i,p in enumerate(pred):
        if i not in usedp:
            rec=next(r for r in rows if r["geometry"] is p)
            metric_rows.append({"gt_index":None,"detection_id":rec["detection_id"],"status":"false_positive",
                "center_error_mm":None,"yaw_error_deg_mod90":None,"corner_rmse_mm":None,
                "predicted_color":rec["color"],"ground_truth_color":None,"color_error":None})
    for rec in rejected:
        metric_rows.append({"gt_index":None,"detection_id":rec["detection_id"],"status":"rejected",
            "center_error_mm":None,"yaw_error_deg_mod90":None,"corner_rmse_mm":None,
            "predicted_color":rec["color"],"ground_truth_color":None,"color_error":None})
    return {"ground_truth_n":m,"estimated_n":n,"matched_n":flow,"missed_n":m-flow,
        "false_positive_n":n-flow,"rejected_n":len(rejected),
        "rejected_reasons":{},"center_error_mm_matched":_summary([x*1000 for x in center]),
        "yaw_error_deg_mod90_matched":_summary(yaw),"corner_rmse_mm_matched":_summary([x*1000 for x in corners]),
        "color_error_n_matched":sum(color_errors),
        "center_le_5mm_n_over_all_gt":sum(x<=.005 for x in center),
        "center_le_5mm_rate_over_all_gt":sum(x<=.005 for x in center)/max(m,1),
        "yaw_le_5deg_n_over_all_gt":sum(x<=5 for x in yaw),
        "yaw_le_5deg_rate_over_all_gt":sum(x<=5 for x in yaw)/max(m,1),
        "corner_le_5mm_n_over_all_gt":sum(x<=.005 for x in corners),
        "corner_le_5mm_rate_over_all_gt":sum(x<=.005 for x in corners)/max(m,1),
        "per_ground_truth":metric_rows,"assignment":pairs}


def _aggregate(mode_rows):
    keys = ("initial_single", "final_single", "accumulated")
    result = {}
    for key in keys:
        ms = [x["metrics"][key] for x in mode_rows]
        gt_n = sum(x["ground_truth_n"] for x in ms)
        matched = sum(x["matched_n"] for x in ms)
        rows = [r for x in ms for r in x["per_ground_truth"]]
        centers = [r["center_error_mm"] for r in rows if r["status"] == "matched"]
        yaws = [r["yaw_error_deg_mod90"] for r in rows if r["status"] == "matched"]
        corners = [r["corner_rmse_mm"] for r in rows if r["status"] == "matched"]
        result[key] = {"ground_truth_n": gt_n, "matched_n": matched,
                       "missed_n": sum(x["missed_n"] for x in ms),
                       "false_positive_n": sum(x["false_positive_n"] for x in ms),
                       "rejected_n": sum(x["rejected_n"] for x in ms),
                       "center_le_5mm_n_all_gt": sum(x["center_le_5mm_n_over_all_gt"] for x in ms),
                       "center_le_5mm_rate_all_gt": sum(x["center_le_5mm_n_over_all_gt"] for x in ms)/max(gt_n,1),
                       "corner_le_5mm_n_all_gt": sum(x["corner_le_5mm_n_over_all_gt"] for x in ms),
                       "corner_le_5mm_rate_all_gt": sum(x["corner_le_5mm_n_over_all_gt"] for x in ms)/max(gt_n,1),
                       "yaw_le_5deg_n_all_gt": sum(x["yaw_le_5deg_n_over_all_gt"] for x in ms),
                       "yaw_le_5deg_rate_all_gt": sum(x["yaw_le_5deg_n_over_all_gt"] for x in ms)/max(gt_n,1),
                       "center_error_mm_matched": _summary(centers), "yaw_error_deg_mod90_matched": _summary(yaws),
                       "corner_rmse_mm_matched": _summary(corners),
                       "color_error_n_matched": sum(x["color_error_n_matched"] for x in ms)}
    return result


def _summary(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    a = np.asarray(values, float)
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def run_episode(index, seed, random_path, output, scene_path, gui=False):
    viewer = None
    keys = KeyState(actions={"P": "pause", "R": "reset"})
    if gui:
        from agentic_sim.mujoco_backend.viewer import Viewer
        viewer = Viewer(keys, "SO-101 Wrist Exploration", (1500, 940), print, visible=True)
    session = MjSession(SessionConfig(scene=scene_path, random_blocks=True, random_seed=seed,
                                      random_episode=index, random_config=str(random_path)),
                        log=lambda s: print(f"[episode {index}] {s}", flush=True), keys=keys,
                        focused=(lambda: True) if not gui else (viewer.focused), viewer=viewer)
    if viewer:
        viewer.attach(session, reset_camera=True)
    manager = MjLivePerception(session.scene_cfg, session.m, hz=2, log=lambda _: None, camera="wrist")
    session.live_perception = manager
    # Disable opportunistic transit captures; every recorded prediction corresponds to a named reached view.
    manager.next_at = time.monotonic() + 1e9
    manager.last_capture["wrist"] = float("inf")
    sweep = plan_wrist_sweep(session.scene_cfg)
    episode_dir = output / f"episode_{index:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    accumulator = WristEstimateAccumulator(gate_m=0.025)
    route_rows, baseline_capture = [], None
    rejected_observations = 0
    robot_bodies = _robot_bodies(session.m)
    map_path = episode_dir / "latest.json"
    def persist_map(status="active"):
        rows = accumulator.records()
        payload = {"status": status, "episode": index, "camera": "wrist",
                   "latest_view_id": accumulator.latest_view_id,
                   "current_estimates": [r for r in rows if r["exploration"]["currently_visible_in_latest_view"]],
                   "historical_estimates": [r for r in rows if not r["exploration"]["currently_visible_in_latest_view"]],
                   "accumulator": accumulator.summary()}
        map_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (episode_dir / "accumulated.json").write_text(json.dumps({"records": rows,
            "summary": accumulator.summary()}, indent=2), encoding="utf-8")
    try:
        for _ in range(120):
            _render_tick(session, viewer, "Settling randomized tray before exploration")
        motion_reference = np.asarray(session.block_positions(), float)
        session._exploration_motion_audit = {"reference_xyz_m": motion_reference,
            "robot_bodies": robot_bodies, "sampled_frames": 0, "max_block_drift_mm": 0.0,
            "max_block_linear_speed_m_s": 0.0, "max_block_contact_count": 0,
            "max_robot_block_contact_count": 0, "robot_block_contact_frames": 0}
        ok, reason = _move_to(session, BASELINE_POSE, sweep, viewer)
        if not ok:
            raise RuntimeError(f"fixed paired baseline pose unreachable: {reason}")
        _dwell(session, viewer, 45, "Settling at initial wrist observation pose")
        baseline_capture, _, _ = _capture(manager, session, "initial_baseline", BASELINE_POSE,
                                           episode_dir / "views" / "initial_baseline", viewer, accumulator)
        rejected_observations += sum(r["geometry"]["status"] != "estimated"
                                     for r in baseline_capture["record"]["detections"])
        persist_map()
        baseline_truth, gt_colors = _pose_truth(session)
        initial_xyz = np.asarray(session.block_positions(), float)
        initial_contacts = _block_monitor(session, initial_xyz, robot_bodies)

        reached, skipped = [], []
        for vi, target in enumerate(sweep.points_xyz_m):
            view_id = f"sweep_{vi:02d}"
            ok, reason = _move_to(session, target, sweep, viewer)
            if not ok:
                skipped.append({"view_id": view_id, "target_xyz_m": list(target), "reason": reason})
                route_rows.append({"view_id": view_id, "status": "skipped", "reason": reason})
                continue
            _dwell(session, viewer, 30, f"Settling at wrist sweep view {view_id}")
            monitor = _block_monitor(session, initial_xyz, robot_bodies)
            capture, _, _ = _capture(manager, session, view_id, target,
                                     episode_dir / "views" / view_id, viewer, accumulator)
            persist_map()
            reached.append(view_id)
            route_rows.append({"view_id": view_id, "status": "reached", "target_xyz_m": list(target),
                               "measured_ee_xyz_m": capture["record"]["ee_measured_xyz_m"],
                               "candidate_n":len(capture["record"]["detections"]),
                               "rejected_n":sum(r["geometry"]["status"]!="estimated" for r in capture["record"]["detections"]),
                               **monitor})
            rejected_observations += route_rows[-1]["rejected_n"]
        # Save a separate single final-frame baseline at the last physically reached view.
        if reached:
            last_target = next(np.asarray(r["target_xyz_m"]) for r in reversed(route_rows) if r["status"] == "reached")
            # The last sweep view is already a reached physical pose; remain there
            # for a final settling interval and capture a distinct single frame.
            _dwell(session, viewer, 60, "Settling at final wrist observation pose")
            final_capture, _, _ = _capture(manager, session, "final_single", last_target,
                                            episode_dir / "views" / "final_single", viewer, accumulator)
            rejected_observations += sum(r["geometry"]["status"] != "estimated"
                                         for r in final_capture["record"]["detections"])
            persist_map()
        else:
            final_capture = baseline_capture
        for _ in range(120):
            _render_tick(session, viewer, "Settling after wrist scan")
        final_truth, final_colors = _pose_truth(session)
        final_xyz = np.asarray(session.block_positions(), float)
        final_monitor = _block_monitor(session, initial_xyz, robot_bodies)

        baseline_rows = baseline_capture["record"]["detections"]
        final_rows = final_capture["record"]["detections"]
        accumulated_rows = accumulator.records()
        edge = session.sc.edge
        floor = session.sc.trays["storage_pink"]["floor_top"]
        metrics = {
            "initial_single": _metrics_for(baseline_rows,
                [r for r in baseline_rows if r["geometry"]["status"] != "estimated"],
                baseline_truth, gt_colors, edge, floor),
            "final_single": _metrics_for(final_rows,
                [r for r in final_rows if r["geometry"]["status"] != "estimated"],
                final_truth, final_colors, edge, floor),
            "accumulated": _metrics_for(accumulated_rows, [], final_truth, final_colors, edge, floor),
        }
        drift = np.linalg.norm(final_xyz-initial_xyz, axis=1)
        episode = {"episode": index, "seed": seed, "gt_denominator_per_mode": len(final_truth),
                   "paired_baseline_pose_xyz_m": list(BASELINE_POSE),
                   "sweep_planner": {"source": "configured pink-tray geometry plus fixed robot/camera mount mapping",
                                     "points_xyz_m": [list(p) for p in sweep.points_xyz_m],
                                     "transit_z_m": sweep.transit_z_m, "max_segment_m": sweep.max_segment_m},
                   "view_status": {"reached": reached, "skipped": skipped, "route": route_rows},
                   "accumulator": accumulator.summary(), "metrics": metrics,
                   "rejected_observation_n_across_views": rejected_observations,
                   "block_motion_audit": {"baseline_to_final_block_drift_mm": (drift*1000).tolist(),
                       "max_block_drift_mm": float(drift.max(initial=0)*1000),
                       "max_block_linear_speed_m_s": final_monitor["max_block_linear_speed_m_s"],
                       "max_observed_robot_block_contacts": max([x.get("robot_block_contact_count",0)
                                                                    for x in route_rows if x["status"]=="reached"]+[0]),
                       "final_block_contact_count": final_monitor["block_contact_count"],
                       "baseline_contact_count": initial_contacts["block_contact_count"],
                       "all_motion_sampled_frames": session._exploration_motion_audit["sampled_frames"],
                       "all_motion_max_block_drift_mm_from_pre_route": session._exploration_motion_audit["max_block_drift_mm"],
                       "all_motion_max_block_linear_speed_m_s": session._exploration_motion_audit["max_block_linear_speed_m_s"],
                       "all_motion_max_block_contact_count": session._exploration_motion_audit["max_block_contact_count"],
                       "all_motion_max_robot_block_contact_count": session._exploration_motion_audit["max_robot_block_contact_count"],
                       "robot_block_contact_frames": session._exploration_motion_audit["robot_block_contact_frames"],
                       "truth_policy": "baseline scored against settled pre-scan poses; accumulated/final scored against settled post-scan poses"},
                   "view_records": [],
                   "settled_ground_truth_before_scan": baseline_truth,
                   "settled_ground_truth_after_scan": final_truth,
                   "ground_truth_colors_after_scan": final_colors,
                   "limitations": "view sweep is bounded and heuristic; no collision-free guarantee; spatial association may merge close objects or leave duplicate clusters"}
        # Avoid duplicating image bytes; individual view JSON files and the cluster provenance are the source records.
        episode["view_records"] = [{"view_id": baseline_capture["view_id"], "path": baseline_capture["rgb_path"]},
                                    *[{"view_id": x, "path": str(episode_dir / "views" / x / "view.json")} for x in reached],
                                    {"view_id": final_capture["view_id"], "path": final_capture["rgb_path"]}]
        episode["accumulated_records"] = accumulated_rows
        episode["accumulated_records_path"] = str(episode_dir / "accumulated.json")
        episode["latest_state_path"] = str(map_path)
        episode["block_motion_audit"]["pre_route_to_final_drift_mm"] = float(
            np.linalg.norm(np.asarray(session.block_positions(), float)-motion_reference, axis=1).max(initial=0)*1000)
        persist_map("complete")
        (episode_dir / "evaluation.json").write_text(json.dumps(episode, indent=2), encoding="utf-8")
        return episode
    except ResetCancelled as exc:
        accumulator.reset()
        latest = {"status": "cancelled", "reason": str(exc), "episode": index,
                  "camera": "wrist", "current_estimates": [], "historical_estimates": [],
                  "accumulator": accumulator.summary()}
        (episode_dir / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
        (episode_dir / "accumulated.json").write_text(json.dumps({"records": [],
            "summary": accumulator.summary()}, indent=2), encoding="utf-8")
        raise
    finally:
        if session.live_perception:
            session.live_perception = None
        manager.close()
        session.finish()
        if viewer:
            viewer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default="outputs/perception-exploration/mujoco-final")
    p.add_argument("--scene", default="configs/scenes/lab_20261120.yaml")
    p.add_argument("--gui", action="store_true", help="show the actual controller-driven single-episode sweep")
    args = p.parse_args()
    if args.episodes <= 0 or (args.gui and args.episodes != 1):
        p.error("episodes must be positive; --gui currently supports exactly one episode")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    random_cfg = load("configs/random_blocks.yaml")
    random_cfg["count_range"] = [8, 8]
    random_cfg["check_ik"] = True
    random_path = output / "random_blocks_eval.yaml"
    random_path.write_text(yaml.safe_dump(random_cfg, sort_keys=False), encoding="utf-8")
    all_rows = []
    try:
        for i in range(args.episodes):
            result = run_episode(i, args.seed, random_path, output, args.scene, args.gui)
            all_rows.append(result)
            print(f"completed episode {i+1}/{args.episodes}: views reached={len(result['view_status']['reached'])}, "
                  f"accumulated clusters={result['accumulator']['cluster_n']}, "
                  f"drift={result['block_motion_audit']['max_block_drift_mm']:.2f} mm", flush=True)
    except ResetCancelled as exc:
        print(f"exploration canceled: {exc}; accumulated map was cleared", flush=True)
        return
    summary = {"seed": args.seed, "episodes_n": len(all_rows),
               "ground_truth_n_per_mode": sum(x["metrics"]["initial_single"]["ground_truth_n"] for x in all_rows),
               "modes": _aggregate(all_rows),
               "view_counts": {"reached": sum(len(x["view_status"]["reached"]) for x in all_rows),
                               "skipped": sum(len(x["view_status"]["skipped"]) for x in all_rows)},
               "accumulation": {"observations": sum(x["accumulator"]["observation_n"] for x in all_rows),
                                "clusters": sum(x["accumulator"]["cluster_n"] for x in all_rows),
                                "merged_observations": sum(x["accumulator"]["merged_observation_n"] for x in all_rows),
                                "rejected_observations_across_views": sum(x["rejected_observation_n_across_views"] for x in all_rows)},
               "block_motion": {"episodes_max_drift_over_2mm": sum(x["block_motion_audit"]["max_block_drift_mm"]>2 for x in all_rows),
                                "max_drift_mm": max(x["block_motion_audit"]["max_block_drift_mm"] for x in all_rows),
                                "max_pre_route_motion_drift_mm": max(x["block_motion_audit"]["all_motion_max_block_drift_mm_from_pre_route"] for x in all_rows),
                                "episodes_with_robot_block_contact": sum(x["block_motion_audit"]["robot_block_contact_frames"]>0 for x in all_rows)}}
    (output / "evaluation.json").write_text(json.dumps({"summary": summary, "episodes": all_rows}, indent=2), encoding="utf-8")
    with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode","mode","gt_n","matched_n","missed_n","false_positive_n",
            "center_le_5mm_rate_all_gt","corner_le_5mm_rate_all_gt","yaw_le_5deg_rate_all_gt","center_mean_mm_matched",
            "yaw_mean_deg_mod90_matched","corner_mean_mm_matched","color_error_n_matched"])
        writer.writeheader()
        for ep in all_rows:
            for mode, m in ep["metrics"].items():
                writer.writerow({"episode":ep["episode"],"mode":mode,"gt_n":m["ground_truth_n"],
                    "matched_n":m["matched_n"],"missed_n":m["missed_n"],"false_positive_n":m["false_positive_n"],
                    "center_le_5mm_rate_all_gt":m["center_le_5mm_rate_over_all_gt"],
                    "corner_le_5mm_rate_all_gt":m["corner_le_5mm_rate_over_all_gt"],
                    "yaw_le_5deg_rate_all_gt":m["yaw_le_5deg_rate_over_all_gt"],
                    "center_mean_mm_matched":m["center_error_mm_matched"]["mean"],
                    "yaw_mean_deg_mod90_matched":m["yaw_error_deg_mod90_matched"]["mean"],
                    "corner_mean_mm_matched":m["corner_rmse_mm_matched"]["mean"],
                    "color_error_n_matched":m["color_error_n_matched"]})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
