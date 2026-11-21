#!/usr/bin/env python3
"""Capture image-only Isaac global/wrist perception across randomized episodes."""
from __future__ import annotations

import argparse, csv, json, math, os, sys, time, queue
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher

SessionConfig = TeleopSession = None
quat_wxyz_to_mat = SIGNS = rendered_camera = _Capture = None
_camera_metrics = _cross_camera = summarize = write_summary = None


def truth(session):
    poses, colors = [], list(session.block_color_names)
    for block in session.blocks:
        pos = block.data.root_pos_w[0].cpu().numpy().astype(float)
        q = block.data.root_quat_w[0].cpu().numpy().astype(float)
        rot = quat_wxyz_to_mat(q)
        yaw = math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3])) % (math.pi/2)
        local = SIGNS.copy()
        local[:, :2] *= session.edge/2
        local[:, 2] = (local[:, 2]-.5)*session.edge
        poses.append({"center_xyz_m": pos.tolist(), "yaw_rad": yaw,
                      "corners_xyz_m": (local @ rot.T + pos).tolist(), "quat_wxyz": q.tolist()})
    return poses, colors


def capture(session, deadline_s=60):
    # Reuse native sensor viewports; estimator is pure RGB + same-frame calibration.
    from agentic_sim.perception.live import LiveEstimator
    from agentic_sim.perception.vertices import CameraGeometry
    from agentic_sim.scene_layout import tray_parts
    from omni.kit.app import get_app
    views=session.sensor_views
    bases={k:CameraGeometry.from_scene(session.scene_cfg,k,[0.]*5 if k=="wrist" else None) for k in ("global","wrist")}
    floor=tray_parts(session.scene_cfg["trays"]["storage_pink"])[1]["floor_top"]
    evaluators={k:LiveEstimator(session.edge,floor) for k in bases}
    inboxes = {k: queue.SimpleQueue() for k in bases}
    captures = {}
    for key in bases:
        captures[key] = _Capture(inboxes[key], 0)
        views.windows[key].viewport_api.schedule_capture(captures[key])
    start = time.monotonic()
    captured = {}
    while time.monotonic()-start < deadline_s and len(captured) < 2:
        session.sim.render()
        get_app().update()
        for key in bases:
            if key in captured or inboxes[key].empty():
                continue
            generation, rgb, view, projection, frame, at, error = inboxes[key].get()
            if error: raise RuntimeError(f"{key} ByteCapture failed: {error}")
            camera = rendered_camera(bases[key], view, projection, rgb.shape[1], rgb.shape[0])
            q = session.robot.data.joint_pos[0,session.arm_ids].cpu().numpy() if key == "wrist" else None
            expected = CameraGeometry.from_scene(session.scene_cfg,key,q).mount_from_cv
            p_err=float(np.linalg.norm(camera.mount_from_cv[:3,3]-expected[:3,3]))
            r_err_mat=camera.mount_from_cv[:3,:3] @ expected[:3,:3].T
            r_err=math.degrees(math.acos(float(np.clip((np.trace(r_err_mat)-1)/2,-1,1))))
            if p_err > .01 or r_err > 5:
                raise RuntimeError(f"{key} captured camera pose differs from live FK/calibration: {p_err*1000:.2f}mm, {r_err:.2f}deg")
            overlay, record = evaluators[key].predict(rgb, camera, frame, at)
            record["camera_pose_check"]={"translation_error_mm":p_err*1000,"rotation_error_deg":r_err,"passed":True}
            record["rgb"] = rgb
            record["overlay"] = overlay
            captured[key] = record
    if len(captured) != 2:
        raise TimeoutError(f"Isaac camera capture timed out; completed {sorted(captured)}")
    return captured


def run(episodes, seed, output, scene):
    global SessionConfig, TeleopSession, quat_wxyz_to_mat, SIGNS, rendered_camera, _Capture
    global _camera_metrics, _cross_camera, summarize, write_summary
    from agentic_sim.isaac.session import SessionConfig, TeleopSession
    from agentic_sim.kinematics import quat_wxyz_to_mat
    from agentic_sim.perception.vertices import SIGNS
    from agentic_sim.perception.live import rendered_camera
    from agentic_sim.isaac.live_perception import _Capture
    from agentic_sim.perception.evaluation import camera_metrics, cross_camera, summarize
    _camera_metrics, _cross_camera = camera_metrics, cross_camera
    write_summary = None
    output.mkdir(parents=True, exist_ok=True)
    cfg_path = output / "random_blocks_eval.yaml"
    cfg = yaml.safe_load(Path("configs/random_blocks.yaml").read_text())
    cfg["count_range"] = [8, 8]
    cfg["check_ik"] = True
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    all_eps, csv_rows = [], []
    for i in range(episodes):
        print(f"[episode {i}] building Isaac session", flush=True)
        s = TeleopSession(SessionConfig(scene=scene, random_blocks=True, random_seed=seed,
                random_episode=i, random_config=str(cfg_path), live_perception=False,
                perception_camera="both", perception_hz=2.0, device="cpu"),
                log=lambda x: print(f"[episode {i}] {x}", flush=True))
        try:
            # Move to a reproducible reachable viewing pose independent of block GT.
            target = [0.10, 0.10, 0.15]
            start = s.ctrl.state.pos.copy()
            for fraction in np.linspace(0.0, 1.0, 31)[1:]:
                waypoint = start + (np.asarray(target)-start)*fraction
                if not s.ctrl.propose(waypoint, 0.0, "evaluation"):
                    raise RuntimeError(f"Isaac observation waypoint {waypoint.tolist()} rejected by IK: {s.ctrl.state.message}")
            settled=0
            for _ in range(360):
                s.step()
                q=s.robot.data.joint_pos[0,s.arm_ids].cpu().numpy()
                if float(np.max(np.abs(q-s.ctrl.state.q_cmd))) < .025: settled += 1
                else: settled=0
                if _ >= 119 and settled >= 30: break
            gt_before, _ = truth(s)
            images = capture(s)
            gt, colors = truth(s)
            drift = max(float(np.linalg.norm(np.asarray(a["center_xyz_m"])-np.asarray(b["center_xyz_m"]))) for a,b in zip(gt_before,gt))
            cam_results, epdir = {}, output / f"episode_{i:03d}"
            epdir.mkdir(exist_ok=True)
            for key in ("global", "wrist"):
                record = images[key]
                detections = record["detections"]
                rejected = [r for r in detections if r["geometry"]["status"] != "estimated"]
                metrics = _camera_metrics(detections, rejected, gt, colors)
                record["evaluation"] = metrics
                cam_results[key] = {"record": record, "metrics": metrics}
                import cv2
                rgb = record["rgb"]
                overlay = record["overlay"]
                cv2.imwrite(str(epdir/f"{key}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(epdir/f"{key}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                record.pop("rgb", None); record.pop("overlay", None)
                for row in metrics["per_ground_truth"]:
                    csv_rows.append({"episode": i, "seed": seed, "camera": key, **row})
            cross = _cross_camera(images["global"]["detections"], images["wrist"]["detections"])
            ep = {"episode": i, "seed": seed, "random_spawn": s.random_spawn,
                  "settled_ground_truth": gt, "settled_gt_max_drift_during_capture_mm": drift*1000,
                  "ground_truth_colors": colors, "cameras": cam_results,
                  "cross_camera": cross, "camera_visibility": {k: {"captured": True,
                      "candidate_n": len(images[k]["detections"]), "estimated_n": cam_results[k]["metrics"]["estimated_n"]}
                      for k in ("global", "wrist")}}
            all_eps.append(ep)
            (epdir/"evaluation.json").write_text(json.dumps(ep, indent=2))
            (epdir/"ground_truth.json").write_text(json.dumps({"episode": i, "seed": seed,
                "poses": gt, "colors": colors, "source": "settled Isaac RigidObject root state after image capture"}, indent=2))
            print(f"episode {i+1}/{episodes}: global {cam_results['global']['metrics']['matched_n']}/8; "
                  f"wrist {cam_results['wrist']['metrics']['matched_n']}/8", flush=True)
        finally:
            s.close()
            # A Kit viewport can retain the just-cleared stage across SimulationContext rebuilds.
            # Recreate the resident sensor widgets so the next episode renders its new camera prims.
            import agentic_sim.isaac.sensor_views as sensor_views_module
            for window in list(sensor_views_module._instance.windows.values()):
                try: window.destroy()
                except Exception: pass
            sensor_views_module._instance = None
    summary = summarize(all_eps, seed)
    (output/"evaluation.json").write_text(json.dumps({"engine":"Isaac Lab", "summary":summary, "episodes":all_eps}, indent=2))
    fields = ["episode", "seed", "camera", "gt_index", "detection_id", "status", "center_error_mm",
              "yaw_error_deg_mod90", "corner_rmse_mm", "predicted_color", "ground_truth_color", "color_error"]
    with (output/"evaluation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(csv_rows)
    lines=["# Isaac live perception validation","",f"Isaac Lab randomized episodes: {summary['episodes_n']}; seed {summary['seed']}. Eight blocks per episode.","Prediction uses rendered RGB and same-frame camera matrices. Ground truth comes from settled Isaac rigid-body poses after capture. Matching uses 100 mm center gate, with misses included in the all-GT pass denominator.",""]
    max_drift=max(e["settled_gt_max_drift_during_capture_mm"] for e in all_eps)
    max_pose=max(max(e["cameras"][k]["record"]["camera_pose_check"]["translation_error_mm"] for k in ("global","wrist")) for e in all_eps)
    lines += [f"All ten captures passed camera pose comparison against the configured global extrinsic and measured wrist FK; maximum translation discrepancy {max_pose:.6f} mm. Maximum settled GT drift while capturing: {max_drift:.6f} mm.",""]
    for k,m in summary["cameras"].items():
        lines += [f"## {k}","",f"GT {m['ground_truth_n']}; matched {m['matched_n']}; missed {m['missed_n']}; false positive {m['false_positive_n']}; rejected {m['rejected_n']}.",f"Center mm {m['matched_center_error_mm']}; ≤5 mm/all GT {m['center_le_5mm_over_all_gt']:.3f}.",f"Yaw modulo 90° {m['matched_yaw_error_deg_mod90']}; ≤5°/all GT {m['yaw_le_5deg_over_all_gt']:.3f}.",f"Eight-corner RMSE mm {m['matched_corner_rmse_mm']}; ≤5 mm/all GT {m['corner_le_5mm_over_all_gt']:.3f}.",""]
    lines += ["## Cross-camera","",f"Predicted-center nearest one-to-one ≤15 mm, no GT: {summary['cross_camera']['matched_n']} pairs; color disagreements {summary['cross_camera']['color_disagreement_n']}.",""]
    (output/"summary.md").write_text("\n".join(lines))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10); p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--output", default="outputs/perception-validation/isaac")
    p.add_argument("--scene", default="configs/scenes/lab_20261120.yaml")
    args, kit_args = p.parse_known_args()
    if args.episodes < 10: p.error("--episodes must be at least 10")
    launcher = AppLauncher({"headless": True, "device": "cpu", "experience": "/workspace/isaaclab/apps/isaaclab.python.kit"})
    try:
        result = run(args.episodes, args.seed, Path(args.output), args.scene)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    finally:
        # Kit shutdown can hang on headless GPU teardown; container process is dedicated to this run.
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__": main()
