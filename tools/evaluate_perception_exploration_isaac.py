#!/usr/bin/env python3
"""Paired Isaac wrist exploration evaluation using the shared image-only sweep planner."""
from __future__ import annotations

import argparse, csv, json, math, os, queue, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); os.chdir(ROOT)

import numpy as np
import yaml
from isaaclab.app import AppLauncher


def truth_snapshot(session):
    from agentic_sim.kinematics import quat_wxyz_to_mat
    from agentic_sim.perception.vertices import SIGNS
    rows=[]; colors=list(session.block_color_names)
    for block in session.blocks:
        p=block.data.root_pos_w[0].cpu().numpy().astype(float)
        q=block.data.root_quat_w[0].cpu().numpy().astype(float)
        rot=quat_wxyz_to_mat(q)
        yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]))%(math.pi/2)
        local=SIGNS.copy(); local[:,:2]*=session.edge/2; local[:,2]=(local[:,2]-.5)*session.edge
        rows.append({"center_xyz_m":p.tolist(),"yaw_rad":yaw,"corners_xyz_m":(local@rot.T+p).tolist(),"quat_wxyz":q.tolist()})
    return rows,colors


def jsonable(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.floating,)): return float(x)
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,dict): return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    return x


def run_episode(session,episode,seed,out,planner,metrics_api):
    from agentic_sim.isaac.live_perception import _Capture
    from agentic_sim.perception.live import LiveEstimator,rendered_camera
    from agentic_sim.perception.vertices import CameraGeometry
    from agentic_sim.scene_layout import tray_parts
    from omni.kit.app import get_app

    sweep=planner.plan_wrist_sweep(session.scene_cfg)
    accumulator=planner.WristEstimateAccumulator(gate_m=.025)
    floor=tray_parts(session.scene_cfg["trays"]["storage_pink"])[1]["floor_top"]
    estimator=LiveEstimator(session.edge,floor)
    baselines=[]; views=[]; all_rows=[]; rejected_total=[]; step_log=[]; motion=[]
    last_positions=None

    def sample_motion(event,segment=None):
        nonlocal last_positions
        positions=[]; speeds=[]
        for idx,b in enumerate(session.blocks):
            p=b.data.root_pos_w[0].cpu().numpy().astype(float)
            positions.append(p)
            vel=getattr(b.data,"root_vel_w",None)
            if vel is not None:
                v=vel[0,:3].cpu().numpy().astype(float)
                speeds.append((idx,float(np.linalg.norm(v))))
        max_speed=max((v for _,v in speeds),default=None)
        if max_speed is not None and max_speed>=.02:
            motion.append({"event":event,"segment":segment,"sim_time_s":float(session.t_sim),"max_block_speed_m_s":max_speed,
                           "blocks_over_0_02m_s":[i for i,v in speeds if v>=.02]})
        last_positions=np.asarray(positions)
        return max_speed

    def advance(event,segment=None):
        session.step()
        sample_motion(event,segment)
        get_app().update()

    def move_to(target,label,max_seg_m,settle=True):
        start=session.ctrl.state.pos.copy(); target=np.asarray(target,float)
        dist=float(np.linalg.norm(target-start)); n=max(1,int(math.ceil(dist/max_seg_m)))
        for k in range(1,n+1):
            waypoint=start+(target-start)*(k/n)
            if np.allclose(waypoint,session.ctrl.state.pos,atol=1e-9):
                continue
            if not session.ctrl.propose(waypoint,0.0,"exploration"):
                raise RuntimeError(f"episode {episode} target {label} waypoint {waypoint.tolist()} rejected: {session.ctrl.state.message}")
            advance("move",label)
        if not settle:
            step_log.append({"label":label,"target_xyz_m":target.tolist(),"segments_n":n,"settle_frames":0})
            return
        stable=0; qerror=None; eeerror=None
        for frame in range(1,121):
            advance("settle",label)
            q=session.robot.data.joint_pos[0,session.arm_ids].cpu().numpy()
            qerror=float(np.max(np.abs(q-session.ctrl.state.q_cmd)))
            ee=session.ctrl.chain.ee_pose_mount(q)[:3,3]
            eeerror=float(np.linalg.norm(ee-target))
            # The constrained IK targets a down-facing tool pose. Its reported
            # task-space residual is occasionally just above 8 mm while joint
            # error is already settled; allow 12 mm, still logged per waypoint.
            if qerror<.04 and eeerror<.012: stable+=1
            else: stable=0
            if stable>=10: break
        if stable<10:
            raise TimeoutError(f"episode {episode} timed out settling at {label}: qerr={qerror:.4f}, EEerr={eeerror*1000:.1f}mm")
        step_log.append({"label":label,"target_xyz_m":target.tolist(),"segments_n":n,"settle_frames":frame,
                         "max_q_error_rad":qerror,"actual_ee_error_mm":eeerror*1000})

    def capture(view_id):
        q=session.robot.data.joint_pos[0,session.arm_ids].cpu().numpy().astype(float)
        base=CameraGeometry.from_scene(session.scene_cfg,"wrist",q)
        inbox=queue.SimpleQueue(); cap=_Capture(inbox,0)
        session.sensor_views.windows["wrist"].viewport_api.schedule_capture(cap)
        start=time.monotonic(); message=None
        while time.monotonic()-start<45:
            session.sim.render(); get_app().update()
            if not inbox.empty():
                _,rgb,view,projection,frame,at,error=inbox.get()
                if error: raise RuntimeError(f"wrist ByteCapture failed: {error}")
                cam=rendered_camera(base,view,projection,rgb.shape[1],rgb.shape[0])
                p_err=float(np.linalg.norm(cam.mount_from_cv[:3,3]-base.mount_from_cv[:3,3]))
                rmat=cam.mount_from_cv[:3,:3]@base.mount_from_cv[:3,:3].T
                r_err=math.degrees(math.acos(float(np.clip((np.trace(rmat)-1)/2,-1,1))))
                if p_err>.01 or r_err>5: raise RuntimeError(f"wrist frame/FK mismatch {p_err*1000:.2f}mm,{r_err:.2f}deg")
                overlay,record=estimator.predict(rgb,cam,frame,at)
                record["camera_pose_check"]={"translation_error_mm":p_err*1000,"rotation_error_deg":r_err,"passed":True}
                return {"view_id":view_id,"rgb":rgb,"overlay":overlay,"record":record,"camera":cam,"captured_at_s":at}
        raise TimeoutError(f"episode {episode} timed out waiting for wrist capture {view_id}; {message}")

    def score_visible_points(captured,gt):
        # Center-in-frame is a post-capture GT coverage diagnostic. Projection
        # rejects points behind the camera; count only valid projected centers.
        count=0
        for row in gt:
            try: uv=captured["camera"].project(np.asarray([row["center_xyz_m"]],float))[0]
            except ValueError: continue
            count += int(0<=uv[0]<captured["camera"].width and 0<=uv[1]<captured["camera"].height)
        return count

    def save_capture(captured,stem):
        import cv2
        cv2.imwrite(str(out/f"{stem}_rgb.png"),cv2.cvtColor(captured["rgb"],cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out/f"{stem}_overlay.png"),cv2.cvtColor(captured["overlay"],cv2.COLOR_RGB2BGR))

    # Paired fixed wrist baseline; this is a new controlled baseline, not the prior home-pose metric.
    move_to([.10,.10,.15],"fixed_baseline",.005)
    baseline=capture("baseline"); baselines=baseline["record"]["detections"]
    gt_baseline,colors=truth_snapshot(session)
    rejected_base=[r for r in baselines if r["geometry"]["status"]!="estimated"]
    metrics_api.camera_metrics(baselines,rejected_base,gt_baseline,colors)
    save_capture(baseline,"baseline")
    all_rows.extend(baselines)
    # The paired accumulated policy may retain observations from its initial
    # baseline as well as the six exploration views.
    accumulator.add_view("baseline",baseline["captured_at_s"],baselines)

    # Raised transit, then six bounded observation positions from the shared planner.
    move_to([.16,.18,.18],"scan_entry",.005)
    move_to([.16,.18,sweep.transit_z_m],"transit_entry",.005)
    current=np.array([.16,.18,sweep.transit_z_m],float)
    for view_id,point in enumerate(sweep.points_xyz_m):
        point=np.asarray(point,float)
        move_to([current[0],current[1],sweep.transit_z_m],f"view_{view_id}_lift",min(.005,sweep.max_segment_m),settle=False)
        move_to([point[0],point[1],sweep.transit_z_m],f"view_{view_id}_transit",min(.005,sweep.max_segment_m),settle=False)
        move_to(point,f"view_{view_id}_observe",min(.005,sweep.max_segment_m))
        captured=capture(f"view_{view_id}"); views.append(captured)
        gt_view,_=truth_snapshot(session)
        rows=captured["record"]["detections"]
        rejected_total.extend(r for r in rows if r["geometry"]["status"]!="estimated")
        accumulator.add_view(f"view_{view_id}",captured["captured_at_s"],rows)
        save_capture(captured,f"view_{view_id:02d}")
        visible=score_visible_points(captured,gt_view)
        view_metrics=metrics_api.camera_metrics(rows,[r for r in rows if r["geometry"]["status"]!="estimated"],gt_view,colors)
        step_log.append({"label":f"view_{view_id}_capture","frame_number":captured["record"]["frame_number"],
                         "candidate_n":len(rows),"estimated_n":view_metrics["estimated_n"],
                         "matched_n":view_metrics["matched_n"],"projected_gt_centers_in_frame_n":visible,
                         "camera_pose_check":captured["record"]["camera_pose_check"],
                         "settled_gt_snapshot":gt_view,"detections":rows})
        all_rows.extend(rows)
        current=point.copy()

    gt_final,colors_final=truth_snapshot(session)
    baseline_metrics=metrics_api.camera_metrics(baselines,rejected_base,gt_baseline,colors)
    last=views[-1]["record"]["detections"]
    last_rejected=[r for r in last if r["geometry"]["status"]!="estimated"]
    last_metrics=metrics_api.camera_metrics(last,last_rejected,gt_final,colors_final)
    accumulated=accumulator.records()
    accum_metrics=metrics_api.camera_metrics(accumulated,rejected_total,gt_final,colors_final)
    bpos=np.asarray([g["center_xyz_m"] for g in gt_baseline]); fpos=np.asarray([g["center_xyz_m"] for g in gt_final])
    drift=np.linalg.norm(fpos-bpos,axis=1)
    return {"episode":episode,"seed":seed,"random_spawn":session.random_spawn,
        "planner":{"name":"plan_wrist_sweep","points_xyz_m":[list(p) for p in sweep.points_xyz_m],"transit_z_m":sweep.transit_z_m,
                   "observation_z_m":sweep.observation_z_m,"max_segment_m":sweep.max_segment_m,"view_n":len(views)},
        "baseline_definition":"controller-reached wrist pose [0.10,0.10,0.15] mount XYZ; distinct paired baseline, not prior 12/80 home-pose evaluation",
        "baseline":{"record":baseline["record"],"metrics":baseline_metrics},
        "last_frame":{"view_id":views[-1]["view_id"],"record":views[-1]["record"],"metrics":last_metrics},
        "accumulated":{"policy":"WristEstimateAccumulator best max fit_iou; spatial-only gate 25mm, no color/GT",
                        "summary":accumulator.summary(),"records":accumulated,"metrics":accum_metrics},
        "per_view":step_log,"settled_ground_truth":{"baseline":gt_baseline,"final":gt_final,"colors":colors_final},
        "block_drift":{"max_xyz_mm":float(drift.max()*1000) if len(drift) else 0.0,"per_block_xyz_mm":(drift*1000).tolist(),
                        "high_velocity_contact_proxy_n":len(motion),"threshold_m_s":.02,"events":motion,
                        "exact_contact_sensor_available":False},
        "motion":{"waypoints_and_settle":step_log,"final_actual_ee_xyz_m":session.ctrl.chain.ee_pose_mount(session.robot.data.joint_pos[0,session.arm_ids].cpu().numpy())[:3,3].tolist(),
                  "planner_max_segment_m":sweep.max_segment_m,"controller_joint_rate_limit_rad_s":session.ctrl.p.joint_rate_limit,
                  "qpos_teleport_used":False}}


def aggregate(episodes,mode):
    rows=[r for e in episodes for r in e[mode]["metrics"]["per_ground_truth"]]
    matches=[r for r in rows if r["status"]=="matched"]
    gt=sum(len(e["settled_ground_truth"]["final"]) for e in episodes)
    def stat(key):
        a=[r[key] for r in matches]
        if not a:return {"n":0,"mean":None,"median":None,"p95":None,"max":None}
        v=np.asarray(a,float);return {"n":len(v),"mean":float(v.mean()),"median":float(np.median(v)),"p95":float(np.percentile(v,95)),"max":float(v.max())}
    return {"ground_truth_n":gt,"matched_n":len(matches),"missed_n":sum(r["status"]=="missed" for r in rows),
            "false_positive_n":sum(r["status"]=="false_positive" for r in rows),
            "rejected_n":sum(e[mode]["metrics"]["rejected_n"] for e in episodes),
            "center_error_mm_matched":stat("center_error_mm"),"yaw_error_deg_mod90_matched":stat("yaw_error_deg_mod90"),
            "corner_rmse_mm_matched":stat("corner_rmse_mm"),"center_le_5mm_rate_over_all_gt":sum(r["center_error_mm"]<=5 for r in matches)/max(1,gt),
            "yaw_le_5deg_rate_over_all_gt":sum(r["yaw_error_deg_mod90"]<=5 for r in matches)/max(1,gt),
            "corner_le_5mm_rate_over_all_gt":sum(r["corner_rmse_mm"]<=5 for r in matches)/max(1,gt),
            "color_error_n_matched":sum(bool(r["color_error"]) for r in matches)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes",type=int,default=10); p.add_argument("--seed",type=int,default=20261121)
    p.add_argument("--output",default="outputs/perception-validation/isaac/exploration")
    p.add_argument("--scene",default="configs/scenes/lab_20261120.yaml")
    a,_=p.parse_known_args()
    if a.episodes<1: p.error("--episodes must be positive")
    launcher=AppLauncher({"headless":True,"device":"cpu","experience":"/workspace/isaaclab/apps/isaaclab.python.kit"})
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    try:
        from agentic_sim.isaac.session import SessionConfig,TeleopSession
        from agentic_sim.perception.exploration import plan_wrist_sweep,WristEstimateAccumulator
        from agentic_sim.perception.evaluation import camera_metrics
        scene=yaml.safe_load(Path(a.scene).read_text())
        rb=yaml.safe_load(Path("configs/random_blocks.yaml").read_text()); rb["count_range"]=[8,8]; rb["check_ik"]=True
        cfgpath=out/"random_blocks_eval.yaml";cfgpath.write_text(yaml.safe_dump(rb,sort_keys=False))
        eps=[]
        for i in range(a.episodes):
            print(f"[explore {i}] build session",flush=True)
            s=TeleopSession(SessionConfig(scene=a.scene,random_blocks=True,random_seed=a.seed,random_episode=i,
                    random_config=str(cfgpath),live_perception=False,device="cpu"),
                    log=lambda x:print(f"[explore {i}] {x}",flush=True))
            epdir=out/f"episode_{i:03d}";epdir.mkdir(exist_ok=True)
            try:
                # Allow randomized block contact settling with the home arm before scoring baseline.
                from omni.kit.app import get_app
                for _ in range(120): s.step();get_app().update()
                ep=run_episode(s,i,a.seed,epdir,type("API",(),{"plan_wrist_sweep":staticmethod(plan_wrist_sweep),"WristEstimateAccumulator":WristEstimateAccumulator})(),
                               type("Metrics",(),{"camera_metrics":staticmethod(camera_metrics)}))
                eps.append(ep)
                (epdir/"evaluation.json").write_text(json.dumps(jsonable(ep),indent=2))
                (epdir/"ground_truth.json").write_text(json.dumps(jsonable(ep["settled_ground_truth"]),indent=2))
                print(f"episode {i+1}/{a.episodes}: baseline {ep['baseline']['metrics']['matched_n']}/8, last {ep['last_frame']['metrics']['matched_n']}/8, accumulated {ep['accumulated']['metrics']['matched_n']}/8",flush=True)
            finally:
                s.close()
        summary={"engine":"Isaac Lab","seed":a.seed,"episodes_n":len(eps),"ground_truth_per_episode":8,
                 "planner":"plan_wrist_sweep; observations are RGB plus same-frame camera matrices only; GT is used after capture for scoring/drift",
                 "paired_metrics":{m:aggregate(eps,m) for m in ("baseline","last_frame","accumulated")},
                 "mean_block_drift_xyz_mm":float(np.mean([e["block_drift"]["max_xyz_mm"] for e in eps])),
                 "max_block_drift_xyz_mm":float(max(e["block_drift"]["max_xyz_mm"] for e in eps)),
                 "contact_measurement":"no ContactSensor configured; block root-velocity >=0.02m/s events are drift/contact proxies only",
                 "episodes":eps}
        (out/"evaluation.json").write_text(json.dumps(jsonable(summary),indent=2))
        fields=["episode","mode","gt_index","detection_id","status","center_error_mm","yaw_error_deg_mod90","corner_rmse_mm","predicted_color","ground_truth_color","color_error"]
        with (out/"evaluation.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for e in eps:
                for m in ("baseline","last_frame","accumulated"):
                    for r in e[m]["metrics"]["per_ground_truth"]:w.writerow({"episode":e["episode"],"mode":m,**r})
        lines=["# Isaac wrist exploration paired evaluation","",f"Episodes: {len(eps)}, seed: {a.seed}, eight blocks per episode.",
               "The shared wrist planner receives only configured tray geometry. View predictions consume wrist RGB and same-frame camera calibration. GT poses/colors are read separately after capture for scoring and drift only.",
               "Motion uses the real TeleopSession controller and rate-limited PhysX steps; no qpos teleport. Scan waypoints are interpolated to <=5 mm with max 120-frame convergence timeout.",
               "The accumulated mode includes the fixed-baseline capture followed by all six sweep captures. It retains estimates by predicted spatial association and best fit IoU; GT is not used for association or selection.",""]
        for m,vv in summary["paired_metrics"].items():lines += [f"## {m}","",f"GT {vv['ground_truth_n']}; matched {vv['matched_n']}; missed {vv['missed_n']}; FP {vv['false_positive_n']}; rejected {vv['rejected_n']}.",f"Center mean {vv['center_error_mm_matched']['mean']} mm; <=5mm/all GT {vv['center_le_5mm_rate_over_all_gt']:.3f}. Yaw mean {vv['yaw_error_deg_mod90_matched']['mean']} deg mod90; <=5deg/all GT {vv['yaw_le_5deg_rate_over_all_gt']:.3f}. Corner RMSE mean {vv['corner_rmse_mm_matched']['mean']} mm; <=5mm/all GT {vv['corner_le_5mm_rate_over_all_gt']:.3f}.",""]
        cap_views=[v for ep in eps for v in ep["per_view"] if v.get("label","").endswith("_capture")]
        projected=[v["projected_gt_centers_in_frame_n"] for v in cap_views]
        lines += ["## Motion and coverage","",f"Max block drift across episodes: {summary['max_block_drift_xyz_mm']:.3f} mm; root-speed contact proxies >=0.02m/s are recorded per episode. Exact contact sensor unavailable.",
                  f"Captured {len(cap_views)} sweep views. GT center-in-frame projection counts: {sum(projected)} / {8*len(projected)} total, mean {float(np.mean(projected)) if projected else 0:.2f}/8 per view, range {min(projected) if projected else 0}–{max(projected) if projected else 0}/8. This is projected coverage, not an occlusion measurement.",""]
        (out/"summary.md").write_text("\n".join(lines))
        print(json.dumps({k:v for k,v in summary.items() if k!="episodes"},indent=2),flush=True)
    except BaseException:
        import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    finally:
        sys.stdout.flush();sys.stderr.flush();os._exit(0)

if __name__=="__main__": main()
