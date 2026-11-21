#!/usr/bin/env python3
"""Evaluate prediction-only target-color search with Isaac wrist-camera captures.

Ground truth is read only after search reaches ``found`` or ``exhausted``. The
target color is sampled from the episode's existing block color list, then the
search receives only that color and image-derived records.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import random
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); os.chdir(ROOT)

import numpy as np
import yaml
from isaaclab.app import AppLauncher

from evaluate_perception_exploration_isaac import jsonable,truth_snapshot


def _plain(value):
    """Convert the shared search result (dict/dataclass/simple object) to JSON."""
    if hasattr(value,"__dataclass_fields__"):
        from dataclasses import asdict
        value=asdict(value)
    elif hasattr(value,"__dict__"):
        value=vars(value)
    return jsonable(value)


def _is_found(value):
    if isinstance(value,dict):
        return value.get("found") is True or str(value.get("status","")).lower() in {"found","confirmed"}
    if isinstance(value,str): return value.lower() in {"found","confirmed"}
    return bool(getattr(value,"found",False)) or str(getattr(value,"status","")).lower() in {"found","confirmed"}


def _save_capture(out,captured,stem):
    """Write into this episode directory and fail loudly on invalid paths."""
    import cv2
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    paths=(out/f"{stem}_rgb.png",out/f"{stem}_overlay.png")
    images=(captured["rgb"],captured["overlay"])
    for path,rgb in zip(paths,images):
        if not cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)) or not path.is_file():
            raise OSError(f"could not save capture PNG: {path}")
    return [str(p.name) for p in paths]


def _score_target(search_result,observed,truth,colors,target_color,edge):
    """GT-only terminal audit; never called until the search is terminal."""
    from agentic_sim.perception.evaluation import camera_metrics
    rows=[r for view in observed for r in view["detections"]]
    target_rows=[r for r in rows if r.get("color")==target_color and r.get("geometry",{}).get("status")=="estimated"]
    rejected=[r for r in rows if r.get("geometry",{}).get("status")!="estimated"]
    scored=camera_metrics(target_rows,rejected,truth,colors)
    target_gt=sum(c==target_color for c in colors)
    found=_is_found(search_result)
    result=search_result if isinstance(search_result,dict) else _plain(search_result)
    confirmed=result.get("confirmed") if isinstance(result,dict) else None
    confirmed_score=None
    if isinstance(confirmed,dict) and confirmed.get("center_xyz_m") is not None and len(confirmed.get("vertices_xyz_m",[]))==8:
        geometry={"status":"estimated","center_xyz_m":confirmed["center_xyz_m"],
                  "yaw_rad_mod_pi_over_2":confirmed.get("yaw_rad_mod_pi_over_2",0.0),
                  "vertices":[{"id":i,"xyz_m":xyz} for i,xyz in enumerate(confirmed["vertices_xyz_m"])]}
        pseudo={"detection_id":"search-confirmation","color":target_color,"geometry":geometry}
        cm=camera_metrics([pseudo],[],truth,colors)
        assigned=next((r for r in cm["per_ground_truth"] if r["status"]=="matched"),None)
        if assigned is not None:
            confirmed_score={"gt_index":assigned["gt_index"],"true_color_match":assigned["ground_truth_color"]==target_color,
                             "center_error_mm":assigned["center_error_mm"],"yaw_error_deg_mod90":assigned["yaw_error_deg_mod90"],
                             "corner_rmse_mm":assigned["corner_rmse_mm"],"predicted_color":target_color,
                             "ground_truth_color":assigned["ground_truth_color"]}
    true_confirmation=bool(confirmed_score and confirmed_score["true_color_match"] and
                           confirmed_score["center_error_mm"]<=5.0 and confirmed_score["corner_rmse_mm"]<=5.0)
    return {"scoring_uses_ground_truth":True,"target_gt_count":target_gt,"target_estimated_record_n":len(target_rows),
            "target_true_positive_n":int(true_confirmation),"target_missed_n":max(0,target_gt-int(true_confirmation)),
            "confirmed_geometry_score":confirmed_score,
            "search_declared_found":found,"false_confirmation":bool(found and not true_confirmation),
            "post_score_success_rule":"confirmed target color matches GT, center error <=5mm, corner RMSE <=5mm",
            "search_result_center_error_gate_mm":5.0,"gate_note":"search gate is prediction-only; GT matching here audits true color, center, and all-corner geometry after termination"}


def run_episode(session,episode,seed,out,search_api,view_budget):
    from agentic_sim.isaac.live_perception import _Capture
    from agentic_sim.perception.live import LiveEstimator,rendered_camera
    from agentic_sim.perception.vertices import CameraGeometry
    from agentic_sim.scene_layout import tray_parts
    from omni.kit.app import get_app

    target=search_api.choose_existing_color(list(session.block_color_names),seed,episode)
    search=search_api.TargetColorSearch(target)
    planned_views=search_api.plan_target_search_views(session.scene_cfg)
    floor=tray_parts(session.scene_cfg["trays"]["storage_pink"])[1]["floor_top"]
    estimator=LiveEstimator(session.edge,floor)
    observed=[]; motion=[]; path=[]

    def advance(label):
        session.step(); get_app().update()
        speeds=[]
        for i,b in enumerate(session.blocks):
            vel=getattr(b.data,"root_vel_w",None)
            if vel is not None:
                speed=float(np.linalg.norm(vel[0,:3].cpu().numpy()))
                if speed>=.02: speeds.append({"block_index":i,"speed_m_s":speed})
        if speeds: motion.append({"label":label,"sim_time_s":float(session.t_sim),"high_speed_blocks":speeds})

    def move_to(target_xyz,label,settle=True):
        start=session.ctrl.state.pos.copy(); target_xyz=np.asarray(target_xyz,float)
        delta=target_xyz-start; n=max(1,int(math.ceil(np.linalg.norm(delta)/.005)))
        for k in range(1,n+1):
            waypoint=start+delta*(k/n)
            if np.allclose(waypoint,session.ctrl.state.pos,atol=1e-9): continue
            if not session.ctrl.propose(waypoint,0.0,"target_color_search"):
                raise RuntimeError(f"episode {episode} {label} rejected by controller IK: {session.ctrl.state.message}")
            advance(label)
        if settle:
            stable=0
            for frame in range(1,121):
                advance(label+"_settle")
                q=session.robot.data.joint_pos[0,session.arm_ids].cpu().numpy()
                qerr=float(np.max(np.abs(q-session.ctrl.state.q_cmd)))
                ee=session.ctrl.chain.ee_pose_mount(q)[:3,3]
                eerr=float(np.linalg.norm(ee-target_xyz))
                if qerr<.04 and eerr<.012: stable+=1
                else: stable=0
                if stable>=10: break
            if stable<10: raise TimeoutError(f"episode {episode} {label} convergence timeout: qerr={qerr:.4f}, EEerr={eerr*1000:.1f}mm")
        path.append({"label":label,"target_xyz_m":target_xyz.tolist(),"segments_n":n,"settled":settle,
                     "controller_rate_limit_rad_s":session.ctrl.p.joint_rate_limit,"qpos_teleport_used":False})

    def capture(view_id):
        q=session.robot.data.joint_pos[0,session.arm_ids].cpu().numpy().astype(float)
        base=CameraGeometry.from_scene(session.scene_cfg,"wrist",q)
        inbox=queue.SimpleQueue(); cap=_Capture(inbox,0)
        session.sensor_views.windows["wrist"].viewport_api.schedule_capture(cap)
        start=time.monotonic()
        while time.monotonic()-start<45:
            session.sim.render(); get_app().update()
            if inbox.empty(): continue
            _,rgb,view,projection,frame,at,error=inbox.get()
            if error: raise RuntimeError(f"ByteCapture failed: {error}")
            cam=rendered_camera(base,view,projection,rgb.shape[1],rgb.shape[0])
            perr=float(np.linalg.norm(cam.mount_from_cv[:3,3]-base.mount_from_cv[:3,3]))
            rmat=cam.mount_from_cv[:3,:3]@base.mount_from_cv[:3,:3].T
            rerr=math.degrees(math.acos(float(np.clip((np.trace(rmat)-1)/2,-1,1))))
            if perr>.01 or rerr>5: raise RuntimeError(f"camera/FK mismatch {perr*1000:.2f}mm,{rerr:.2f}deg")
            overlay,record=estimator.predict(rgb,cam,frame,at)
            origin=cam.mount_from_cv[:3,3].copy()
            record.update({"view_id":view_id,"camera":"wrist","captured_at_s":float(at),"mount_from_cv":cam.mount_from_cv.copy(),
                           "camera_origin_xyz_m":origin,"camera_pose_check":{"translation_error_mm":perr*1000,"rotation_error_deg":rerr,"passed":True}})
            return {"view_id":view_id,"rgb":rgb,"overlay":overlay,"record":record,"camera":cam,
                    "detections":record["detections"],"captured_at_s":float(at),"camera_origin_xyz_m":origin}
        raise TimeoutError(f"capture timeout at {view_id}")

    def observe_view(view_id):
        captured=capture(view_id)
        _save_capture(out,captured,f"{len(observed):02d}_{view_id}")
        observed.append({"view_id":view_id,"captured_at_s":captured["captured_at_s"],
                         "camera_origin_xyz_m":captured["camera_origin_xyz_m"].tolist(),
                         "camera_pose_check":captured["record"]["camera_pose_check"],
                         "record":captured["record"],"detections":captured["detections"]})
        result=search.observe(captured["record"])
        state=search.result()
        return {"observe_return":_plain(result),"search_result":_plain(state),"found":_is_found(state)}

    # Fixed paired view followed by the shared six-view sweep and four
    # configured elevated views. Sampling
    # color reads only the episode's colors; GT pose data is not inspected.
    move_to([.10,.10,.15],"baseline")
    status=observe_view("baseline")
    if not status["found"]:
        move_to([.16,.18,.18],"scan_entry")
        move_to([.16,.18,.24],"transit_entry")
        current=np.array([.16,.18,.24],float)
        for view_i,(planned_id,point) in enumerate(planned_views[:max(0,view_budget-1)]):
            point=np.asarray(point,float)
            move_to([current[0],current[1],.24],f"{planned_id}_lift",settle=False)
            move_to([point[0],point[1],.24],f"{planned_id}_transit",settle=False)
            move_to(point,f"{planned_id}_observe")
            status=observe_view(planned_id)
            if status["found"]: break
            current=point.copy()
    if not status["found"] and view_budget>1:
        search.exhaust("view_budget_exhausted")
        terminal=search.result()
    else: terminal=search.result()

    # Only now expose truth for scoring. It never enters planner, motion, or
    # search.observe; the sampled target color itself came from existing names.
    truth,colors=truth_snapshot(session)
    score=_score_target(terminal,observed,truth,colors,target,session.edge)
    return {"episode":episode,"seed":seed,"target_color":target,"target_color_source":"sampled from existing session.block_color_names; only target_color passed to search",
            "terminal_status":"found" if _is_found(terminal) else "exhausted","view_n":len(observed),"view_budget":view_budget,
            "search_result":_plain(terminal),"views":observed,
            "ground_truth_snapshot_post_terminal":{"truth":truth,"colors":colors},"post_terminal_score":score,
            "motion":{"waypoints":path,"high_velocity_contact_proxy_events":motion,"exact_contact_sensor_available":False,
                      "qpos_teleport_used":False}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes",type=int,default=10); p.add_argument("--seed",type=int,default=20261121)
    p.add_argument("--view-budget",type=int,default=11);p.add_argument("--output",default="outputs/perception-target-search/isaac-final")
    p.add_argument("--scene",default="configs/scenes/lab_20261120.yaml")
    a,_=p.parse_known_args()
    if a.episodes<1 or a.view_budget<1: p.error("episodes and view budget must be positive")
    launcher=AppLauncher({"headless":True,"device":"cpu","experience":"/workspace/isaaclab/apps/isaaclab.python.kit"})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    try:
        from agentic_sim.isaac.session import SessionConfig,TeleopSession
        from agentic_sim.perception import exploration as search_api
        scene=yaml.safe_load(Path(a.scene).read_text())
        rb=yaml.safe_load(Path("configs/random_blocks.yaml").read_text());rb["count_range"]=[8,8];rb["check_ik"]=True
        cfgpath=out/"random_blocks_eval.yaml";cfgpath.write_text(yaml.safe_dump(rb,sort_keys=False))
        eps=[]
        for i in range(a.episodes):
            print(f"[target-search {i}] build session",flush=True)
            s=TeleopSession(SessionConfig(scene=a.scene,random_blocks=True,random_seed=a.seed,random_episode=i,
                random_config=str(cfgpath),live_perception=False,device="cpu"),log=lambda x:print(f"[target-search {i}] {x}",flush=True))
            epdir=out/f"episode_{i:03d}";epdir.mkdir(parents=True,exist_ok=True)
            try:
                from omni.kit.app import get_app
                for _ in range(120): s.step();get_app().update()
                ep=run_episode(s,i,a.seed,epdir,search_api,a.view_budget)
                (epdir/"evaluation.json").write_text(json.dumps(jsonable(ep),indent=2))
                eps.append(ep)
                print(f"episode {i+1}/{a.episodes}: target={ep['target_color']} {ep['terminal_status']} after {ep['view_n']} views; false-confirm={ep['post_terminal_score']['false_confirmation']}",flush=True)
            finally:s.close()
        summary={"engine":"Isaac Lab","seed":a.seed,"episodes_n":len(eps),"view_budget":a.view_budget,
                 "found_n":sum(e["terminal_status"]=="found" for e in eps),"exhausted_n":sum(e["terminal_status"]=="exhausted" for e in eps),
                 "false_confirmation_n":sum(e["post_terminal_score"]["false_confirmation"] for e in eps),
                 "true_target_present_n":sum(e["post_terminal_score"]["target_gt_count"]>0 for e in eps),
                 "confirmed_center_error_mm_mean":float(np.mean([e["post_terminal_score"]["confirmed_geometry_score"]["center_error_mm"] for e in eps if e["post_terminal_score"]["confirmed_geometry_score"]])) if any(e["post_terminal_score"]["confirmed_geometry_score"] for e in eps) else None,
                 "confirmed_corner_rmse_mm_mean":float(np.mean([e["post_terminal_score"]["confirmed_geometry_score"]["corner_rmse_mm"] for e in eps if e["post_terminal_score"]["confirmed_geometry_score"]])) if any(e["post_terminal_score"]["confirmed_geometry_score"] for e in eps) else None,
                 "mean_views":float(np.mean([e["view_n"] for e in eps])) if eps else 0,
                 "sampling":"target color selected from each live episode's block_color_names; search receives color only",
                 "scoring":"GT snapshot read after terminal status only; used for target true color and center/corner audit",
                 "episodes":eps}
        (out/"evaluation.json").write_text(json.dumps(jsonable(summary),indent=2))
        with (out/"evaluation.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["episode","target_color","status","views","target_gt_count","true_positive_n","false_confirmation","confirmed_center_error_mm","confirmed_corner_rmse_mm"]);w.writeheader()
            for e in eps:
                cs=e["post_terminal_score"]["confirmed_geometry_score"] or {}
                w.writerow({"episode":e["episode"],"target_color":e["target_color"],"status":e["terminal_status"],"views":e["view_n"],"target_gt_count":e["post_terminal_score"]["target_gt_count"],"true_positive_n":e["post_terminal_score"]["target_true_positive_n"],"false_confirmation":e["post_terminal_score"]["false_confirmation"],"confirmed_center_error_mm":cs.get("center_error_mm"),"confirmed_corner_rmse_mm":cs.get("corner_rmse_mm")})
        lines=["# Isaac target-color search evaluation","",f"Episodes {len(eps)}, seed {a.seed}; found {summary['found_n']}; exhausted {summary['exhausted_n']}; false confirmations {summary['false_confirmation_n']} / {summary['found_n']} found; mean views {summary['mean_views']:.2f}.",
               f"For found targets, GT center error mean {summary['confirmed_center_error_mm_mean']:.3f} mm and eight-corner symmetry RMSE mean {summary['confirmed_corner_rmse_mm_mean']:.3f} mm; success requires true color and each error <=5 mm.",
               "Target color was sampled from the current episode's existing block colors. Search used target color and wrist RGB-derived records; GT was read only after found/exhausted. A found result requires two distinct views with accepted same-color geometry, center agreement <=5mm, and all eight vertices (shared search contract).", "",
               f"Each PNG is checked after write under its episode directory. Exact contact sensing is unavailable; root-speed proxies are recorded.",""]
        for e in eps:lines.append(f"- Episode {e['episode']}: {e['target_color']} {e['terminal_status']} in {e['view_n']}/{a.view_budget} views; target GT={e['post_terminal_score']['target_gt_count']}; false confirmation={e['post_terminal_score']['false_confirmation']}.")
        (out/"summary.md").write_text("\n".join(lines)+"\n")
        print(json.dumps({k:v for k,v in summary.items() if k!="episodes"},indent=2),flush=True)
    except BaseException:
        import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    finally:sys.stdout.flush();sys.stderr.flush();os._exit(0)

if __name__=="__main__":main()
