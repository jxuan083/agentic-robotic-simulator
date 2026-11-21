#!/usr/bin/env python3
"""Exercise both-camera live perception JSON through reset and random scene rebuild."""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); os.chdir(ROOT)
from isaaclab.app import AppLauncher

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def wait_both(session,path,limit=300):
    from omni.kit.app import get_app
    for frame in range(limit):
        session.step()
        # Explicit Kit tick is required for hidden/headless full-UI viewports to dispatch ByteCapture callbacks.
        get_app().update()
        state=read(path)
        cameras=(state or {}).get("cameras",{})
        if all(cameras.get(k,{}).get("latest") is not None for k in ("global","wrist")):
            return frame+1,state
    raise TimeoutError(f"both cameras did not update JSON; latest state: {read(path)}")

def verify_reset_json(path):
    state=read(path)
    cameras=(state or {}).get("cameras",{})
    if not all(cameras.get(k,{}).get("latest") is None for k in ("global","wrist")):
        raise AssertionError(f"reset/rebuild retained stale camera estimates: {state}")
    return state

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="outputs/perception-validation/isaac/live-smoke.json")
    p.add_argument("--scene",default="configs/scenes/lab_20261120.yaml")
    p.add_argument("--seed",type=int,default=20261121)
    a,_=p.parse_known_args()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    launcher=AppLauncher({"headless":True,"device":"cpu","experience":"/workspace/isaaclab/apps/isaaclab.python.kit"})
    try:
        from agentic_sim.isaac.session import SessionConfig,TeleopSession
        target=out.with_name("live-perception.json")
        target.unlink(missing_ok=True)
        results={"mode":"Kit full UI experience, headless renderer; no desktop connection",
                 "perception_json_path":str(target),"camera":"both","steps":[]}
        session=None
        try:
            cfg=SessionConfig(scene=a.scene,random_blocks=True,random_seed=a.seed,random_episode=0,
                live_perception=True,perception_camera="both",perception_hz=2.0,perception_json=str(target),device="cpu")
            session=TeleopSession(cfg,log=lambda m:print("[smoke] "+m,flush=True))
            if set(session.live_perception.children)!={"global","wrist"}: raise AssertionError("both live camera workers were not created")
            if set(session.sensor_views.windows)!={"global","wrist"}: raise AssertionError("both sensor viewport windows were not created")
            n,state=wait_both(session,target)
            results["steps"].append({"event":"initial_estimates","frames":n,
                "camera_updates":{k:{"frame_number":state["cameras"][k]["latest"]["frame_number"],
                    "detections_n":len(state["cameras"][k]["latest"]["detections"]),
                    "error":state["cameras"][k]["error"]} for k in ("global","wrist")}})
            session.do_reset()
            cleared=verify_reset_json(target)
            results["steps"].append({"event":"operator_reset_invalidates_old_results","cleared":True,
                "camera_latest_null":{k:cleared["cameras"][k]["latest"] is None for k in ("global","wrist")}})
            n,state=wait_both(session,target)
            results["steps"].append({"event":"operator_reset_recovery","frames":n,
                "camera_updates":{k:{"frame_number":state["cameras"][k]["latest"]["frame_number"],
                    "detections_n":len(state["cameras"][k]["latest"]["detections"])} for k in ("global","wrist")}})
            session.close(); session=None
            # New randomized episode is a full stage / camera viewport rebuild, sharing only the JSON path.
            cfg=SessionConfig(scene=a.scene,random_blocks=True,random_seed=a.seed,random_episode=1,
                live_perception=True,perception_camera="both",perception_hz=2.0,perception_json=str(target),device="cpu")
            session=TeleopSession(cfg,log=lambda m:print("[rebuild] "+m,flush=True))
            startup_state=read(target)
            startup_cameras=(startup_state or {}).get("cameras",{})
            stale_on_start={k:startup_cameras.get(k,{}).get("latest") is not None for k in ("global","wrist")}
            results["steps"].append({"event":"random_episode_rebuild_startup_json","random_episode":1,
                "stale_estimate_present":stale_on_start})
            # Exercise the real Reset path after the new randomized episode is live.
            session.do_reset()
            cleared=verify_reset_json(target)
            results["steps"].append({"event":"random_episode_rebuild_then_operator_reset_clears_old_results","random_episode":1,
                "cleared":True,"camera_latest_null":{k:cleared["cameras"][k]["latest"] is None for k in ("global","wrist")}})
            n,state=wait_both(session,target)
            results["steps"].append({"event":"random_episode_rebuild_recovery","frames":n,
                "random_episode":1,"camera_updates":{k:{"frame_number":state["cameras"][k]["latest"]["frame_number"],
                    "detections_n":len(state["cameras"][k]["latest"]["detections"]),
                    "error":state["cameras"][k]["error"]} for k in ("global","wrist")}})
            results["status"]="passed"
        finally:
            if session is not None: session.close()
        out.write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2),flush=True)
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    finally:
        sys.stdout.flush();sys.stderr.flush();os._exit(0)

if __name__=="__main__": main()
