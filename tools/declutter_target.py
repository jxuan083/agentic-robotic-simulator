#!/usr/bin/env python3
"""MuJoCo run of the declutter loop, then the target localization, on messy layouts.

Each episode: a random upright layout made messy (blocks stacked, leaning, pushed into contact; tools/
evaluate_height_check.py), a target colour, then up to max_moves rounds of
  global RGB + depth -> height check per colour segment -> rule (agentic_sim.task.declutter) -> pick and place on a
  free spot (scripted PickAt / Place, standing in for ACT; a Push when the pick fails) -> arm home
and finally the usual localization (global coarse, wrist refine). Perception and rules see only images; ground truth
is read only to name the block expected under the gripper (lift check, as the task executor does) and to score.

  MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/declutter_target.py --episodes 20 --seed 3
  ... --video              offscreen WebM per episode
  ... --target any         target colour drawn from all colours (default: a hidden one when there is one)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import yaml

import evaluate_perception_exploration as common
from evaluate_height_check import ground_truth, messify, render_global
from locate_target_color import VideoRecorder, _write_json, locate
from agentic_sim.config import load, v
from agentic_sim.mujoco_backend.live_perception import MjLivePerception
from agentic_sim.mujoco_backend.session import MjSession, SessionConfig
from agentic_sim.perception.exploration import score_target_confirmation
from agentic_sim.perception.height import backproject, height_check, polygon_gap
from agentic_sim.random_blocks import grasp_ik_checker
from agentic_sim.scene_layout import trays
from agentic_sim.task.declutter import (free_spot, grasp_yaw_offset, item_gap, item_key, items_from,
                                        localize_candidate, plan, push_path)
from agentic_sim.task.skills import Home, Motion, PickAt, Place, Push, World, wrap_quarter

TASK_CONFIG = "configs/task.yaml"


class Driver:
    """session.task stand-in: steps the current skill once per control frame (no scenario, no task panel)."""
    done = False
    scenario = None

    def __init__(self):
        self.skill = None
        self.ex = SimpleNamespace(panel_lines=lambda: ["declutter: " + (self.skill.phase if self.skill else "idle")],
                                  view=lambda: {"declutter_skill": self.skill.name if self.skill else None})

    def step(self, obs):
        if self.skill is not None:
            self.skill.step(obs)

    def close(self):
        pass


def _in_tray_xy(xy, tray, margin=0.01):
    cx, cy = tray["center"]
    c, s = math.cos(-tray["yaw"]), math.sin(-tray["yaw"])
    lx = c * (xy[0] - cx) - s * (xy[1] - cy)
    ly = s * (xy[0] - cx) + c * (xy[1] - cy)
    hx, hy = tray["inner_half"]
    return abs(lx) <= hx + margin and abs(ly) <= hy + margin


def perceive(session, manager, hc, noise_m, rng, tray):
    """Items in the storage tray from one global RGB + depth capture (segments elsewhere, e.g. the gripper, dropped)."""
    rgb, depth, cam, _ = render_global(manager, session, noise_m, rng)
    dets = manager.estimators["global"].split_detectors[1].predict(rgb).detections
    checks = [height_check(backproject(depth, d.mask, cam, hc.get("erode_px", 2)), manager.floor, manager.edge, hc,
                           seed=i) for i, d in enumerate(dets)]
    items = items_from([d.color for d in dets], checks)
    import cv2
    overlay = rgb.copy()                        # for the video panel: green = upright, red = a problem
    for d, c in zip(dets, checks):
        if c["status"] in ("side_face", "no_depth"):
            continue
        x0, y0, x1, y1 = (int(v_) for v_ in d.bbox_xyxy)
        col = (0, 170, 0) if c["status"] == "upright" else (230, 0, 0)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), col, 2)
        cv2.putText(overlay, c["status"], (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    manager.latest["global"] = {"record": None, "overlay_rgb": overlay}
    return [it for it in items if _in_tray_xy(it["xy"], tray)], rgb


def run_skill(session, driver, skill, status, timeout_s=40.0):
    driver.skill = skill
    t_end = session.t_sim + timeout_s
    label = getattr(session, "_target_search_ui", "") or ""
    label = label.split(" || ")[0]
    while skill.status == "running" and session.t_sim < t_end:
        status(f"{label} || {skill.name}: {skill.phase}")
        common._render_tick(session, None, skill.phase)
    driver.skill = None
    if skill.result is None:
        return {"skill": skill.name, "status": "failed", "failure_reason": "timeout"}
    return skill.result.to_dict()


def choose_target(colors, truth, rng, mode, close_gap):
    """hidden: a colour every block of which is covered (invisible to the global camera), else a colour with a
    problem block (stacked, tilted, covered or crowded), else any. any: any colour in the scene."""
    if mode == "hidden":
        visible = {colors[k] for k, t in enumerate(truth) if not t["covered"]}
        hidden = sorted({colors[k] for k, t in enumerate(truth) if t["covered"]} - visible)
        if hidden:
            return str(rng.choice(hidden)), "hidden"
        prob = set()
        for k, t in enumerate(truth):
            crowded = any(o != k and polygon_gap(t["footprint_xy_m"], u["footprint_xy_m"]) < close_gap
                          for o, u in enumerate(truth))
            if t["label"] != "upright" or t["covered"] or crowded:
                prob.add(colors[k])
        clean = {colors[k] for k, t in enumerate(truth) if t["label"] == "upright" and not t["covered"]
                 and all(polygon_gap(t["footprint_xy_m"], u["footprint_xy_m"]) >= close_gap
                         for o, u in enumerate(truth) if o != k)}
        prob -= clean                                   # a clean block of the colour needs no declutter
        if prob:
            return str(rng.choice(sorted(prob))), "problem"
    return str(rng.choice(sorted(set(colors)))), "any"


def run_episode(ep, args, cfg, task_cfg, reachable, random_path, out):
    rng = np.random.default_rng([args.seed, ep])
    session = MjSession(SessionConfig(scene=args.scene, random_blocks=True, random_seed=args.seed, random_episode=ep,
                                      random_config=str(random_path)), log=lambda *_: None)
    manager = MjLivePerception(session.scene_cfg, session.m, hz=2, log=lambda _: None, camera="both")
    session.live_perception = manager
    manager.last_capture["global"] = manager.last_capture["wrist"] = float("inf")
    edge, floor = manager.edge, manager.floor
    ep_dir = out / f"episode_{ep:03d}"
    tray = trays(session.scene_cfg)["storage_pink"]
    hc, dc = cfg["height_check"], cfg["declutter"]
    rules = dict(dc, close_gap_m=hc["close_gap_m"], hull_margin_m=hc.get("hull_margin_m", 0.0),
                 global_lookalikes=cfg.get("global_lookalikes", {}), edge_m=edge)
    base_xy = [float(x) for x in v(session.scene_cfg["robot"]["base_pos"])[:2]]
    recorder = None
    if args.video:
        recorder = session._frame_hook = VideoRecorder(session, manager, ep_dir / "video.webm")

    def status(text):
        session._target_search_ui = text

    try:
        for _ in range(60):
            common._render_tick(session, None, "settling")
        ops = messify(session, rng, int(rng.integers(1, args.max_ops + 1)), edge, floor, args.kinds.split(","))
        for _ in range(int(2.0 / session.control_dt)):
            common._render_tick(session, None, "settling")
        truth0 = ground_truth(session, edge, floor)
        colors = list(session.block_color_names)
        target, scenario = choose_target(colors, truth0, rng, args.target, hc["close_gap_m"])
        status(f"Target {target} ({scenario})")

        world = World(trays={"storage": tray}, edge=edge, colors=colors, init_pos=list(session.sc.block_init_pos),
                      home=np.asarray(session.ctrl.p.home_ee, float), slots=[])
        events = []
        ctx = SimpleNamespace(ctrl=session.ctrl, world=world, cfg=task_cfg,
                              motion=Motion(session.ctrl, task_cfg, session.control_dt),
                              emit=lambda kind, **d: events.append({"t": round(session.t_sim, 3), "kind": kind, **d}),
                              ledger=SimpleNamespace(revision=0), on_grasp=lambda b: None,
                              on_release=lambda b: None)
        driver = Driver()
        session.task = driver
        t0 = session.t_sim
        failed, excluded, confirmed, crowded_seen, rounds = {}, set(), set(), set(), []
        confirmed_pose = {}                     # item key -> wrist pose (centre, yaw) of a confirmed target
        decision, result, moves = None, None, 0
        if args.no_declutter:   # baseline: the localizer alone on the messy scene
            session.task = None
            result, _, _, _ = locate(session, manager, target, cfg, ep_dir, reachable, None, status)
            decision = {"action": "localize", "reason": "no declutter (baseline)"}
        for r in range(0 if args.no_declutter else dc["max_moves"] + dc.get("max_localize_attempts", 3) + 1):
            items, _ = perceive(session, manager, hc, args.depth_noise_mm_at_1m / 1000, rng, tray)
            decision = plan(items, target, rules, base_xy, failed, excluded, confirmed, crowded_seen)
            row = {"round": r, "items": [{k: it[k] for k in ("color", "status", "xy", "layers", "tilt_deg")}
                                         for it in items], "decision": decision}
            rounds.append(row)
            if decision["action"] == "localize":
                session.task = None
                cand = localize_candidate(items[decision["target"]], target, edge, floor, base_xy)
                result, _, _, _ = locate(session, manager, target, cfg, ep_dir, reachable, None, status,
                                         first_candidates=[cand])
                session.task = driver
                row["localize"] = {k: result.get(k) for k in ("status", "reason", "center_xyz_m")}
                run_skill(session, driver, Home(ctx), status)
                tkey = item_key(items[decision["target"]])
                if result["status"] not in ("found", "unreachable"):
                    crowded = decision["reason"] == "check the colour with the wrist" and any(
                        item_gap(items[decision["target"]], o, rules["hull_margin_m"]) < rules["close_gap_m"]
                        for j, o in enumerate(items) if j != decision["target"])
                    (crowded_seen if crowded and tkey not in crowded_seen else excluded).add(tkey)
                    crowded_seen.discard(tkey) if tkey in excluded else None
                    continue
                # the confirmed block (nearest item to the wrist pose): done if nothing needs moving around it
                c = np.asarray(result["center_xyz_m"][:2])
                k = int(np.argmin([np.linalg.norm(np.asarray(it["xy"]) - c) for it in items]))
                confirmed.add(item_key(items[k]))
                confirmed_pose[item_key(items[k])] = (list(result["center_xyz_m"]), float(result["yaw_rad_mod_pi_over_2"]))
                again = plan(items, target, rules, base_xy, failed, excluded, confirmed, crowded_seen)
                row["after_confirm"] = again
                if again["action"] == "localize" and again["target"] == k:
                    break
                continue
            if decision["action"] != "move":
                break
            if moves == dc["max_moves"]:
                decision = {"action": "gave_up", "reason": f"still blocked after {moves} moves"}
                row["decision"] = decision
                break
            moves += 1
            result = None              # a move can shift the target: an earlier localization is stale
            it = dict(items[decision["item"]])
            pos = [it["xy"][0], it["xy"][1], it["top_z_m"] - edge / 2]
            if item_key(it) in confirmed_pose:          # a confirmed target: its wrist pose (a merged item's centre
                pos, it["yaw_rad"] = confirmed_pose[item_key(it)]   # lies between blocks)
                it["status"] = "upright"
            # ground truth only for the lift check: the block nearest the perceived pose
            blocks = session.block_positions()
            truth_b = int(np.argmin([np.linalg.norm(np.asarray(b) - pos) for b in blocks]))   # 3D: stacks share xy
            obs = session.task_obs()
            row["pick_check"] = {"perceived_pos": [round(x, 4) for x in pos], "perceived_yaw_deg":
                                 round(math.degrees(it["yaw_rad"]), 1), "truth_pos": np.round(blocks[truth_b], 4).tolist(),
                                 "truth_yaw_deg_mod90": round(math.degrees(obs.block_yaw[truth_b]) % 90, 1)}
            avoid = [] if decision["target"] is None or decision["target"] == decision["item"] else \
                [items[decision["target"]]["xy"]]
            # the moved block's own place stays blocked: a hidden block may lie under it
            spot = free_spot(items, tray, edge, reachable, base_xy, avoid, dc["place_clearance_m"],
                             dc["place_avoid_target_m"])
            status(f"Target {target} | move {it['color']} ({decision['reason']})")
            acts = []
            pick = None
            ctrl = session.ctrl
            base_off = wrap_quarter(it["yaw_rad"] - ctrl.yaw_for(np.array([pos[0], pos[1], 0.0]), 0.0))
            safe_z, gdz, gap = (task_cfg["motion"][k] for k in ("safe_z", "grasp_dz", "fixed_finger_gap"))

            def feasible(o):   # IK (gripper tilt within the controller's limit) above and at the grasp point
                y = ctrl.yaw_for(np.array([pos[0], pos[1], 0.0]), o)
                g = np.asarray(pos[:2]) - gap * np.array([math.cos(y), math.sin(y)])
                return all(ctrl.chain.solve_ik_tilt(np.array([g[0], g[1], z]), y, ctrl.q_home, ctrl.p.max_tilt_deg)[1]
                           for z in (safe_z, pos[2] + gdz))

            off = grasp_yaw_offset(items, decision["item"], base_off,
                                   lambda o: ctrl.yaw_for(np.array([pos[0], pos[1], 0.0]), o),
                                   ctrl.p.yaw_offset_limit, edge, feasible, dc.get("finger_room_m", 0.018))
            row["grasp_yaw_offset_deg"] = None if off is None else round(math.degrees(off), 1)
            if spot is not None and off is not None and it["status"] != "merged":   # merged: centre between blocks
                pick = run_skill(session, driver, PickAt(ctx, block=truth_b, pos=pos, yaw=it["yaw_rad"],
                                                         yaw_offset=off), status)
                acts.append(pick)
                if pick["status"] == "succeeded":
                    acts.append(run_skill(session, driver, Place(ctx, block=truth_b, xy=spot, tray="storage"), status))
            if pick is not None and pick["status"] != "succeeded":
                acts.append(run_skill(session, driver, Home(ctx), status))   # a failed pick can leave the wrist turned
            if pick is None or pick["status"] != "succeeded":
                push_z = floor + edge / 2 + task_cfg["motion"]["grasp_dz"]

                def push_ok(a, b):
                    d = np.asarray(b) - np.asarray(a)
                    y = ctrl.yaw_for(np.array([a[0], a[1], 0.0]),
                                     wrap_quarter(math.atan2(d[1], d[0]) - ctrl.yaw_for(np.array([a[0], a[1], 0.0]), 0.0)))
                    return all(ctrl.chain.solve_ik_tilt(np.array([q[0], q[1], z]), y, ctrl.q_home, ctrl.p.max_tilt_deg)[1]
                               for q, z in ((a, safe_z), (a, push_z), (b, push_z)))
                path = push_path(items, decision["item"], tray, edge, decision["target"], dc["push_distance_m"],
                                 feasible=push_ok)
                if path is not None:
                    acts.append(run_skill(session, driver, Push(ctx, start=path[0], end=path[1], z=push_z), status))
                key = (round(it["xy"][0], 2), round(it["xy"][1], 2))
                failed[key] = failed.get(key, 0) + 1
            acts.append(run_skill(session, driver, Home(ctx), status))
            row.update(spot=spot, actions=acts)
        session.task = None
        declutter_s = session.t_sim - t0
        if result is None or result["status"] not in ("found", "unreachable"):
            gave_up = decision["action"] in ("gave_up", "move")    # "move": the rounds ran out after a move
            result = {"status": "gave_up" if gave_up else "not_found",
                      "reason": decision["reason"] if decision["action"] != "move" else "rounds ran out after a move",
                      "localize": result}
        result["declutter_rounds"] = len(rounds)
        result["moves"] = moves
        _write_json(ep_dir / "target.json", result)              # frozen before any ground truth is read
        if recorder is not None:
            common._dwell(session, None, 60, "done")
        # ---- scoring only from here on ----
        truth, tcolors = common._pose_truth(session)
        truth_after = ground_truth(session, edge, floor)
        found = result if result["status"] in ("found", "unreachable") else None
        score = score_target_confirmation(found, target, truth, tcolors)
        m = score["matched_gt_index"]
        if m is not None:   # corners as a set: a block rolled onto another face keeps its corners, not their order
            pv, tv = np.asarray(found["vertices_xyz_m"]), np.asarray(truth[m]["corners_xyz_m"])
            score["corner_set_rmse_mm"] = float(np.sqrt(np.mean(
                np.min(np.linalg.norm(pv[:, None] - tv[None], axis=2), axis=1) ** 2)) * 1000)
            score["pose_ok"] = bool(score["color_correct"] and score["center_error_mm"] <= 5
                                    and score["corner_set_rmse_mm"] <= 5)
        ok_block = None
        if m is not None:
            t = truth_after[m]
            gap = min((polygon_gap(t["footprint_xy_m"], u["footprint_xy_m"]) for o, u in enumerate(truth_after)
                       if o != m), default=1.0)
            ok_block = {"label": t["label"], "covered": t["covered"], "min_gap_mm": round(gap * 1000, 1),
                        "reachable": bool(reachable(truth[m]["center_xyz_m"], math.degrees(truth[m]["yaw_rad"])))}
        success = bool(result["status"] == "found" and score.get("pose_ok") and ok_block
                       and ok_block["label"] == "upright" and not ok_block["covered"]
                       and ok_block["min_gap_mm"] >= hc["close_gap_m"] * 1000 and ok_block["reachable"])
        out_of_tray = sum(1 for t in truth_after if t["label"] == "off_tray")
        acts = [a for rw in rounds for a in rw.get("actions", []) if a["skill"] != "home"]
        episode = {"episode": ep, "target_color": target, "scenario": scenario, "ops": ops, "result": result,
                   "rounds": rounds, "score": score, "target_block": ok_block, "success": success,
                   "declutter_sim_s": round(declutter_s, 2),
                   "picks": sum(a["skill"] == "pick_at" for a in acts),
                   "pick_failures": sum(a["skill"] == "pick_at" and a["status"] != "succeeded" for a in acts),
                   "places_uncertain": sum(a["skill"] == "place" and a["status"] != "succeeded" for a in acts),
                   "pushes": sum(a["skill"] == "push" for a in acts),
                   "blocks_out_of_tray": out_of_tray, "truth_before": truth0, "truth_after": truth_after,
                   "colors": colors}
        _write_json(ep_dir / "evaluation.json", episode)
        print(f"[episode {ep}] {target} ({scenario}): {result['status']} after {result['moves']} moves"
              f" -> success {success}", flush=True)
        return episode
    finally:
        if recorder is not None:
            session._frame_hook = None
            recorder.close()
        session.task = None
        manager.close()
        session.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--max-ops", type=int, default=3)
    p.add_argument("--kinds", default="stack,lean,touch")
    p.add_argument("--target", choices=("hidden", "any"), default="hidden")
    p.add_argument("--depth-noise-mm-at-1m", type=float, default=5.0)
    p.add_argument("--video", action="store_true")
    p.add_argument("--only", type=int, nargs="+", help="run just these episode indices (same scenes as in a full run)")
    p.add_argument("--no-declutter", action="store_true", help="baseline: localize directly, move nothing")
    p.add_argument("--scene", default="configs/scenes/lab_20261120.yaml")
    p.add_argument("--config", default="configs/target_localization.yaml")
    p.add_argument("--output", default="outputs/declutter/run")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    random_cfg = load("configs/random_blocks.yaml")
    random_cfg["count_range"] = [args.blocks, args.blocks]
    random_path = out / "random_blocks.yaml"
    random_path.write_text(yaml.safe_dump(random_cfg, sort_keys=False), encoding="utf-8")
    cfg, task_cfg = load(args.config), load(TASK_CONFIG)
    reachable = grasp_ik_checker(load(args.scene), "configs/teleop.yaml")
    eps = [run_episode(i, args, cfg, task_cfg, reachable, random_path, out) for i in (args.only or range(args.episodes))]
    by = {}
    for e in eps:
        by.setdefault(e["scenario"], []).append(e)
    summary = {
        "episodes_n": len(eps), "success_n": sum(e["success"] for e in eps),
        "by_scenario": {k: f"{sum(e['success'] for e in es)}/{len(es)}" for k, es in sorted(by.items())},
        "outcomes": {s: sum(e["result"]["status"] == s for e in eps)
                     for s in sorted({e["result"]["status"] for e in eps})},
        "moves_mean": round(float(np.mean([e["result"]["moves"] for e in eps])), 2),
        "picks": sum(e["picks"] for e in eps), "pick_failures": sum(e["pick_failures"] for e in eps),
        "pushes": sum(e["pushes"] for e in eps), "places_uncertain": sum(e["places_uncertain"] for e in eps),
        "blocks_out_of_tray": sum(e["blocks_out_of_tray"] for e in eps),
        "declutter_sim_s_mean": round(float(np.mean([e["declutter_sim_s"] for e in eps])), 1),
    }
    _write_json(out / "evaluation.json", {"args": vars(args), "summary": summary,
                                          "episodes": [{k: e[k] for k in ("episode", "target_color", "scenario",
                                                                          "success")} | {"status": e["result"]["status"],
                                                                                         "moves": e["result"]["moves"]}
                                                       for e in eps]})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
