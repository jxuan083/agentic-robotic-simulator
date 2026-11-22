"""Acquire: find a block of one colour with the cameras, clearing the tray where needed, until it can be grasped.

  sk = Acquire(executor, color="red", purpose="deliver")   # executor.cameras, .loc_cfg, .reachable must be set
  each frame: sk.step(obs)
  sk.result.evidence["outcome"]:
    {"status": "found", "center_xyz_m", "yaw_rad", "grasp_yaw_offset", "block", ...}   pose for the grasp skill
    {"status": "holding", "block"}          the target was moved for the declutter and the wrist saw its colour in
                                            the gripper: it is still held, ready to be delivered
    {"status": "unreachable" | "not_found" | "gave_up", "reason"}
  avoid_grasp={"pos", "yaw_offset"}: a grasp of this colour that failed there (the gripper rested on a neighbour or
  the tray wall, or the block slipped): when the target is found at the same place again, the other finger
  orientation is used, or, when there is none, the target is pushed clear first

Each round looks again with the global camera (RGB + depth height check, perception.height), lets the declutter
rules (task.declutter.plan) pick one action and does it: a wrist look at the target (the coarse-to-fine localization
of perception.localize), or one move of a block (pick and set down in a free spot of the storage tray, push when the
pick fails). The arm returns home after every move so the next global look is not blocked. Sub-skills (pick_at,
place, push, home) run inside this skill, frame by frame.

A cancel (order changed) is taken at once while looking or moving the wrist camera, otherwise after the current
move (a block in the gripper is set down first).

Simulation truth: only cameras.block_id_near() (which block a pose names, for the executor's bookkeeping and the
lift check log). Positions, colours and decisions come from the cameras.
"""
from __future__ import annotations

import math

import numpy as np

from agentic_sim.perception.height import backproject, height_check
from agentic_sim.perception.localize import global_candidates, plan_wrist_view, refine_with_wrist, target_pose
from agentic_sim.perception.vertices import CameraGeometry
from agentic_sim.task.declutter import (free_spot, grasp_yaw_offset, in_tray_xy, item_gap, item_key, items_from,
                                        localize_candidate, mount_conflicts, plan, push_path)
from agentic_sim.task.skills import Home, Place, PickAt, Push, Skill, _Cancel, _Fail, wrap_quarter

TRANSIT_Z = 0.13          # wrist views: travel at this height, then descend (as tools/locate_target_color Route)


class Acquire(Skill):
    name = "acquire"
    cancel_phases = ("look", "view")

    def __init__(self, ctx, **params):
        super().__init__(ctx, **params)
        self.sub = None
        self.rounds: list = []

    # ---- cancel: at once while looking, otherwise between moves -----------------------------------------------
    def cancel(self, reason: str, revision: int) -> bool:
        if self.status != "running":
            return False
        self.cancel_reason, self.revision = reason, revision
        return True

    @property
    def cancellable(self) -> bool:
        return self.status == "running"

    def _between_moves(self):
        if self.cancel_reason is not None:
            raise _Cancel()

    def on_cancel(self):
        yield from ()

    # ---- helpers ----------------------------------------------------------------------------------------------------
    def run(self, sk):
        """Drive a sub-skill to its end, one frame per yield. Returns its result dict."""
        self.sub = sk
        while True:
            sk.step(self.obs)
            if sk.status != "running":
                break
            yield
        self.sub = None
        r = sk.result.to_dict()
        if sk.name == "pick_at":
            self.holding = sk.result.holding if sk.status == "succeeded" else None
        elif sk.name == "place" and sk.status == "succeeded":
            self.holding = None
        return r

    def say(self, text: str):
        self.ctx.status_text = f"{self.params['color']}: {text}"

    def perceive(self):
        """Items in the storage tray from one global RGB + depth capture."""
        import cv2
        cams, hc = self.ctx.cameras, self.ctx.loc_cfg["height_check"]
        rgb, depth, cam = cams.global_rgbd()
        dets = cams.segment_global(rgb)
        checks = [height_check(backproject(depth, d.mask, cam, hc.get("erode_px", 2)), cams.floor, cams.edge, hc,
                               seed=i) for i, d in enumerate(dets)]
        overlay = rgb.copy()                       # panel: green = upright, red = a problem
        for d, c in zip(dets, checks):
            if c["status"] in ("side_face", "no_depth"):
                continue
            x0, y0, x1, y1 = (int(v) for v in d.bbox_xyxy)
            col = (0, 170, 0) if c["status"] == "upright" else (230, 0, 0)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), col, 2)
            cv2.putText(overlay, c["status"], (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                        cv2.LINE_AA)
        cams.show("global", overlay)
        tray = self.world.trays["storage"]
        return [it for it in items_from([d.color for d in dets], checks) if in_tray_xy(it["xy"], tray)]

    def go_view(self, ee):
        """Wrist view pose: up to the transit height, across, down (yaw offset 0, as the view was planned)."""
        s = self.ctrl.state
        z = max(TRANSIT_Z, float(s.pos[2]), float(ee[2]))
        for p in ([s.pos[0], s.pos[1], z], [ee[0], ee[1], z], ee):
            yield from self.move(p, 0.0)

    def localize(self, first: dict):
        """Coarse-to-fine localization (as tools/locate_target_color.locate), `first` tried before the global
        camera's own candidates. Returns found / unreachable (wrist-confirmed pose) or not_found."""
        cams, cfg, color = self.ctx.cameras, self.ctx.loc_cfg, self.params["color"]
        self.set_phase("look")
        global_record = cams.global_record()
        cands = [first] + [c for c in global_candidates(global_record, color, cfg, self.ctx.base_xy)
                           if np.hypot(c["center_xyz_m"][0] - first["center_xyz_m"][0],
                                       c["center_xyz_m"][1] - first["center_xyz_m"][1]) > 0.02]
        out_of_reach, views = None, 0
        for ci, cand in enumerate(cands):
            confirmed = color_only = None
            for z in cfg["view_ee_z"]:
                plan_ = plan_wrist_view(self.ctrl.chain, self.ctrl.p, cams.scene, cand["center_xyz_m"], z,
                                        self.ctrl.state.q_goal)
                if plan_ is None:
                    continue
                self.set_phase("view")
                self.say(f"wrist look at candidate {ci + 1}/{len(cands)} ({cand['global_color']})")
                try:
                    yield from self.go_view(plan_["ee_xyz_m"])
                except _Fail:
                    continue
                yield from self.wait(int(cfg.get("settle_frames", 30)) * self.ctx.dt)
                record, rgb = cams.wrist()
                views += 1
                verdict = refine_with_wrist(record, cand, color, cfg, rgb)
                if verdict["status"] == "confirmed":
                    confirmed = (verdict, record)
                    break
                if verdict["status"] == "color_only" and color_only is None:
                    color_only = (verdict, record)
                if verdict["status"] == "wrong_color":
                    color_only = None
                    break
            hit = confirmed or color_only
            if hit is None:
                continue
            res = target_pose(hit[0], cand, hit[1], global_record)
            yaw = res["yaw_rad_mod_pi_over_2"] or 0.0
            res["graspable"] = bool(self.ctx.reachable(res["center_xyz_m"], math.degrees(yaw)))
            res["views"] = views
            if res["graspable"]:
                return res
            if out_of_reach is None:
                out_of_reach = dict(res, status="unreachable",
                                    reason="the wrist confirmed the colour, but the block is out of grasp reach")
        if out_of_reach is not None:
            return out_of_reach
        return {"status": "not_found", "views": views,
                "reason": f"{len(cands)} candidate(s), none confirmed by the wrist camera"}

    def mount_points(self, q):
        """Sample points of the wrist camera mount (mount frame) for joint angles q (forward kinematics)."""
        lo, hi = (np.asarray(c, float) for c in self.ctx.loc_cfg["declutter"]["wrist_mount_box_cv_m"])
        g = np.stack(np.meshgrid(*[np.linspace(a, b, n) for a, b, n in zip(lo, hi, (4, 5, 3))]), -1).reshape(-1, 3)
        t = CameraGeometry.from_scene(self.ctx.cameras.scene, "wrist", joints_rad=q).mount_from_cv
        return g @ t[:3, :3].T + t[:3, 3]

    def grasp_ik(self, pos, o):
        """IK at the grasp point for yaw offset o: joint angles or None."""
        ctrl, m = self.ctrl, self.cfg["motion"]
        y = ctrl.yaw_for(np.array([pos[0], pos[1], 0.0]), o)
        g = np.asarray(pos[:2]) - float(m["fixed_finger_gap"]) * np.array([math.cos(y), math.sin(y)])
        q, ok, _ = ctrl.chain.solve_ik_tilt(np.array([g[0], g[1], pos[2] + float(m["grasp_dz"])]), y, ctrl.q_home,
                                            ctrl.p.max_tilt_deg)
        return q if ok else None

    def grasp_offset(self, items, i, pos, yaw, exclude=None):
        """Finger orientation: the smallest turn, unless the fingers lack room there (task.declutter). An orientation
        needs IK above and at the grasp point, and the wrist camera mount clear of other blocks there; `exclude`
        (an offset that failed) is not used again."""
        ctrl, m = self.ctrl, self.cfg["motion"]
        safe_z, gap = float(m["safe_z"]), float(m["fixed_finger_gap"])
        heading = lambda o: ctrl.yaw_for(np.array([pos[0], pos[1], 0.0]), o)
        margin = float(self.ctx.loc_cfg["declutter"].get("mount_margin_m", 0.003))

        def feasible(o):
            if exclude is not None and abs(wrap_quarter(o - exclude)) < 0.3 and abs(o - exclude) < 0.8:
                return False
            y = heading(o)
            g = np.asarray(pos[:2]) - gap * np.array([math.cos(y), math.sin(y)])
            q = self.grasp_ik(pos, o)
            return q is not None and \
                ctrl.chain.solve_ik_tilt(np.array([g[0], g[1], safe_z]), y, ctrl.q_home, ctrl.p.max_tilt_deg)[1] and \
                not mount_conflicts(items, i, self.mount_points(q), margin)

        base = wrap_quarter(yaw - heading(0.0))
        return grasp_yaw_offset(items, i, base, heading, ctrl.p.yaw_offset_limit, self.world.edge, feasible,
                                self.ctx.loc_cfg["declutter"].get("finger_room_m", 0.018))

    def mount_blocker(self, items, k, result):
        """When no finger orientation clears the wrist camera mount at the grasp of target k (or only a turn over
        mount_big_offset_rad does): the nearest movable block under the mount (smallest-turn orientation), else None."""
        it = dict(items[k], xy=list(result["center_xyz_m"][:2]), yaw_rad=float(result["yaw_rad_mod_pi_over_2"] or 0.0))
        items = items[:k] + [it] + items[k + 1:]
        off = self.grasp_offset(items, k, result["center_xyz_m"], it["yaw_rad"])
        big = float(self.ctx.loc_cfg["declutter"].get("mount_big_offset_rad", 1.0))
        if off is not None and abs(off) <= big:   # a large turn puts the gripper body on the block: clear instead
            return None
        base = wrap_quarter(it["yaw_rad"] - self.ctrl.yaw_for(np.array([it["xy"][0], it["xy"][1], 0.0]), 0.0))
        q = self.grasp_ik(result["center_xyz_m"], base)
        if q is None:
            return None
        hits = mount_conflicts(items, k, self.mount_points(q),
                               float(self.ctx.loc_cfg["declutter"].get("mount_margin_m", 0.003)))
        hits = [j for j in hits if items[j]["status"] in ("upright", "stacked", "tilted", "off_layer")]
        return min(hits, key=lambda j: np.hypot(*(np.asarray(items[j]["xy"]) - np.asarray(it["xy"])))) if hits else None

    def held_color(self, block_from_ee, yaw_rad, shrink: float = 0.5):
        """Colour of the block in the gripper from one wrist capture: the cube expected at the EE (FK of the commanded
        joints) plus its offset at the grasp, projected into the wrist image and shrunk toward its centre (the
        fingers, yellow themselves, frame it)."""
        import cv2
        from agentic_sim.perception.classical import ClassicalConfig
        from agentic_sim.perception.color import estimate_color
        from agentic_sim.perception.localize import CLASSICAL_CONFIG, _record_camera
        record, rgb = self.ctx.cameras.wrist()
        ee = self.ctrl.chain.ee_pose_mount(self.ctrl.state.q_goal)[:3, 3]
        c, h = ee + np.asarray(block_from_ee, float), self.world.edge / 2
        rot = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0], [math.sin(yaw_rad), math.cos(yaw_rad), 0],
                        [0, 0, 1]])
        corners = c + np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)]) @ rot.T
        px = np.asarray(_record_camera(record).project(corners), np.float32)
        hull = cv2.convexHull(px).reshape(-1, 2)
        hull = hull.mean(0) + (1 - shrink) * (hull - hull.mean(0))
        mask = np.zeros(rgb.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
        if not mask.any():
            return None
        return estimate_color(rgb, mask.astype(bool), ClassicalConfig.from_yaml(CLASSICAL_CONFIG).color)

    # ---- the loop ---------------------------------------------------------------------------------------------------
    def body(self):
        cfg, color = self.ctx.loc_cfg, self.params["color"]
        hc, dc = cfg["height_check"], cfg["declutter"]
        edge, cams = self.world.edge, self.ctx.cameras
        tray, base_xy = self.world.trays["storage"], self.ctx.base_xy
        rules = dict(dc, close_gap_m=hc["close_gap_m"], hull_margin_m=hc.get("hull_margin_m", 0.0),
                     global_lookalikes=cfg.get("global_lookalikes", {}), edge_m=edge)
        m = self.cfg["motion"]
        failed, excluded, confirmed, crowded_seen = {}, set(), set(), set()
        confirmed_pose = {}                     # item key -> wrist pose (centre, yaw) of a confirmed target
        moves, result, decision, items, found_at = 0, None, None, [], None
        avoid, exclude = self.params.get("avoid_grasp"), None
        self.evidence["retry"] = None
        self.evidence["rounds"] = self.rounds
        for r in range(int(dc["max_moves"]) + int(dc.get("max_localize_attempts", 3)) + 1):
            self._between_moves()
            if float(np.linalg.norm(self.ctrl.state.pos - self.world.home)) > 1e-3:
                self.set_phase("home")
                yield from self.run(Home(self.ctx))   # the arm out of the global camera's view
            self.set_phase("look")
            self.say("global look")
            items = self.perceive()
            self.evidence["seen"] = {c: sum(it["color"] == c for it in items) for c in {it["color"] for it in items}}
            decision = plan(items, color, rules, base_xy, failed, excluded, confirmed, crowded_seen)
            row = {"round": r, "decision": decision,
                   "items": [{k: it[k] for k in ("color", "status", "xy", "layers")} for it in items]}
            self.rounds.append(row)
            if decision["action"] == "localize":
                t = decision["target"]
                self.say(f"localize ({decision['reason']})")
                result = yield from self.localize(localize_candidate(items[t], color, edge, cams.floor, base_xy))
                row["localize"] = {k: result.get(k) for k in ("status", "reason", "center_xyz_m")}
                tkey = item_key(items[t])
                if result["status"] not in ("found", "unreachable"):
                    crowded = decision["reason"] == "check the colour with the wrist" and any(
                        item_gap(items[t], o, rules["hull_margin_m"]) < rules["close_gap_m"]
                        for j, o in enumerate(items) if j != t)
                    if crowded and tkey not in crowded_seen:
                        crowded_seen.add(tkey)
                    else:
                        crowded_seen.discard(tkey)
                        excluded.add(tkey)
                    continue
                c = np.asarray(result["center_xyz_m"][:2])
                k = int(np.argmin([np.linalg.norm(np.asarray(it["xy"]) - c) for it in items]))
                confirmed.add(item_key(items[k]))
                confirmed_pose[item_key(items[k])] = (list(result["center_xyz_m"]),
                                                      float(result["yaw_rad_mod_pi_over_2"] or 0.0))
                found_at = (items, k)
                again = plan(items, color, rules, base_xy, failed, excluded, confirmed, crowded_seen)
                row["after_confirm"] = again
                if again["action"] == "localize" and again["target"] == k:
                    blocker = self.mount_blocker(items, k, result) if result["status"] == "found" else None
                    if (blocker is None or moves == int(dc["max_moves"])) and result["status"] == "found" and \
                            avoid is not None and avoid.get("yaw_offset") is not None and np.linalg.norm(
                                np.asarray(result["center_xyz_m"][:2]) - np.asarray(avoid["pos"][:2])) < 0.015:
                        # the last grasp failed here: another finger orientation, else push the target clear
                        it = dict(items[k], xy=list(result["center_xyz_m"][:2]),
                                  yaw_rad=float(result["yaw_rad_mod_pi_over_2"] or 0.0))
                        alt = self.grasp_offset(items[:k] + [it] + items[k + 1:], k, result["center_xyz_m"],
                                                it["yaw_rad"], exclude=float(avoid["yaw_offset"]))
                        if alt is not None or moves == int(dc["max_moves"]):
                            exclude = float(avoid["yaw_offset"]) if alt is not None else None
                            self.evidence["retry"] = "other finger orientation" if alt is not None else None
                            break
                        # no other orientation: move the nearest neighbour (the gripper rested on it), or, with none
                        # close, push the target towards the tray centre (the gripper rested on the wall)
                        reach = float(dc.get("retry_neighbour_m", 0.03))
                        near = [j for j, o in enumerate(items)
                                if j != k and item_gap(items[k], o, rules["hull_margin_m"]) < reach]
                        if near:
                            j = min(near, key=lambda j: item_gap(items[k], items[j], rules["hull_margin_m"]))
                            decision = {"action": "move", "item": j, "target": k,
                                        "reason": "the last grasp failed here: move the nearest neighbour"}
                        else:
                            decision = {"action": "move", "item": k, "target": k,
                                        "reason": "the last grasp failed here: push the target to the tray centre"}
                        row["decision_after_confirm"] = decision
                        self.evidence["retry"] = "neighbour" if near else "push"
                        moves += 1
                        avoid = result = found_at = None
                        done = yield from self.move_item(items, decision, confirmed_pose, failed, rules, row,
                                                         push_only=not near)
                        if done is not None:
                            result = done
                            break
                        continue
                    if blocker is None or moves == int(dc["max_moves"]):
                        break
                    # the grasp would put the wrist camera mount on a block: move that block first
                    decision = {"action": "move", "item": blocker, "target": k,
                                "reason": "under the wrist camera mount at the grasp"}
                    row["decision_after_confirm"] = decision
                    moves += 1
                    result = found_at = None
                    done = yield from self.move_item(items, decision, confirmed_pose, failed, rules, row)
                    if done is not None:
                        result = done
                        break
                continue
            if decision["action"] != "move":
                break
            if moves == int(dc["max_moves"]):
                decision = {"action": "gave_up", "reason": f"still blocked after {moves} moves"}
                row["decision"] = decision
                break
            moves += 1
            result = found_at = None            # a move can shift the target: an earlier localization is stale
            done = yield from self.move_item(items, decision, confirmed_pose, failed, rules, row)
            if done is not None:
                result = done
                break
        self.set_phase("done")
        if result is None or result["status"] not in ("found", "unreachable", "holding"):
            gave_up = decision is not None and decision["action"] in ("gave_up", "move")
            result = {"status": "gave_up" if gave_up else "not_found",
                      "reason": (decision or {}).get("reason", "") if not (decision and decision["action"] == "move")
                      else "rounds ran out after a move"}
        if result["status"] == "found":         # the grasp: block id (bookkeeping) and finger orientation
            items, k = found_at
            it = dict(items[k], xy=list(result["center_xyz_m"][:2]),
                      yaw_rad=float(result["yaw_rad_mod_pi_over_2"] or 0.0))
            result["grasp_yaw_offset"] = self.grasp_offset(items[:k] + [it] + items[k + 1:], k,
                                                           result["center_xyz_m"], it["yaw_rad"], exclude)
            result["block"] = cams.block_id_near(result["center_xyz_m"])
        result["moves"] = moves
        self.evidence["outcome"] = result
        if result["status"] not in ("found", "holding"):
            raise _Fail(f"{result['status']}: {result.get('reason', '')}")

    def move_item(self, items, decision, confirmed_pose, failed, rules, row, push_only: bool = False):
        """Move one block (pick and set down, or push; only push with push_only). Returns the outcome when the target
        itself ends up held and the wrist saw its colour in the gripper, else None."""
        cams, dc, edge = self.ctx.cameras, self.ctx.loc_cfg["declutter"], self.world.edge
        tray, base_xy, m = self.world.trays["storage"], self.ctx.base_xy, self.cfg["motion"]
        color = self.params["color"]
        i, t = decision["item"], decision["target"]
        it = dict(items[i])
        pos = [it["xy"][0], it["xy"][1], it["top_z_m"] - edge / 2]
        if item_key(it) in confirmed_pose:      # a confirmed target: its wrist pose (a merged item's centre lies
            pos, it["yaw_rad"] = confirmed_pose[item_key(it)]   # between blocks)
            it["status"] = "upright"
        b = cams.block_id_near(pos)
        avoid = [] if t is None or t == i else [items[t]["xy"]]
        spot = free_spot(items, tray, edge, self.ctx.reachable, base_xy, avoid, dc["place_clearance_m"],
                         dc["place_avoid_target_m"])
        self.say(f"move {it['color']} ({decision['reason']})")
        off = self.grasp_offset(items, i, pos, it["yaw_rad"])
        row.update(spot=spot, grasp_yaw_offset_deg=None if off is None else round(math.degrees(off), 1))
        pick = None
        if push_only:
            yield from self.run(Home(self.ctx))
        elif spot is not None and off is not None and it["status"] != "merged":   # merged: centre between blocks
            self.set_phase("move")
            pick = yield from self.run(PickAt(self.ctx, block=b, pos=pos, yaw=it["yaw_rad"], yaw_offset=off,
                                              color=it["color"]))
            if pick["status"] == "succeeded":
                b = pick["evidence"].get("held_block", b)
            if pick["status"] == "succeeded" and i == t:
                self.set_phase("check_held")
                self.say("wrist: colour of the block in the gripper")
                row["held_color"] = self.held_color(pick["evidence"]["block_from_ee_m"], it["yaw_rad"])
                if row["held_color"] == color:
                    return {"status": "holding", "block": b, "reason": "the wrist saw the target colour in the gripper"}
            if pick["status"] == "succeeded":
                yield from self.run(Place(self.ctx, block=b, xy=spot, tray="storage", color=it["color"]))
        if pick is None or pick["status"] != "succeeded":
            if pick is not None:
                yield from self.run(Home(self.ctx))    # a failed pick can leave the wrist turned
            ctrl = self.ctrl
            push_z = cams.floor + edge / 2 + float(m["grasp_dz"])

            def push_ok(a, c):
                d = np.asarray(c) - np.asarray(a)
                y = ctrl.yaw_for(np.array([a[0], a[1], 0.0]),
                                 wrap_quarter(math.atan2(d[1], d[0]) - ctrl.yaw_for(np.array([a[0], a[1], 0.0]), 0.0)))
                return all(ctrl.chain.solve_ik_tilt(np.array([q[0], q[1], z]), y, ctrl.q_home, ctrl.p.max_tilt_deg)[1]
                           for q, z in ((a, float(m["safe_z"])), (a, push_z), (c, push_z)))

            if push_only:                       # towards the tray centre
                u = np.asarray(tray["center"], float) - np.asarray(it["xy"], float)
                u = u / max(float(np.linalg.norm(u)), 1e-6)
                p0 = np.asarray(it["xy"], float)
                a, c = p0 - (edge / 2 + 0.02) * u, p0 + float(dc["push_distance_m"]) * u
                path = ([float(x) for x in a], [float(x) for x in c]) if push_ok(a, c) else None
            else:
                path = push_path(items, i, tray, edge, t, dc["push_distance_m"], feasible=push_ok)
            if path is not None:
                self.set_phase("move")
                yield from self.run(Push(self.ctx, start=path[0], end=path[1], z=push_z))
            key = item_key(it)
            failed[key] = failed.get(key, 0) + 1
        return None
