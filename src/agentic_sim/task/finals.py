"""Finals whole-order scene: a shortage in the middle of the order, answered by a substitute colour or a restock.

  name: finals_shortage
  finals:
    roles: {shortage: red, priority: blue, substitute: yellow}   # A x2, B x1, C x1 in the storage tray
    layout: {config: configs/random_blocks_finals.yaml, seed: 1}
    standby_after: 2            # verified deliveries before the standby step
    answer: substitute          # batch: substitute | wait, injected once the shortage question exists; none: chat
    answer_after_s: 2.0
    restock_after_observations: 3
  timeout_s: 240

The order is B x1 then A x2 (B first). After the second verified delivery the executor is sent to standby; once it
reports the arm settled there with an empty, open gripper, the scene removes the A blocks still in the storage tray.
The executor finds out from its own next observation and asks; it is never told. After "wait", the scene puts one
removed block back at another legal pose once the executor has looked at the tray `restock_after_observations` times.
Block IDs are used here only to set the scene up, change it and score it; the executor chooses from observations.
Scene changes are state-triggered (executor events), never by elapsed time.
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np

from agentic_sim.config import load, v
from agentic_sim.scene_layout import block_in_tray, palette_by_name

ROLE_KEYS = ("shortage", "priority", "substitute")


def resolve(spec: dict) -> dict:
    """A scenario with its finals section expanded: block colours [A, A, B, C] and the order B x1, A x2 (B first)."""
    if "finals" not in spec:
        return spec
    out = copy.deepcopy(spec)
    roles = out["finals"]["roles"]
    a, b, c = (roles[k] for k in ROLE_KEYS)
    if len({a, b, c}) != 3:
        raise ValueError(f"finals roles must be three different colours: {roles}")
    out["blocks"] = [a, a, b, c]
    if out["finals"].get("order", "scenario") == "scenario":
        out["order"] = {b: 1, a: 2}      # chat: the user orders in the chat window ("兩紅一藍，藍色先放")
    out.setdefault("timeout_s", 240)
    return out


def check_palette(spec: dict, scene: dict):
    names = set(palette_by_name(scene["block"]["extra_palette"]))
    bad = [c for c in spec["finals"]["roles"].values() if c not in names]
    if bad:
        raise ValueError(f"finals roles {bad} are not palette colours ({sorted(names)})")


def layout(scene: dict, spec: dict, teleop_path: str, log=print):
    """The seeded legal layout of the four blocks in the storage tray (reachability, margins, yaw-aware gaps)."""
    from agentic_sim.random_blocks import grasp_ik_checker, sample_scene
    check_palette(spec, scene)
    f = spec["finals"]
    settings = dict(load(f["layout"]["config"]))
    settings["fixed_colors"] = list(spec["blocks"])
    settings["count_range"] = [len(spec["blocks"])] * 2
    checker = grasp_ik_checker(scene, teleop_path) if settings.get("check_ik", True) else None
    out, colors, record = sample_scene(scene, settings, seed=int(f["layout"]["seed"]), region="storage",
                                       reachable=checker)
    record["roles"] = dict(f["roles"])
    log(f"finals layout: {record}")
    return out, colors, record


def restock_pose(scene: dict, spec: dict, teleop_path: str, occupied: list, removed_pose: dict, attempt_seed: str):
    """Another legal pose in the storage tray for the block put back: inside the tray with the configured margins,
    clear of the blocks now there (yaw-aware), reachable, and away from where it was removed (at least one edge plus
    the gap, and a different yaw)."""
    from agentic_sim.random_blocks import REGION_TRAY, grasp_ik_checker, legal_position
    from agentic_sim.scene_layout import tray_parts
    settings = dict(load(spec["finals"]["layout"]["config"]))
    edge = float(v(scene["block"]["edge"]))
    info = tray_parts(scene["trays"][REGION_TRAY["storage"]])[1]
    reachable = grasp_ik_checker(scene, teleop_path)
    rng = random.Random(attempt_seed)
    lo, hi = settings.get("yaw_range_deg", [-45.0, 45.0])
    placed = [tuple(p["xy"]) for p in occupied]
    yaws = [p["yaw_deg"] for p in occupied]
    old = np.asarray(removed_pose["xy"])
    away = edge + float(settings["gap_m"])
    z = info["floor_top"] + edge / 2 + 0.0005
    for _ in range(int(settings["max_attempts_per_layout"]) * int(settings["layout_retries"])):
        lx, ly = (rng.uniform(-h, h) for h in info["inner_half"])
        c, s = math.cos(info["yaw"]), math.sin(info["yaw"])
        xy = (info["center"][0] + c * lx - s * ly, info["center"][1] + s * lx + c * ly)
        yaw = rng.uniform(float(lo), float(hi))
        if np.linalg.norm(np.asarray(xy) - old) < away:
            continue
        if abs((yaw - removed_pose["yaw_deg"] + 45) % 90 - 45) < 10:
            continue
        if not legal_position(scene, xy, placed, "storage", settings, yaw_deg=yaw, placed_yaws=yaws):
            continue
        if not reachable((*xy, z), yaw):
            continue
        return {"xy": [float(xy[0]), float(xy[1])], "z": z, "yaw_deg": float(yaw),
                "from_removed_mm": float(np.linalg.norm(np.asarray(xy) - old) * 1000)}
    raise ValueError("no legal restock pose in the storage tray")


def _yaw_deg(q_wxyz) -> float:
    w, x, y, zq = (float(c) for c in q_wxyz)
    return math.degrees(math.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq)))


class FinalsDirector:
    """Scene side of the finals run: standby request, removal, (batch) answer, restock, scoring.

    world: object with remove_block(b), place_block(b, xyz, yaw_deg), block_pose(b) -> (xyz, quat_wxyz).
    """

    def __init__(self, spec: dict, executor, world, scene: dict, teleop_path: str, log=print):
        self.spec, self.ex, self.world, self.scene, self.teleop, self.log = spec, executor, world, scene, teleop_path, log
        f = spec["finals"]
        self.roles = {k: f["roles"][k] for k in ROLE_KEYS}
        self.standby_after = int(f.get("standby_after", 2))
        self.answer = f.get("answer") or None
        self.answer_after = float(f.get("answer_after_s", 2.0))
        self.restock_after = int(f.get("restock_after_observations", 3))
        # acceptance of a fresh look after the answer: once the question is up (before the answer), the scene moves
        # the substitute block to another legal pose; the executor must find it there with the cameras
        self.move_substitute = bool(f.get("move_substitute", False))
        self.moved: list = []
        self.perception = f.get("perception", "camera")   # camera: choose / recheck with the cameras; sim_truth
        if self.perception not in ("camera", "sim_truth"):
            raise ValueError("finals perception must be camera or sim_truth")
        self.removed: list = []          # {block, pose}
        self.restocked: list = []
        self.scene_events: list = []
        self.question = None             # first shortage question event
        self.questions = 0
        self.answer_event = None
        self.answer_due = None
        self.restock_obs = 0
        self.standby_sent = False
        self.standby_event = None
        self.picks: list = []            # (t, block, colour)
        self.delivered_at: dict = {}     # block -> position when verified
        self.delivered_disp: dict = {}   # block -> max mm moved after its verification
        self.c_disp = 0.0
        self.c_init = None
        self.pending: list = []          # scene changes applied at the next frame
        self.wait_ee0 = None             # EE when the "wait" answer was taken; max move until the next pick
        self.wait_ee_mm = 0.0
        executor.hooks.append(self._on_event)

    # ---- scene log -----------------------------------------------------------------------------------
    def _scene(self, action: str, **data):
        e = self.ex.emit("scene", action=action, **data)
        self.scene_events.append(e)
        self.log(f"finals scene: {action} {data}")

    # ---- executor events ----------------------------------------------------------------------------------
    def _on_event(self, e: dict):
        k = e["kind"]
        if k == "verified" and e["delta"] > 0:
            self.delivered_at[e["block"]] = None      # position taken at the next frame
            n = sum(e["ledger"]["delivered"].values())
            if n == self.standby_after and not self.standby_sent:
                self.standby_sent = True
                self.ex.standby()
        elif k == "standby_reached" and self.standby_event is None:
            self.standby_event = e
            if e["ok"]:
                self.pending.append(("remove", e))
            else:
                self._scene("remove_refused", reason="arm not settled with an empty open gripper", standby=e)
        elif k == "question" and e.get("qkind") == "shortage":
            self.questions += 1
            if self.question is None:
                self.question = e
                if self.move_substitute:
                    self.pending.append(("move_substitute", e))
                if self.answer:
                    self.answer_due = e["t"] + self.answer_after
        elif k == "answer":
            if self.answer_event is None:
                self.answer_event = e
        elif k == "observation" and e.get("reason") == "restock_wait":
            if self.answer_event is not None and self.answer_event.get("choice") == "wait" and not self.restocked:
                self.restock_obs += 1
                if self.restock_obs == self.restock_after:
                    self.pending.append(("restock", e))
        elif k == "skill_start" and e.get("skill") == "pick":
            self.picks.append((e["t"], e.get("block"), e.get("color")))

    # ---- per frame (before the executor step) -------------------------------------------------------------
    def step(self, obs):
        ex = self.ex
        for action, e in self.pending:
            if action == "remove":
                self._remove(obs)
            elif action == "restock":
                self._restock(obs, e)
            elif action == "move_substitute":
                self._move_substitute(obs)
        self.pending = []
        if self.answer_due is not None and obs.t + 1e-9 >= self.answer_due:
            self.answer_due = None
            q = self.question
            kw = {"color": self.roles["substitute"]} if self.answer == "substitute" else {}
            ex.answer(q["id"], self.answer, **kw)
            self._scene("batch_answer", question=q["id"], choice=self.answer, **kw)
        for b, p0 in list(self.delivered_at.items()):
            p = np.asarray(obs.blocks[b])
            if p0 is None:
                self.delivered_at[b] = p.copy()
                continue
            self.delivered_disp[b] = max(self.delivered_disp.get(b, 0.0), float(np.linalg.norm(p - p0)) * 1000)
        ae = self.answer_event
        if ae is not None and ae.get("choice") == "wait" and not any(p[0] >= ae["t"] for p in self.picks):
            if self.wait_ee0 is None:
                self.wait_ee0 = np.asarray(obs.ee).copy()
            self.wait_ee_mm = max(self.wait_ee_mm, float(np.linalg.norm(np.asarray(obs.ee) - self.wait_ee0)) * 1000)
        c = self._blocks_of("substitute")[0]
        if self.c_init is None:
            self.c_init = np.asarray(obs.blocks[c]).copy()
        if getattr(self, "_c_frame_stale", False):
            self._c_frame_stale = False
        elif self.answer_event is None or self.answer_event.get("choice") == "wait":   # by the answer actually given
            self.c_disp = max(self.c_disp, float(np.linalg.norm(np.asarray(obs.blocks[c]) - self.c_init)) * 1000)

    def _blocks_of(self, role: str) -> list:
        return [b for b, col in enumerate(self.ex.world.colors) if col == self.roles[role]]

    def _remove(self, obs):
        ex = self.ex
        settled = obs.qd_max < float(ex.cfg["motion"]["settle_qd"])
        jaw_open = ex.ctrl.state.gripper >= ex.ctrl.p.gripper_open - 1e-6
        if not (settled and jaw_open and ex.holding is None):
            self._scene("remove_refused", reason="arm moving or gripper not empty and open", qd=obs.qd_max)
            return
        storage = ex.world.trays["storage"]
        for b in self._blocks_of("shortage"):
            if b in ex.delivered_blocks or not block_in_tray(obs.blocks[b], storage, ex.world.edge):
                continue
            pos, quat = self.world.block_pose(b)
            pose = {"xy": [float(pos[0]), float(pos[1])], "z": float(pos[2]), "yaw_deg": _yaw_deg(quat)}
            self.world.remove_block(b)
            self.removed.append({"block": b, "pose": pose, "t": obs.t})
            self._scene("remove", block=b, color=ex.world.colors[b], pose=pose, arm_qd=round(float(obs.qd_max), 4),
                        jaw_open=bool(jaw_open))

    def _move_substitute(self, obs):
        """Move C to another legal, reachable pose in the storage tray (at least one edge plus the gap away)."""
        ex = self.ex
        c = self._blocks_of("substitute")[0]
        storage = ex.world.trays["storage"]
        occupied = []
        for b in range(len(ex.world.colors)):
            if b == c or not block_in_tray(obs.blocks[b], storage, ex.world.edge):
                continue
            pos, quat = self.world.block_pose(b)
            occupied.append({"xy": [float(pos[0]), float(pos[1])], "yaw_deg": _yaw_deg(quat)})
        pos, quat = self.world.block_pose(c)
        old = {"xy": [float(pos[0]), float(pos[1])], "z": float(pos[2]), "yaw_deg": _yaw_deg(quat)}
        seed = f"{self.spec['finals']['layout']['seed']}:move_substitute"
        pose = restock_pose(self.scene, self.spec, self.teleop, occupied, old, seed)
        self.world.place_block(c, [*pose["xy"], pose["z"]], pose["yaw_deg"])
        self.moved.append({"block": c, "from": old, "to": pose, "t": obs.t})
        self.c_init = np.array([*pose["xy"], pose["z"]])   # C's own displacement: from where the scene put it
        self._c_frame_stale = True     # this frame's observation was taken before the move
        self._scene("move_substitute", block=c, color=ex.world.colors[c], from_pose=old, pose=pose)

    def _restock(self, obs, e):
        ex = self.ex
        if not self.removed:
            return
        r = self.removed[0]
        storage = ex.world.trays["storage"]
        occupied = []
        for b in range(len(ex.world.colors)):
            if b == r["block"] or not block_in_tray(obs.blocks[b], storage, ex.world.edge):
                continue
            pos, quat = self.world.block_pose(b)
            occupied.append({"xy": [float(pos[0]), float(pos[1])], "yaw_deg": _yaw_deg(quat)})
        seed = f"{self.spec['finals']['layout']['seed']}:restock"
        pose = restock_pose(self.scene, self.spec, self.teleop, occupied, r["pose"], seed)
        self.world.place_block(r["block"], [*pose["xy"], pose["z"]], pose["yaw_deg"])
        self.restocked.append({"block": r["block"], "pose": pose, "t": obs.t, "after_observations": self.restock_obs})
        self._scene("restock", block=r["block"], color=ex.world.colors[r["block"]], pose=pose,
                    removed_pose=r["pose"], after_observations=self.restock_obs)

    # ---- scoring -----------------------------------------------------------------------------------------------
    def checks(self, obs) -> list:
        ex, w = self.ex, self.ex.world
        a, b, c = (self.roles[k] for k in ROLE_KEYS)
        out = []

        def add(name, ok, **detail):
            out.append({"name": name, "ok": bool(ok), **detail})

        placed = {}
        for i, p in enumerate(obs.blocks):
            if block_in_tray(p, w.trays["placement"], w.edge):
                placed[w.colors[i]] = placed.get(w.colors[i], 0) + 1
        verified = [e for e in ex.events if e["kind"] == "verified" and e["delta"] > 0]
        add("priority_first", bool(verified) and verified[0]["color"] == b,
            first=verified[0]["color"] if verified else None, expected=b)
        add("standby_before_removal", self.standby_event is not None and self.standby_event["ok"] and bool(self.removed),
            standby=self.standby_event, removed=self.removed)
        add("one_shortage_question", self.questions == 1, questions=self.questions)
        q = self.question
        exp_q = {"missing": {a: 1}, "delivered": {b: 1, a: 1}}
        add("question_content", q is not None and q.get("missing") == exp_q["missing"]
            and q.get("delivered") == exp_q["delivered"] and c in (q.get("substitute_candidates") or [])
            and set(q.get("options", [])) == {"substitute", "wait"}, expected=exp_q, asked=q)
        t_q = q["t"] if q else None
        t_ans = self.answer_event["t"] if self.answer_event else None
        early = [p for p in self.picks if t_q is not None and p[0] >= t_q and (t_ans is None or p[0] < t_ans)]
        add("no_pick_before_answer", q is not None and t_ans is not None and not early, picks=early)
        add("delivered_untouched", all(d < 3.0 for d in self.delivered_disp.values()) and len(self.delivered_disp) >= 2,
            limit_mm=3.0, observed_mm={k: round(x, 2) for k, x in self.delivered_disp.items()})
        choice = self.answer_event.get("choice") if self.answer_event else None
        if choice == "substitute":
            add("final_placement", placed == {a: 1, b: 1, c: 1}, expected={a: 1, b: 1, c: 1}, observed=placed)
            led = ex.ledger.snapshot()
            add("ledger", led["done"] and led["target"] == {b: 1, a: 1, c: 1} and led["delivered"] == {b: 1, a: 1, c: 1},
                ledger=led)
            add("substitute_untouched_until_answer", self.c_disp < 2.0, observed_mm=round(self.c_disp, 2))
            ok, detail = self._fresh_look(obs)
            add("fresh_look_after_answer", ok, **detail)
        elif choice == "wait":
            add("final_placement", placed == {a: 2, b: 1}, expected={a: 2, b: 1}, observed=placed)
            led = ex.ledger.snapshot()
            add("ledger", led["done"] and led["target"] == {b: 1, a: 2} and led["delivered"] == {b: 1, a: 2},
                ledger=led)
            cb = self._blocks_of("substitute")[0]
            add("substitute_stays_in_storage", block_in_tray(obs.blocks[cb], w.trays["storage"], w.edge)
                and self.c_disp < 2.0, observed_mm=round(self.c_disp, 2))
            rs = self.restocked[0] if self.restocked else None
            away_mm = 1000 * (w.edge + float(load(self.spec["finals"]["layout"]["config"])["gap_m"]))
            add("restock_new_pose", rs is not None and rs["pose"]["from_removed_mm"] >= away_mm - 1e-6,
                limit_mm=round(away_mm, 1), restock=rs)
            waited = [p for p in self.picks if t_ans is not None and p[0] >= t_ans and (rs is None or p[0] < rs["t"])]
            add("stays_at_standby_while_waiting", self.wait_ee0 is not None and self.wait_ee_mm < 3.0,
                limit_mm=3.0, observed_mm=round(self.wait_ee_mm, 2))
            add("no_pick_before_restock", rs is not None and not waited and self.restock_obs >= self.restock_after,
                picks=waited, observations_before_restock=self.restock_obs)
        else:
            add("answered", False, choice=choice)
        return out

    def _fresh_look(self, obs):
        """After the substitute answer: a new storage observation before the first pick, and that pick's target (its
        first approach) is where C was observed then; when the scene moved C, the new pose, not the old one."""
        ev = self.ex.events
        t_a = self.answer_event["t"]
        pick = next((e for e in ev if e["kind"] == "skill_start" and e.get("skill") == "pick" and e["t"] >= t_a), None)
        if pick is None:
            return False, {"reason": "no pick after the answer"}
        looks = [e for e in ev if e["kind"] == "observation" and t_a <= e["t"] <= pick["t"]]
        target = (pick.get("target") or {}).get("xy")
        c = self._blocks_of("substitute")[0]
        detail = {"answer_t": t_a, "observations_between": [{"t": e["t"], "n": e["n"], "reason": e["reason"],
                                                             "source": e["source"]} for e in looks],
                  "pick_t": pick["t"], "pick_target_xy": target}
        if self.perception == "sim_truth":     # no camera target: the fresh observation is what is checked
            return bool(looks), detail
        ok = bool(looks) and target is not None
        if looks and target is not None:
            seen = looks[-1]["poses"].get(str(pick.get("block")), {}).get("xy")
            detail["observed_xy_after_answer"] = seen
            ok = ok and seen is not None and np.linalg.norm(np.subtract(seen, target)) < 1e-6
        if self.moved:
            m = self.moved[0]
            detail.update(moved_from_xy=m["from"]["xy"], moved_to_xy=m["to"]["xy"])
            if target is not None:
                to_new = float(np.linalg.norm(np.subtract(target, m["to"]["xy"]))) * 1000
                to_old = float(np.linalg.norm(np.subtract(target, m["from"]["xy"]))) * 1000
                detail.update(target_to_new_mm=round(to_new, 2), target_to_old_mm=round(to_old, 2))
                ok = ok and to_new < 10.0 and to_old > 1000 * self.ex.world.edge
        return ok, detail

    def record(self) -> dict:
        return {"roles": self.roles, "perception": self.perception, "removed": self.removed, "moved": self.moved, "restocked": self.restocked,
                "scene_events": self.scene_events, "question": self.question,
                "answer": self.answer_event, "standby": self.standby_event}
