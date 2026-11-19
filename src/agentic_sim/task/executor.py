"""Task executor: chooses the next skill from the ledger, handles order changes, questions and answers.

  ex = TaskExecutor(ctrl, world, cfg, dt=1/60, log=print, event_log="outputs/task/run.jsonl")
  ex.order({"red": 3})                       # queued; applied at the next step
  each control frame:  ex.step(obs)          # before ctrl.joint_command(); obs: skills.Obs (sim truth here)
  ex.answer(qid, "return")                   # reply to ex.question

One skill runs at a time; it only proposes EE targets and gripper goals to the shared controller. Rules (issue #5):
- "total" is the new complete order, "add" adds to it. Each change is a new ledger revision; the running skill is
  re-checked against it at once: kept (and re-tagged with the new revision) if it still serves the order, otherwise
  cancelled by phase (agentic_sim.task.skills). Every frame's command carries the revision it was issued for.
- Holding a block the new order does not need: stop, keep it, ask the user (return / place_at / deliver). No release
  and no return home before the answer.
- Delivered beyond the order (released before the change): reported, then retrieved to the storage tray.
- Shortage: do what can be done first, then ask (substitute colour / reduce / wait). Never substitute on its own.
- A delivery counts only after the placed block is observed at rest in the tray with the gripper open and clear.
"""

from __future__ import annotations

import json
import os

import numpy as np

from agentic_sim.scene_layout import block_in_tray
from agentic_sim.task.ledger import Ledger
from agentic_sim.task.skills import Home, Hold, Motion, Obs, Pick, Place, World

HELD_OPTIONS = ("return", "place_at", "deliver")
SHORTAGE_OPTIONS = ("substitute", "reduce", "wait")


class TaskExecutor:
    def __init__(self, ctrl, world: World, cfg: dict, dt: float, log=print, event_log: str | None = None):
        self.ctrl, self.world, self.cfg, self.dt, self.log = ctrl, world, cfg, dt, log
        self.ledger = Ledger()
        self.motion = Motion(ctrl, cfg, dt)
        self.skill = None
        self.holding: int | None = None
        self.purpose: str | None = None       # what the held block is for: deliver / deliver_extra / return / ...
        self.place_xy = None                  # place_at answer
        self.question: dict | None = None
        self._qseq = 0
        self._inputs: list = []
        self.events: list = []
        self.hooks: list = []
        self.results: list = []
        self.cmd_log: list = []               # (t, revision) of every frame with a skill commanding the arm
        self.delivered_blocks: dict = {}      # block -> colour, verified in the placement tray
        self.attempts: dict = {}
        self.skipped: set = set()
        self.skill_counts: dict = {}          # (skill, colour) -> started so far
        self.state = "idle"                   # idle / running / waiting / done
        self.restock_wait = False
        self._asked_shortage = None
        self._reported_over: set = set()
        self._pending_cancel = None           # latency record of the cancel in progress
        self.cancels: list = []
        self.obs: Obs | None = None
        self.t = 0.0
        self._event_log = None
        if event_log:
            os.makedirs(os.path.dirname(event_log) or ".", exist_ok=True)
            self._event_log = open(event_log, "w")

    # ---- inputs (server / panel / scenario); applied at the next step ----------------------
    @property
    def max_per_color(self) -> int:
        return int((self.cfg.get("order") or {}).get("max_per_color", 8))

    def projected_target(self) -> dict:
        """The order once every queued order is applied (queued answers are not included)."""
        t = dict(self.ledger.target)
        for item in list(self._inputs):
            if item[0] == "order":
                t = Ledger.compose(t, item[1], item[2])
        return t

    def _over_cap(self, target: dict) -> dict:
        return {c: n for c, n in target.items() if n > self.max_per_color}

    def order(self, counts: dict, mode: str = "total"):
        """Queue an order. Refused at once (ValueError, nothing queued) when the mode or a count is invalid or the
        result would ask for more than max_per_color blocks of one colour."""
        over = self._over_cap(Ledger.compose(self.projected_target(), counts, mode))
        if over:
            raise ValueError(f"at most {self.max_per_color} blocks per colour; this order would give {over}")
        self._inputs.append(("order", dict(counts), mode))

    def answer(self, qid: int, choice: str, **kw):
        self._inputs.append(("answer", int(qid), str(choice), kw))

    # ---- events ------------------------------------------------------------------------------
    def _write(self, e: dict):
        self.events.append(e)
        if self._event_log:
            self._event_log.write(json.dumps(e, ensure_ascii=False) + "\n")
            self._event_log.flush()
        for h in list(self.hooks):
            h(e)

    def emit(self, kind: str, **data) -> dict:
        e = {"t": round(float(self.t), 4), "rev": self.ledger.revision, "kind": kind, **data}
        self._write(e)
        return e

    def close(self):
        if self._event_log:
            self._event_log.close()
            self._event_log = None

    # ---- helpers ----------------------------------------------------------------------------------
    def color(self, b: int) -> str:
        return self.world.colors[b]

    def _in(self, b: int, tray: str) -> bool:
        return bool(block_in_tray(self.obs.blocks[b], self.world.trays[tray], self.world.edge))

    def available(self, color: str) -> list:
        """Blocks of `color` lying in the storage tray, not delivered, not held, not given up on."""
        return [b for b, c in enumerate(self.world.colors)
                if c == color and b != self.holding and b not in self.delivered_blocks and b not in self.skipped
                and self._in(b, "storage")]

    def _free(self, xy, ignore: int | None = None, radius: float = 0.04) -> bool:
        return all(float(np.linalg.norm(p[:2] - np.asarray(xy))) > radius
                   for i, p in enumerate(self.obs.blocks) if i != ignore)

    def free_slot(self, b: int):
        for xy in self.cfg["placement_slots"]:
            if self._free(xy, ignore=b):
                return list(map(float, xy))
        return None

    def return_xy(self, b: int):
        """The block's own start position if free, else the nearest free start position in the storage tray."""
        own = np.asarray(self.world.init_pos[b][:2])
        cands = sorted((np.asarray(p[:2]) for p in self.world.init_pos), key=lambda p: float(np.linalg.norm(p - own)))
        for xy in cands:
            if self._free(xy, ignore=b):
                return [float(xy[0]), float(xy[1])]
        return None

    def _counts_in(self, tray: str) -> dict:
        out = {}
        for b in range(len(self.world.colors)):
            if b != self.holding and self._in(b, tray):
                out[self.color(b)] = out.get(self.color(b), 0) + 1
        return out

    # ---- callbacks from skills --------------------------------------------------------------------
    def on_grasp(self, b: int):
        self.holding = b

    def on_release(self, b: int):
        if self.holding == b:
            self.holding = None

    # ---- one frame ----------------------------------------------------------------------------------
    def step(self, obs: Obs):
        self.obs, self.t = obs, obs.t
        self._process_inputs()
        if self.skill is not None:
            self.skill.step(obs)
            if self.skill.status != "running":
                self._skill_done(self.skill)
        self._process_inputs()          # hooks fired during the skill step (scenario triggers)
        if self.skill is None and self.question is None and self.ledger.revision > 0:
            self._plan()
            if self.skill is not None:
                self.skill.step(obs)    # first frame of the new skill: no idle frame between skills
        if self.skill is not None:
            self.cmd_log.append((round(float(obs.t), 4), self.skill.revision))
        self._track_cancel()

    def _start(self, cls, **params):
        c = self.color(params["block"]) if params.get("block") is not None else None
        key = (cls.name, c)
        self.skill_counts[key] = self.skill_counts.get(key, 0) + 1
        params.setdefault("color", c)
        params["nth"] = self.skill_counts[key]
        self.skill = cls(self, **params)
        self.state = "running"
        self.emit("skill_start", skill=cls.name, **{k: v for k, v in params.items() if k != "holding"})

    # ---- planning ----------------------------------------------------------------------------------------
    def _plan(self):
        led = self.ledger
        if self.holding is not None:
            b = self.holding
            c = self.color(b)
            if self.purpose in ("retrieve", "return"):
                xy = self.return_xy(b)
                return self._start(Place, block=b, xy=xy, tray="storage", purpose="return")
            if self.purpose == "place_at":
                xy = self.place_xy
                tray = next((t for t in ("storage", "placement") if block_in_tray([xy[0], xy[1], 0.0],
                                                                                 self.world.trays[t], 0.0)), None)
                return self._start(Place, block=b, xy=xy, tray=tray, purpose="place_at")
            if self.purpose == "deliver_extra" or led.need(c) > 0:
                xy = self.free_slot(b)
                if xy is None:
                    return self._ask_full(b)
                return self._start(Place, block=b, xy=xy, tray="placement",
                                   purpose="deliver_extra" if self.purpose == "deliver_extra" else "deliver")
            return self._ask_held(b)
        overs = led.overs()
        if overs:
            c = next(iter(overs))
            b = [x for x, col in self.delivered_blocks.items() if col == c][-1]   # the last one delivered
            self.purpose = "retrieve"
            return self._start(Pick, block=b, purpose="retrieve")
        needed = led.needed()
        for c in needed:
            av = self.available(c)
            if av:
                self.restock_wait = False
                self.purpose = "deliver"
                return self._start(Pick, block=av[0], purpose="deliver")
        if needed:
            if not self.restock_wait and self._asked_shortage != led.revision:
                self._ask_shortage(needed)
            self.state = "waiting"
            return
        if self.state != "done":
            if float(np.linalg.norm(self.ctrl.state.pos - self.world.home)) > 1e-3:
                return self._start(Home)
            self.state = "done"
            self.emit("task_done", ledger=led.snapshot(), placement=self._counts_in("placement"))

    # ---- questions ---------------------------------------------------------------------------------------
    def _ask(self, kind: str, text: str, options, **data):
        self._qseq += 1
        self.question = {"id": self._qseq, "kind": kind, "rev": self.ledger.revision, "text": text,
                         "options": list(options), **data}
        self.state = "waiting"
        self.emit("question", qkind=kind, **{k: v for k, v in self.question.items() if k != "kind"})

    def _ask_held(self, b: int):
        c = self.color(b)
        self._start(Hold, block=b)
        self._ask("held_item", f"Holding 1 {c}, which the new order does not need. Return it to its place "
                  f"(return), put it somewhere else (place_at x y), or deliver it anyway (deliver)?",
                  HELD_OPTIONS, block=b, color=c)

    def _ask_full(self, b: int):
        self._start(Hold, block=b)
        self._ask("held_item", f"No free slot in the placement tray for the {self.color(b)} block in the gripper. "
                  f"Return it (return) or put it somewhere else (place_at x y)?", ("return", "place_at"),
                  block=b, color=self.color(b))

    def _ask_shortage(self, needed: dict):
        led = self.ledger
        self._asked_shortage = led.revision
        stock = {}
        for b, c in enumerate(self.world.colors):
            if b != self.holding and b not in self.delivered_blocks and b not in self.skipped and self._in(b, "storage"):
                stock[c] = stock.get(c, 0) + 1
        delivered = {c: n for c, n in led.delivered.items() if n}
        txt = ("Delivered " + (", ".join(f"{n} {c}" for c, n in delivered.items()) or "nothing") +
               "; still missing " + ", ".join(f"{n} {c}" for c, n in needed.items()) +
               " and none left in the storage tray. Left there: " +
               (", ".join(f"{n} {c}" for c, n in stock.items()) or "nothing") +
               ". Substitute another colour (substitute <colour>), reduce the order (reduce), or wait for more (wait)?")
        self._ask("shortage", txt, SHORTAGE_OPTIONS, delivered=delivered, missing=dict(needed), available=stock)

    # ---- inputs ---------------------------------------------------------------------------------------------
    def _process_inputs(self):
        while self._inputs:
            item = self._inputs.pop(0)
            if item[0] == "order":
                self._apply_order(item[1], item[2])
            else:
                self._apply_answer(*item[1:])

    def _apply_order(self, counts, mode):
        over = self._over_cap(Ledger.compose(self.ledger.target, counts, mode))
        if over:   # checked again here: something else (a shortage answer) may have changed the order meanwhile
            self.emit("order_rejected", mode=mode, counts=dict(counts),
                      reason=f"at most {self.max_per_color} per colour: {over}")
            return
        e = self.ledger.set_order(counts, mode, self.t)
        self._write(e)
        self.restock_wait = False
        if self.state == "done":
            self.state = "running"
        q = self.question
        if q is not None and q["kind"] == "shortage":
            self.emit("question_dropped", id=q["id"], reason="order_changed")
            self.question = None
        if q is not None and q["kind"] == "held_item" and self.ledger.need(q["color"]) > 0:
            self.emit("question_dropped", id=q["id"], reason="order_changed")
            self.question = None
            self.purpose = "deliver"
            if self.skill is not None and self.skill.name == "hold":
                self.skill.cancel("order_changed", self.ledger.revision)
        self._report_over()
        self._reconcile()

    def _reconcile(self):
        """Re-check the running skill against the new revision: keep (re-tag) or cancel."""
        sk, led = self.skill, self.ledger
        if sk is None or sk.status != "running":
            return
        b = sk.params.get("block")
        c = self.color(b) if b is not None else None
        purpose = sk.params.get("purpose")
        keep = True
        if sk.name == "pick":
            keep = led.over(c) > 0 if purpose == "retrieve" else led.need(c) > 0
        elif sk.name == "place":
            if purpose == "deliver":
                keep = led.need(c) > 0 or not sk.cancellable
        elif sk.name == "home":
            keep = False
        elif sk.name == "hold":
            keep = self.question is not None
        if keep:
            sk.revision = led.revision
            self.emit("skill_kept", skill=sk.name, phase=sk.phase, block=b)
            return
        if sk.cancel("order_changed", led.revision):
            self._pending_cancel = {"t_change": self.t, "rev": led.revision, "skill": sk.name, "phase": sk.phase,
                                    "block": b, "t_arm_stopped": None, "t_done": None}
            self.emit("cancel_requested", skill=sk.name, phase=sk.phase, block=b)
        else:   # past the point of no return (the block is being let go): finish, then handle what follows
            sk.revision = led.revision
            self.emit("skill_kept", skill=sk.name, phase=sk.phase, block=b, reason="not_cancellable")

    def _apply_answer(self, qid, choice, kw):
        q = self.question
        if q is None or q["id"] != qid:
            self.emit("answer_rejected", id=qid, choice=choice, reason="no such open question")
            return
        if choice not in q["options"]:
            self.emit("answer_rejected", id=qid, choice=choice, reason=f"options are {q['options']}")
            return
        self.emit("answer", id=qid, choice=choice, **kw)
        led = self.ledger
        if q["kind"] == "held_item":
            if choice == "place_at":
                xy = kw.get("xy")
                if not xy or len(xy) != 2:
                    self.emit("answer_rejected", id=qid, choice=choice, reason="place_at needs xy: [x, y]")
                    return
                self.place_xy = [float(xy[0]), float(xy[1])]
            if choice == "deliver":
                e = led.accept_extra(q["color"], 1, self.t)
                self._write(e)
            self.purpose = {"return": "return", "place_at": "place_at", "deliver": "deliver_extra"}[choice]
            self.question = None
            if self.skill is not None and self.skill.name == "hold":
                self.skill.cancel("answered", led.revision)
            return
        # shortage
        if choice == "wait":
            self.question = None
            self.restock_wait = True
            return
        new = dict(led.target)
        if choice == "substitute":
            sub = kw.get("color")
            if sub not in set(self.world.colors):
                self.emit("answer_rejected", id=qid, choice=choice, reason=f"unknown colour {sub!r}")
                return
            for c, n in q["missing"].items():
                new[c] = new.get(c, 0) - n
                new[sub] = new.get(sub, 0) + n
        else:   # reduce
            for c, n in q["missing"].items():
                new[c] = new.get(c, 0) - n
        self.question = None
        self._apply_order({c: n for c, n in new.items() if n > 0}, "total")

    # ---- results ------------------------------------------------------------------------------------------------
    def _skill_done(self, sk):
        r = sk.result
        self.skill = None
        self.results.append(r.to_dict())
        self.emit("skill_done", **r.to_dict())
        b = sk.params.get("block")
        led = self.ledger
        if self._pending_cancel is not None and r.status == "cancelled":
            self._pending_cancel["t_done"] = self.t
        if sk.name == "pick":
            if r.status in ("succeeded", "cancelled"):
                self.holding = r.holding
            elif r.holding is not None and r.failure_reason != "dropped":
                self.holding = r.holding        # e.g. the lift timed out: the block is still in the gripper
            else:
                self.holding = None
                self.attempts[b] = self.attempts.get(b, 0) + 1
                if r.failure_reason == "dropped":
                    self.skipped.add(b)
                    self._write(led.mark_uncertain(self.color(b), self.t, b, r.failure_reason))
                elif self.attempts[b] >= int(self.cfg["retry"]["pick_attempts"]):
                    self.skipped.add(b)
                    self.emit("block_skipped", block=b, color=self.color(b), attempts=self.attempts[b],
                              reason=r.failure_reason)
            self.purpose = sk.params.get("purpose") if self.holding is not None else None
        elif sk.name == "place":
            if r.status == "succeeded":
                self.holding = None
                c = self.color(b)
                tray = sk.params.get("tray")
                if tray == "placement":
                    self.delivered_blocks[b] = c
                    self._write(led.verified(c, +1, self.t, b))
                elif b in self.delivered_blocks:
                    del self.delivered_blocks[b]
                    self._write(led.verified(c, -1, self.t, b))
                self.purpose = None
                self._report_over()
            elif r.status == "cancelled":
                self.holding = b
            else:   # uncertain / failed (dropped)
                self.holding = None
                self.purpose = None
                self._write(led.mark_uncertain(self.color(b), self.t, b, r.failure_reason))
                if r.failure_reason == "dropped":
                    self.skipped.add(b)
        elif sk.name == "home":
            pass

    def _report_over(self):
        led = self.ledger
        for c in led.colors():
            extra = led.accepted_extra.get(c, 0)
            over = led.over(c)
            key = (led.revision, c, led.delivered.get(c, 0))
            if (over > 0 or (extra > 0 and led.delivered.get(c, 0) > led.target.get(c, 0))) and key not in self._reported_over:
                self._reported_over.add(key)
                self.emit("over_delivery", color=c, over=over, accepted=extra, delivered=led.delivered.get(c, 0),
                          target=led.target.get(c, 0), action="retrieve" if over > 0 else "kept (user approved)")

    def _track_cancel(self):
        pc = self._pending_cancel
        if pc is None:
            return
        if pc["t_arm_stopped"] is None and self.obs.qd_max < float(self.cfg["stop"]["arm_stopped_qd"]) \
                and self.t > pc["t_change"]:
            pc["t_arm_stopped"] = self.t
        if pc["t_done"] is not None and pc["t_arm_stopped"] is not None:
            rec = {"skill": pc["skill"], "phase": pc["phase"], "block": pc["block"], "rev": pc["rev"],
                   "t_change": round(pc["t_change"], 4),
                   "arm_stopped_s": round(pc["t_arm_stopped"] - pc["t_change"], 4),
                   "abort_done_s": round(pc["t_done"] - pc["t_change"], 4)}
            self.cancels.append(rec)
            self.emit("stop_latency", **rec)
            self._pending_cancel = None

    # ---- display -----------------------------------------------------------------------------------------------------
    def view(self) -> dict:
        led = self.ledger
        sk = self.skill
        return {"state": self.state, "revision": led.revision, "ledger": led.snapshot(),
                "holding": None if self.holding is None else {"block": self.holding, "color": self.color(self.holding)},
                "skill": None if sk is None else {"name": sk.name, "phase": sk.phase, "block": sk.params.get("block"),
                                                  "purpose": sk.params.get("purpose"), "color": sk.params.get("color"),
                                                  "nth": sk.params.get("nth")},
                "question": self.question, "placement": self._counts_in("placement") if self.obs else {},
                "storage": self._counts_in("storage") if self.obs else {},
                "colors_in_scene": sorted(set(self.world.colors)), "events": len(self.events)}

    def panel_lines(self) -> list:
        """(text, kind) lines for the simulator panels; kind: text / dim / ask."""
        v = self.view()
        led = v["ledger"]
        out = [(f"state {v['state']}   order rev {v['revision']}", "dim")]
        for c in sorted(set(led["target"]) | set(led["delivered"]), key=lambda c: list(led["target"]).index(c)
                        if c in led["target"] else 99):
            t, d = led["target"].get(c, 0), led["delivered"].get(c, 0)
            extra = f"  need {t - d}" if t > d else (f"  over {d - t}" if d > t else "  done")
            out.append((f"{c:<11} order {t}  delivered {d}{extra}", "text"))
        h = v["holding"]
        out.append((f"holding: {h['color']} (#{h['block']})" if h else "holding: nothing", "dim"))
        s = v["skill"]
        out.append((f"skill: {s['name']} / {s['phase']}" + (f"  ({s['purpose']})" if s.get("purpose") else "")
                    if s else "skill: -", "dim"))
        if v["question"]:
            q = v["question"]
            out.append((f"Q{q['id']}: {q['text']}", "ask"))
        return out
