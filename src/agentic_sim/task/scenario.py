"""Reproducible task scenarios (configs/tasks/*.yaml): initial order, order changes and answers triggered by executor
events (skill phases, verified deliveries, questions), and the checks evaluated at the end. Pure Python.

  name: change_during_descend
  blocks: [red, red, red, blue, blue, yellow]     # colour per block (scene layout slots)
  order: {red: 3}
  timeout_s: 150                                  # simulated seconds
  events:
    - when: {event: phase, skill: pick, color: red, nth: 2, phase: descend}
      after_s: 0.4                                # simulated delay after the trigger
      order: {total: {red: 1, blue: 2}}
    - when: {event: question, kind: held_item}
      after_s: 2.0
      answer: {choice: return}                    # or {choice: substitute, color: yellow}, {choice: place_at, xy: [..]}
  checks:
    - {final_placement: {red: 1, blue: 2}}
    - ...

Triggers fire on executor events, so they hit the same skill phase in every run and on every engine.
"""

from __future__ import annotations

import numpy as np

from agentic_sim.scene_layout import block_in_tray


def _match(when: dict, e: dict) -> bool:
    kind = when["event"]
    if kind == "verified":
        return (e["kind"] == "verified" and e["delta"] > 0 and e["color"] == when["color"]
                and e["ledger"]["delivered"].get(when["color"], 0) == int(when["delivered"]))
    if kind == "phase":
        return (e["kind"] == "phase" and e["skill"] == when["skill"] and e["phase"] == when["phase"]
                and (when.get("color") is None or e.get("color") == when["color"])
                and (when.get("nth") is None or e.get("nth") == int(when["nth"])))
    if kind == "question":
        return e["kind"] == "question" and (when.get("kind") is None or e.get("qkind") == when["kind"])
    raise ValueError(f"unknown trigger event {kind!r}")


class TaskScenario:
    def __init__(self, spec: dict, executor, log=print):
        self.spec, self.ex, self.log = spec, executor, log
        self.name = spec["name"]
        self.timeout = float(spec.get("timeout_s", 150))
        self.events_spec = list(spec.get("events", []))
        self.fired = [False] * len(self.events_spec)
        self.timers: list = []        # (t, index)
        self.applied: list = []       # (t, index, action)
        self.done = False
        self.timed_out = False
        self.t_done = None
        self.started = False
        w = executor.world
        self.init = [np.asarray(p, float) for p in w.init_pos]
        n = len(self.init)
        self.max_disp = [0.0] * n     # 3D, mm
        self.max_rise = [0.0] * n     # z, mm
        self.disp_before_answer: dict = {}   # block -> max mm before the first answer
        self.answered = False
        self.hold_ref = None          # (ee, block-ee) at the start of a hold
        self.hold_dev = {"ee_mm": 0.0, "block_rel_mm": 0.0, "frames": 0}
        self.retrieve_ref = None      # positions of the other placed blocks when a retrieval starts
        self.retrieve_dev = 0.0
        self.retrieve_seen = False
        executor.hooks.append(self._on_event)

    # ---- events -------------------------------------------------------------------------------------
    def _on_event(self, e: dict):
        for i, ev in enumerate(self.events_spec):
            if self.fired[i]:
                continue
            when = ev["when"]
            if _match(when, e):
                self.fired[i] = True
                self.timers.append((e["t"] + float(ev.get("after_s", 0.0)), i))
                self.log(f"scenario: trigger {i} {when} at t={e['t']:.3f}")
                if float(ev.get("after_s", 0.0)) == 0.0:
                    self._fire_due(e["t"])
        k = e["kind"]
        if k == "phase" and e["skill"] == "hold" and e["phase"] == "hold":
            self.hold_ref = "pending"
        if k == "answer":
            self.answered = True
            self.hold_ref = None
        if k == "skill_start" and e.get("skill") == "pick" and e.get("purpose") == "retrieve":
            self.retrieve_ref = "pending"
            self.retrieve_block = e["block"]
            self.retrieve_seen = True
        if k == "skill_done" and e.get("skill") == "place" and e.get("params", {}).get("purpose") == "return":
            self.retrieve_ref = None

    def _fire_due(self, t: float):
        due = [x for x in self.timers if x[0] <= t + 1e-9]
        self.timers = [x for x in self.timers if x[0] > t + 1e-9]
        for _, i in due:
            ev = self.events_spec[i]
            if "order" in ev:
                mode, counts = next(iter(ev["order"].items()))
                self.ex.order(counts, mode)
                self.applied.append((t, i, {"order": {mode: counts}}))
                self.log(f"scenario: order {mode} {counts} at t={t:.3f}")
            if "answer" in ev:
                q = self.ex.question
                a = dict(ev["answer"])
                choice = a.pop("choice")
                qid = q["id"] if q else -1
                self.ex.answer(qid, choice, **a)
                self.applied.append((t, i, {"answer": {"id": qid, "choice": choice, **a}}))
                self.log(f"scenario: answer Q{qid} {choice} {a} at t={t:.3f}")

    # ---- per frame (before the executor step) -------------------------------------------------------------
    def step(self, obs):
        if not self.started:
            self.started = True
            if self.spec.get("order"):     # no order: it comes from the chat / server
                self.ex.order(self.spec["order"], "total")
                self.log(f"scenario {self.name}: order {self.spec['order']}")
        self._fire_due(obs.t)
        for i, p in enumerate(obs.blocks):
            d = float(np.linalg.norm(p - self.init[i])) * 1000.0
            self.max_disp[i] = max(self.max_disp[i], d)
            self.max_rise[i] = max(self.max_rise[i], float(p[2] - self.init[i][2]) * 1000.0)
            if not self.answered:
                self.disp_before_answer[i] = max(self.disp_before_answer.get(i, 0.0), d)
        ex = self.ex
        if self.hold_ref == "pending":
            b = ex.holding
            self.hold_ref = (obs.ee.copy(), None if b is None else obs.blocks[b] - obs.ee)
        elif self.hold_ref is not None:
            ee0, rel0 = self.hold_ref
            self.hold_dev["ee_mm"] = max(self.hold_dev["ee_mm"], float(np.linalg.norm(obs.ee - ee0)) * 1000.0)
            b = ex.holding
            if rel0 is not None and b is not None:
                rel = obs.blocks[b] - obs.ee
                self.hold_dev["block_rel_mm"] = max(self.hold_dev["block_rel_mm"],
                                                    float(np.linalg.norm(rel - rel0)) * 1000.0)
            self.hold_dev["frames"] += 1
        if self.retrieve_ref == "pending":
            tray = ex.world.trays["placement"]
            self.retrieve_ref = {i: obs.blocks[i].copy() for i in range(len(obs.blocks))
                                 if i != self.retrieve_block and block_in_tray(obs.blocks[i], tray, ex.world.edge)}
        elif isinstance(self.retrieve_ref, dict):
            for i, p0 in self.retrieve_ref.items():
                self.retrieve_dev = max(self.retrieve_dev, float(np.linalg.norm(obs.blocks[i] - p0)) * 1000.0)
        if not self.done:
            if ex.state == "done" and not self.timers and all(self.fired):
                self.done, self.t_done = True, obs.t
            elif obs.t > self.timeout:
                self.done, self.timed_out, self.t_done = True, True, obs.t
                self.log(f"scenario {self.name}: timeout at t={obs.t:.1f} (state {ex.state})")

    # ---- checks -------------------------------------------------------------------------------------------------
    def checks(self, obs) -> list:
        ex, w = self.ex, self.ex.world
        led = ex.ledger.snapshot()
        out = []

        def add(name, ok, **detail):
            out.append({"name": name, "ok": bool(ok), **detail})

        placed = {}
        for b, p in enumerate(obs.blocks):
            if block_in_tray(p, w.trays["placement"], w.edge):
                placed[w.colors[b]] = placed.get(w.colors[b], 0) + 1
        add("finished", not self.timed_out, t_sim=round(float(obs.t), 2), state=ex.state)
        for c in self.spec.get("checks", []):
            key, arg = (c, None) if isinstance(c, str) else next(iter(c.items()))
            if key == "final_placement":
                add(key, placed == arg, expected=arg, observed=placed, observation="sim_truth")
            elif key == "ledger":
                ok = all(led[k] == v for k, v in arg.items())
                add(key, ok and ("done" in arg or led["done"]), expected=arg, ledger=led)   # done: false = over left
            elif key == "max_displacement":
                obs_mm = {b: round(self.max_disp[b], 2) for b in arg["blocks"]}
                add(key, all(v < float(arg["mm"]) for v in obs_mm.values()), limit_mm=arg["mm"], observed_mm=obs_mm)
            elif key == "not_lifted":
                b = arg["block"]
                add(key, self.max_rise[b] < float(arg["mm"]) and self.max_disp[b] < float(arg["mm"]),
                    block=b, max_rise_mm=round(self.max_rise[b], 2), max_disp_mm=round(self.max_disp[b], 2),
                    limit_mm=arg["mm"])
            elif key == "hold_still":
                d = self.hold_dev
                add(key, d["frames"] > 0 and d["ee_mm"] < float(arg["mm"]) and d["block_rel_mm"] < float(arg["mm"]),
                    limit_mm=arg["mm"], ee_mm=round(d["ee_mm"], 2), block_rel_mm=round(d["block_rel_mm"], 2),
                    frames=d["frames"])
            elif key == "no_drop":
                drops = [e for e in ex.events if e["kind"] == "dropped"]
                add(key, not drops, drops=drops)
            elif key == "no_stale_commands":
                changes = [(e["t"], e["rev"]) for e in ex.events if e["kind"] == "order"]
                stale = [(t, r, tc, rc) for tc, rc in changes for t, r in ex.cmd_log if t >= tc and r < rc]
                add(key, not stale, order_changes=changes, stale_frames=len(stale), first=stale[:3])
            elif key == "question":
                qs = [e for e in ex.events if e["kind"] == "question" and e.get("qkind") == arg["kind"]]
                ok = bool(qs) and all(qs[0].get(k) == v for k, v in arg.items() if k != "kind")
                add(key, ok, expected=arg, asked=qs[0] if qs else None)
            elif key == "over_delivery":
                ev = [e for e in ex.events if e["kind"] == "over_delivery" and e["color"] == arg["color"]]
                ok = bool(ev) and all(ev[0].get(k) == v for k, v in arg.items())
                add(key, ok, expected=arg, reported=ev[:2])
            elif key == "no_over_delivery_retrieval":
                add(key, not self.retrieve_seen, retrieved=self.retrieve_seen)
            elif key == "neighbours_still":
                add(key, self.retrieve_seen and self.retrieve_dev < float(arg["mm"]), limit_mm=arg["mm"],
                    observed_mm=round(self.retrieve_dev, 2), retrieval=self.retrieve_seen)
            elif key == "untouched_until_answer":
                obs_mm = {b: round(self.disp_before_answer.get(b, 0.0), 2) for b in arg["blocks"]}
                add(key, self.answered and all(v < float(arg["mm"]) for v in obs_mm.values()),
                    limit_mm=arg["mm"], observed_mm=obs_mm)
            elif key == "block_in":
                b, tray = arg["block"], arg["tray"]
                p = obs.blocks[b]
                inside = block_in_tray(p, w.trays[tray], w.edge)
                d = float(np.linalg.norm(p[:2] - self.init[b][:2])) * 1000.0
                add(key, inside and ("near_init_mm" not in arg or d < float(arg["near_init_mm"])), block=b, tray=tray,
                    inside=bool(inside), from_init_mm=round(d, 2), limit_mm=arg.get("near_init_mm"))
            elif key == "stop_latency":
                add(key, bool(ex.cancels), cancels=ex.cancels)
            elif key == "skills_ok":
                bad = [r for r in ex.results if r["status"] in ("failed", "uncertain")]
                add(key, not bad, failed=bad)
            else:
                raise ValueError(f"unknown check {key!r}")
        return out

    def result(self, obs, engine: str, extra: dict | None = None) -> dict:
        checks = self.checks(obs)
        return {"engine": engine, "scenario": self.name, "passed": sum(c["ok"] for c in checks),
                "checks_total": len(checks), "checks": checks, "ledger": self.ex.ledger.snapshot(),
                "applied": [{"t": round(t, 4), "event": i, **a} for t, i, a in self.applied],
                "stop_latency": self.ex.cancels, "skills": self.ex.results, "t_sim": round(float(obs.t), 3),
                **(extra or {})}
