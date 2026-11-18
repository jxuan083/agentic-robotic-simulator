"""task_* protocol commands, shared by the Isaac and MuJoCo servers (mixed into their command handlers).

  task_order  {counts: {red: 1, blue: 2}, mode: total|add, request_id?}   the first order hands the arm to the executor
  task_answer {id, choice, color?, xy?, request_id?}                       reply to the open question
  task_state  {}                                              ledger, holding, skill / phase, question
  task_events {since?}                                        important executor events from index `since` on
  task_stop   {}                                              give the arm back to keyboard / mouse

request_id: a request repeated with the same id (a transport retry) is applied once; the first response is returned
again with "duplicate": true. A deliberate repeat (e.g. the same "add one blue" said twice) uses a new id.

The handler needs self.session (with start_task / stop_task / task), self._require_session() and self._job_running().
"""

from __future__ import annotations

from agentic_sim.task.ledger import MODES


# events a client (chat layer) needs; phase / skill_start events stay on the simulator panel
REPORT_EVENTS = ("order", "question", "question_dropped", "answer", "answer_rejected", "over_delivery", "verified",
                 "uncertain", "dropped", "block_skipped", "task_done", "skill_done")


class TaskCommands:
    def _dedup(self, request_id, fn):
        if request_id is None:
            return fn()
        seen = self.__dict__.setdefault("_task_requests", {})
        if request_id in seen:
            return dict(seen[request_id], duplicate=True)
        out = fn()
        seen[request_id] = out
        if len(seen) > 1000:                       # bounded: oldest ids first out
            seen.pop(next(iter(seen)))
        return out

    def _task_session(self):
        self._require_session()
        if self._job_running():
            raise RuntimeError(f"acceptance run {self.job['scenario']} is in progress")
        return self.session

    def cmd_task_order(self, counts, mode="total", request_id=None):
        return self._dedup(request_id, lambda: self._task_order(counts, mode))

    def _task_order(self, counts, mode):
        s = self._task_session()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        bad = {c: n for c, n in counts.items() if not isinstance(n, int) or isinstance(n, bool) or n < 0}
        if bad:
            raise ValueError(f"counts must be non-negative integers: {bad}")
        if s.task is None:
            s.start_task()
        known = set(s.task.ex.world.colors)
        unknown = [c for c, n in counts.items() if n > 0 and c not in known]
        s.task.ex.order(counts, mode)
        out = {"queued": True, "mode": mode, "counts": counts, "colors_in_scene": sorted(known)}
        if unknown:
            out["note"] = f"no block of colour {unknown} in this scene: the executor will report a shortage"
        return out

    def cmd_task_answer(self, id, choice, color=None, xy=None, request_id=None):  # noqa: A002  (protocol name)
        return self._dedup(request_id, lambda: self._task_answer(id, choice, color, xy))

    def _task_answer(self, id, choice, color, xy):  # noqa: A002
        s = self._task_session()
        if s.task is None:
            raise RuntimeError("no task running (send task_order first)")
        kw = {k: v for k, v in (("color", color), ("xy", xy)) if v is not None}
        s.task.ex.answer(id, choice, **kw)
        return {"queued": True}

    def cmd_task_state(self):
        self._require_session()
        s = self.session
        if s.task is None:   # no order yet: the colours on the table (a chat layer needs them before the first order)
            from agentic_sim.task.runner import default_color_names
            colors = s.block_color_names or default_color_names(s.scene_cfg, int(s.settings()["blocks"]))
            stock = {}
            for c in colors:
                stock[c] = stock.get(c, 0) + 1
            return {"active": False, "colors_in_scene": sorted(stock), "storage": stock,
                    "ledger": {"revision": 0, "target": {}, "delivered": {}, "need": {}, "over": {}}}
        return {"active": True, **s.task.ex.view(), "cancels": s.task.ex.cancels}

    def cmd_task_events(self, since=0):
        self._require_session()
        s = self.session
        if s.task is None:
            return {"active": False, "next": 0, "events": []}
        ev = s.task.ex.events
        since = max(0, int(since))
        out = []
        for i in range(since, len(ev)):
            e = ev[i]
            if e["kind"] not in REPORT_EVENTS:
                continue
            if e["kind"] == "skill_done" and e.get("status") not in ("failed", "uncertain"):
                continue
            out.append(dict(e, i=i))
        return {"active": True, "next": len(ev), "events": out}

    def cmd_task_stop(self):
        s = self._task_session()
        s.stop_task()
        return {"active": False}
