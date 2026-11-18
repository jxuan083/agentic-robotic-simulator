"""task_* protocol commands, shared by the Isaac and MuJoCo servers (mixed into their command handlers).

  task_order  {counts: {red: 1, blue: 2}, mode: total|add}   the first order hands the arm to the executor
  task_answer {id, choice, color?, xy?}                       reply to the open question
  task_state  {}                                              ledger, holding, skill / phase, question
  task_stop   {}                                              give the arm back to keyboard / mouse

The handler needs self.session (with start_task / stop_task / task), self._require_session() and self._job_running().
"""

from __future__ import annotations

from agentic_sim.task.ledger import MODES


class TaskCommands:
    def _task_session(self):
        self._require_session()
        if self._job_running():
            raise RuntimeError(f"acceptance run {self.job['scenario']} is in progress")
        return self.session

    def cmd_task_order(self, counts, mode="total"):
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

    def cmd_task_answer(self, id, choice, color=None, xy=None):  # noqa: A002  (protocol field name)
        s = self._task_session()
        if s.task is None:
            raise RuntimeError("no task running (send task_order first)")
        kw = {k: v for k, v in (("color", color), ("xy", xy)) if v is not None}
        s.task.ex.answer(id, choice, **kw)
        return {"queued": True}

    def cmd_task_state(self):
        self._require_session()
        s = self.session
        if s.task is None:
            return {"active": False}
        return {"active": True, **s.task.ex.view(), "cancels": s.task.ex.cancels}

    def cmd_task_stop(self):
        s = self._task_session()
        s.stop_task()
        return {"active": False}
