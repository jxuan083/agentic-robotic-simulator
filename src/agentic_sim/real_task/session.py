"""The real arm behind the simulators' task protocol (agentic_sim.task.server_cmds): the chat layer (tools/chat.py,
tools/chat_web.py --backend real) talks to it unchanged.

  session = RealSession(arm, log=print)        # scans the trays once at park (colours for the chat before an order)
  handler = RealHandler(session)               # cmd_task_* + status / pause / resume / scan (+ fake_scene)
  every frame (~30 Hz):  server.poll(); session.step()
"""

from __future__ import annotations

from types import SimpleNamespace

from agentic_sim.task.server_cmds import TaskCommands
from agentic_sim.real_task.executor import RealTaskExecutor, real_task_config
from agentic_sim.real_task.tracker import Tracker


class RealSession:
    def __init__(self, arm, log=print, event_log: str | None = None, task_cfg: str = "configs/task.yaml",
                 standby_after: int | None = None):
        self.arm, self.log, self.event_log, self.task_cfg = arm, log, event_log, task_cfg
        self.standby_after = standby_after
        self.tracker = Tracker()
        self.task = None
        self.paused = False
        self.scene_cfg = {}
        self.scan()

    # ---- what the task commands use -------------------------------------------------------------------
    @property
    def block_color_names(self) -> list:
        return [c for c, t in zip(self.tracker.colors, self.tracker.tray) if t == "storage"]

    def settings(self) -> dict:
        return {"blocks": len(self.block_color_names)}

    def scan(self):
        """D455 look at park (no task running)."""
        if self.task is not None:
            raise RuntimeError("a task is running: the executor takes its own observations")
        blocks, clear, reason = self.arm.scan()
        self.tracker.update(blocks)
        self.log(f"D455: storage {self.tracker.counts('storage')}, placement {self.tracker.counts('placement')}"
                 + ("" if clear else f" (unclear: {reason})"))
        return {"storage": self.tracker.counts("storage"), "placement": self.tracker.counts("placement"),
                "clear": clear, "reason": reason}

    def start_task(self):
        ex = RealTaskExecutor(self.arm, self.tracker, real_task_config(self.arm, self.task_cfg, self.standby_after),
                              log=self.log, event_log=self.event_log)
        ex.hooks.append(lambda e: e["kind"] == "block_removed" and self.tracker.remove(e["block"]))
        self.task = SimpleNamespace(ex=ex)
        self.log("task started (real arm)")

    def stop_task(self):
        if self.task is not None:
            self.task.ex.close()
        self.task = None

    def step(self):
        if self.task is not None and not self.paused:
            ex = self.task.ex
            ex.step(ex.make_obs())

    def pause(self):
        self.paused = True
        self.arm.pause()

    def resume(self):
        self.paused = False
        self.arm.resume()


class RealHandler(TaskCommands):
    def __init__(self, session: RealSession, fake: bool = False):
        self.session, self.fake, self.job = session, fake, None

    def _require_session(self):
        return self.session

    def _job_running(self) -> bool:
        return False

    def cmd_status(self):
        s = self.session
        return {"backend": "real" + ("-fake" if self.fake else ""), "experiment": "real-right", "options": None,
                "paused": s.paused, "task": s.task is not None}

    def cmd_pause(self):
        self.session.pause()
        return {}

    def cmd_resume(self):
        self.session.resume()
        return {}

    def cmd_scan(self):
        return self.session.scan()

    def cmd_task_continue(self):
        """The operator is done at the standby point (e.g. took the shortage colour away)."""
        s = self.session
        if s.task is None or not s.task.ex.waiting_operator:
            raise RuntimeError("the arm is not waiting for the operator")
        s.task.ex.operator_continue()
        return {"continued": True}

    def cmd_fake_scene(self, op, color, xy=None):
        """--fake only: the operator takes a block away (op remove) or puts one in the pink tray (op add, xy m)."""
        if not self.fake:
            raise RuntimeError("fake_scene is for the simulated table only")
        if op == "remove":
            return {"removed": self.session.arm.remove(color)}
        if op == "add":
            self.session.arm.add(color, xy or (0.25, 0.15))
            return {"added": color}
        raise ValueError("op is remove or add")
