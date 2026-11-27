"""TaskExecutor on the real arm: the decisions (ledger, orders, questions, shortage, no_return) are the executor's own;
only what touches the world is replaced:

  skills       Pick / Place / Home / Hold / Observe -> agentic_sim.real_task.skills (hardware routines in a thread)
  _in()        tray membership from the last D455 scan (Tracker), not from simulator positions
  associate()  D455 items already carry their tracker ID
  reachable()  the arm's safe grasp zone + IK (RealArm) instead of the simulator's EE controller
  slots        the beige slots P1-P4 of the rig (configs/real_slots.json via the D455 pose); the plush toy has its
               own slot P0 (configs/real_plush.json) and is picked by its ACT policy (pick param method "act")

Always the finals settings: snapshot observations (choices only from the last look, taken at park), shortage
options substitute / wait, no_return (nothing is taken back; a held block the order no longer needs is delivered or
held until removed on site).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from agentic_sim.config import load
from agentic_sim.task.executor import TaskExecutor
from agentic_sim.task.skills import Obs, World
from agentic_sim.real_task.skills import SKILLS
from agentic_sim.real_task.tracker import EDGE_M, PLUSH

AWAY = np.array([9.0, 9.0, 0.0])   # obs position of a block that is in no tray (taken away / not seen)


class CtrlStub:
    """The few controller fields the executor reads outside its skills."""

    def __init__(self, arm):
        self.arm = arm
        self.pan = np.array([*arm.pan_xy, 0.0], float)
        self.p = SimpleNamespace(gripper_open=1.0, gripper_closed=0.0, home_ee=arm.home_xyz)

    @property
    def state(self):
        at_park = self.arm.at_park()
        return SimpleNamespace(pos=np.asarray(self.arm.home_xyz if at_park else self.arm.ee(), float),
                               gripper=1.0 if self.arm.gripper_open() else 0.0)


def real_task_config(arm, path: str = "configs/task.yaml", standby_after: int | None = None) -> dict:
    cfg = load(path)
    cfg["standby_after"] = standby_after
    cfg["no_return"] = True
    cfg["observation"] = {**(cfg.get("observation") or {}), "snapshot": True}
    cfg["shortage"] = {**cfg["shortage"], "options": ["substitute", "wait"]}
    cfg["standby"] = {**(cfg.get("standby") or {}), "observe_after_s": 0.2}
    cfg["placement_slots"] = [list(map(float, s)) for s in arm.slots]
    return cfg


class RealTaskExecutor(TaskExecutor):
    def __init__(self, arm, tracker, cfg: dict, dt: float = 1 / 30, log=print, event_log: str | None = None):
        world = World(trays={"storage": {}, "placement": {}}, edge=EDGE_M, colors=tracker.colors, init_pos=[],
                      home=np.asarray(arm.home_xyz, float), slots=[list(s) for s in cfg["placement_slots"]])
        super().__init__(CtrlStub(arm), world, cfg, dt, log=log, event_log=event_log)
        self.arm, self.tracker = arm, tracker
        self.camera = "d455"        # observations come from Observe skills (the D455 at park)
        self.t_origin = time.monotonic()
        # Finals: after this many verified deliveries the arm goes to standby (park) and takes no new look until
        # the operator is done (operator_continue): there someone takes the shortage colour away (11/22 rules).
        self.standby_after = cfg.get("standby_after")
        self.waiting_operator = False
        self.hooks.append(self._standby_hook)

    def _standby_hook(self, e: dict):
        if e["kind"] == "verified" and self.standby_after and len(self.delivery_seq) == int(self.standby_after) \
                and not self.waiting_operator:
            self.waiting_operator = True
            self.standby()
            self.emit("waiting_operator", delivered=len(self.delivery_seq))

    def operator_continue(self):
        if self.waiting_operator:
            self.waiting_operator = False
            self.standby_after = None           # once per task
            if self.inventory is not None:
                self.inventory["stale"] = True
            self.emit("operator_continue")

    def _look(self, reason: str):
        if self.waiting_operator:
            self.state = "waiting"
            return None
        return super()._look(reason)

    def _start(self, cls, **params):
        if cls.name == "pick" and "block" in params:     # shown in the chat's progress line
            params["method"] = "act" if self.tracker.colors[params["block"]] == PLUSH else "program"
        return super()._start(SKILLS.get(cls.name, cls), **params)

    def free_slot(self, b: int):
        if self.tracker.colors[b] == PLUSH:
            xy = getattr(self.arm, "plush_slot", None)
            return list(map(float, xy)) if xy is not None and self._free(xy, ignore=b, radius=0.06) else None
        return super().free_slot(b)

    def _in(self, b: int, tray: str) -> bool:
        return b < len(self.tracker) and self.tracker.tray[b] == tray

    def associate(self, items: list) -> tuple:
        blocks, poses = {}, {}
        for it in items:
            b = it["block"]
            if b == self.holding or b in self.delivered_blocks:
                continue
            blocks.setdefault(it["color"], []).append(b)
            poses[b] = {"xy": list(it["xy"]), "yaw": float(it["yaw"]), "color": it["color"], "source": it["source"]}
        return blocks, poses, []

    def reachable(self, xy) -> bool:
        tr = self.tracker
        if any(c == PLUSH and abs(tr.pos[b][0] - xy[0]) < 1e-3 and abs(tr.pos[b][1] - xy[1]) < 1e-3
               for b, c in enumerate(tr.colors)):
            return True            # the plush ACT was shown where the plush is put out (no program-grasp zone)
        key = (round(float(xy[0]), 3), round(float(xy[1]), 3))
        if key not in self._reach_cache:
            self._reach_cache[key] = bool(self.arm.reachable(key))
        return self._reach_cache[key]

    def make_obs(self) -> Obs:
        ee = np.asarray(self.arm.ee(), float)
        tr = self.tracker
        blocks = [ee.copy() if b == self.holding else (tr.pos[b] if tr.tray[b] else AWAY) for b in range(len(tr))]
        return Obs(t=time.monotonic() - self.t_origin, ee=ee, qd_max=0.0, jaw=0.0, blocks=blocks,
                   block_yaw=list(tr.yaw), block_speed=[0.0] * len(tr))
