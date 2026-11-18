"""Session glue shared by the Isaac and MuJoCo sessions: builds the executor (and an optional scenario) for a scene.

  tr = TaskRunner(ctrl, scene_cfg, block_init_pos, colors, dt, log, scenario="configs/tasks/x.yaml",
                  event_log="outputs/task/x.jsonl")
  each unpaused control frame, before ctrl.joint_command():   tr.step(obs)
  tr.done                         scenario finished (or timed out)
  tr.result(obs, "mujoco")        scenario result dict (checks)

While a task runs the executor is the only control source: the sessions drop keyboard / mouse / panel motion input.
"""

from __future__ import annotations

import math

import numpy as np

from agentic_sim.config import load
from agentic_sim.scene_layout import palette_by_name, trays
from agentic_sim.task.executor import TaskExecutor
from agentic_sim.task.scenario import TaskScenario
from agentic_sim.task.skills import Obs, World

TASK_CONFIG = "configs/task.yaml"


def default_color_names(scene: dict, n: int) -> list:
    """Colour names of the default layout: block 0 is the scene's red block, blocks 1.. take palette colours 1.."""
    names = list(palette_by_name(scene["block"]["extra_palette"]))
    return ["red"] + [names[i % len(names)] for i in range(1, n)]


def scenario_colors(path: str | None) -> list | None:
    return list(load(path)["blocks"]) if path else None


def quat_yaw(wxyz) -> float:
    w, x, y, z = (float(c) for c in wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class TaskRunner:
    def __init__(self, ctrl, scene_cfg: dict, block_init_pos, colors: list, dt: float, log=print,
                 scenario: str | None = None, event_log: str | None = None, task_cfg: str = TASK_CONFIG):
        cfg = load(task_cfg)
        tr = trays(scene_cfg)
        from agentic_sim.config import v
        world = World(trays={"storage": tr["storage_pink"], "placement": tr["placement_beige"]},
                      edge=float(v(scene_cfg["block"]["edge"])), colors=list(colors),
                      init_pos=[np.asarray(p, float) for p in block_init_pos], home=np.asarray(ctrl.p.home_ee, float),
                      slots=[list(map(float, s)) for s in cfg["placement_slots"]])
        self.ex = TaskExecutor(ctrl, world, cfg, dt, log=log, event_log=event_log)
        self.scenario = TaskScenario(load(scenario), self.ex, log) if scenario else None
        self.scenario_path = scenario
        self.log = log
        log(f"task executor ready: blocks {colors}" + (f", scenario {scenario}" if scenario else ""))

    def step(self, obs: Obs):
        if self.scenario is not None:
            self.scenario.step(obs)
        self.ex.step(obs)

    @property
    def done(self) -> bool:
        return self.scenario is not None and self.scenario.done

    def result(self, obs: Obs, engine: str, extra: dict | None = None) -> dict:
        return self.scenario.result(obs, engine, extra)

    def close(self):
        self.ex.close()
