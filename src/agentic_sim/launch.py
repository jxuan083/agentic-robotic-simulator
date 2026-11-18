"""Turn (backend, experiment, mode, options) into the command that runs it once (no server). Pure Python.

  isaac   teleop      docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py --scene ... [options]
          acceptance  tools/run_acceptance.sh <scenario> [--no-video]          (app arguments via ISAAC_ARGS)
  mujoco  teleop      uv run scripts/mujoco_ee_teleop.py --scene ... [options]
          acceptance  tools/run_acceptance_mujoco.sh <scenario> [--no-video]   (app arguments via MUJOCO_ARGS)
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from agentic_sim.experiments import Experiment

MODES = ("teleop", "acceptance")
IMPLEMENTED_BACKENDS = ("isaac", "mujoco")


class LaunchError(ValueError):
    pass


@dataclass
class RunOptions:
    focus: bool = False
    color_dr: bool = False
    color_dr_seed: int | None = None
    blocks: int = 1                 # number of blocks in the scene (1-8)
    video: bool = True              # acceptance only: record the app window

    def merged(self, forced: dict) -> "RunOptions":
        """Options with a scenario's forced options applied."""
        d = dict(self.__dict__)
        d.update(forced)
        return RunOptions(**d)


@dataclass
class Command:
    argv: list
    env: dict = field(default_factory=dict)  # extra environment variables
    notes: list = field(default_factory=list)  # things the user should know (e.g. forced options)

    def shell(self) -> str:
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        return (env + " " if env else "") + " ".join(shlex.quote(a) for a in self.argv)


def _app_args(exp: Experiment, opts: RunOptions, backend: str = "isaac") -> list:
    args = ["--scene", exp.scene, "--teleop", exp.teleop]
    if exp.color_dr:
        args += ["--color_dr_config", exp.color_dr]
    if opts.focus and backend == "isaac":
        args.append("--focus")
    if int(opts.blocks) != 1:
        args += ["--blocks", str(int(opts.blocks))]
    if opts.color_dr:
        if not exp.color_dr:
            raise LaunchError(f"experiment {exp.name!r} has no colour DR config")
        args.append("--color_dr")
        if opts.color_dr_seed is not None:
            args += ["--color_dr_seed", str(int(opts.color_dr_seed))]
    return args


def build(backend: str, exp: Experiment, mode: str, opts: RunOptions | None = None,
          scenario: str | None = None) -> Command:
    opts = opts or RunOptions()
    if not exp.supports(backend):
        raise LaunchError(f"experiment {exp.name!r} does not support backend {backend!r} (supports {list(exp.backends)})")
    if backend not in IMPLEMENTED_BACKENDS:
        raise LaunchError(f"backend {backend!r} is not implemented yet")
    if mode not in MODES:
        raise LaunchError(f"unknown mode {mode!r} (known: {list(MODES)})")

    if mode == "teleop":
        if scenario:
            raise LaunchError("a scenario is only used in acceptance mode")
        if backend == "mujoco":
            return Command(["uv", "run", "scripts/mujoco_ee_teleop.py"] + _app_args(exp, opts, backend))
        return Command(["docker/isaac/run.sh", "/isaac-sim/python.sh", "scripts/isaac_ee_teleop.py"]
                       + _app_args(exp, opts))

    if not scenario:
        raise LaunchError(f"acceptance mode needs a scenario "
                          f"(available: {[a.scenario for a in exp.acceptance_for(backend)]})")
    entry = exp.scenario(scenario, backend)
    notes = []
    if entry.options:
        notes.append(f"scenario {scenario} forces options {entry.options}")
    run = opts.merged(entry.options)
    app_args = " ".join(shlex.quote(a) for a in _app_args(exp, run, backend))
    if backend == "mujoco":
        argv = ["tools/run_acceptance_mujoco.sh", scenario] + ([] if run.video else ["--no-video"])
        return Command(argv, {"MUJOCO_ARGS": app_args, "ACCEPT_OUT": f"outputs/acceptance/mujoco/{exp.name}"}, notes)
    argv = ["tools/run_acceptance.sh", scenario] + ([] if run.video else ["--no-video"])
    env = {"ISAAC_ARGS": app_args, "ACCEPT_OUT": f"outputs/acceptance/{exp.name}"}
    return Command(argv, env, notes)
