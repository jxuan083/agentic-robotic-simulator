"""Turn (backend, experiment, mode, options) into the command that runs it once (no server). Pure Python.

  isaac   teleop      docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py --scene ... [options]
          acceptance  tools/run_acceptance.sh <scenario> [--no-video]          (app arguments via ISAAC_ARGS)
  mujoco  teleop      uv run scripts/mujoco_ee_teleop.py --scene ... [options]
          acceptance  tools/run_acceptance_mujoco.sh <scenario> [--no-video]   (app arguments via MUJOCO_ARGS)
  isaac   record      docker/isaac/run.sh ... scripts/isaac_ee_teleop.py --scene ... --block_colors ... --target_color ...
                      --record_hz ... --record <dir>   (demonstration capture, issue #13; build_record)
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
class RecordOptions:
    """Demonstration capture (issue #13) from the launcher. Defaults: the lab layout with a yellow block."""
    block_colors: list = field(default_factory=lambda: ["red", "yellow", "blue"])
    target_color: str = "yellow"
    record_hz: float = 20.0
    out_dir: str = "outputs/demos"
    wrist_lens: str | None = None   # None = the scene default ("real" in lab_20261120)


RECORD_EXPERIMENT = "lab_20261120"


@dataclass
class RunOptions:
    focus: bool = False
    color_dr: bool = False
    color_dr_seed: int | None = None
    blocks: int = 1                 # number of blocks in the scene (1-8)
    block_colors: list | None = None  # colour name per block (task scenes); sets the block count
    wrist_lens: str | None = None   # wrist camera image: "real" (raw-looking, distorted) | "pinhole"; None = scene default
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
    if opts.block_colors:
        args += ["--block_colors", ",".join(opts.block_colors)]
    elif int(opts.blocks) != 1:
        args += ["--blocks", str(int(opts.blocks))]
    if opts.color_dr:
        if not exp.color_dr:
            raise LaunchError(f"experiment {exp.name!r} has no colour DR config")
        args.append("--color_dr")
        if opts.color_dr_seed is not None:
            args += ["--color_dr_seed", str(int(opts.color_dr_seed))]
    return args


def palette_names(exp: Experiment) -> list:
    """Block colour names the experiment's scene accepts (its extra palette), read with PyYAML only."""
    from agentic_sim.config import load
    path = (load(exp.scene).get("block") or {}).get("extra_palette")
    return [c["name"] for c in load(path)["colors"]] if path else []


def check_record(exp: Experiment, rec: RecordOptions) -> list:
    """Problems of a capture setup, as messages for the operator ([] when it can start)."""
    out = []
    names = palette_names(exp)
    if not 1 <= len(rec.block_colors) <= 8:
        out.append(f"積木數量 {len(rec.block_colors)}：需要 1 到 8 顆")
    unknown = [c for c in rec.block_colors if names and c not in names]
    if unknown:
        out.append(f"不認得的積木顏色 {unknown}（可用：{', '.join(names)}）")
    if rec.target_color not in rec.block_colors:
        out.append(f"目標顏色 {rec.target_color} 不在積木配置 {','.join(rec.block_colors)} 裡")
    if not 0 < rec.record_hz <= 60:
        out.append(f"錄製頻率 {rec.record_hz:g} Hz：需要在 0 到 60 Hz 之間（控制頻率 60 Hz）")
    if rec.wrist_lens not in (None, "real", "pinhole"):
        out.append(f"腕部鏡頭模式 {rec.wrist_lens}：只有 real 或 pinhole")
    return out


def build_record(backend: str, exp: Experiment, rec: RecordOptions) -> Command:
    """The capture command (the same one the launcher runs and shows for copying)."""
    if backend != "isaac":
        raise LaunchError("錄示範目前只支援 Isaac（MuJoCo 沒有錄製）")
    if not exp.supports(backend):
        raise LaunchError(f"experiment {exp.name!r} does not support backend {backend!r}")
    problems = check_record(exp, rec)
    if problems:
        raise LaunchError("；".join(problems))
    args = _app_args(exp, RunOptions(block_colors=list(rec.block_colors)), backend)
    if rec.wrist_lens:
        args += ["--wrist_lens", rec.wrist_lens]
    args += ["--target_color", rec.target_color, "--record_hz", f"{rec.record_hz:g}", "--record", rec.out_dir]
    return Command(["docker/isaac/run.sh", "/isaac-sim/python.sh", "scripts/isaac_ee_teleop.py"] + args)


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
