"""Turn (backend, experiment, mode, options) into the command that runs it once (no server). Pure Python.

  isaac   teleop      docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py --scene ... [options]
          acceptance  tools/run_acceptance.sh <scenario> [--no-video]          (app arguments via ISAAC_ARGS)
  mujoco  teleop      uv run scripts/mujoco_ee_teleop.py --scene ... [options]
          acceptance  tools/run_acceptance_mujoco.sh <scenario> [--no-video]   (app arguments via MUJOCO_ARGS)
  isaac   record      docker/isaac/run.sh ... scripts/isaac_ee_teleop.py --scene ... --block_colors ... --target_color ...
                      --record_hz ... --record <dir>   (demonstration capture, issue #13; build_record)
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from dataclasses import dataclass, field

from agentic_sim.experiments import Experiment

MODES = ("teleop", "acceptance")
IMPLEMENTED_BACKENDS = ("isaac", "mujoco")


class LaunchError(ValueError):
    pass


@dataclass
class RecordOptions:
    """Demonstration capture (issue #13) from the launcher. Defaults: the lab layout with a yellow block, live
    perception off (a keyboard / mouse demonstration does not need it; it costs about half the frame rate)."""
    block_colors: list = field(default_factory=lambda: ["red", "yellow", "blue"])
    target_color: str = "yellow"
    record_hz: float = 20.0
    out_dir: str = "outputs/demos"
    wrist_lens: str | None = None   # None = the scene default ("real" in lab_20261120)
    perception: dict = field(default_factory=dict)   # agentic_sim.perception_cli.options(): {} = off
    run_id: str | None = None       # labels this launch's recorder events (the launcher reads only its own)
    screen_video_dir: str | None = None   # right-column screen video control directory (the launcher runs its agent)


RECORD_EXPERIMENT = "lab_20261120"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTROL_HZ = 60                     # control steps per simulated second (agentic_sim.isaac.session.CONTROL_DT)


def record_rate(hz: float) -> tuple:
    """(stride in control steps, actual recorded rate): the recorder keeps every stride-th control step."""
    stride = max(1, int(round(CONTROL_HZ / float(hz))))
    return stride, CONTROL_HZ / stride


def record_dir(path: str, root: str = REPO_ROOT) -> str:
    """The capture output directory relative to the repo. The Isaac container mounts only the repo (at its working
    directory), so a directory outside it would be written inside the container and lost; refused."""
    base = Path(root).resolve()
    full = (base / path).resolve()
    if full == base or not full.is_relative_to(base):
        raise LaunchError(f"錄製輸出 {path} 必須在 repo 裡的子目錄（Isaac 容器只掛載 repo {root}）")
    return full.relative_to(base).as_posix()


@dataclass
class RunOptions:
    random_blocks: bool = False
    random_seed: int | None = None
    random_region: str = "storage"
    random_config: str = "configs/random_blocks.yaml"
    random_episode: int = 0           # advanced on Reset; seed + episode reproduce the layout
    live_perception: bool = False
    perception_camera: str = "both"
    perception_hz: float = 2.0
    perception_json: str | None = None
    perception_stage: str = "estimate"   # "capture": capture only (benchmark)
    perception_delay: float = 0.0        # test only: late estimates
    perception_fault: str | None = None  # test only: CAMERA:AFTER:DURATION failing estimates
    annotations: str = "on"              # camera cells start annotated ("on") or raw ("off")
    focus: bool = False
    color_dr: bool = False
    color_dr_seed: int | None = None
    blocks: int = 1                 # number of blocks in the scene (1-8)
    block_colors: list | None = None  # colour name per block (task scenes); sets the block count
    wrist_lens: str | None = None   # wrist camera image: "real" (raw-looking, distorted) | "pinhole"; None = scene default
    video: bool = True              # acceptance only: record the app window
    screen_video_dir: str | None = None   # teleop: right-column screen video control directory (agentic_sim.screen_video)
    task: str | None = None         # server: task scenario run in the loaded scene (finals: tools/finals.py chat)

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
    from agentic_sim.perception_cli import args as perception_args
    args += perception_args(opts)
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
    if opts.random_blocks:
        args += ["--random-blocks", "--random-region", opts.random_region, "--random-config", opts.random_config]
        if opts.random_seed is not None:
            args += ["--random-seed", str(opts.random_seed)]
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
    if not (rec.record_hz == rec.record_hz and 0 < rec.record_hz <= CONTROL_HZ):   # also refuses NaN
        out.append(f"錄製頻率 {rec.record_hz:g} Hz：需要大於 0、最多 {CONTROL_HZ} Hz（控制頻率）")
    try:
        record_dir(rec.out_dir)
    except LaunchError as e:
        out.append(str(e))
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
    # perception only when asked for, and then with every option spelled out (camera, rate, JSON path)
    args = _app_args(exp, RunOptions(block_colors=list(rec.block_colors), **rec.perception), backend)
    if rec.wrist_lens:
        args += ["--wrist_lens", rec.wrist_lens]
    args += ["--target_color", rec.target_color, "--record_hz", f"{rec.record_hz:g}", "--record", record_dir(rec.out_dir)]
    if rec.run_id:
        args += ["--record_run_id", rec.run_id]
    if rec.screen_video_dir:
        args += ["--screen_video_dir", rec.screen_video_dir]
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
        sv = ["--screen_video_dir", opts.screen_video_dir] if opts.screen_video_dir else []
        if backend == "mujoco":
            return Command(["uv", "run", "scripts/mujoco_ee_teleop.py"] + _app_args(exp, opts, backend) + sv)
        return Command(["docker/isaac/run.sh", "/isaac-sim/python.sh", "scripts/isaac_ee_teleop.py"]
                       + _app_args(exp, opts) + sv)

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
