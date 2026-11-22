"""MuJoCo version of the minimal desk scene with keyboard EE teleop for one SO-101 (runs on the host, no GPU needed).

    uv run scripts/mujoco_ee_teleop.py [--blocks 3] [--color_dr --color_dr_seed 7]
    uv run scripts/mujoco_ee_teleop.py --script configs/acceptance/keyboard_pick_place.yaml \
        --log out.csv --result out.json --exit_after_script
    uv run scripts/mujoco_ee_teleop.py --task configs/tasks/hold_return.yaml --task_log out.jsonl \
        --result out.json --exit_after_script         # task executor scenario (issue #5)

Same scene YAML, teleop config, key map and EE controller as the Isaac version (docs/mujoco-backend.md).
--headless runs without a window (scripts only; the keys still go through the same KeyState).
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from agentic_sim.config import load  # noqa: E402
from agentic_sim.mujoco_backend.model import ROBOT_CONFIG  # noqa: E402
from agentic_sim.mujoco_backend.session import MjSession, SessionConfig, session_config_with  # noqa: E402
from agentic_sim.teleop.keys import MJ_ACTION_KEYS, KeyState  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
from agentic_sim.random_blocks import add_arguments
add_arguments(parser)
from agentic_sim.perception_cli import add_arguments as add_perception_arguments
add_perception_arguments(parser)
parser.add_argument("--metrics_log", default=None, help="one JSON line per host second: UI FPS, RTF, perception "
                    "result age and latency (benchmarks)")
parser.add_argument("--screen_video_dir", default=None, help="control directory of the right-column screen video "
                    "(this script runs an agent in its own process when none is running; default: outputs/screen_video/control/<new id>)")
parser.add_argument("--metrics_run_id", default=None, help="id written into every --metrics_log line (benchmarks bind "
                    "their report to this run)")
parser.add_argument("--scene", default="configs/scenes/lab_current_minimal.yaml")
parser.add_argument("--teleop", default="configs/teleop.yaml")
parser.add_argument("--script", default=None, help="YAML input script for automated runs")
parser.add_argument("--log", default=None, help="CSV debug log path")
parser.add_argument("--result", default=None, help="JSON summary path (script / trajectory runs)")
parser.add_argument("--trajectory", default=None, help="scripted EE target sequence (configs/trajectories/*.yaml); "
                    "replaces keyboard / mouse input, ends the run with --exit_after_script")
parser.add_argument("--task", default=None, help="task scenario (configs/tasks/*.yaml): the task executor drives the "
                    "arm; ends the run with --exit_after_script")
parser.add_argument("--task_log", default=None, help="executor event log (JSONL)")
parser.add_argument("--block_colors", default=None, help="comma-separated colour names, one per block")
parser.add_argument("--exit_after_script", action="store_true")
parser.add_argument("--max_time", type=float, default=0.0, help="exit after N wall seconds (0 = run until closed)")
parser.add_argument("--color_dr", action="store_true", help="enable colour domain randomization (default off)")
parser.add_argument("--color_dr_seed", type=int, default=None)
parser.add_argument("--color_dr_config", default="configs/color_randomization.yaml")
parser.add_argument("--color_dr_log", default=None)
parser.add_argument("--blocks", type=int, default=None, help="number of blocks, 1-8 (default: the scene's count)")
parser.add_argument("--wrist_lens", choices=["real", "pinhole"], default=None,
                    help="wrist camera image: raw-looking (distorted) or ideal pinhole (default: the scene's lens_mode)")
parser.add_argument("--headless", action="store_true", help="no window (keyboard-only scripts)")
parser.add_argument("--hidden", action="store_true",
                    help="development checks only: invisible window, focus treated as present (not an acceptance run)")
parser.add_argument("--focus", action="store_true", help="start in the operator workspace (fullscreen, cameras below; "
                    "M toggles it)")
args = parser.parse_args()
try:
    from agentic_sim.perception_cli import parse_fault
    parse_fault(args.perception_fault)
except ValueError as e:
    parser.error(str(e))


def screen_video_dir(_d=[]):
    """Control directory of the right-column screen video, the same for every session of this run."""
    if not _d:
        import uuid
        from agentic_sim.screen_video import control_dir
        _d.append(args.screen_video_dir or control_dir(uuid.uuid4().hex[:12]))
    return _d[0]


def log(msg):
    print(f"[mujoco_teleop] {msg}", flush=True)


def main():
    keys = KeyState(actions=MJ_ACTION_KEYS)
    viewer = None
    if not args.headless:
        from agentic_sim.mujoco_backend.viewer import Viewer
        vc = load(ROBOT_CONFIG)["viewer"]
        viewer = Viewer(keys, vc["title"], vc["size"], log, visible=not args.hidden)
    if args.script and viewer is None:
        log("headless: scripted keys only")
    focused = (lambda: True) if args.hidden else (viewer.focused if viewer else None)
    agent = None
    if viewer is not None and not args.hidden:   # on the host already: record this window without tools/sim.py
        from agentic_sim.screen_video import AgentThread, agent_running, platform_support
        if platform_support()[0] and not agent_running(screen_video_dir()):   # Linux / X11 only
            agent = AgentThread(screen_video_dir(), log=log)
    session = MjSession(SessionConfig(
        scene=args.scene, teleop=args.teleop,
        random_blocks=args.random_blocks, random_seed=args.random_seed,
        random_region=args.random_region, random_config=args.random_config, color_dr_config=args.color_dr_config, color_dr=args.color_dr,
        color_dr_seed=args.color_dr_seed, color_dr_log=args.color_dr_log, blocks=args.blocks, wrist_lens=args.wrist_lens, script=args.script,
        log=args.log, result=args.result, trajectory=args.trajectory, task=args.task, task_log=args.task_log,
        block_colors=args.block_colors.split(",") if args.block_colors else None,
        live_perception=args.live_perception, perception_camera=args.perception_camera,
        perception_hz=args.perception_hz,
        perception_json=args.perception_json, perception_stage=args.perception_stage,
        perception_delay=args.perception_delay, perception_fault=args.perception_fault, annotations=args.annotations,
        metrics_log=args.metrics_log, metrics_run_id=args.metrics_run_id,
        screen_video_dir=screen_video_dir()), log,
        keys=keys, focused=focused, viewer=viewer)
    if viewer:
        viewer.attach(session, reset_camera=True)
        if args.focus:
            viewer.set_workspace(True)
    t_start = time.monotonic()
    while True:
        now = time.monotonic()
        if viewer and viewer.should_close():
            break
        if session.rebuild_request is not None:  # block count changed in the panel: rebuild (about 0.2 s)
            cfg = session_config_with(session.cfg, session.rebuild_request)
            script = session.handover_script()
            session.finish()
            session = MjSession(cfg, log, keys=keys, focused=focused, script=script, viewer=viewer)
            if viewer:
                viewer.attach(session)
        running = session.step()
        if viewer:
            viewer.render(session)
        if not running and args.exit_after_script:
            break
        if args.max_time and now - t_start > args.max_time:
            break
        spare = session.control_dt - (time.monotonic() - now)  # real time: physics never runs ahead of the operator
        if spare > 0:
            time.sleep(spare)
    session.finish()
    if viewer:
        viewer.close()                       # the window is gone: a running screen video ends and is joined
    if agent is not None:
        for path in agent.close():
            log(f"screen video saved: {path}")


main()
