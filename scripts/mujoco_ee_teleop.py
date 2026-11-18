"""MuJoCo version of the minimal desk scene with keyboard EE teleop for one SO-101 (runs on the host, no GPU needed).

    uv run scripts/mujoco_ee_teleop.py [--blocks 3] [--color_dr --color_dr_seed 7]
    uv run scripts/mujoco_ee_teleop.py --script configs/acceptance/keyboard_pick_place.yaml \
        --log out.csv --result out.json --exit_after_script

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
from agentic_sim.teleop.keys import KeyState  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scene", default="configs/scenes/lab_current_minimal.yaml")
parser.add_argument("--teleop", default="configs/teleop.yaml")
parser.add_argument("--script", default=None, help="YAML input script for automated runs")
parser.add_argument("--log", default=None, help="CSV debug log path")
parser.add_argument("--result", default=None, help="JSON summary path (script / trajectory runs)")
parser.add_argument("--trajectory", default=None, help="scripted EE target sequence (configs/trajectories/*.yaml); "
                    "replaces keyboard / mouse input, ends the run with --exit_after_script")
parser.add_argument("--exit_after_script", action="store_true")
parser.add_argument("--max_time", type=float, default=0.0, help="exit after N wall seconds (0 = run until closed)")
parser.add_argument("--color_dr", action="store_true", help="enable colour domain randomization (default off)")
parser.add_argument("--color_dr_seed", type=int, default=None)
parser.add_argument("--color_dr_config", default="configs/color_randomization.yaml")
parser.add_argument("--color_dr_log", default=None)
parser.add_argument("--blocks", type=int, default=None, help="number of blocks, 1-8 (default: the scene's count)")
parser.add_argument("--headless", action="store_true", help="no window (keyboard-only scripts)")
parser.add_argument("--hidden", action="store_true",
                    help="development checks only: invisible window, focus treated as present (not an acceptance run)")
parser.add_argument("--focus", action="store_true", help="accepted for menu compatibility (no editor panels here)")
args = parser.parse_args()


def log(msg):
    print(f"[mujoco_teleop] {msg}", flush=True)


def main():
    keys = KeyState(actions={"P": "pause", "R": "reset"})
    viewer = None
    if not args.headless:
        from agentic_sim.mujoco_backend.viewer import Viewer
        vc = load(ROBOT_CONFIG)["viewer"]
        viewer = Viewer(keys, vc["title"], vc["size"], log, visible=not args.hidden)
    if args.script and viewer is None:
        log("headless: scripted keys only")
    focused = (lambda: True) if args.hidden else (viewer.focused if viewer else None)
    session = MjSession(SessionConfig(
        scene=args.scene, teleop=args.teleop, color_dr_config=args.color_dr_config, color_dr=args.color_dr,
        color_dr_seed=args.color_dr_seed, color_dr_log=args.color_dr_log, blocks=args.blocks, script=args.script,
        log=args.log, result=args.result, trajectory=args.trajectory), log, keys=keys, focused=focused,
        viewer=viewer)
    if viewer:
        viewer.attach(session, reset_camera=True)
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
        viewer.close()


main()
