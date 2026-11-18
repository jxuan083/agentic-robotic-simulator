"""Isaac Sim minimal desk scene with keyboard / mouse EE teleop for one SO-101 (issue #1).

Run inside the pinned container (see docs/isaac-ee-teleop.md):
    docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py
Optional:
    --script configs/acceptance/keyboard_pick_place.yaml   drive the UI with injected input events
    --log outputs/run.csv                                  per-frame debug log
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

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
parser.add_argument("--profile", action="store_true", help="print per-section frame timing every 5 s")
parser.add_argument("--focus", action="store_true", help="start in the focus (operation) layout")
parser.add_argument("--color_dr", action="store_true", help="enable colour domain randomization (default off)")
parser.add_argument("--color_dr_seed", type=int, default=None, help="seed (default: from the config)")
parser.add_argument("--color_dr_config", default="configs/color_randomization.yaml")
parser.add_argument("--color_dr_log", default=None, help="JSONL of the drawn colours (default: from the config)")
parser.add_argument("--blocks", type=int, default=None, help="number of blocks, 1-8 (default: the scene's count)")
AppLauncher.add_app_launcher_args(parser)
# One small articulation: CPU PhysX steps faster than the GPU pipeline here, and the viewport should not wait for
# vsync / the 60 Hz main-loop limiter (measured: 16 -> 40 fps on the tested machine, see docs/isaac-ee-teleop.md).
parser.set_defaults(device="cpu", kit_args="--/app/vsync=false --/app/runLoops/main/rateLimitEnabled=false")
args = parser.parse_args()
app_launcher = AppLauncher(args)
app = app_launcher.app

import time  # noqa: E402

from agentic_sim.isaac.session import CONTROL_DT, SessionConfig, TeleopSession, session_config_with  # noqa: E402


def log(msg):
    print(f"[ee_teleop] {msg}", flush=True)


def main():
    session = TeleopSession(SessionConfig(
        scene=args.scene, teleop=args.teleop, color_dr_config=args.color_dr_config, focus=args.focus,
        color_dr=args.color_dr, color_dr_seed=args.color_dr_seed, color_dr_log=args.color_dr_log, blocks=args.blocks,
        script=args.script, log=args.log, result=args.result, profile=args.profile, device=args.device,
        trajectory=args.trajectory), log)
    t_start = time.monotonic()
    while app.is_running():
        now = time.monotonic()
        if session.rebuild_request is not None:  # block count changed in the panel: rebuild in this process
            cfg = session_config_with(session.cfg, session.rebuild_request)
            layout, script = session.focus_mode, session.handover_script()
            session.close(keep_layout=True)
            session = TeleopSession(cfg, log, focus_layout=layout, carry_script=script)
        if not session.step() and args.exit_after_script:
            break
        if args.max_time and now - t_start > args.max_time:
            break
        # pace to real time so physics does not run faster than the operator sees
        spare = CONTROL_DT - (time.monotonic() - now)
        if spare > 0:
            time.sleep(spare)
    session.finish()
    session.kb.close()


main()
sys.stdout.flush()
os._exit(0)  # SimulationApp.close() can hang in Isaac Sim 5.1; all outputs are flushed above
