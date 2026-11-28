#!/usr/bin/env python3
"""Explicit physical replay of one named follower from a real episode."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_episode import Episode
from agentic_sim.real_motion import load_setup
from agentic_sim.real_round import LiveBackend, replay_round, replay_steps
from real_teleop import ConsoleOperator, provenance, stop_keys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--setup", type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, help="new directory for the replay result and measured trajectory")
    parser.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args(argv)
    try:
        episode = Episode(args.episode)
        if episode.manifest["source"] != "real":
            raise ValueError("physical replay requires a real capture")
        if not args.allow_motion or not sys.stdin.isatty() or args.setup is None:
            raise ValueError("physical replay requires --allow-motion, --setup and an interactive terminal")
        from agentic_sim.real_hardware import load_profile

        plan, _ = load_setup(args.setup)
        profile = load_profile(args.config)
        backend = LiveBackend(profile, args.arm, provenance("tools/replay_real_follower.py"))
        replay_steps(episode, backend, plan)
        if not Path(backend.pair["follower"]["port"]).expanduser().is_char_device():
            raise ValueError("selected follower serial device is unavailable")
        output = args.output or args.episode.parent / "replays" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        input("Restore the opening scene, verify follower, support, path and hands clear. Press Enter to replay. ")
        with stop_keys() as poll:
            result = replay_round(episode, backend, plan, output, ConsoleOperator(poll))
        print(f"{result['status']}: {output / 'result.json'}")
        return 0 if result["status"] == "complete" else 1
    except KeyboardInterrupt:
        print("Interrupted. Check follower support and torque on site.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, KeyError, TypeError, ValueError, ImportError, EOFError) as error:
        print(f"replay: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
