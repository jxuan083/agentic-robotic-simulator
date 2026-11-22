#!/usr/bin/env python3
"""Explicit physical replay of one named follower from a real episode."""

import argparse
from pathlib import Path
import signal
import sys
from time import sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_control import JOINTS
from agentic_sim.real_episode import Episode


def replay_targets(episode, arm):
    if episode.manifest["source"] != "real":
        raise ValueError("physical replay requires a real capture")
    if arm not in episode.manifest["arms"]:
        raise ValueError(f"episode has no arm named {arm}")
    if not episode.validate()["ok"]:
        raise ValueError("physical replay requires a complete, valid episode")
    return [row["arms"][arm] for row in episode.rows]


def play(device, steps):
    # Compare the first target while the selected follower is still unpowered.
    current = device.read()
    for i, step in enumerate(steps):
        if i:
            sleep((step["command_ns"] - steps[i - 1]["command_ns"]) / 1e9)
            current = device.read()
        target = step["command"]
        if any(abs(target[joint] - current[joint]) > (10 if joint == "gripper" else 5) for joint in JOINTS):
            raise RuntimeError(f"follower is too far from the replay target at frame {i}")
        if i == 0:
            device.arm()
        device.send(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True, help="One exact arm name in the episode and local profile")
    parser.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args(argv)
    try:
        episode = Episode(args.episode)
        steps = replay_targets(episode, args.arm)
        if not args.allow_motion or not sys.stdin.isatty():
            raise ValueError("physical replay requires --allow-motion and an interactive terminal")
        from agentic_sim.real_hardware import calibration_ref, load_profile, open_follower

        spec = load_profile(args.config)["arms"][args.arm]["follower"]
        if calibration_ref("follower", spec) != episode.manifest["arms"][args.arm]["follower"]:
            raise ValueError("selected follower calibration differs from the episode")
        if not Path(spec["port"]).expanduser().is_char_device():
            raise ValueError("selected follower serial device is unavailable")
        input(f"Support {args.arm}, match the first pose and clear its path. Press Enter to replay. ")

        def interrupt(*_args):
            raise KeyboardInterrupt

        handler = signal.signal(signal.SIGTERM, interrupt)
        try:
            with open_follower(args.arm, spec) as device:
                play(device, steps)
        finally:
            signal.signal(signal.SIGTERM, handler)
        print(f"{args.arm}: replayed {len(steps)} steps")
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, KeyError, TypeError, ValueError, ImportError, EOFError) as error:
        print(f"replay: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
