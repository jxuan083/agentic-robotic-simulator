#!/usr/bin/env python3
"""Record SO-101 teleoperation, generate a no-device sample, or read an episode."""

import argparse
from contextlib import contextmanager
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_episode import Episode, stream_name, write_json


def provenance():
    result = {"python": platform.python_version(), "entrypoint": "tools/real_teleop.py"}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        result["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


@contextmanager
def stop_keys():
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)

    def poll():
        if not select.select([fd], [], [], 0)[0]:
            return False
        key = os.read(fd, 1)
        if key in (b"\x1b", b""):
            raise KeyboardInterrupt
        return key.lower() == b"q"

    def interrupt(*_args):
        raise KeyboardInterrupt

    handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        tty.setcbreak(fd)
        yield poll
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)
        signal.signal(signal.SIGTERM, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("check", help="Inspect device and calibration paths without opening them")
    check.add_argument("--config", type=Path, required=True)
    record = commands.add_parser("record", help="Record the leader/follower pairs named in a local profile")
    record.add_argument("--config", type=Path, required=True)
    record.add_argument("--layout", required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--allow-motion", action="store_true")
    fake = commands.add_parser("fake", help="Write a deterministic policy-to-manual example without devices")
    fake.add_argument("--output", type=Path, required=True)
    fake.add_argument("--arms", nargs="+", default=["right"])
    read = commands.add_parser("read", help="Validate an episode and optionally open it in a local browser")
    read.add_argument("episode", type=Path)
    read.add_argument("--html", action="store_true", help="Write replay.html next to the episode manifest")
    read.add_argument("--frame", type=int, help="Include this control sample in the report")
    args = parser.parse_args(argv)

    try:
        if args.action == "read":
            episode = Episode(args.episode)
            report = episode.validate()
            if args.frame is not None:
                if not 0 <= args.frame < len(episode.rows):
                    raise ValueError("frame out of range")
                report["sample"] = episode.rows[args.frame]
            if args.html and report["ok"]:
                from agentic_sim.real_viewer import write_viewer

                write_viewer(episode)
                report["viewer"] = "replay.html"
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ok"] else 1
        if args.action == "fake":
            from agentic_sim.real_fake import record_fake

            arms = [stream_name(name) for name in args.arms]
            if not 1 <= len(arms) <= 2 or len(set(arms)) != len(arms):
                raise ValueError("specify one or two distinct arm names")
            trace = record_fake(args.output, arms, provenance())
            write_json(args.output / "fake_device_trace.json", trace)
            print(args.output)
            return 0
        from agentic_sim.real_hardware import check_devices, load_profile

        profile = load_profile(args.config)
        if args.action == "check":
            check_devices(profile)
            print("Profile, paths and dependencies available:", ", ".join(profile["arms"]))
            return 0
        if not args.allow_motion or not sys.stdin.isatty():
            raise ValueError("record requires --allow-motion and an interactive terminal")
        if args.output.exists():
            raise ValueError(f"episode already exists: {args.output}")
        check_devices(profile)
        from agentic_sim.real_record import record_live

        print("Selected follower pairs:", ", ".join(profile["arms"]), flush=True)
        input("Place followers in supported poses, clear the workspace, and press Enter to arm. ")
        with stop_keys() as poll:
            record_live(args.output, profile, provenance=provenance() | {"lerobot": version("lerobot")},
                        layout=args.layout, stop_requested=poll)
        print(args.output)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Existing episode data was retained.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, KeyError, TypeError, ValueError, ImportError, EOFError) as error:
        print(f"{args.action}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
