#!/usr/bin/env python3
"""Open the SO-101 collection desk or run a headless fake acceptance session."""

import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_collection import Collection, locked_session
from agentic_sim.real_episode import write_json
from agentic_sim.real_motion import OFF, load_setup, near
from agentic_sim.real_round import LiveBackend
from real_teleop import provenance


def real_backend(config):
    from agentic_sim.real_hardware import check_devices, load_profile

    profile = load_profile(config)
    if list(profile["arms"]) != ["right"] or set(profile["cameras"]) != {"global", "wrist"}:
        raise ValueError("collection requires exactly the right arm and global/wrist camera streams")
    check_devices(profile)
    return LiveBackend(profile, "right", provenance("tools/real_collection.py"))


def teach(backend, path, waypoint_count):
    if not sys.stdin.isatty() or path.exists() or waypoint_count < 0:
        raise ValueError("teach requires an interactive terminal, a new setup file and a nonnegative waypoint count")
    data = json.loads((ROOT / "configs" / "real_collection.setup.example.json").read_text(encoding="utf-8"))
    motion = data["motion"]
    motion["calibrations"] = backend.calibrations
    motion["arm"] = backend.arm
    motion["version"] = input("起姿版本：").strip()
    motion["layout"] = input("架位 layout ID：").strip()
    motion["workspace"] = input("已驗工作區版本：").strip()
    motion["park_support"] = input("停放支撐條件（固定墊或架位描述）：").strip()
    if not all(motion[key] for key in ("version", "layout", "workspace", "park_support")):
        raise ValueError("version, layout, workspace and support description are required")
    with backend.devices(leader=False) as (_, follower):
        def sample(label):
            input(f"支撐 follower、移到已確認的 {label}，停穩後按 Enter 讀值：")
            first = follower.read()
            backend.wait(.2)
            second = follower.read()
            if follower.torque_states() != OFF or not near(first, second):
                raise RuntimeError("teach sample was moving or torque was not off")
            return second

        motion["park"] = sample("停放位")
        motion["via"] = [sample(f"通道點 {index + 1}") for index in range(waypoint_count)]
        motion["start"] = sample("標準起姿")
        input("支撐 follower 放回停放位後按 Enter：")
        if not near(follower.read(), motion["park"]) or follower.torque_states() != OFF:
            raise RuntimeError("park and torque-off could not be confirmed")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, data)
    print(f"{path}: fill the verified joint limits and approved tray positions before starting collection")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--setup", type=Path, help="local taught motion and tray-layout JSON")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--demo", action="store_true", help="headless fake workflow, using a new session directory")
    parser.add_argument("--teach", type=int, metavar="WAYPOINTS", help="saves read-only taught poses")
    parser.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.demo and not args.fake:
            raise ValueError("--demo requires --fake")
        if args.teach is not None:
            if args.fake or not args.config or not args.setup:
                raise ValueError("--teach requires real --config and --setup")
            teach(real_backend(args.config), args.setup, args.teach)
            return 0
        if args.session is None:
            raise ValueError("--session is required")
        with locked_session(args.session):
            if args.fake:
                from agentic_sim.real_collection_fake import FakeCollectionBackend, demo, fake_setup

                if args.demo:
                    facts = demo(args.session, provenance("tools/real_collection.py"))
                    print(json.dumps(facts, ensure_ascii=False, indent=2))
                    return 0
                plan, layout = fake_setup()
                backend = FakeCollectionBackend(plan)
            else:
                if not args.config or not args.setup or not args.allow_motion or not sys.stdin.isatty():
                    raise ValueError("real collection requires --config, --setup, --allow-motion and a desktop terminal")
                plan, layout = load_setup(args.setup)
                backend = real_backend(args.config)
            from agentic_sim.real_collection_ui import Window

            window = Window(Collection(args.session, backend, plan, layout))
            for kind in (signal.SIGINT, signal.SIGTERM):
                signal.signal(kind, lambda *_: window.halt() if window.worker is not None else window.close())
            window.mainloop()
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError, EOFError) as error:
        print(f"collection: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
