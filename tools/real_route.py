#!/usr/bin/env python3
"""Teach and replay the single-arm park <-> channel route with the SO-101 leader.

  record   follower on its park support (torque off) -> follow the leader to the channel -> c
           saves the route -> r replays it in reverse to park -> confirm support -> torque off
  run      from park: replay forward to the channel, hold; t = leader task test, r = reverse home
  recover  follower still holding with torque on: follow the route back to park and release
  record-drop  park -> channel, then teach channel -> drop pose over a tray (--tray pink|beige)
  setup    combine the park-channel route and drop routes into a collection setup JSON
  show     offline summary of a route file
"""

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import select
import signal
import sys
import termios
import time
import tty

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_motion import OFF  # noqa: E402
from agentic_sim.real_route import (  # noqa: E402
    ENGAGE_GAP, PLAY_SPEED, Driver, Released, archive_existing, body_gap, build_setup, load_route,
    record_drop, recover, record, roundtrip, sample_park, save_route, tune_route,
)


class Terminal:
    """cbreak keys for the operator; Ctrl-C is turned into a hold request instead of an exit."""

    def __init__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("run this in a desktop terminal next to the arm")
        self.fd = sys.stdin.fileno()
        self.pending = []
        self.shown = False

    def __enter__(self):
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.previous = signal.signal(signal.SIGINT, lambda *_: self.pending.append("esc"))
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        signal.signal(signal.SIGINT, self.previous)
        self._clear()

    def key(self, wait=0.0):
        if not self.pending and select.select([self.fd], [], [], wait)[0]:
            data = os.read(self.fd, 1)
            if data == b"\x1b":
                while select.select([self.fd], [], [], 0.01)[0]:  # swallow arrow-key sequences
                    os.read(self.fd, 8)
                return "esc"
            if data in (b"\n", b"\r"):
                return "enter"
            return data.decode(errors="ignore").lower() or None
        return self.pending.pop(0) if self.pending else None

    def _clear(self):
        if self.shown:
            sys.stdout.write("\r\x1b[K")
            self.shown = False

    def say(self, text):
        self._clear()
        print(text, flush=True)

    def status(self, text):
        sys.stdout.write("\r" + text + "\x1b[K")
        sys.stdout.flush()
        self.shown = True

    def choose(self, prompt, options):
        self.say(prompt)
        while True:
            key = self.key(0.2)
            if key in options:
                self.say(f"→ {options[key]}")
                return key

    def word(self, prompt, expected):
        self.say(prompt)
        typed = ""
        while True:
            key = self.key(0.2)
            if key == "esc":
                self.say("")
                return False
            if key == "enter":
                self.say("")
                return typed == expected
            if key:
                typed += key.upper()
                sys.stdout.write(key.upper())
                sys.stdout.flush()


def run_dir(route, mode):
    return route.parent / f"{route.stem}-runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{mode}"


@contextmanager
def real_devices(config, arm, *, leader, resume=False):
    from agentic_sim.real_hardware import calibration_ref, check_devices, load_profile, open_follower, open_leader

    profile = load_profile(config)
    if list(profile["arms"]) != [arm]:
        raise ValueError(f"{config}: single-arm profile must list exactly the {arm} pair")
    check_devices(profile)
    pair = profile["arms"][arm]
    refs = {role: calibration_ref(role, spec) for role, spec in pair.items()}
    with ExitStack() as stack:
        hand = stack.enter_context(open_leader(arm, pair["leader"])) if leader else None
        follower = stack.enter_context(open_follower(arm, pair["follower"], managed=True, resume=resume))
        yield hand, follower, refs, profile["control_hz"]


@contextmanager
def fake_devices(*, leader, resume=False):
    from agentic_sim.real_route_fake import FakeFollower, FakeLeader, operator_path

    clock = time.monotonic_ns
    follower = FakeFollower(clock, torque=resume)
    refs = {role: {"id": f"fake_right_{role}", "version": "synthetic-v1"} for role in ("leader", "follower")}
    yield (FakeLeader(clock, [(t + 4.0, p) for t, p in operator_path()]) if leader else None), follower, refs, 60


def check_calibration(data, refs, *, leader):
    roles = ("leader", "follower") if leader else ("follower",)
    for role in roles:
        if data["calibrations"][role] != refs[role]:
            raise RuntimeError(f"{role} calibration changed since the route was taught; teach a new route")


def show(path):
    data, track = load_route(path)
    ranges = {j: (min(p[j] for p in track.points), max(p[j] for p in track.points)) for j in track.points[0]}
    print(json.dumps({
        "route": str(path), "arm": data["arm"], "layout_id": data["layout_id"], "created_at": data["created_at"],
        "park": data["park"]["pose"], "channel": data["channel"]["readback"],
        "waypoints": len(track.points), "recorded_moving_seconds": round(sum(track.seconds), 2),
        "min_playback_seconds_at_cap": round(sum(body_gap(a, b) for a, b in zip(track.points, track.points[1:]))
                                             / PLAY_SPEED, 2),
        "joint_ranges_along_route": ranges, "verified": data["verified"],
    }, ensure_ascii=False, indent=2))


def leader_gripper(config, arm, samples=10):
    """Read-only: the leader's current gripper opening (percent), averaged over a few reads."""
    from agentic_sim.real_hardware import check_devices, load_profile, open_leader

    profile = load_profile(config)
    check_devices(profile)
    with open_leader(arm, profile["arms"][arm]["leader"]) as leader:
        values = []
        for _ in range(samples):
            values.append(leader.read()["gripper"])
            time.sleep(0.05)
    return sum(values) / len(values), max(values) - min(values)


def tune(args):
    data, _ = load_route(args.route)
    grip = args.grip
    if grip is None:
        if not args.config:
            raise SystemExit("tune needs --grip PCT or --config to read the leader gripper")
        grip, spread = leader_gripper(args.config, data["arm"])
        if spread > 2.0:
            raise SystemExit(f"leader gripper moved while reading ({spread:.1f} %); hold it still and retry")
        print(f"leader gripper reads {grip:.1f} %")
    tuned = tune_route(data, grip=grip)
    old = archive_existing(args.route)
    save_route(args.route, tuned)
    t = tuned["tuned"]
    print(f"saved {args.route} (previous: {old.name})")
    print(f"waypoints {t['waypoints_before']} -> {t['waypoints_after']}, playback about "
          f"{sum(tuned['segment_seconds']):.1f} s (taught {sum(data['segment_seconds']):.1f} s incl. pauses); "
          f"gripper {data['park']['pose']['gripper']:.1f} % until waypoint {t['open_from_waypoint']}, then "
          f"{grip:.1f} %. Verify again: route-test.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "run", "recover", "record-drop", "setup", "show", "tune"):
        p = sub.add_parser(name)
        p.add_argument("--route", type=Path, required=True, help="route JSON, e.g. outputs/routes/right-park-channel.json")
        if name == "tune":
            p.add_argument("--grip", type=float, help="fixed gripper opening %% at the channel (cube-ready)")
            p.add_argument("--config", type=Path, help="read the opening from the leader gripper instead (read-only)")
            continue
        if name == "setup":
            p.add_argument("--drop", type=Path, action="append", default=[], help="drop route JSON (repeat per tray)")
            p.add_argument("--cards", type=Path, help="JSON with the cards section (seed, reserve_distance_mm, zones)")
            p.add_argument("--version", required=True, help="start/setup version, e.g. right-20261126-v1")
            p.add_argument("--workspace", required=True, help="verified workspace description/version")
            p.add_argument("--speed", type=float, default=20.0, help="route speed deg/s (<= 25)")
            p.add_argument("--out", type=Path, required=True)
        if name in ("show", "setup"):
            continue
        p.add_argument("--config", type=Path, help="single-arm local profile, e.g. outputs/single.local.yaml")
        p.add_argument("--arm", default="right")
        p.add_argument("--fake", action="store_true", help="rehearse the key flow with simulated arms")
        p.add_argument("--allow-motion", action="store_true")
        p.add_argument("--rate", type=float, default=1.0, help="playback speed factor vs. the recorded timing")
        if name != "recover":
            p.add_argument("--relative", action="store_true",
                           help="follow leader displacement instead of absolute angles (fallback)")
        if name == "record":
            p.add_argument("--layout", required=True, help="layout ID, e.g. lab-right-20261125")
            p.add_argument("--support", required=True, help="park support description, e.g. 前臂靠在固定墊塊上")
            p.add_argument("--replace", action="store_true", help="archive an existing route file and teach anew")
        if name == "run":
            p.add_argument("--no-leader", action="store_true", help="playback only, no leader task test")
        if name == "record-drop":
            p.add_argument("--tray", required=True, choices=("pink", "beige"))
            p.add_argument("--out", type=Path, required=True, help="drop route JSON to write")
            p.add_argument("--release", type=float, default=60.0, help="gripper opening %% to release the block")
            p.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "show":
        show(args.route)
        return 0
    if args.command == "tune":
        return tune(args)
    if args.command == "setup":
        data, track = load_route(args.route)
        if not data["verified"].get("reverse_to_park"):
            parser.error("verify the reverse route to park before building a setup")
        drops = [json.loads(path.read_text(encoding="utf-8")) for path in args.drop]
        example = json.loads((ROOT / "configs/real_collection.setup.example.json").read_text(encoding="utf-8"))
        cards = json.loads(args.cards.read_text(encoding="utf-8")) if args.cards else example["cards"]
        setup = build_setup(data, track, drops, version=args.version, workspace=args.workspace,
                            cards=cards.get("cards", cards), speed=args.speed)
        from agentic_sim.real_motion import MotionPlan

        MotionPlan.load(setup["motion"])  # same validation as the collection desk
        if args.out.exists():
            parser.error(f"{args.out} exists")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(setup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{args.out}: {len(setup['motion']['via'])} via points, drops {sorted(setup['motion']['drops'])}"
              + ("" if any(setup["cards"]["zones"].values()) else "; fill cards.zones before collecting"))
        return 0
    if not args.fake and (not args.config or not args.allow_motion):
        parser.error("hardware use needs --config and --allow-motion (or --fake)")
    if not 0.2 <= args.rate <= 2.0:
        parser.error("--rate must be within 0.2..2.0")

    route = args.route
    leader_needed = args.command in ("record", "record-drop") or (args.command == "run" and not args.no_leader)
    if args.command == "record-drop" and args.out.exists() and not args.replace:
        parser.error(f"{args.out} exists; pass --replace to archive it")
    if args.command == "record":
        if route.exists() and not args.replace:
            parser.error(f"{route} exists; pass --replace to archive it and teach a new route")
        data = track = None
    else:
        data, track = load_route(route)
        if data["arm"] != args.arm:
            parser.error(f"route was taught on the {data['arm']} arm")

    from real_teleop import provenance

    driver = None
    try:
        with Terminal() as term:
            opener = fake_devices if args.fake else lambda **kw: real_devices(args.config, args.arm, **kw)
            if args.command == "record":
                term.say("教示停放位→通道位：確認 follower 已靠在停放支撐上、夾爪為空、工作區淨空。")
                if term.choose("y 開始（只讀取停放位，扭力保持關閉）／n 離開", {"y": "開始", "n": "離開"}) != "y":
                    return 0
            else:
                where = "已沿路線停在中途並保持扭力" if args.command == "recover" else "已靠在停放支撐上、扭力關閉"
                term.say(f"確認 follower {where}，夾爪為空，路線淨空，有人可隨時按 Esc。")
                if term.choose("y 開始／n 離開", {"y": "開始", "n": "離開"}) != "y":
                    return 0
            with opener(leader=leader_needed, resume=args.command == "recover") as (hand, follower, refs, hz):
                if data is not None:
                    check_calibration(data, refs, leader=leader_needed)
                driver = Driver(follower, term, run_dir(route, args.command), hz=hz, rate=args.rate,
                                relative=getattr(args, "relative", False))
                driver.state.update(command=args.command, route=str(route), fake=args.fake,
                                    provenance=provenance("tools/real_route.py"), control_hz=hz)
                if args.command == "record":
                    archived = archive_existing(route) if args.replace else None
                    if archived:
                        term.say(f"舊路線已保存為 {archived}")
                    samples = sample_park(follower, time.sleep)
                    meta = {"arm": args.arm, "layout_id": args.layout, "park_support": args.support,
                            "calibrations": refs, "control_hz": hz,
                            "mapping": "relative" if args.relative else "absolute",
                            "engage_gap": ENGAGE_GAP}
                    term.say("停放位已讀取（扭力關閉、靜止）。")
                    record(driver, hand, route, meta, samples)
                elif args.command == "run":
                    roundtrip(driver, hand, route, data, track)
                elif args.command == "record-drop":
                    if args.replace:
                        archive_existing(args.out)
                    record_drop(driver, hand, data, track, args.out, args.tray, args.release)
                else:
                    if follower.torque_states() == OFF:
                        term.say("follower 扭力已全關，不需要接手；請確認支撐。")
                        return 0
                    recover(driver, data, track)
                final = follower.torque_states()
                driver.stage("finished", torque=final)
                term.say("結束：六軸扭力 " + ("全關" if final == OFF else f"仍開 {final}，follower 保持中，請現場處理"))
                return 0 if final == OFF else 2
    except Released as done:
        print(done)
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError) as error:
        if driver is not None and driver.armed:
            try:
                driver.hold(f"fatal: {error}")
            except Exception:  # noqa: BLE001 - report the original failure below
                pass
            driver.stage("operator_required", reason=str(error), pose=driver.last)
            print(f"real_route: {error}\n扭力可能仍開啟、follower 保持中。扶住手臂後可執行 "
                  f"`python tools/real_route.py recover --route {route} ...` 沿路線回停放位。", file=sys.stderr)
            return 2
        print(f"real_route: {error}", file=sys.stderr)
        return 1
    finally:
        if driver is not None:
            driver.close_logs()


if __name__ == "__main__":
    sys.exit(main())
