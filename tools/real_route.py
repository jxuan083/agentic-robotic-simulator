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
    record_drop, recover, record, roundtrip, sample_park, save_route, tune_route, drop_test,
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


@contextmanager
def episode_capture(root, config, arm, refs, hz, layout):
    """Cameras + episode writer for one park-start grasp demonstration (same files as the collection desk)."""
    from threading import Event

    from agentic_sim.real_episode import EpisodeWriter
    from agentic_sim.real_hardware import load_profile, open_camera
    from agentic_sim.real_record import CameraWorker
    from real_teleop import provenance

    profile = load_profile(config)
    with ExitStack() as stack:
        streams, cameras = {}, {}
        for name, spec in profile["cameras"].items():
            for attempt in range(3):  # the wrist camera sometimes misses its first frame after opening
                try:
                    streams[name], info = stack.enter_context(open_camera(spec))
                    break
                except TimeoutError:
                    if attempt == 2:
                        raise
                    time.sleep(1)
            cameras[name] = {k: v for k, v in spec.items() if k not in ("device", "depth")} | info
        writer = EpisodeWriter(root, source="real", arms={arm: refs}, cameras=cameras,
                               provenance=provenance("tools/real_route.py grasp") | {"start_pose": "park",
                                                                                     "leader_mapping": "driven hand-over, relative"},
                               hz=hz, layout=layout)
        stopped = Event()
        workers = [CameraWorker(name, profile["cameras"][name], streams[name], writer, stopped) for name in cameras]
        started, error = [], None

        def check():
            for worker in workers:
                worker.check()

        try:
            for worker in workers:
                worker.thread.start()
                started.append(worker)
            deadline = time.monotonic() + 8
            while any(w.last_frame is None and w.error is None for w in workers) and time.monotonic() < deadline:
                time.sleep(.02)
            check()
            yield writer, check
            check()
        except BaseException as failure:
            error = failure
            raise
        finally:
            stopped.set()
            for worker in started:
                worker.thread.join()
            writer.close(error=error)


def set_aside(root, session, why, line=None):
    """Move a cancelled or failed episode out of the session into <session>-rejected/ (raw files kept, as the lab
    rules ask, but never seen by training or the session counters) and log it there. Returns the new path."""
    rejected = session.parent / f"{session.name}-rejected"
    rejected.mkdir(parents=True, exist_ok=True)
    target = rejected / f"{root.name}-{time.strftime('%H%M%S')}"
    root.rename(target)
    with (rejected / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps((line or {}) | {"episode": root.name, "moved_to": target.name, "why": why},
                            ensure_ascii=False) + "\n")
    return target


def round_state(index):
    """(round number, successful placements in it, blocks left in the source tray) from the session index."""
    state = (0, 0, 0)
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            row = json.loads(line).get("round")
            if row:
                state = (row["n"], row["placed_after"], row.get("left_after", 0))
    return state


def grasp_session(driver, hand, data, args, refs, hz):
    """Terminal loop of park-start pick-and-place demonstrations, until q.

    Default (rounds): the source tray holds the finals set (2 blue, 1 red, 1 yellow); each episode the D455
    opening frame picks a random remaining block, the operator places it in the next free slot, and the next
    episode starts from the scene as it is (4, 3, 2, 1 blocks left). Re-lay the set when the slots are full.
    --pick manual: the old flow, colour key per episode and the block put back by hand."""
    import random

    from agentic_sim.real_grasp import (
        COLOR_NAME, COLORS, ROUND_SET, TRAY_NAME, finalize, grasp_round, pick_place_round, random_pick,
    )

    term = driver.term
    drops = {d["tray"]: d for d in (json.loads(path.read_text(encoding="utf-8")) for path in args.drop)}
    for tray, drop in drops.items():
        if drop["calibrations"] != data["calibrations"] or body_gap(drop["waypoints"][0], data["channel"]["command"]) > 3:
            raise ValueError(f"drop route {tray} does not match this route's calibration or start pose")
    quads = json.loads(args.trays.read_text(encoding="utf-8"))["quads"]
    slots = json.loads(args.slots.read_text(encoding="utf-8"))
    episodes = args.session / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    index = args.session / "index.jsonl"
    rounds = args.pick == "random" and not args.auto_place
    used = {}
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            dest = json.loads(line).get("result", {}).get("destination")
            used[dest] = used.get(dest, 0) + 1
    set_text = "、".join(f"{n} {COLOR_NAME[c]}" for c, n in ROUND_SET.items())
    rng = random.Random()
    n, placed, left = round_state(index)
    if rounds:
        destination = args.destination or "beige"
        source = "pink" if destination == "beige" else "beige"
        size = sum(ROUND_SET.values())
        if len(slots[destination]) < size:
            raise ValueError(f"{args.slots} has {len(slots[destination])} {destination} slots; a round needs {size}")
        resume = n and placed < size and left > 0 and term.choose(
            f"上次第 {n} 輪已放 {placed} 顆、{TRAY_NAME[source]}還剩 {left} 顆。c 接著錄這一輪／n 重擺新一輪",
            {"c": "接著錄", "n": "新一輪"}) == "c"
        if not resume:
            placed = size
    while True:
        number = 1 + len(list(episodes.glob("P*")))
        if rounds:
            if placed >= size:
                key = term.choose(f"新一輪：方塊全部放回{TRAY_NAME[source]}重擺（{set_text}；近端 x ≲ 100 mm，位置與轉角隨機，"
                                  f"偶爾讓同色貼近），{TRAY_NAME[destination]}格位清空，手離開後按 a；q 結束",
                                  {"a": "開始", "q": "結束"})
                if key == "q":
                    return
                n, placed = n + 1, 0
            else:
                key = term.choose(f"第 {n} 輪第 {placed + 1}/{size} 筆（P{number:04}）：場景照舊、手離開後按 a／"
                                  "n 放棄這一輪、重擺／q 結束", {"a": "錄", "n": "重擺", "q": "結束"})
                if key == "q":
                    return
                if key == "n":
                    placed = size
                    continue
            color = None
        else:
            key = term.choose(f"第 {number} 筆：先擺好方塊、手離開盤子與手臂，再選目標顏色 r 紅／b 藍／y 黃；q 結束",
                              {"r": "紅", "b": "藍", "y": "黃", "q": "結束"})
            if key == "q":
                return
            color = COLORS[key]
            destination = args.destination or {"p": "pink", "b": "beige"}[
                term.choose("放到哪一盤？p 粉盤／b 米盤", {"p": "粉盤", "b": "米盤"})]
        k = placed if rounds else used.get(destination, 0) % len(slots[destination])
        slot = {"index": k + 1, "xy": slots[destination][k]}
        if not args.auto_place:
            term.say(f"放置：{TRAY_NAME[destination]}第 {slot['index']} 個位置，照固定順序接在上一顆旁邊"
                     f"（參考 x {slot['xy'][0]}, y {slot['xy'][1]} mm；落點由 D455 在停放位量）。")
        root = episodes / f"P{number:04}"
        with episode_capture(root, args.config, args.arm, refs, hz, data["layout_id"]) as (writer, check):
            if args.auto_place:
                result = grasp_round(driver, hand, data, drops, writer, check, color)
            else:
                result = pick_place_round(driver, hand, data, writer, check, color=color, destination=destination,
                                          slot=slot, quads=quads, choose=random_pick(rng, source) if rounds else None)
        if result.get("demo_frames") is None:
            if root.exists():
                set_aside(root, args.session, "cancelled before the demonstration")
            term.say("這筆取消，未錄到示範。")
            if rounds and result.get("scene", {}).get("candidates") == 0:
                placed = size
            continue
        color = result.get("color", color)
        round_info = None
        if rounds:
            scene = result["scene"]
            before = scene["candidates"]
            ok = bool(result.get("success"))
            round_info = {"n": n, "step": placed + 1, "size": size, "left_before": before,
                          "placed_before": placed, "placed_after": placed + ok, "left_after": before - ok,
                          "counts_before": scene["counts"], "expected_set": ROUND_SET}
            if placed == 0 and scene["counts"] != ROUND_SET:
                term.say(f"注意：D455 在{TRAY_NAME[source]}看到 {scene['counts']}，跟決賽組合 {ROUND_SET} 不同（照樣記錄）。")
        choice = result.get("target_choice")
        report, review = finalize(root, color=color, destination=result.get("destination"),
                                  success=result.get("success"), quads=quads,
                                  target_pose=choice and (choice["x_mm"], choice["y_mm"]),
                                  slot=None if args.auto_place else slot, landed=result.get("landed"),
                                  extra=round_info and {"round": round_info, "scene": result["scene"]})
        target = review["target"]["d455"]
        line = {"episode": root.name, "color": color, "result": result, "ok": report["ok"],
                "target": target and {k: target[k] for k in ("tray", "x_mm", "y_mm", "yaw_deg")},
                "training_ranges": review.get("training_ranges"), "reasons": report["review_reasons"]}
        if round_info:
            line["round"] = round_info
        if not result.get("success"):
            # Failed or discarded: set aside and record the same step again, so the session holds successes only.
            moved = set_aside(root, args.session, "failed" if not result.get("cancelled") else "discarded", line)
            term.say(f"這筆不收（原檔移到 {moved.parent.name}/{moved.name}）。"
                     + ("掉落或夾走的方塊放回粉盤（可以換新位置）、這一格若有方塊請拿走，按 a 重錄這一格。" if rounds
                        else "重擺後再錄。"))
            continue
        used[destination] = used.get(destination, 0) + 1
        with index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        where = f"{target['tray']} x {target['x_mm']:.0f} y {target['y_mm']:.0f} mm" if target else "D455 未找到目標"
        term.say(f"{root.name}：讀回 {'OK' if report['ok'] else '有問題'}，{report.get('observed_control_hz')} Hz，"
                 f"目標 {where}，訓練段 {review.get('training_ranges')}"
                 + ("" if not report["review_reasons"] else f"；待確認：{'；'.join(report['review_reasons'])}"))
        if round_info:
            placed = round_info["placed_after"]
            if round_info["left_after"] <= 0:
                placed = size


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
    if grip is None and not args.config and "tuned" in data:
        grip = data["tuned"]["gripper_open_pct"]
        print(f"keeping gripper {grip:.1f} %")
    if grip is None:
        if not args.config:
            raise SystemExit("tune needs --grip PCT or --config to read the leader gripper")
        grip, spread = leader_gripper(args.config, data["arm"])
        if spread > 2.0:
            raise SystemExit(f"leader gripper moved while reading ({spread:.1f} %); hold it still and retry")
        print(f"leader gripper reads {grip:.1f} %")
    tuned = tune_route(data, grip=grip, straight_from=args.straight_from, hold_roll=args.hold_roll)
    old = archive_existing(args.route)
    save_route(args.route, tuned)
    t = tuned["tuned"]
    print(f"saved {args.route} (previous: {old.name})")
    print(f"waypoints {t['waypoints_before']} -> {t['waypoints_after']}, playback about "
          f"{sum(tuned['segment_seconds']):.1f} s (before {sum(data['segment_seconds']):.1f} s); "
          f"gripper {data['park']['pose']['gripper']:.1f} % until waypoint {t['open_from_waypoint']}, then "
          f"{grip:.1f} %. Verify again: route-test.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "run", "recover", "record-drop", "drop-test", "grasp", "setup", "show", "tune"):
        p = sub.add_parser(name)
        p.add_argument("--route", type=Path, required=True, help="route JSON, e.g. outputs/routes/right-park-channel.json")
        if name == "tune":
            p.add_argument("--grip", type=float, help="fixed gripper opening %% at the channel (cube-ready)")
            p.add_argument("--config", type=Path, help="read the opening from the leader gripper instead (read-only)")
            p.add_argument("--hold-roll", action="store_true",
                           help="keep wrist_roll at the park value along the route (no wrist turn)")
            p.add_argument("--straight-from", type=int,
                           help="waypoint index in the route file after which to go straight to the channel")
            continue
        if name == "setup":
            p.add_argument("--drop", type=Path, action="append", default=[], help="drop route JSON (repeat per tray)")
            p.add_argument("--cards", type=Path, help="JSON with the cards section (seed, reserve_distance_mm, zones)")
            p.add_argument("--version", required=True, help="start/setup version, e.g. right-20261126-v1")
            p.add_argument("--workspace", required=True, help="verified workspace description/version")
            p.add_argument("--speed", type=float, default=20.0, help="route speed deg/s (<= 25)")
            p.add_argument("--trays", type=Path, help="fixed D455 tray corners JSON (tools/real_locate.py --save-trays)")
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
        p.add_argument("--manual-align", action="store_true",
                       help="align the leader by hand instead of driving it to the follower pose")
        if name == "record":
            p.add_argument("--layout", required=True, help="layout ID, e.g. lab-right-20261125")
            p.add_argument("--support", required=True, help="park support description, e.g. 前臂靠在固定墊塊上")
            p.add_argument("--replace", action="store_true", help="archive an existing route file and teach anew")
        if name == "run":
            p.add_argument("--no-leader", action="store_true", help="playback only, no leader task test")
        if name == "drop-test":
            p.add_argument("--drop", type=Path, required=True, help="taught drop route JSON to replay")
        if name == "grasp":
            p.add_argument("--drop", type=Path, action="append", required=True, help="drop route JSON (per tray)")
            p.add_argument("--trays", type=Path, required=True, help="fixed D455 tray corners JSON")
            p.add_argument("--session", type=Path, required=True, help="output folder, e.g. outputs/261125-park")
            p.add_argument("--slots", type=Path, default=ROOT / "configs/real_slots.json", help="placement slots JSON")
            p.add_argument("--destination", choices=("beige", "pink"), default="beige",
                           help="finals flow is pink (storage) -> beige (delivery slots); --destination pink only "
                                "for a separately tagged reverse set")
            p.add_argument("--any-destination", dest="destination", action="store_const", const=None,
                           help="ask for the destination tray each episode")
            p.add_argument("--pick", choices=("random", "manual"), default="random",
                           help="random: finals-set rounds, D455 picks a random block each episode, slots fill up; "
                                "manual: colour key per episode")
            p.add_argument("--auto-place", action="store_true",
                           help="old flow: record to the lift only, then place with the taught drop route")
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
                            cards=cards.get("cards", cards), speed=args.speed,
                            trays=json.loads(args.trays.read_text(encoding="utf-8")) if args.trays else None)
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
    leader_needed = args.command in ("record", "record-drop", "grasp") or (args.command == "run" and not args.no_leader)
    if args.command == "grasp" and args.fake:
        parser.error("grasp records cameras: run it on the lab machine")
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
                                relative=getattr(args, "relative", False),
                                drive_leader=not getattr(args, "manual_align", False))
                if track is not None:
                    driver.leader_route = track  # a folded leader is driven out along the taught route
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
                elif args.command == "grasp":
                    grasp_session(driver, hand, data, args, refs, hz)
                elif args.command == "drop-test":
                    drop_test(driver, data, track, json.loads(args.drop.read_text(encoding="utf-8")))
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
