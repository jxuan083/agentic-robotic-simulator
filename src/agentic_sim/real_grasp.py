"""One pick-and-place demonstration from the park pose, driven from a terminal (no task cards).

  park (torque on, small reseat) -> leader handed over at park ->
  [record: unfold, approach, close, lift, carry, place in the chosen slot, open, lift] c ->
  (straight up to 12 cm when c is pressed away from the start pose) start pose -> park -> torque off
With --program-place (lab_day grasp default since 11/26): the follower takes the route to the start pose, hand-over
there, [record "manual": start -> grasp -> stable lift] c [record "script": programmed carry into the slot, open,
rise over the slot] -> start pose -> park; the leader holds at c and is driven home.

The demonstration covers the whole pick and place (no scripted detour back to the start pose before placing).
Cameras run for the whole round: the opening global frame shows the untouched layout and gives the target
block (fixed tray corners); a frame taken back at park (arm folded, trays unobstructed) gives the landing. review.json gets
start/close/lift/release/end and both targets. Training ranges: grasp [start, lift], place [lift, end].
"""

from agentic_sim.real_control import JOINTS, pose
from agentic_sim.real_route import (
    CARRY_SQUEEZE, Paused, Track, _simplify, body_gap, follow, park_and_release, place_from_channel, retimed,
)

END_NEAR_START = 30.0   # body deg: after c the arm goes straight to the start pose only from this close
RETURN_LIFT_Z = 0.12    # m, grasp centre: after c farther away, first straight up to this height (clears rims, blocks)
RETURN_MIN_Z = 0.05     # m: lowest grasp-centre height allowed on the straight move from there to the start pose
WRIST_FREE = "wrist_roll"   # turned to square up with the block during the demo; the straight move back restores it


def reach_gap(a, b):
    """body_gap without wrist_roll: how far the arm is from the start pose, ignoring a wrist turned in open air.
    The way back always ends at the full start pose (wrist included) before the fixed route to park."""
    return max(abs(a[j] - b[j]) for j in JOINTS[:-1] if j != WRIST_FREE)

COLORS = {"r": "red", "b": "blue", "y": "yellow"}
COLOR_NAME = {"red": "紅", "blue": "藍", "yellow": "黃"}
TRAY_NAME = {"pink": "粉盤", "beige": "米盤"}
ROUND_SET = {"blue": 2, "red": 1, "yellow": 1}   # finals storage: one round = these blocks, picked in random order
RETRACE_ERROR = 2.0
CROWDED_MM = 45.0       # centre distance to a block already in the destination tray that leaves no room for the fingers


def placed_block(before, after, color, tray, slot):
    """The block this episode put in the tray: of the destination blocks of that colour seen at park after the
    episode, the one farthest from every same-colour block that was already there in the opening frame.
    Adds the offset to the nominal slot (a reference only: conditioning uses the landing) and the distance to the
    nearest other block in the tray."""
    def dist(a, b):
        return ((a["x_mm"] - b["x_mm"]) ** 2 + (a["y_mm"] - b["y_mm"]) ** 2) ** .5

    there = [b for b in before if b["tray"] == tray]
    new = [b for b in after if b["tray"] == tray and b["color"] == color]
    if not new:
        return None
    old = [b for b in there if b["color"] == color]
    landed = max(new, key=lambda b: min((dist(b, o) for o in old), default=float("inf")))
    others = [b for b in after if b["tray"] == tray and b is not landed]
    reference = {"x_mm": slot["xy"][0], "y_mm": slot["xy"][1]}
    return landed | {"slot_offset_mm": round(dist(landed, reference), 1),
                     "neighbour_mm": round(min(dist(landed, b) for b in others), 1) if others else None}


def taken_block(before, after, color, tray, near_mm=20.0):
    """The block of that colour that left the source tray (opening frame vs park after the episode), or None when
    not exactly one did. 11/27 P0006: with two blues the operator took the other one than the D455 target."""
    def dist(a, b):
        return ((a["x_mm"] - b["x_mm"]) ** 2 + (a["y_mm"] - b["y_mm"]) ** 2) ** .5

    left = [b for b in after if b["tray"] == tray and b["color"] == color]
    gone = [b for b in before if b["tray"] == tray and b["color"] == color
            and all(dist(b, a) > near_mm for a in left)]
    return gone[0] if len(gone) == 1 else None


def lift_and_return(reading, channel, follower_id):
    """After c away from the start pose (placed in the beige tray and lifted): straight up to RETURN_LIFT_Z over
    the same spot (IK on the follower's calibration-file joint map), then to the start pose. Returns a Track, or None
    when the lift has no IK or the move would pass below RETURN_MIN_Z (the caller falls back to the menu)."""
    import numpy as np
    from pathlib import Path
    from agentic_sim.real_kinematics import ArmModel, find_calibration

    try:
        model = ArmModel.from_file(find_calibration(follower_id), Path(__file__).resolve().parents[2])
    except FileNotFoundError:     # no calibration file (rehearsal): the caller falls back to the old way back
        return None
    now = dict(reading)
    ee = model.ee(now)
    points = [now]
    if ee[2] < RETURN_LIFT_Z:
        up, _ = model.hover(np.array([ee[0], ee[1], RETURN_LIFT_Z]), model.radial(ee[:2]), now, 30.0)
        if up is None:
            return None
        points.append(up | {"gripper": now["gripper"]})
    end = pose(channel) | {"gripper": now["gripper"]}
    for a, b in zip(points[-1:], [end]):
        for t in np.linspace(0, 1, 15):
            if model.ee({j: a[j] + (b[j] - a[j]) * t for j in a})[2] < RETURN_MIN_Z - 1e-3:
                return None
    return retimed(points + [end])


def retrace_to(samples, channel):
    """Reverse the recorded path back to where it passed closest to the start pose, then straight to it.
    The demonstration starts at park, so reversing all of it would fold the arm back holding the block."""
    points = [p for _, p in samples]
    nearest = min(range(len(points)), key=lambda i: body_gap(points[i], channel))
    tail = points[nearest:]
    tail = [tail[i] for i in sorted(_simplify(tail, RETRACE_ERROR))][::-1]
    grip = tail[0]["gripper"]
    return retimed([*(p | {"gripper": grip} for p in tail), pose(channel) | {"gripper": grip}])


def control_row(writer, sent, actual):
    now = writer.clock()
    return {"timestamp_ns": now, "command_source": "manual", "task_is_policy": 0, "event": None,
            "arms": {"right": {"command": pose(sent), "command_ns": now, "readback": pose(actual), "readback_ns": now}}}


def record_demo(driver, leader, writer, check_cameras):
    """Teleop from park until c; every follower command becomes an episode control row.
    Returns (samples, kept): the driven path (also when discarded, to retrace it safely) and whether c ended it."""
    rows = [0]
    driver.camera_error = None

    def on_send(sent, actual):
        if driver.camera_error:
            return
        rows[0] += 1
        if rows[0] % 15 == 0:
            try:
                check_cameras()
            except RuntimeError as error:
                # 11/27 P0015: the wrist camera stalled ~1 s mid-demonstration and the whole tool exited with the
                # arm reaching out. Now the recording stops, the follower holds, and the arm goes back by itself.
                driver.camera_error = str(error)
                raise Paused(f"相機中斷（{error}）") from error
        writer.add_control(control_row(writer, sent, actual))

    samples = [(driver.clock(), dict(driver.target))]
    driver.on_send = on_send
    try:
        while True:
            try:
                driver.stage("recording")
                driver.term.say(driver.demo_prompt)
                driver.teleop(leader, {"c"}, samples)
                return samples, True
            except Paused as pause:
                driver.term.say(f"已暫停並原地保持：{pause}")
                if driver.camera_error:
                    driver.term.say("這筆已停錄、不能用。leader 先放著；按 d 後 follower 沿剛才的路徑退回起姿、再回停放位"
                                    "（路上不錄），結果請按 f。")
                    choice = driver.menu("d 放棄這筆、自動退回／x 緊急原地解扭力", {"d": "放棄", "x": "緊急解力"})
                    return samples, False
                choice = driver.menu("t 重新交接 leader 繼續這筆／d 放棄這筆（之後回停放位）／x 緊急原地解扭力",
                                     {"t": "繼續", "d": "放棄", "x": "緊急解力"})
                if choice == "d" or not driver.align_leader(leader, driver.read()):
                    return samples, False
    finally:
        driver.on_send = None


def grasp_round(driver, leader, data, drops, writer, check_cameras, color):
    """Park -> demonstration -> placement -> park. Returns the operator result dict."""
    park, channel, support = data["park"]["pose"], data["channel"]["command"], data["park_support"]
    route = Track(data["waypoints"], data["segment_seconds"])
    driver.leader_route = route
    driver.arm_at(park)
    if not driver.align_leader(leader, driver.read()):
        park_and_release(driver, Track([driver.target, park], [0.0]), park, support)
        return {"cancelled": True}
    driver.demo_prompt = "錄製中：從停靠位用 leader 出收折區、接近、閉爪夾起並抬穩，停約 1 秒後按 c；Esc 暫停。腕部不要轉。"
    samples, kept = record_demo(driver, leader, writer, check_cameras)
    demo_frames = writer.count
    if not kept:
        driver.term.say("這筆已放棄。")
    options = {"n": "不放置"} | {k: f"放進{TRAY_NAME[t]}" for k, t in (("p", "pink"), ("b", "beige")) if t in drops}
    choice = driver.menu("方塊抬穩了。p 放進粉盤／b 放進米盤／n 不放置（回起姿後人工取下）", options) if kept else "n"
    destination = {"p": "pink", "b": "beige"}.get(choice)
    driver.target = driver.target | {"gripper": max(0.0, driver.read()["gripper"] - CARRY_SQUEEZE)}
    follow(driver, retrace_to(samples + [(driver.clock(), dict(driver.target))], channel), ("抬穩位", "起姿"))
    if destination:
        place_from_channel(driver, drops[destination])
    else:
        driver.menu("在起姿保持。取下夾爪中的方塊（若有）、回程淨空後按 r 回停放位", {"r": "回停放位"})
        driver.target = driver.target | {"gripper": channel["gripper"]}
    park_and_release(driver, route.reverse(), park, support)
    outcome = driver.menu("結果：s 成功（夾對色並放進目的盤）／f 失敗或掉落", {"s": "成功", "f": "失敗"})
    return {"color": color, "destination": destination, "placed": outcome == "s" and destination is not None,
            "success": outcome == "s" and kept, "demo_frames": demo_frames, "cancelled": not kept}


def finalize(root, *, color, destination, success, quads, target_pose=None, slot=None, landed=None, extra=None,
             start="park", mode="demonstration"):
    """Readback, D455 target from the opening frame (fixed trays) and review.json.

    Frames: start (first motion), close, lift (first still moment after lifting), and for demonstrated placement
    release and end. Ranges: grasp [start, lift]; place [lift, end] when the release is in the episode."""
    import json

    from agentic_sim.real_episode import Episode, write_json
    from agentic_sim.real_review import confirmed_frames, locate_opening, review_episode

    report = review_episode(root, source="real", arm="right", rgb_depth={"tray_quads": quads})
    episode = Episode(root)
    probe = {"color": color, "pose": list(target_pose or (0.0, 0.0)) + [0.0]}
    target = located = None
    for tray in ("pink", "beige"):
        result = locate_opening(episode, probe | {"source_tray": tray}, quads)
        if result and result.get("target") and (target is None or result["target"]["card_offset_mm"]
                                                 < target["card_offset_mm"]):
            located, target = result, result["target"]
    marks = {key: value["frame"] for key, value in report.get("candidates", {}).items()}
    review = {"outcome": {"color": color, "destination": destination, "success": success,
                          "placed": bool(success and destination)},
              "target": {"tray": target and target["tray"], "color": color, "frame": "tray-local mm, deg",
                         "card_pose": None, "actual_pose": None, "d455": target},
              "place_target": slot and {"tray": destination, "slot": slot["index"], "xy_mm": slot["xy"],
                                        "landed_d455": landed},
              "start_pose": start} | (extra or {})
    if {"start", "close", "lift"} <= marks.keys():
        frames = confirmed_frames(episode, {k: marks[k] for k in ("start", "close", "lift")})
        for extra in ("release", "end"):
            if extra in marks:
                frames[extra] = {"frame": marks[extra], "timestamp_ns": episode.rows[marks[extra]]["timestamp_ns"]}
        review |= {"frames": frames, "training_range": [frames["start"]["frame"], frames["lift"]["frame"]],
                   "training_ranges": {"grasp": [frames["start"]["frame"], frames["lift"]["frame"]]}}
        if "release" in frames and "end" in frames:
            review["training_ranges"]["place"] = [frames["lift"]["frame"], frames["end"]["frame"]]
    else:
        review["frames"] = None
        report["review_reasons"].append("no gripper closure / lift found: mark close and lift by hand")
    if target is None:
        report["review_reasons"].append(f"D455 found no {color} block in the opening frame")
    elif target["same_colour_in_tray"] > 1 and target_pose is None:
        report["review_reasons"].append(f"more than one {color} block in the {target['tray']} tray")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["task_card"] = {"color": color, "source_tray": target and target["tray"], "destination": destination,
                             "pose": target and [target["x_mm"], target["y_mm"], target["yaw_deg"]],
                             "slot": slot and slot["index"], "slot_xy": slot and slot["xy"],
                             "start": start, "layout_group": None}
    manifest["operator_result"] = review["outcome"]
    manifest["mode"] = mode    # "demonstration" (leader) or "script" (tools/auto_collect.py); real_to_lerobot --modes
    write_json(root / "manifest.json", manifest)
    write_json(root / "review.json", review | {"review_reasons": report["review_reasons"], "d455_blocks": located})
    return report, review

def refresh_marks(root):
    """Recompute only the review marks (frames, training ranges) and readback of a finished episode with the current
    keyframe rules; every other review field (target, round, script / takeover records, landing) is kept."""
    import json
    import time
    from pathlib import Path

    from agentic_sim.real_episode import Episode, write_json
    from agentic_sim.real_review import confirmed_frames, review_episode

    root = Path(root)
    review = json.loads((root / "review.json").read_text(encoding="utf-8"))
    trays = ((review.get("d455_blocks") or {}).get("trays") or {})
    quads = {k: v["quad_px"] for k, v in trays.items()} or None
    report = review_episode(root, source="real", arm="right", rgb_depth={"tray_quads": quads})
    episode = Episode(root)
    marks = {key: value["frame"] for key, value in report.get("candidates", {}).items()}
    for key in ("frames", "training_range", "training_ranges"):
        review.pop(key, None)
    if {"start", "close", "lift"} <= marks.keys():
        frames = confirmed_frames(episode, {k: marks[k] for k in ("start", "close", "lift")})
        for extra in ("release", "end"):
            if extra in marks:
                frames[extra] = {"frame": marks[extra], "timestamp_ns": episode.rows[marks[extra]]["timestamp_ns"]}
        review |= {"frames": frames, "training_range": [frames["start"]["frame"], frames["lift"]["frame"]],
                   "training_ranges": {"grasp": [frames["start"]["frame"], frames["lift"]["frame"]]}}
        if "release" in frames and "end" in frames:
            review["training_ranges"]["place"] = [frames["lift"]["frame"], frames["end"]["frame"]]
    else:
        review["frames"] = None
    review["review_reasons"] = report["review_reasons"]
    review["marks_refreshed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(root / "review.json", review)
    return review


def latest_global(writer):
    """The newest saved global colour frame of the running episode (BGR), or None."""
    import cv2

    from agentic_sim.real_episode import _image_name

    count = getattr(writer, "camera_counts", {}).get("global", 0)
    if not count:
        return None
    return cv2.imread(str(writer.root / "global" / _image_name(count - 1)))


def describe(target, blocks):
    """Terminal wording for the chosen block; with same-colour neighbours, its side relative to the nearest one
    (arm's view: +y is left, +x is away from the base)."""
    name = f"{COLOR_NAME.get(target['color'], target['color'])}方塊（{TRAY_NAME[target['tray']]} " \
           f"x {target['x_mm']:.0f} y {target['y_mm']:.0f} mm）"
    same = [b for b in blocks if b is not target and b["color"] == target["color"] and b["tray"] == target["tray"]]
    if not same:
        return name
    other = min(same, key=lambda b: (b["x_mm"] - target["x_mm"]) ** 2 + (b["y_mm"] - target["y_mm"]) ** 2)
    dx, dy = target["x_mm"] - other["x_mm"], target["y_mm"] - other["y_mm"]
    side = ("靠左" if dy > 0 else "靠右") if abs(dy) >= abs(dx) else ("離手臂較遠" if dx > 0 else "離手臂較近")
    return f"{len(same) + 1} 顆同色中{side}（手臂視角）的{name}"


def random_pick(rng, source):
    """Chooser for pick_place_round: a uniformly random block in the source tray, so the order carries no
    information and the policy has to use the target condition. Returns (block, scene) or (None, scene)."""
    def choose(blocks):
        pool = [b for b in blocks if b["tray"] == source]
        counts = {c: sum(b["color"] == c for b in pool) for c in sorted({b["color"] for b in pool})}
        scene = {"source": source, "candidates": len(pool), "counts": counts,
                 "elsewhere": len(blocks) - len(pool), "selection": "random"}
        return (rng.choice(pool) if pool else None), scene
    return choose


def choose_target(term, blocks, color):
    """The operator picks which block of the colour when there is more than one (advanced finals: a given block)."""
    same = [b for b in blocks if b["color"] == color]
    if len(same) <= 1:
        return same[0] if same else None
    keys = {str(i + 1): b for i, b in enumerate(same[:9])}
    lines = "／".join(f"{k} {TRAY_NAME[b['tray']]} x {b['x_mm']:.0f} y {b['y_mm']:.0f}" for k, b in keys.items())
    return keys[term.choose(f"有 {len(same)} 顆{color}方塊，要夾哪一顆？{lines}", {k: k for k in keys})]


def leader_home_goals(now, channel, route, rest, follower_id):
    """Driven leader from where the demonstration ended back to its folded rest: straight up and to the start pose
    (the follower's way back, lift_and_return), then the taught route reversed, ending at the rest pose read before
    the hand-over (or the route's park pose)."""
    grip = float(now["gripper"])
    back = lift_and_return(now, channel, follower_id)
    goals = list(back.points[1:]) if back is not None else [pose(channel) | {"gripper": grip}]
    rev = route.reverse()
    goals += [pose(p) | {"gripper": grip} for p in rev.points[1:-1]]
    goals.append(pose(rest) if rest is not None else pose(rev.points[-1]))
    return goals


def leader_start_goals(now, channel, follower_id):
    """Driven leader from where the demonstration ended to the start pose (the first part of leader_home_goals):
    between park rounds it waits there for the next hand-over."""
    grip = float(now["gripper"])
    back = lift_and_return(now, channel, follower_id)
    return list(back.points[1:]) if back is not None else [pose(channel) | {"gripper": grip}]


def park_all(driver, leader, data):
    """End of a stretch of rounds at the start pose (a failure to clear up, a new round to lay out, the end): the
    follower takes the fixed route to park and is released there, the waiting leader is driven home to its fold."""
    if not getattr(driver, "at_start", False):
        return
    route = Track(data["waypoints"], data["segment_seconds"])
    park, channel, support = data["park"]["pose"], data["channel"]["command"], data["park_support"]
    driver.target = driver.target | {"gripper": channel["gripper"]}
    park_and_release(driver, route.reverse(), park, support)
    driver.at_start = False
    park_leader_home(driver, leader, data)


def park_leader_home(driver, leader, data):
    """A leader waiting at the start pose: driven home along the reversed route to its own fold and released."""
    if not getattr(driver, "leader_waiting", False):
        return
    rev = Track(data["waypoints"], data["segment_seconds"]).reverse()
    grip = float(pose(leader.read())["gripper"])
    goals = [pose(p) | {"gripper": grip} for p in rev.points[1:-1]]
    goals.append(pose(driver.leader_rest) if driver.leader_rest is not None else pose(rev.points[-1]))
    driver.park_leader(leader, goals)
    driver.leader_waiting = False


def script_recorder(writer, check_cameras):
    """driver.on_send hook for programmed motion inside an episode: rows tagged command_source "script"."""
    rows = [0]

    def on_send(sent, actual):
        rows[0] += 1
        if rows[0] % 15 == 0 and not getattr(writer, "camera_error", None):
            try:
                check_cameras()
            except RuntimeError as error:   # never stop a carry over the trays: finish the place, fail the episode
                writer.camera_error = str(error)
        writer.add_control(control_row(writer, sent, actual) | {"command_source": "script", "task_is_policy": 1})
    return on_send


def program_place(driver, placer, slot_mount, label):
    """After a demonstration that ended at a stable lift: carry the held block into the slot with IK
    (agentic_sim.real_place), open, rise over the slot. With a recorder on driver.on_send these are the episode's
    "script" rows (the IK is planned while the follower keeps sending, so the rows have no gap)."""
    from agentic_sim.real_place import HELD_PCT, hold_while
    grip = float(driver.read()["gripper"])
    if grip < HELD_PCT:
        driver.term.say(f"夾爪讀數 {grip:.1f}%：像是空爪，不放置，直接回起姿。")
        return {"held": False, "gripper_pct": round(grip, 1)}
    xy, yaw = slot_mount
    current = dict(driver.target)
    plan = hold_while(driver, lambda: placer.plan_place(xy, yaw, yaw, current["gripper"], current))
    err = plan if isinstance(plan, str) else None
    if err:
        driver.term.say(f"程式放置沒有執行：{err}")
    else:
        placer.execute_place(plan, label)
    return {"held": True, "gripper_pct": round(grip, 1), "slot_mount_m": [round(float(v), 4) for v in xy],
            "error": err}


def pick_place_round(driver, leader, data, writer, check_cameras, *, color, destination, slot, quads, choose=None,
                     placer=None, slot_mount=None, stay=False):
    """Park -> demonstrated pick and place (ending near the start pose) -> park. Returns the operator result.
    With choose (see random_pick) the target block, and so its colour, is picked from the opening D455 frame.
    With placer (agentic_sim.real_place.Placer) and slot_mount ((x, y) m, tray yaw): the demonstration ends at the
    stable lift (c) and the program places the block (the place height and release varied too much by hand)."""
    from agentic_sim.real_locate import locate

    park, channel, support = data["park"]["pose"], data["channel"]["command"], data["park_support"]
    route = Track(data["waypoints"], data["segment_seconds"])
    driver.leader_route = route
    opening = latest_global(writer)
    blocks = locate(opening, quads)["blocks"] if opening is not None else []
    scene = None
    if choose is not None:
        target, scene = choose(blocks)
        if target is None:
            driver.term.say(f"D455 開場畫面在{TRAY_NAME[scene['source']]}沒找到方塊，這筆不錄。")
            return {"cancelled": True, "demo_frames": None, "scene": scene}
        color = target["color"]
        driver.term.say(f"這筆夾：{describe(target, blocks)}")
    else:
        target = choose_target(driver.term, blocks, color) if opening is not None else None
    if target is None:
        driver.term.say(f"D455 開場畫面沒找到{color}方塊；仍可錄，讀回時再確認。")
    if placer is not None:
        driver.demo_prompt = (f"錄製中：從起姿 → 夾起{color}方塊 → 抬起約 5 cm 停穩，按 c（不用放）。"
                              f"按 c 後 leader 會上扭力定住，手放開；程式把方塊放進{TRAY_NAME[destination]} P{slot['index']}，"
                              "follower 回停放位後 leader 自動收回。Esc 暫停；腕部只在 ±30° 內對齊方塊。")
    else:
        driver.demo_prompt = (f"錄製中：從停靠位出收折區 → 夾起{color}方塊並抬穩 → 放進{TRAY_NAME[destination]} P{slot['index']}"
                              "（overlay 圖上的格位：米盤靠基座、靠粉盤的角落，P1 最近；貼近盤面再張爪）→ 抬起退回起姿附近，停穩按 c。"
                              "Esc 暫停；腕部只在 ±30° 內對齊方塊。")
    if not (stay and getattr(driver, "at_start", False)):
        driver.arm_at(park)
        if placer is not None:
            # The demonstration starts at the start pose (ACT learns start -> lift); the fixed route gets there.
            follow(driver, route, ("停放位", "起姿"))
    # else: since the last round the follower holds at the start pose and the leader waits there (stay)
    driver.at_start = placer is not None
    if not driver.align_leader(leader, driver.read()):
        if stay:
            park_all(driver, leader, data)
        else:
            park_and_release(driver, route.reverse() if placer is not None else Track([driver.target, park], [0.0]),
                             park, support)
            driver.at_start = False
        return {"cancelled": True, "demo_frames": None}
    driver.leader_waiting = False
    driver.at_start = False          # the demonstration moves the arm away: a stop now holds it there (recover)
    back, programmed = None, None
    while True:
        samples, kept = record_demo(driver, leader, writer, check_cameras)
        if kept and placer is not None:
            break
        if not kept or reach_gap(driver.read(), channel) <= END_NEAR_START:
            break
        back = lift_and_return(driver.target, channel, data["calibrations"]["follower"]["id"])
        if back is not None:      # c after placing and lifting: straight up, then to the start pose (not recorded)
            driver.term.say("錄製結束。夾爪先垂直抬高，再回起姿與停放位。")
            break
        choice = driver.menu(f"離起姿還有 {reach_gap(driver.last, channel):.0f}°。t 重新交接繼續錄、把手臂帶回起姿附近"
                             "／r 沿剛才的路徑退回起姿（這段不錄）", {"t": "繼續錄", "r": "退回"})
        if choice == "r" or not driver.align_leader(leader, driver.read()):
            break
    demo_frames = writer.count
    leader_end = None
    if kept and placer is not None:
        # The recording goes on through the programmed placing (rows "script") up to the rise over the slot.
        from agentic_sim.real_place import hold_while

        def hold():                      # c: the leader holds where the operator left it
            driver.hold_leader(leader)
            return pose(leader.read())
        driver.on_send = script_recorder(writer, check_cameras)
        try:
            leader_end = hold_while(driver, hold)
            driver.term.say("leader 已上扭力定住，手可以放開。")
            programmed = program_place(driver, placer, slot_mount, f"{TRAY_NAME[destination]} P{slot['index']}")
        finally:
            driver.on_send = None
        back = lift_and_return(driver.target, channel, data["calibrations"]["follower"]["id"])
        if back is None:
            driver.term.say("從這裡回起姿的路徑算不出來，改沿示範路徑退回。")
    if back is not None:
        follow(driver, back, ("結束位", "起姿"))
    elif reach_gap(driver.read(), channel) <= END_NEAR_START:
        follow(driver, retimed([dict(driver.target), pose(channel) | {"gripper": driver.target["gripper"]}]),
               ("結束位", "起姿"))
    else:
        follow(driver, retrace_to(samples + [(driver.clock(), dict(driver.target))], channel), ("結束位", "起姿"))
    if not kept or (programmed and programmed.get("held") and programmed.get("error")):
        driver.menu("在起姿保持。若夾爪裡還有方塊請取下，回程淨空後按 r 回停放位", {"r": "回停放位"})
    driver.target = driver.target | {"gripper": channel["gripper"]}
    trouble = not kept or (programmed and programmed.get("held") and programmed.get("error"))
    if stay and placer is not None and leader_end is not None and not trouble:
        # Stay at the start pose for the next round (no trip through the fold area): the follower holds there and
        # the leader is driven from where c left it back to the start pose, where it waits torque-on.
        driver.at_start = True
        driver.leader_waiting = driver.bring_leader(leader, leader_start_goals(
            leader_end, channel, data["calibrations"]["follower"]["id"]))
    else:
        park_and_release(driver, route.reverse(), park, support)
        driver.at_start = False
        if leader_end is not None:
            driver.park_leader(leader, leader_home_goals(leader_end, channel, route, driver.leader_rest,
                                                         data["calibrations"]["follower"]["id"]))
    # Landing is read only now, the arm at park or at the start pose: the D455 sees both trays past it there.
    final = latest_global(writer) if kept else None
    after = locate(final, quads)["blocks"] if final is not None else None
    landed = placed_block(blocks, after, color, destination, slot) if after is not None else None
    if after is not None and target is not None:
        taken = taken_block(blocks, after, color, target["tray"])
        if taken is not None and ((taken["x_mm"] - target["x_mm"]) ** 2 + (taken["y_mm"] - target["y_mm"]) ** 2) ** .5 > 25:
            driver.term.say(f"注意：夾走的是另一顆（x {taken['x_mm']:.0f} y {taken['y_mm']:.0f}），不是指定的 "
                            f"x {target['x_mm']:.0f} y {target['y_mm']:.0f}；目標改記成實際夾走的那顆。")
            target = taken | {"relabelled_from": [target["x_mm"], target["y_mm"]]}
    seen = (f"D455（停放位）：落點{TRAY_NAME[destination]} x {landed['x_mm']:.0f} y {landed['y_mm']:.0f}，"
            f"離參考位置 {landed['slot_offset_mm']:.0f} mm" + (
                f"，離旁邊方塊只有 {landed['neighbour_mm']:.0f} mm（太擠，下一顆可能撞到）" if
                landed["neighbour_mm"] is not None and landed["neighbour_mm"] < CROWDED_MM else "") + "。"
            if landed else "D455 沒在目的盤看到新放的方塊。")
    camera_error = getattr(driver, "camera_error", None) or getattr(writer, "camera_error", None)
    if camera_error:
        seen = f"錄製中相機中斷（{camera_error}），這筆不能用，請按 f。" + seen
    outcome = driver.menu(seen + "結果：s 成功（收進資料）／f 失敗（移出、重錄這一格）", {"s": "成功", "f": "失敗"})
    return {"color": color, "destination": destination, "slot": slot, "target_choice": target, "scene": scene,
            "landed": landed, "program_place": programmed, "placed": landed is not None and outcome == "s", "success": outcome == "s" and kept,
            "demo_frames": demo_frames, "cancelled": not kept}
