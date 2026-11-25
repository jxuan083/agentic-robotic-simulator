"""One pick-and-place demonstration from the park pose, driven from a terminal (no task cards).

  park (torque on, small reseat) -> leader handed over at park ->
  [record: unfold, approach, close, lift, carry, place in the chosen slot, open, retreat near the start pose] c ->
  start pose -> park -> torque off

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

    def on_send(sent, actual):
        rows[0] += 1
        if rows[0] % 15 == 0:
            check_cameras()
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


def finalize(root, *, color, destination, success, quads, target_pose=None, slot=None, landed=None, extra=None):
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
              "start_pose": "park"} | (extra or {})
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
                             "start": "park", "layout_group": None}
    manifest["operator_result"] = review["outcome"]
    write_json(root / "manifest.json", manifest)
    write_json(root / "review.json", review | {"review_reasons": report["review_reasons"], "d455_blocks": located})
    return report, review

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


def pick_place_round(driver, leader, data, writer, check_cameras, *, color, destination, slot, quads, choose=None):
    """Park -> demonstrated pick and place (ending near the start pose) -> park. Returns the operator result.
    With choose (see random_pick) the target block, and so its colour, is picked from the opening D455 frame."""
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
    driver.demo_prompt = (f"錄製中：從停靠位出收折區 → 夾起{color}方塊並抬穩 → 放進{TRAY_NAME[destination]}第 {slot['index']} 個位置"
                          "（照固定順序接在上一顆旁邊，貼近盤面再張爪）→ 抬起退回起姿附近，停穩按 c。"
                          "Esc 暫停，腕部不要轉。")
    driver.arm_at(park)
    if not driver.align_leader(leader, driver.read()):
        park_and_release(driver, Track([driver.target, park], [0.0]), park, support)
        return {"cancelled": True, "demo_frames": None}
    while True:
        samples, kept = record_demo(driver, leader, writer, check_cameras)
        if not kept or body_gap(driver.read(), channel) <= END_NEAR_START:
            break
        choice = driver.menu(f"離起姿還有 {body_gap(driver.last, channel):.0f}°。t 重新交接繼續錄、把手臂帶回起姿附近"
                             "／r 沿剛才的路徑退回起姿（這段不錄）", {"t": "繼續錄", "r": "退回"})
        if choice == "r" or not driver.align_leader(leader, driver.read()):
            break
    demo_frames = writer.count
    if body_gap(driver.read(), channel) <= END_NEAR_START:
        follow(driver, retimed([dict(driver.target), pose(channel) | {"gripper": driver.target["gripper"]}]),
               ("結束位", "起姿"))
    else:
        follow(driver, retrace_to(samples + [(driver.clock(), dict(driver.target))], channel), ("結束位", "起姿"))
    if not kept:
        driver.menu("在起姿保持。若夾爪裡還有方塊請取下，回程淨空後按 r 回停放位", {"r": "回停放位"})
    driver.target = driver.target | {"gripper": channel["gripper"]}
    park_and_release(driver, route.reverse(), park, support)
    # Landing is read only now: with the arm folded at park it neither hides the trays nor shades the colour masks.
    final = latest_global(writer) if kept else None
    landed = placed_block(blocks, locate(final, quads)["blocks"], color, destination, slot) if final is not None else None
    seen = (f"D455（停放位）：落點{TRAY_NAME[destination]} x {landed['x_mm']:.0f} y {landed['y_mm']:.0f}，"
            f"離參考位置 {landed['slot_offset_mm']:.0f} mm" + (
                f"，離旁邊方塊只有 {landed['neighbour_mm']:.0f} mm（太擠，下一顆可能撞到）" if
                landed["neighbour_mm"] is not None and landed["neighbour_mm"] < CROWDED_MM else "") + "。"
            if landed else "D455 沒在目的盤看到新放的方塊。")
    outcome = driver.menu(seen + "結果：s 成功／f 失敗", {"s": "成功", "f": "失敗"})
    return {"color": color, "destination": destination, "slot": slot, "target_choice": target, "scene": scene,
            "landed": landed, "placed": landed is not None and outcome == "s", "success": outcome == "s" and kept,
            "demo_frames": demo_frames, "cancelled": not kept}
