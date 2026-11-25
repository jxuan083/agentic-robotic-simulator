"""One grasp demonstration from the park pose, driven from a terminal (no task cards).

  park (torque on, small reseat) -> leader handed over at park -> [record: unfold, approach, close, lift] c ->
  retrace to the start pose -> taught drop route of the chosen tray -> start pose -> park -> torque off

The recorded control rows cover only the demonstration (park to stable lift); cameras run for the whole round,
so the opening global frame shows the untouched layout. The target block is found in that frame (fixed tray
corners), and review.json gets start/close/lift and the D455 target like the collection desk.
"""

from agentic_sim.real_control import JOINTS, pose
from agentic_sim.real_route import (
    CARRY_SQUEEZE, Paused, Track, _simplify, body_gap, follow, park_and_release, place_from_channel, retimed,
)

COLORS = {"r": "red", "b": "blue", "y": "yellow"}
TRAY_NAME = {"pink": "粉盤", "beige": "米盤"}
RETRACE_ERROR = 2.0


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
                driver.term.say("錄製中：從停靠位用 leader 出收折區、接近、閉爪夾起並抬穩，停約 1 秒後按 c；Esc 暫停。腕部不要轉。")
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


def finalize(root, *, color, destination, success, quads):
    """Readback, D455 target from the opening frame (fixed trays) and review.json (start/close/lift)."""
    import json

    from agentic_sim.real_episode import Episode, write_json
    from agentic_sim.real_review import confirmed_frames, locate_opening, review_episode

    report = review_episode(root, source="real", arm="right", rgb_depth={"tray_quads": quads})
    episode = Episode(root)
    probe = {"color": color, "pose": [0.0, 0.0, 0.0]}
    target = None
    located = None
    for tray in ("pink", "beige"):
        result = locate_opening(episode, probe | {"source_tray": tray}, quads)
        if result and result.get("target"):
            located, target = result, result["target"]
            break
    marks = {key: value["frame"] for key, value in report.get("candidates", {}).items()}
    review = {"outcome": {"color": color, "destination": destination, "success": success,
                          "placed": bool(success and destination)},
              "target": {"tray": target and target["tray"], "color": color, "frame": "tray-local mm, deg",
                         "card_pose": None, "actual_pose": None, "d455": target},
              "start_pose": "park"}
    if {"close", "lift"} <= marks.keys():
        frames = confirmed_frames(episode, {"start": 0, "close": marks["close"], "lift": marks["lift"]})
        review |= {"frames": frames, "training_range": [frames["start"]["frame"], frames["lift"]["frame"]]}
    else:
        review["frames"] = None
        report["review_reasons"].append("no gripper closure / lift found: mark close and lift by hand")
    if target is None:
        report["review_reasons"].append(f"D455 found no single {color} block in the opening frame")
    elif target["same_colour_in_tray"] > 1:
        report["review_reasons"].append(f"more than one {color} block in the {target['tray']} tray")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["task_card"] = {"color": color, "source_tray": target and target["tray"], "destination": destination,
                             "pose": target and [target["x_mm"], target["y_mm"], target["yaw_deg"]],
                             "start": "park", "layout_group": None}
    manifest["operator_result"] = review["outcome"]
    write_json(root / "manifest.json", manifest)
    write_json(root / "review.json", review | {"review_reasons": report["review_reasons"], "d455_blocks": located})
    return report, review
