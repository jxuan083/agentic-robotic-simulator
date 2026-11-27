"""Park <-> channel route: teach with the leader, reverse replay, pause/obstruction, recovery."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_sim.real_motion import OFF, ON  # noqa: E402
from agentic_sim.real_motion import MotionPlan  # noqa: E402
from agentic_sim.real_route import (  # noqa: E402
    BODY, RELEASE_LAG, Driver, Released, Track, body_gap, build_setup, compress, load_route, record_drop,
    recover, record, roundtrip, sample_park,
)
from agentic_sim.real_route_fake import (  # noqa: E402
    CHANNEL, PARK, VIA, FakeClock, FakeFollower, FakeLeader, ScriptedTerminal, operator_path,
)

META = {"arm": "right", "layout_id": "fake-layout", "park_support": "fake support",
        "calibrations": {"leader": {"id": "l", "version": "1"}, "follower": {"id": "f", "version": "1"}}}


class RouteCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.route = self.tmp / "route.json"

    def driver(self, follower, term, clock, name, **kw):
        return Driver(follower, term, self.tmp / name, hz=60, clock=clock, wait=clock.sleep, **kw)

    def teach(self, keyframes=None, keys=None, answers=("r", "y"), **follower_kw):
        clock = FakeClock()
        follower = FakeFollower(clock, **follower_kw)
        leader = FakeLeader(clock, keyframes or operator_path())
        term = ScriptedTerminal(clock, keys if keys is not None else [(0.3, "enter"), (7.5, "c")], answers)
        driver = self.driver(follower, term, clock, "record")
        data = record(driver, leader, self.route, META, sample_park(follower, clock.sleep))
        driver.close_logs()
        return data, follower, term

    def assert_released_at_park(self, follower):
        self.assertEqual(follower.torque_states(), OFF)
        self.assertLessEqual(body_gap(follower.state, PARK), RELEASE_LAG)

    def test_teach_then_reverse_to_park(self):
        data, follower, term = self.teach()
        self.assertIsNotNone(data)
        self.assert_released_at_park(follower)
        saved, track = load_route(self.route)
        self.assertIsNotNone(saved["verified"]["reverse_to_park"])
        self.assertLess(body_gap(saved["channel"]["readback"], CHANNEL), 3.0)
        self.assertEqual(track.points[0], saved["park"]["pose"])
        self.assertLess(len(track.points), 80, "route should be simplified")
        # The reverse never left the corridor of the forward path: wrist roll stays low until the
        # arm has left the support (the taught order), also on the way back.
        for _, cmd in follower.sent:
            if cmd["shoulder_lift"] < -95:
                self.assertLess(cmd["wrist_roll"], 12.0)

    def test_range_edge_only_clips(self):
        # Leader pushes elbow beyond the follower's calibrated range: command clips, no abort.
        beyond = dict(PARK) | {"elbow_flex": 130.0}
        frames = [(0.0, PARK), (1.0, PARK), (2.0, beyond), (3.5, PARK),
                  *[(t + 4.0, p) for t, p in operator_path(offset=0)[1:]]]
        data, follower, term = self.teach(frames, keys=[(0.3, "enter"), (11.5, "c")])
        self.assertIsNotNone(data)
        self.assertTrue(all(cmd["elbow_flex"] <= 97.0 + 1e-9 for _, cmd in follower.sent))
        self.assertFalse(any("已暫停" in line for line in term.lines))
        self.assert_released_at_park(follower)

    def test_leader_far_from_follower_waits(self):
        far = dict(PARK) | {"wrist_roll": 60.0}
        frames = [(0.0, far), (2.0, far), (3.0, PARK), *[(t + 3.0, p) for t, p in operator_path(offset=0)[1:]]]
        # Enter at 0.3 s is refused (gap), the second Enter at 3.5 s engages.
        data, follower, term = self.teach(frames, keys=[(0.3, "enter"), (3.5, "enter"), (10.5, "c")])
        self.assertIsNotNone(data)
        self.assertTrue(any("差距仍超過門檻" in line for line in term.lines))
        self.assert_released_at_park(follower)

    def run_route(self, keys=(), answers=("r", "y"), **follower_kw):
        clock = FakeClock()
        follower = FakeFollower(clock, **follower_kw)
        term = ScriptedTerminal(clock, keys, answers)
        driver = self.driver(follower, term, clock, f"run-{len(list(self.tmp.iterdir()))}")
        self.last_follower = follower
        data, track = load_route(self.route)
        try:
            roundtrip(driver, None, self.route, data, track)
        finally:
            driver.close_logs()
        return follower, term

    def test_roundtrip(self):
        self.teach()
        follower, _ = self.run_route()
        self.assert_released_at_park(follower)
        self.assertIsNotNone(load_route(self.route)[0]["verified"]["forward_to_channel"])

    def test_obstruction_pauses_then_backs_out(self):
        self.teach()

        def block(state, now):  # elbow cannot go below 60 deg (an obstacle on the way out)
            return state | {"elbow_flex": max(60.0, state["elbow_flex"])}

        # Paused by the stall detector, b = go back to park, confirm support.
        follower, term = self.run_route(answers=("b", "y"), block=block)
        self.assertTrue(any("停滯" in line for line in term.lines))
        self.assert_released_at_park(follower)

    def test_operator_stop_continue_and_refused_emergency(self):
        self.teach()
        # Esc mid-way; x then a wrong word re-asks; g continues; at channel r; support y.
        follower, term = self.run_route(keys=[(1.0, "esc")], answers=("x", "no", "g", "r", "y"))
        self.assert_released_at_park(follower)
        self.assertTrue(any("操作者暫停" in line for line in term.lines))

    def test_emergency_release(self):
        self.teach()
        with self.assertRaises(Released):
            self.run_route(keys=[(1.0, "esc")], answers=("x", "OFF"))
        self.assertEqual(self.last_follower.torque_states(), OFF)

    def test_recover_from_mid_route(self):
        self.teach()
        _, track = load_route(self.route)
        clock = FakeClock()
        middle = track.points[len(track.points) // 2]
        follower = FakeFollower(clock, start=middle, torque=True)
        term = ScriptedTerminal(clock, answers=("r", "y"))
        driver = self.driver(follower, term, clock, "recover")
        data, track = load_route(self.route)
        self.assertTrue(recover(driver, data, track))
        driver.close_logs()
        self.assert_released_at_park(follower)

    def test_slow_follower_slows_playback_instead_of_aborting(self):
        self.teach()
        follower, term = self.run_route(tau=0.35)  # 6x slower servo
        self.assert_released_at_park(follower)
        self.assertFalse(any("停滯" in line for line in term.lines))

    def test_relative_mapping(self):
        clock = FakeClock()
        follower = FakeFollower(clock)
        shift = {j: (20.0 if j in BODY else 0.0) for j in PARK}
        frames = [(t, {j: p[j] + shift[j] for j in p}) for t, p in operator_path(offset=0)]
        leader = FakeLeader(clock, frames)
        term = ScriptedTerminal(clock, [(0.3, "enter"), (7.5, "c")], ("r", "y"))
        driver = self.driver(follower, term, clock, "relative", relative=True)
        self.assertIsNotNone(record(driver, leader, self.route, META, sample_park(follower, clock.sleep)))
        driver.close_logs()
        self.assert_released_at_park(follower)
        self.assertLess(body_gap(load_route(self.route)[0]["channel"]["readback"], CHANNEL), 3.0)


class DropCase(unittest.TestCase):
    setUp = RouteCase.setUp
    driver = RouteCase.driver
    teach = RouteCase.teach
    assert_released_at_park = RouteCase.assert_released_at_park

    def test_drop_route_and_collection_setup(self):
        self.teach()
        data, track = load_route(self.route)
        drop_pose = dict(CHANNEL) | {"shoulder_pan": 20.0, "shoulder_lift": -40.0}
        clock = FakeClock()
        follower = FakeFollower(clock)
        leader = FakeLeader(clock, [(0.0, CHANNEL), (22.0, CHANNEL), (26.0, drop_pose)])
        term = ScriptedTerminal(clock, [(20.0, "enter"), (30.0, "c")], ("r", "y"))
        driver = self.driver(follower, term, clock, "drop")
        out = record_drop(driver, leader, data, track, self.tmp / "drop-pink.json", "pink", 60.0)
        driver.close_logs()
        self.assertIsNotNone(out)
        self.assert_released_at_park(follower)
        self.assertLess(body_gap(out["drop_readback"], drop_pose), 3.0)
        self.assertLess(body_gap(out["waypoints"][0], data["channel"]["command"]), 1e-9)
        cards = {"seed": 1, "reserve_distance_mm": 20, "zones": {}}
        setup = build_setup(data, track, [out], version="v1", workspace="fake", cards=cards)
        plan = MotionPlan.load(setup["motion"])
        self.assertEqual(sorted(plan.drops), ["pink"])
        self.assertLess(body_gap(plan.start, data["channel"]["command"]), 1e-9)

        # Replay it without the leader, holding a block: close at the channel, carry, open, back home.
        from agentic_sim.real_route import drop_test

        clock = FakeClock()
        follower = FakeFollower(clock)
        term = ScriptedTerminal(clock, answers=("b", "r", "y"))
        driver = self.driver(follower, term, clock, "drop-test")
        self.assertTrue(drop_test(driver, data, track, out))
        driver.close_logs()
        self.assert_released_at_park(follower)
        at_drop = [cmd for _, cmd in follower.sent if body_gap(cmd, drop_pose) < 1.0]
        self.assertTrue(at_drop)
        self.assertLess(min(cmd["gripper"] for cmd in at_drop), 1.0)        # carried closed
        self.assertGreaterEqual(max(cmd["gripper"] for cmd in at_drop), 60.0)  # released at the drop pose


class TrackCase(unittest.TestCase):
    def test_compress_drops_idle_time_and_keeps_shape(self):
        a, b = dict(PARK), dict(CHANNEL)
        samples = [(i * 16_666_667, a) for i in range(120)]  # 2 s idle
        samples += [(2e9 + i * 16_666_667, {j: a[j] + (b[j] - a[j]) * i / 60 for j in a}) for i in range(61)]
        track = compress(samples)
        self.assertEqual(track.points[0], a)
        self.assertLess(body_gap(track.points[-1], b), 1e-9)
        self.assertLess(sum(track.seconds), 1.2)
        self.assertLessEqual(len(track.points), 4)

    def test_tail_and_head_back(self):
        pts = [dict(PARK, shoulder_pan=float(i)) for i in range(4)]
        track = Track(pts, [1, 2, 3])
        cur = dict(PARK, shoulder_pan=1.5)
        self.assertEqual([p["shoulder_pan"] for p in track.tail(1, cur).points], [1.5, 2, 3])
        self.assertEqual(track.tail(1, cur).seconds, [2, 3])
        self.assertEqual([p["shoulder_pan"] for p in track.head_back(1, cur).points], [1.5, 1, 0])
        self.assertEqual(track.head_back(1, cur).seconds, [2, 1])


if __name__ == "__main__":
    unittest.main()


def test_tune_route_removes_pauses_and_fixes_gripper():
    from agentic_sim.real_route import PLAY_SPEED, Track, body_gap, tune_route

    park = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 90.0, "wrist_flex": 60.0,
            "wrist_roll": 0.0, "gripper": 1.0}
    points = [park | {"shoulder_lift": -100.0 + 5 * i, "elbow_flex": 90.0 - 4 * i, "gripper": 1.0 + (i % 3) * 4}
              for i in range(13)]
    seconds = [0.2] * 12
    seconds[2] = 25.0  # operator paused while teaching
    data = {"waypoints": points, "segment_seconds": seconds, "park": {"pose": park},
            "channel": {"command": points[-1], "readback": points[-1]},
            "verified": {"reverse_to_park": "run", "forward_to_channel": "run"}}
    tuned = tune_route(data, grip=45.0)
    track = Track(tuned["waypoints"], tuned["segment_seconds"])
    assert max(track.seconds) < 2.0
    assert all(s >= body_gap(a, b) / PLAY_SPEED - 1e-9 for a, b, s in zip(track.points, track.points[1:], track.seconds))
    grips = [p["gripper"] for p in track.points]
    assert grips[0] == 1.0 and grips[-1] == 45.0 and set(grips) == {1.0, 45.0}
    opened = grips.index(45.0)
    assert track.points[opened]["shoulder_lift"] - park["shoulder_lift"] >= 45.0
    assert tuned["channel"]["command"]["gripper"] == 45.0
    assert tuned["verified"] == {"reverse_to_park": None, "forward_to_channel": None}
    assert track.points[0] == park
    held = tune_route(data, grip=45.0, hold_roll=True)
    assert {p["wrist_roll"] for p in held["waypoints"]} == {park["wrist_roll"]}


class DrivenHandoverCase(RouteCase):
    """Leader driven to the follower pose, countdown, release while the follower holds, clutch from there."""

    def teach_driven(self, start, moves, keys, answers, sag=None):
        from agentic_sim.real_route_fake import FakeDrivenLeader

        clock = FakeClock()
        follower = FakeFollower(clock)
        leader = FakeDrivenLeader(clock, start, moves, sag=sag)
        term = ScriptedTerminal(clock, keys, answers)
        driver = self.driver(follower, term, clock, "record")
        data = record(driver, leader, self.route, META, sample_park(follower, clock.sleep))
        driver.close_logs()
        return data, follower, leader, term, driver

    def test_driven_handover_teaches_route_and_ignores_sag(self):
        from agentic_sim.real_route import HANDOVER_COUNTDOWN_S, HANDOVER_GUARD_S

        start = dict(PARK) | {"wrist_roll": 50.0, "shoulder_pan": 20.0}  # leader left far from the follower
        to_via = {j: VIA[j] - PARK[j] for j in VIA}
        to_channel = {j: CHANNEL[j] - PARK[j] for j in CHANNEL}
        moves = [(1.0, dict.fromkeys(PARK, 0.0)), (3.0, to_via), (5.5, to_channel)]
        data, follower, leader, term, driver = self.teach_driven(start, moves, [(16.0, "c")], ("a", "r", "y"))
        self.assertIsNotNone(data)
        self.assert_released_at_park(follower)
        self.assertEqual([kind for kind, _ in leader.log], ["on", "off"])
        on, off = leader.log[0][1], leader.log[1][1]
        self.assertGreaterEqual((off - on) / 1e9, HANDOVER_COUNTDOWN_S)  # released only after the countdown
        self.assertFalse(leader.driven)
        # The follower never followed the sag: its first teleop targets stay at park.
        first = [cmd for t, cmd in follower.sent if t <= off + (HANDOVER_GUARD_S + 0.15) * 1e9]
        self.assertTrue(first and all(body_gap(cmd, PARK) < 1.0 for cmd in first))
        saved, _ = load_route(self.route)
        self.assertLess(body_gap(saved["channel"]["readback"], CHANNEL), 3.0)
        self.assertTrue(any("已交接" in line for line in term.lines))

    def test_leader_dropped_after_release_falls_back_to_manual(self):
        data, follower, leader, term, _ = self.teach_driven(
            PARK, [], [(12.0, "esc")], ("a",), sag={"shoulder_lift": -25.0})
        self.assertIsNone(data)
        self.assertFalse(leader.driven)
        self.assertTrue(any("沒有扶穩" in line for line in term.lines))
        self.assertEqual(follower.torque_states(), OFF)

    def test_cancel_during_countdown_releases_only_when_supported(self):
        # Esc during the countdown: leader keeps torque until the operator presses Enter.
        data, follower, leader, term, _ = self.teach_driven(
            PARK, [], [(1.5, "esc"), (6.0, "enter")], ("a", "q"))
        self.assertIsNone(data)
        on, off = leader.log[0][1], leader.log[1][1]
        self.assertGreaterEqual(off / 1e9, 6.0)
        self.assertFalse(leader.driven)
        self.assertEqual(follower.torque_states(), OFF)


class ReseatCase(RouteCase):
    def test_small_offset_from_park_is_reseated_under_torque(self):
        clock = FakeClock()
        follower = FakeFollower(clock, start=dict(PARK) | {"wrist_roll": PARK["wrist_roll"] - 9.8})
        term = ScriptedTerminal(clock, answers=("s",))
        driver = self.driver(follower, term, clock, "reseat")
        driver.arm_at(PARK)
        self.assertEqual(follower.torque_states(), ON)
        self.assertLess(body_gap(follower.read(), PARK), 2.0)
        self.assertTrue(all(abs(cmd["wrist_roll"] - PARK["wrist_roll"]) <= 9.8 + 1e-6 for _, cmd in follower.sent))

    def test_declined_or_large_offset_is_not_moved(self):
        for start, answers in ((dict(PARK) | {"wrist_roll": PARK["wrist_roll"] - 9.8}, ("q",)),
                               (dict(PARK) | {"shoulder_pan": PARK["shoulder_pan"] + 25.0}, ())):
            clock = FakeClock()
            follower = FakeFollower(clock, start=start)
            driver = self.driver(follower, ScriptedTerminal(clock, answers=answers), clock, f"reseat-{len(answers)}")
            with self.assertRaises(RuntimeError):
                driver.arm_at(PARK)
            self.assertEqual(follower.torque_states(), OFF)
            self.assertFalse(follower.sent)


class SagCase(RouteCase):
    def test_steady_gravity_sag_above_gate_does_not_pause(self):
        from agentic_sim.real_route import GATE_LAG, PUSH_LAG

        clock = FakeClock()
        sag = dict.fromkeys(PARK, 0.0) | {"shoulder_lift": (GATE_LAG + PUSH_LAG) / 2}  # e.g. stretched arm
        follower = FakeFollower(clock, sag=sag)
        term = ScriptedTerminal(clock)
        driver = self.driver(follower, term, clock, "sag")
        driver.arm_at(PARK)
        driver.play(Track([PARK, CHANNEL], [1.0]), "往通道位")  # raises Paused if treated as a stall
        self.assertLess(body_gap(follower.goal, CHANNEL), 1e-9)

    def test_lag_beyond_push_still_pauses(self):
        from agentic_sim.real_route import PUSH_LAG, Paused

        clock = FakeClock()
        sag = dict.fromkeys(PARK, 0.0) | {"shoulder_lift": PUSH_LAG + 3.0}
        follower = FakeFollower(clock, sag=sag)
        driver = self.driver(follower, ScriptedTerminal(clock), clock, "blocked")
        driver.arm_at(PARK)
        with self.assertRaises(Paused):
            driver.play(Track([PARK, CHANNEL], [1.0]), "往通道位")


def test_lift_home_raises_first_then_goes_straight():
    from agentic_sim.real_route import lift_home

    grasp = dict(CHANNEL) | {"shoulder_pan": -56.0, "shoulder_lift": 0.4, "elbow_flex": 28.7, "gripper": 16.5}
    track = lift_home(grasp, CHANNEL)
    assert len(track.points) == 3
    assert track.points[1] == grasp | {"shoulder_lift": CHANNEL["shoulder_lift"]}
    assert body_gap(track.points[-1], CHANNEL) < 1e-9 and track.points[-1]["gripper"] == 16.5
    assert len(lift_home(CHANNEL | {"shoulder_pan": 10.0}, CHANNEL).points) == 2  # already at channel height


class LeaderRouteCase(RouteCase):
    def test_folded_leader_follows_route_and_opens_gripper_last(self):
        from agentic_sim.real_route_fake import FakeDrivenLeader

        clock = FakeClock()
        follower = FakeFollower(clock)
        term = ScriptedTerminal(clock)
        driver = self.driver(follower, term, clock, "leader-route")
        route = Track([PARK, VIA, CHANNEL], [1.0, 1.0])
        driver.leader_route = route
        start = dict(PARK) | {"shoulder_pan": PARK["shoulder_pan"] + 4.0, "gripper": 1.0}
        reference = dict(CHANNEL) | {"gripper": 26.0}
        goals = driver._leader_goals(start, reference)
        self.assertEqual(len(goals), 4)  # VIA, CHANNEL (gripper held), reference body, then gripper
        self.assertTrue(all(g["gripper"] == 1.0 for g in goals[:-1]))
        self.assertEqual(goals[-1]["gripper"], 26.0)
        leader = FakeDrivenLeader(clock, start)
        leader.drive_on()
        self.assertEqual(driver._drive_leader(leader, reference), "arrived")
        # Passed close to the route's via pose on the way (not a straight joint move).
        driver.leader_route = None
        self.assertEqual(len(driver._leader_goals(start, reference)), 2)


def test_retrace_home_reverses_the_driven_path():
    from agentic_sim.real_route import retrace_home

    out = [(i, dict(CHANNEL) | {"shoulder_lift": CHANNEL["shoulder_lift"] + 4 * i,
                                "elbow_flex": CHANNEL["elbow_flex"] - 5 * min(i, 8), "gripper": 17.0})
           for i in range(16)]
    track = retrace_home(out, CHANNEL)
    lifts = [p["shoulder_lift"] for p in track.points]
    assert lifts == sorted(lifts, reverse=True)          # monotonically back, no swing past the channel
    assert body_gap(track.points[-1], CHANNEL) < 1e-9 and track.points[-1]["gripper"] == 17.0
    assert body_gap(track.points[0], out[-1][1]) < 1e-9


class FakeWriter:
    def __init__(self, clock):
        self.clock_fn, self.origin, self.rows, self.count = clock, clock(), [], 0

    def clock(self):
        return self.clock_fn() - self.origin

    def add_control(self, row):
        self.rows.append(row)
        self.count += 1


class GraspRoundCase(RouteCase):
    def test_park_start_round_records_only_the_demo_and_places(self):
        from agentic_sim.real_grasp import grasp_round
        from agentic_sim.real_route_fake import FakeDrivenLeader

        self.teach()
        data, _ = load_route(self.route)
        channel = data["channel"]["command"]
        drop = {"tray": "pink", "waypoints": [channel, dict(channel) | {"shoulder_pan": channel["shoulder_pan"] - 15}],
                "release_gripper": 35.0}
        grasp = dict(CHANNEL) | {"shoulder_lift": CHANNEL["shoulder_lift"] + 20, "elbow_flex": CHANNEL["elbow_flex"] - 10}
        moves = [(0.5, dict.fromkeys(PARK, 0.0)), (3.0, {j: VIA[j] - PARK[j] for j in PARK}),
                 (6.0, {j: grasp[j] - PARK[j] for j in PARK})]
        clock = FakeClock()
        follower = FakeFollower(clock)
        leader = FakeDrivenLeader(clock, PARK, moves)
        term = ScriptedTerminal(clock, [(16.0, "c")], ("a", "p", "r", "y", "s"))
        driver = self.driver(follower, term, clock, "grasp")
        writer = FakeWriter(clock)
        checks = []
        result = grasp_round(driver, leader, data, {"pink": drop}, writer, lambda: checks.append(1), "red")
        driver.close_logs()
        self.assertEqual(result["destination"], "pink")
        self.assertTrue(result["success"] and result["placed"])
        self.assert_released_at_park(follower)
        self.assertEqual(result["demo_frames"], writer.count)          # rows stop at c
        self.assertGreater(writer.count, 100)
        self.assertTrue(checks)
        first = writer.rows[0]["arms"]["right"]["readback"]
        self.assertLess(body_gap(first, PARK), 4.0)                    # the demo starts at park
        # Placement went out along the drop route (pan turned) after the demo.
        self.assertTrue(any(cmd["shoulder_pan"] < channel["shoulder_pan"] - 10 for _, cmd in follower.sent))


def test_retrace_to_stops_where_the_demo_passed_the_start_pose():
    from agentic_sim.real_grasp import retrace_to

    path = [dict(PARK)] + [dict(CHANNEL) | {"shoulder_lift": CHANNEL["shoulder_lift"] + 3 * i} for i in range(10)]
    samples = list(enumerate(path))
    track = retrace_to(samples, CHANNEL)
    self_lifts = [p["shoulder_lift"] for p in track.points]
    assert all(p["shoulder_lift"] >= CHANNEL["shoulder_lift"] - 1e-9 for p in track.points)  # never back toward park
    assert self_lifts == sorted(self_lifts, reverse=True)


class PickPlaceRoundCase(RouteCase):
    def test_demonstrated_place_ends_near_start_then_parks(self):
        from agentic_sim.real_grasp import pick_place_round
        from agentic_sim.real_route_fake import FakeDrivenLeader

        self.teach()
        data, _ = load_route(self.route)
        grasp = dict(CHANNEL) | {"shoulder_lift": CHANNEL["shoulder_lift"] + 20, "elbow_flex": CHANNEL["elbow_flex"] - 10}
        place = dict(grasp) | {"shoulder_pan": grasp["shoulder_pan"] + 40}
        d = lambda pose_: {j: pose_[j] - PARK[j] for j in PARK}  # noqa: E731
        moves = [(0.5, dict.fromkeys(PARK, 0.0)), (3.0, d(VIA)), (6.0, d(grasp)), (9.0, d(place)),
                 (12.0, d(CHANNEL) | {"shoulder_pan": 4.0})]
        clock = FakeClock()
        follower = FakeFollower(clock)
        leader = FakeDrivenLeader(clock, PARK, moves)
        term = ScriptedTerminal(clock, [(22.0, "c")], ("a", "y", "s"))
        driver = self.driver(follower, term, clock, "pick-place")
        writer = FakeWriter(clock)
        result = pick_place_round(driver, leader, data, writer, lambda: None, color="red", destination="beige",
                                  slot={"index": 1, "xy": [70, 60]}, quads=None)
        driver.close_logs()
        self.assert_released_at_park(follower)
        self.assertTrue(result["success"]) and self.assertEqual(result["demo_frames"], writer.count)
        last = writer.rows[-1]["arms"]["right"]["readback"]
        self.assertLess(body_gap(last, CHANNEL), 30)          # the demonstration ended near the start pose
        self.assertTrue(any(cmd["shoulder_pan"] > grasp["shoulder_pan"] + 30 for _, cmd in follower.sent))

    def test_wrist_turned_at_the_end_still_goes_straight_and_is_restored_before_the_route(self):
        from agentic_sim.real_grasp import pick_place_round
        from agentic_sim.real_route_fake import FakeDrivenLeader

        self.teach()
        data, _ = load_route(self.route)
        grasp = dict(CHANNEL) | {"shoulder_lift": CHANNEL["shoulder_lift"] + 20, "wrist_roll": CHANNEL["wrist_roll"] + 30}
        d = lambda pose_: {j: pose_[j] - PARK[j] for j in PARK}  # noqa: E731
        moves = [(0.5, dict.fromkeys(PARK, 0.0)), (3.0, d(VIA)), (6.0, d(grasp)),
                 (9.0, d(CHANNEL) | {"wrist_roll": CHANNEL["wrist_roll"] + 35 - PARK["wrist_roll"]})]
        clock = FakeClock()
        follower = FakeFollower(clock)
        term = ScriptedTerminal(clock, [(18.0, "c")], ("a", "y", "s"))   # no "far from the start pose" menu
        driver = self.driver(follower, term, clock, "pick-wrist")
        result = pick_place_round(driver, FakeDrivenLeader(clock, PARK, moves), data, FakeWriter(clock), lambda: None,
                                  color="red", destination="beige", slot={"index": 1, "xy": [70, 30]}, quads=None)
        driver.close_logs()
        self.assert_released_at_park(follower)
        self.assertTrue(result["success"])
        sent = [cmd for _, cmd in follower.sent]
        # the demo turned the wrist past the start-pose angle (the fake route itself only turns it the other way)
        turned = max(i for i, cmd in enumerate(sent) if cmd["wrist_roll"] - CHANNEL["wrist_roll"] > 20)
        at_start = next(i for i in range(turned, len(sent)) if body_gap(sent[i], CHANNEL) < 1.0)
        self.assertLess(turned, at_start)                     # the wrist is back before the route to park starts

    def test_program_place_round_hands_over_at_the_start_pose_and_parks_the_leader(self):
        from agentic_sim.real_grasp import pick_place_round
        from agentic_sim.real_route import follow
        from agentic_sim.real_route_fake import FakeDrivenLeader

        self.teach()
        data, _ = load_route(self.route)
        lift = dict(CHANNEL) | {"shoulder_lift": CHANNEL["shoulder_lift"] + 15, "elbow_flex": CHANNEL["elbow_flex"] - 8,
                                "gripper": 17.0}
        d = lambda pose_: {j: pose_[j] - CHANNEL[j] for j in CHANNEL}  # noqa: E731  (after the hand-over at the start)
        moves = [(0.5, dict.fromkeys(CHANNEL, 0.0)), (4.0, d(lift))]
        clock = FakeClock()
        follower = FakeFollower(clock)
        rest = dict(PARK) | {"shoulder_pan": PARK["shoulder_pan"] + 2.0, "gripper": 1.0}   # the leader's own fold
        leader = FakeDrivenLeader(clock, rest, moves)
        term = ScriptedTerminal(clock, [(30.0, "c")], ("a", "y", "s"))
        driver = self.driver(follower, term, clock, "program-place")
        placed = []

        class StubPlacer:
            def plan_place(self, xy, yaw, held, grip, current):
                import time as _time
                _time.sleep(0.05)                    # IK wall time: the follower keeps sending meanwhile
                return [dict(current) | {"shoulder_pan": 30.0}], dict(current) | {"shoulder_pan": 30.0}, \
                    dict(current) | {"shoulder_pan": 30.0, "gripper": grip}

            def execute_place(self, plan, label):
                placed.append(label)
                path, over, low = plan
                follow(driver, Track([dict(driver.target), *path, low, low | {"gripper": 40.0}, over | {"gripper": 40.0}],
                                     [0.0] * 4), ("方塊上方", label))

        writer = FakeWriter(clock)
        result = pick_place_round(driver, leader, data, writer, lambda: None, color="red", destination="beige",
                                  slot={"index": 2, "xy": [50, 132]}, quads=None,
                                  placer=StubPlacer(), slot_mount=((0.21, -0.12), 0.0))
        driver.close_logs()
        self.assert_released_at_park(follower)
        self.assertTrue(result["success"])
        self.assertEqual(len(placed), 1)                                   # the program placed the held block
        self.assertTrue(result["program_place"]["held"])
        first = writer.rows[0]["arms"]["right"]["readback"]
        self.assertLess(body_gap(first, CHANNEL), 5.0)                     # the demo starts at the start pose
        sources = [r["command_source"] for r in writer.rows]
        k = sources.index("script")
        self.assertEqual(result["demo_frames"], k)                         # people up to c ...
        self.assertTrue(all(src == "manual" for src in sources[:k]) and all(src == "script" for src in sources[k:]))
        self.assertLess(body_gap(writer.rows[k - 1]["arms"]["right"]["readback"], lift), 6.0)   # ... c at the lift
        stamps = [r["timestamp_ns"] for r in writer.rows]
        self.assertLess(max(b - a for a, b in zip(stamps, stamps[1:])), 1.5e9 / 60)          # no gap at the switch
        self.assertEqual(writer.rows[-1]["arms"]["right"]["command"]["gripper"], 40.0)       # ... through the release
        kinds = [k for k, _ in leader.log]
        self.assertEqual(kinds, ["on", "off", "on", "off"])                # hand-over, then held at c and parked
        self.assertLess(body_gap(leader.goal, rest), 1e-6)                 # driven home to its own fold

    def test_a_camera_stall_mid_demonstration_stops_the_episode_and_brings_the_arm_home(self):
        """11/27 P0015: the wrist camera stalled ~1 s while the arm reached out and the tool exited (emergency
        release by hand). Now the recording stops, the follower retraces to the start pose and parks."""
        from agentic_sim.real_grasp import pick_place_round
        from agentic_sim.real_route_fake import FakeDrivenLeader

        self.teach()
        data, _ = load_route(self.route)
        reach = dict(CHANNEL) | {"shoulder_lift": CHANNEL["shoulder_lift"] + 15, "elbow_flex": CHANNEL["elbow_flex"] - 8}
        d = lambda pose_: {j: pose_[j] - CHANNEL[j] for j in CHANNEL}  # noqa: E731
        clock = FakeClock()
        follower = FakeFollower(clock)
        leader = FakeDrivenLeader(clock, dict(PARK), [(0.5, dict.fromkeys(CHANNEL, 0.0)), (4.0, d(reach))])
        term = ScriptedTerminal(clock, [], ("a", "d", "r", "y", "f"))
        driver = self.driver(follower, term, clock, "camera-stall")
        began = clock()

        def check_cameras():
            if clock() - began > 30e9:
                raise RuntimeError("wrist: no recent camera frame")

        writer = FakeWriter(clock)
        result = pick_place_round(driver, leader, data, writer, check_cameras, color="red", destination="beige",
                                  slot={"index": 1, "xy": [50, 190]}, quads=None,
                                  placer=object(), slot_mount=((0.21, -0.07), 0.0))
        driver.close_logs()
        self.assert_released_at_park(follower)
        self.assertTrue(result["cancelled"])
        self.assertFalse(result["success"])
        self.assertTrue(writer.rows)                                       # rows up to the stall, none after

    def test_stay_rounds_wait_at_the_start_pose_and_park_both_arms_at_the_end(self):
        """11/27: between rounds the follower holds at the start pose and the leader is driven from c back to the
        start pose and waits there torque-on; the next round hands over there; park_all takes both home."""
        from agentic_sim.real_grasp import park_all, pick_place_round
        from agentic_sim.real_route import follow
        from agentic_sim.real_route_fake import FakeDrivenLeader

        self.teach()
        data, _ = load_route(self.route)
        lift = dict(CHANNEL) | {"shoulder_lift": CHANNEL["shoulder_lift"] + 15, "elbow_flex": CHANNEL["elbow_flex"] - 8,
                                "gripper": 17.0}
        d = {j: lift[j] - CHANNEL[j] for j in CHANNEL}
        clock = FakeClock()
        follower = FakeFollower(clock)
        rest = dict(PARK) | {"shoulder_pan": PARK["shoulder_pan"] + 2.0, "gripper": 1.0}
        leader = FakeDrivenLeader(clock, rest, [(0.5, dict.fromkeys(CHANNEL, 0.0)), (4.0, d)])
        term = ScriptedTerminal(clock, [(30.0, "c"), (75.0, "c")], ("a", "s", "a", "s", "y"))
        driver = self.driver(follower, term, clock, "stay")

        class StubPlacer:
            def plan_place(self, xy, yaw, held, grip, current):
                return [dict(current) | {"shoulder_pan": 30.0}], dict(current) | {"shoulder_pan": 30.0}, \
                    dict(current) | {"shoulder_pan": 30.0, "gripper": grip}

            def execute_place(self, plan, label):
                path, over, low = plan
                follow(driver, Track([dict(driver.target), *path, low, low | {"gripper": 40.0}, over | {"gripper": 40.0}],
                                     [0.0] * 4), ("方塊上方", label))

        def one_round():
            writer = FakeWriter(clock)
            return pick_place_round(driver, leader, data, writer, lambda: None, color="red", destination="beige",
                                    slot={"index": 1, "xy": [50, 190]}, quads=None, placer=StubPlacer(),
                                    slot_mount=((0.21, -0.07), 0.0), stay=True), writer

        r1, _ = one_round()
        self.assertTrue(r1["success"])
        self.assertEqual(follower.torque_states(), ON)                       # holding at the start pose ...
        self.assertLess(body_gap(follower.state, CHANNEL), 3.0)
        self.assertTrue(driver.at_start and driver.leader_waiting)           # ... the leader waits there
        self.assertLess(body_gap(leader.goal, CHANNEL), 1e-6)
        self.assertTrue(leader.driven)
        r2, w2 = one_round()                                                   # no arm_at: torque already on
        self.assertTrue(r2["success"])
        first = w2.rows[0]["arms"]["right"]["readback"]
        self.assertLess(body_gap(first, CHANNEL), 5.0)                       # the second demo starts at the start pose
        if self._testMethodName.endswith("_after_an_error"):
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
            from real_route import safe_park
            safe_park(driver, leader, data)          # what grasp_session does when anything stops it
        else:
            park_all(driver, leader, data)
        driver.close_logs()
        self.assert_released_at_park(follower)
        self.assertFalse(leader.driven)
        self.assertLess(body_gap(leader.goal, rest), 1e-6)                  # home to its own fold
        self.assertEqual([k for k, _ in leader.log], ["on", "off", "on", "on", "off", "on", "off"])

    def test_stay_rounds_wait_at_the_start_pose_and_park_both_arms_after_an_error(self):
        """An error between rounds: both arms go home by their fixed ways before any torque is released."""
        self.test_stay_rounds_wait_at_the_start_pose_and_park_both_arms_at_the_end()

    def test_plush_round_takes_the_depth_target_and_places_from_its_grasp_height(self):
        """11/28: the plush toy: target from the opening depth (find_plush), the program releases it at P0 from the
        height it was taken at, the landing comes from depth too."""
        from unittest import mock

        import numpy as np

        from agentic_sim.real_grasp import pick_place_round
        from agentic_sim.real_route import follow
        from agentic_sim.real_route_fake import FakeDrivenLeader

        self.teach()
        data, _ = load_route(self.route)
        lift = dict(CHANNEL) | {"shoulder_lift": CHANNEL["shoulder_lift"] + 15, "gripper": 17.0}
        clock = FakeClock()
        follower = FakeFollower(clock)
        leader = FakeDrivenLeader(clock, dict(PARK) | {"gripper": 1.0},
                                  [(0.5, dict.fromkeys(CHANNEL, 0.0)), (4.0, {j: lift[j] - CHANNEL[j] for j in CHANNEL})])
        term = ScriptedTerminal(clock, [(30.0, "c")], ("a", "s", "y"))
        driver = self.driver(follower, term, clock, "plush")
        asked = {}

        class StubPlacer:
            corr_at = staticmethod(lambda xyz: np.zeros(3))
            model = type("M", (), {"ee": staticmethod(lambda p: np.array([0.3, -0.1, 0.05 + 0.0 * p["shoulder_lift"]]))})

            def plan_place(self, xy, yaw, held, grip, current, **heights):
                asked.update(heights, xy=xy)
                return [dict(current)], dict(current), dict(current) | {"gripper": grip}

            def execute_place(self, plan, label):
                path, over, low = plan
                follow(driver, Track([dict(driver.target), low | {"gripper": 40.0}, over | {"gripper": 40.0}],
                                     [0.0] * 2), ("玩偶上方", label))

        def plush(x, y, tray):
            return {"x": x, "y": y, "z_top": 0.1, "yaw": 0.5, "head": [x, y], "tray": tray, "x_mm": x * 1000,
                    "y_mm": y * 1000, "yaw_deg": 28.6, "pixel": [300.0, 200.0], "height_mm": 98.0,
                    "size_mm": [88.0, 60.0], "upright": True}
        seen = [[plush(0.30, -0.10, None)], [plush(0.215, -0.20, "beige")]]
        with mock.patch("agentic_sim.real_grasp.find_plush", side_effect=lambda *a: seen.pop(0)):
            result = pick_place_round(driver, leader, data, FakeWriter(clock), lambda: None, color=None,
                                      destination="beige", slot={"index": 0, "xy": [55, 58]}, quads=None,
                                      placer=StubPlacer(), slot_mount=((0.215, -0.197), 0.0), stay=True,
                                      plush_cam=object())
        driver.close_logs()
        self.assertTrue(result["success"])
        self.assertEqual(result["color"], "plush")
        self.assertEqual(result["target_choice"]["x_mm"], 300.0)
        self.assertAlmostEqual(asked["place_z"], 0.05 + 0.015)          # grasp height + floor + release gap
        self.assertAlmostEqual(asked["carry_z"], 0.05 + 0.083)          # bottom 6 cm over the rims
        self.assertLess(result["landed"]["slot_offset_mm"], 5.0)

    def test_round_chooser_sets_the_colour_and_cancels_on_an_empty_tray(self):
        from agentic_sim.real_grasp import pick_place_round
        from agentic_sim.real_route_fake import FakeDrivenLeader

        self.teach()
        data, _ = load_route(self.route)
        clock = FakeClock()
        follower = FakeFollower(clock)
        term = ScriptedTerminal(clock, [], ())
        driver = self.driver(follower, term, clock, "pick-empty")
        result = pick_place_round(driver, FakeDrivenLeader(clock, PARK, []), data, FakeWriter(clock), lambda: None,
                                  color=None, destination="beige", slot={"index": 1, "xy": [70, 30]}, quads=None,
                                  choose=lambda blocks: (None, {"source": "pink", "candidates": 0}))
        driver.close_logs()
        self.assertIsNone(result["demo_frames"])
        self.assertFalse(follower.sent)                       # nothing moved: no block to pick


def _block(color, x, y, tray="pink"):
    return {"color": color, "tray": tray, "x_mm": x, "y_mm": y, "yaw_deg": 0.0}


def test_random_pick_only_takes_source_tray_blocks_and_counts_the_scene():
    import random

    from agentic_sim.real_grasp import random_pick

    blocks = [_block("blue", 40, 50), _block("blue", 60, 120), _block("red", 80, 180), _block("yellow", 70, 30, "beige")]
    choose = random_pick(random.Random(0), "pink")
    seen = set()
    for _ in range(50):
        target, scene = choose(blocks)
        assert target["tray"] == "pink"
        seen.add(id(target))
    assert len(seen) == 3                                     # every pink block gets picked at some point
    assert scene["counts"] == {"blue": 2, "red": 1} and scene["candidates"] == 3 and scene["elsewhere"] == 1
    assert choose([_block("red", 70, 30, "beige")])[0] is None


def test_describe_names_the_side_among_same_colour_blocks():
    from agentic_sim.real_grasp import describe

    left, right, red = _block("blue", 60, 150), _block("blue", 65, 60), _block("red", 50, 100)
    blocks = [left, right, red]
    assert "靠左" in describe(left, blocks) and "靠右" in describe(right, blocks)
    assert "同色" not in describe(red, blocks)
    near, far = _block("blue", 40, 100), _block("blue", 110, 105)
    assert "較近" in describe(near, [near, far]) and "較遠" in describe(far, [near, far])


def test_round_state_resumes_from_the_last_round_row(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("real_route_tool", Path(__file__).resolve().parents[1] / "tools/real_route.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    index = tmp_path / "index.jsonl"
    assert tool.round_state(index) == (0, 0, 0)
    rows = [{"episode": "P0001"}, {"episode": "P0002", "round": {"n": 1, "placed_after": 1, "left_after": 3}},
            {"episode": "P0003", "round": {"n": 1, "placed_after": 2, "left_after": 2}}]
    index.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert tool.round_state(index) == (1, 2, 2)


def test_placed_block_is_the_new_one_and_reports_crowding():
    from agentic_sim.real_grasp import placed_block

    before = [_block("blue", 70, 30, "beige"), _block("red", 70, 95, "beige"), _block("blue", 60, 120)]
    after = [_block("blue", 72, 32, "beige"), _block("red", 70, 95, "beige"), _block("blue", 75, 130, "beige")]
    landed = placed_block(before, after, "blue", "beige", {"index": 3, "xy": [70, 160]})
    assert (landed["x_mm"], landed["y_mm"]) == (75, 130)          # not the blue already there
    assert landed["slot_offset_mm"] == 30.4 and landed["neighbour_mm"] == 35.4
    assert placed_block(before, after, "yellow", "beige", {"index": 3, "xy": [70, 160]}) is None


def test_set_aside_moves_the_episode_out_of_the_session(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("real_route_tool2", Path(__file__).resolve().parents[1] / "tools/real_route.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    session = tmp_path / "261126-park"
    ep = session / "episodes" / "P0003"
    ep.mkdir(parents=True)
    (ep / "manifest.json").write_text("{}")
    moved = tool.set_aside(ep, session, "failed", {"round": {"n": 1}})
    assert not ep.exists() and (moved / "manifest.json").exists()
    assert moved.parent == tmp_path / "261126-park-rejected" and moved.name.startswith("P0003-")
    row = json.loads((moved.parent / "index.jsonl").read_text())
    assert row["why"] == "failed" and row["moved_to"] == moved.name and row["round"] == {"n": 1}
    assert not list((session / "episodes").glob("P*"))         # the next episode reuses the number


def test_program_moves_widen_the_lag_gate_with_speed_but_still_stop_on_an_obstacle(tmp_path):
    """11/27: the programmed place trailed its command by ~4 deg (servo lag + sag) and played at 13.6 deg/s under a
    45 cap. With the speed-proportional allowance a lagging servo runs near the cap; a blocked one still pauses."""
    import pytest

    from agentic_sim.real_route import LAG_PER_SPEED, Driver, Paused, Track
    from agentic_sim.real_route_fake import CHANNEL, FakeClock, FakeFollower, ScriptedTerminal

    def run(allowance, block=None):
        clock = FakeClock()
        follower = FakeFollower(clock, start=CHANNEL, torque=True, tau=0.1, block=block,
                                sag=dict.fromkeys(CHANNEL, 0.0) | {"shoulder_lift": -2.5, "elbow_flex": 2.5})
        driver = Driver(follower, ScriptedTerminal(clock), tmp_path / f"r{allowance}{block is None}", hz=60,
                        clock=clock, wait=clock.sleep, drive_leader=False)
        driver.target = dict(CHANNEL)
        driver.play_speed, driver.lag_per_speed = 75.0, allowance
        goal = dict(CHANNEL) | {"shoulder_pan": CHANNEL["shoulder_pan"] + 60, "elbow_flex": CHANNEL["elbow_flex"] - 40}
        began = clock()
        driver.play(Track([dict(CHANNEL), goal], [0.0]), "test")
        return (clock() - began) / 1e9

    slow, fast = run(0.0), run(LAG_PER_SPEED)
    assert fast < 1.3 < slow                         # 60 deg at 75 deg/s = 0.8 s
    wall = lambda state, now: state | {"shoulder_pan": min(state["shoulder_pan"], CHANNEL["shoulder_pan"] + 15)}
    with pytest.raises(Paused):
        run(LAG_PER_SPEED, block=wall)


def test_a_program_move_from_a_standing_sag_starts_at_once(tmp_path):
    """11/27 S0001: standing at the grasp with ~5 deg of sag the lift crept at PUSH_RATE for 1.2 s."""
    from agentic_sim.real_route import LAG_PER_SPEED, Driver, Track
    from agentic_sim.real_route_fake import CHANNEL, FakeClock, FakeFollower, ScriptedTerminal

    clock = FakeClock()
    follower = FakeFollower(clock, start=CHANNEL, torque=True, tau=0.08,
                            sag=dict.fromkeys(CHANNEL, 0.0) | {"shoulder_lift": -5.5, "elbow_flex": 5.0})
    driver = Driver(follower, ScriptedTerminal(clock), tmp_path / "sag", hz=60, clock=clock, wait=clock.sleep,
                    drive_leader=False)
    driver.target = dict(CHANNEL)
    driver.hold("settle")
    for _ in range(60):                                  # stand still: the sag builds up
        driver.send(driver.target)
        driver.tick(clock())
    driver.play_speed, driver.lag_per_speed = 75.0, LAG_PER_SPEED
    goal = dict(CHANNEL) | {"shoulder_lift": CHANNEL["shoulder_lift"] - 10}
    pts = [dict(CHANNEL) | {"shoulder_lift": CHANNEL["shoulder_lift"] - 10 * s} for s in (0.02, 0.1, 0.3, 0.6, 1.0)]
    began = clock()
    driver.play(Track([dict(CHANNEL), *pts], [0.15] * 5), "lift")
    assert (clock() - began) / 1e9 < 1.0                 # 0.75 s planned
    assert goal["shoulder_lift"] == driver.target["shoulder_lift"]
