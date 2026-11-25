"""Park <-> channel route: teach with the leader, reverse replay, pause/obstruction, recovery."""

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
