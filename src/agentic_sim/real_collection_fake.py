"""Deterministic adapters exercising production task, replay and torque paths."""

from contextlib import contextmanager
from itertools import combinations

from agentic_sim.real_collection import Collection
from agentic_sim.real_control import JOINTS
from agentic_sim.real_episode import EpisodeWriter, write_json
from agentic_sim.real_fake import FakeLeader, fake_png, fake_pose
from agentic_sim.real_motion import MotionPlan, OFF, ON


def fake_setup():
    plan = {
        "version": "fake-start-v1", "layout": "fake-table", "arm": "right", "workspace": "fake-envelope",
        "calibrations": {role: {"id": f"fake_right_{role}", "version": "synthetic-v1"}
                         for role in ("leader", "follower")},
        "park_support": "fake dock", "park": fake_pose(-6, 70), "via": [fake_pose(-3, 70)],
        "start": fake_pose(0, 70), "limits": dict.fromkeys(JOINTS, [-25, 25]) | {"gripper": [0, 100]},
        "speed": dict.fromkeys(JOINTS, 20), "rgb_depth": {},
        "drops": {"pink": {"waypoints": [fake_pose(0, 70), fake_pose(-4, 70, 2)], "release_gripper": 80},
                  "beige": {"waypoints": [fake_pose(0, 70), fake_pose(4, 70, 2)], "release_gripper": 80}},
    }
    layouts = {
        "seed": 20261124, "reserve_distance_mm": 18,
        "zones": {f"{tray}_{region}": [[20 + 22 * x, start + 24 * y, (x + y) % 4 * 15]
                                      for x in range(5) for y in range(3)]
                  for tray in ("pink", "beige") for region, start in (("near", 12), ("far", 90))},
    }
    return MotionPlan.load(plan), layouts


class FakeCollectionFollower:
    def __init__(self, backend):
        self.backend = backend
        self.state = dict(backend.plan.park)
        self.torque = False

    def read(self):
        return dict(self.state)

    def mark(self, kind, **more):
        self.backend.trace.append({"event": kind, "timestamp_ns": self.backend.clock(), **more})

    def arm(self):
        self.mark("hold_measured_before_torque", command=self.read())
        self.torque = True
        self.mark("enable", torque=self.torque_states())

    def torque_states(self):
        return dict(ON if self.torque else OFF)

    def send(self, target):
        if not self.torque:
            raise RuntimeError("fake follower received a command while torque was off")
        self.mark("send", command=dict(target), readback_before=self.read())
        self.state = dict(target)
        return self.read()

    def close(self):
        if not self.torque or self.state != self.backend.plan.park:
            raise RuntimeError("fake follower torque release was attempted outside the parking pose")
        self.torque = False
        self.mark("disable", readback=self.read())


class FakeCollectionBackend:
    source = "fake"
    arm = "right"
    hz = 20

    def __init__(self, plan, provenance=None):
        self.plan = plan
        self.calibrations = plan.calibrations
        self.provenance = provenance or {}
        self.time = 1_000_000
        self.trace = []

    def clock(self):
        self.time += 30_000
        return self.time

    def wait(self, seconds):
        self.time += round(seconds * 1e9)

    @contextmanager
    def devices(self, *, leader):
        device = FakeCollectionFollower(self)
        device.mark("open", leader=leader)
        try:
            yield FakeLeader() if leader else None, device
        finally:
            device.mark("disconnect", torque=device.torque_states())

    @contextmanager
    def capture(self, root, plan):
        cameras = {name: {"width": 192, "height": 128, "fps": self.hz,
                          "calibration": {"id": f"fake-{name}", "version": "synthetic-v1"}}
                   for name in ("global", "wrist")}
        writer = EpisodeWriter(root, source=self.source, arms={self.arm: self.calibrations}, cameras=cameras,
                               provenance=self.provenance, hz=self.hz, layout=plan.layout)
        origin = self.clock()
        writer.clock = lambda: self.clock() - origin

        def images(row):
            for i, name in enumerate(cameras):
                writer.add_image(name, fake_png((writer.count - 1) % 12, i), row["timestamp_ns"] + i * 100_000)

        error = None
        try:
            yield writer, lambda: None, images
        except BaseException as failure:
            error = failure
            raise
        finally:
            writer.close(error=error)


class FakeOperator:
    def __init__(self, color="red", *, dagger=False, replay=False, cancel_at=None, support=True, halt_at=None):
        self.color = color
        self.dagger = dagger
        self.replay = replay
        self.cancel_at = cancel_at
        self.safe_support = support
        self.halt_at = halt_at
        self.frame = -1

    def status(self, state):
        pass

    def halted(self):
        return self.halt_at is not None and self.frame >= self.halt_at

    def finished(self, frame):
        return (self.cancel_at is not None and frame >= self.cancel_at) or (not self.replay and frame >= 18)

    def takeover(self, frame):
        return self.dagger and frame == 3

    def tick(self, leader, frame):
        self.frame = frame
        if self.dagger and frame <= 3:
            leader.state = fake_pose(0, 70)
            return
        pan = min(frame, 3) * .6
        lift = max(0, min(frame - 9, 4)) * .9
        grip = 70 - min(max(frame - 6, 0), 5) * 3 + min(max(frame - 13, 0), 5) * 3
        leader.state = fake_pose(pan, grip, lift)

    def finish(self):
        return {"empty": True, "placed": self.cancel_at is None, "color": self.color,
                "cancelled": self.cancel_at is not None}

    def support(self, support):
        return self.safe_support

    def carry(self, tray):
        self.placed_into = tray
        return self.cancel_at is None


def fake_policy(frame):
    if frame != 0:
        return []
    return [{"right": fake_pose(pan, 70)} for pan in (1.0, 2.0, 23.0, 24.0)]


def demo(root, provenance=None):
    """Run the full acceptance sequence inside a test-only session."""
    plan, layout = fake_setup()
    backend = FakeCollectionBackend(plan, provenance)
    session = Collection(root, backend, plan, layout)
    card, pending = session.current()
    if pending or session.state["attempts"]:
        raise ValueError("--demo needs a new fake session; omit it to open an existing session")
    first = session.record(FakeOperator(card["color"]), card["pose"])
    session.approve(first["episode"])
    if session.progress()[0] or session.current()[1]["status"] != "needs_replay":
        raise AssertionError("first demonstration advanced without a replay")
    first_replay = session.replay(first["episode"], FakeOperator(card["color"], replay=True), layout_ready=True)
    repeated = session.replay(first["episode"], FakeOperator(card["color"], replay=True), layout_ready=True)
    if not first_replay["passed"] or not repeated["passed"] or session.progress()[0] != 1:
        raise AssertionError("first replay gate or replay accounting failed")
    card, _ = session.current()
    second = session.record(FakeOperator(card["color"]), card["pose"])
    reopened = Collection(root, backend, plan, layout)
    if reopened.progress()[0] != 2 or reopened.current()[0]["id"] == card["id"]:
        raise AssertionError("automatic acceptance or session resume failed")
    dagger = reopened.dagger(FakeOperator(dagger=True), fake_policy)
    if (not dagger["takeovers"] or dagger["takeovers"][0]["discarded_policy_actions"] != 2
            or any(row.get("command", {}).get("shoulder_pan") in (23, 24) for row in backend.trace)):
        raise AssertionError("a stale policy command reached the fake follower")
    write_json(root / "fake_device_trace.json", backend.trace)
    summary = {"source": "fake", "demonstrations": reopened.progress()[0], "replays": len(reopened.state["replays"]),
               "next_card": reopened.current()[0]["id"], "first": first["episode"], "second": second["episode"],
               "first_replay": first_replay["record"], "dagger": dagger["episode"],
               "takeover": dagger["takeovers"][0]}
    write_json(root / "fake_acceptance.json", summary)
    return summary
