"""Deterministic devices and a portable two-camera recording fixture."""

from contextlib import contextmanager
from itertools import count
from struct import pack
from zlib import compress, crc32

from agentic_sim.real_control import ControlRouter, JOINTS, open_pairs
from agentic_sim.real_episode import EpisodeWriter, PNG_SIGNATURE


def fake_pose(pan=0, grip=70, lift=0):
    return dict.fromkeys(JOINTS, 0) | {"shoulder_pan": pan, "shoulder_lift": lift, "gripper": grip}


class FakeLeader:
    def __init__(self):
        self.state = fake_pose()

    def read(self):
        return dict(self.state)


class FakeFollower:
    def __init__(self, name, trace):
        self.name = name
        self.trace = trace
        self.state = fake_pose()
        self.armed = False

    def arm(self):
        self.armed = True

    def read(self):
        return dict(self.state)

    def send(self, target):
        if not self.armed:
            raise RuntimeError("fake follower is not armed")
        self.trace["commands"].append({"arm": self.name, "command": dict(target)})
        # A half-step lag makes the last policy target distinguishable from measured position.
        self.state = {joint: (self.state[joint] + target[joint]) / 2 for joint in JOINTS}
        return target


class FakeRig:
    def __init__(self):
        self.leaders = {}
        self.trace = {"opened": [], "closed": [], "commands": []}

    @contextmanager
    def open_leader(self, name, _spec):
        device = self.leaders[name] = FakeLeader()
        self.trace["opened"].append(["leader", name])
        try:
            yield device
        finally:
            self.trace["closed"].append(["leader", name])

    @contextmanager
    def open_follower(self, name, _spec):
        self.trace["opened"].append(["follower", name])
        try:
            yield FakeFollower(name, self.trace)
        finally:
            self.trace["closed"].append(["follower", name])


def _chunk(tag, data):
    return pack(">I", len(data)) + tag + data + pack(">I", crc32(data, crc32(tag)))


def fake_png(frame, camera, width=192, height=128):
    """A visibly synthetic table and a moving square, using no imaging dependency."""
    panel = bytes((193, 165, 165)) if camera == 0 else bytes((154, 180, 185))
    background = bytes((230, 230, 225))
    marker = bytes((221, 50, 48)) if camera == 0 else bytes((45, 135, 215))
    x = 26 + 10 * frame + 4 * camera
    y = 58 - 3 * min(frame, 8)
    rows = []
    for j in range(height):
        row = bytearray(background * width)
        if 22 <= j < height - 17:
            row[12 * 3:(width - 12) * 3] = panel * (width - 24)
        if y <= j < y + 23:
            row[x * 3:(x + 23) * 3] = marker * 23
        rows.append(b"\x00" + row)
    return (PNG_SIGNATURE + _chunk(b"IHDR", pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", compress(b"".join(rows))) + _chunk(b"IEND", b""))


def record_fake(root, arms, provenance):
    rig = FakeRig()
    calibration = {
        name: {role: {"id": f"fake_{name}_{role}", "version": "synthetic-v1"} for role in ("leader", "follower")}
        for name in arms
    }
    camera_names = ("global", "wrist") if len(arms) == 1 else ("global", *(f"{name}_wrist" for name in arms))
    cameras = {
        name: {"width": 192, "height": 128, "fps": 20,
               "calibration": {"id": f"fake_{name}", "version": "synthetic-v1"}}
        for name in camera_names
    }
    writer = EpisodeWriter(root, source="fake", arms=calibration, cameras=cameras, provenance=provenance,
                           hz=20, layout="synthetic-table")
    failure = None
    try:
        with open_pairs(calibration, rig.open_leader, rig.open_follower) as pairs:
            router = ControlRouter(pairs, start="policy")
            chunk = [{name: fake_pose(pan + 2 * i) for i, name in enumerate(arms)} for pan in (2, 4, 80, 85)]
            generation = router.generation
            router.enqueue_policy(chunk, generation=generation)
            for frame in range(12):
                elapsed = frame * 50_000_000
                if frame == 2:
                    router.take_over(elapsed)
                    rig.trace["stale_chunk_accepted"] = router.enqueue_policy(chunk[2:], generation=generation)
                if frame >= 2:
                    for i, name in enumerate(arms):
                        rig.leaders[name].state = fake_pose(
                            pan=25 + (1 if i == 0 else -1) * (frame - 2),
                            grip=58 - 4 * min(frame - 2, 7), lift=2 * max(0, frame - 4),
                        )
                ticks = count(elapsed + 1_000_000, 1_000_000)
                writer.add_control(router.step(lambda: next(ticks)))
                for index, name in enumerate(cameras):
                    writer.add_image(name, fake_png(frame, index), elapsed + (index + 2) * 1_000_000)
    except BaseException as error:
        failure = error
        raise
    finally:
        writer.close(error=failure)
    return rig.trace
