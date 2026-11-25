"""Deterministic stand-ins for the park <-> channel route: lagging follower, scripted leader and keys."""

from math import exp

from agentic_sim.real_control import JOINTS, pose
from agentic_sim.real_motion import OFF, ON
from agentic_sim.real_route import BODY, lerp

# 2026-11-25 right arm readings (leader-teach-05 / channel-only-01), degrees and percent open.
PARK = {"shoulder_pan": -1.67, "shoulder_lift": -109.05, "elbow_flex": 97.23,
        "wrist_flex": 70.55, "wrist_roll": 5.67, "gripper": 1.05}
CHANNEL = {"shoulder_pan": -1.41, "shoulder_lift": -48.40, "elbow_flex": 31.91,
           "wrist_flex": 83.03, "wrist_roll": 80.22, "gripper": 14.11}
# Leave the support first, then roll the wrist away from the folded area.
VIA = {"shoulder_pan": -1.5, "shoulder_lift": -80.0, "elbow_flex": 75.0,
       "wrist_flex": 75.0, "wrist_roll": 5.67, "gripper": 5.0}
LIMITS = dict.fromkeys(BODY, (-110.0, 110.0)) | {"elbow_flex": (-97.0, 97.0), "gripper": (0.0, 100.0)}


class FakeClock:
    def __init__(self):
        self.now = 1_000_000_000

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0, round(seconds * 1e9))


class FakeFollower:
    """First-order servo with gravity sag; `block(state)` can clamp the state (an obstacle)."""

    def __init__(self, clock, start=PARK, *, tau=0.06, sag=None, block=None, torque=False):
        self.clock = clock
        self.limits = dict(LIMITS)
        self.state = pose(start)
        self.goal = dict(self.state) if torque else None
        self.torque = torque
        self.sag = sag or dict.fromkeys(JOINTS, 0.0) | {"shoulder_lift": -0.6, "elbow_flex": 0.7}
        self.block = block
        self.tau = tau
        self.t = clock()
        self.sent = []

    def _advance(self):
        dt = (self.clock() - self.t) / 1e9
        self.t = self.clock()
        if self.torque and self.goal is not None and dt > 0:
            k = 1 - exp(-dt / self.tau)
            for j in JOINTS:
                aim = self.goal[j] + (self.sag[j] if j in BODY else 0.0)
                self.state[j] += (aim - self.state[j]) * k
            self.state["gripper"] = min(100.0, max(0.0, self.state["gripper"]))
        if self.block:
            self.state = self.block(self.state, self.clock())

    def read(self):
        self._advance()
        return dict(self.state)

    def send(self, target):
        if not self.torque:
            raise RuntimeError("fake follower got a command with torque off")
        self._advance()
        target = pose(target)
        for j in BODY:
            low, high = self.limits[j]
            if not low - 1e-9 <= target[j] <= high + 1e-9:
                raise RuntimeError(f"fake follower: {j} command outside calibrated range")
        self.goal = target
        self.sent.append((self.clock(), dict(target)))
        return dict(target)

    def arm(self):
        if self.torque:
            raise RuntimeError("fake follower already armed")
        self._advance()
        self.torque = True
        self.goal = {j: min(self.limits[j][1], max(self.limits[j][0], v)) for j, v in self.state.items()}

    def resume(self):
        if not self.torque:
            raise RuntimeError("resume requires torque on")
        self.goal = self.read()

    def close(self):
        self._advance()
        self.torque = False
        self.goal = None

    def torque_states(self):
        return dict(ON if self.torque else OFF)


class FakeLeader:
    """Leader pose as a function of time: list of (seconds since creation, pose) keyframes."""

    def __init__(self, clock, keyframes):
        self.clock = clock
        self.origin = clock()
        self.keyframes = [(t, pose(p)) for t, p in keyframes]

    def read(self):
        t = (self.clock() - self.origin) / 1e9
        frames = self.keyframes
        if t <= frames[0][0]:
            return dict(frames[0][1])
        for (t0, a), (t1, b) in zip(frames, frames[1:]):
            if t <= t1:
                return lerp(a, b, (t - t0) / (t1 - t0))
        return dict(frames[-1][1])


class FakeDrivenLeader:
    """Leader with torque: servoes to its goal while driven; after release it sags, then follows the operator's
    displacement keyframes (seconds after release, joint delta)."""

    def __init__(self, clock, start, moves=(), *, sag=None, tau=0.08):
        self.clock = clock
        self.state = pose(start)
        self.goal = None
        self.driven = False
        self.released_at = None
        self.base = None
        self.moves = [(t, pose(d)) for t, d in moves]
        self.sag = sag if sag is not None else {"shoulder_lift": -4.0, "elbow_flex": 3.0}
        self.tau = tau
        self.t = clock()
        self.log = []

    def _delta(self, s):
        zero = dict.fromkeys(JOINTS, 0.0)
        frames = [(0.0, zero), *self.moves]
        for (t0, a), (t1, b) in zip(frames, frames[1:]):
            if s <= t1:
                return lerp(a, b, max(0.0, (s - t0) / (t1 - t0)))
        return dict(frames[-1][1])

    def _advance(self):
        dt = (self.clock() - self.t) / 1e9
        self.t = self.clock()
        if self.driven and dt > 0:
            k = 1 - exp(-dt / self.tau)
            for j in JOINTS:
                self.state[j] += (self.goal[j] - self.state[j]) * k
        elif self.released_at is not None:
            s = (self.clock() - self.released_at) / 1e9
            delta = self._delta(s)
            self.state = {j: self.base[j] + self.sag.get(j, 0.0) * min(1.0, s / 0.3) + delta[j] for j in JOINTS}

    def read(self):
        self._advance()
        return dict(self.state)

    def drive_on(self):
        self.goal = self.read()
        self.driven = True
        self.log.append(("on", self.clock()))

    def drive(self, target):
        if not self.driven:
            raise RuntimeError("fake leader driven with torque off")
        self._advance()
        self.goal = pose(target)

    def release(self):
        self._advance()
        self.driven = False
        self.released_at = self.clock()
        self.base = dict(self.state)
        self.log.append(("off", self.clock()))


def operator_path(start=PARK, offset=3.0):
    """Leader near park, then out of the support, then to the channel."""
    near = {j: v + (offset if j in BODY else 0.0) for j, v in start.items()}
    return [(0.0, near), (1.5, start), (3.5, VIA), (6.0, CHANNEL)]


class ScriptedTerminal:
    """Timed keys for motion loops and an answer queue for menus."""

    def __init__(self, clock, keys=(), answers=()):
        self.clock = clock
        self.keys = sorted(keys)
        self.answers = list(answers)
        self.lines = []
        self.prompts = []

    def key(self, wait=0.0):
        if self.keys and self.keys[0][0] * 1e9 <= self.clock() - 1_000_000_000:
            return self.keys.pop(0)[1]
        if wait:
            self.clock.sleep(wait)
        return None

    def say(self, text):
        self.lines.append(text)

    def status(self, text):
        self.last_status = text

    def choose(self, prompt, options):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"no scripted answer for: {prompt}")
        answer = self.answers.pop(0)
        if answer not in options:
            raise AssertionError(f"{answer!r} not offered in: {prompt}")
        return answer

    def word(self, prompt, expected):
        self.prompts.append(prompt)
        return self.answers.pop(0) == expected
