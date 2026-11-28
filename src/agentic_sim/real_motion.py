"""Common single-follower alignment, measured retreat and torque ownership."""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
from time import monotonic_ns, sleep

from agentic_sim.real_control import JOINTS, pose


OFF = dict.fromkeys(JOINTS, 0)
ON = dict.fromkeys(JOINTS, 1)
AT_TARGET = dict.fromkeys(JOINTS, 0.8) | {"gripper": 2.0}
TRACKING = dict.fromkeys(JOINTS, 5.0) | {"gripper": 10.0}
ROUTE_PERIOD = 0.05
BODY = JOINTS[:-1]
# Lenient rules shared with tools/real_route.py: a lagging follower slows a route instead of stopping it,
# and range edges clip commands. Only a persistent lag (obstruction) faults, after holding in place.
SETTLED = dict.fromkeys(JOINTS, 2.5) | {"gripper": 5.0}   # gravity sag allowance at a settled target
CORRIDOR = dict.fromkeys(JOINTS, 3.0) | {"gripper": 100.0}  # start-of-route match (gripper free)
READ_MARGIN = 3.0       # deg a readback may sit outside the approved limits (e.g. park on its support)
GATE_LAG = 4.0          # route interpolation waits while the body lag exceeds this
PUSH_LAG = 8.0          # between GATE_LAG and this it creeps on at PUSH_RATE: steady gravity sag on stretched poses
PUSH_RATE = 0.3
REACH_SETTLED = dict.fromkeys(JOINTS, 6.0) | {"gripper": 5.0}  # settle allowance at stretched drop poses
STALL_LAG = 15.0        # teleop lag treated as an obstruction ...
STALL_SECONDS = 1.0     # ... when it persists (routes: gated time)
CARRY_SQUEEZE = 3.0  # gripper % below the grasp readback while carrying (= the follower's closing-gap cap)
TASK_STEP = dict.fromkeys(JOINTS, 4.0) | {"gripper": 10.0}  # max change per task command


def near(actual, target, tolerance=AT_TARGET):
    return all(abs(actual[joint] - target[joint]) <= tolerance[joint] for joint in JOINTS)


@dataclass(frozen=True)
class MotionPlan:
    version: str
    layout: str
    arm: str
    calibrations: dict
    park_support: str
    route: tuple
    limits: dict
    speed: dict
    workspace: str
    revision: str
    rgb_depth: dict
    drops: dict = None

    @classmethod
    def load(cls, data):
        for key in ("version", "layout", "arm", "park_support", "workspace"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ValueError(f"motion setup needs {key}")
        calibrations = data["calibrations"]
        if set(calibrations) != {"leader", "follower"} or any(
            not ref.get("id") or not ref.get("version") for ref in calibrations.values()
        ):
            raise ValueError("motion setup needs both calibration references")
        route = (pose(data["park"]), *(pose(point) for point in data.get("via", [])), pose(data["start"]))
        limits = {}
        if set(data["limits"]) != set(JOINTS):
            raise ValueError("motion setup needs six approved workspace limits")
        for joint in JOINTS:
            low, high = data["limits"][joint]
            if not all(type(value) in (int, float) and isfinite(value) for value in (low, high)) or low >= high:
                raise ValueError(f"invalid workspace limits for {joint}")
            if joint == "gripper" and not 0 <= low < high <= 100:
                raise ValueError("invalid gripper workspace limits")
            limits[joint] = (low, high)
        speed = pose(data["speed"])
        for joint in JOINTS:
            if not 0 < speed[joint] <= (30 if joint == "gripper" else 25):
                raise ValueError(f"approved alignment/retreat speed out of range for {joint}")
        revision = sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
        drops = {}
        for tray, drop in (data.get("drops") or {}).items():
            points = tuple(pose(point) for point in drop["waypoints"])
            release = drop["release_gripper"]
            if len(points) < 2 or not near(points[0], route[-1], CORRIDOR):
                raise ValueError(f"drop route {tray} must start at the standard start (channel)")
            if type(release) not in (int, float) or not 0 < release <= 100:
                raise ValueError(f"drop route {tray} needs a release gripper opening")
            drops[tray] = {"waypoints": points, "release_gripper": float(release)}
        plan = cls(data["version"], data["layout"], data["arm"], calibrations, data["park_support"],
                   route, limits, speed, data["workspace"], revision, data.get("rgb_depth", {}), drops)
        for point in (*route, *(p for d in drops.values() for p in d["waypoints"])):
            plan.in_workspace(point)
        return plan

    @property
    def start(self):
        return self.route[-1]

    @property
    def park(self):
        return self.route[0]

    def in_workspace(self, target, margin=0.0):
        for joint, (low, high) in self.limits.items():
            if not low - margin <= target[joint] <= high + margin:
                raise RuntimeError(f"{joint} is outside the approved workspace")

    def clip(self, target):
        return {j: min(high, max(low, target[j])) for j, (low, high) in self.limits.items()}

    def origin(self, current):
        # Interpolation starts at a confirmed taught waypoint, never at an arbitrary pose.
        index = min(range(len(self.route)), key=lambda i: sum(
            ((current[joint] - self.route[i][joint]) / AT_TARGET[joint]) ** 2 for joint in JOINTS
        ))
        if not near(current, self.route[index], CORRIDOR):
            raise RuntimeError("follower is outside the verified start corridor; support and reposition it")
        return index

    def reference(self):
        return {"version": self.version, "revision": self.revision, "workspace": self.workspace}


def load_setup(path):
    setup = json.loads(Path(path).read_text(encoding="utf-8"))
    return MotionPlan.load(setup["motion"]), setup["cards"]


def simplified(points):
    """Drop stationary samples and keep the path within the route tolerance (fewer return steps)."""
    from agentic_sim.real_route import _simplify

    moving = []
    for point in points:
        if not moving or not near(point, moving[-1], dict.fromkeys(JOINTS, .15) | {"gripper": .3}):
            moving.append(point)
    if len(moving) <= 2:
        return moving
    return [moving[i] for i in sorted(_simplify(moving))]


class Motion:
    def __init__(self, device, plan, emit, *, clock=monotonic_ns, wait=sleep, halted=lambda: False):
        self.device = device
        self.plan = plan
        self.emit = emit
        self.clock = clock
        self.wait = wait
        self.halted = halted
        self.phase = "preflight"
        self.last = None
        self.path = []
        self.armed = False
        self.command = None
        self.stall_since = None

    def event(self, kind, **fields):
        self.emit({"event": kind, "phase": self.phase, "timestamp_ns": self.clock(), **fields})

    def ensure_running(self):
        if self.halted():
            raise RuntimeError("operator stopped motion; support the follower and check torque on site")

    def read(self):
        self.ensure_running()
        sample = pose(self.device.read())
        self.plan.in_workspace(sample, READ_MARGIN)
        self.last = sample
        if self.phase == "task":
            self.path.append(self.last)
        return self.last

    def send(self, target):
        self.ensure_running()
        requested = pose(target)
        target = self.plan.clip(requested)
        if self.phase == "task":
            # Rate limit relative to the previous command and watch for an obstruction instead of
            # stopping on the first large leader step or range edge.
            previous = self.command or self.last
            target = {j: min(previous[j] + TASK_STEP[j], max(previous[j] - TASK_STEP[j], target[j])) for j in JOINTS}
            lag = max(abs(self.last[j] - previous[j]) for j in BODY)
            now = self.clock()
            if lag <= STALL_LAG:
                self.stall_since = None
            elif self.stall_since is None:
                self.stall_since = now
            elif now - self.stall_since > STALL_SECONDS * 1e9:
                self._hold(f"task stall lag {lag:.1f}")
                raise RuntimeError(f"follower stalled ({lag:.0f} deg behind); held in place")
        sent = pose(self.device.send(target))
        self.command = sent
        self.event("command", requested=requested, command=sent, previous_readback=self.last)
        return sent

    def _hold(self, reason):
        """Best effort: command the measured pose so the follower stops pushing."""
        try:
            actual = pose(self.device.read())
            self.command = pose(self.device.send(self.plan.clip(actual)))
            self.event("hold", reason=reason, readback=actual)
        except Exception:  # noqa: BLE001 - the caller reports the original fault
            pass

    def _settle(self, target, allowance=SETTLED):
        previous = None
        stable = 0
        for _ in range(80):
            actual = self.read()
            if previous is not None and near(actual, target, allowance) and near(
                actual, previous, dict.fromkeys(JOINTS, 0.15) | {"gripper": 0.3}
            ):
                stable += 1
                if stable == 3:
                    self.event("settled", target=target, readback=actual)
                    return actual
            else:
                stable = 0
            previous = actual
            self.wait(ROUTE_PERIOD)
        raise RuntimeError(f"{self.phase}: follower did not reach and settle at the target")

    def _move(self, target, *, settle=True, allowance=SETTLED):
        target = self.plan.clip(pose(target))
        start = self.command if self.command is not None else self.plan.clip(self.read())
        delta = {joint: target[joint] - start[joint] for joint in JOINTS}
        steps = max(1, ceil(max(abs(delta[joint]) / self.plan.speed[joint] for joint in JOINTS) / ROUTE_PERIOD))
        progress, stalled = 1.0, 0.0
        while progress < steps + 1:
            step = min(steps, int(progress))
            sent = self.send(target if step == steps else
                             {joint: start[joint] + delta[joint] * step / steps for joint in JOINTS})
            self.wait(ROUTE_PERIOD)
            lag = max(abs(self.read()[j] - sent[j]) for j in BODY)
            if lag <= GATE_LAG:
                progress, stalled = step + 1.0, 0.0
            elif lag <= PUSH_LAG:  # steady gravity sag: creep on, the growing command error adds torque
                progress, stalled = progress + PUSH_RATE, 0.0
            else:  # wait for the follower instead of aborting; fault only on a persistent stall
                stalled += ROUTE_PERIOD
                if stalled > STALL_SECONDS:
                    self._hold(f"route stall lag {lag:.1f}")
                    raise RuntimeError(f"{self.phase}: follower stalled ({lag:.0f} deg behind); held in place")
        if settle:
            self._settle(target, allowance)

    def align(self):
        initial = self.read()
        waypoint = self.plan.origin(initial)
        self.event("start_measured", readback=initial, waypoint=waypoint)
        if self.device.torque_states() != OFF:
            raise RuntimeError("follower torque must be confirmed off at entry")
        self.ensure_running()
        self.armed = True  # A partial arm() can already have energized a motor.
        self.device.arm()
        self.event("torque_on", readback=initial)
        self.phase = "align"
        for target in self.plan.route[waypoint + 1:-1]:
            self._move(target, settle=False)
        self._move(self.plan.start)
        if self.device.torque_states() != ON:
            raise RuntimeError("could not confirm all six axes are holding")
        self.phase = "task"
        actual = self.read()
        self.event("task_ready", readback=actual)
        return actual

    def place(self, tray):
        """After a lifted grasp: carry to the standard start, follow the taught drop route with the
        gripper held, release, and reverse back to the standard start (the task path is then done)."""
        drop = self.plan.drops[tray]
        self.phase = "carry"
        held = self.read()
        # Keep squeezing the block while carrying: a goal at the readback opening holds with no force.
        grip = max(0.0, held["gripper"] - CARRY_SQUEEZE)
        self.command = None
        self._move(held)
        # Back to the standard start along the measured task path, reversed (a straight joint move from a
        # stretched grasp swings the arm back over itself).
        for point in reversed(simplified(self.path)):
            self._move(point | {"gripper": grip}, settle=False)
        self._move(self.plan.start | {"gripper": grip}, settle=False)
        self.phase = "place"
        for point in drop["waypoints"][1:]:
            self._move(point | {"gripper": grip}, settle=False)
        self._settle(self.command, REACH_SETTLED)
        release = drop["release_gripper"]
        self._move(self.command | {"gripper": max(grip, release)}, allowance=REACH_SETTLED)
        self.event("released_block", tray=tray, readback=self.last)
        self.phase = "place_return"
        for point in reversed(drop["waypoints"][:-1]):
            self._move(point | {"gripper": max(grip, release)}, settle=False)
        self._settle(self.command)
        self.path = []  # back at the standard start: the task path no longer needs to be retraced
        self.event("placed", tray=tray, readback=self.last)

    def park(self, finish, confirm_support):
        self.phase = "hold"
        if self.device.torque_states() != ON:
            raise RuntimeError("torque state cannot be confirmed before retreat")
        held = self.read()
        # Command the measured pose first, including on a controlled cancellation.
        self.command = None
        self.send(self.plan.clip(held))
        self._settle(self.command)
        if not finish():
            raise RuntimeError("clearance or empty gripper could not be confirmed; follower held for operator")
        self.phase = "return"
        # Follow measured task poses in reverse while keeping the empty gripper open.
        grip = held["gripper"]
        for point in (*reversed(simplified(self.path)), *reversed(self.plan.route)):
            self._move(point | {"gripper": grip}, settle=False)
        self.phase = "park"
        self._move(self.plan.park)
        if not confirm_support(self.plan.park_support):
            raise RuntimeError("park support not confirmed; follower held for operator")
        parked = self._settle(self.plan.park)
        self.event("support_confirmed", support=self.plan.park_support, readback=parked)
        self.ensure_running()
        self.device.close()
        torque = self.device.torque_states()
        if torque != OFF:
            raise RuntimeError("torque off was not verified on every joint")
        self.armed = False
        self.event("torque_off", torque=torque)

    def fault(self, error):
        self.event("operator_required" if self.armed else "preflight_rejected", reason=str(error),
                   last_readback=self.last, torque_unknown=self.armed)
