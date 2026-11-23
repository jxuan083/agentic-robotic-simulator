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
        plan = cls(data["version"], data["layout"], data["arm"], calibrations, data["park_support"],
                   route, limits, speed, data["workspace"], revision, data.get("rgb_depth", {}))
        for point in route:
            plan.in_workspace(point)
        return plan

    @property
    def start(self):
        return self.route[-1]

    @property
    def park(self):
        return self.route[0]

    def in_workspace(self, target):
        for joint, (low, high) in self.limits.items():
            if not low <= target[joint] <= high:
                raise RuntimeError(f"{joint} is outside the approved workspace")

    def origin(self, current):
        # Interpolation starts at a confirmed taught waypoint, never at an arbitrary pose.
        index = min(range(len(self.route)), key=lambda i: sum(
            ((current[joint] - self.route[i][joint]) / AT_TARGET[joint]) ** 2 for joint in JOINTS
        ))
        if not near(current, self.route[index]):
            raise RuntimeError("follower is outside the verified start corridor; support and reposition it")
        return index

    def reference(self):
        return {"version": self.version, "revision": self.revision, "workspace": self.workspace}


def load_setup(path):
    setup = json.loads(Path(path).read_text(encoding="utf-8"))
    return MotionPlan.load(setup["motion"]), setup["cards"]


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

    def event(self, kind, **fields):
        self.emit({"event": kind, "phase": self.phase, "timestamp_ns": self.clock(), **fields})

    def ensure_running(self):
        if self.halted():
            raise RuntimeError("operator stopped motion; support the follower and check torque on site")

    def read(self):
        self.ensure_running()
        sample = pose(self.device.read())
        self.plan.in_workspace(sample)
        self.last = sample
        if self.phase == "task":
            self.path.append(self.last)
        return self.last

    def send(self, target):
        self.ensure_running()
        target = pose(target)
        self.plan.in_workspace(target)
        if self.phase == "task" and not near(self.last, target, TRACKING):
            raise RuntimeError("task target exceeds the tracking envelope")
        sent = pose(self.device.send(target))
        self.event("command", requested=target, command=sent, previous_readback=self.last)
        return sent

    def _settle(self, target):
        previous = None
        stable = 0
        for _ in range(40):
            actual = self.read()
            if previous is not None and near(actual, target) and near(
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

    def _move(self, target, *, settle=True):
        start = self.read()
        delta = {joint: target[joint] - start[joint] for joint in JOINTS}
        steps = ceil(max(abs(delta[joint]) / self.plan.speed[joint] for joint in JOINTS) / ROUTE_PERIOD)
        for step in range(1, steps + 1):
            sent = self.send({joint: start[joint] + delta[joint] * step / steps for joint in JOINTS})
            self.wait(ROUTE_PERIOD)
            if not near(self.read(), sent, TRACKING):
                raise RuntimeError(f"{self.phase}: follower tracking error")
        if settle:
            self._settle(target)

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
        for target in self.plan.route[waypoint:]:
            self._move(target)
        if self.device.torque_states() != ON:
            raise RuntimeError("could not confirm all six axes are holding")
        self.phase = "task"
        actual = self.read()
        self.event("task_ready", readback=actual)
        return actual

    def park(self, finish, confirm_support):
        self.phase = "hold"
        if self.device.torque_states() != ON:
            raise RuntimeError("torque state cannot be confirmed before retreat")
        held = self.read()
        # Command the measured pose first, including on a controlled cancellation.
        self.send(held)
        self._settle(held)
        if not finish():
            raise RuntimeError("clearance or empty gripper could not be confirmed; follower held for operator")
        self.phase = "return"
        # Follow measured task poses in reverse while keeping the empty gripper open.
        grip = held["gripper"]
        for point in (*reversed(self.path), *reversed(self.plan.route)):
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
