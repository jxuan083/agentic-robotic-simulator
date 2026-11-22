"""Route policy chunks and hand-guided targets to explicitly selected SO-101 pairs."""

from collections import deque
from contextlib import ExitStack, contextmanager
from math import isfinite


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
UNITS = dict.fromkeys(JOINTS[:-1], "degree") | {"gripper": "percent_open"}


def pose(values):
    if not isinstance(values, dict) or set(values) != set(JOINTS):
        raise ValueError("pose must contain exactly the six named SO-101 joints")
    result = {}
    for joint in JOINTS:
        value = values[joint]
        if type(value) not in (int, float) or not isfinite(value):
            raise ValueError(f"invalid value for {joint}")
        if joint == "gripper" and not 0 <= value <= 100:
            raise ValueError("gripper must be between 0 and 100 percent open")
        result[joint] = float(value)
    return result


@contextmanager
def open_pairs(arms, open_leader, open_follower):
    """Connect and check every named pair before any follower is armed."""
    if not arms or len(arms) > 2:
        raise ValueError("select one or two named arm pairs")
    with ExitStack() as stack:
        pairs = {
            name: (stack.enter_context(open_leader(name, spec["leader"])),
                   stack.enter_context(open_follower(name, spec["follower"])))
            for name, spec in arms.items()
        }
        for _, follower in pairs.values():
            follower.arm()
        yield pairs


class ControlRouter:
    """Single-loop controller. The generation rejects chunks from a superseded owner."""

    def __init__(self, pairs, *, start="manual"):
        if not pairs or start not in ("policy", "manual"):
            raise ValueError("a controller needs at least one arm and a valid initial source")
        self.pairs = dict(pairs)
        self.source = start
        self.generation = 0
        self._policy = deque()
        self._origins = None
        self._takeover = None

    def enqueue_policy(self, chunk, *, generation):
        if self.source != "policy" or generation != self.generation:
            return False
        actions = []
        for action in chunk:
            if set(action) != set(self.pairs):
                raise ValueError("policy action must target precisely the selected arms")
            actions.append({name: pose(action[name]) for name in self.pairs})
        self._policy.extend(actions)
        return True

    def take_over(self, timestamp_ns):
        if self.source == "manual":
            return
        self._takeover = (len(self._policy), timestamp_ns)
        self._policy.clear()
        self.generation += 1
        self.source = "manual"
        self._origins = None

    def step(self, clock):
        """Send one whole step and capture post-command readback from each follower."""
        event = None
        if self.source == "manual":
            leaders = {name: pose(leader.read()) for name, (leader, _) in self.pairs.items()}
            if self._origins is None:
                starts = {name: pose(follower.read()) for name, (_, follower) in self.pairs.items()}
                self._origins = leaders, starts
                if self._takeover is not None:
                    discarded, switch_ns = self._takeover
                    event = {
                        "type": "takeover",
                        "discarded_policy_actions": discarded,
                        "switch_ns": switch_ns,
                        "generation": self.generation,
                        "aligned_readback": starts,
                    }
                    self._takeover = None
            initial_leaders, starts = self._origins
            targets = {}
            for name, leader in leaders.items():
                target = {j: starts[name][j] + leader[j] - initial_leaders[name][j] for j in JOINTS}
                target["gripper"] = min(100, max(0, target["gripper"]))
                targets[name] = target
        else:
            if not self._policy:
                raise RuntimeError("no fresh policy action is queued")
            targets = self._policy.popleft()

        arms = {}
        for name, (_, follower) in self.pairs.items():
            command = pose(follower.send(pose(targets[name])))
            command_ns = clock()
            readback = pose(follower.read())
            arms[name] = {
                "command": command, "command_ns": command_ns,
                "readback": readback, "readback_ns": clock(),
            }
        return {"timestamp_ns": clock(), "command_source": self.source,
                "task_is_policy": int(self.source == "policy"), "arms": arms, "event": event}
