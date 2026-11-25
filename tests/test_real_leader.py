"""Collection-desk leader hand-over: route out of the fold, countdown, release, sag guard."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_sim.real_leader import COUNTDOWN_S, handover  # noqa: E402
from agentic_sim.real_route_fake import CHANNEL, PARK, VIA, FakeClock, FakeDrivenLeader  # noqa: E402


def run(leader, clock, reference, halted=lambda: False):
    events, said = [], []
    after = handover(leader, reference, route=[PARK, VIA, CHANNEL], clock=clock, wait=clock.sleep, hz=30,
                     say=said.append, halted=halted, event=lambda kind, **f: events.append(kind))
    return after, events, said


def test_folded_leader_is_driven_via_route_and_released_after_countdown():
    clock = FakeClock()
    leader = FakeDrivenLeader(clock, dict(PARK) | {"gripper": 1.0})
    after, events, said = run(leader, clock, dict(CHANNEL) | {"gripper": 26.0})
    assert events[:2] == ["leader_drive_plan", "leader_torque_on"] and "leader_arrived" in events
    assert events[-2:] == ["leader_torque_off", "leader_engaged"]
    on, off = leader.log[0][1], leader.log[1][1]
    assert (off - on) / 1e9 >= COUNTDOWN_S and not leader.driven
    assert any("握住 leader" in text for text in said)


def test_dropped_leader_raises_after_release():
    clock = FakeClock()
    leader = FakeDrivenLeader(clock, CHANNEL, sag={"shoulder_lift": -30.0})
    with pytest.raises(RuntimeError, match="not supported"):
        run(leader, clock, CHANNEL)
    assert not leader.driven


def test_stop_releases_the_leader():
    clock = FakeClock()
    leader = FakeDrivenLeader(clock, PARK)
    start = clock()
    with pytest.raises(RuntimeError, match="stopped"):
        run(leader, clock, CHANNEL, halted=lambda: clock() - start > 1e9)
    assert not leader.driven
