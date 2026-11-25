"""Test the live motion state machine with an instrumented fake follower."""

from collections import Counter
from contextlib import contextmanager
import json
from math import dist
from pathlib import Path
import subprocess
import sys

import pytest

from agentic_sim.real_collection import Collection, locked_session, make_cards
from agentic_sim.real_collection_fake import FakeCollectionBackend, FakeOperator, fake_policy, fake_setup
from agentic_sim.real_episode import Episode, write_json
from agentic_sim.real_motion import OFF, ON
from agentic_sim.real_round import replay_round


ROOT = Path(__file__).resolve().parents[1]


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def session(tmp_path):
    plan, layout = fake_setup()
    backend = FakeCollectionBackend(plan)
    return Collection(tmp_path, backend, plan, layout)


def record(session, **options):
    card, _ = session.current()
    return session.record(FakeOperator(card["color"], **options), card["pose"])


def test_command_chronology_and_first_replay_gate_survive_restart(session):
    entry = record(session)
    source = entry["episode"]
    session.approve(source)
    assert session.current()[1]["status"] == "needs_replay"
    assert session.progress()[0] == 0
    original = session.root / source
    start = (original / "control.jsonl").read_bytes()
    with pytest.raises(ValueError, match="confirm clearance"):
        session.replay(source, FakeOperator(replay=True), layout_ready=False)
    reopened = Collection(session.root, session.backend, session.plan, session.layout)
    assert reopened.current()[1]["episode"] == source

    operation = reopened.replay(source, FakeOperator(replay=True), layout_ready=True)
    replay_dir = reopened.root / operation["record"]
    assert operation["passed"] and operation["source_episode_id"] == "D001-01"
    assert reopened.progress()[0] == 1
    recorded = Episode(original)
    assert recorded.rows[0]["arms"]["right"]["readback"] == session.plan.start
    actual = jsonl(replay_dir / "motion.jsonl")
    replayed = [row for row in actual if row["event"] == "replay_frame"]
    assert [row["command"] for row in replayed] == [row["arms"]["right"]["command"] for row in recorded.rows]
    assert [row["readback"] for row in replayed] == [row["command"] for row in replayed]
    relative = [row["timestamp_ns"] - replayed[0]["timestamp_ns"] for row in replayed]
    expected = [row["recorded_command_ns"] - replayed[0]["recorded_command_ns"] for row in replayed]
    assert all(abs(a - b) <= 500_000 for a, b in zip(relative, expected))

    for folder in (original, replay_dir):
        motion = jsonl(folder / "motion.jsonl")
        events = [row["event"] for row in motion]
        assert events[0:2] == ["start_measured", "torque_on"]
        assert events.index("settled") < events.index("task_ready") < events.index("support_confirmed")
        assert motion[-2]["readback"] == session.plan.park and motion[-1]["torque"] == OFF
        assert any(row["phase"] == "align" and row["event"] == "command" for row in motion)
        assert any(row["phase"] == "return" and row["event"] == "command" for row in motion)

    session = reopened
    session.replay(source, FakeOperator(replay=True), layout_ready=True)
    assert session.progress()[0] == 1
    assert (original / "control.jsonl").read_bytes() == start
    assert record(session)["status"] == "accepted"
    resumed = Collection(session.root, session.backend, session.plan, session.layout)
    assert resumed.progress()[0] == 2 and resumed.current()[0]["id"] == "D003"


def test_fake_dagger_holds_same_torque_and_discards_pending_policy(session):
    entry = session.dagger(FakeOperator(dagger=True), fake_policy)
    root = session.root / entry["episode"]
    episode = Episode(root)
    event, = entry["takeovers"]
    assert event["frame"] == 3 and event["discarded_policy_actions"] == 2
    assert episode.rows[0]["arms"]["right"]["command"] == session.plan.start
    before, after = episode.rows[2:4]
    assert before["arms"]["right"]["readback"] == after["arms"]["right"]["command"]
    assert [row["command_source"] for row in episode.rows[:5]] == ["policy"] * 3 + ["manual"] * 2
    trace = session.backend.trace
    assert [row["event"] for row in trace if row["event"] != "send"] == [
        "open", "hold_measured_before_torque", "enable", "disable", "disconnect"
    ]
    assert not {23, 24}.intersection(row["command"]["shoulder_pan"] for row in trace if row["event"] == "send")
    assert session.progress()[0] == 0 and episode.validate()["ok"]


def test_controlled_cancel_parks_but_does_not_count(session):
    episode = record(session, cancel_at=6)
    assert episode["status"] == "retry"
    assert session.progress()[0] == 0
    assert not session.state["operator_recovery_required"]
    events = jsonl(session.root / episode["episode"] / "motion.jsonl")
    assert any(row["phase"] == "return" for row in events) and events[-1]["event"] == "torque_off"


@pytest.mark.parametrize("options", [{"support": False}, {"halt_at": 4}])
def test_unconfirmed_support_or_stop_never_disengages(session, options):
    with pytest.raises(RuntimeError):
        record(session, **options)
    entry = session.state["attempts"][-1]
    events = jsonl(session.root / entry["episode"] / "motion.jsonl")
    assert events[-1]["event"] == "operator_required" and events[-1]["last_readback"]
    assert "disable" not in [row["event"] for row in session.backend.trace]
    assert session.backend.trace[-1]["torque"] == ON
    assert session.state["operator_recovery_required"]
    with pytest.raises(RuntimeError, match="verify follower support"):
        record(session)


def test_off_route_pose_or_wrong_replay_setup_is_rejected_before_motion(session):
    original = record(session)
    path = session.root / original["episode"]
    manifest = Episode(path).manifest
    manifest["motion_setup"]["version"] = "obsolete"
    write_json(path / "manifest.json", manifest)
    count = len(session.backend.trace)
    with pytest.raises(ValueError, match="setup differs"):
        replay_round(Episode(path), session.backend, session.plan, session.root / "wrong-version",
                     FakeOperator(replay=True))
    assert len(session.backend.trace) == count

    open_devices = session.backend.devices

    @contextmanager
    def off_route(*, leader):
        with open_devices(leader=leader) as (hand, follower):
            follower.state["shoulder_pan"] = 14
            yield hand, follower

    session.backend.devices = off_route
    with pytest.raises(RuntimeError, match="outside the verified start corridor"):
        session.preflight(FakeOperator())
    assert [row["event"] for row in session.backend.trace[count:]] == ["open", "disconnect"]
    assert session.backend.trace[-1]["torque"] == OFF


def test_held_out_positions_and_quota_balance_are_reproducible(session):
    cards, held_out = make_cards(session.layout)
    assert (cards, held_out) == make_cards(session.layout)
    assert len(cards) == 120 and len(held_out) == 20
    assert Counter((row["tray"], row["region"]) for row in held_out) == {
        ("pink", "near"): 5, ("pink", "far"): 5, ("beige", "near"): 5, ("beige", "far"): 5
    }
    quotas = Counter((row["source_tray"], row["color"], row["region"], row["kind"]) for row in cards)
    assert all(value == (6 if key[-1] == "normal" else 4) for key, value in quotas.items())
    assert all(dist(card["pose"][:2], reserved["pose"][:2]) >= session.layout["reserve_distance_mm"]
               for card in cards for reserved in held_out if card["source_tray"] == reserved["tray"])
    with pytest.raises(ValueError, match="held-out"):
        session.record(FakeOperator(), held_out[0]["pose"])
    assert session.state["attempts"] == [] and session.backend.trace == []


def test_interrupted_session_preserves_card_and_requires_operator_recovery(session):
    session.state["attempts"].append({"card": "D001", "episode": "episodes/D001-01", "status": "running"})
    session.save()
    with locked_session(session.root):
        with pytest.raises(RuntimeError, match="already open"):
            with locked_session(session.root):
                pytest.fail("duplicate session lock")
        resumed = Collection(session.root, session.backend, session.plan, session.layout)
        assert resumed.current()[0]["id"] == "D001"
        assert resumed.current()[1]["status"] == "interrupted"
        assert resumed.state["operator_recovery_required"]


def test_fake_demo_cli_runs_without_hardware_drivers(tmp_path):
    result = subprocess.run([sys.executable, "-S", str(ROOT / "tools/real_collection.py"),
                             "--fake", "--demo", "--session", str(tmp_path)],
                            text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert (report["demonstrations"], report["replays"], report["next_card"]) == (2, 2, "D003")
    assert (tmp_path / "fake_device_trace.json").is_file()


def test_each_card_uses_its_own_position():
    import json
    from pathlib import Path
    from agentic_sim.real_collection import make_cards

    layout = json.loads((Path(__file__).resolve().parents[1] / "configs/real_collection.cards.template.json").read_text())
    cards, reserves = make_cards(layout)
    assert len(cards) == 120 and len(reserves) == 20
    positions = [(card["source_tray"], *card["pose"][:2]) for card in cards]
    assert len(set(positions)) == 120
    held = {(row["tray"], *row["pose"][:2]) for row in reserves}
    assert not held & set(positions)
    groups = {card["layout_group"] for card in cards}
    assert all(group.split("-")[0] in ("pink", "beige") for group in groups) and len(groups) > 20


def test_cards_place_distractor_blocks():
    import json
    from collections import Counter
    from math import dist
    from pathlib import Path
    from agentic_sim.real_collection import make_cards

    layout = json.loads((Path(__file__).resolve().parents[1] / "configs/real_collection.cards.template.json").read_text())
    cards, _ = make_cards(layout)
    assert Counter(card["scene"] for card in cards) == {"target_only": 40, "near_distractor": 40, "far_distractors": 40}
    for card in cards:
        poses = [card["pose"]] + [d["pose"] for d in card["distractors"]]
        assert all(dist(a[:2], b[:2]) >= 55 for i, a in enumerate(poses) for b in poses[i + 1:])
        if card["scene"] == "target_only":
            assert not card["distractors"]
        if card["scene"] == "near_distractor":
            assert any(d["relation"] == "near" for d in card["distractors"])
    near = [d for card in cards for d in card["distractors"] if d["relation"] == "near"]
    same = sum(d["color"] == card["color"] for card in cards for d in card["distractors"] if d["relation"] == "near")
    assert 10 <= same <= len(near) - 10
    assert make_cards(layout)[0] == cards  # deterministic for the session
