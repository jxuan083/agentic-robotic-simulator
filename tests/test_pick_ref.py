"""A block named by position ("左上角那顆紅色"): the resolver, and the executor taking (or refusing) that block."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentic_sim.task.pick_ref import DEFAULT_FRAME, resolve  # noqa: E402
from test_task import FakeWorld  # noqa: E402

GRID = [{"block": 0, "xy": [0.30, 0.10]}, {"block": 1, "xy": [0.30, -0.10]},
        {"block": 2, "xy": [0.15, 0.10]}, {"block": 3, "xy": [0.15, -0.10]}]


@pytest.mark.parametrize("ref,want", [("top_left", 0), ("top_right", 1), ("bottom_left", 2), ("bottom_right", 3),
                                      ("farthest", 0), ("nearest", 3)])
def test_resolve_corners_and_distance(ref, want):
    pan = [0.0, -0.10]          # nearest / farthest: from the pan axis
    best, tie = resolve(GRID, ref, DEFAULT_FRAME, pan)
    assert best["block"] == want and not tie


def test_resolve_ties_single_and_frame():
    row = [{"block": 0, "xy": [0.20, 0.10]}, {"block": 1, "xy": [0.20, -0.10]}]
    assert resolve(row, "farthest", pan_xy=[0.0, 0.0])[1]                    # same distance: words do not decide
    assert resolve(row, "leftmost")[0]["block"] == 0
    assert resolve([GRID[3]], "top_left") == (GRID[3], False)                # only one: that one
    assert resolve([], "top_left") == (None, False)
    across = {"up": [-1.0, 0.0], "left": [0.0, -1.0]}                        # seen from across the table
    assert resolve(GRID, "top_left", across)[0]["block"] == 3
    with pytest.raises(ValueError):
        resolve(GRID, "middle")


def run(w, cond, max_s=200.0):
    w.run(lambda w: cond(w), max_s=w.t + max_s)


def kinds(w, k):
    return [e for e in w.ex.events if e["kind"] == k]


def test_executor_picks_the_named_block():
    # three reds in a row along +x at the same y: "top" (farther from the base) is block 1
    w = FakeWorld(["red", "red", "red", "blue"])
    w.step()
    assert {b["block"]: b["reachable"] for b in w.ex.pick_view()["storage_blocks"]} == {0: True, 1: True, 2: True, 3: True}
    w.ex.order({"red": 1}, "add", pick={"color": "red", "ref": "top_left"})
    run(w, lambda w: w.ex.state == "done")
    t = kinds(w, "pick_target")
    assert len(t) == 1 and t[0]["block"] == 1 and t[0]["ref"] == "top_left"
    assert list(w.ex.delivered_blocks) == [1]
    assert w.ex.ledger.snapshot()["delivered"] == {"red": 1} and w.ex.pins == []


@pytest.mark.parametrize("prefer", [False, True])
def test_named_block_out_of_reach_is_taken_off_the_order(prefer):
    w = FakeWorld(["red", "red", "red", "blue"])
    w.ex.prefer_reachable = prefer      # default False: unnamed picks keep the original order
    w.step()
    far = w.ex.block_xy(1)
    w.ex._reach_cache[(round(far[0], 3), round(far[1], 3))] = False     # as if block 1 lay beyond the reach
    w.ex.order({"red": 2}, "total")
    w.ex.order({"red": 1}, "add", pick={"color": "red", "ref": "top_left"})
    run(w, lambda w: w.ex.state == "done")
    r = kinds(w, "pick_rejected")
    assert len(r) == 1 and r[0]["block"] == 1 and r[0]["reason"] == "unreachable" and r[0]["reachable_left"] == 2
    led = w.ex.ledger.snapshot()
    assert led["target"] == {"red": 2} and led["delivered"] == {"red": 2}
    # the unnamed picks: original order (block 0, then 1) unless prefer_reachable skips the unreachable block 1
    assert sorted(w.ex.delivered_blocks) == ([0, 2] if prefer else [0, 1])


def test_pick_order_checks_and_a_new_total_clears_pins():
    w = FakeWorld(["red", "red", "blue"])
    w.step()
    with pytest.raises(ValueError):
        w.ex.order({"red": 1}, "add", pick={"color": "red", "ref": "middle"})
    with pytest.raises(ValueError):
        w.ex.order({"red": 2}, "add", pick={"color": "red", "ref": "top_left"})
    with pytest.raises(ValueError):
        w.ex.order({"red": 1}, "total", pick={"color": "red", "ref": "top_left"})
    w.ex.order({"red": 1}, "add", pick={"color": "red", "ref": "top_left"})
    w.ex.order({"blue": 1}, "total")
    w.step()
    w.step()
    assert w.ex.pins == []


def test_pick_passes_the_real_protocol_check():
    from agentic_sim.server.protocol import ProtocolError, _check_args
    _check_args("task_order", {"counts": {"red": 1}, "mode": "add", "pick": {"color": "red", "ref": "top_left"}}, 1)
    _check_args("task_order", {"counts": {"red": 1}}, 1)                     # the original form still passes
    with pytest.raises(ProtocolError):
        _check_args("task_order", {"counts": {"red": 1}, "pick": "top_left"}, 1)


def test_task_state_view_is_unchanged_for_the_panel():
    """view() (read by the simulator panel every frame) has the original keys; the block list is only in task_state."""
    w = FakeWorld(["red", "blue"])
    w.step()
    assert set(w.ex.view()) == {"state", "revision", "ledger", "holding", "skill", "question", "placement", "storage",
                                "colors_in_scene", "events"}
    assert {"storage_blocks", "pick_frame", "pan_xy", "pins"} <= set(w.ex.pick_view())
