"""Operator workspace clocks and status (pure Python): UI FPS, RTF per running segment, perception result age."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agentic_sim.workspace_status import (NO_RESULT, Clocks, MetricsLog, cell_view, perception_item,  # noqa: E402
                                          status_lines)


def _run(c, host0, sim0, n, dt_host, rtf, paused=False):
    h, s = host0, sim0
    for _ in range(n):
        c.frame(h, s, paused)
        h += dt_host
        if not paused:
            s += dt_host * rtf
    return h, s


def test_ui_fps_and_rtf_are_separate_clocks():
    c = Clocks(window_s=1.0)
    h, s = _run(c, 100.0, 0.0, 61, 1 / 30, 0.25)          # 30 frames per host second, simulation at 0.25 x
    assert abs(c.ui_fps - 30) < 0.5 and abs(c.rtf - 0.25) < 1e-6


def test_pause_stops_the_rtf_and_no_rtf_spans_a_pause():
    c = Clocks(window_s=1.0)
    h, s = _run(c, 0.0, 0.0, 40, 1 / 20, 1.0)
    h, s2 = _run(c, h, s, 40, 1 / 20, 1.0, paused=True)   # paused: simulated time stays
    assert s2 == s and c.rtf is None and c.paused
    assert c.ui_fps is not None                            # the window keeps drawing while paused
    h, s = _run(c, h + 5.0, s, 3, 1 / 20, 0.5)             # resumed (5 s of host time passed while paused)
    assert abs(c.rtf - 0.5) < 1e-6                         # measured only after the pause, not 0.5 / 5 s


def test_reset_starts_a_new_segment():
    c = Clocks()
    h, s = _run(c, 0.0, 0.0, 30, 1 / 30, 1.0)
    c.reset()
    assert c.rtf is None
    c.frame(h, 0.0, False)                                 # Reset: simulated time starts over
    assert c.rtf is None
    c.frame(h + 0.1, 0.05, False)
    assert abs(c.rtf - 0.5) < 1e-6
    c2 = Clocks()
    _run(c2, 0.0, 5.0, 10, 0.1, 1.0)
    c2.frame(1.0, 0.0, False)                              # simulated time went back without reset(): new segment
    assert c2.rtf is None


def test_perception_age_uses_the_capture_time_and_stale_results_are_not_drawn_on_new_frames():
    assert perception_item(None, 10.0)["age_s"] is None
    assert cell_view(True, perception_item({"latest": None}, 10.0)) == ("raw", f"raw (annotation: {NO_RESULT})")
    st = {"latest": {"captured_at_monotonic_s": 9.0, "completed_at_monotonic_s": 9.6, "frame_number": 12}}
    it = perception_item(st, 10.0)
    assert abs(it["age_s"] - 1.0) < 1e-9 and abs(it["latency_s"] - 0.6) < 1e-9 and not it["stale"]
    assert cell_view(True, it)[0] == "annotated" and "1.0 s ago" in cell_view(True, it)[1]
    late = perception_item(st, 13.5)                       # an estimate delayed past STALE_S
    assert late["stale"] and cell_view(True, late) == ("raw", "raw (annotation too old: 4.5 s ago)")
    assert cell_view(False, it) == ("raw", "raw")          # display off: raw, whatever the estimator does


def test_status_lines_and_metrics_log(tmp_path):
    c = Clocks()
    _run(c, 0.0, 0.0, 31, 1 / 30, 0.5)
    lines = status_lines({"clocks": c, "target_color": "yellow", "task": None, "jaw": 0.21, "gripper_cmd": -0.17,
                          "gripper_open": False, "recording": {"state": "recording", "frames": 42,
                                                               "episode": "outputs/demos/x_yellow"},
                          "paused": False, "perception": {"global": perception_item(None, 1.0)}})
    text = "\n".join(lines)
    assert "Target colour: yellow" in text and "Task: manual teleop" in text and "Gripper: closed" in text
    assert "Recording: RECORDING  42 frames" in text and "RTF 0.50" in text and f"global: {NO_RESULT}" in text
    log = MetricsLog(str(tmp_path / "m.jsonl"), period_s=0.5)
    for i in range(5):
        log.sample(i * 0.3, i * 0.1, c, {"global": perception_item(None, 0.0)})
    log.close()
    rows = [json.loads(line) for line in open(tmp_path / "m.jsonl")]
    assert len(rows) == 2 and rows[0]["perception"]["global"]["age_s"] is None and "rtf" in rows[0]
