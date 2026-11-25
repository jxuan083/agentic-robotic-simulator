"""PR #17 round 3 (pure Python, no simulator): injected keys move by simulated time, a perception error takes the
camera cell back to the raw image, the benchmark report reads only its own run, and the screen video start / stop."""
import json
import os
import stat
import sys
import time

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from agentic_sim.config import load  # noqa: E402
from agentic_sim.kinematics import So101Chain  # noqa: E402
from agentic_sim.teleop import script_core  # noqa: E402
from agentic_sim.teleop.ee_target import load_controller  # noqa: E402
from agentic_sim.workspace_status import cell_view, perception_item, status_lines, Clocks  # noqa: E402

CONTROL_DT = 1 / 60


# ---- 1. injected keys move by simulated time ----------------------------------------------------------------------
def _scripted_x(fps, scripted, monkeypatch, pause_frames=0):
    """Hold W until target x passes +30 mm; every frame advances one control step of simulated time (as both
    engines do), the window draws at `fps`. Returns the final target x offset (m) and the per-frame steps."""
    clock = {"t": 0.0}
    monkeypatch.setattr(script_core.time, "monotonic", lambda: clock["t"])
    chain = So101Chain.from_dict(yaml.safe_load(open(os.path.join(ROOT, "configs/so101_chain.yaml"))))
    ctrl = load_controller(chain, load("configs/teleop.yaml"))
    x0 = float(ctrl.state.pos[0])
    held = set()

    class Script(script_core.ScriptCore):
        def _key(self, name, down):
            (held.add if down else held.discard)(name)

    sc = Script([{"hold": ["W"], "until": f"target[0] >= {x0 + 0.030}", "sim_timeout": 5}], log=lambda *a: None)
    t_sim, steps, frame = 0.0, [], 0
    while not sc.done:
        host_dt = 1.0 / fps
        clock["t"] += host_dt
        paused = frame < pause_frames
        if not paused:
            before = float(ctrl.state.pos[0])
            dt_in = script_core.input_dt(host_dt, CONTROL_DT, scripted)
            ctrl.step_velocity([1.0 if "W" in held else 0.0, 0.0, 0.0], 0.0, dt_in, "keyboard")
            steps.append(float(ctrl.state.pos[0]) - before)
            t_sim += CONTROL_DT
        sc.step({"t_sim": t_sim, "target": ctrl.state.pos.copy()})
        frame += 1
    return float(ctrl.state.pos[0]) - x0, steps


def test_injected_keys_move_by_simulated_time_whatever_the_frame_rate(monkeypatch):
    fast, _ = _scripted_x(60, True, monkeypatch)
    slow, steps = _scripted_x(13, True, monkeypatch)
    assert abs(fast - slow) < 1e-9                         # the same motion at 60 and 13 fps
    assert max(steps) <= 0.08 * CONTROL_DT + 1e-9          # one control step: 1.33 mm
    assert fast - 0.030 < 0.08 * CONTROL_DT + 1e-9         # the until threshold is passed by less than one step
    host_slow, host_steps = _scripted_x(13, False, monkeypatch)   # the old host-time path: ~6 mm per frame
    assert max(host_steps) > 0.005 and host_slow - 0.030 > fast - 0.030


def test_a_person_keeps_host_time_and_a_pause_accumulates_nothing(monkeypatch):
    assert script_core.input_dt(0.05, CONTROL_DT, False) == 0.05     # operator feel: speed per host second
    assert script_core.input_dt(0.05, CONTROL_DT, True) == CONTROL_DT
    paused, steps = _scripted_x(13, True, monkeypatch, pause_frames=40)   # 3 s paused before the hold moves
    plain, _ = _scripted_x(13, True, monkeypatch)
    assert abs(paused - plain) < 1e-9 and max(steps) <= 0.08 * CONTROL_DT + 1e-9   # no jump after the pause


def test_marks_summary_is_json_ready():
    import numpy as np
    out = script_core.marks_summary({"grasp": {"target": np.array([0.23, 0.155, 0.029]), "t_sim": 12.3456789,
                                               "status": "ok", "stats": {"x": 1}}})
    assert json.loads(json.dumps(out)) == {"grasp": {"target": [0.23, 0.155, 0.029], "t_sim": 12.34568,
                                                     "status": "ok"}}


# ---- 2. a perception error takes the cell back to the raw image -----------------------------------------------------
def test_an_error_shows_the_raw_image_until_a_new_valid_result():
    good = {"latest": {"captured_at_monotonic_s": 9.5, "completed_at_monotonic_s": 9.8, "frame_number": 7}}
    assert cell_view(True, perception_item(good, 10.0))[0] == "annotated"
    failed = dict(good, error="injected test fault (--perception-fault)")       # 0.5 s later: the next one failed
    it = perception_item(failed, 10.0)
    mode, text = cell_view(True, it)
    assert mode == "raw" and "perception error: injected test fault" in text and it["stale"]
    other = perception_item(good, 10.0)                                          # the other camera is not affected
    assert cell_view(True, other)[0] == "annotated"
    recovered = {"latest": {"captured_at_monotonic_s": 14.0, "completed_at_monotonic_s": 14.3, "frame_number": 9}}
    assert cell_view(True, perception_item(recovered, 14.5))[0] == "annotated"
    # unchanged: older than 2 s, no result yet, display off
    assert cell_view(True, perception_item(good, 12.0))[0] == "raw"
    assert cell_view(True, perception_item({"latest": None}, 1.0))[0] == "raw"
    assert cell_view(False, perception_item(good, 10.0)) == ("raw", "raw")
    lines = "\n".join(status_lines({"clocks": Clocks(), "perception": {"global": it}, "recording": None}))
    assert "ERROR: injected test fault" in lines


def test_the_injected_fault_fails_one_camera_for_a_while_then_recovers(monkeypatch):
    from agentic_sim.perception import live
    from agentic_sim.perception_cli import parse_fault
    assert parse_fault("global:0.5:4") == ("global", 0.5, 4.0) and parse_fault(None) is None
    for bad in ("both:1:1", "global:1", "global:-1:1", "wrist:0:0", "global:x:1"):
        with pytest.raises(ValueError):
            parse_fault(bad)
    clock = {"t": 100.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    fault = live.InjectedFault.parse("global:0.5:4")
    run = fault.wrap(lambda *a: "ok")
    assert run() == "ok" and fault.window() == (100.5, 104.5)
    clock["t"] = 101.0
    with pytest.raises(RuntimeError, match="injected test fault"):
        run()
    clock["t"] = 104.6
    assert run() == "ok"


def test_the_fault_option_reaches_the_command_only_when_set():
    from types import SimpleNamespace

    from agentic_sim.perception_cli import args
    base = dict(live_perception=True, perception_camera="both", perception_hz=2.0, perception_json=None)
    assert "--perception-fault" not in args(SimpleNamespace(**base))
    assert args(SimpleNamespace(**base, perception_fault="wrist:1:2"))[-2:] == ["--perception-fault", "wrist:1:2"]


# ---- 3. the benchmark report reads only its own run -----------------------------------------------------------------
def test_bench_report_binds_metrics_and_episodes_to_the_run_id(tmp_path, monkeypatch):
    from tools import bench_workspace as bw
    monkeypatch.setattr(bw, "ROOT", str(tmp_path))
    monkeypatch.setattr(bw, "figure", lambda *a: None)
    out = "custom_bench"
    base = tmp_path / out
    rec = base / "record_raw"
    rec.mkdir(parents=True)
    # the records go under --out, bound to the run id
    cmds = dict(bw.runs(["isaac"], out))
    assert bw.record_dir(out, "raw") in cmds["isaac_record_raw"]
    cmd = bw.run_command(cmds["isaac_record_raw"], out, "isaac_record_raw", "run2")
    assert cmd[cmd.index("--record_run_id") + 1] == "run2" and cmd[cmd.index("--metrics_run_id") + 1] == "run2"

    def episode(name, run_id, success, manifest_run_id=None, event_run_id=None, write_manifest=True):
        d = rec / name
        d.mkdir(exist_ok=True)
        if write_manifest:
            (d / "manifest.json").write_text(json.dumps({
                "capture_run_id": manifest_run_id or run_id, "operator": "scripted_input", "control_frames": 234,
                "result": {"success": success, "reason": "placed" if success else "closed"}}))
        with open(rec / "recorder_events.jsonl", "a") as f:
            f.write(json.dumps({"event": "saved", "run_id": event_run_id or run_id,
                                "episode": f"{out}/record_raw/{name}", "success": success, "frames": 234}) + "\n")
    episode("first_yellow", "run1", True)                       # first run: success
    with open(base / "isaac_record_raw.jsonl", "w") as f:       # metrics: an old line of run1, then run2
        f.write(json.dumps({"run_id": "run1", "t_sim": 5, "ui_fps": 27, "rtf": 0.4, "paused": False}) + "\n")
        f.write(json.dumps({"run_id": "run2", "t_sim": 5, "ui_fps": 13, "rtf": 0.2, "paused": False}) + "\n")
    (base / "isaac_record_raw.run.json").write_text(json.dumps({"key": "isaac_record_raw", "run_id": "run1",
                                                                  "exit": 0}))
    bw.report(type("A", (), {"out": out})())
    s1 = json.load(open(base / "summary.json"))["isaac_record_raw"]
    assert s1["episode"]["path"].endswith("first_yellow") and s1["episode"]["success"] is True
    assert s1["episode"]["frames"] == 234 and s1["successes"] == 1 and s1["ui_fps"]["median"] == 27
    # second run aborted before saving: the report must not take the first run's episode
    (base / "isaac_record_raw.run.json").write_text(json.dumps({"key": "isaac_record_raw", "run_id": "run2",
                                                                  "exit": "timeout"}))
    bw.report(type("A", (), {"out": out})())
    s2 = json.load(open(base / "summary.json"))["isaac_record_raw"]
    assert s2["episode"] == bw.NO_EPISODE and s2["episodes"] == [] and s2["ui_fps"]["median"] == 13
    assert bw.NO_EPISODE in (base / "summary.md").read_text()


# ---- 4. screen video start / stop -----------------------------------------------------------------------------------
FAKE_TOOL = """#!/usr/bin/env bash
# stands in for tools/record_isaac_window.sh: records until SIGINT / SIGTERM, then writes the file
OUT=$2
echo "window $ISAAC_WINDOW_ID" >&2
trap 'printf video > "$OUT"; echo "wrote $OUT" >&2; exit 0' INT TERM
while true; do sleep 0.05; done
"""


def _tool(tmp_path, body):
    p = tmp_path / "tool.sh"
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


def _wait(cond, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def linux_recording_host(monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("recording helper uses Linux process signals and bash")
    # These tests supply a fake recorder and window ID; no real X server is needed.
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.delenv("ARS_SCREEN_VIDEO", raising=False)


@pytest.mark.usefixtures("linux_recording_host")
def test_screen_video_start_stop_gives_a_new_file_each_time(tmp_path, monkeypatch):
    from agentic_sim import screen_video as sv
    monkeypatch.setattr(sv, "ROOT", str(tmp_path))
    ctl = "outputs/screen_video/control/t1"
    client = sv.ScreenVideoClient(ctl, lambda: 0x4400007, "mujoco", log=lambda *a: None)
    client.start()                                      # no agent yet: say so, claim nothing
    assert "helper not running" in client.text() and not client.recording()
    agent = sv.AgentThread(ctl, log=lambda *a: None, tool=_tool(tmp_path, FAKE_TOOL))
    try:
        assert _wait(client.agent_alive)
        paths = []
        for _ in range(2):
            client.start()
            assert client.recording()
            assert _wait(lambda: (client.status(time.monotonic() + 1) or {}).get("state") == "recording")
            assert "REC" in client.text() and client.button_text() == "Stop screen video"
            client.stop()
            assert _wait(lambda: (client.status(time.monotonic() + 1) or {}).get("state") == "saved")
            path = client.status()["path"]
            assert os.path.exists(tmp_path / path) and f"saved {path}" in client.text()
            assert not client.recording() and client.button_text() == "Record screen"
            paths.append(path)
        assert paths[0] != paths[1]
    finally:
        saved = agent.close()
    assert saved == paths


@pytest.mark.usefixtures("linux_recording_host")
def test_screen_video_tool_that_cannot_start_reports_the_reason(tmp_path, monkeypatch):
    from agentic_sim import screen_video as sv
    monkeypatch.setattr(sv, "ROOT", str(tmp_path))
    ctl = "outputs/screen_video/control/t2"
    client = sv.ScreenVideoClient(ctl, lambda: 1, "isaac", log=lambda *a: None)
    agent = sv.AgentThread(ctl, log=lambda *a: None, tool=str(tmp_path / "missing.sh"))
    try:
        assert _wait(client.agent_alive)
        client.start()
        assert _wait(lambda: (client.status(time.monotonic() + 1) or {}).get("state") == "failed")
        assert "failed" in client.text() and "cannot start" in client.text()
        assert client.status()["saved"] == [] and not client.recording()
    finally:
        agent.close()
    broken = _tool(tmp_path, "#!/usr/bin/env bash\necho \"window 'x' not found\" >&2\nexit 1\n")
    agent = sv.AgentThread("outputs/screen_video/control/t3", log=lambda *a: None, tool=broken)
    client = sv.ScreenVideoClient("outputs/screen_video/control/t3", lambda: 1, "isaac", log=lambda *a: None)
    try:
        assert _wait(client.agent_alive)
        client.start()
        assert _wait(lambda: (client.status(time.monotonic() + 1) or {}).get("state") == "failed")
        assert "not found" in client.text() and not os.path.exists(tmp_path / client.status()["path"])
    finally:
        agent.close()


@pytest.mark.usefixtures("linux_recording_host")
def test_closing_the_simulator_ends_the_recording_and_keeps_the_file(tmp_path, monkeypatch):
    from agentic_sim import screen_video as sv
    monkeypatch.setattr(sv, "ROOT", str(tmp_path))
    ends = _tool(tmp_path, "#!/usr/bin/env bash\nsleep 0.5\nprintf video > \"$2\"\nexit 0\n")  # the window closes
    ctl = "outputs/screen_video/control/t4"
    client = sv.ScreenVideoClient(ctl, lambda: 1, "isaac", log=lambda *a: None)
    agent = sv.AgentThread(ctl, log=lambda *a: None, tool=ends)
    assert _wait(client.agent_alive)
    client.start()
    assert _wait(lambda: (client.status(time.monotonic() + 1) or {}).get("state") in ("recording", "saved"))
    saved = agent.close()                                  # the launcher ends: the agent waits for the file
    assert len(saved) == 1 and os.path.exists(tmp_path / saved[0])



def test_a_key_hold_times_out_in_simulated_time_a_wait_hold_in_host_time(monkeypatch):
    """At 0.5 x real time, a 2 s key hold lasts 2 simulated s (4 host s); a hold with no keys gives up after 2 host s."""
    clock = {"t": 0.0}
    monkeypatch.setattr(script_core.time, "monotonic", lambda: clock["t"])

    class Script(script_core.ScriptCore):
        def _key(self, name, down):
            pass

    for keys, host_s in ((["W"], 4.0), ([], 2.0)):
        clock["t"], t_sim = 0.0, 0.0
        sc = Script([{"hold": keys, "until": "False", "timeout": 2.0}], log=lambda *a: None)
        while not sc.done:
            sc.step({"t_sim": t_sim})
            clock["t"] += 0.1
            t_sim += 0.05
        assert abs(clock["t"] - host_s) < 0.25, (keys, clock["t"])


def test_bench_report_rejects_episodes_it_cannot_verify(tmp_path, monkeypatch):
    """A saved event of this run that points at an old manifest (another run id), or at a missing manifest, is listed
    with the reason and never counted as a success; a verified episode of the same run is kept."""
    from tools import bench_workspace as bw
    monkeypatch.setattr(bw, "ROOT", str(tmp_path))
    monkeypatch.setattr(bw, "figure", lambda *a: None)
    out, rec = "b", tmp_path / "b" / "record_raw"
    (rec / "old_yellow").mkdir(parents=True)
    (rec / "old_yellow" / "manifest.json").write_text(json.dumps({"capture_run_id": "run1", "control_frames": 300,
                                                                  "result": {"success": True}}))
    (rec / "bad_yellow").mkdir()
    (rec / "bad_yellow" / "manifest.json").write_text("{not json")
    (rec / "good_yellow").mkdir()
    (rec / "good_yellow" / "manifest.json").write_text(json.dumps({"capture_run_id": "run2", "control_frames": 120,
                                                                   "result": {"success": False, "reason": "closed"}}))
    with open(rec / "recorder_events.jsonl", "w") as f:
        for name in ("old_yellow", "gone_yellow", "bad_yellow", "good_yellow"):
            f.write(json.dumps({"event": "saved", "run_id": "run2", "episode": f"{out}/record_raw/{name}",
                                "success": True, "frames": 300}) + "\n")
    valid, rejected = bw.run_episodes(str(rec), "run2")
    assert [e["path"].split("/")[-1] for e in valid] == ["good_yellow"] and valid[0]["success"] is False
    assert valid[0]["frames"] == 120                               # from the manifest, not the event
    reasons = {r["path"].split("/")[-1]: r["reason"] for r in rejected}
    assert reasons["old_yellow"].startswith("run id mismatch") and reasons["gone_yellow"] == "manifest missing"
    assert reasons["bad_yellow"].startswith("manifest unreadable")
    (tmp_path / out / "isaac_record_raw.run.json").write_text(json.dumps({"key": "isaac_record_raw",
                                                                          "run_id": "run2", "exit": 0}))
    bw.report(type("A", (), {"out": out})())
    s = json.load(open(tmp_path / out / "summary.json"))["isaac_record_raw"]
    assert s["successes"] == 0 and len(s["rejected_episodes"]) == 3
    md = (tmp_path / out / "summary.md").read_text()
    row = [ln for ln in md.splitlines() if "old_yellow" in ln][0]
    assert "run id mismatch" in row and "True" not in row          # never "mismatch" and "success" together


@pytest.mark.parametrize("plat,env,reason", [("win32", {"DISPLAY": ":1"}, "not available on this platform"),
                                             ("linux", {}, "DISPLAY unset"),
                                             ("linux", {"DISPLAY": ":1", "ARS_SCREEN_VIDEO": "off"}, "turned off")])
def test_screen_video_is_disabled_where_the_platform_cannot_record(tmp_path, monkeypatch, capsys, plat, env, reason):
    """Windows (no X11 window ids, no shell tool), no X display, or forced off: the button is disabled, the right
    column says why, nothing is requested and no helper is started."""
    from agentic_sim import screen_video as sv
    monkeypatch.setattr(sv, "ROOT", str(tmp_path))
    monkeypatch.setattr(sv.sys, "platform", plat)
    for k in ("DISPLAY", "ARS_SCREEN_VIDEO"):
        monkeypatch.delenv(k, raising=False)
    for k, val in env.items():
        monkeypatch.setenv(k, val)
    if plat == "win32":
        monkeypatch.setattr(sv.platform, "system", lambda: "Windows")
    ok, why = sv.platform_support()
    assert not ok and reason in why
    client = sv.ScreenVideoClient("outputs/screen_video/control/w", lambda: 1, "mujoco", log=lambda *a: None)
    client.toggle()
    client.start()
    assert not client.enabled and not client.recording() and client.button_text() == "Screen video unavailable"
    assert reason in client.text() and not (tmp_path / "outputs").exists()     # no request, no control directory
    import sim
    agent = sim.screen_agent("outputs/screen_video/control/w")
    with agent:
        pass
    assert agent.saved() == [] and "此平台未提供視窗錄影" in capsys.readouterr().out
