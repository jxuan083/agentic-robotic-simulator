"""Shared entry (tools/sim.py): record a demonstration and read it back. Pure Python, no simulator.

The capture command is built from the same options that are shown for copying; the recorder's event file decides
which episode was just saved (older episodes in the same directory are not taken for it); the readback runs the
command it shows."""
import io
import json
import os
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from agentic_sim import demo_entry  # noqa: E402
from agentic_sim.experiments import load_experiment  # noqa: E402
from agentic_sim.launch import Command, LaunchError, RecordOptions, build_record, check_record  # noqa: E402

EXP = load_experiment("lab_20261120")


def test_record_command_defaults_and_options():
    argv = build_record("isaac", EXP, RecordOptions()).argv
    assert argv[:3] == ["docker/isaac/run.sh", "/isaac-sim/python.sh", "scripts/isaac_ee_teleop.py"]
    s = " ".join(argv)
    assert "--scene configs/scenes/lab_20261120.yaml" in s and "--block_colors red,yellow,blue" in s
    assert "--target_color yellow --record_hz 20 --record outputs/demos" in s and "--wrist_lens" not in s
    s = " ".join(build_record("isaac", EXP, RecordOptions(block_colors=["blue", "red"], target_color="red",
                                                           record_hz=10, out_dir="x/y", wrist_lens="pinhole")).argv)
    assert "--block_colors blue,red" in s and "--wrist_lens pinhole --target_color red --record_hz 10 --record x/y" in s


def test_record_setup_problems_are_reported_before_isaac_starts():
    assert check_record(EXP, RecordOptions()) == []
    assert "不在積木配置" in check_record(EXP, RecordOptions(target_color="purple"))[0]
    assert "不認得的積木顏色" in check_record(EXP, RecordOptions(block_colors=["red", "gold"], target_color="red"))[0]
    with pytest.raises(LaunchError):
        build_record("isaac", EXP, RecordOptions(target_color="purple"))
    with pytest.raises(LaunchError, match="只支援 Isaac"):
        build_record("mujoco", EXP, RecordOptions())


FAKE_RECORDER = textwrap.dedent("""
    import json, os, sys, time
    out, run_id = sys.argv[1], sys.argv[2]
    path = os.path.join(out, "recorder_events.jsonl")
    def ev(rid=run_id, **k):
        with open(path, "a") as f:
            f.write(json.dumps(dict(k, run_id=rid, pid=55)) + "\\n")
    # another recorder writing to the same directory meanwhile (same container pid 55): not this run
    ev(rid="other", event="saved", episode=os.path.join(out, "other_run"), success=True, reason="placed", frames=7)
    ep = os.path.join(out, "new_yellow")
    ev(event="episode", episode=ep, target_color="yellow", state="waiting_cameras")
    ev(event="recording", episode=ep, frames=1)
    ev(event="progress", episode=ep, frames=20)
    ev(event="saved", episode=ep, success=True, reason="placed", frames=31, duration_s=4.5)
    ev(event="episode", episode=os.path.join(out, "next_yellow"), target_color="yellow")
    ev(event="saved", episode=os.path.join(out, "next_yellow"), success=False, reason="closed", frames=0, duration_s=0)
    ev(event="closed", episodes=[ep])
""")


def test_recording_reports_only_the_episodes_of_this_run(tmp_path):
    out = tmp_path / "demos"
    out.mkdir()
    # an older run in the same directory: must not be taken for the new one
    (out / "recorder_events.jsonl").write_text(json.dumps({"event": "saved", "episode": str(out / "old"),
                                                           "success": True, "frames": 99, "pid": 55,
                                                           "run_id": "abc"}) + "\n")
    script = tmp_path / "fake.py"
    script.write_text(FAKE_RECORDER)
    buf = io.StringIO()
    res = demo_entry.run_recording(Command([sys.executable, str(script), str(out), "abc"]), str(tmp_path), "demos",
                                   "yellow", dict(os.environ), "logs", "abc", out=buf, poll_s=0.05)
    assert res["exit_code"] == 0 and not res["interrupted"]
    assert [e["episode"] for e in res["saved"]] == [str(out / "new_yellow"), str(out / "next_yellow")]
    text = buf.getvalue()
    assert "兩台相機就緒" in text and "已存檔（成功：放進米色盤）：31 幀" in text
    assert "old" not in text and "other_run" not in text
    assert os.path.exists(tmp_path / res["log"])


def test_status_line_states():
    st = demo_entry.Status("yellow")
    assert "啟動 Isaac" in st.line()
    st.apply({"event": "episode", "episode": "e1", "target_color": "yellow"})
    assert st.line() == "等待相機…"
    assert st.apply({"event": "recording", "episode": "e1", "frames": 1}) == "兩台相機就緒，開始錄製"
    assert st.apply({"event": "progress", "episode": "e1", "frames": 40}) is None
    assert st.line().startswith("錄製中　目標 yellow　已錄 40 幀")
    assert "已存檔" in st.apply({"event": "saved", "episode": "e1", "success": True, "reason": "placed", "frames": 41,
                                 "duration_s": 6.0})


def _episode(tmp_path):
    """A complete recorded episode (the capture tests' synthetic pick and place)."""
    from test_capture import _pick_place, _recorder
    rec, lay = _recorder(tmp_path)
    _pick_place(rec, lay)
    return rec.episodes[0]["path"]


def test_menu_readback_matches_the_cli_and_points_at_a_bad_frame(tmp_path, monkeypatch, capfd):
    import sim
    import read_episode
    ep = _episode(tmp_path)
    # the menu runs the command it shows: run it in-process here (uv is the same interpreter in the tests)
    monkeypatch.setattr(sim.subprocess, "call", lambda argv, cwd=None: read_episode.main(argv[argv.index(ep):]))
    assert sim.readback_flow(ep, False, False, False) == 0
    menu_out = capfd.readouterr().out
    assert read_episode.main([ep]) == 0
    cli_out = capfd.readouterr().out
    report = lambda s: s[s.index("episode："):]  # noqa: E731
    assert report(menu_out) == report(cli_out) and "讀回結果：通過" in cli_out
    # one wrist row names a file that does not exist: both fail and name camera, frame and reason
    rows = [json.loads(line) for line in open(os.path.join(ep, "control.jsonl"))]
    rows[5]["images"]["wrist"] = "missing.png"
    with open(os.path.join(ep, "control.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    assert sim.readback_flow(ep, False, False, False) == 1
    menu_out = capfd.readouterr().out
    assert read_episode.main([ep]) == 1
    cli_out = capfd.readouterr().out
    assert report(menu_out) == report(cli_out)
    assert "wrist 第 5 幀：control row names missing.png" in cli_out and "資料不完整" in cli_out


def test_readback_without_a_recording_never_guesses(tmp_path, monkeypatch, capsys):
    import sim
    monkeypatch.setattr(sim.Session, "saved", [])
    (tmp_path / "outputs/demos/old_yellow").mkdir(parents=True)
    (tmp_path / "outputs/demos/old_yellow/manifest.json").write_text("{}")
    monkeypatch.setattr(sim, "ROOT", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda *_: "1")
    expected = os.path.join("outputs/demos", "old_yellow")
    assert sim.pick_episode("outputs/demos") == expected
    out = capsys.readouterr().out
    assert "這次還沒有錄製" in out and f"既有：{expected}" in out
    # after a recording, the episode the recorder reported is the one read back
    monkeypatch.setattr(sim.Session, "saved", [{"episode": "outputs/demos/new_yellow", "success": True, "frames": 30}])
    assert sim.pick_episode("outputs/demos") == "outputs/demos/new_yellow"


def test_interactive_record_stays_in_the_menu_until_the_target_is_in_the_layout(monkeypatch, capsys):
    import sim
    answers = iter(["purple", "layout", "red,purple", "", "n", "n"])   # ..., perception off, do not start
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    rec = RecordOptions()
    assert sim.record_flow(EXP, rec, False, True) == 0          # "n": shown, not started
    out = capsys.readouterr().out
    assert "需要修正：目標顏色 purple 不在積木配置 red,yellow,blue 裡" in out
    assert rec.block_colors == ["red", "purple"] and rec.target_color == "purple"
    assert "--block_colors red,purple --target_color purple" in out


def test_record_output_must_be_inside_the_repo():
    """The Isaac container mounts only the repo: another directory would be written inside the container and lost."""
    from agentic_sim.launch import REPO_ROOT, record_dir
    assert record_dir("outputs/demos") == "outputs/demos"
    assert record_dir(os.path.join(REPO_ROOT, "outputs", "x")) == "outputs/x"   # mapped back
    for bad in ("/tmp/demos", "../demos", ".", "outputs/../.."):
        with pytest.raises(LaunchError, match="必須在 repo"):
            record_dir(bad)
        assert any("必須在 repo" in m for m in check_record(EXP, RecordOptions(out_dir=bad)))
    argv = build_record("isaac", EXP, RecordOptions(out_dir=os.path.join(REPO_ROOT, "outputs", "x"))).argv
    assert argv[argv.index("--record") + 1] == "outputs/x"


def test_zero_rate_is_refused_and_the_actual_rate_is_shown(capsys):
    import sim
    from agentic_sim.launch import record_rate
    assert record_rate(20) == (3, 20.0) and record_rate(25) == (2, 30.0) and record_rate(60) == (1, 60.0)
    for hz in ("0", "-5", "61", "nan"):
        assert sim.main(["--mode", "record", "--record-hz", hz, "--dry-run"]) == 2
        assert "錄製頻率" in capsys.readouterr().out
    assert sim.main(["--mode", "record", "--record-hz", "25", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "要求 25 Hz，實際 30 Hz（每 2 個控制步" in out and "--record_hz 25" in out


def test_perception_options_are_passed_only_when_asked(capsys):
    import sim
    assert sim.main(["--mode", "record", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "即時感知：關" in out and "--live-perception" not in out
    assert sim.main(["--mode", "record", "--live-perception", "--perception-camera", "wrist", "--perception-hz", "1.5",
                     "--perception-json", "outputs/p.json", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "--live-perception --perception-camera wrist --perception-hz 1.5 --perception-json outputs/p.json" in out
    assert "即時感知：開（wrist，最多 1.5 Hz，JSON outputs/p.json" in out


def test_the_launch_passes_its_run_id_to_the_recorder():
    argv = build_record("isaac", EXP, RecordOptions(run_id="r123")).argv
    assert argv[argv.index("--record_run_id") + 1] == "r123"
    assert "--record_run_id" not in build_record("isaac", EXP, RecordOptions()).argv


def test_the_launcher_gives_every_simulator_a_screen_video_control_directory(capsys):
    """Sim (both engines) and record runs carry --screen_video_dir; the launcher runs the host agent for it."""
    import sim
    assert sim.main(["--mode", "record", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "--screen_video_dir outputs/screen_video/control/" in out and "畫面錄影" in out
    for backend in ("isaac", "mujoco"):
        assert sim.main(["--backend", backend, "--experiment", "lab_20261120", "--mode", "teleop", "--direct",
                         "--dry-run"]) == 0
        assert "--screen_video_dir outputs/screen_video/control/" in capsys.readouterr().out
    argv = build_record("isaac", EXP, RecordOptions()).argv
    assert "--screen_video_dir" not in argv                 # only when the launcher gives one
