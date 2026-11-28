"""Exercise the episode path through fake devices, including policy handover."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from struct import pack
import subprocess
import sys
from types import SimpleNamespace
from zlib import compress

import pytest

from agentic_sim.real_control import JOINTS, ControlRouter, open_pairs
from agentic_sim.real_episode import Episode, EpisodeWriter, PNG_SIGNATURE
from agentic_sim.real_fake import FakeRig, _chunk, fake_pose, record_fake


ROOT = Path(__file__).resolve().parents[1]
TELEOP = ROOT / "tools" / "real_teleop.py"


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("arms", [("right",), ("left", "right")])
def test_named_devices_receive_only_their_commands(tmp_path, arms):
    target = tmp_path / "sample"
    trace = record_fake(target, arms, {"test": True})
    episode = Episode(target)
    report = episode.validate()
    assert report["ok"] and report["arms"] == list(arms)
    assert report["control_frames"] == 12
    assert report["warnings"] == []
    assert report["control_timing"]["configured_hz"] == 20
    assert report["control_timing"]["expected_period_ns"] == 50_000_000
    assert report["control_timing"]["max_interval"]["interval_ns"] == 50_000_000
    assert trace["opened"] == [[role, name] for name in arms for role in ("leader", "follower")]
    assert trace["closed"] == list(reversed(trace["opened"]))
    assert trace["commands"] == [
        {"arm": name, "command": row["arms"][name]["command"]}
        for row in episode.rows for name in arms
    ]
    if len(arms) == 2:
        assert set(episode.frames) == {"global", "left_wrist", "right_wrist"}
        assert episode.rows[3]["arms"]["left"]["command"]["shoulder_pan"] != (
            episode.rows[3]["arms"]["right"]["command"]["shoulder_pan"])


def test_takeover_drops_queued_and_late_policy_actions(tmp_path):
    target = tmp_path / "sample"
    trace = record_fake(target, ["right"], {})
    episode = Episode(target)
    rows = episode.rows
    commands = [row["command"]["shoulder_pan"] for row in trace["commands"]]
    assert commands == [2, 4, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
    assert trace["stale_chunk_accepted"] is False
    assert [row["task_is_policy"] for row in rows] == [1, 1] + [0] * 10
    assert rows[2]["arms"]["right"]["command"] == rows[1]["arms"]["right"]["readback"]
    event, = episode.validate()["takeovers"]
    assert (event["frame"], event["timestamp_ns"], event["discarded_policy_actions"]) == (2, 103_000_000, 2)
    assert event["switch_ns"] == 100_000_000
    assert event["aligned_readback"]["right"] == rows[2]["arms"]["right"]["command"]


def test_route_rejects_an_unselected_arm_and_unwinds_partial_open():
    rig = FakeRig()
    arms = {name: {"leader": {}, "follower": {}} for name in ("left", "right")}

    @contextmanager
    def fail_follower(name, spec):
        if name == "right":
            raise OSError("right follower unavailable")
        with rig.open_follower(name, spec) as device:
            yield device

    with pytest.raises(OSError, match="right follower"):
        with open_pairs(arms, rig.open_leader, fail_follower):
            pytest.fail("partial rig was armed")
    assert rig.trace["closed"] == list(reversed(rig.trace["opened"]))
    assert not rig.trace["commands"]

    with open_pairs({"right": arms["right"]}, rig.open_leader, rig.open_follower) as pairs:
        router = ControlRouter(pairs, start="policy")
        with pytest.raises(ValueError, match="precisely"):
            router.enqueue_policy([{"left": fake_pose()}], generation=0)
    assert not rig.trace["commands"]


def test_read_cli_on_relocated_sample_without_site_packages(tmp_path):
    target = tmp_path / "original" / "sample"
    record_fake(target, ["right"], {})
    moved = tmp_path / "different-machine" / "sample"
    shutil.move(target, moved)
    command = [sys.executable, "-S", str(TELEOP), "read", str(moved), "--html"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["ok"] and report["source"] == "fake"
    html = (moved / "replay.html").read_text()
    assert "global/000002.png" in html and "wrist/000002.png" in html
    assert all(name in html for name in JOINTS)
    assert str(target) not in html
    (moved / "wrist" / "000002.png").unlink()
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert any(issue["stream"] == "wrist" and issue["frame"] == 2
               for issue in json.loads(result.stdout)["issues"])


def test_reader_reports_missing_camera_index_and_frame_number(tmp_path):
    target = tmp_path / "sample"
    record_fake(target, ["right"], {})
    index = target / "global" / "frames.jsonl"
    rows = index.read_text().splitlines()
    index.write_text("\n".join(rows[:4] + rows[5:]) + "\n")
    (target / "wrist" / "frames.jsonl").unlink()
    report = Episode(target).validate()
    assert not report["ok"]
    assert any(issue["stream"] == "wrist" and "index unreadable" in issue["reason"] for issue in report["issues"])
    assert any(issue["stream"] == "global" and issue["frame"] == 4 for issue in report["issues"])


@pytest.mark.parametrize("swap", [False, True])
def test_read_cli_rejects_wrong_rgb_filename_even_when_png_is_valid(tmp_path, swap):
    target = tmp_path / "sample"
    record_fake(target, ["right"], {})
    rows = Episode(target).frames["wrist"]
    if swap:
        rows[1]["rgb"] = "000002.png"
    rows[2]["rgb"] = "000001.png"
    _write_rows(target / "wrist" / "frames.jsonl", rows)

    result = subprocess.run([sys.executable, "-S", str(TELEOP), "read", str(target)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert not report["ok"]
    mismatches = [issue for issue in report["issues"] if issue.get("image") == "rgb"]
    assert [(issue["stream"], issue["frame"], issue["expected"], issue["actual"]) for issue in mismatches] == (
        ([("wrist", 1, "000001.png", "000002.png")] if swap else [])
        + [("wrist", 2, "000002.png", "000001.png")]
    )
    duplicates = [issue for issue in report["issues"] if "duplicate image reference" in issue["reason"]]
    assert [issue["frame"] for issue in duplicates] == ([] if swap else [2])


def test_reader_rejects_wrong_depth_filename_with_correct_rgb(tmp_path):
    record_fake(tmp_path / "rgb", ["right"], {})
    original = Episode(tmp_path / "rgb")
    manifest = original.manifest
    cameras = manifest["cameras"]
    cameras["wrist"]["depth"] = True
    width, height = cameras["wrist"]["width"], cameras["wrist"]["height"]
    depth_header = _chunk(b"IHDR", pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0))
    writer = EpisodeWriter(tmp_path / "rgbd", source="fake", arms=manifest["arms"],
                           cameras=cameras, provenance={}, hz=20)
    for row in original.rows:
        writer.add_control(row)
    for camera, frames in original.frames.items():
        for frame in frames:
            pixels = (b"\x00" + pack(">H", frame["frame"] + 1) * width) * height
            depth = PNG_SIGNATURE + depth_header + _chunk(b"IDAT", compress(pixels)) + _chunk(b"IEND", b"")
            rgb = (original.root / camera / frame["rgb"]).read_bytes()
            writer.add_image(camera, rgb, frame["capture_ns"], depth=depth if camera == "wrist" else None)
    writer.close()
    episode = Episode(writer.root)
    assert episode.validate()["ok"]
    rows = episode.frames["wrist"]
    rows[2]["depth"] = "000001_depth.png"
    _write_rows(episode.root / "wrist" / "frames.jsonl", rows)

    report = Episode(episode.root).validate()
    assert not report["ok"]
    mismatch, = [issue for issue in report["issues"] if issue.get("image")]
    assert (mismatch["stream"], mismatch["frame"], mismatch["image"]) == ("wrist", 2, "depth")
    assert (mismatch["expected"], mismatch["actual"]) == ("000002_depth.png", "000001_depth.png")


def test_read_cli_warns_about_an_omitted_and_renumbered_control_row(tmp_path):
    target = tmp_path / "sample"
    record_fake(target, ["right"], {})
    episode = Episode(target)
    rows = episode.rows
    del rows[5]
    for index, row in enumerate(rows):
        row["frame"] = index
    _write_rows(target / "control.jsonl", rows)
    episode.manifest["control_frames"] = len(rows)
    (target / "manifest.json").write_text(json.dumps(episode.manifest))

    result = subprocess.run([sys.executable, "-S", str(TELEOP), "read", str(target), "--html"],
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    interval = {"before_frame": 4, "before_ns": 203_000_000, "after_frame": 5, "after_ns": 303_000_000,
                "interval_ns": 100_000_000}
    assert report["ok"] and not report["issues"]
    assert report["control_timing"] == {
        "configured_hz": 20, "expected_period_ns": 50_000_000,
        "gap_threshold_ns": 75_000_000, "max_interval": interval,
    }
    warning, = report["warnings"]
    assert warning["stream"] == "control" and warning["frame"] == 5
    assert {key: warning[key] for key in interval} == interval
    assert (target / "replay.html").is_file()


@pytest.mark.parametrize("adjustment_ns", [2_000_000, -50_000_000, -51_000_000])
def test_reader_handles_jitter_and_still_rejects_non_increasing_control_time(tmp_path, adjustment_ns):
    target = tmp_path / "sample"
    record_fake(target, ["right"], {})
    rows = Episode(target).rows
    rows[5]["timestamp_ns"] += adjustment_ns
    _write_rows(target / "control.jsonl", rows)

    report = Episode(target).validate()
    if adjustment_ns > 0:
        assert report["ok"] and not report["warnings"]
        assert report["control_timing"]["max_interval"]["interval_ns"] == 52_000_000
    else:
        assert not report["ok"]
        assert any(issue["frame"] == 5 and "non-increasing timestamp" in issue["reason"]
                   for issue in report["issues"])


def test_explicit_physical_replay_rejects_fake_without_drivers(tmp_path):
    target = tmp_path / "sample"
    record_fake(target, ["right"], {})
    result = subprocess.run([sys.executable, "-S", str(ROOT / "tools/replay_real_follower.py"),
                             str(target), "--config", "missing.yaml", "--arm", "right", "--allow-motion"],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "physical replay requires a real capture" in result.stderr


def test_startup_goal_failure_still_disengages_every_follower_motor():
    from agentic_sim.real_hardware import Follower

    class Bus:
        model_resolution_table = {"sts3215": 4096}
        model_ctrl_table = {"sts3215": {"Torque_Enable": (40, 1)}}
        calibration = dict.fromkeys(JOINTS, SimpleNamespace(range_min=500, range_max=3600))
        motors = {j: SimpleNamespace(id=i, model="sts3215") for i, j in enumerate(JOINTS, 1)}
        port_handler = None

        def __init__(self):
            self.torque = 0
            self.packet_handler = SimpleNamespace(read1ByteTxRx=self.torque_status)

        def sync_read(self, *_args, **_kwargs):
            return dict.fromkeys(JOINTS, 2048)

        def sync_write(self, register, action, **_kwargs):
            if register == "Goal_Position":
                self.torque = 1  # Device behavior observed in the lab on Goal_Position writes.
            elif register == "Torque_Enable":
                assert action == dict.fromkeys(JOINTS, 0)
                self.torque = 0

        def read(self, _register, joint, **_kwargs):
            return 2047 if joint == "gripper" else 2048

        def torque_status(self, _port, _motor, _address):
            return self.torque, 0, 32  # valid readback even with a latched overload flag

        def _is_comm_success(self, status):
            return status == 0

    bus = Bus()
    follower = Follower(bus)
    with pytest.raises(RuntimeError, match="startup target"):
        follower.arm()
    assert bus.torque == 1
    follower.close()
    assert bus.torque == 0
