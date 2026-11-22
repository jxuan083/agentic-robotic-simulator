from agentic_sim.experiments import load_experiment
from agentic_sim.launch import RunOptions, _app_args
from agentic_sim.mujoco_backend.live_perception import camera_selection
from agentic_sim.mujoco_backend.live_perception import MjLivePerception


def test_live_camera_selection_has_default_both_and_single_modes():
    assert camera_selection("both") == ("global", "wrist")
    assert camera_selection("global") == ("global",)
    assert camera_selection("wrist") == ("wrist",)


def test_launch_propagates_both_camera_and_json_output_to_each_backend():
    exp = load_experiment("lab_20261120")
    opts = RunOptions(live_perception=True, perception_camera="both", perception_hz=1.5,
                      perception_json="outputs/perception-live/latest.json")
    for backend in ("isaac", "mujoco"):
        args = _app_args(exp, opts, backend)
        assert args[args.index("--perception-camera") + 1] == "both"
        assert args[args.index("--perception-hz") + 1] == "1.5"
        assert args[args.index("--perception-json") + 1] == "outputs/perception-live/latest.json"


def test_mujoco_reset_discards_previous_camera_frame_and_advances_generation():
    manager = MjLivePerception.__new__(MjLivePerception)
    manager.specs = {"global": {}, "wrist": {}}
    manager.selected = ("global", "wrist")
    manager.generation = {"global": 2, "wrist": 2}
    manager.latest = {key: {"record": {"frame_number": 9}} for key in manager.specs}
    manager.errors = {"global": "old"}
    manager.last_capture = {key: 7.0 for key in manager.specs}
    manager.json_path = None
    manager._write_json = lambda: None
    manager.invalidate()
    assert manager.latest == {}
    assert manager.errors == {}
    assert manager.generation == {"global": 3, "wrist": 3}
    assert manager.last_capture == {}


def test_stage_delay_and_annotations_are_passed_only_when_not_default():
    import argparse
    from agentic_sim.perception_cli import add_arguments, options
    ap = argparse.ArgumentParser()
    add_arguments(ap)
    exp = load_experiment("lab_20261120")
    plain = options(ap.parse_args(["--live-perception"]))
    assert set(plain) == {"live_perception", "perception_camera", "perception_hz", "perception_json"}
    args = _app_args(exp, RunOptions(**plain), "isaac")
    assert "--perception-stage" not in args and "--annotations" not in args and "--perception-delay" not in args
    bench = options(ap.parse_args(["--live-perception", "--perception-stage", "capture", "--annotations", "off",
                                   "--perception-delay", "3"]))
    args = _app_args(exp, RunOptions(**bench), "mujoco")
    s = " ".join(args)
    assert "--perception-stage capture" in s and "--perception-delay 3.0" in s and "--annotations off" in s
    assert options(ap.parse_args(["--annotations", "off"])) == {}       # perception off: nothing is passed
