import json
import colorsys
import math
import sys
import types

import numpy as np

from agentic_sim.perception import BlockPerception, PerceptionConfig, PerceptionError, estimate_color
from agentic_sim.perception.cli import _write_frame
from agentic_sim.perception.types import Detection, FrameDetections
from agentic_sim.capture_utils import capture_output_paths, validate_capture_options, write_video_frames


def test_color_crop_is_relative_to_offset_mask_extent():
    rgb = np.full((40, 50, 3), (10, 20, 240), dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=bool)
    mask[1:13, 2:14] = True
    rgb[mask] = (240, 20, 20)
    assert estimate_color(rgb, mask) == "red"


def test_hue_boundaries_are_gapless_and_green_uses_brightness():
    def rgb_for_cv_hue(hue):
        r, g, b = colorsys.hsv_to_rgb(hue / 180, 1.0, 0.85)
        return tuple(round(v * 255) for v in (r, g, b))

    probes = [
        (10.4, "red"), (10.6, "orange"), (24.4, "orange"), (24.6, "yellow"),
        (36.4, "yellow"), (36.6, "light_green"), (84.8, "light_green"),
        (85.2, "cyan"), (99.4, "cyan"), (99.6, "blue"), (132.4, "blue"),
        (132.6, "purple"), (169.8, "purple"), (170.2, "red"),
    ]
    for hue, expected in probes:
        rgb = np.empty((12, 12, 3), dtype=np.uint8)
        rgb[:] = rgb_for_cv_hue(hue)
        assert estimate_color(rgb, np.ones((12, 12), dtype=bool)) == expected, hue
    dark = np.full((12, 12, 3), rgb_for_cv_hue(55), dtype=np.uint8)
    light = np.full((12, 12, 3), rgb_for_cv_hue(55), dtype=np.uint8)
    # Preserve hue and saturation while selecting the configured brightness split.
    dark[:] = tuple(round(v * 255) for v in colorsys.hsv_to_rgb(55 / 180, 1, .35))
    light[:] = tuple(round(v * 255) for v in colorsys.hsv_to_rgb(55 / 180, 1, .85))
    mask = np.ones((12, 12), dtype=bool)
    assert estimate_color(dark, mask) == "dark_green"
    assert estimate_color(light, mask) == "light_green"


def test_color_estimation_uses_mask_and_rejects_gray():
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:16, 4:16] = True
    rgb[mask] = (240, 20, 20)
    rgb[0:2, 0:2] = (20, 20, 240)  # outside mask must not affect result
    assert estimate_color(rgb, mask) == "red"
    rgb[mask] = (120, 120, 120)
    assert estimate_color(rgb, mask) == "unknown"


def test_empty_model_output_and_rgb_bgr_contract():
    class Empty:
        task = "segment"
        def predict(self, image, **kwargs):
            assert tuple(image[0, 0]) == (3, 2, 1)
            return [type("Result", (), {"masks": None, "boxes": None})()]

    detector = BlockPerception(PerceptionConfig(), model=Empty())
    frame = detector.predict(np.array([[[1, 2, 3]]], dtype=np.uint8), frame_id=2, timestamp_s=1.25)
    assert frame.detections == []
    assert frame.image_size == (1, 1)
    assert frame.to_dict()["detections"] == []


def test_json_contract_excludes_mask_array_and_ids_are_separate():
    mask = np.ones((2, 3), dtype=bool)
    frame = FrameDetections("video-a:4", 0.5, (2, 3), [
        Detection(0, 19, "block", .9, "blue", (0., 0., 3., 2.), (1., .5), mask),
        Detection(1, None, "block", .7, "unknown", (1., 0., 3., 2.), (2., .5), mask),
    ])
    data = frame.to_dict({0: "a.png", 1: "b.png"})
    assert data["detections"][0]["detection_id"] == 0
    assert data["detections"][0]["track_id"] == 19
    assert data["detections"][1]["track_id"] is None
    assert data["detections"][0]["mask_path"] == "a.png"
    assert "mask" not in data["detections"][0]
    json.dumps(data)


def test_model_mask_is_restored_and_detection_ids_are_frame_local():
    class Boxes:
        xyxy = np.array([[1., 1., 5., 5.]])
        conf = np.array([.95])
        cls = np.array([0])
        id = None

    class Model:
        task = "segment"
        def predict(self, image, **kwargs):
            result = type("Result", (), {
                "boxes": Boxes(), "masks": type("Masks", (), {"data": np.ones((1, 6, 8))})(),
            })()
            return [result]

    detector = BlockPerception(PerceptionConfig(), model=Model())
    rgb = np.full((6, 8, 3), (20, 20, 240), dtype=np.uint8)
    a = detector.predict(rgb, frame_id=0)
    b = detector.predict(rgb, frame_id=1)
    assert a.detections[0].mask.shape == rgb.shape[:2]
    assert a.detections[0].centroid_uv == (3.5, 2.5)
    assert a.detections[0].detection_id == b.detections[0].detection_id == 0
    assert a.detections[0].track_id is None


def test_non_native_masks_are_rejected_instead_of_stretched():
    class Boxes:
        xyxy = np.array([[1., 1., 5., 5.]])
        conf = np.array([.95])
        cls = np.array([0])
        id = None

    class Model:
        task = "segment"
        def predict(self, image, **kwargs):
            result = type("Result", (), {
                "boxes": Boxes(), "masks": type("Masks", (), {"data": np.ones((1, 2, 2))})(),
            })()
            return [result]

    detector = BlockPerception(PerceptionConfig(), model=Model())
    try:
        detector.predict(np.zeros((6, 8, 3), dtype=np.uint8))
    except PerceptionError as exc:
        assert "instead of original image size" in str(exc)
    else:
        raise AssertionError("non-native mask size must be rejected")


def test_track_id_is_from_tracker_and_reset_preserves_ultralytics_path_list(monkeypatch):
    monkeypatch.setitem(sys.modules, "lap", types.ModuleType("lap"))
    class Boxes:
        xyxy = np.array([[1., 1., 5., 5.]])
        conf = np.array([.95])
        cls = np.array([0])
        id = np.array([41])

    class Tracker:
        def __init__(self): self.reset_count = 0
        def reset(self): self.reset_count += 1

    class Model:
        task = "segment"
        predictor = type("Predictor", (), {"trackers": [Tracker()], "vid_path": ["clip.mp4"]})()
        def track(self, image, **kwargs):
            result = type("Result", (), {
                "boxes": Boxes(), "masks": type("Masks", (), {"data": np.ones((1, 6, 8))})(),
            })()
            return [result]

    model = Model()
    detector = BlockPerception(PerceptionConfig(), model=model)
    a = detector.predict(np.zeros((6, 8, 3), dtype=np.uint8), frame_id=0, track=True)
    b = detector.predict(np.zeros((6, 8, 3), dtype=np.uint8), frame_id=1, track=True)
    assert a.detections[0].track_id == b.detections[0].track_id == 41
    assert a.detections[0].detection_id == b.detections[0].detection_id == 0
    detector.reset_tracker()
    assert model.predictor.trackers[0].reset_count == 1
    assert model.predictor.vid_path == [None]


def test_local_segmentation_checkpoint_loads_once_without_downloading(tmp_path):
    checkpoint = tmp_path / "block.pt"
    checkpoint.write_bytes(b"local test checkpoint")
    calls = []

    class LocalYOLO:
        task = "segment"
        def __init__(self, path):
            calls.append(path)
        def predict(self, image, **kwargs):
            return [type("Result", (), {"masks": None, "boxes": None})()]

    old = {name: sys.modules.get(name) for name in (
        "ultralytics", "ultralytics.utils", "ultralytics.utils.checks")}
    root = types.ModuleType("ultralytics")
    root.__path__ = []
    root.YOLO = LocalYOLO
    utils = types.ModuleType("ultralytics.utils")
    utils.__path__ = []
    utils.AUTOINSTALL = True
    checks = types.ModuleType("ultralytics.utils.checks")
    checks.AUTOINSTALL = True
    sys.modules.update({"ultralytics": root, "ultralytics.utils": utils,
                        "ultralytics.utils.checks": checks})
    try:
        config = PerceptionConfig(model_path=str(checkpoint))
        detector = BlockPerception(config)
        detector.predict(np.zeros((2, 2, 3), dtype=np.uint8))
        detector.predict(np.zeros((2, 2, 3), dtype=np.uint8))
        assert calls == [str(checkpoint)]
        assert utils.AUTOINSTALL is checks.AUTOINSTALL is False
    finally:
        for name, module in old.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_missing_checkpoint_and_detection_only_model_fail_clearly(tmp_path):
    try:
        BlockPerception(PerceptionConfig(model_path=str(tmp_path / "absent.pt")))
    except PerceptionError as exc:
        assert "checkpoint not found" in str(exc)
    else:
        raise AssertionError("missing local weights must fail without automatic download")
    try:
        BlockPerception(PerceptionConfig(), model=type("DetectOnly", (), {"task": "detect"})())
    except PerceptionError as exc:
        assert "not an Ultralytics segmentation model" in str(exc)
    else:
        raise AssertionError("detection-only models must be rejected")


def test_cli_writes_rgb_correct_overlay_masks_and_metadata(tmp_path):
    class CV2:
        COLOR_RGB2BGR = 1
        FONT_HERSHEY_SIMPLEX = 0
        def __init__(self, fail=False):
            self.writes = []
            self.fail = fail
        def imwrite(self, path, image):
            self.writes.append((path, image.copy()))
            return not self.fail
        def cvtColor(self, image, code):
            assert code == self.COLOR_RGB2BGR
            return image[..., ::-1]
        def rectangle(self, *args, **kwargs): pass
        def putText(self, *args, **kwargs): pass

    mask = np.ones((3, 4), dtype=bool)
    frame = FrameDetections(0, 0.0, (3, 4), [
        Detection(0, None, "block", .8, "red", (0., 0., 4., 3.), (1.5, 1.), mask),
    ])
    cv2 = CV2()
    _write_frame(frame, tmp_path, np.zeros((3, 4, 3), dtype=np.uint8), cv2, jsonl=False)
    overlay_write = next(image for path, image in cv2.writes if path.endswith("overlay.jpg"))
    # Selection color is RGB red-ish; after RGB->BGR for imwrite it must be blue channel last.
    assert tuple(overlay_write[1, 1]) == (16, 16, 51)
    record = json.loads((tmp_path / "frame_0.json").read_text())
    assert record["detections"][0]["mask_path"].endswith("frame_0_det_0_mask.png")
    failing = CV2(fail=True)
    try:
        _write_frame(frame, tmp_path, np.zeros((3, 4, 3), dtype=np.uint8), failing, jsonl=False)
    except OSError as exc:
        assert "failed to write mask" in str(exc)
    else:
        raise AssertionError("failed cv2.imwrite must not be ignored")


def test_predict_track_mode_switch_keeps_detections_and_callbacks_restored(monkeypatch):
    def on_predict_start(*args): pass
    def on_predict_postprocess_end(*args): pass
    on_predict_start.__module__ = "ultralytics.trackers.track"
    on_predict_postprocess_end.__module__ = "ultralytics.trackers.track"
    callbacks = {"on_predict_start": [on_predict_start],
                 "on_predict_postprocess_end": [on_predict_postprocess_end]}

    class Boxes:
        xyxy = np.array([[1., 1., 5., 5.]])
        conf = np.array([.95])
        cls = np.array([0])
        id = np.array([73])
    class Model:
        task = "segment"
        def __init__(self):
            self.predictor = type("Predictor", (), {"callbacks": callbacks})()
            self.callbacks = callbacks
        def _result(self, tracking):
            # Models on older Ultralytics versions may filter detections unless the
            # tracking callback is suspended around the plain predict call.
            if not tracking and self.callbacks["on_predict_postprocess_end"]:
                return type("Result", (), {"boxes": type("Empty", (), {
                    "xyxy": np.empty((0, 4)), "conf": np.empty(0), "cls": np.empty(0, int), "id": None})(),
                    "masks": type("Masks", (), {"data": np.empty((0, 6, 8))})()})()
            return type("Result", (), {"boxes": Boxes(),
                "masks": type("Masks", (), {"data": np.ones((1, 6, 8))})()})()
        def predict(self, image, **kwargs): return [self._result(False)]
        def track(self, image, **kwargs): return [self._result(True)]

    monkeypatch.setitem(sys.modules, "lap", types.ModuleType("lap"))
    model = Model()
    detector = BlockPerception(PerceptionConfig(), model=model)
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    first = detector.predict(rgb, track=True)
    plain = detector.predict(rgb)
    third = detector.predict(rgb, track=True)
    assert (first.frame_id, plain.frame_id, third.frame_id) == (0, 1, 2)
    assert first.detections[0].track_id == third.detections[0].track_id == 73
    assert plain.detections and plain.detections[0].track_id is None
    assert model.callbacks["on_predict_postprocess_end"] == [on_predict_postprocess_end]


def test_byte_track_reports_missing_lap_without_autoinstall(monkeypatch):
    monkeypatch.setitem(sys.modules, "lap", None)
    class Model:
        task = "segment"
        def track(self, *args, **kwargs): raise AssertionError("must fail before tracker call")
    detector = BlockPerception(PerceptionConfig(), model=Model())
    try:
        detector.predict(np.zeros((2, 2, 3), dtype=np.uint8), track=True)
    except PerceptionError as exc:
        assert "ByteTrack requires the `lap` package" in str(exc)
    else:
        raise AssertionError("missing lap should be an actionable setup error")


def test_capture_options_paths_and_video_writer_cleanup(tmp_path):
    for args in ((0, 16., 10, 10), (1, 0., 10, 10), (1, math.inf, 10, 10),
                 (1, 16., 0, 10), (1, 16., 10, -1), (True, 16., 10, 10)):
        try:
            validate_capture_options(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid capture options accepted: {args}")
    validate_capture_options(1, 16., 10, 10)
    output = tmp_path / "capture"
    output.mkdir()
    sentinel = output / "user-image.png"
    sentinel.write_bytes(b"keep")
    capture_output_paths(output)
    assert sentinel.read_bytes() == b"keep"

    class Writer:
        def __init__(self, opened=True, fail_write=False):
            self.opened, self.fail_write, self.released = opened, fail_write, False
        def isOpened(self): return self.opened
        def write(self, frame):
            if self.fail_write: raise RuntimeError("write failed")
        def release(self): self.released = True
    class CV2:
        def __init__(self, writer): self.writer = writer
        def VideoWriter_fourcc(self, *args): return 0
        def VideoWriter(self, *args): return self.writer
    for writer in (Writer(opened=False), Writer(fail_write=True)):
        try:
            write_video_frames(CV2(writer), output / "x.mp4", 16., 10, 10, [object()])
        except RuntimeError:
            pass
        else:
            raise AssertionError("failed writer should raise")
        assert writer.released

