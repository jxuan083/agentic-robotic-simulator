import json

import cv2
import numpy as np

from agentic_sim.perception.annotations import export_yolo_candidates, reset_yolo_export
from agentic_sim.perception.classical import ClassicalBlockPerception, ClassicalConfig
from agentic_sim.perception.types import Detection, FrameDetections


def _scene(red_x=30, blue_x=70, include_red=True, height=100, width=140):
    rgb = np.full((height, width, 3), 128, np.uint8)
    if include_red:
        rgb[30:60, red_x:red_x + 25] = (240, 20, 20)
    rgb[30:60, blue_x:blue_x + 25] = (20, 40, 240)
    return rgb


def _config(**kwargs):
    values = dict(min_area_px=60, min_side_px=5, close_kernel=3, open_kernel=1,
                  max_area_fraction=.5)
    values.update(kwargs)
    return ClassicalConfig(**values)


def test_classical_finds_nearby_different_colors_and_does_not_merge_with_default_close():
    detector = ClassicalBlockPerception(_config())
    rgb = np.full((90, 140, 3), 128, np.uint8)
    rgb[25:60, 20:55] = (240, 20, 20)
    rgb[25:60, 59:94] = (20, 40, 240)  # four-pixel gap
    frame = detector.predict(rgb)
    assert len(frame.detections) == 2
    assert [d.color for d in frame.detections] == ["red", "blue"]
    assert all(d.mask.shape == rgb.shape[:2] for d in frame.detections)
    assert detector.config.close_kernel == 3


def test_repository_classical_yaml_uses_verified_small_close_kernel():
    config = ClassicalConfig.from_yaml("configs/perception-classical.yaml")
    assert config.close_kernel == 3
    assert ClassicalBlockPerception(config).config.close_kernel == 3


def test_roi_excludes_colorful_background_and_morphology_stays_inside_roi():
    rgb = _scene()
    rgb[0:8, 0:8] = (20, 240, 20)
    detector = ClassicalBlockPerception(_config(roi_xyxy_normalized=(.15, .15, .85, .85),
                                                close_kernel=11))
    frame = detector.predict(rgb)
    assert len(frame.detections) == 2
    assert all(d.bbox_xyxy[0] >= 21 and d.bbox_xyxy[2] <= 119 for d in frame.detections)
    assert not any(d.mask[:15].any() or d.mask[:, :21].any() or d.mask[:, 119:].any()
                   for d in frame.detections)


def test_color_position_tracker_matches_ids_and_expires_without_reuse():
    detector = ClassicalBlockPerception(_config(track_max_distance_px=20, track_max_missed_frames=1))
    a = detector.predict(_scene(30, 75), track=True)
    ids = {d.color: d.track_id for d in a.detections}
    b = detector.predict(_scene(35, 70), track=True)
    assert {d.color: d.track_id for d in b.detections} == ids
    detector.predict(_scene(0, 70, include_red=False), track=True)
    detector.predict(_scene(0, 70, include_red=False), track=True)
    c = detector.predict(_scene(40, 70), track=True)
    red_new = next(d.track_id for d in c.detections if d.color == "red")
    assert red_new != ids["red"]
    detector.reset_tracker()
    d = detector.predict(_scene(40, 70), track=True)
    assert min(item.track_id for item in d.detections) > max(ids.values())


def test_export_aggregates_frame_labels_saves_rgb_and_marks_skips(tmp_path):
    rgb = _scene(20, 80)
    mask1 = np.zeros(rgb.shape[:2], bool); mask1[30:60, 20:45] = True
    mask2 = np.zeros(rgb.shape[:2], bool); mask2[30:60, 80:105] = True
    frame = FrameDetections(0, None, rgb.shape[:2], [
        Detection(0, None, "block", .8, "red", (20., 30., 45., 60.), (32., 44.), mask1),
        Detection(1, None, "block", .8, "blue", (80., 30., 105., 60.), (92., 44.), mask2),
    ])
    export = tmp_path / "export"
    reset_yolo_export(export)
    rows = export_yolo_candidates(frame, cv2, export, "scene.png", rgb)
    label = export / "labels/scene.txt"
    image = cv2.imread(str(export / "images/scene.png"), cv2.IMREAD_COLOR)
    assert len(label.read_text().splitlines()) == 2
    assert all(row["label_line_index"] in (0, 1) and row["status"] == "needs_review" for row in rows)
    assert np.array_equal(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), rgb)
    coords = [float(v) for v in label.read_text().splitlines()[0].split()[1:]]
    assert all(0 <= value <= 1 for value in coords)

    # A ring has a hole and cannot safely become one simple YOLO polygon.
    ring = np.zeros(rgb.shape[:2], bool); ring[10:40, 10:40] = True; ring[18:32, 18:32] = False
    frame.detections = [Detection(0, None, "block", .8, "red", (10., 10., 40., 40.),
                                  (20., 20.), ring)]
    rows = export_yolo_candidates(frame, cv2, export, "scene.png", rgb)
    assert rows[0]["status"] == "skipped"
    assert rows[0]["skip_reason"] == "mask_contains_hole"
    assert rows[0]["frame_status"] == "incomplete_needs_review"
    assert (export / "labels/scene.txt").read_text() == ""
    unrelated = export / "notes.txt"; unrelated.write_text("keep")
    reset_yolo_export(export)
    assert unrelated.read_text() == "keep"
    assert not (export / "images/scene.png").exists()
    assert not (export / "labels/scene.txt").exists()
    manifest_rows = [json.loads(line) for line in (export / "manifest.jsonl").read_text().splitlines()]
    assert manifest_rows == []


def test_export_reset_refuses_to_delete_input_source(tmp_path):
    export = tmp_path / "export"
    reset_yolo_export(export)
    source = export / "images/frame.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"input")
    (export / "manifest.jsonl").write_text(json.dumps({"image_path": "images/frame.png"}) + "\n")
    try:
        reset_yolo_export(export, protect_paths=(source,))
    except ValueError as exc:
        assert "input source is managed output" in str(exc)
    else:
        raise AssertionError("export reset must preserve a source it would otherwise delete")
    assert source.read_bytes() == b"input"
