"""Export conservative, explicitly unreviewed YOLO segmentation candidates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def reset_yolo_export(export_dir: str | Path, *, protect_paths=()) -> None:
    """Remove only files listed by the previous run's own manifest, then reset it."""
    root = Path(export_dir).resolve()
    manifest = root / "manifest.jsonl"
    managed = []
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("label_path", "image_path"):
                value = row.get(key)
                if not value:
                    continue
                path = (root / value).resolve()
                if path.is_relative_to(root) and path.is_file():
                    managed.append(path)
    protected = {Path(path).resolve() for path in protect_paths}
    conflict = protected.intersection(managed)
    if conflict:
        paths = ", ".join(str(path) for path in sorted(conflict))
        raise ValueError(f"cannot reset YOLO export because an input source is managed output: {paths}")
    for path in set(managed):
        path.unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    manifest.write_text("", encoding="utf-8")


def export_yolo_candidates(frame, cv2, export_dir: str | Path, source_name: str,
                           image_rgb: np.ndarray, *, frame_tag: str | None = None) -> list[dict]:
    """Write paired images/labels; reject holes and disconnected mask islands."""
    root = Path(export_dir)
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.image_size
    rows = []
    stem = f"{Path(source_name).stem}{frame_tag or ''}"
    label_path = labels / f"{stem}.txt"
    image_path = images / f"{stem}.png"
    if not cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write candidate image: {image_path}")
    label_lines = []
    for detection in frame.detections:
        contours, hierarchy = cv2.findContours(detection.mask.astype(np.uint8), cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)
        reason = None
        if hierarchy is None or not contours:
            reason = "empty_mask"
        else:
            tree = hierarchy[0]
            roots = [i for i, node in enumerate(tree) if node[3] < 0]
            if len(roots) != 1:
                reason = "multiple_disconnected_components"
            elif any(node[3] == roots[0] for node in tree):
                reason = "mask_contains_hole"
        line_index = len(label_lines) if reason is None else None
        if reason is None:
            root_index = roots[0]
            contour = contours[root_index]
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.002), True).reshape(-1, 2)
            if len(polygon) < 3:
                reason = "polygon_has_fewer_than_three_points"
            else:
                normalized = np.column_stack((polygon[:, 0] / width, polygon[:, 1] / height))
                normalized = np.clip(normalized, 0.0, 1.0)
                coords = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
                label_lines.append(f"0 {coords}")
        row = {
            "source": source_name,
            "frame_id": frame.frame_id,
            "detection_id": detection.detection_id,
            "label_line_index": line_index,
            "color_candidate": detection.color,
            "class_id": 0,
            "label_path": str(label_path.relative_to(root)),
            "image_path": str(image_path.relative_to(root)),
            "status": "needs_review" if reason is None else "skipped",
            "reviewed": False,
            "skip_reason": reason,
            "frame_status": "incomplete_needs_review" if reason else "needs_review",
            "warning": "candidate annotation, not ground truth; inspect and correct manually before training",
        }
        rows.append(row)
    label_path.write_text("".join(line + "\n" for line in label_lines), encoding="utf-8")
    if not rows:
        rows.append({
            "source": source_name, "frame_id": frame.frame_id, "detection_id": None,
            "label_line_index": None, "color_candidate": None, "class_id": 0,
            "label_path": str(label_path.relative_to(root)),
            "image_path": str(image_path.relative_to(root)), "status": "needs_review",
            "reviewed": False, "skip_reason": None, "frame_status": "needs_review",
            "warning": "empty candidate frame, not an approved negative example",
        })
    elif any(row["status"] == "skipped" for row in rows):
        for row in rows:
            row["frame_status"] = "incomplete_needs_review"
    with manifest.open("a", encoding="utf-8") as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return rows
