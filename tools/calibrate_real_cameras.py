#!/usr/bin/env python3
"""Capture camera-only calibration data and fit it offline."""

import argparse
import json
from pathlib import Path
import sys
from time import monotonic_ns, time_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from agentic_sim.camera_calibration import (
    MARKERS, charuco_board, compare_workspace, coverage_regions, depth_holdout,
    detect_board, dictionary, find_markers, fit_wrist, measurements, read_png, save_png,
    workspace_geometry,
)
from agentic_sim.real_episode import Episode, stream_name, write_json


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def record_measurements(root, args):
    target = root / "measurements.json"
    if target.exists():
        raise ValueError(f"measurements already exist: {target}; use a new folder for changed print or layout")
    labels = ("第 1 頁板面全寬", "第 1 頁板面全高", "板上一個 marker 黑框邊長",
              "ID 40 黑框邊長", "ID 41 黑框邊長", "ID 42 黑框邊長",
              "第 1 頁 100 mm 檢查線實長", "第 2 頁 100 mm 檢查線實長")
    values = args.mm if args.mm else [float(input(f"{label}（mm）：")) for label in labels]
    result = measurements(args.layout, values, args.precision_mm)
    root.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    print(target)


def _capture_spec(config, camera):
    from agentic_sim.real_hardware import load_profile

    spec = load_profile(config)["cameras"][camera]
    if camera == "global" and (spec["backend"] != "realsense" or not spec.get("depth")):
        raise ValueError("global reference requires the formal RealSense RGB-D profile")
    return spec


def _draw_grid(image, covered):
    for row in range(3):
        for col in range(3):
            x, y = 8 + 14 * col, 24 + 14 * row
            color = (0, 200, 0) if (row, col) in covered else (0, 128, 255)
            cv2.rectangle(image, (x, y), (x + 10, y + 10), color, -1)


def _global_overlay(image, corners):
    overlay = image.copy()
    for marker, points in corners.items():
        cv2.polylines(overlay, [points.round().astype(np.int32)], True, (0, 220, 0), 2, cv2.LINE_AA)
        top = tuple(points[0].round().astype(int))
        cv2.putText(overlay, str(marker), (top[0], max(16, top[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 220, 0), 2, cv2.LINE_AA)
        for index, (x, y) in enumerate(points.round().astype(int)):
            cv2.circle(overlay, (x, y), 3, (20, 230, 255), -1)
            cv2.putText(overlay, str(index), (x + 4, y + 3), cv2.FONT_HERSHEY_SIMPLEX, .35, (20, 230, 255), 1)
    return overlay


def _key_for(name, image):
    cv2.imshow(name, image)
    key = cv2.waitKey(15) & 0xff
    return ord("q") if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 else key


def _require_gui():
    if any("GUI:" in line and "NONE" in line for line in cv2.getBuildInformation().splitlines()):
        raise RuntimeError("use a GUI-enabled OpenCV environment for capture; fitting can run headlessly")


def _write_global(output, image, depth, timing, stamp, info, camera, measured, links, compare):
    if image.shape != (camera["height"], camera["width"], 3):
        raise ValueError("global capture does not match the configured native RGB size")
    if depth is None or depth.dtype != np.uint16 or depth.shape != image.shape[:2]:
        raise ValueError("global depth must be a matching native 16-bit frame")
    record = dict(measurements=measured, camera=camera | info, device_timing=timing,
                  host_monotonic_ns=stamp, host_unix_ns=time_ns(), layout_id=measured["layout_id"],
                  files={"color": "rgb.png", "depth": "depth.png", "annotation": "markers.png"})
    output.mkdir(parents=True, exist_ok=False)
    save_png(output / "rgb.png", image)
    save_png(output / "depth.png", depth)
    write_json(output / "reference.json", record)
    try:
        record |= workspace_geometry(image, info, measured, links=links)
    except (KeyError, ValueError, cv2.error) as error:
        record["geometry_error"] = str(error)
        write_json(output / "reference.json", record)
        raise
    corners = {int(marker): np.array(row["corners_px"]) for marker, row in record["markers"].items()}
    save_png(output / "markers.png", _global_overlay(image, corners))
    write_json(output / "reference.json", record)
    if compare:
        write_json(output / "comparison.json", compare_workspace(load(compare / "reference.json"), record))
    return record


def capture_global(root, args, measured):
    output = root / stream_name(args.name)
    if output.exists():
        raise ValueError(f"capture already exists: {output}")
    links = load(args.links) if args.links else None
    spec = _capture_spec(args.config, "global")
    if args.episode:
        episode = Episode(args.episode)
        if not episode.validate()["ok"] or episode.manifest["source"] != "real":
            raise ValueError("choose a complete real episode with a valid disk readback")
        meta = episode.manifest["cameras"]["global"]
        if any(meta[key] != spec[key] for key in ("width", "height", "fps", "calibration")):
            raise ValueError("global recording differs from the supplied camera profile")
        with (episode.root / "motion.jsonl").open(encoding="utf-8") as handle:
            ready = next((row for line in handle if (row := json.loads(line))["event"] == "task_ready"), None)
        if ready is None:
            raise ValueError("episode has no settled standard-start event")
        stamp = ready["timestamp_ns"]
        frame = min(episode.frames["global"], key=lambda row: abs(row["capture_ns"] - stamp))
        if abs(frame["capture_ns"] - stamp) > 1e9 / meta["fps"]:
            raise ValueError("no global RGB-D frame close to the settled standard start")
        image = read_png(episode.root / "global" / frame["rgb"])
        depth = read_png(episode.root / "global" / frame["depth"])
        provenance = {"source_episode_id": episode.manifest["episode_id"], "global_frame": frame["frame"],
                      "capture_from_episode_start_ns": frame["capture_ns"], "task_ready_from_episode_start_ns": stamp,
                      "serial_source": "supplied camera profile matching the episode calibration"}
        _write_global(output, image, depth, frame.get("device_timing"), None, meta,
                      spec | {"capture_provenance": provenance}, measured, links, args.compare)
        print(output / "reference.json")
        return

    from agentic_sim.real_hardware import open_camera

    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary(), parameters)
    name = "D455 reference | S save | Q quit"
    _require_gui()
    try:
        with open_camera(spec) as (read, info):
            while True:
                image, depth, timing = read()
                stamp = monotonic_ns()
                try:
                    corners = find_markers(image, detector)
                    missing = sorted(set(MARKERS) - corners.keys())
                    status = f"Missing {missing}" if missing else "40 / 41 / 42 visible. S: save native RGB + depth"
                except ValueError as error:
                    missing, corners, status = list(MARKERS), {}, str(error)
                overlay = _global_overlay(image, corners)
                cv2.putText(overlay, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
                key = _key_for(name, overlay)
                if key in (27, ord("q")):
                    return
                if key == ord("s") and not missing:
                    poses = _write_global(output, image, depth, timing, stamp, info, spec, measured, links, args.compare)
                    print(output / "reference.json")
                    if poses["base_from_rgb_m"] is None:
                        print("saved relative tray positions; measured base-from-marker40 is pending")
                    return
    finally:
        cv2.destroyAllWindows()


def capture_wrist(root, args, measured):
    from agentic_sim.real_hardware import open_camera

    output = root / "wrist"
    spec = _capture_spec(args.config, "wrist")
    board = charuco_board(measured)
    detector = cv2.aruco.CharucoDetector(board)
    _require_gui()
    if args.sharpness <= 0:
        raise ValueError("sharpness threshold must be positive")
    (output / "frames").mkdir(parents=True, exist_ok=True)
    manifest = output / "observations.json"
    size = spec["width"], spec["height"]
    if manifest.exists():
        data = load(manifest)
        if data["camera"] != spec or data["measurements"] != measured or data["focus_note"] != args.focus_note:
            raise ValueError("resume with the same camera profile, board and focus; use a new layout folder if changed")
    else:
        data = {"camera": spec, "measurements": measured, "focus_note": args.focus_note, "image_size": list(size),
                "frames": []}
        write_json(manifest, data)
    signatures = [np.asarray(row["signature"]) for row in data["frames"]]
    covered = {tuple(cell) for row in data["frames"] for cell in row["coverage_cells"]}
    last = monotonic_ns()
    previous = None
    name = "Wrist ChArUco | S save | Q finish"
    try:
        with open_camera(spec) as (read, _):
            while True:
                image, _, _ = read()
                stamp = monotonic_ns()
                if image.shape != (size[1], size[0], 3):
                    raise ValueError("wrist capture does not match the configured native RGB size")
                view = detect_board(image, board, detector)
                signature = view.signature(size) if view else None
                clear = view is not None and view.sharpness >= args.sharpness
                stable = clear and previous is not None and np.linalg.norm(signature - previous) < .08
                new = clear and all(np.linalg.norm(signature - saved) >= .10 for saved in signatures)
                previous = signature if clear else None
                preview = image.copy()
                if view:
                    cv2.aruco.drawDetectedCornersCharuco(preview, view.pixels.reshape(-1, 1, 2),
                                                         view.ids.reshape(-1, 1))
                _draw_grid(preview, covered)
                state = "no usable board" if not view else f"{len(view.ids)} corners | sharpness {view.sharpness:.0f}"
                cv2.putText(preview, f"{len(signatures)} saved | {state}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (0, 255, 255), 1)
                key = _key_for(name, preview)
                if key in (27, ord("q")):
                    print(f"{manifest} ({len(signatures)} views)")
                    return
                automatic = not args.manual and stable and stamp - last >= 700_000_000
                if new and (key == ord("s") or automatic):
                    filename = f"{len(signatures):04}.png"
                    if (output / "frames" / filename).exists():
                        raise OSError(f"unindexed wrist frame exists at {filename}; preserve it before retrying")
                    save_png(output / "frames" / filename, image)
                    cells = coverage_regions(view.pixels, size)
                    data["frames"].append({"path": f"frames/{filename}", "host_monotonic_ns": stamp,
                                           "host_unix_ns": time_ns(), "corner_ids": view.ids.tolist(),
                                           "corners_px": view.pixels.tolist(), "signature": signature.tolist(),
                                           "coverage_cells": sorted(cells), "sharpness": view.sharpness})
                    write_json(manifest, data)
                    signatures.append(signature)
                    covered |= cells
                    last = stamp
    finally:
        cv2.destroyAllWindows()


def validate_global(root, args):
    output = root / stream_name(args.name)
    geometry = load(output / "reference.json")
    if args.links or "geometry_error" in geometry:
        rgb = read_png(output / "rgb.png")
        geometry |= workspace_geometry(rgb, geometry["camera"], geometry["measurements"],
                                       links=load(args.links) if args.links else None)
        corners = {int(marker): np.array(row["corners_px"]) for marker, row in geometry["markers"].items()}
        save_png(output / "markers.png", _global_overlay(rgb, corners))
        geometry.pop("geometry_error", None)
        write_json(output / "reference.json", geometry)
    if args.holdout:
        native = read_png(output / "depth.png")
        results = depth_holdout(native, geometry["camera"], geometry, load(args.holdout))
        write_json(output / "depth_holdout.json", {
            "results": results,
            "per_tray_near_far_complete": len(results) == 4 and all("xy_error_mm" in row for row in results),
        })
        print(output / "depth_holdout.json")
    else:
        print(output / "reference.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    actions = parser.add_subparsers(dest="action", required=True)
    measure = actions.add_parser("measure", help="record the actual printed dimensions in mm")
    measure.add_argument("--layout", required=True)
    measure.add_argument("--precision-mm", type=float, default=1)
    measure.add_argument("--mm", type=float, nargs=8, metavar="MM", help="width height board-marker 40 41 42 line1 line2")
    global_cmd = actions.add_parser("global", help="save a D455 RGB-D reference with markers 40/41/42")
    global_cmd.add_argument("--config", type=Path, required=True)
    global_cmd.add_argument("--name", default="global")
    global_cmd.add_argument("--compare", type=Path, help="folder of the prior reference.json")
    global_cmd.add_argument("--links", type=Path, help="measured marker-to-base and tray transforms")
    global_cmd.add_argument("--episode", type=Path, help="use RGB-D at the recorded settled start, without a camera")
    wrist = actions.add_parser("wrist", help="record diverse wrist ChArUco views")
    wrist.add_argument("--config", type=Path, required=True)
    wrist.add_argument("--focus-note", default="same focus and resolution as collection")
    wrist.add_argument("--sharpness", type=float, default=45, help="laplacian variance accepted for a view")
    wrist.add_argument("--manual", action="store_true", help="save clear new views only when S is pressed")
    fit = actions.add_parser("fit-wrist", help="compare Brown and fisheye with held-out views")
    fit.add_argument("--holdout", nargs="+", default=[], help="optional names such as 0003.png 0007.png")
    check = actions.add_parser("check-global", help="apply measured frame links and validate independent depth points")
    check.add_argument("--name", default="global")
    check.add_argument("--links", type=Path)
    check.add_argument("--holdout", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "measure":
            record_measurements(args.root, args)
            return 0
        measured = load(args.root / "measurements.json")
        if args.action == "global":
            capture_global(args.root, args, measured)
        elif args.action == "wrist":
            capture_wrist(args.root, args, measured)
        elif args.action == "check-global":
            validate_global(args.root, args)
        else:
            capture = args.root / "wrist"
            results = fit_wrist(capture, measured, args.holdout)
            for model, result in results["models"].items():
                print(f"{model}: held-out RMS {result['holdout']['rms_px']:.3f} px, "
                      f"edge RMS {result['holdout']['edge_rms_px']} px")
            for split in ("training_grid", "holdout_grid"):
                missing = results[split]["missing"]
                labels = [("top", "middle", "bottom")[y] + "-" + ("left", "center", "right")[x] for y, x in missing]
                if labels:
                    print(f"{split}: add coverage at {', '.join(labels)}")
            print(capture / "intrinsics.json")
    except (EOFError, ImportError, OSError, RuntimeError, KeyError, TypeError, ValueError, cv2.error) as error:
        print(f"camera calibration: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
