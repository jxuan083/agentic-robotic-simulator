#!/usr/bin/env python3
"""Read a converted dataset back with LeRobot and compare every frame with its source episode.

  .venv-train/bin/python tools/check_real_dataset.py outputs/datasets/right-act-v1 [--max-image-diff 12]

Per LeRobot episode (meta/trc_sources.jsonl): frame count; observation.state = the previous control row's readback;
action = the row's command; task = the instruction; is_handoff only on the handoff row; each camera image matches
the paired source PNG (mean absolute difference per channel, 0-255; video encoding is lossy). Exit 1 on a mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_control import JOINTS  # noqa: E402
from agentic_sim.real_dataset import MARKER_CAMERA, draw_markers  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--max-image-diff", type=float, default=12.0, help="mean |difference| per pixel channel")
    a = ap.parse_args(argv)
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(a.dataset)
    conv = json.loads((root / "meta" / "trc_conversion.json").read_text())
    lang = conv["options"]["language"]
    sources = [json.loads(line) for line in open(root / "meta" / "trc_sources.jsonl", encoding="utf-8")]
    ds = LeRobotDataset(conv["options"]["repo_id"], root=root)
    problems, worst = [], 0.0
    if ds.meta.total_episodes != len(sources):
        problems.append(f"{ds.meta.total_episodes} LeRobot episodes, {len(sources)} source records")
    for src in sources:
        e = src["lerobot_episode"]
        ep = ds.meta.episodes[e]
        lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        if hi - lo != len(src["rows"]):
            problems.append(f"episode {e}: {hi - lo} frames, source segment has {len(src['rows'])}")
            continue
        root_src = Path(src["source_episode"])
        rows = [json.loads(line) for line in open(root_src / "control.jsonl")]
        frames = {c: [json.loads(line) for line in open(root_src / c / "frames.jsonl")] for c in src["camera_frames"]}
        arm = conv["options"]["arm"]
        for k, i in enumerate(src["rows"]):
            item = ds[lo + k]
            prev = rows[i - 1]
            want_state = [prev["arms"][arm]["readback"][j] for j in JOINTS]
            want_action = [rows[i]["arms"][arm]["command"][j] for j in JOINTS]
            if not np.allclose(item["observation.state"].numpy(), want_state, atol=1e-4):
                problems.append(f"episode {e} row {i}: state differs")
            if not np.allclose(item["action"].numpy(), want_action, atol=1e-4):
                problems.append(f"episode {e} row {i}: action differs")
            if item["task"] != src["instruction"][lang]:
                problems.append(f"episode {e} row {i}: task {item['task']!r}")
            if src.get("condition") and not np.allclose(item["observation.environment_state"].numpy(),
                                                        src["condition"], atol=1e-4):
                problems.append(f"episode {e} row {i}: environment_state differs from the condition")
            if int(item["is_handoff"].item()) != int(i == src["handoff"]):
                problems.append(f"episode {e} row {i}: is_handoff {int(item['is_handoff'].item())}")
            for c, idx in src["camera_frames"].items():
                img = Image.open(root_src / c / frames[c][idx[k]]["rgb"]).convert("RGB")
                if src.get("markers") and c == MARKER_CAMERA:
                    img = draw_markers(img, src["markers"])
                want = np.asarray(img, np.float32)
                got = item[f"observation.images.{c}"].numpy()       # C, H, W in [0, 1]
                got = np.transpose(got, (1, 2, 0)) * 255.0
                diff = float(np.abs(got - want).mean())
                worst = max(worst, diff)
                if diff > a.max_image_diff:
                    problems.append(f"episode {e} row {i} {c}: image differs by {diff:.1f}")
    n = sum(len(s["rows"]) for s in sources)
    print(json.dumps({"dataset": str(root), "episodes": len(sources), "frames": n, "fps": ds.fps,
                      "features": sorted(ds.features), "tasks": sorted(set(ds.meta.tasks.index)),
                      "worst_image_mean_abs_diff": round(worst, 2), "problems": problems[:20],
                      "ok": not problems}, ensure_ascii=False, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
