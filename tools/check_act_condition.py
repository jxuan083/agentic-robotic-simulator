#!/usr/bin/env python3
"""Does a conditioned ACT checkpoint use its target? Same opening frame, other target blocks -> other actions.

  .venv-train/bin/python tools/check_act_condition.py outputs/train/<run>/checkpoints/last/pretrained_model \
      outputs/datasets/<name> [--episodes 20]

For the first frame of each LeRobot episode, the other blocks D455 saw in the source tray of the opening frame
(review.json d455_blocks) become alternative targets: the condition's target x, y, yaw is swapped (the place part
is kept) and, when the dataset was converted with --markers, the global image is redrawn from the source PNG with
the circle on that block. The whole predicted action chunk is compared with the one for the true target.

  chunk_diff_deg   mean |difference| over the chunk and the body joints (alternative vs true target)
  pan_toward       share of alternatives where shoulder_pan at the chunk end moves the same way as the block
                   lies relative to the true target (+y = left = larger pan on the right arm; checked, not assumed:
                   the sign is also reported per case)
A policy that ignores the condition gives chunk_diff_deg near 0. Train-set episodes only show that the condition is
read; held-out episodes (not converted into the training set) show whether it generalises.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_control import JOINTS  # noqa: E402
from agentic_sim.real_dataset import MARKER_CAMERA, draw_markers  # noqa: E402


def alternatives(review: dict, target: dict) -> list:
    blocks = ((review.get("d455_blocks") or {}).get("blocks")) or []
    return [b for b in blocks if b["tray"] == target["tray"]
            and math.hypot(b["x_mm"] - target["x_mm"], b["y_mm"] - target["y_mm"]) > 20]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="…/checkpoints/<step>/pretrained_model")
    ap.add_argument("dataset")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args(argv)
    import numpy as np
    from PIL import Image
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    root = Path(a.dataset)
    conv = json.loads((root / "meta" / "trc_conversion.json").read_text())
    if not conv.get("condition_names") and not conv.get("marker_style"):
        print("dataset has neither a condition nor markers", file=sys.stderr)
        return 1
    sources = [json.loads(line) for line in open(root / "meta" / "trc_sources.jsonl", encoding="utf-8")]
    ds = LeRobotDataset(conv["options"]["repo_id"], root=root)
    policy = ACTPolicy.from_pretrained(a.checkpoint)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=a.checkpoint)
    body = [JOINTS.index(j) for j in JOINTS[:-1]]
    pan = JOINTS.index("shoulder_pan")

    def chunk(obs):
        with torch.no_grad():
            out = policy.predict_action_chunk(pre({k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}))
            return post(out).squeeze(0).cpu().numpy()

    cases, skipped = [], []
    for src in sources[:a.episodes]:
        ep_root = Path(src["source_episode"])
        review = json.loads((ep_root / "review.json").read_text(encoding="utf-8"))
        target = (review.get("target") or {}).get("d455")
        others = alternatives(review, target) if target else []
        if not others:
            skipped.append({"episode": src["source_episode"], "reason": "no other block in the source tray"})
            continue
        lo = int(ds.meta.episodes[src["lerobot_episode"]]["dataset_from_index"])
        item = ds[lo]
        obs = {k: v.unsqueeze(0) for k, v in item.items() if k.startswith("observation.") and torch.is_tensor(v)}
        obs["task"] = [item["task"]]
        policy.reset()
        base = chunk(obs)
        raw = None
        if src.get("markers"):
            frames = [json.loads(line) for line in open(ep_root / MARKER_CAMERA / "frames.jsonl")]
            raw = Image.open(ep_root / MARKER_CAMERA / frames[src["camera_frames"][MARKER_CAMERA][0]]["rgb"]).convert("RGB")
        for b in others:
            alt = dict(obs)
            if src.get("condition"):
                yaw = math.radians(4 * float(b.get("yaw_deg") or 0.0))
                cond = list(src["condition"])
                cond[:4] = [b["x_mm"], b["y_mm"], math.sin(yaw), math.cos(yaw)]
                alt["observation.environment_state"] = torch.tensor([cond], dtype=torch.float32)
            if raw is not None:
                img = draw_markers(raw.copy(), src["markers"] | {"target": b["pixel"]})
                arr = np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0
                alt[f"observation.images.{MARKER_CAMERA}"] = torch.from_numpy(arr).unsqueeze(0)
            policy.reset()
            moved = chunk(alt)
            diff = float(np.abs(moved[:, body] - base[:, body]).mean())
            d_pan = float(moved[-1, pan] - base[-1, pan])
            d_y = b["y_mm"] - target["y_mm"]
            cases.append({"episode": ep_root.name, "alt": [round(b["x_mm"]), round(b["y_mm"])], "dy_mm": round(d_y),
                          "chunk_diff_deg": round(diff, 2), "pan_delta_deg": round(d_pan, 2),
                          "same_sign": (d_pan > 0) == (d_y > 0) if abs(d_y) > 20 and abs(d_pan) > 0.5 else None})
    signs = [c["same_sign"] for c in cases if c["same_sign"] is not None]
    summary = {"checkpoint": a.checkpoint, "dataset": str(root), "cases": len(cases),
               "chunk_diff_deg_mean": round(float(np.mean([c["chunk_diff_deg"] for c in cases])), 2) if cases else None,
               "pan_same_sign_share": round(sum(signs) / len(signs), 2) if signs else None,
               "skipped": skipped[:10], "examples": cases[:12]}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if cases else 1


if __name__ == "__main__":
    sys.exit(main())
