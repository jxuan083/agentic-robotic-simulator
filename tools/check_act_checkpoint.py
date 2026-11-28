#!/usr/bin/env python3
"""Load an ACT checkpoint with its saved pre/post-processors and run select_action on dataset frames.

  .venv-train/bin/python tools/check_act_checkpoint.py outputs/train/act_right_smoke/checkpoints/last/pretrained_model \
      outputs/datasets/fake-act [--frames 8]

Checks the inference path only (load, normalise, predict, unnormalise; action shape and finite values), not the
policy's quality. Prints the predicted and recorded actions of the first frames.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="…/checkpoints/<step>/pretrained_model")
    ap.add_argument("dataset")
    ap.add_argument("--frames", type=int, default=8)
    a = ap.parse_args(argv)
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(a.checkpoint)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=a.checkpoint)
    conv = json.load(open(f"{a.dataset}/meta/trc_conversion.json"))
    ds = LeRobotDataset(conv["options"]["repo_id"], root=a.dataset)
    policy.reset()
    rows, ok = [], True
    with torch.no_grad():
        for i in range(min(a.frames, len(ds))):
            item = ds[i]
            obs = {k: v.unsqueeze(0) for k, v in item.items()
                   if k.startswith("observation.") and isinstance(v, torch.Tensor)}
            obs["task"] = [item["task"]]
            act = post(policy.select_action(pre(obs)))
            act = act.squeeze(0).cpu()
            ok = ok and act.shape == item["action"].shape and bool(torch.isfinite(act).all())
            rows.append({"frame": i, "predicted": [round(float(x), 2) for x in act],
                         "recorded": [round(float(x), 2) for x in item["action"]]})
    print(json.dumps({"checkpoint": a.checkpoint, "device": str(next(policy.parameters()).device),
                      "action_dim": int(item["action"].shape[0]), "ok": ok, "first_frames": rows[:3]}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
