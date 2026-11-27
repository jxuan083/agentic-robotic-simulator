#!/usr/bin/env python3
"""Real SO-101 episodes -> a LeRobot dataset for training (docs/real-training.md).

  .venv-train/bin/python tools/real_to_lerobot.py outputs/261126-park --out outputs/datasets/park-act-v1 \
      --condition target+place --markers                 # park-start rounds: whole pick-and-place, conditioned
  .venv-train/bin/python tools/real_to_lerobot.py outputs/261124-right --out outputs/datasets/right-act-v1 --segment handoff
  .venv-train/bin/python tools/real_to_lerobot.py <session> --out <dir> --segment review --language zh
  .venv-train/bin/python tools/real_to_lerobot.py <session> --out <dir> --handoff-jitter 3 --modes demonstration,dagger

Selection, camera pairing and exclusions are in agentic_sim.real_dataset (standard library). This entry needs the
training environment (LeRobot 0.4.3). Next to LeRobot's own files it writes meta/trc_sources.jsonl (one line per
LeRobot episode: the source episode, segment, start offset, both instructions, review frames, camera offsets) and
meta/trc_conversion.json (options, versions, included / excluded episodes with the reason).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_control import JOINTS  # noqa: E402
from agentic_sim.real_dataset import (  # noqa: E402
    CONDITION_NAMES, CONDITIONS, MARKER, MARKER_CAMERA, SEGMENTS, draw_markers, find_episodes, plan_episode,
)


def features(cameras: dict, n_condition: int = 0) -> dict:
    f = {"observation.state": {"dtype": "float32", "shape": (len(JOINTS),), "names": list(JOINTS)},
         "action": {"dtype": "float32", "shape": (len(JOINTS),), "names": list(JOINTS)},
         "task_is_policy": {"dtype": "int64", "shape": (1,), "names": None},
         "is_handoff": {"dtype": "int64", "shape": (1,), "names": None}}
    for name, cam in cameras.items():
        f[f"observation.images.{name}"] = {"dtype": "video", "shape": (cam["height"], cam["width"], 3),
                                           "names": ["height", "width", "channels"]}
    if n_condition:
        f["observation.environment_state"] = {"dtype": "float32", "shape": (n_condition,),
                                              "names": CONDITION_NAMES[:n_condition]}
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="sessions or episode directories")
    ap.add_argument("--out", required=True, help="new dataset directory (must not exist)")
    ap.add_argument("--repo-id", default="local/so101-right", help="LeRobot repo id written into the metadata")
    ap.add_argument("--arm", default="right")
    ap.add_argument("--segment", choices=SEGMENTS, default="task",
                    help="task: park-start pick and place (default); grasp / place: one half; handoff / review: "
                         "older start-pose episodes; full: whole recording")
    ap.add_argument("--condition", choices=CONDITIONS, default="none",
                    help="target condition written as observation.environment_state (ACT reads it as one token)")
    ap.add_argument("--markers", action="store_true",
                    help=f"draw the target circle / landing square on the {MARKER_CAMERA} camera images")
    ap.add_argument("--include-failed", action="store_true", help="keep episodes the operator marked failed")
    ap.add_argument("--handoff-jitter", type=int, default=0, help="also start k frames before / after the handoff")
    ap.add_argument("--max-offset-ms", type=float, default=25.0, help="camera frame vs observation readback")
    ap.add_argument("--late-ms", type=float, default=70.0, help="hard limit for a late frame after a dropped one")
    ap.add_argument("--late-run", type=int, default=2, help="consecutive rows allowed beyond --max-offset-ms")
    ap.add_argument("--modes", default="demonstration", help="recording modes to include, comma-separated")
    ap.add_argument("--max-still-s", type=float, default=0.3,
                    help="still stretches (arm and gripper command) keep only their first N s (0 = keep all)")
    ap.add_argument("--regrasp", choices=("keep", "skip"), default="keep",
                    help="episodes that close, reopen and close the gripper again (a corrected miss)")
    ap.add_argument("--language", choices=("en", "zh"), default="en", help="instruction written as LeRobot 'task'")
    ap.add_argument("--allow-fake", action="store_true", help="include source=fake episodes (pipeline tests only)")
    ap.add_argument("--video", action=argparse.BooleanOptionalAction, default=True,
                    help="store camera streams as video (default) or PNG images")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        ap.error(f"{out} exists; choose a new --out")

    import numpy as np
    from PIL import Image
    import importlib.metadata as md
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    plans = [plan_episode(p, arm=a.arm, segment=a.segment, max_offset_ms=a.max_offset_ms, allow_fake=a.allow_fake,
                          handoff_jitter=a.handoff_jitter, modes=tuple(a.modes.split(",")),
                          condition_kind=a.condition, include_failed=a.include_failed,
                          late_ms=a.late_ms, late_run=a.late_run, max_still_s=a.max_still_s, regrasp=a.regrasp)
             for p in find_episodes(a.paths)]
    used = [p for p in plans if p["included"]]
    for p in plans:
        print(f"[{'ok' if p['included'] else 'skip'}] {p['episode']}: {p['reason']}", flush=True)
    if not used:
        print("no episode to convert", file=sys.stderr)
        return 1
    manifests = [json.loads((Path(p["episode"]) / "manifest.json").read_text()) for p in used]
    cams = manifests[0]["cameras"]
    hz = {m["control_hz"] for m in manifests}
    if len(hz) != 1 or any(m["cameras"].keys() != cams.keys() or any(
            (m["cameras"][c]["width"], m["cameras"][c]["height"]) != (cams[c]["width"], cams[c]["height"])
            for c in cams) for m in manifests):
        print("episodes differ in control rate or camera set / size; convert them separately", file=sys.stderr)
        return 1
    fps = hz.pop()
    if a.markers and MARKER_CAMERA not in cams:
        print(f"--markers needs a {MARKER_CAMERA!r} camera", file=sys.stderr)
        return 1
    n_condition = len(used[0]["condition"] or [])
    ds = LeRobotDataset.create(a.repo_id, fps=int(fps), features=features(cams, n_condition), root=out,
                               robot_type="so101_follower", use_videos=a.video)
    sources = []
    for plan, man in zip(used, manifests):
        task = plan["instruction"][a.language] or "pick and place the block"
        for seg in plan["segments"]:
            for s in seg["samples"]:
                frame = {"observation.state": np.asarray(s["state"], np.float32),
                         "action": np.asarray(s["action"], np.float32),
                         "task_is_policy": np.asarray([s["task_is_policy"]], np.int64),
                         "is_handoff": np.asarray([s["is_handoff"]], np.int64), "task": task}
                if n_condition:
                    frame["observation.environment_state"] = np.asarray(plan["condition"], np.float32)
                for c, im in s["images"].items():
                    img = Image.open(im["path"]).convert("RGB")
                    if a.markers and c == MARKER_CAMERA:
                        img = draw_markers(img, plan["markers"])
                    frame[f"observation.images.{c}"] = np.asarray(img)
                ds.add_frame(frame)
            ds.save_episode()
            sources.append({"lerobot_episode": len(sources), "source_episode": plan["episode"],
                            "episode_id": plan["episode_id"], "source": plan["source"], "mode": plan["mode"],
                            "segment": a.segment, "start": seg["start"], "end": seg["end"],
                            "start_offset": seg["start_offset"], "handoff": plan["handoff"],
                            "review_frames": plan["review_frames"], "instruction": plan["instruction"],
                            "condition": plan["condition"], "markers": plan["markers"] if a.markers else None,
                            "round": plan["round"], "scene": plan["scene"],
                            "task_card": man.get("task_card"), "provenance": man.get("provenance"),
                            "rows": [x["row"] for x in seg["samples"]],
                            "camera_frames": {c: [x["images"][c]["frame"] for x in seg["samples"]] for c in cams},
                            "camera_offset_ms": plan["stats"]["camera_offset_ms"]})
    ds.finalize()
    meta = out / "meta"
    with open(meta / "trc_sources.jsonl", "w", encoding="utf-8") as f:
        for r in sources:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import torch
    (meta / "trc_conversion.json").write_text(json.dumps({
        "options": vars(a), "fps": fps, "cameras": cams, "joint_names": list(JOINTS),
        "condition_names": CONDITION_NAMES[:n_condition], "marker_style": MARKER if a.markers else None,
        "units": "body joints: calibrated degrees; gripper: percent open (as recorded; no joint mapping)",
        "observation_rule": "state = previous control row readback; images = frame nearest that readback, "
                            "captured before this row's command; action = this row's command",
        "versions": {"python": sys.version.split()[0], "lerobot": md.version("lerobot"), "torch": torch.__version__},
        "included": [{"episode": p["episode"], "copies": p["stats"]["copies"], "rows": p["stats"]["rows"]}
                     for p in used],
        "excluded": [{"episode": p["episode"], "reason": p["reason"]} for p in plans if not p["included"]],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}: {len(sources)} episodes, {sum(len(r['rows']) for r in sources)} frames, fps {fps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
