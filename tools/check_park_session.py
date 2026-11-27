#!/usr/bin/env python3
"""Quick read-only check of the latest park-start episodes between collection blocks (no LeRobot, no writing).

  python3 tools/check_park_session.py outputs/261126-park [--last 20]

Runs the converter's own selection (agentic_sim.real_dataset.plan_episode) on the newest episodes for both datasets
- ACT: segment grasp (start .. lift, the operator's rows), condition target;
- VLA: segment task (start .. rise over the slot, programmed place included), no condition -
and prints, per episode, whether each would take it and why not, plus what to fix at the desk: D455 target / landing
found, crowded placement, wrist_roll travel in the grasp segment (the wrist may turn up to ~30 deg; the programmed
place turns it to the tray, not counted), demonstration length and late camera rows. The share of exclusions counts
the ACT dataset. Exit 1 when more than --max-skip of the checked episodes would be excluded for a
data problem (episodes the operator marked failed are excluded from training but not counted here).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_dataset import find_episodes, plan_episode  # noqa: E402

WRIST_TURN_DEG = 30.0   # demos may turn the wrist up to ~30 deg to square up with the block
CROWDED_MM = 45.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--last", type=int, default=20, help="newest episodes to check (0 = all)")
    ap.add_argument("--condition", default="target", help="ACT condition the training will use")
    ap.add_argument("--max-skip", type=float, default=0.2, help="excluded share that fails the check")
    ap.add_argument("--modes", default="demonstration,script", help="recording modes training takes (auto: script)")
    a = ap.parse_args(argv)
    episodes = find_episodes([a.session])
    episodes = episodes[-a.last:] if a.last else episodes
    lines, skipped, success, rounds = [], 0, 0, set()
    for ep in episodes:
        modes = tuple(a.modes.split(","))
        plan = plan_episode(ep, segment="grasp", condition_kind=a.condition, modes=modes)
        vla = plan_episode(ep, segment="task", condition_kind="none", modes=modes)
        review = json.loads((ep / "review.json").read_text()) if (ep / "review.json").exists() else {}
        result = json.loads((ep / "manifest.json").read_text()).get("operator_result") or {}
        success += bool(result.get("success"))
        rnd = review.get("round") or {}
        if rnd:
            rounds.add(rnd["n"])
        notes = []
        landed = (review.get("place_target") or {}).get("landed_d455")
        if not (review.get("target") or {}).get("d455"):
            notes.append("D455 未找到目標")
        if result.get("success") and not landed:
            notes.append("沒量到落點")
        if landed and landed.get("neighbour_mm") is not None and landed["neighbour_mm"] < CROWDED_MM:
            notes.append(f"太擠 {landed['neighbour_mm']:.0f} mm")
        if plan["included"]:
            rolls = [s["state"][4] for s in plan["segments"][0]["samples"]]
            travel = max(rolls) - min(rolls)
            if travel > WRIST_TURN_DEG:
                notes.append(f"夾取段 wrist_roll 轉了 {travel:.0f}°")
            if plan.get("regrasp"):
                notes.append("補夾（閉爪→張開→再閉）")
            cut = plan["stats"].get("trimmed_rows", 0)
            if cut:
                notes.append(f"停頓截掉 {cut / 30:.1f}s")
            late = sum(plan["stats"].get("late_camera_rows", {}).values())
            if late:
                notes.append(f"相機延遲 {late} 列")
            length = f"{plan['stats']['rows'] / 30:.1f}s"
        else:
            skipped += "failed" not in plan["reason"]   # a failed demonstration is not a data problem
            length = "-"
        act = "ok  " if plan["included"] else "skip"
        v = "ok  " if vla["included"] else "skip"
        where = f"R{rnd.get('n', '?')}-{rnd.get('step', '?')}" if rnd else "    "
        why = [] if plan["included"] else [f"ACT: {plan['reason']}"]
        if not vla["included"] and vla["reason"] != plan["reason"]:
            why.append(f"VLA: {vla['reason']}")
        lines.append(f"ACT {act} VLA {v} {ep.name} {where} {result.get('success') and '成功' or '失敗'} {length:>6}  "
                     + "".join(w + "  " for w in why) + "；".join(notes))
    print("\n".join(lines))
    n = len(episodes)
    share = skipped / n if n else 0.0
    print(f"\n{n} 筆（輪次 {sorted(rounds)}）：成功 {success}，ACT 資料問題排除 {skipped}（{share:.0%}）；"
          f"夾取段長度是 ACT 段（起姿 → 抬起）")
    if share > a.max_skip:
        print(f"排除超過 {a.max_skip:.0%}：先看上面的原因再繼續收")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
