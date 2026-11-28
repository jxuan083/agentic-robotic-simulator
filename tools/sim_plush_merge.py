#!/usr/bin/env python3
"""Merge plush datasets recorded in parallel by tools/sim_plush_collect.py (shards with different --seed).

  .venv-train/bin/python tools/sim_plush_merge.py outputs/plush/datasets/shards/s1 ... --out <dataset>

LeRobot's aggregate_datasets joins the episodes in the order given; meta/sim_sources.jsonl of the shards is
concatenated with lerobot_episode renumbered, so episode k of the merged set still names its seed and layout.
The first N episodes of the result are the N-episode training subsets (every shard is an independent random draw).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="local/sim-plush")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        ap.error(f"{out} exists")
    from lerobot.datasets.aggregate import aggregate_datasets

    shards = [Path(s) for s in a.shards]
    aggregate_datasets([a.repo_id] * len(shards), a.repo_id, roots=shards, aggr_root=out)
    rows, offset, collections = [], 0, []
    for s in shards:
        kept = 0
        for line in (s / "meta" / "sim_sources.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line) | {"shard": s.name}
            if r.get("kept"):
                r["lerobot_episode"] = offset + kept
                kept += 1
            rows.append(r)
        offset += kept
        collections.append(json.loads((s / "meta" / "sim_collection.json").read_text(encoding="utf-8")))
    with open(out / "meta" / "sim_sources.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "meta" / "sim_collection.json").write_text(json.dumps(
        {"merged_from": [s.name for s in shards], "episodes": offset,
         "attempts": sum(c["attempts"] for c in collections), "shards": collections},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}: {offset} episodes from {len(shards)} shards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
