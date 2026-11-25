#!/usr/bin/env python3
"""Trim the task-card block positions on site and check that 120 cards + 20 held-out positions still fit.

  show  CARDS                         numbered positions per zone, then the card/held-out check
  drop  CARDS pink_far:3,5 ...        remove numbered positions (numbers from `show`), in place
  drop  CARDS --max-x pink=200 ...    remove positions beyond a tray-local x (mm), e.g. an unreachable far edge
        (also --max-y / --min-y)
  drop  CARDS --yaw 60                remove every position with this yaw

Positions are tray-local [x mm, y mm, yaw deg]: origin at the inner tray corner nearest the arm base on its right,
+x forward, +y left. The previous file is kept as CARDS.previous-NN.json.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_sim.real_collection import make_cards  # noqa: E402


def check(cards):
    try:
        made, reserves = make_cards(cards)
    except ValueError as error:
        print(f"NOT ENOUGH: {error}")
        return False
    print(f"OK: {len(made)} task cards, {len(reserves)} held-out positions")
    distinct = {}
    for card in made:
        distinct.setdefault(f"{card['source_tray']}_{card['region']}", set()).add(tuple(card["pose"][:2]))
    for zone, points in sorted(distinct.items()):
        note = "" if len(points) >= 30 else "  (fewer than 30: some positions repeat; keep more points if possible)"
        print(f"  {zone}: {len(points)} distinct card positions{note}")
    return True


def show(cards):
    for zone, points in cards["zones"].items():
        print(f"{zone} ({len(points)}):")
        for i, (x, y, yaw) in enumerate(points):
            print(f"  {i:3d}  x {x:6.1f}  y {y:6.1f}  yaw {yaw:5.1f}")
    return check(cards)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("show")
    p.add_argument("cards", type=Path)
    p = sub.add_parser("drop")
    p.add_argument("cards", type=Path)
    p.add_argument("items", nargs="*", help="zone:i,j,... using the numbers printed by show")
    p.add_argument("--max-x", action="append", default=[], help="tray=mm: drop positions with x above this")
    p.add_argument("--max-y", action="append", default=[], help="tray=mm: drop positions with y above this")
    p.add_argument("--min-y", action="append", default=[], help="tray=mm: drop positions with y below this")
    p.add_argument("--yaw", type=float, action="append", default=[], help="drop positions with this yaw")
    args = parser.parse_args(argv)
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    if args.command == "show":
        return 0 if show(cards) else 1

    remove = {zone: set() for zone in cards["zones"]}
    for item in args.items:
        zone, _, indices = item.partition(":")
        if zone not in remove or not indices:
            parser.error(f"expected zone:i,j with zone in {sorted(remove)}")
        remove[zone] |= {int(i) for i in indices.split(",")}
    def bounds(items):
        out = {}
        for item in items:
            tray, _, mm = item.partition("=")
            out[tray] = float(mm)
        return out

    max_x, max_y, min_y = bounds(args.max_x), bounds(args.max_y), bounds(args.min_y)
    for zone, points in cards["zones"].items():
        tray = zone.split("_")[0]
        for i, (x, y, yaw) in enumerate(points):
            if (x > max_x.get(tray, float("inf")) or y > max_y.get(tray, float("inf"))
                    or y < min_y.get(tray, float("-inf")) or yaw in args.yaw):
                remove[zone].add(i)
    for zone, indices in remove.items():
        if any(i >= len(cards["zones"][zone]) for i in indices):
            parser.error(f"{zone}: index out of range")
        cards["zones"][zone] = [p for i, p in enumerate(cards["zones"][zone]) if i not in indices]
        if indices:
            print(f"{zone}: removed {len(indices)}, {len(cards['zones'][zone])} left")
    n = 1
    while (old := args.cards.with_name(f"{args.cards.stem}.previous-{n:02}{args.cards.suffix}")).exists():
        n += 1
    args.cards.rename(old)
    args.cards.write_text(json.dumps(cards, indent=1) + "\n", encoding="utf-8")
    print(f"saved {args.cards} (previous: {old.name})")
    return 0 if check(cards) else 1


if __name__ == "__main__":
    sys.exit(main())
