"""Send orders and answers to the task executor of a running simulator server (issue #5).

  uv run tools/task.py --backend mujoco order red=3                 # new total order (the executor takes the arm)
  uv run tools/task.py --backend mujoco order red=1 blue=2           # change it: a new total, not an addition
  uv run tools/task.py --backend mujoco order --add blue=1           # add to the current order
  uv run tools/task.py --backend mujoco state [--watch]              # ledger, holding, skill / phase, question
  uv run tools/task.py --backend mujoco answer 1 return              # reply to question 1
  uv run tools/task.py --backend mujoco answer 2 substitute --color yellow
  uv run tools/task.py --backend mujoco answer 1 place_at --xy 0.30 0.05
  uv run tools/task.py --backend mujoco stop                         # give the arm back to keyboard / mouse

Load a scene with named block colours first, e.g.
  uv run tools/sim.py --backend mujoco --experiment minimal_pick_place --mode teleop --block-colors red,red,red,blue,blue,yellow
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from agentic_sim.server.client import RemoteError, SimClient  # noqa: E402


def counts_arg(items) -> dict:
    out = {}
    for it in items:
        c, _, n = it.partition("=")
        if not c or not n.isdigit():
            raise SystemExit(f"expected colour=count, got {it!r}")
        out[c] = int(n)
    return out


def show(st: dict):
    if not st.get("active"):
        print("task executor not active (send an order first)")
        return
    led = st["ledger"]
    print(f"state {st['state']}  order revision {st['revision']}")
    for c in led["target"] | led["delivered"]:
        print(f"  {c:<11} order {led['target'].get(c, 0)}  delivered {led['delivered'].get(c, 0)}")
    h, sk, q = st.get("holding"), st.get("skill"), st.get("question")
    print("  holding: " + (f"{h['color']} (block {h['block']})" if h else "nothing"))
    print("  skill: " + (f"{sk['name']} / {sk['phase']}" if sk else "-"))
    if q:
        print(f"  question {q['id']} ({q['kind']}): {q['text']}")
        print(f"  answer with: uv run tools/task.py --backend <backend> answer {q['id']} <{'|'.join(q['options'])}>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["isaac", "mujoco"], required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("order")
    o.add_argument("--add", action="store_true", help="add to the current order instead of replacing it")
    o.add_argument("counts", nargs="+", help="colour=count ...")
    s = sub.add_parser("state")
    s.add_argument("--watch", action="store_true", help="print every second until Ctrl+C")
    s.add_argument("--json", action="store_true")
    a = sub.add_parser("answer")
    a.add_argument("id", type=int)
    a.add_argument("choice")
    a.add_argument("--color")
    a.add_argument("--xy", type=float, nargs=2)
    sub.add_parser("stop")
    args = ap.parse_args(argv)
    try:
        with SimClient.for_backend(args.backend, name="task-cli") as c:
            if args.cmd == "order":
                r = c.call("task_order", counts=counts_arg(args.counts), mode="add" if args.add else "total")
                print(json.dumps(r, ensure_ascii=False))
            elif args.cmd == "answer":
                kw = {k: v for k, v in (("color", args.color), ("xy", args.xy)) if v is not None}
                print(json.dumps(c.call("task_answer", id=args.id, choice=args.choice, **kw), ensure_ascii=False))
            elif args.cmd == "stop":
                print(json.dumps(c.call("task_stop")))
            else:
                while True:
                    st = c.call("task_state")
                    print(json.dumps(st, ensure_ascii=False, indent=1)) if args.json else show(st)
                    if not args.watch:
                        break
                    time.sleep(1.0)
                    print()
    except (RemoteError, ConnectionError, OSError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
