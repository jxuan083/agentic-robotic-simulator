#!/usr/bin/env python3
"""Order -> delivery with the cameras (M1): the task executor finds blocks with Acquire, not the simulation truth.

  MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/evaluate_task_perception.py --episodes 30 --seed 11

Each episode: random blocks in the storage tray, messed up (stacked / leaning / touching, as
tools/declutter_target.py), then one order of two colours (one block each): a colour with a problem block (hidden
under another, stacked, leaning or crowded) and another colour in the scene. The executor runs until the order is
done, it asks the shortage question, or the time limit. Success: the placement tray holds exactly the ordered
colours (simulation truth, read only after the run) and the ledger agrees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_perception_exploration as common                     # noqa: E402
from declutter_target import choose_target                           # noqa: E402
from evaluate_height_check import ground_truth, messify             # noqa: E402
from locate_target_color import VideoRecorder, _write_json          # noqa: E402
from agentic_sim.config import load, v                               # noqa: E402
from agentic_sim.mujoco_backend.session import MjSession, SessionConfig   # noqa: E402
from agentic_sim.scene_layout import block_in_tray, trays            # noqa: E402


def run_episode(ep, args, random_path, out):
    rng = np.random.default_rng([args.seed, ep])
    session = MjSession(SessionConfig(scene=args.scene, random_blocks=True, random_seed=args.seed, random_episode=ep,
                                      random_config=str(random_path), task_perception=True),
                        log=lambda *_: None)
    ep_dir = out / f"episode_{ep:03d}"
    recorder = None
    try:
        for _ in range(60):
            common._render_tick(session, None, "settling")
        cams_edge = float(v(session.scene_cfg["block"]["edge"]))
        floor = trays(session.scene_cfg)["storage_pink"]["floor_top"]
        ops = messify(session, rng, int(rng.integers(1, args.max_ops + 1)), cams_edge, floor, args.kinds.split(","))
        for _ in range(int(2.0 / session.control_dt)):
            common._render_tick(session, None, "settling")
        colors = list(session.block_color_names)
        truth0 = ground_truth(session, cams_edge, floor)
        first, scenario = choose_target(colors, truth0, rng, "hidden", 0.015)
        others = sorted(set(colors) - {first})
        order = {first: 1}
        if others and args.colors > 1:
            order[str(rng.choice(others))] = 1
        tr = session.start_task()
        ex = tr.ex
        if args.video:
            recorder = session._frame_hook = VideoRecorder(session, session.live_perception, ep_dir / "video.webm")
        ex.order(order)
        t0 = session.t_sim
        while session.t_sim - t0 < args.time_limit_s:
            label = f"order {order} | {ex.status_text}" if ex.status_text else f"order {order}"
            session._target_search_ui = label
            common._render_tick(session, None, label)
            if ex.state == "done" or (ex.question is not None and ex.question["kind"] == "shortage"):
                break
        if recorder is not None:
            common._dwell(session, None, 60, "done")
        # ---- scoring (simulation truth) from here on ----
        placement = trays(session.scene_cfg)["placement_beige"]
        in_place = sorted(c for c, p in zip(colors, session.block_positions())
                          if block_in_tray(p, placement, cams_edge))
        wanted = sorted(c for c, n in order.items() for _ in range(n))
        led = ex.ledger.snapshot()
        success = bool(ex.state == "done" and in_place == wanted)
        skills = [e for e in ex.events if e["kind"] == "skill_done"]
        acq = [e for e in ex.events if e["kind"] in ("target_acquired", "target_not_acquired")]
        episode = {"episode": ep, "order": order, "scenario": scenario, "ops": ops, "success": success,
                   "state": ex.state, "question": ex.question, "placement_colors": in_place, "ledger": led,
                   "sim_s": round(session.t_sim - t0, 1), "acquire": acq,
                   "skills": [{k: e.get(k) for k in ("skill", "status", "failure_reason", "duration")} for e in skills],
                   "acquire_rounds": [e.get("evidence", {}).get("rounds") for e in skills if e["skill"] == "acquire"],
                   "truth_before": truth0, "colors": colors}
        _write_json(ep_dir / "evaluation.json", episode)
        print(f"[episode {ep}] order {order} ({scenario}): {ex.state}, placement {in_place} -> success {success}"
              f" ({episode['sim_s']} s)", flush=True)
        return episode
    finally:
        if recorder is not None:
            session._frame_hook = None
            recorder.close()
        if session.task is not None:
            session.task.close()
        session.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--only", type=int, nargs="+")
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--max-ops", type=int, default=3)
    p.add_argument("--kinds", default="stack,lean,touch")
    p.add_argument("--colors", type=int, default=2, help="colours in the order (one block each)")
    p.add_argument("--time-limit-s", type=float, default=240.0)
    p.add_argument("--video", action="store_true")
    p.add_argument("--scene", default="configs/scenes/lab_20261120.yaml")
    p.add_argument("--output", default="outputs/task-perception/run")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    random_cfg = load("configs/random_blocks.yaml")
    random_cfg["count_range"] = [args.blocks, args.blocks]
    random_path = out / "random_blocks.yaml"
    random_path.write_text(yaml.safe_dump(random_cfg, sort_keys=False), encoding="utf-8")
    eps = [run_episode(i, args, random_path, out) for i in (args.only or range(args.episodes))]
    summary = {"episodes_n": len(eps), "success_n": sum(e["success"] for e in eps),
               "states": {s: sum(e["state"] == s for e in eps) for s in {e["state"] for e in eps}},
               "sim_s_mean": round(float(np.mean([e["sim_s"] for e in eps])), 1),
               "skills": {k: sum(s["skill"] == k for e in eps for s in e["skills"]) for k in
                          ("acquire", "pick_at", "place", "push", "home")}}
    _write_json(out / "evaluation.json", {"summary": summary, "args": vars(args)})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
