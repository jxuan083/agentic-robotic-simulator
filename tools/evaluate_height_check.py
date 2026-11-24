#!/usr/bin/env python3
"""MuJoCo evaluation of the depth height check (agentic_sim.perception.height) on messy layouts.

Each episode starts from a random upright layout, then stacks blocks on others, drops blocks so they lean on a
neighbour, or pushes two blocks into contact, and lets physics settle. The global camera's RGB and aligned depth
(with D455-like noise) go through the colour segmentation and the height check; ground truth (block poses and a
segmentation render) is read only to score.

  MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/evaluate_height_check.py --episodes 40 --seed 5
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import mujoco
import numpy as np
import yaml

from agentic_sim.config import load
from agentic_sim.mujoco_backend.live_perception import MjLivePerception
from agentic_sim.mujoco_backend.session import MjSession, SessionConfig
from agentic_sim.perception.height import PROBLEMS, backproject, close_pairs, height_check, polygon_gap

PROBLEM_LABELS = ("stacked", "tilted", "covered", "multi", "off_tray")


def _quat(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]


def _corners(d, body, edge):
    rot = np.asarray(d.xmat[body]).reshape(3, 3)
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * edge / 2
    return c @ rot.T + d.xpos[body]


def _tilt_deg(d, body):
    rot = np.asarray(d.xmat[body]).reshape(3, 3)
    return math.degrees(math.acos(min(1.0, float(np.abs(rot[2]).max()))))


def messify(session, rng, ops_n, edge, floor, kinds=("stack", "lean", "touch")):
    """Stack / lean / touch blocks (in place, before settling). Returns the operations done."""
    m, d, sc = session.m, session.d, session.sc
    n = len(sc.block_qpos)
    done = []
    for _ in range(ops_n):
        i, j = rng.choice(n, 2, replace=False)
        base = d.qpos[sc.block_qpos[i]:sc.block_qpos[i] + 3].copy()
        adr = sc.block_qpos[j]
        kind = rng.choice(list(kinds))
        yaw = _quat([0, 0, 1], rng.uniform(0, math.pi / 2))
        if kind == "stack":
            off = rng.uniform(-0.008, 0.008, 2)
            pos, q = [base[0] + off[0], base[1] + off[1], base[2] + edge + 0.003], yaw
        elif kind == "lean":
            # one edge on the floor, the top leaning on block i: slide in from far until it touches i
            ang = rng.uniform(0, 2 * math.pi)
            u = np.array([math.cos(ang), math.sin(ang), 0.0])          # from i towards the leaning block
            tilt = _quat([u[1], -u[0], 0], math.radians(rng.uniform(15, 40)))   # top tips towards i
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, tilt, yaw)
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, q)
            low = (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * edge / 2
                   @ rot.reshape(3, 3).T)[:, 2].min()
            pos = None
            for r in np.arange(2.2 * edge, 0.5 * edge, -0.0005):
                trial = [base[0] + r * u[0], base[1] + r * u[1], floor - low + 0.0005]
                d.qpos[adr:adr + 3] = trial
                d.qpos[adr + 3:adr + 7] = q
                mujoco.mj_forward(m, d)
                if any({int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} ==
                       {sc.block_bodies[i], sc.block_bodies[j]} and c.dist < 0 for c in d.contact[:d.ncon]):
                    break
                pos = trial
            pos = pos or trial
        else:
            ang = rng.uniform(0, 2 * math.pi)
            r = edge + rng.uniform(0.0, 0.008)
            pos, q = [base[0] + r * math.cos(ang), base[1] + r * math.sin(ang), base[2] + 0.001], yaw
        d.qpos[adr:adr + 3] = pos
        d.qpos[adr + 3:adr + 7] = q
        d.qvel[sc.block_qvel[j]:sc.block_qvel[j] + 6] = 0
        done.append({"kind": str(kind), "block": int(j), "on_or_next_to": int(i)})
    mujoco.mj_forward(m, d)
    return done


def ground_truth(session, edge, floor):
    d, sc = session.d, session.sc
    corners = [_corners(d, b, edge) for b in sc.block_bodies]
    centers = [np.asarray(d.xpos[b]) for b in sc.block_bodies]
    truth = []
    for k, b in enumerate(sc.block_bodies):
        tilt = _tilt_deg(d, b)
        top = float(corners[k][:, 2].max())
        covered = any(o != k and np.linalg.norm(centers[o][:2] - centers[k][:2]) < 0.9 * edge
                      and centers[o][2] > centers[k][2] + 0.5 * edge for o in range(len(centers)))
        if top < floor + 0.5 * edge:
            label = "off_tray"
        elif tilt > 5.0:
            label = "tilted"
        else:
            label = "upright" if round((top - floor) / edge) == 1 else "stacked"
        truth.append({"label": label, "covered": covered, "tilt_deg": round(tilt, 1), "top_z_m": round(top, 4),
                      "footprint_xy_m": cv2.convexHull(corners[k][:, :2].astype(np.float32)).reshape(-1, 2).tolist()})
    return truth


def render_global(manager, session, noise_1m_m, rng):
    m, d = session.m, session.d
    cam = manager._geometry("global", d)
    r = mujoco.Renderer(m, height=cam.height, width=cam.width)
    cid = manager.cam_ids["global"]
    r.update_scene(d, camera=cid)
    rgb = np.ascontiguousarray(r.render(), np.uint8)
    r.enable_depth_rendering()
    r.update_scene(d, camera=cid)
    depth = r.render().astype(float)
    r.disable_depth_rendering()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cid)
    seg = r.render()
    r.close()
    if noise_1m_m > 0:   # stereo depth error grows with z²; edges get mixed (flying) values
        depth = depth + rng.normal(0, 1, depth.shape) * noise_1m_m * depth ** 2
        lo = cv2.erode(depth.astype(np.float32), np.ones((3, 3), np.uint8))
        hi = cv2.dilate(depth.astype(np.float32), np.ones((3, 3), np.uint8))
        edge = (hi - lo) > 0.005
        mix = rng.uniform(0, 1, depth.shape)
        depth = np.where(edge, lo + mix * (hi - lo), depth)
    geom_body = np.asarray(m.geom_bodyid)
    body_block = {b: k for k, b in enumerate(session.sc.block_bodies)}
    block_px = np.full(seg.shape[:2], -1, int)
    is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    ids = seg[..., 0]
    for b, k in body_block.items():
        block_px[is_geom & np.isin(ids, np.nonzero(geom_body == b)[0])] = k
    return rgb, depth, cam, block_px


def segment_truth(mask, block_px, truth, min_share=0.2):
    vals = block_px[mask]
    vals = vals[vals >= 0]
    if len(vals) == 0:
        return {"blocks": [], "label": "background"}
    counts = Counter(vals.tolist())
    blocks = [k for k, c in counts.most_common() if c >= min_share * len(vals)]
    main = blocks[0]
    if len(blocks) > 1:
        label = "multi"
    elif truth[main]["covered"]:
        label = "covered"
    else:
        label = truth[main]["label"]
    return {"blocks": blocks, "label": label}


def run(args):
    cfg_all = load(args.config)
    cfg = cfg_all["height_check"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    random_cfg = load("configs/random_blocks.yaml")
    random_cfg["count_range"] = [args.blocks, args.blocks]
    random_path = out / "random_blocks.yaml"
    random_path.write_text(yaml.safe_dump(random_cfg, sort_keys=False), encoding="utf-8")
    rows, close_rows, covered_rows = [], [], []
    for ep in range(args.episodes):
        rng = np.random.default_rng([args.seed, ep])
        session = MjSession(SessionConfig(scene=args.scene, random_blocks=True, random_seed=args.seed,
                                          random_episode=ep, random_config=str(random_path)),
                             log=lambda *_: None)
        manager = MjLivePerception(session.scene_cfg, session.m, hz=2, log=lambda *_: None, camera="both")
        edge, floor = manager.edge, manager.floor
        for _ in range(60):
            session.step(dt=session.control_dt)
        ops = messify(session, rng, int(rng.integers(1, args.max_ops + 1)), edge, floor, args.kinds.split(","))
        for _ in range(int(args.settle_s / session.control_dt)):
            session.step(dt=session.control_dt)
        truth = ground_truth(session, edge, floor)
        rgb, depth, cam, block_px = render_global(manager, session, args.depth_noise_mm_at_1m / 1000, rng)
        est = manager.estimators["global"]
        detector = est.detector if args.segmentation == "plain" else est.split_detectors[1]
        dets = detector.predict(rgb).detections
        checks, segs = [], []
        for i, det in enumerate(dets):
            pts = backproject(depth, det.mask, cam, cfg.get("erode_px", 2))
            chk = height_check(pts, floor, edge, cfg, seed=i)
            st = segment_truth(det.mask, block_px, truth)
            checks.append(chk)
            segs.append(st)
            rows.append({"episode": ep, "segment": i, "color": det.color, "truth": st["label"],
                         "truth_blocks": st["blocks"], "pred": chk["status"], "tilt_deg": chk.get("tilt_deg"),
                         "truth_tilt_deg": truth[st["blocks"][0]]["tilt_deg"] if st["blocks"] else None,
                         "layers": chk.get("layers"), "top_area_faces": chk.get("top_area_faces")})
        for k, t in enumerate(truth):   # the key case: a block under another one, invisible to global
            if not t["covered"]:
                continue
            above = [o for o, u in enumerate(truth) if o != k and u["top_z_m"] > t["top_z_m"] + 0.5 * edge
                     and polygon_gap(u["footprint_xy_m"], t["footprint_xy_m"]) == 0.0]
            flagged = any(st["blocks"] and st["blocks"][0] in above and c["status"] in PROBLEMS
                          for c, st in zip(checks, segs))
            covered_rows.append({"episode": ep, "block": k, "top_flagged": flagged})
        pairs = {(a, b) for a, b, _ in close_pairs(checks, cfg["close_gap_m"], cfg.get("hull_margin_m", 0.0))}
        # per block pair: flagged when two valid segments of the pair are close, or one merged segment holds both
        valid = [i for i, c in enumerate(checks) if c["status"] not in ("side_face", "no_depth") and segs[i]["blocks"]]
        flagged_pairs = set()
        for a, b in pairs:
            if segs[a]["blocks"] and segs[b]["blocks"]:   # a background segment (e.g. the gripper) has no block
                flagged_pairs.add(frozenset((segs[a]["blocks"][0], segs[b]["blocks"][0])))
        for i in valid:
            if checks[i]["status"] == "merged":
                bl = segs[i]["blocks"]
                flagged_pairs.update(frozenset((x, y)) for x in bl for y in bl if x != y)
        seen = sorted({segs[i]["blocks"][0] for i in valid} | {k for i in valid for k in segs[i]["blocks"]})
        for x in range(len(seen)):
            for y in range(x + 1, len(seen)):
                ka, kb = seen[x], seen[y]
                gap = polygon_gap(truth[ka]["footprint_xy_m"], truth[kb]["footprint_xy_m"])
                if gap < 2 * cfg["close_gap_m"] and not (truth[ka]["covered"] or truth[kb]["covered"]):
                    close_rows.append({"episode": ep, "blocks": [ka, kb], "truth_gap_mm": round(gap * 1000, 1),
                                       "truth_close": gap < cfg["close_gap_m"],
                                       "flagged": frozenset((ka, kb)) in flagged_pairs})
        ep_dir = out / f"episode_{ep:03d}"
        ep_dir.mkdir(exist_ok=True)
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for det, chk, st in zip(dets, checks, segs):
            x0, y0, x1, y1 = det.bbox_xyxy
            ok = chk["status"] == "upright"
            col = (0, 160, 0) if ok else (0, 0, 255)
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
            cv2.putText(vis, f"{chk['status']} ({st['label']})", (int(x0), int(y0) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.imwrite(str(ep_dir / "global.png"), vis)
        json.dump({"ops": ops, "truth": truth, "checks": checks, "segments": segs}, open(ep_dir / "check.json", "w"),
                  indent=1, default=float)
        manager.close()
        session.finish()
    summary = summarize(rows, close_rows, covered_rows)
    json.dump({"args": vars(args), "summary": summary, "rows": rows, "close": close_rows, "covered": covered_rows},
              open(out / "evaluation.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1))


def summarize(rows, close_rows, covered_rows):
    seg = [r for r in rows if r["truth"] != "background" and r["pred"] not in ("side_face", "no_depth")]
    ignored = Counter(r["pred"] for r in rows if r["truth"] != "background" and r["pred"] in ("side_face", "no_depth"))
    conf = Counter((r["truth"], r["pred"]) for r in seg)
    problem_t = [r for r in seg if r["truth"] in PROBLEM_LABELS]
    upright_t = [r for r in seg if r["truth"] == "upright"]
    tilt_err = [abs(r["tilt_deg"] - r["truth_tilt_deg"]) for r in seg
                if r["truth"] == "tilted" and r["tilt_deg"] is not None]
    return {
        "segments_n": len(seg),
        "confusion_truth_to_pred": {f"{t} -> {p}": n for (t, p), n in sorted(conf.items())},
        "ignored_segments": dict(ignored),
        "problem_flagged": f"{sum(r['pred'] in PROBLEMS for r in problem_t)}/{len(problem_t)}",
        "upright_false_alarm": f"{sum(r['pred'] in PROBLEMS for r in upright_t)}/{len(upright_t)}",
        "covered_block_top_flagged": f"{sum(c['top_flagged'] for c in covered_rows)}/{len(covered_rows)}",
        "tilt_error_deg_median": float(np.median(tilt_err)) if tilt_err else None,
        "close_pairs_truth": sum(c["truth_close"] for c in close_rows),
        "close_pairs_flagged_of_truth": sum(c["truth_close"] and c["flagged"] for c in close_rows),
        "close_pairs_false_flag": sum((not c["truth_close"]) and c["flagged"] for c in close_rows),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--blocks", type=int, default=8, help="blocks per episode")
    p.add_argument("--max-ops", type=int, default=3, help="messy operations per episode (1..N)")
    p.add_argument("--kinds", default="stack,lean,touch", help="messy operations to draw from")
    p.add_argument("--segmentation", choices=("plain", "split"), default="split",
                   help="global colour segmentation: plain, or split by hue band and green value (touching blocks)")
    p.add_argument("--settle-s", type=float, default=2.0)
    p.add_argument("--depth-noise-mm-at-1m", type=float, default=5.0,
                   help="depth noise sigma at 1 m, scaled by z² (D455 spec <=2%% at 4 m is about 5 mm; 0 = perfect)")
    p.add_argument("--scene", default="configs/scenes/lab_20261120.yaml")
    p.add_argument("--config", default="configs/target_localization.yaml")
    p.add_argument("--output", default="outputs/height-check/run")
    run(p.parse_args())


if __name__ == "__main__":
    main()
