#!/usr/bin/env python3
"""D455 pose in the right follower's mount frame from the arm itself (after mounting or re-mounting the camera).

  tools/lab_day.sh d455-capture     # lerobot env: torque off, move the follower by hand to ~8 poses, Enter each
  tools/lab_day.sh d455-fit         # uv env (MuJoCo): fit -> outputs/d455/pose_<date>.json, then use POSE=...

Capture: per pose, the median of 8 aligned depth frames (colour pixels) and the follower reading. Spread the poses
over both trays and heights (gripper over the near / far part of each tray, arm high and low); keep hands out of
the D455 view while Enter is pressed. The first frame is taken with the arm at park for the table plane.

Fit: the SO-101 Menagerie meshes posed by the calibration-file joint map (agentic_sim.real_kinematics) against the
depth points of the arm, plus the bare table (z = -1.6 mm), Gauss-Newton on the 6 camera parameters from the previous
pose (--init). Reports the median point-to-mesh distance per pose; a re-mount within a few cm / degrees of the
previous pose converges (the alignment image keeps it close).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

TABLE_Z = -0.0016
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def capture(args):
    from agentic_sim.real_d455 import open_aligned
    from agentic_sim.real_hardware import check_devices, load_profile, open_follower
    profile = load_profile(args.config)
    check_devices(profile)
    (arm, pair), = profile["arms"].items()
    read, intr, stop = open_aligned(args.serial)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        with open_follower(arm, pair["follower"], managed=True, resume=True) as follower:
            if any(follower.torque_states().values()):
                sys.exit("follower torque is on: release it first (tools/lab_day.sh recover)")
            for _ in range(10):
                read()
            print("扭力關閉。第 1 筆：手臂在停放位（量桌面）。之後手扶 follower 擺姿勢、手離開畫面再按 Enter；輸入 q 結束。")
            while True:
                cmd = input(f"[{len(rows) + 1}] Enter 拍這個姿勢／q 結束：").strip().lower()
                if cmd == "q":
                    break
                reading = follower.read()
                frames = [read() for _ in range(8)]
                after = follower.read()
                moved = max(abs(after[j] - reading[j]) for j in ARM)
                if moved > 1.0:
                    print(f"  手臂在拍攝中動了 {moved:.1f}°，這筆不收，扶穩再拍")
                    continue
                depth = np.median(np.stack([d for _, d in frames]), 0).astype(np.float32)
                name = f"pose{len(rows):02d}"
                np.savez_compressed(out / f"{name}.npz", depth=depth, bgr=frames[-1][0])
                rows.append({"name": name, "reading": {j: float(reading[j]) for j in ARM}})
                print(f"  已拍 {name}：" + " ".join(f"{reading[j]:6.1f}" for j in ARM))
    finally:
        stop()
    (out / "capture.json").write_text(json.dumps({"intrinsics": intr, "serial": args.serial, "poses": rows,
                                                  "follower_id": pair["follower"]["id"],
                                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print(f"{len(rows)} 筆 → {out}；接著跑 tools/lab_day.sh d455-fit")
    return 0


def _rot(rv):
    a = np.linalg.norm(rv)
    if a < 1e-12:
        return np.eye(3)
    k = rv / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def fit(args):
    import mujoco
    from agentic_sim.config import load
    from agentic_sim.mujoco_backend.model import build
    from agentic_sim.real_kinematics import ArmModel, find_calibration

    src = Path(args.capture)
    cap = json.loads((src / "capture.json").read_text())
    K = cap["intrinsics"]
    init = json.loads(Path(args.init).read_text())["mount_from_color"]
    R0, t0 = np.array(init["R"]), np.array(init["t"])
    model = ArmModel.from_file(find_calibration(cap["follower_id"]), ROOT)
    sc = build(load(ROOT / "configs/scenes/lab_20261125.yaml"), 1, colors=["red"])
    m, d = sc.model, mujoco.MjData(sc.model)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    rng = np.random.default_rng(0)

    def descends(b):
        while b > 0:
            if b == base:
                return True
            b = m.body_parentid[b]
        return False
    geoms = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and descends(m.geom_bodyid[g]):
            mid = m.geom_dataid[g]
            vert = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            geoms.append((g, vert[rng.choice(len(vert), min(len(vert), 400), replace=False)]))

    def mesh(reading):
        d.qpos[:] = 0
        d.qpos[sc.arm_qpos] = model.to_shared(reading) + sc.arm_offset
        mujoco.mj_kinematics(m, d)
        return np.concatenate([d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T for g, v in geoms])

    def cam_points(depth, step=2):
        vv, uu = np.nonzero(depth[::step, ::step] > 0.15)
        vv, uu = vv * step, uu * step
        z = depth[vv, uu]
        return np.stack([(uu - K["cx"]) / K["fx"] * z, (vv - K["cy"]) / K["fy"] * z, z], 1)

    poses = []
    for p in cap["poses"]:
        z = np.load(src / f"{p['name']}.npz")
        poses.append({"name": p["name"], "reading": p["reading"], "pc": cam_points(z["depth"]), "mesh": mesh(p["reading"])})
    pm0 = poses[0]["pc"] @ R0.T + t0
    table_sel = (np.abs(pm0[:, 2]) < 0.03) & (np.hypot(pm0[:, 0], pm0[:, 1]) > 0.15) & (pm0[:, 1] < 0.35) \
        & ~((pm0[:, 0] > 0.10) & (pm0[:, 0] < 0.52) & (pm0[:, 1] > -0.30) & (pm0[:, 1] < 0.34))
    table = poses[0]["pc"][np.nonzero(table_sel)[0][rng.choice(int(table_sel.sum()), min(4000, int(table_sel.sum())), replace=False)]]

    def arm_sel(pm):
        return (pm[:, 2] > 0.045) & (np.hypot(pm[:, 0], pm[:, 1]) < 0.45) & (pm[:, 1] < 0.30) & (pm[:, 0] > -0.15)

    def nearest(a, b):
        best, idx = np.full(len(a), np.inf), np.zeros(len(a), int)
        for s in range(0, len(b), 3000):
            dd = ((a[:, None] - b[None, s:s + 3000]) ** 2).sum(-1)
            j = dd.argmin(1)
            dv = dd[np.arange(len(a)), j]
            up = dv < best
            best[up], idx[up] = dv[up], j[up] + s
        return np.sqrt(best), idx

    th = np.zeros(6)
    for it in range(args.iters):
        R, t = _rot(th[:3]) @ R0, t0 + th[3:]
        blocks = []
        for p in poses:
            pm = p["pc"] @ R.T + t
            k = np.nonzero(arm_sel(pm))[0]
            k = k[rng.choice(len(k), min(len(k), 800), replace=False)] if len(k) else k
            dist, j = nearest(pm[k], p["mesh"])
            keep = dist < (0.04 if it < 3 else 0.02 if it < 6 else 0.012)
            blocks.append((p, k[keep], j[keep]))

        def resid(x):
            R_, t_ = _rot(x[:3]) @ R0, t0 + x[3:]
            out = [((p["pc"][k] @ R_.T + t_) - p["mesh"][j]).ravel() for p, k, j in blocks]
            out.append(np.repeat((table @ R_.T + t_)[:, 2] - TABLE_Z, 6))
            return np.concatenate(out)
        r0 = resid(th)
        J = np.stack([(resid(th + e) - r0) / 1e-4 for e in np.eye(6) * 1e-4], 1)
        w = 1 / np.maximum(1, np.abs(r0) / 0.005)
        th = th - np.linalg.solve(J.T @ (J * w[:, None]) + 1e-4 * np.eye(6), J.T @ (w * r0))
        print(f"iter {it}: rms {np.sqrt(np.mean(r0 ** 2)) * 1000:.1f} mm", flush=True)
    R, t = _rot(th[:3]) @ R0, t0 + th[3:]
    report = []
    for p in poses[1:]:
        pm = p["pc"] @ R.T + t
        sel = pm[arm_sel(pm)]
        sel = sel[rng.choice(len(sel), min(len(sel), 2500), replace=False)]
        dist, _ = nearest(sel, p["mesh"])
        report.append({"pose": p["name"], "median_mm": round(float(np.median(dist)) * 1000, 1),
                       "within_10mm": round(float(np.mean(dist < 0.01)), 2), "arm_points": int(len(sel))})
    tz = (table @ R.T + t)[:, 2]
    result = {"mount_from_color": {"R": R.tolist(), "t": t.tolist()},
              "frame": "right follower mount: +X forward, +Y left, +Z up, origin on the table below the base (m); "
                       "columns of R = colour camera x/y/z axes",
              "method": "arm depth (SO-101 meshes, calibration-file joint map) + table plane; tools/finals_calib.py",
              "capture": str(src), "init": args.init, "intrinsics_640x480": K,
              "change_from_init": {"rotation_deg": round(math.degrees(np.linalg.norm(th[:3])), 2),
                                   "translation_mm": np.round(th[3:] * 1000, 1).tolist()},
              "table_z_mm": {"median": round(float(np.median(tz)) * 1000, 1)},
              "per_pose": report,
              "median_mm": round(float(np.median([r["median_mm"] for r in report])), 1) if report else None}
    Path(args.out).write_text(json.dumps(result, indent=1))
    for r in report:
        print(r)
    print(f"camera moved {result['change_from_init']} from the initial pose; table z {result['table_z_mm']['median']} mm; "
          f"median arm residual {result['median_mm']} mm -> {args.out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--serial", required=True)
    c.add_argument("--config", default=str(ROOT / "outputs/single.local.yaml"))
    c.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--capture", required=True)
    f.add_argument("--init", default=str(ROOT / "outputs/d455/pose_261125.json"))
    f.add_argument("--out", required=True)
    f.add_argument("--iters", type=int, default=10)
    a = ap.parse_args(argv)
    return capture(a) if a.cmd == "capture" else fit(a)


if __name__ == "__main__":
    sys.exit(main())
