"""Cross-engine comparison (issue #3 P4): the same scripted EE trajectory in Isaac and MuJoCo.

  DISPLAY= ISAAC_CONTAINER_NAME=ars-plot docker/isaac/run.sh /isaac-sim/python.sh tools/compare_engines.py \
      [--trajectories pick_place reach_sweep] [--dir outputs/compare] [--out outputs/figures/compare]

Inputs: outputs/compare/<trajectory>/<engine>_<run>.{csv,json} from tools/run_trajectory.sh (one row per 60 Hz control
frame, aligned by simulated time). Run 1 is plotted; all runs feed the repeatability numbers. Writes per trajectory:
  <traj>_overview.png  target / EE height, tracking error, jaw angle (pick_place) or z sag (reach_sweep)
  <traj>_joints.png    the five arm joints, measured (solid) and commanded (dashed), both engines
  <traj>_block.png     block position (pick_place)
  compare_summary.json all numbers (per phase, at rest / moving, repeatability, results)
Differences are reported only; nothing is tuned to make the engines agree.
"""

import argparse
import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from agentic_sim.teleop.trajectory import Trajectory  # noqa: E402

JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
COLORS = {"isaac": "#2a6fdb", "mujoco": "#e8862a"}
LABELS = {"isaac": "Isaac Sim (PhysX, Workshop gains)", "mujoco": "MuJoCo 4.0.1 (Menagerie gains)"}
ap = argparse.ArgumentParser()
ap.add_argument("--trajectories", nargs="*", default=["pick_place", "reach_sweep"])
ap.add_argument("--dir", default="outputs/compare")
ap.add_argument("--out", default="outputs/figures/compare")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def load_run(path):
    rows = list(csv.DictReader(open(path)))
    cols = ["t_sim", "tx", "ty", "tz", "ex", "ey", "ez", "bx", "by", "bz", "grip_cmd", "jaw",
            *[f"qcmd_{j}" for j in JOINTS], *[f"q_{j}" for j in JOINTS]]
    d = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    d["t"] = np.round(d.pop("t_sim"), 4)
    d["target"] = np.stack([d["tx"], d["ty"], d["tz"]], 1)
    d["ee"] = np.stack([d["ex"], d["ey"], d["ez"]], 1)
    d["block"] = np.stack([d["bx"], d["by"], d["bz"]], 1)
    d["err_mm"] = np.linalg.norm(d["target"] - d["ee"], axis=1) * 1000
    d["sag_mm"] = (d["tz"] - d["ez"]) * 1000
    d["q"] = np.stack([d[f"q_{j}"] for j in JOINTS], 1)
    d["qcmd"] = np.stack([d[f"qcmd_{j}"] for j in JOINTS], 1)
    return d


def runs(traj, engine):
    out = []
    for p in sorted(glob.glob(os.path.join(a.dir, traj, f"{engine}_[0-9]*.csv"))):
        res = os.path.splitext(p)[0] + ".json"
        out.append((load_run(p), json.load(open(res)) if os.path.exists(res) else {}))
    return out


def pct(x):
    x = np.asarray(x)
    if x.size == 0:
        return None
    return {"mean": round(float(x.mean()), 3), "p95": round(float(np.percentile(x, 95)), 3), "max": round(float(x.max()), 3)}


def shade_phases(ax, tr, tmax):
    edges, lab = [], None
    for k0, k1 in zip(tr.keys[:-1], tr.keys[1:]):
        if k1.label != lab:
            edges.append((k0.t, k1.label))
            lab = k1.label
    for i, (t0, name) in enumerate(edges):
        t1 = edges[i + 1][0] if i + 1 < len(edges) else tmax
        if i % 2 == 0:
            ax.axvspan(t0, t1, color="#000000", alpha=0.04, lw=0)
        ax.text((t0 + t1) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7,
                color="#666")


summary = {}
for traj in a.trajectories:
    tr = Trajectory.load(f"configs/trajectories/{traj}.yaml")
    data = {e: runs(traj, e) for e in ("isaac", "mujoco")}
    if not all(data.values()):
        print(f"{traj}: missing runs {[e for e, v in data.items() if not v]}")
        continue
    # target velocity from the trajectory itself: 'at rest' = target not moving
    S = {}
    for e, rs in data.items():
        d0 = rs[0][0]
        moving = np.r_[0, np.linalg.norm(np.diff(d0["target"], axis=0), axis=1)] > 1e-7
        phases = np.array([tr.phase(t - 1e-6) for t in d0["t"]])
        per_phase = {ph: pct(d0["err_mm"][phases == ph]) for ph in dict.fromkeys(phases) if ph}
        # repeatability: largest EE distance between runs at the same frame
        rep = None
        if len(rs) > 1:
            n = min(len(r[0]["t"]) for r in rs)
            rep = round(float(max(np.linalg.norm(r[0]["ee"][:n] - rs[0][0]["ee"][:n], axis=1).max() for r in rs[1:]))
                        * 1000, 3)
        S[e] = {"runs": len(rs), "results": [r[1] for r in rs],
                "success": [bool(r[1].get("block_in_placement")) for r in rs],
                "tracking_err_mm": pct(d0["err_mm"]), "tracking_err_at_rest_mm": pct(d0["err_mm"][~moving]),
                "tracking_err_moving_mm": pct(d0["err_mm"][moving]), "per_phase_mm": per_phase,
                "z_sag_at_rest_mm": pct(d0["sag_mm"][~moving]),
                "repeatability_ee_max_mm": rep}
        if traj == "pick_place":
            grasp = phases == "grasp"
            S[e]["jaw_end_of_grasp_rad"] = round(float(d0["jaw"][grasp][-1]), 4)
            S[e]["block_lifted_max_z_m"] = round(float(d0["block"][:, 2].max()), 4)
            S[e]["block_final_m"] = [round(float(x), 4) for x in d0["block"][-1]]
    di, dm = data["isaac"][0][0], data["mujoco"][0][0]
    n = min(len(di["t"]), len(dm["t"]))
    S["between_engines"] = {
        "ee_distance_mm": pct(np.linalg.norm(di["ee"][:n] - dm["ee"][:n], axis=1) * 1000),
        "joint_diff_rad": {j: round(float(np.abs(di["q"][:n, k] - dm["q"][:n, k]).max()), 4)
                           for k, j in enumerate(JOINTS)},
        "command_diff_rad": round(float(np.abs(di["qcmd"][:n] - dm["qcmd"][:n]).max()), 5),
    }
    if traj == "pick_place":
        S["between_engines"]["block_distance_mm"] = pct(np.linalg.norm(di["block"][:n] - dm["block"][:n], axis=1) * 1000)
    # physics only: MuJoCo replaying Isaac run 1's joint / jaw commands frame by frame (no IK in the loop)
    rp = os.path.join(a.dir, traj, "mujoco_replay.csv")
    dr = load_run(rp) if os.path.exists(rp) else None
    if dr is not None:
        n2 = min(len(di["t"]), len(dr["t"]))
        S["physics_only_same_commands"] = {
            "command_diff_rad": round(float(np.abs(di["qcmd"][:n2] - dr["qcmd"][:n2]).max()), 6),
            "ee_distance_mm": pct(np.linalg.norm(di["ee"][:n2] - dr["ee"][:n2], axis=1) * 1000),
            "isaac_tracking_err_mm": pct(di["err_mm"][:n2]), "mujoco_tracking_err_mm": pct(dr["err_mm"][:n2]),
            "joint_diff_rad": {j: round(float(np.abs(di["q"][:n2, k] - dr["q"][:n2, k]).max()), 4)
                               for k, j in enumerate(JOINTS)},
            "result": json.load(open(os.path.splitext(rp)[0] + ".json")) if os.path.exists(os.path.splitext(rp)[0] + ".json") else None,
        }
        if traj == "pick_place":
            S["physics_only_same_commands"]["block_distance_mm"] = pct(
                np.linalg.norm(di["block"][:n2] - dr["block"][:n2], axis=1) * 1000)
            S["physics_only_same_commands"]["jaw_diff_rad"] = round(float(np.abs(di["jaw"][:n2] - dr["jaw"][:n2]).max()), 4)
        fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
        axes[0].plot(di["t"], di["err_mm"], color=COLORS["isaac"], lw=1.4, label="Isaac: |target - EE|")
        axes[0].plot(dr["t"], dr["err_mm"], color=COLORS["mujoco"], lw=1.4, label="MuJoCo replaying Isaac's commands")
        axes[0].set_ylabel("tracking error (mm)")
        axes[1].plot(di["t"][:n2], np.linalg.norm(di["ee"][:n2] - dr["ee"][:n2], axis=1) * 1000, color="#555", lw=1.4,
                     label="|EE Isaac - EE MuJoCo| (same joint commands)")
        axes[1].set_ylabel("EE difference (mm)")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, frameon=False, loc="upper right")
            shade_phases(ax, tr, di["t"][-1])
        axes[-1].set_xlabel("simulated time (s)")
        fig.suptitle(f"{traj}: physics only (identical joint and jaw commands in both engines)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{traj}_physics_only.png"), dpi=120)
        plt.close(fig)
    summary[traj] = S

    # ---- overview figure
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    t = di["t"]
    axes[0].plot(t, di["tz"] * 1000, color="#333", lw=1.2, ls="--", label="target")
    for e, rs in data.items():
        d = rs[0][0]
        axes[0].plot(d["t"], d["ez"] * 1000, color=COLORS[e], lw=1.6, label=f"EE: {LABELS[e]}")
        for r, _ in rs[1:]:
            axes[1].plot(r["t"], r["err_mm"], color=COLORS[e], lw=0.6, alpha=0.4)
        axes[1].plot(d["t"], d["err_mm"], color=COLORS[e], lw=1.4, label=f"{e} (run 1; thin = runs 2-{len(rs)})")
        if traj == "pick_place":
            axes[2].plot(d["t"], d["jaw"], color=COLORS[e], lw=1.6, label=f"jaw: {e}")
        else:
            axes[2].plot(d["t"], d["sag_mm"], color=COLORS[e], lw=1.4, label=f"z sag: {e}")
    axes[0].set_ylabel("height z (mm)")
    axes[1].set_ylabel("|target - EE| (mm)")
    if traj == "pick_place":
        axes[2].plot(t, di["grip_cmd"], color="#333", lw=1.0, ls="--", label="jaw command")
        axes[2].axhline(0.17, color="#999", lw=0.8, ls=":")
        axes[2].text(t[0], 0.19, "block contact (about 0.17 rad)", fontsize=7, color="#777")
        axes[2].set_ylabel("jaw angle (rad)")
    else:
        axes[2].set_ylabel("target z - EE z (mm)")
        axes[2].axhline(0, color="#999", lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        shade_phases(ax, tr, t[-1])
    axes[-1].set_xlabel("simulated time (s)")
    fig.suptitle(f"{traj}: same EE target sequence in both engines (60 Hz control, 240 Hz physics)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"{traj}_overview.png"), dpi=120)
    plt.close(fig)

    # ---- joints
    fig, axes = plt.subplots(5, 1, figsize=(11, 9.5), sharex=True)
    for k, j in enumerate(JOINTS):
        for e, rs in data.items():
            d = rs[0][0]
            axes[k].plot(d["t"], d["q"][:, k], color=COLORS[e], lw=1.5, label=f"{e} measured")
            axes[k].plot(d["t"], d["qcmd"][:, k], color=COLORS[e], lw=0.9, ls="--", alpha=0.8, label=f"{e} command")
        axes[k].set_ylabel(f"{j} (rad)", fontsize=8)
        axes[k].grid(alpha=0.25)
        shade_phases(axes[k], tr, t[-1]) if k == 0 else None
    axes[0].legend(fontsize=7, frameon=False, ncol=4, loc="lower right")
    axes[-1].set_xlabel("simulated time (s)")
    fig.suptitle(f"{traj}: arm joints (shared convention; the commands come from the same controller)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"{traj}_joints.png"), dpi=120)
    plt.close(fig)

    if traj == "pick_place":
        fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
        for k, c in enumerate("xyz"):
            axes[k].plot(t, di["target"][:, k] * 1000, color="#333", lw=1.0, ls="--", label="EE target")
            for e, rs in data.items():
                d = rs[0][0]
                axes[k].plot(d["t"], d["block"][:, k] * 1000, color=COLORS[e], lw=1.6, label=f"block: {e}")
            axes[k].set_ylabel(f"{c} (mm)")
            axes[k].grid(alpha=0.25)
            shade_phases(axes[k], tr, t[-1]) if k == 0 else None
        axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
        axes[-1].set_xlabel("simulated time (s)")
        fig.suptitle("pick_place: block position (centre) in both engines", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{traj}_block.png"), dpi=120)
        plt.close(fig)

with open(os.path.join(a.out, "compare_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
for traj, S in summary.items():
    print(f"== {traj}")
    for e in ("isaac", "mujoco"):
        s = S[e]
        print(f"  {e:6s} success {s['success']}  err rest {s['tracking_err_at_rest_mm']}  moving {s['tracking_err_moving_mm']}"
              f"  sag rest {s['z_sag_at_rest_mm']}  repeat {s['repeatability_ee_max_mm']} mm"
              + (f"  jaw {s['jaw_end_of_grasp_rad']} lift {s['block_lifted_max_z_m']}" if traj == "pick_place" else ""))
    print("  between engines:", json.dumps(S["between_engines"]))
    if "physics_only_same_commands" in S:
        p = dict(S["physics_only_same_commands"])
        p.pop("result", None)
        print("  physics only:", json.dumps(p))
print(f"wrote {a.out}/")
