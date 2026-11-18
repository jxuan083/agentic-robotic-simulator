"""Reach before / after the base-height fix (base bottom 30 mm above the table vs on the table), side view.

  DISPLAY= ISAAC_CONTAINER_NAME=ars-plot docker/isaac/run.sh /isaac-sim/python.sh tools/plot_reach_base.py \
      [--old outputs/reach_map_base_old.json] [--new outputs/reach_map.json] [--out outputs/figures/reach_base_fix.png]

Inputs come from tools/reach_map.py (--base_z 0.0 for the old height). Also writes <out>.json with the summary
numbers: farthest reach per height and highest reachable point per distance, for both heights and tilt limits.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--old", default="outputs/reach_map_base_old.json")
ap.add_argument("--new", default="outputs/reach_map.json")
ap.add_argument("--out", default="outputs/figures/reach_base_fix.png")
a = ap.parse_args()


def grid(d, tilt):
    side = d["maps"][tilt]["side"]
    return np.array([[v is not None for v in row] for row in side]), np.array(d["r"]), np.array(d["z"])


def far_at(mask, rs, zs, z):
    i = int(np.argmin(np.abs(zs - z)))
    r = rs[mask[i]]
    return float(r.max()) if r.size else None


def top_at(mask, rs, zs, r):
    j = int(np.argmin(np.abs(rs - r)))
    z = zs[mask[:, j]]
    return float(z.max()) if z.size else None


old, new = json.load(open(a.old)), json.load(open(a.new))
tilts = sorted(new["maps"], key=float)
fig, axes = plt.subplots(1, len(tilts), figsize=(6.2 * len(tilts), 4.6), sharey=True)
summary = {"base_z": {"old": old.get("base_pos", [0, 0, 0])[2], "new": new.get("base_pos", [0, 0, 0])[2]}}
for ax, t in zip(np.atleast_1d(axes), tilts):
    for d, name, color, alpha in ((old, "base 30 mm above the table (before)", "#999999", 0.35),
                                  (new, "base on the table (after)", "#2a6fdb", 0.45)):
        m, rs, zs = grid(d, t)
        ax.contourf(rs, zs, m.astype(float), levels=[0.5, 1.5], colors=[color], alpha=alpha)
        ax.contour(rs, zs, m.astype(float), levels=[0.5], colors=[color], linewidths=1.6)
        ax.plot([], [], color=color, lw=6, alpha=0.6, label=name)
        key = "old" if d is old else "new"
        summary.setdefault(t, {})[key] = {
            "farthest_r_at_z": {str(z): far_at(m, rs, zs, z) for z in (0.024, 0.05, 0.10, 0.15, 0.20, 0.25)},
            "highest_z_at_r": {str(r): top_at(m, rs, zs, r) for r in (0.13, 0.20, 0.25, 0.30)},
        }
    ax.axhline(0.0, color="#555", lw=1)
    ax.text(0.005, 0.004, "table top", fontsize=8, color="#555")
    ax.set_title(f"tilt limit {t} deg" + (" (vertical only)" if float(t) < 10 else ""), fontsize=10)
    ax.set_xlabel("horizontal distance from the pan axis (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
np.atleast_1d(axes)[0].set_ylabel("EE height above the table (m)")
fig.suptitle("Reachable EE positions (IK, joint limits; no self-collision model)", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out, dpi=130)
with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
print(f"wrote {a.out}")
