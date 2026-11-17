"""Plot tools/reach_map.py output: reachable region, vertical-only vs tilted grasp (side view and top view).

  DISPLAY= ISAAC_CONTAINER_NAME=ars-plot docker/isaac/run.sh /isaac-sim/python.sh tools/plot_reach.py \
      [--data outputs/reach_map.json] [--out outputs/figures/reach_map.png]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="outputs/reach_map.json")
ap.add_argument("--out", default="outputs/figures/reach_map.png")
ap.add_argument("--scene", default="configs/scenes/lab_current_minimal.yaml")
a = ap.parse_args()
d = json.load(open(a.data))
keys = sorted(d["maps"], key=float)
old, new = keys[0], keys[-1]
pan = d["pan_origin"]

# 0 = unreachable, 1 = reachable with the new limit only, 2 = reachable vertical-only (also new)
VERT, TILT = "#2a6fdb", "#9db8e8"
cmap = ListedColormap(["#ffffff", TILT, VERT])


def region(m_old, m_new):
    o = np.array([[v is not None for v in row] for row in m_old])
    n = np.array([[v is not None for v in row] for row in m_new])
    return np.where(o, 2, np.where(n, 1, 0))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.0, 1.15]})
r, z = np.array(d["r"]), np.array(d["z"])
g = region(d["maps"][old]["side"], d["maps"][new]["side"])
step = r[1] - r[0]
ax1.imshow(g, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="equal", interpolation="nearest",
           extent=[r[0] - step / 2, r[-1] + step / 2, z[0] - step / 2, z[-1] + step / 2])
ax1.axhline(d["grasp_z"], color="#c0392b", lw=1, ls="--")
ax1.text(r[-1], d["grasp_z"] + 0.004, "grasp height (z min)", color="#c0392b", fontsize=8, ha="right")
ax1.set_xlabel("distance from the pan axis r (m)")
ax1.set_ylabel("height z above the table (m)")
ax1.set_title("Side view (vertical plane through the pan axis)", fontsize=10)
ax1.grid(alpha=0.2)

x, y = np.array(d["x"]), np.array(d["y"])
gt = region(d["maps"][old]["top"], d["maps"][new]["top"])
st = x[1] - x[0]
# top view drawn like the panel's top view: +X up, +Y left
ax2.imshow(gt, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="equal", interpolation="nearest",
           extent=[y[0] - st / 2, y[-1] + st / 2, x[0] - st / 2, x[-1] + st / 2])
ax2.invert_xaxis()
try:
    import yaml
    scene = yaml.safe_load(open(a.scene))
    for name, col in (("storage_pink", "#e75480"), ("placement_beige", "#b08d57")):
        t = scene["trays"][name]
        L, W = t["outer_size"]["value"]
        cx, cy = t["center_xy"]["value"]
        ax2.add_patch(mpatches.Rectangle((cy - W / 2, cx - L / 2), W, L, fill=False, ec=col, lw=1.6))
        ax2.text(cy, cx + L / 2 + 0.008, name.replace("_", " "), color=col, fontsize=8, ha="center")
except Exception as e:  # noqa: BLE001
    print(f"trays not drawn: {e}")
ax2.plot(pan[1], pan[0], "k+", ms=10)
ax2.text(pan[1] - 0.01, pan[0] - 0.03, "pan axis", fontsize=8)
ax2.set_xlabel("y (m, +Y left)")
ax2.set_ylabel("x (m, forward)")
ax2.set_title(f"Top view at grasp height z = {d['grasp_z']:.3f} m", fontsize=10)
ax2.grid(alpha=0.2)

handles = [mpatches.Patch(color=VERT, label=f"vertical grasp (tilt <= {old} deg): earlier reach"),
           mpatches.Patch(color=TILT, label=f"added by tilting up to {new} deg")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("SO-101 teleop reach: IK only (joint limits + tilt limit; self-collision not modelled)", fontsize=11)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out, dpi=130)
print(f"wrote {a.out}")
