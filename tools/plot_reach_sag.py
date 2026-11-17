"""Gravity sag vs reach from acceptance CSVs: how far the actual EE hangs below the target when the arm reaches out.

  DISPLAY= ISAAC_CONTAINER_NAME=ars-plot docker/isaac/run.sh /isaac-sim/python.sh tools/plot_reach_sag.py \
      [--csv outputs/acceptance/minimal_pick_place/far_reach.csv ...] [--out outputs/figures/reach_sag.png]

Only settled frames count (target unchanged for the last 0.5 s, not paused), so the sag is not mixed with motion lag.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = "outputs/acceptance/minimal_pick_place"
ap = argparse.ArgumentParser()
ap.add_argument("--csv", nargs="*", default=[f"{D}/far_reach.csv", f"{D}/robustness.csv",
                                             f"{D}/keyboard_pick_place.csv"])
ap.add_argument("--out", default="outputs/figures/reach_sag.png")
a = ap.parse_args()

pts = []
for p in a.csv:
    if not os.path.exists(p):
        continue
    rows = list(csv.DictReader(open(p)))
    last_change, prev = -1e9, None
    for r in rows:
        t = float(r["t_wall"])
        tgt = (float(r["tx"]), float(r["ty"]), float(r["tz"]))
        if prev is None or max(abs(x - y) for x, y in zip(tgt, prev)) > 1e-6:
            last_change = t
        prev = tgt
        if r["paused"] == "1" or t - last_change < 0.5:
            continue
        pan = (0.0231, 0.0208)
        reach = float(np.hypot(tgt[0] - pan[0], tgt[1] - pan[1]))
        pts.append((reach, (tgt[2] - float(r["ez"])) * 1000, float(r["follow_err_mm"]), tgt[2]))

pts = np.array(pts)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bins = np.arange(0.10, 0.45, 0.02)
idx = np.digitize(pts[:, 0], bins)
xs, med, lo, hi = [], [], [], []
for i in range(1, len(bins)):
    s = pts[idx == i, 1]
    if len(s) >= 5:
        xs.append(0.5 * (bins[i - 1] + bins[i]))
        med.append(np.median(s))
        lo.append(np.percentile(s, 10))
        hi.append(np.percentile(s, 90))
ax.scatter(pts[:, 0], pts[:, 1], s=4, color="#9db8e8", alpha=0.35, label="settled frames")
ax.plot(xs, med, "-o", color="#2a6fdb", lw=2, ms=4, label="median per 2 cm")
ax.fill_between(xs, lo, hi, color="#2a6fdb", alpha=0.12, label="10-90 %")
ax.axvline(0.28, color="#888", lw=1, ls="--")
ax.text(0.282, ax.get_ylim()[1] * 0.92, "earlier reach at grasp height", fontsize=8, color="#555")
ax.set_xlabel("target distance from the pan axis (m)")
ax.set_ylabel("EE below target, z sag (mm)")
ax.set_title("Gravity sag grows with reach (workshop joint gains, settled frames)", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out, dpi=130)
with open(os.path.splitext(a.out)[0] + ".csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["reach_m", "z_sag_mm", "follow_err_mm", "target_z_m"])
    w.writerows(np.round(pts, 5).tolist())
print(f"wrote {a.out} ({len(pts)} settled frames)")
