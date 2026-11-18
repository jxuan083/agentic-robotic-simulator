"""Figures for issue #3 P3 (MuJoCo backend): server load time / memory per backend, IK boundary fix.

  DISPLAY= ISAAC_CONTAINER_NAME=ars-plot docker/isaac/run.sh /isaac-sim/python.sh tools/plot_mujoco_p3.py \
      [--isaac outputs/server_check_v3.json] [--mujoco outputs/mujoco/server_check.json] \
      [--ik outputs/ik_boundary_check.json] [--out outputs/figures/mujoco_p3.png]

(matplotlib is in the Isaac container, not in the host uv environment.)
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--isaac", default="outputs/server_check_v3.json")
ap.add_argument("--mujoco", default="outputs/mujoco/server_check.json")
ap.add_argument("--ik", default="outputs/ik_boundary_check.json")
ap.add_argument("--out", default="outputs/figures/mujoco_p3.png")
a = ap.parse_args()

BLUE, ORANGE, GREY = "#2a6fdb", "#e8862a", "#888888"
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.3))
for path, name, color in ((a.isaac, "Isaac (Kit, container)", BLUE), (a.mujoco, "MuJoCo (host process)", ORANGE)):
    if not os.path.exists(path):
        continue
    loads = json.load(open(path))["loads"]
    xs = [x["i"] + 1 for x in loads]
    ax1.plot(xs, [x["client_s"] for x in loads], "-o", color=color, lw=2, ms=4, label=name)
    r0 = loads[0]["rss_mb"]
    ax2.plot(xs, [x["rss_mb"] - r0 for x in loads], "-o", color=color, lw=2, ms=4, label=name)
ax1.set_yscale("log")
ax1.set_yticks([0.2, 0.5, 1, 2, 3])
ax1.set_yticklabels(["0.2", "0.5", "1", "2", "3"])
ax1.minorticks_off()
ax1.set_xlabel("experiment load on the running server (#)")
ax1.set_ylabel("client-side time (s, log)")
ax1.set_title("Load an experiment on a running server", fontsize=10)
ax2.set_xlabel("experiment load (#)")
ax2.set_ylabel("process memory since first load (MB)")
ax2.set_title("Memory per scene rebuild", fontsize=10)
for ax in (ax1, ax2):
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

if os.path.exists(a.ik):
    d = json.load(open(a.ik))
    heights = sorted({k.split()[0] for k in d["current"]}, key=lambda s: float(s[2:]))

    def per_height(rep):
        return [sum(v["transient"] for k, v in rep.items() if k.split()[0] == h) for h in heights]

    n = [sum(v["runs"] for k, v in d["current"].items() if k.split()[0] == h) for h in heights]
    xs = range(len(heights))
    w = 0.38
    if "before" in d:
        b = per_height(d["before"])
        ax3.bar([x - w / 2 for x in xs], b, w, color=GREY, label=f"before ({d.get('rev', 'old')})")
    c = per_height(d["current"])
    ax3.bar([x + w / 2 for x in xs], c, w, color=BLUE, label="after (shortened null-space step)")
    ax3.set_xticks(list(xs))
    ax3.set_xticklabels([f"{h[2:]} m\n({k} runs)" for h, k in zip(heights, n)], fontsize=8)
    ax3.set_xlabel("EE height while holding a move key (5 directions)")
    ax3.set_ylabel("runs with a transient rejection")
    ax3.set_title("Rejected, then moved on > 5 mm (not the real limit)", fontsize=10)
    ax3.legend(fontsize=8, frameon=False, loc="upper left")
    ax3.grid(alpha=0.25, axis="y")
fig.tight_layout()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out, dpi=130)
print(f"wrote {a.out}")
