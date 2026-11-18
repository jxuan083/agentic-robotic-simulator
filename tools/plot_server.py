"""Figures for the persistent-server measurements (issue #3, P2).

  DISPLAY= ISAAC_CONTAINER_NAME=ars-plot docker/isaac/run.sh /isaac-sim/python.sh tools/plot_server.py \
      [--probe_dir outputs/probe] [--check outputs/server_check.json] [--out outputs/figures/server_rebuild.png]

Left: process memory growth per scene rebuild for three teardown variants (tools/probe_scene_rebuild.py).
Right: client-side time to load an experiment on the running server (tools/server_check.py).
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

ap = argparse.ArgumentParser()
ap.add_argument("--probe_dir", default="outputs/probe")
ap.add_argument("--check", default="outputs/server_check.json")
ap.add_argument("--out", default="outputs/figures/server_rebuild.png")
a = ap.parse_args()

VARIANTS = [("scene_rebuild.jsonl", "teardown only", "#2a6fdb"),
            ("rebuild_undo.jsonl", "+ clear undo stack", "#e8862a"),
            ("rebuild_undo+close.jsonl", "+ clear undo + close stage", "#3a9d5d")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for fname, label, color in VARIANTS:
    p = os.path.join(a.probe_dir, fname)
    if not os.path.exists(p):
        continue
    rows = [json.loads(x) for x in open(p)]
    r0 = rows[0]["rss_mb"]
    xs = [r["round"] for r in rows]
    ys = [r["rss_mb"] - r0 for r in rows]
    ax1.plot(xs, ys, "-o", color=color, lw=2, ms=4, label=label)
    if fname == VARIANTS[0][0]:  # the longest run; the shorter variants end on top of each other
        ax1.annotate(f"{ys[-1]:.0f} MB", (xs[-1], ys[-1]), textcoords="offset points", xytext=(4, 0), fontsize=8,
                     va="center", color="#333")
ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax1.set_xlabel("scene rebuild (round)")
ax1.set_ylabel("process RSS growth since round 0 (MB)")
ax1.set_title("Memory per rebuild in one Kit process (~65 MB each)", fontsize=10)
ax1.legend(fontsize=8, frameon=False, loc="upper left")
ax1.grid(alpha=0.25)

if os.path.exists(a.check):
    rep = json.load(open(a.check))
    loads = rep["loads"]
    xs = [x["i"] for x in loads]
    ax2.bar([x - 0.2 for x in xs], [x["client_s"] for x in loads], width=0.4, color="#2a6fdb",
            label="client: load_experiment round trip")
    ax2.bar([x + 0.2 for x in xs], [x["server_s"] for x in loads], width=0.4, color="#9db8e8",
            label="server: build new scene only")
    ax2.axhline(3.0, color="#c0392b", lw=1.2, ls="--")
    ax2.text(xs[-1] + 0.5, 3.0, "3 s target", color="#c0392b", fontsize=8, va="bottom", ha="right")
    ax2.set_ylim(0, 3.6)
    ax2.set_xlabel("load on the running server")
    ax2.set_ylabel("seconds")
    ax2.set_title("Loading an experiment on the running server (cold start ~30 s)", fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.grid(alpha=0.25, axis="y")

fig.tight_layout()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out, dpi=130)
print(f"wrote {a.out}")
