"""Figures for the task executor acceptance runs (issue #5); runs in the Isaac container (matplotlib there).

  DISPLAY= ISAAC_CONTAINER_NAME=ars-plot docker/isaac/run.sh /isaac-sim/python.sh tools/plot_task.py \
      [--root outputs/task] [--out outputs/figures/task] [--run 1]

Per scenario (<scenario>_timeline.png): both engines, EE height over simulated time with the running skill as
background bands, order changes / questions / answers as vertical lines, and the verified deliveries per colour.
Summary (summary.png + summary.json): checks passed per scenario and engine over all runs, time to finish, and the
stop latency of every cancel.
"""

import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SKILL_COLOR = {"pick": "#2a78d6", "place": "#eb6834", "hold": "#1baf7a", "home": "#eda100"}   # categorical 1-4
BLOCK_COLOR = {"red": "#e34948", "blue": "#2a78d6", "yellow": "#eda100"}                      # the blocks' colours
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ENGINES = ("isaac", "mujoco")
SCENARIOS = ("change_between_picks", "change_during_descend", "hold_return", "hold_deliver",
             "retrieve_over_delivery", "shortage_substitute")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="outputs/task")
ap.add_argument("--out", default="outputs/figures/task")
ap.add_argument("--run", default="1", help="run shown in the timelines")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def events(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def frames(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t_sim"]) for r in rows])
    ez = np.array([float(r["ez"]) for r in rows]) * 1000.0
    return t, ez


def ee_track(path):
    rows = list(csv.DictReader(open(path)))
    return (np.array([float(r["t_sim"]) for r in rows]),
            np.array([[float(r[k]) for k in ("ex", "ey", "ez")] for r in rows]))


def approach_stop(csv_path, t_change, window=0.1, eps=1e-4):
    """Time from the order change until the measured EE no longer moves in the direction it was moving just before
    (per-frame advance below eps m): when the arm stopped pursuing the old goal."""
    t, ee = ee_track(csv_path)
    i0 = int(np.searchsorted(t, t_change - 1e-6))
    j = int(np.searchsorted(t, t_change - window))
    d0 = ee[i0] - ee[max(j, 0)]
    n = float(np.linalg.norm(d0))
    if n < 1e-5:
        return 0.0
    d0 /= n
    for i in range(i0, len(t) - 1):
        if float(np.dot(ee[i + 1] - ee[i], d0)) < eps:
            return float(t[i + 1] - t_change)
    return None


def skill_spans(ev):
    spans, cur = [], None
    for e in ev:
        if e["kind"] == "skill_start":
            cur = (e["skill"], e["t"])
        elif e["kind"] == "skill_done" and cur:
            spans.append((cur[0], cur[1], e["t"], e["status"]))
            cur = None
    return spans


def timeline(name):
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex="col", gridspec_kw={"height_ratios": [3, 1.3]},
                             squeeze=False)
    any_data = False
    for col, eng in enumerate(ENGINES):
        base = os.path.join(a.root, name, f"{eng}_{a.run}")
        ev, fr = events(base + ".jsonl"), frames(base + ".csv")
        ax, axl = axes[0, col], axes[1, col]
        ax.set_title(f"{eng}", loc="left", fontsize=10, color=INK)
        if fr is None or not ev:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=INK2)
            continue
        any_data = True
        t, ez = fr
        for sk, t0, t1, st in skill_spans(ev):
            ax.axvspan(t0, t1, color=SKILL_COLOR.get(sk, "#999"), alpha=0.16, lw=0)
            if st == "cancelled":
                ax.text(t1, 3, "cancelled", rotation=90, fontsize=7, color=INK2, ha="right", va="bottom")
        ax.plot(t, ez, color=INK, lw=1.4)
        for e in ev:
            if e["kind"] == "order":
                ax.axvline(e["t"], color=INK, lw=1.2, ls="--")
                c = ", ".join(f"{v} {k}" for k, v in e["counts"].items())
                ax.text(e["t"], ax.get_ylim()[1] if False else 112, f" rev {e['rev']}: {c}", fontsize=7.5,
                        color=INK, va="top")
            elif e["kind"] == "question":
                ax.axvline(e["t"], color=INK2, lw=1.0, ls=":")
                ax.text(e["t"], 100, f" Q{e['id']} {e['qkind']}", fontsize=7.5, color=INK2, va="top")
            elif e["kind"] == "answer":
                ax.axvline(e["t"], color=INK2, lw=1.0, ls="-.")
                ax.text(e["t"], 90, f" answer: {e['choice']}" + (f" {e['color']}" if e.get("color") else ""),
                        fontsize=7.5, color=INK2, va="top")
        ax.set_ylim(0, 115)
        ax.set_ylabel("EE height (mm)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        # verified deliveries per colour (ledger)
        led_t, counts = [0.0], [{}]
        for e in ev:
            if "ledger" in e:
                led_t.append(e["t"])
                counts.append(dict(e["ledger"]["delivered"]))
        led_t.append(t[-1])
        counts.append(counts[-1])
        colors = sorted({c for d in counts for c in d})
        for c in colors:
            axl.step(led_t, [d.get(c, 0) for d in counts], where="post", color=BLOCK_COLOR.get(c, INK2), lw=2,
                     label=c)
        axl.set_ylim(-0.3, 3.3)
        axl.set_yticks([0, 1, 2, 3])
        axl.set_ylabel("delivered")
        axl.set_xlabel("simulated time (s)")
        axl.grid(axis="y", color=GRID, lw=0.6)
        axl.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=3)
    if not any_data:
        plt.close(fig)
        return False
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in SKILL_COLOR.values()]
    fig.legend(handles + [plt.Line2D([], [], color=INK, lw=1.4)], list(SKILL_COLOR) + ["EE height"],
               loc="upper right", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"{name}: skills, order changes and deliveries (run {a.run})", x=0.01, ha="left", fontsize=11,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(a.out, f"{name}_timeline.png"), dpi=130)
    plt.close(fig)
    return True


summary = {}
for name in SCENARIOS:
    timeline(name)
    for eng in ENGINES:
        runs = []
        for p in sorted(glob.glob(os.path.join(a.root, name, f"{eng}_[0-9]*.json"))):
            r = json.load(open(p))
            cancels = []
            for c in r.get("stop_latency", []):
                c = dict(c)
                c["approach_stopped_s"] = approach_stop(p[:-5] + ".csv", c["t_change"])
                cancels.append(c)
            r["stop_latency"] = cancels
            runs.append({"run": os.path.basename(p).split("_")[-1][:-5], "passed": r["passed"],
                         "checks": r["checks_total"], "t_sim": r["t_sim"],
                         "failed": [c["name"] for c in r["checks"] if not c["ok"]],
                         "stop_latency": r.get("stop_latency", []),
                         "hold_still": next((c for c in r["checks"] if c["name"] == "hold_still"), None)})
        summary.setdefault(name, {})[eng] = runs
with open(os.path.join(a.out, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)

# summary figure: time to finish (dots per run) and stop latency (dots per cancel)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 2]})
eng_col = {"isaac": "#2a78d6", "mujoco": "#eb6834"}
for i, name in enumerate(SCENARIOS):
    for k, eng in enumerate(ENGINES):
        runs = summary[name][eng]
        ys = [r["t_sim"] for r in runs]
        y0 = i + (k - 0.5) * 0.3
        ax1.scatter(ys, [y0] * len(ys), s=26, color=eng_col[eng], edgecolor="white", lw=0.8, zorder=3,
                    label=eng if i == 0 else None)
        ok = sum(r["passed"] == r["checks"] for r in runs)
        if runs:
            ax1.text(max(ys) + 0.6, y0, f"{ok}/{len(runs)} passed", va="center", fontsize=7.5, color=INK2)
ax1.set_yticks(range(len(SCENARIOS)))
ax1.set_yticklabels(SCENARIOS)
ax1.invert_yaxis()
ax1.set_xlabel("simulated time to finish (s)")
ax1.grid(axis="x", color=GRID, lw=0.6)
ax1.legend(frameon=False, fontsize=8, loc="lower right")
ax1.set_title("Scenario runs (dot = one run)", loc="left", fontsize=10, color=INK)
lat_names = [n for n in SCENARIOS if any(r["stop_latency"] for e in ENGINES for r in summary[n][e])]
for i, name in enumerate(lat_names):
    for k, eng in enumerate(ENGINES):
        cs = [c for r in summary[name][eng] for c in r["stop_latency"]]
        y0 = i + (k - 0.5) * 0.3
        ax2.scatter([c["approach_stopped_s"] * 1000 for c in cs], [y0] * len(cs), s=26, color=eng_col[eng],
                    edgecolor="white", lw=0.8, zorder=3, label=f"{eng}: stopped toward old goal" if i == 0 else None)
        ax2.scatter([c["abort_done_s"] * 1000 for c in cs], [y0] * len(cs), s=26, facecolor="white",
                    edgecolor=eng_col[eng], lw=1.2, zorder=3, label=f"{eng}: abort motion done" if i == 0 else None)
ax2.set_yticks(range(len(lat_names)))
ax2.set_yticklabels(lat_names)
ax2.invert_yaxis()
ax2.set_xlabel("time after the order change (ms)")
ax2.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, columnspacing=0.8)
ax2.set_xlim(left=0)
ax2.grid(axis="x", color=GRID, lw=0.6)
ax2.set_title("Stop latency (measured only)", loc="left", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(a.out, "summary.png"), dpi=130, bbox_inches="tight")
print(json.dumps({n: {e: [f"{r['passed']}/{r['checks']}" for r in summary[n][e]] for e in ENGINES}
                  for n in SCENARIOS}, indent=1))
print(f"wrote {a.out}")
