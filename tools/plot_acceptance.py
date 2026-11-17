"""Evidence figures from acceptance runs (CSV / log written by scripts/isaac_ee_teleop.py).

    docker/isaac/run.sh /isaac-sim/python.sh tools/plot_acceptance.py \
        --runs outputs/acceptance --bench outputs/bench --out outputs/figures
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from agentic_sim.config import load, v  # noqa: E402

# validated categorical slots (dataviz reference palette, light mode), text inks, surface
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BAD = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "lines.linewidth": 2.0, "legend.frameon": False,
})


def read_csv(path):
    rows = list(csv.DictReader(open(path)))
    cols = {}
    for k in rows[0]:
        try:
            cols[k] = [float(r[k]) for r in rows]
        except ValueError:
            cols[k] = [r[k] for r in rows]
    return cols


def edges(t, flags):
    """[(t_start, t_end)] spans where flags is truthy."""
    spans, start = [], None
    for ti, f in zip(t, flags):
        if f and start is None:
            start = ti
        elif not f and start is not None:
            spans.append((start, ti))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


def gripper_events(d):
    ev, g = [], d["grip_cmd"]
    for i in range(1, len(g)):
        if g[i] < g[i - 1] - 1e-6 and (not ev or ev[-1][1] != "close"):
            ev.append((d["t_wall"][i], "close"))
        elif g[i] > g[i - 1] + 1e-6 and (not ev or ev[-1][1] != "open"):
            ev.append((d["t_wall"][i], "open"))
    return ev


def timeline(d, title, path):
    t = d["t_wall"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a.plot(t, [z * 100 for z in d["tz"]], color=S1, label="EE target z")
    a.plot(t, [z * 100 for z in d["ez"]], color=S2, label="EE actual z")
    a.plot(t, [z * 100 for z in d["bz"]], color=S3, label="block z")
    a.set_ylabel("height (cm)")
    a.set_title(title)
    a.legend(loc="center left", labelcolor=INK2)
    b.plot(t, d["grip_cmd"], color=S1, label="jaw command")
    b.plot(t, d["jaw"], color=S2, label="jaw measured")
    b.set_ylabel("jaw angle (rad)")
    b.set_xlabel("wall time (s)")
    b.legend(loc="center left", labelcolor=INK2)
    for ax in (a, b):
        for te, kind in gripper_events(d):
            ax.axvline(te, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
            if ax is a:
                ax.annotate(kind, (te, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -12),
                            textcoords="offset points", color=INK2, fontsize=9)
    # session change = reset
    for i in range(1, len(t)):
        if d["session"][i] != d["session"][i - 1]:
            for ax in (a, b):
                ax.axvline(t[i], color=INK2, linewidth=1)
            a.annotate("reset", (t[i], 0.5), xycoords=("data", "axes fraction"), xytext=(3, 0),
                       textcoords="offset points", color=INK2, fontsize=9)
    b.annotate("jaw stops at ~0.18 rad on the block\n(command is fully closed: physical contact)",
               xy=(0.6, 0.27), xycoords="axes fraction", color=INK2, fontsize=9, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def topview(runs, scene, path):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for name, color in (("storage_pink", "#f2a6bd"), ("placement_beige", "#e6d7b0")):
        tr = scene["trays"][name]
        L, W = v(tr["outer_size"])
        cx, cy = v(tr["center_xy"])
        # plot with +Y to the left (operator view from behind the robot): x-axis = -y
        ax.add_patch(Rectangle((-(cy + W / 2), cx - L / 2), W, L, facecolor=color, edgecolor=INK2, linewidth=1,
                               alpha=0.6, zorder=0))
        ax.annotate(name.replace("_", " "), (-(cy), cx + L / 2 + 0.01), ha="center", color=INK2, fontsize=9)
    for (label, d), color in zip(runs, (S1, S2)):
        ax.plot([-y for y in d["ey"]], d["ex"], color=color, label=f"EE path ({label})", linewidth=1.6)
    d = runs[0][1]
    ax.plot([-d["by"][0]], [d["bx"][0]], "s", color=S3, markersize=9, label="block start", zorder=5)
    i_end = next((i for i in range(len(d["session"])) if d["session"][i] != d["session"][0]), len(d["session"])) - 1
    ax.plot([-d["by"][i_end]], [d["bx"][i_end]], "s", color=S3, markersize=9, markerfacecolor=SURF,
            markeredgewidth=2, label="block placed", zorder=5)
    ax.plot([0], [0], "o", color=INK2, markersize=8)
    ax.annotate("robot base", (0, 0), xytext=(6, -12), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-0.38, 0.38)
    ax.set_ylim(-0.05, 0.47)
    ticks = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{-x:+.1f}" for x in ticks])
    ax.set_xlabel("y (m, +Y = left)")
    ax.set_ylabel("x (m, forward)")
    ax.set_title("Top view: EE paths, keyboard vs mouse pick and place")
    ax.legend(loc="lower left", labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def robustness(d, path):
    t = d["t_wall"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
    for s0, s1 in edges(t, [s in ("unreachable", "joint_limit", "ik_no_solution", "joint_jump") for s in d["status"]]):
        a.axvspan(s0, s1, color=BAD, alpha=0.12, linewidth=0)
    a.plot(t, [x * 100 for x in d["tx"]], color=S1, label="EE target x")
    a.plot(t, [x * 100 for x in d["ex"]], color=S2, label="EE actual x")
    a.set_ylabel("x (cm)")
    a.set_title("Robustness run: unreachable / focus loss / pause / reset")
    h, _ = a.get_legend_handles_labels()
    a.legend(h + [Patch(color=BAD, alpha=0.12, label="blocked: joint limit / no IK / jump (last valid target kept)")],
             [x.get_label() for x in h] + ["blocked: joint limit / no IK / jump (last valid target kept)"],
             loc="center right", labelcolor=INK2, fontsize=9)
    for s0, s1 in edges(t, [f == 0 for f in d["focused"]]):
        b.axvspan(s0, s1, color=S3, alpha=0.18, linewidth=0)
        b.annotate("focus lost", ((s0 + s1) / 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -12),
                   textcoords="offset points", ha="center", color=INK2, fontsize=9)
    for s0, s1 in edges(t, [p == 1 for p in d["paused"]]):
        b.axvspan(s0, s1, color=MUTED, alpha=0.2, linewidth=0)
        b.annotate("paused", ((s0 + s1) / 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -12),
                   textcoords="offset points", ha="center", color=INK2, fontsize=9)
    b.plot(t, [y * 100 for y in d["ty"]], color=S1, label="EE target y")
    held = [y * 100 if k else None for y, k in zip(d["ty"], d["keys"])]
    b.plot(t, held, color=S2, linewidth=4, alpha=0.5, label="a key is held")
    for i in range(1, len(t)):
        if d["session"][i] != d["session"][i - 1]:
            b.axvline(t[i], color=INK2, linewidth=1)
            b.annotate("reset (D still held)", (t[i], 0.35), xycoords=("data", "axes fraction"), xytext=(4, 0),
                       textcoords="offset points", color=INK2, fontsize=9)
    b.set_ylabel("y (cm)")
    b.set_xlabel("wall time (s)")
    b.legend(loc="lower left", ncol=2, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bench(bench_dir, path):
    order = [("gpu_vsync", "GPU PhysX\nvsync on"), ("cpu_vsync", "CPU PhysX\nvsync on"),
             ("gpu_novsync", "GPU PhysX\nvsync off"), ("cpu_novsync", "CPU PhysX\nvsync off\n(default)")]
    rows = []
    for key, label in order:
        logs = glob.glob(os.path.join(bench_dir, key, "*.log"))
        if not logs:
            continue
        parts = {"physics": [], "render": [], "other": [], "fps": []}
        passed = total = 0
        for lg in logs:
            txt = open(lg).read()
            for m in re.finditer(r"profile ms/frame: input\+ik=([\d.]+), physics=([\d.]+), render=([\d.]+), "
                                 r"ui\+log=([\d.]+)\s+fps=([\d.]+)", txt):
                ik, ph, rd, ui, fps = map(float, m.groups())
                parts["physics"].append(ph)
                parts["render"].append(rd)
                parts["other"].append(ik + ui)
                parts["fps"].append(fps)
            res = lg[:-4] + ".json"
            if os.path.exists(res):
                j = json.load(open(res))
                passed += sum(c["ok"] for c in j["checks"])
                total += len(j["checks"])
        mean = {k: sum(x) / len(x) for k, x in parts.items()}
        rows.append((label, mean, min(parts["physics"]), max(parts["physics"]), passed, total))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = list(range(len(rows)))[::-1]
    for yi, (label, m, lo, hi, p, n) in zip(y, rows):
        left = 0.0
        for k, color in (("physics", S1), ("render", S2), ("other", S3)):
            ax.barh(yi, m[k] - 0.4, left=left + 0.2, height=0.55, color=color, label=k if yi == y[0] else None)
            left += m[k]
        ax.annotate(f"{left:.0f} ms/frame · {m['fps']:.0f} fps · checks {p}/{n}", (left + 1, yi), va="center",
                    color=INK, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("mean time per frame (ms)  ·  physics = 4 substeps of 1/240 s")
    ax.set_xlim(0, max(sum(r[1][k] for k in ("physics", "render", "other")) for r in rows) * 1.55)
    ax.axvline(1000 / 60, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.annotate("60 fps budget", (1000 / 60, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -10),
                textcoords="offset points", color=INK2, fontsize=9)
    ax.set_title("Frame time by physics device and vsync (keyboard + mouse runs, same machine)")
    ax.legend(loc="lower right", ncol=3, labelcolor=INK2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rows


def real_focus(d, path):
    """A scripted run during which the OS focus moved to another window: input is dropped, nothing moves."""
    t = d["t_wall"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
    for ax in (a, b):
        for s0, s1 in edges(t, [f == 0 for f in d["focused"]]):
            ax.axvspan(s0, s1, color=S3, alpha=0.18, linewidth=0)
    a.plot(t, [z * 100 for z in d["tz"]], color=S1, label="EE target z")
    a.plot(t, [z * 100 for z in d["ez"]], color=S2, label="EE actual z")
    a.set_ylabel("height (cm)")
    a.set_title("Real focus loss (another desktop window took focus) during a scripted keyboard run")
    h, _ = a.get_legend_handles_labels()
    a.legend(h + [Patch(color=S3, alpha=0.18)], [x.get_label() for x in h] + ["Isaac window not focused"],
             loc="upper center", labelcolor=INK2, fontsize=9)
    b.plot(t, d["grip_cmd"], color=S1, label="jaw command")
    b.set_ylabel("jaw cmd (rad)")
    b.set_ylim(-0.25, 0.6)
    b.set_xlabel("wall time (s)")
    b.annotate("C (close) was held by the script while unfocused: dropped, jaw stays open",
               xy=(0.5, 0.45), xycoords="axes fraction", ha="center", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def viewport_drag(d, path):
    """Viewport handle drags: target path, blocked spans, and per-frame target step (no jumps)."""
    t = d["t_wall"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for ax in (a, b):
        for s0, s1 in edges(t, [x == 1 for x in d["handle_drag"]]):
            ax.axvspan(s0, s1, color=MUTED, alpha=0.12, linewidth=0)
        for s0, s1 in edges(t, [x in ("joint_limit", "ik_no_solution", "joint_jump") for x in d["status"]]):
            ax.axvspan(s0, s1, color=BAD, alpha=0.15, linewidth=0)
    a.plot(t, [x * 100 for x in d["tx"]], color=S1, label="target x")
    a.plot(t, [y * 100 for y in d["ty"]], color=S2, label="target y")
    a.plot(t, [z * 100 for z in d["tz"]], color=S3, label="target z")
    a.set_ylabel("target (cm)")
    a.set_title("Viewport handle: drags (grey), blocked at the reach boundary (red)")
    h, _ = a.get_legend_handles_labels()
    a.legend(h + [Patch(color=MUTED, alpha=0.12), Patch(color=BAD, alpha=0.15)],
             [x.get_label() for x in h] + ["handle drag active", "blocked (reason in log)"],
             loc="upper right", labelcolor=INK2, fontsize=8, ncol=2)
    steps = [0.0] + [1000 * ((d["tx"][i] - d["tx"][i - 1]) ** 2 + (d["ty"][i] - d["ty"][i - 1]) ** 2
                             + (d["tz"][i] - d["tz"][i - 1]) ** 2) ** 0.5 for i in range(1, len(t))]
    b.plot(t, steps, color=S1, linewidth=1.4, label="target step per frame")
    b.axhline(15, color=BAD, linewidth=1, linestyle=(0, (3, 3)))
    b.annotate("jump threshold 15 mm", (t[0], 15), xytext=(3, 3), textcoords="offset points", color=INK2, fontsize=9)
    b.set_ylabel("step (mm)")
    b.set_xlabel("wall time (s)")
    b.set_ylim(0, 20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/acceptance")
    ap.add_argument("--bench", default="outputs/bench")
    ap.add_argument("--scene", default="configs/scenes/lab_current_minimal.yaml")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--focus_run", default="outputs/real_focus_loss/gpu_vsync_keyboard_pick_place.csv")
    ap.add_argument("--only", default=None, help="comma list: timelines,topview,robustness,bench,focus,viewport")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None

    def want(k):
        return only is None or k in only

    if want("timelines") or want("topview"):
        kb = read_csv(os.path.join(a.runs, "keyboard_pick_place.csv"))
        ms = read_csv(os.path.join(a.runs, "mouse_pick_place.csv"))
        if want("timelines"):
            timeline(kb, "Keyboard-only pick and place: heights and jaw", os.path.join(a.out, "keyboard_timeline.png"))
            timeline(ms, "Mouse-only pick and place: heights and jaw", os.path.join(a.out, "mouse_timeline.png"))
        if want("topview"):
            topview([("keyboard", kb), ("mouse", ms)], load(a.scene), os.path.join(a.out, "topview_paths.png"))
    if want("robustness"):
        robustness(read_csv(os.path.join(a.runs, "robustness.csv")), os.path.join(a.out, "robustness_timeline.png"))
    if want("viewport") and os.path.exists(os.path.join(a.runs, "viewport_drag.csv")):
        viewport_drag(read_csv(os.path.join(a.runs, "viewport_drag.csv")), os.path.join(a.out, "viewport_drag_timeline.png"))
    if want("bench") and os.path.isdir(a.bench):
        for label, m, lo, hi, p, n in bench(a.bench, os.path.join(a.out, "physics_device_benchmark.png")):
            print(f"{label.replace(chr(10), ' ')}: physics {m['physics']:.1f} ms (min {lo:.1f}, max {hi:.1f}), "
                  f"render {m['render']:.1f} ms, other {m['other']:.1f} ms, fps {m['fps']:.1f}, checks {p}/{n}")
    if want("focus") and os.path.exists(a.focus_run):
        real_focus(read_csv(a.focus_run), os.path.join(a.out, "real_focus_loss.png"))
    print("figures in", a.out)


if __name__ == "__main__":
    main()
