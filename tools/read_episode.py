#!/usr/bin/env python3
"""Read a recorded demonstration back from disk (issue #13): completeness, the three grasp moments, figures.

  uv run tools/read_episode.py outputs/demos/<episode>                       # report (stdout + readback.json)
  uv run --with matplotlib tools/read_episode.py outputs/demos/<episode> --figures

Report: frames, duration, missing fields / frames / images, and the moments "gripper closes", "block leaves the
storage tray", "gripper opens over the placement tray" with the command and state at each (order checked: close <
lift < release, the jaw lags its command, the block is still low at close and in the placement tray at release).
--figures writes <episode>/readback/moments.png (global and wrist image at -0.25 s, the moment, +0.25 s for each
moment) and timeline.png (jaw command vs measured, block height, EE height, with the moments marked).
Needs numpy only (matplotlib for --figures); the files are read with the same code as the tests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from agentic_sim.capture.episode import Episode, check, grasp_moments  # noqa: E402

MOMENTS = (("close", "夾爪閉合"), ("lift", "積木離盤"), ("release", "目標盤放開"))


def order_checks(ep: Episode, mo: dict) -> dict:
    out = {}
    f = {k: mo[k]["frame"] for k, _ in MOMENTS if k in mo}
    out["all_three_found"] = len(f) == 3
    if len(f) == 3:
        out["close_before_lift_before_release"] = f["close"] < f["lift"] < f["release"]
    if "close" in mo:
        c = mo["close"]
        out["block_on_tray_at_close"] = c["block_z"] < ep.rows[0]["blocks"][mo["target_block"]]["pos"][2] + 0.005
        out["jaw_settles_after_close_cmd"] = c.get("jaw_settled_frame") is not None and c["jaw_settled_frame"] > c["frame"]
    if "release" in mo:
        from agentic_sim.scene_layout import block_in_tray
        tray = ep.manifest["trays"]["placement_beige"]
        p = mo["release"]["block_pos"]
        out["block_over_placement_at_release"] = bool(block_in_tray([p[0], p[1], tray["floor_top"]], tray,
                                                                    ep.manifest["block_edge"]))
        end = ep.rows[-1]["blocks"][mo["target_block"]]["pos"]
        out["block_in_placement_at_end"] = bool(block_in_tray(end, tray, ep.manifest["block_edge"]))
    return out


def figures(ep: Episode, mo: dict, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from agentic_sim.image_io import read_png
    os.makedirs(out_dir, exist_ok=True)
    hz = ep.manifest["record_hz"]
    d = max(1, int(round(0.25 * hz)))
    cams = ep.cameras
    found = [(k, label) for k, label in MOMENTS if k in mo]
    fig, axes = plt.subplots(len(found) * len(cams), 3, figsize=(12, 2.9 * len(found) * len(cams)))
    axes = np.atleast_2d(axes)
    for i, (k, label) in enumerate(found):
        f0 = mo[k]["frame"]
        for j, cam in enumerate(cams):
            for c, f in enumerate((f0 - d, f0, f0 + d)):
                ax = axes[i * len(cams) + j, c]
                ax.axis("off")
                f = min(max(f, 0), len(ep.rows) - 1)
                p = ep.image_path(cam, f)
                if p and os.path.exists(p):
                    ax.imshow(read_png(p))
                r = ep.rows[f]
                ax.set_title(f"{k} ({'moment' if f == f0 else f'{(f - f0) / hz:+.2f} s'}) {cam}  frame {f}  "
                             f"t {r['t_sim']:.2f} s\njaw cmd {r['command']['gripper']:.3f} meas "
                             f"{r['state']['joint_pos'][5]:.3f} rad, block z "
                             f"{r['blocks'][mo['target_block']]['pos'][2] * 1000:.0f} mm", fontsize=7)
    fig.suptitle(f"{ep.manifest['episode_id']}: close = gripper closes, lift = block leaves the storage tray, "
                 "release = gripper opens over the placement tray", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "moments.png"), dpi=90)
    plt.close(fig)

    t = ep.array("t_sim")
    t = t - t[0]
    tb = mo["target_block"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, ep.array("command.gripper"), label="jaw command")
    ax[0].plot(t, [r["state"]["joint_pos"][5] for r in ep.rows], label="jaw measured")
    ax[0].set_ylabel("jaw (rad)")
    ax[1].plot(t, [r["blocks"][tb]["pos"][2] * 1000 for r in ep.rows], color="C3", label=f"{ep.manifest['target_color']} block z")
    ax[1].set_ylabel("block z (mm)")
    ax[2].plot(t, [r["state"]["ee_pos"][2] * 1000 for r in ep.rows], label="EE measured z")
    ax[2].plot(t, [r["command"]["ee_pos"][2] * 1000 for r in ep.rows], "--", label="EE command z")
    ax[2].set_ylabel("EE z (mm)")
    ax[2].set_xlabel("simulated time since the episode start (s)")
    for k, _ in found:
        for a in ax:
            a.axvline(t[mo[k]["frame"]], color="k", lw=0.8, ls=":")
        ax[0].text(t[mo[k]["frame"]], ax[0].get_ylim()[1], f" {k}", fontsize=8, va="top")
    for a in ax:
        a.legend(fontsize=7, loc="upper right")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "timeline.png"), dpi=90)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args(argv)
    ep = Episode(a.episode)
    m = ep.manifest
    rep = check(ep)
    mo = grasp_moments(ep)
    oc = order_checks(ep, mo)
    t = ep.array("t_sim")
    summary = {
        "episode": os.path.abspath(a.episode), "schema": m.get("schema"), "source": m.get("source"),
        "operator": m.get("operator"), "scene": m.get("scene"), "target_color": m.get("target_color"),
        "status": m.get("status"), "result": m.get("result"), "frames": len(ep.rows),
        "record_hz": m.get("record_hz"), "duration_s": round(float(t[-1] - t[0]), 3) if len(t) else 0.0,
        "check": rep, "moments": {k: {kk: mo[k][kk] for kk in ("frame", "step", "t_sim", "gripper_cmd", "jaw",
                                                               "block_z", "images")} | (
            {"jaw_settled_frame": mo[k].get("jaw_settled_frame")} if k == "close" else {})
            for k, _ in MOMENTS if k in mo},
        "order": oc,
    }
    summary["ok"] = bool(rep["ok"] and all(oc.values()) and (m.get("result") or {}).get("success"))
    out = os.path.join(a.episode, "readback")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "readback.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if a.figures:
        figures(ep, mo, out)
        print(f"figures: {out}/moments.png, {out}/timeline.png")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
