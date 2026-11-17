#!/usr/bin/env python3
"""Simulator menu: pick backend -> experiment -> mode -> options, then run it.

  uv run tools/sim.py                      interactive menu
  uv run tools/sim.py --list               list experiments (and server status)
  uv run tools/sim.py --backend isaac --experiment minimal_pick_place --mode teleop --focus
  uv run tools/sim.py --backend isaac --experiment minimal_pick_place --mode acceptance --scenario robustness --no-video
  add --dry-run to print the command instead of running it

Needs only PyYAML (plain `python3 tools/sim.py` also works where PyYAML is installed).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agentic_sim.experiments import BACKENDS, ExperimentError, list_experiments, load_experiment  # noqa: E402
from agentic_sim.launch import IMPLEMENTED_BACKENDS, MODES, LaunchError, RunOptions, build  # noqa: E402
from agentic_sim.server.client import probe  # noqa: E402

MODE_TEXT = {"teleop": "手動操作（keyboard／mouse）", "acceptance": "自動驗收（注入事件，視窗需保持焦點）"}


def server_text(backend: str) -> str:
    info = probe(backend)
    if info is None:
        return "伺服器未啟動"
    st = info.get("status") or {}
    return f"伺服器執行中（實驗：{st.get('experiment') or '無'}）"


def backend_text(b: str) -> str:
    if b not in IMPLEMENTED_BACKENDS:
        return f"{b:7s}  尚未實作"
    return f"{b:7s}  {server_text(b)}"


def ask(prompt: str, choices: list, labels: list | None = None, default: int | None = None) -> int:
    """Numbered choice; returns the index. 'q' quits."""
    labels = labels or [str(c) for c in choices]
    print(f"\n{prompt}")
    for i, lab in enumerate(labels, 1):
        print(f"  {i}. {lab}")
    hint = f" [預設 {default + 1}]" if default is not None else ""
    while True:
        s = input(f"選擇 1-{len(choices)}{hint}（q 離開）：").strip().lower()
        if s == "q":
            raise SystemExit(0)
        if not s and default is not None:
            return default
        if s.isdigit() and 1 <= int(s) <= len(choices):
            return int(s) - 1
        print("  無效的選擇")


def ask_yes(prompt: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        s = input(f"{prompt} [{d}]：").strip().lower()
        if not s:
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False


def ask_int(prompt: str, default: int) -> int:
    while True:
        s = input(f"{prompt} [預設 {default}]：").strip()
        if not s:
            return default
        try:
            return int(s)
        except ValueError:
            print("  請輸入整數")


def interactive() -> tuple:
    b = ask("選擇模擬器：", list(BACKENDS), [backend_text(b) for b in BACKENDS],
            default=0)
    backend = BACKENDS[b]
    if backend not in IMPLEMENTED_BACKENDS:
        print(f"{backend} backend 尚未實作。")
        raise SystemExit(1)
    exps = list_experiments(backend=backend)
    if not exps:
        print(f"沒有支援 {backend} 的實驗（configs/experiments/）。")
        raise SystemExit(1)
    e = ask("選擇實驗：", exps, [f"{x.name}：{x.description}" for x in exps], default=0)
    exp = exps[e]
    m = ask("選擇模式：", list(MODES), [MODE_TEXT[m] for m in MODES], default=0)
    mode = MODES[m]
    opts = RunOptions()
    scenario = None
    if mode == "acceptance":
        s = ask("選擇驗收情境：", list(exp.acceptance),
                [a.scenario + (f"（固定選項 {a.options}）" if a.options else "") for a in exp.acceptance], default=0)
        scenario = exp.acceptance[s].scenario
        opts.video = ask_yes("錄影（只錄應用程式視窗）？", True)
    else:
        opts.focus = ask_yes("專注模式（隱藏編輯器面板）？", True)
    if exp.color_dr and not (scenario and "color_dr" in exp.scenario(scenario).options):
        opts.color_dr = ask_yes("開啟顏色 DR？", False)
        if opts.color_dr:
            opts.color_dr_seed = ask_int("DR seed", 0)
    return backend, exp, mode, opts, scenario


def print_list():
    for b in BACKENDS:
        print(backend_text(b))
    print()
    for e in list_experiments():
        print(f"{e.name}  [{', '.join(e.backends)}]  {e.description}")
        if e.acceptance:
            print("    驗收：" + ", ".join(a.scenario for a in e.acceptance))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list experiments and server status")
    ap.add_argument("--backend", choices=BACKENDS)
    ap.add_argument("--experiment")
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--scenario", help="acceptance scenario (acceptance mode)")
    ap.add_argument("--focus", action="store_true")
    ap.add_argument("--color-dr", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="colour DR seed")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
    a = ap.parse_args(argv)

    try:
        if a.list:
            print_list()
            return 0
        if a.backend or a.experiment or a.mode:
            if not (a.backend and a.experiment and a.mode):
                ap.error("non-interactive use needs --backend, --experiment and --mode")
            backend, exp, mode, scenario = a.backend, load_experiment(a.experiment), a.mode, a.scenario
            opts = RunOptions(focus=a.focus, color_dr=a.color_dr, color_dr_seed=a.seed, video=not a.no_video)
        else:
            if not sys.stdin.isatty():
                ap.error("no TTY: use --backend/--experiment/--mode (or --list)")
            backend, exp, mode, opts, scenario = interactive()
        cmd = build(backend, exp, mode, opts, scenario)
    except (ExperimentError, LaunchError, KeyError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 2

    for n in cmd.notes:
        print(f"注意：{n}")
    print(f"\n$ {cmd.shell()}\n", flush=True)
    if a.dry_run:
        return 0
    env = dict(os.environ, **cmd.env)
    if not env.get("DISPLAY"):
        print("注意：沒有設定 DISPLAY，Isaac 視窗可能無法開啟（例如 export DISPLAY=:1）")
    try:
        return subprocess.call(cmd.argv, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
