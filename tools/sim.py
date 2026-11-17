#!/usr/bin/env python3
"""Simulator menu: pick backend -> experiment -> mode -> options, then run it.

  uv run tools/sim.py                      interactive menu
  uv run tools/sim.py --list               list experiments and server status
  uv run tools/sim.py --server up|down|status          manage the persistent server (Isaac)
  uv run tools/sim.py --backend isaac --experiment minimal_pick_place --mode teleop --focus
  uv run tools/sim.py --backend isaac --experiment minimal_pick_place --mode acceptance --scenario robustness --no-video

When the backend's server is running, the request goes to it (no restart, scene already loaded); otherwise the
experiment runs once in its own process (`--direct` forces that). `--dry-run` prints the direct command only.
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
from agentic_sim.server import runner  # noqa: E402
from agentic_sim.server.client import RemoteError, SimClient, probe  # noqa: E402

MODE_TEXT = {"teleop": "手動操作（keyboard／mouse）", "acceptance": "自動驗收（注入事件，視窗需保持焦點）"}
SERVER_SCRIPTS = {"isaac": "docker/isaac/server.sh"}


def server_text(backend: str) -> str:
    info = probe(backend)
    if info is None:
        return "伺服器未啟動" if backend in SERVER_SCRIPTS else "（無常駐伺服器）"
    st = info.get("status") or {}
    warn = "，" + st["warning"] if st.get("warning") else ""
    return f"伺服器執行中（實驗：{st.get('experiment') or '無'}，記憶體 +{st.get('rss_growth_mb', '?')} MB{warn}）"


def backend_text(b: str) -> str:
    if b not in IMPLEMENTED_BACKENDS:
        return f"{b:7s}  尚未實作"
    return f"{b:7s}  {server_text(b)}"


def server_cmd(backend: str, action: str) -> int:
    script = SERVER_SCRIPTS.get(backend)
    if script is None:
        print(f"{backend} 沒有常駐伺服器", file=sys.stderr)
        return 2
    return subprocess.call([os.path.join(ROOT, script), action], cwd=ROOT)


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


def interactive(dry_run: bool) -> tuple:
    """Returns (backend, exp, mode, opts, scenario, use_server) or raises SystemExit after a server action."""
    b = ask("選擇模擬器：", list(BACKENDS), [backend_text(b) for b in BACKENDS], default=0)
    backend = BACKENDS[b]
    if backend not in IMPLEMENTED_BACKENDS:
        print(f"{backend} backend 尚未實作。")
        raise SystemExit(1)
    use_server = False
    if backend in SERVER_SCRIPTS and not dry_run:
        if probe(backend) is not None:
            k = ask("伺服器執行中：", ["use", "down"], ["使用伺服器", "停止伺服器（docker compose down）"], default=0)
            if k == 1:
                raise SystemExit(server_cmd(backend, "down"))
            use_server = True
        elif ask_yes("啟動常駐伺服器？（之後不必重新載入；選 n 則只執行這一次）", True):
            if server_cmd(backend, "up") != 0:
                raise SystemExit(1)
            use_server = True
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
    opts.focus = ask_yes("專注模式（隱藏編輯器面板）？", mode == "teleop")
    if exp.color_dr and not (scenario and "color_dr" in exp.scenario(scenario).options):
        opts.color_dr = ask_yes("開啟顏色 DR？（之後也可在模擬器面板切換）", False)
        if opts.color_dr:
            opts.color_dr_seed = ask_int("DR seed", 0)
    if mode == "teleop":  # acceptance scripts are written for one block
        while True:
            opts.blocks = ask_int("積木數量（1-8，之後也可在模擬器面板調整）", 1)
            if 1 <= opts.blocks <= 8:
                break
            print("請輸入 1 到 8")
    return backend, exp, mode, opts, scenario, use_server


def print_list():
    for b in BACKENDS:
        print(backend_text(b))
    print()
    for e in list_experiments():
        print(f"{e.name}  [{', '.join(e.backends)}]  {e.description}")
        if e.acceptance:
            print("    驗收：" + ", ".join(a.scenario for a in e.acceptance))


def run_on_server(backend, exp, mode, opts, scenario) -> int:
    with SimClient.for_backend(backend, timeout=60.0, name="sim.py") as c:
        if mode == "teleop":
            runner.teleop(c, exp, opts)
            return 0
        if not scenario:
            raise LaunchError(f"acceptance mode needs a scenario (available: {[a.scenario for a in exp.acceptance]})")
        job = runner.acceptance(c, exp, opts, scenario, recorder_factory=runner.WindowRecorder)
        ok = job.get("state") == "done" and job.get("passed") == job.get("checks") and not job.get("warning")
        return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list experiments and server status")
    ap.add_argument("--server", choices=["up", "down", "status"], help="manage the backend's persistent server")
    ap.add_argument("--backend", choices=BACKENDS)
    ap.add_argument("--experiment")
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--scenario", help="acceptance scenario (acceptance mode)")
    ap.add_argument("--focus", action="store_true")
    ap.add_argument("--color-dr", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="colour DR seed")
    ap.add_argument("--blocks", type=int, default=1, choices=range(1, 9), metavar="1-8", help="number of blocks")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--direct", action="store_true", help="run in its own process even if a server is running")
    ap.add_argument("--dry-run", action="store_true", help="print the direct command, do not run it")
    a = ap.parse_args(argv)

    try:
        if a.list:
            print_list()
            return 0
        if a.server:
            return server_cmd(a.backend or "isaac", a.server)
        if a.backend or a.experiment or a.mode:
            if not (a.backend and a.experiment and a.mode):
                ap.error("non-interactive use needs --backend, --experiment and --mode")
            backend, exp, mode, scenario = a.backend, load_experiment(a.experiment), a.mode, a.scenario
            opts = RunOptions(focus=a.focus, color_dr=a.color_dr, color_dr_seed=a.seed, blocks=a.blocks,
                              video=not a.no_video)
            use_server = not (a.direct or a.dry_run) and backend in SERVER_SCRIPTS and probe(backend) is not None
        else:
            if not sys.stdin.isatty():
                ap.error("no TTY: use --backend/--experiment/--mode (or --list)")
            backend, exp, mode, opts, scenario, use_server = interactive(a.dry_run)
        if use_server:
            if not exp.supports(backend):
                raise LaunchError(f"experiment {exp.name!r} does not support backend {backend!r}")
            print(f"使用執行中的 {backend} 伺服器")
            return run_on_server(backend, exp, mode, opts, scenario)
        cmd = build(backend, exp, mode, opts, scenario)
    except (ExperimentError, LaunchError, KeyError, RemoteError, ConnectionError, TimeoutError) as e:
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
