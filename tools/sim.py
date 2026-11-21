#!/usr/bin/env python3
"""Simulator menu: simulate (backend -> experiment -> mode -> options), record a demonstration, or read one back.

  uv run tools/sim.py                      interactive menu
  uv run tools/sim.py --list               list experiments and server status
  uv run tools/sim.py --server up|down|status [--backend mujoco]   manage a persistent server (default Isaac)
  uv run tools/sim.py --backend isaac --experiment minimal_pick_place --mode teleop --focus
  uv run tools/sim.py --backend isaac --experiment minimal_pick_place --mode acceptance --scenario robustness --no-video
  uv run tools/sim.py --mode record [--target-color yellow --block-colors red,yellow,blue --record-hz 20
                      --record-dir outputs/demos --wrist-lens real]      record demonstrations (Isaac, lab_20261120)
  uv run tools/sim.py --mode readback --episode outputs/demos/<episode> [--figures]   read one back (offline)

When the backend's server is running, the request goes to it (no restart, scene already loaded); otherwise the
experiment runs once in its own process (`--direct` forces that). `--dry-run` prints the direct command only.
Needs only PyYAML (plain `python3 tools/sim.py` also works where PyYAML is installed); recording starts the Isaac
container, readback runs tools/read_episode.py with uv (NumPy there). Both print the command they run, for copying.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agentic_sim.experiments import BACKENDS, ExperimentError, list_experiments, load_experiment  # noqa: E402
from agentic_sim.launch import (IMPLEMENTED_BACKENDS, MODES, RECORD_EXPERIMENT, LaunchError, RecordOptions,  # noqa: E402
                                RunOptions, build, build_record, check_record, record_rate)
from agentic_sim.server import runner  # noqa: E402
from agentic_sim.server.client import RemoteError, SimClient, probe  # noqa: E402

ENTRY_MODES = ("record", "readback")       # demonstration capture (issue #13) and readback, besides MODES
LOG_DIR = "outputs/sim_entry"
MODE_TEXT = {"teleop": "手動操作（keyboard／mouse）", "acceptance": "自動驗收（注入事件，視窗需保持焦點）"}
SERVER_SCRIPTS = {"isaac": "docker/isaac/server.sh", "mujoco": "tools/mujoco_server.sh"}
SERVER_STOP_TEXT = {"isaac": "停止伺服器（docker compose down）", "mujoco": "停止伺服器"}


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
            k = ask("伺服器執行中：", ["use", "down"], ["使用伺服器", SERVER_STOP_TEXT[backend]], default=0)
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
        entries = exp.acceptance_for(backend)
        s = ask("選擇驗收情境：", list(entries),
                [a.scenario + (f"（固定選項 {a.options}）" if a.options else "") for a in entries], default=0)
        scenario = entries[s].scenario
        opts.video = ask_yes("錄影（只錄應用程式視窗）？", True)
    if backend == "isaac":  # MuJoCo's window has no editor panels
        opts.focus = ask_yes("專注模式（隱藏編輯器面板）？", mode == "teleop")
    if exp.color_dr and not (scenario and "color_dr" in exp.scenario(scenario, backend).options):
        opts.color_dr = ask_yes("開啟顏色 DR？（之後也可在模擬器面板切換）", False)
        if opts.color_dr:
            opts.color_dr_seed = ask_int("DR seed", 0)
    if mode == "teleop":  # acceptance scripts are written for one block
        while True:
            opts.blocks = ask_int("積木數量（1-8" + ("，之後也可在模擬器面板調整）" if backend == "isaac" else "）"), 1)
            if 1 <= opts.blocks <= 8:
                break
            print("請輸入 1 到 8")
    return backend, exp, mode, opts, scenario, use_server


class Session:
    """What this menu session recorded: the readback takes the episode the recorder reported, never a guess."""
    saved: list = []


def wrist_lens_default(exp) -> str:
    from agentic_sim.config import load
    wr = (load(exp.scene).get("sensor_cameras") or {}).get("wrist") or {}
    return str(wr.get("lens_mode") or "real") if wr.get("lens") else "pinhole"


def record_setup_text(exp, rec: RecordOptions) -> str:
    lens = rec.wrist_lens or f"場景預設（{wrist_lens_default(exp)}）"
    try:
        stride, actual = record_rate(rec.record_hz)
        rate = (f"{actual:g} Hz（每 {stride} 個控制步）" if abs(actual - rec.record_hz) < 1e-9 else
                f"要求 {rec.record_hz:g} Hz，實際 {actual:g} Hz（每 {stride} 個控制步，控制頻率 60 Hz）")
    except (ValueError, ZeroDivisionError, OverflowError):
        rate = f"{rec.record_hz:g} Hz（無效）"
    p = rec.perception
    perc = (f"開（{p['perception_camera']}，最多 {p['perception_hz']:g} Hz" +
            (f"，JSON {p['perception_json']}" if p.get("perception_json") else "") + "；操作畫面約慢一半）"
            if p else "關")
    return "\n".join([
        "錄示範的設定：",
        f"  模擬器：Isaac　實驗：{exp.name}　操作：keyboard／mouse（Isaac 視窗）",
        f"  積木配置（依槽位）：{','.join(rec.block_colors)}",
        f"  目標顏色：{rec.target_color}",
        f"  錄製：{rate}，global 與腕部 RGB，腕部鏡頭 {lens}",
        f"  即時感知：{perc}",
        f"  輸出：{rec.out_dir}/<日期_時間>_<顏色>/",
        "  成功（積木放進米色盤、夾爪打開、停 1 秒）時自動存檔；按 R 開始下一筆；關閉 Isaac 視窗結束。"])


def record_flow(exp, rec: RecordOptions, dry_run: bool, interactive_ask: bool) -> int:
    """Show the setup and the capture command, run it, report what the recorder saved."""
    while True:
        print("\n" + record_setup_text(exp, rec))
        if interactive_ask:
            s = input(f"目標顏色（Enter = {rec.target_color}；輸入 layout 修改積木配置）：").strip()
            if s == "layout":
                cols = input(f"積木顏色，依槽位以逗號分隔 [目前 {','.join(rec.block_colors)}]：").strip()
                if cols:
                    rec.block_colors = [c.strip() for c in cols.split(",") if c.strip()]
                continue
            if s:
                rec.target_color = s
        problems = check_record(exp, rec)
        if not problems:
            break
        for msg in problems:
            print(f"  需要修正：{msg}")
        if not interactive_ask:
            return 2
    from agentic_sim.demo_entry import new_run_id
    if interactive_ask and not rec.perception and ask_yes("同時開啟即時感知？（錄鍵鼠示範通常不需要）", False):
        from agentic_sim.perception_cli import DEFAULTS
        rec.perception = {k: v for k, v in DEFAULTS.items()} | {"live_perception": True}
    rec.run_id = rec.run_id or new_run_id()
    cmd = build_record("isaac", exp, rec)
    print("\n採集命令（可複製到另一個終端單獨執行，需要 DISPLAY）：")
    print(f"$ {cmd.shell()}\n", flush=True)
    if dry_run:
        return 0
    if interactive_ask and not ask_yes("開始錄製？", True):
        return 0
    if probe("isaac") is not None:
        print("注意：Isaac 常駐伺服器也在執行；錄製會另開一個 Isaac，兩個同時跑會很慢")
    env = dict(os.environ, **cmd.env)
    if not env.get("DISPLAY"):
        print("注意：沒有設定 DISPLAY，Isaac 視窗可能無法開啟（例如 export DISPLAY=:1）")
    from agentic_sim.demo_entry import run_recording
    from agentic_sim.launch import record_dir
    res = run_recording(cmd, ROOT, record_dir(rec.out_dir), rec.target_color, env, LOG_DIR, rec.run_id)
    Session.saved += [ev for ev in res["saved"] if ev.get("frames")]
    print("\n這次存檔的 episode：" if res["saved"] else "\n這次沒有存檔任何 episode。")
    for ev in res["saved"]:
        print(f"  {'成功' if ev.get('success') else '未完成'}（{ev.get('reason')}）{ev.get('frames')} 幀 "
              f"{ev.get('duration_s')} s：{ev.get('episode')}")
    return 0 if res["exit_code"] == 0 or res["interrupted"] else 1


def pick_episode(out_dir: str) -> str | None:
    """The episode to read back: one this session recorded (as the recorder reported it), else an existing one the
    operator names explicitly."""
    if Session.saved:
        good = [ev for ev in Session.saved if ev.get("success")]
        last = (good or Session.saved)[-1]
        if len(Session.saved) == 1:
            print(f"\n這次錄的上一筆：{last['episode']}")
            return last["episode"]
        i = ask("這次錄了好幾筆，選一筆：", Session.saved,
                [f"{'成功' if ev.get('success') else '未完成'} {ev.get('frames')} 幀：{ev['episode']}" for ev in Session.saved],
                default=Session.saved.index(last))
        return Session.saved[i]["episode"]
    from agentic_sim.demo_entry import recent_episodes
    print("\n這次還沒有錄製。可以讀取既有的 episode（不會當成剛錄的那一筆）：")
    eps = recent_episodes(ROOT, out_dir)
    labels = [f"既有：{e}" for e in eps] + ["輸入路徑"]
    i = ask("選擇 episode：", eps + [None], labels, default=None)
    if i < len(eps):
        return eps[i]
    path = input("episode 資料夾路徑：").strip()
    return path or None


def readback_flow(episode: str, figures: bool, dry_run: bool, interactive_ask: bool) -> int:
    from agentic_sim.demo_entry import readback_command
    if not os.path.exists(os.path.join(ROOT, episode, "manifest.json")) and not os.path.exists(
            os.path.join(episode, "manifest.json")):
        print(f"錯誤：{episode} 不是 episode 資料夾（沒有 manifest.json）", file=sys.stderr)
        return 2
    cmd = readback_command(episode)
    fig = readback_command(episode, figures=True)
    print("\n讀回命令（離線工具，不需要 Isaac 或模型服務）：")
    print(f"$ {cmd.shell()}")
    print(f"雙相機對照圖：\n$ {fig.shell()}\n", flush=True)
    if dry_run:
        return 0
    rc = subprocess.call((fig if figures else cmd).argv, cwd=ROOT)
    if interactive_ask and not figures and ask_yes("產生雙相機對照圖（moments.png、timeline.png）？", False):
        subprocess.call(fig.argv, cwd=ROOT)
    return rc


def menu(dry_run: bool) -> tuple | None:
    """Top menu. Returns the simulate choice (as interactive()) or None after a record / readback action."""
    while True:
        last = f"（這次錄的：{Session.saved[-1]['episode']}）" if Session.saved else "（這次還沒錄；可選既有 episode）"
        op = ask("選擇操作：", ["sim", "record", "readback"],
                 ["模擬：手動操作或自動驗收", f"錄示範：Isaac、{RECORD_EXPERIMENT}、keyboard／mouse",
                  "檢查上一筆示範：從磁碟讀回" + last], default=0)
        if op == 0:
            return interactive(dry_run)
        try:
            if op == 1:
                record_flow(load_experiment(RECORD_EXPERIMENT), RecordOptions(), dry_run, True)
            else:
                ep = pick_episode(RecordOptions().out_dir)
                if ep:
                    readback_flow(ep, False, dry_run, True)
        except (ExperimentError, LaunchError) as e:
            print(f"錯誤：{e}", file=sys.stderr)


def print_list():
    for b in BACKENDS:
        print(backend_text(b))
    print()
    for e in list_experiments():
        print(f"{e.name}  [{', '.join(e.backends)}]  {e.description}")
        for b in e.backends:
            print(f"    驗收（{b}）：" + ", ".join(a.scenario for a in e.acceptance_for(b)))


def run_on_server(backend, exp, mode, opts, scenario) -> int:
    with SimClient.for_backend(backend, timeout=60.0, name="sim.py") as c:
        if mode == "teleop":
            runner.teleop(c, exp, opts)
            return 0
        if not scenario:
            raise LaunchError(f"acceptance mode needs a scenario "
                              f"(available: {[a.scenario for a in exp.acceptance_for(backend)]})")
        title = runner.window_title(backend)
        job = runner.acceptance(c, exp, opts, scenario, backend=backend,
                                recorder_factory=lambda p: runner.WindowRecorder(p, window=title))
        ok = job.get("state") == "done" and job.get("passed") == job.get("checks") and not job.get("warning")
        return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from agentic_sim.perception_cli import add_arguments as add_perception_arguments
    add_perception_arguments(ap)
    ap.add_argument("--list", action="store_true", help="list experiments and server status")
    ap.add_argument("--server", choices=["up", "down", "status"], help="manage the backend's persistent server")
    ap.add_argument("--backend", choices=BACKENDS)
    ap.add_argument("--experiment")
    ap.add_argument("--mode", choices=MODES + ENTRY_MODES)
    ap.add_argument("--scenario", help="acceptance scenario (acceptance mode)")
    ap.add_argument("--focus", action="store_true")
    ap.add_argument("--color-dr", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="colour DR seed")
    ap.add_argument("--blocks", type=int, default=1, choices=range(1, 9), metavar="1-8", help="number of blocks")
    ap.add_argument("--block-colors", default=None, help="colour name per block, comma-separated (task scenes, e.g. "
                    "red,red,blue); sets the block count")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--target-color", default=None, help="record: colour of the block to pick (default yellow)")
    ap.add_argument("--record-hz", type=float, default=None, help="record: images and rows per second (default 20)")
    ap.add_argument("--record-dir", default=None, help="record: output root (default outputs/demos)")
    ap.add_argument("--wrist-lens", choices=["real", "pinhole"], default=None, help="record: wrist lens mode "
                    "(default: the scene's)")
    ap.add_argument("--episode", help="readback: episode directory")
    ap.add_argument("--figures", action="store_true", help="readback: also write the two-camera figures")
    ap.add_argument("--direct", action="store_true", help="run in its own process even if a server is running")
    ap.add_argument("--dry-run", action="store_true", help="print the direct command, do not run it")
    a = ap.parse_args(argv)

    try:
        if a.list:
            print_list()
            return 0
        if a.server:
            return server_cmd(a.backend or "isaac", a.server)
        if a.mode == "readback":
            if not a.episode:
                ap.error("--mode readback needs --episode")
            return readback_flow(a.episode, a.figures, a.dry_run, False)
        if a.mode == "record":
            if (a.backend or "isaac") != "isaac":
                ap.error("--mode record runs on Isaac only")
            d = RecordOptions()
            from agentic_sim.perception_cli import options as perception_options
            rec = RecordOptions(block_colors=a.block_colors.split(",") if a.block_colors else d.block_colors,
                                target_color=a.target_color or d.target_color,
                                record_hz=d.record_hz if a.record_hz is None else a.record_hz,   # 0 is refused, not defaulted
                                out_dir=a.record_dir or d.out_dir, wrist_lens=a.wrist_lens,
                                perception=perception_options(a))
            return record_flow(load_experiment(a.experiment or RECORD_EXPERIMENT), rec, a.dry_run, False)
        if a.backend or a.experiment or a.mode:
            if not (a.backend and a.experiment and a.mode):
                ap.error("non-interactive use needs --backend, --experiment and --mode")
            backend, exp, mode, scenario = a.backend, load_experiment(a.experiment), a.mode, a.scenario
            colors = a.block_colors.split(",") if a.block_colors else None
            opts = RunOptions(live_perception=a.live_perception, perception_camera=a.perception_camera,
                              perception_hz=a.perception_hz, perception_json=a.perception_json,
                              focus=a.focus and backend == "isaac", color_dr=a.color_dr, color_dr_seed=a.seed,
                              blocks=len(colors) if colors else a.blocks, block_colors=colors, video=not a.no_video)
            use_server = not (a.direct or a.dry_run) and backend in SERVER_SCRIPTS and probe(backend) is not None
        else:
            if not sys.stdin.isatty():
                ap.error("no TTY: use --backend/--experiment/--mode (or --list)")
            chosen = menu(a.dry_run)
            backend, exp, mode, opts, scenario, use_server = chosen
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
        print("注意：沒有設定 DISPLAY，模擬器視窗可能無法開啟（例如 export DISPLAY=:1）")
    try:
        return subprocess.call(cmd.argv, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
