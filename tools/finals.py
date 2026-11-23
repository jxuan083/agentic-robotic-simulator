#!/usr/bin/env python3
"""Finals whole-order scene in MuJoCo (docs/finals-shortage.md): one run, or the replayable acceptance batch.

  uv run tools/finals.py run --roles red,blue,yellow --seed 11 --answer substitute   # one run, answer injected
  uv run tools/finals.py run --roles red,blue,yellow --seed 11 --answer wait
  uv run tools/finals.py batch                         # 3 role sets x 2 seeds x {substitute, wait} = 12 runs
  uv run tools/finals.py batch --sets red,blue,yellow --seeds 5,6 --out outputs/finals/my_batch
  uv run tools/finals.py chat --roles red,blue,yellow --seed 11     # interactive: order and answer in the chat
  uv run tools/finals.py report outputs/finals/<run>/*.jsonl         # step timeline of runs

Roles are shortage A (x2), priority B (x1), substitute C (x1); the order is B x1 then A x2. Every run writes its
resolved scenario file next to its results, so `scripts/mujoco_ee_teleop.py --task <that file>` replays it exactly.
Runs open the MuJoCo window (acceptance is watched); --video records only that window.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from agentic_sim.config import load  # noqa: E402

BASE = "configs/tasks/finals_shortage.yaml"
SCENE = "configs/scenes/lab_20261120.yaml"     # lab scene: storage / placement trays and both sensor cameras
# Acceptance sets: three (A, B, C) role sets over eight palette colours, two preselected seeds each.
SETS = [("red", "blue", "yellow"), ("purple", "orange", "dark_green"), ("cyan", "yellow", "light_green")]
SEEDS = {SETS[0]: (11, 12), SETS[1]: (21, 22), SETS[2]: (31, 32)}
ANSWERS = ("substitute", "wait")


def scenario(roles, seed: int, answer: str | None, perception: str = "camera", order: str = "scenario",
             move_substitute: bool = False) -> dict:
    spec = load(BASE)
    if move_substitute:
        spec["finals"]["move_substitute"] = True
    spec["finals"]["perception"] = perception
    spec["finals"]["order"] = order
    a, b, c = roles
    spec["finals"]["roles"] = {"shortage": a, "priority": b, "substitute": c}
    spec["finals"]["layout"]["seed"] = int(seed)
    spec["finals"]["answer"] = answer
    spec["name"] = (f"finals_{a}_{b}_{c}_s{seed}_{answer or 'chat'}" + ("" if perception == "camera" else "_truth")
                    + ("_moved" if move_substitute else ""))
    return spec


def window_title() -> str:
    import re
    return re.search(r"title: *(.+)", open("configs/mujoco_so101.yaml").read()).group(1).strip()


def run_one(spec: dict, out: str, video: bool, extra: list) -> dict:
    import yaml
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, spec["name"])
    with open(base + ".yaml", "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)
    for ext in (".json", ".jsonl", ".csv", ".log", ".mp4"):
        if os.path.exists(base + ext):
            os.remove(base + ext)
    cmd = ["uv", "run", "--quiet", "--with", "opencv-python-headless>=4.9", "scripts/mujoco_ee_teleop.py", "--scene", SCENE, "--task", base + ".yaml", "--task_log", base + ".jsonl",
           "--log", base + ".csv", "--result", base + ".json", "--exit_after_script", "--max_time", "900", *extra]
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":1")
    env.setdefault("XAUTHORITY", f"/run/user/{os.getuid()}/gdm/Xauthority")
    t0 = time.monotonic()
    with open(base + ".log", "w") as log:
        app = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        rec = None
        if video:
            for _ in range(600):
                if "reset: session" in open(base + ".log").read():
                    break
                time.sleep(0.5)
            rec = subprocess.Popen(["tools/record_isaac_window.sh", "video", base + ".mp4", "900"],
                                   env={**env, "ISAAC_WINDOW_NAME": window_title()},
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rc = app.wait()
        if rec is not None:
            rec.wait()
    wall = time.monotonic() - t0
    if not os.path.exists(base + ".json"):
        return {"name": spec["name"], "ok": False, "error": f"no result (exit {rc}); see {base}.log", "wall_s": wall}
    r = json.load(open(base + ".json"))
    fin = r.get("finals", {})
    failed = [c["name"] for c in r["checks"] if not c["ok"]]
    return {"name": spec["name"], "roles": fin.get("roles"), "seed": spec["finals"]["layout"]["seed"],
            "answer": spec["finals"]["answer"], "ok": not failed and r["passed"] == r["checks_total"],
            "passed": r["passed"], "checks": r["checks_total"], "failed": failed, "t_sim": r["t_sim"],
            "wall_s": round(wall, 1), "layout": r.get("random_spawn"), "removed": fin.get("removed"),
            "restocked": fin.get("restocked"), "result": base + ".json"}


STEPS = (("verified", "交付核對"), ("standby_reached", "待命停穩"), ("scene:remove", "場景移除"),
         ("observation_unclear", "看不清"), ("question", "提問"), ("answer", "回答"), ("scene:restock", "場景補回"),
         ("task_done", "結單"))


def timeline(jsonl: str) -> list:
    """The steps of one run from its event log: delivery -> standby -> removal -> observation and question ->
    answer -> restock or order change -> final ledger."""
    import json as _json
    from agentic_sim.chat.zh import Words
    words = Words(load("configs/chat.yaml")["colors"])
    rows = []
    for line in open(jsonl):
        e = _json.loads(line)
        k = e["kind"] if e["kind"] != "scene" else f"scene:{e.get('action')}"
        if k == "observation" and e.get("reason") in ("standby", "restock_wait"):
            rows.append((e["t"], "觀測", f"{e['reason']}：{e.get('counts')}（{e.get('source')}，看清 {e.get('clear')}）"))
        elif k == "order":
            rows.append((e["t"], "訂單", f"第 {e['rev']} 版 {e['ledger']['target']}"))
        for key, name in STEPS:
            if k == key:
                detail = {"verified": lambda: f"{e['color']}（帳本 {e['ledger']['delivered']}）",
                          "standby_reached": lambda: f"停穩 {e.get('ok')}，夾爪張開 {e.get('jaw_open')}，qd {e.get('qd_max')}",
                          "scene:remove": lambda: f"{e.get('color')} #{e.get('block')} 於 {e.get('pose')}",
                          "question": lambda: words.question(e),
                          "answer": lambda: f"{e.get('choice')} {e.get('color') or ''}",
                          "scene:restock": lambda: f"{e.get('color')} #{e.get('block')} 至 {e.get('pose')}",
                          "task_done": lambda: f"帳本 {e['ledger']['delivered']}，米盤 {e.get('placement')}",
                          }.get(key, lambda: e.get("reason", ""))()
                rows.append((e["t"], name, detail))
    return rows


def summary_md(rows: list) -> str:
    lines = ["| 角色 (A, B, C) | seed | 回答 | 結果 | 檢查 | 模擬秒 |", "|---|---:|---|---|---:|---:|"]
    for r in rows:
        roles = r.get("roles") or {}
        lines.append(f"| {roles.get('shortage')}, {roles.get('priority')}, {roles.get('substitute')} | {r.get('seed')} "
                     f"| {r.get('answer')} | {'通過' if r['ok'] else '失敗：' + ', '.join(r.get('failed') or [r.get('error', '')])} "
                     f"| {r.get('passed')}/{r.get('checks')} | {r.get('t_sim')} |")
    for ans in ANSWERS:
        sub = [r for r in rows if r.get("answer") == ans]
        if sub:
            lines.append(f"\n{ans}: {sum(r['ok'] for r in sub)}/{len(sub)}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("run", help="one run")
    one.add_argument("--roles", default="red,blue,yellow", help="A,B,C: shortage, priority, substitute colours")
    one.add_argument("--seed", type=int, default=11)
    one.add_argument("--answer", choices=ANSWERS, required=True)
    one.add_argument("--out", default="outputs/finals/runs")
    one.add_argument("--move-substitute", action="store_true",
                     help="acceptance: once the question is up (before the answer) the scene moves C elsewhere")
    bat = sub.add_parser("batch", help="the acceptance batch (every set x seed x answer)")
    bat.add_argument("--sets", default=None, help="role sets 'a,b,c;a,b,c' (default: the three acceptance sets)")
    bat.add_argument("--seeds", default=None, help="seeds for every set, e.g. 5,6 (default: preselected per set)")
    bat.add_argument("--out", default=None)
    rep = sub.add_parser("report", help="step timeline (Markdown) of runs from their event logs (*.jsonl)")
    rep.add_argument("logs", nargs="+")
    chat = sub.add_parser("chat", help="interactive: MuJoCo server, model service and chat window (tools/lab.py)")
    chat.add_argument("--roles", default="red,blue,yellow")
    chat.add_argument("--seed", type=int, default=11)
    chat.add_argument("--out", default="outputs/finals/chat")
    chat.add_argument("--move-substitute", action="store_true", help="acceptance: move C after the question")
    chat.add_argument("--lab", nargs=argparse.REMAINDER, default=[], help="more tools/lab.py options (--model, --type)")
    for p in (one, bat):
        p.add_argument("--perception", choices=("camera", "sim_truth"), default="camera",
                       help="camera (default): choose and recheck with the cameras; sim_truth: task flow only")
        p.add_argument("--video", action="store_true", help="record the MuJoCo window only")
        p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="more scripts/mujoco_ee_teleop.py options")
    a = ap.parse_args()
    if a.cmd == "report":
        for path in a.logs:
            print(f"\n### {os.path.basename(path)}\n\n| 模擬秒 | 步驟 | 內容 |\n|---:|---|---|")
            for t, name, detail in timeline(path):
                print(f"| {t:.2f} | {name} | {detail} |")
        return 0
    if a.cmd == "chat":
        import yaml
        roles = tuple(a.roles.split(","))
        spec = scenario(roles, a.seed, None, "camera", order="chat", move_substitute=a.move_substitute)
        spec["name"] += time.strftime("_%Y%m%d_%H%M%S")
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, spec["name"] + ".yaml")
        with open(path, "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)
        zh = {k: v["zh"].rstrip("色") for k, v in load("configs/chat.yaml")["colors"].items()}
        r, b, c = (zh.get(x, x) for x in roles)
        print(f"[finals] 場景 {path}：粉盤 {r}×2、{b}、{c}。在聊天下單，例如「我要兩{r}一{b}，{b}先放」；"
              f"完成後事件紀錄與結果寫在場景檔旁邊。")
        cmd = ["uv", "run", "--quiet", "tools/lab.py", "--backend", "mujoco", "--task", path,
               "--chat-transcript", os.path.splitext(path)[0] + ".chat.jsonl", *a.lab]
        return subprocess.call(cmd)
    if a.cmd == "run":
        roles = tuple(a.roles.split(","))
        r = run_one(scenario(roles, a.seed, a.answer, a.perception, move_substitute=a.move_substitute), a.out, a.video,
                    a.extra)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0 if r["ok"] else 1
    sets = [tuple(s.split(",")) for s in a.sets.split(";")] if a.sets else SETS
    out = a.out or time.strftime("outputs/finals/batch_%Y%m%d_%H%M%S")
    rows = []
    for roles in sets:
        if len(roles) != 3 or len(set(roles)) != 3:
            ap.error(f"a role set needs three different colours: {roles}")
        seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else SEEDS.get(roles, (1, 2))
        for seed in seeds:
            for ans in ANSWERS:
                r = run_one(scenario(roles, seed, ans, a.perception), out, a.video, a.extra)
                rows.append(r)
                print(f"[finals] {r['name']}: {'ok' if r['ok'] else 'FAILED ' + str(r.get('failed') or r.get('error'))}",
                      flush=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(summary_md(rows))
    print(summary_md(rows))
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
