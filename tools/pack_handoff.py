#!/usr/bin/env python3
"""Pack a lab round: handoff index + notes + worktree outputs into one ZIP, verify it, archive the handoff dir.

  python tools/pack_handoff.py --date 261126 --include outputs/261126-right outputs/routes \
      outputs/261126-right.setup.json outputs/261126-cards.json --notes ../notes/261126_實機實驗紀錄.md

Layout (same as earlier rounds): the handoff directory's files at the ZIP root, worktree files under
`agentic-robotic-simulator-real-teleop/<path>`, and SHA256SUMS for every entry. PNG/MP4 are stored, the rest
deflated. After the CRC test and the required-item check the handoff directory moves to handoff_origin/.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]          # the real-teleop worktree
LAB = ROOT.parent                                   # AllenHZL/
STORED = {".png", ".jpg", ".jpeg", ".mp4", ".mkv", ".zip"}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def session_summary(path):
    state = json.loads((path / "session.json").read_text(encoding="utf-8"))
    attempts = state.get("attempts", [])
    count = lambda status: sum(a.get("status") == status for a in attempts)  # noqa: E731
    return {"session": str(path.relative_to(ROOT)), "attempts": len(attempts), "accepted": count("accepted"),
            "needs_review": count("needs_review"), "needs_replay": count("needs_replay"),
            "retry_or_interrupted": count("retry") + count("interrupted"),
            "replays": len(state.get("replays", [])), "dagger": len(state.get("dagger", [])),
            "first_replay_episode": state.get("first_replay_episode")}


def index(date, name, includes, notes, sessions):
    lines = [f"# {date}｜2080 Ti 實機採集交接", "",
             f"工作樹 `{ROOT.name}`，分支 `{git('branch', '--show-current')}`，commit `{git('rev-parse', 'HEAD')}`"
             + ("（有未提交變更）" if git("status", "--porcelain", "--untracked-files=no") else "") + "。",
             f"打包時間 {datetime.now().astimezone().isoformat(timespec='seconds')}。路徑相對於 ZIP 根目錄。", "",
             "## 採集進度", ""]
    for s in sessions:
        lines.append(f"- `{s['session']}`：有效示範 {s['accepted']}，待審 {s['needs_review']}，待重播 {s['needs_replay']}，"
                     f"重錄／中斷 {s['retry_or_interrupted']}，實機重播 {s['replays']}，DAgger {s['dagger']}；"
                     f"首筆重播 `{s['first_replay_episode']}`")
    if not sessions:
        lines.append("- 本包沒有採集 session。")
    lines += ["", "## 檔案", ""]
    lines += [f"- `{ROOT.name}/{p}`" for p in includes]
    lines += [f"- `notes/{Path(n).name}`" for n in notes]
    lines += ["", "## 實測結果（現場補）", "", "- 路線去程／回程時間：", "- 讀回 `control_timing` 實測頻率：",
              "- 到範圍邊界或暫停：", "", "## 退出確認（現場補）", "", "- 右 follower 六軸扭力讀回：", "",
              "## 待做", "", "- ", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True, help="YYMMDD, e.g. 261126")
    parser.add_argument("--name", default="2080Ti_實機採集")
    parser.add_argument("--include", nargs="*", default=[], help="worktree-relative files or directories")
    parser.add_argument("--notes", nargs="*", default=[], help="notes files to copy into the handoff")
    args = parser.parse_args(argv)

    folder = f"{args.date}_{args.name}"
    handoff = LAB / "handoff" / folder
    archive = LAB / "handoff_origin" / folder
    target = LAB / f"{folder}交接.zip"
    if archive.exists() or target.exists():
        parser.error(f"{archive} or {target} already exists")
    includes = []
    for item in args.include:
        path = (ROOT / item).resolve()
        if not path.exists() or ROOT not in path.parents:
            parser.error(f"{item}: not found inside the worktree")
        includes.append(path.relative_to(ROOT))
    sessions = [session_summary(ROOT / p) for p in includes if (ROOT / p / "session.json").is_file()]

    handoff.mkdir(parents=True, exist_ok=True)
    (handoff / "notes").mkdir(exist_ok=True)
    for note in args.notes:
        shutil.copy2(note, handoff / "notes" / Path(note).name)
    readme = handoff / "README.md"
    if not readme.exists():
        readme.write_text(index(args.date, folder, includes, args.notes, sessions), encoding="utf-8")
        print(f"wrote {readme}; fill the on-site sections, then run this command again with the same arguments")
        return 2

    entries = [(p, p.relative_to(handoff).as_posix()) for p in sorted(handoff.rglob("*")) if p.is_file()
               and p.name != "SHA256SUMS"]
    for rel in includes:
        source = ROOT / rel
        files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
        entries += [(p, f"{ROOT.name}/{p.relative_to(ROOT).as_posix()}") for p in files]
    sums = []
    tmp = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
        for path, arcname in entries:
            digest = hashlib.sha256()
            with path.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(block)
            sums.append(f"{digest.hexdigest()}  {arcname}")
            zf.write(path, arcname, zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED)
        zf.writestr("SHA256SUMS", "\n".join(sums) + "\n")
    (handoff / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    with zipfile.ZipFile(tmp) as zf:
        bad = zf.testzip()
        names = set(zf.namelist())
    required = {"README.md", "SHA256SUMS"} | {f"{ROOT.name}/{s['session']}/session.json" for s in sessions}
    if bad or not required <= names or len(names) != len(entries) + 1:
        tmp.unlink()
        raise SystemExit(f"ZIP check failed: bad={bad}, missing={sorted(required - names)}")
    tmp.rename(target)
    archive.parent.mkdir(exist_ok=True)
    shutil.move(str(handoff), str(archive))
    size = target.stat().st_size / 1e9
    print(f"{target} ({len(entries)} files, {size:.2f} GB) passed CRC and required-item checks; "
          f"handoff moved to {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
