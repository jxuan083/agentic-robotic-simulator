"""Text report of a readback summary (tools/read_episode.py writes the summary to <episode>/readback/readback.json).

Pure Python, no NumPy: the launcher (tools/sim.py) and the readback CLI print the same text. The verdict separates
the three ways an episode can fail: incomplete data (missing frames / images, mismatched files or steps), the task
not completed (no success), and grasp moments that are missing or out of order."""

from __future__ import annotations

MOMENTS = (("close", "夾爪閉合"), ("lift", "積木離盤"), ("release", "目標盤放開"))
KINDS = (("missing_in_row", "控制列缺圖"), ("missing_in_frames_jsonl", "相機索引缺幀"), ("name_mismatch", "檔名不符"),
         ("step_mismatch", "步號不符"), ("missing_files", "缺影像檔"))
RESULT_TEXT = {"placed": "放進米色盤", "reset": "按 R 重來", "closed": "關閉視窗"}


def verdict(s: dict) -> list:
    """Failure categories of a summary ([] when the readback passed)."""
    out = []
    if not s["check"]["ok"]:
        out.append("資料不完整（缺幀或錯檔）")
    if not (s.get("result") or {}).get("success"):
        out.append("任務未完成")
    if not all((s.get("order") or {"all_three_found": False}).values()):
        out.append("抓放時刻缺少或順序不符")
    return out


def text(s: dict, path: str | None = None) -> str:
    """path: the episode as the caller named it (default: the absolute path in the summary)."""
    c = s["check"]
    res = s.get("result") or {}
    reason = res.get("reason")
    lines = [
        f"episode：{path or s['episode']}",
        f"操作者：{s.get('operator')}　場景：{s.get('scene')}　目標顏色：{s.get('target_color')}",
        f"任務結果：{'成功' if res.get('success') else '未完成'}（{RESULT_TEXT.get(reason, reason)}）",
        f"幀數：{s['frames']}（{s.get('record_hz')} Hz，{s.get('duration_s')} s）",
        f"完整性：{'通過' if c['ok'] else '失敗'}",
    ]
    if c.get("missing_fields"):
        lines.append(f"  缺欄位：{c['missing_fields']}")
    if c.get("frame_gaps") or c.get("step_gaps"):
        lines.append(f"  幀號不連續：{c.get('frame_gaps')}　步號間隔不符：{c.get('step_gaps')}")
    for cam, v in c["images"].items():
        counts = v.get("counts") or {k: len(v.get(k) or []) for k, _ in KINDS}
        lines.append(f"  {cam}：{v['listed']} 幀；" + "，".join(f"{label} {counts.get(k, 0)}" for k, label in KINDS))
    errors = c.get("errors") or []
    if errors:
        lines.append(f"  錯誤 {c.get('error_count', len(errors))} 筆，前 {min(10, len(errors))} 筆：")
        lines += [f"    {e['camera']} 第 {e['frame']} 幀：{e['reason']}" for e in errors[:10]]
    lines.append("三個時刻（幀／模擬步／t_sim）：")
    mo = s.get("moments") or {}
    for k, label in MOMENTS:
        m = mo.get(k)
        lines.append(f"  {label}（{k}）：" + (f"幀 {m['frame']}／步 {m['step']}／{m['t_sim']:.2f} s" if m else "找不到"))
    order = s.get("order") or {}
    bad = [k for k, ok in order.items() if not ok]
    lines.append("  先後與狀態檢查：" + ("通過" if order and not bad else "未通過 " + ", ".join(bad or ["all_three_found"])))
    v = verdict(s)
    lines.append("讀回結果：" + ("通過" if not v else "失敗：" + "；".join(v)))
    return "\n".join(lines)
