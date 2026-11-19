"""Traditional Chinese wording: colour names, counts, questions and executor event reports (fixed templates)."""

from __future__ import annotations

NUM = {0: "零", 1: "一", 2: "兩", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
CHOICE_ZH = {"return": "放回原位", "place_at": "放到指定位置", "deliver": "照樣送出", "substitute": "換成其他顏色",
             "reduce": "減少數量", "wait": "等補料"}


class Words:
    def __init__(self, colors_cfg: dict):
        self.cfg = colors_cfg

    def color(self, c: str) -> str:
        return self.cfg.get(c, {}).get("zh", c)

    def counts(self, d: dict, empty: str = "無") -> str:
        """{'red': 1, 'blue': 2} -> '紅色 1、藍色 2'."""
        items = [(c, n) for c, n in (d or {}).items() if n]
        return "、".join(f"{self.color(c)} {n}" for c, n in items) if items else empty

    def choice(self, q: dict, choice: str, color: str | None = None) -> str:
        if choice == "substitute" and color:
            return f"換成{self.color(color)}"
        return CHOICE_ZH.get(choice, choice)

    # ---- questions -------------------------------------------------------------------------------
    def question(self, q: dict) -> str:
        k = q.get("qkind") or q.get("kind")     # an event carries the question kind as qkind
        if k == "held_item":
            if "deliver" in q.get("options", []):
                return (f"手上的{self.color(q['color'])}積木，新的需求用不到。要放回原位、放到其他位置，還是照樣送出？"
                        f"（問題 {q['id']}）")
            return f"放置盤沒有空位放手上的{self.color(q['color'])}積木。要放回原位，還是放到其他位置？（問題 {q['id']}）"
        if k == "shortage":
            return (f"已交付 {self.counts(q.get('delivered'))}，還缺 {self.counts(q.get('missing'))}，來源盤已經沒有了。"
                    f"來源盤剩下：{self.counts(q.get('available'))}。要換成其他顏色、減少數量，還是等補料？"
                    f"（問題 {q['id']}）")
        return f"{q.get('text', '')}（問題 {q.get('id')}）"

    # ---- executor events -> reports (None = not reported in the chat) ---------------------------------
    def event(self, e: dict) -> str | None:
        k = e["kind"]
        if k == "order":
            return f"需求已生效（第 {e['rev']} 版）：{self.counts(e['ledger']['target'])}。"
        if k == "question":
            return self.question(e)
        if k == "question_dropped":
            return f"問題 {e['id']} 已取消：新的需求已經用得到手上的積木，會直接送出。"
        if k == "over_delivery":
            if e.get("over", 0) > 0:
                return f"{self.color(e['color'])}多交了 {e['over']} 顆，會從放置盤取回來源盤。"
            return f"{self.color(e['color'])}比需求多 {e['delivered'] - e['target']} 顆（你同意送出的），不取回。"
        if k == "task_done":
            return (f"完成：放置盤有 {self.counts(e.get('placement'))}，帳本核對過的交付是 "
                    f"{self.counts(e['ledger']['delivered'])}。")
        if k == "order_rejected":
            return f"異常：需求沒有套用，帳本維持原樣（{e.get('reason', '')}）。"
        if k == "answer_rejected":
            return f"異常：回答沒有被接受（{e.get('reason', '')}）。"
        if k == "uncertain":
            return f"異常：{self.color(e['color'])}積木的結果無法確認（{e.get('reason', '')}）。"
        if k == "dropped":
            return "異常：搬運中積木掉落。"
        if k == "block_skipped":
            return f"異常：{self.color(e['color'])}積木夾取失敗 {e.get('attempts')} 次，先跳過。"
        if k == "skill_done":
            return f"異常：{e['skill']} 動作{'失敗' if e['status'] == 'failed' else '結果不確定'}（{e.get('failure_reason', '')}）。"
        return None

    def status(self, st: dict) -> str:
        """Progress from the ledger only (what the executor has verified)."""
        if not st.get("active"):
            return "目前沒有任務。可以直接說要什麼，例如「我要三顆紅色」。"
        led = st["ledger"]
        parts = [f"需求（第 {led['revision']} 版）：{self.counts(led['target'])}",
                 f"已核對交付：{self.counts(led['delivered'])}"]
        if led.get("need"):
            parts.append(f"還要：{self.counts(led['need'])}")
        if led.get("over"):
            parts.append(f"多交待取回：{self.counts(led['over'])}")
        h = st.get("holding")
        parts.append(f"手上：{self.color(h['color'])}" if h else "手上：沒有")
        state = {"running": "進行中", "waiting": "等待中", "done": "已完成", "idle": "待命"}.get(st.get("state"), "")
        text = "；".join(parts) + f"。狀態：{state}。"
        if st.get("question"):
            text += " 待回答：" + self.question(st["question"])
        return text
