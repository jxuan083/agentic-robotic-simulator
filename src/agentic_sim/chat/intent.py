"""Intent schema, prompt and the checks between the model and the task executor (pure Python).

The model returns one intent (JSON, constrained by INTENT schema):
  order_total  items = the complete new order                    "總共一紅兩藍"
  order_add    items = added to the current order                 "再加一藍"
  order_set    items = only these colours change, others stay     "藍色改成兩顆"
  order_pick   one block named by position (pick_ref)             "我要左上角那顆紅色"
  answer       reply to the open question (question_id, choice, substitute_color / xy)
  status       progress question: answered from the ledger only
  clarify      ambiguous: ask back (reply), nothing is sent
  reject       not a task request the system can do (reply)
  stop         emergency stop / halt: pointed to the stop controls (the chat never drives the arm)

plan(intent, state) checks it against the live task state (colours, counts, question id and options) and turns it into
one Action: send a task command, or say something. Nothing the model writes reaches the executor unchecked.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from agentic_sim.task.pick_ref import REF_ZH, REFS, resolve

INTENTS = ("order_total", "order_add", "order_set", "order_pick", "answer", "status", "clarify", "reject", "stop")
CHOICES = ("return", "place_at", "deliver", "substitute", "reduce", "wait")
NO_RETURN_CHOICES = ("hold", "removed")     # configs/chat.yaml no_return (the executor's task.yaml no_return)


PICK_RULE = """- order_pick：用位置指定某一顆積木（「我要左上角那顆紅色」「拿最左邊的藍色」「離手臂最近的那顆黃的」）。items 只放那個顏色、數量 1，pick_ref 填位置：
  top_left＝左上角，top_right＝右上角，bottom_left＝左下角，bottom_right＝右下角，leftmost＝最左邊，rightmost＝最右邊，farthest＝最遠／最上面，nearest＝最近／離手臂最近／最下面。
  位置說法對不上這些（例如「中間那顆」「第三顆」）就 clarify。其他 intent 的 pick_ref 是 null。
"""


NO_RETURN_RULE = """  hold＝停住等人取下／先拿著不要動，removed＝已經取下／拿走了。
"""


def schema(palette: list, max_count: int, pick: bool = False, no_return: bool = False) -> dict:
    """pick: also the order_pick intent and its pick_ref field (configs/chat.yaml pick_by_position). no_return: also
    the hold / removed answers. Both off, the schema is the one the utterance table was measured with."""
    s = _schema(palette, max_count)
    if no_return:
        s["properties"]["choice"]["anyOf"][0]["enum"] += list(NO_RETURN_CHOICES)
    if not pick:
        s["properties"]["intent"]["enum"].remove("order_pick")
        del s["properties"]["pick_ref"]
        s["required"].remove("pick_ref")
    return s


def _schema(palette: list, max_count: int) -> dict:
    nullable_color = {"anyOf": [{"type": "string", "enum": list(palette)}, {"type": "null"}]}
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(INTENTS)},
            "items": {"type": "array", "items": {
                "type": "object",
                "properties": {"color": {"type": "string", "enum": list(palette)},
                               # the model may write what was said (e.g. 10); plan() rejects counts above max_count
                               "count": {"type": "integer", "minimum": 0, "maximum": 99}},
                "required": ["color", "count"], "additionalProperties": False}},
            "question_id": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "choice": {"anyOf": [{"type": "string", "enum": list(CHOICES)}, {"type": "null"}]},
            "substitute_color": nullable_color,
            "xy": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                             {"type": "null"}]},
            "pick_ref": {"anyOf": [{"type": "string", "enum": list(REFS)}, {"type": "null"}]},
            "reply": {"type": "string"},
        },
        "required": ["intent", "items", "question_id", "choice", "substitute_color", "xy", "pick_ref", "reply"],
        "additionalProperties": False,
    }


def system_prompt(colors_cfg: dict, max_count: int, pick: bool = False, no_return: bool = False) -> str:
    """pick: add the order_pick rule (configs/chat.yaml pick_by_position); no_return: the hold / removed answers. Both
    off, the prompt is the measured one."""
    p = _system_prompt(colors_cfg, max_count)
    if not pick:
        p = p.replace(PICK_RULE, "")
    if not no_return:
        p = p.replace(NO_RETURN_RULE, "")
    return p


def _system_prompt(colors_cfg: dict, max_count: int) -> str:
    vocab = "\n".join(f"- {c}: {'、'.join(v['words'])}" for c, v in colors_cfg.items())
    return f"""你是積木分揀機械臂的任務助理。使用者用繁體中文下單、改單、回答問題或詢問進度。
你只負責判斷使用者的意圖，輸出一個 JSON 物件，不控制手臂，也不判斷是否完成（完成與否由系統核對）。

顏色只能用下列代號（左邊）；右邊是使用者可能的說法：
{vocab}
只說「綠」或「綠色」時無法分辨淺綠或深綠，要用 clarify 反問。

intent 規則：
- order_total：使用者給出完整的新需求（「總共…」「我要…」「只要…」「全部改成…」「幫我放…到放置盤」「拿…給我」）。items 列出每個顏色與數量；沒列到的顏色就是不要。把某顏色的積木放進放置盤就是下單，不是直接控制手臂。
- order_add：在原需求上追加（「再加…」「多…」「另外再…」）。items 是追加的數量。
- order_set：只改某些顏色的數量，其他顏色維持（「藍色改成兩顆」「紅的只要一顆」「不要黃色」=黃色 0）。items 只列被改的顏色。
{PICK_RULE}- 「改成兩藍」這類沒說清楚其他顏色是否保留的話：如果目前需求裡還有其他顏色，用 clarify 反問「原本的X色要保留嗎？」；如果目前沒有其他顏色或沒有需求，當作 order_total。
- 使用者回答你的反問時，結合前面的對話決定最後的 order_total／order_set。
- answer：回答系統目前的問題。question_id 填目前問題的 id，choice 從該問題的 options 選：
  return＝放回原位／放回去，place_at＝放到指定位置（要有座標 xy，單位公尺；沒有座標就 clarify），deliver＝照樣送出／還是放進去，
  substitute＝換成別的顏色（substitute_color 填顏色），reduce＝減少數量／不用補了，wait＝等補料。
{NO_RETURN_RULE}- status：詢問進度、數量、狀態（「現在做到哪」「還差幾顆」）。
- clarify：意思不清楚、顏色或數量不確定時反問，reply 寫一句繁體中文的問題。「幾顆」「一些」「多一點」沒有說數字，就是數量不確定。數量照使用者說的填，不要自己改小。
- reject：不是任務需求或系統做不到（例如直接控制手臂往哪裡移動、不在色票裡的顏色、超過 {max_count} 顆），reply 寫一句繁體中文的說明。
- stop：要求立刻停止、急停、暫停手臂。
數字「一／二／兩／三…」與阿拉伯數字相同。items 在 answer、status、clarify、reject、stop 時是空陣列。
items 依使用者要的放置先後排列；說了「藍色先放」「先放藍色」就把藍色排第一，其他照說的順序。
reply 在 order 與 answer 時可留空字串。只輸出 JSON。"""


def state_for_prompt(st: dict, words) -> str:
    """The part of the task state the model needs (ledger, holding, open question, colours in the scene)."""
    led = st.get("ledger") or {}
    q = st.get("question")
    d = {"目前需求": led.get("target", {}), "已核對交付": led.get("delivered", {}), "還要": led.get("need", {}),
         "手上": (st.get("holding") or {}).get("color"), "場景中的顏色": st.get("colors_in_scene", []),
         "目前問題": None if not q else {"id": q["id"], "kind": q["kind"], "options": q["options"],
                                         "說明": words.question(q)}}
    return json.dumps(d, ensure_ascii=False)


@dataclass
class Action:
    kind: str                          # send / say
    text: str = ""                     # what the chat shows (say) or confirms (send)
    cmd: str | None = None
    args: dict = field(default_factory=dict)
    intent: str = ""
    reply: str = ""                    # for say: ask (asks back, waits for more) / reject (cannot do it) / info
                                       # (status, stop, no open question) / error (could not understand the model)
    followup: str = ""                 # ask for a missing parameter of a chosen disposal: "xy" (place_at) / "color"


def _items(intent: dict) -> dict:
    out = {}
    for it in intent.get("items") or []:
        out[str(it["color"])] = int(it["count"])
    return out


COLOR_CHARS = "紅藍黃綠橘橙青紫"
NUM_CHARS = "一二兩三四五六七八九十0-9０-９"
VAGUE_COUNT = re.compile(r"幾[顆個]|一些|一點|多一點|若干|些許|幾")
WHOLE_ORDER = re.compile(r"總共|全部|只要|都要|其他|其餘|保留|不要|就好|一共|合計")
CHANGE_TO = re.compile(rf"(^|[^{COLOR_CHARS}色的])改成\s*[{NUM_CHARS}]")


def ambiguity(text: str, intent: dict, st: dict, words) -> str | None:
    """Program-side guard for the two order phrasings the model cannot be trusted with (it guessed in the utterance
    table): a vague count ("再來幾顆藍色") and "改成兩藍" while the order has other colours (keep them or not?).
    Returns the question to ask back, or None."""
    if intent.get("intent") not in ("order_total", "order_add", "order_set"):
        return None
    if VAGUE_COUNT.search(text) and not re.search(rf"[{NUM_CHARS}]\s*[顆個]", text.replace("幾", "")):
        return "要幾顆？請說數字，例如「再加兩顆藍色」。"
    target = (st.get("ledger") or {}).get("target") or {}
    # colours the user actually said (not the model's items: it may have filled the others in by guessing)
    named = {c for c, v in words.cfg.items() if any(w in text for w in v.get("words", []))}
    others = [c for c, n in target.items() if n > 0 and c not in named]
    if CHANGE_TO.search(text) and others and not WHOLE_ORDER.search(text):
        keep = "、".join(words.color(c) for c in others)
        return f"原本的{keep}要保留嗎？請說「保留{keep}」或「只要…」。"
    return None


FIRST = re.compile(r"先(?:放|送|拿|交)?\s*(?P<a>[^，。,\s]{1,4}?)(?:色|的)?(?=[，。,\s]|$)|(?P<b>[^，。,\s]{1,4}?)(?:色|的)?\s*先(?:放|送|拿|交)")


def first_color(text: str, words) -> str | None:
    """The colour the user wants placed first ("藍色先放", "先放藍色"), from the text itself."""
    for m in FIRST.finditer(text):
        said = (m.group("a") or m.group("b") or "").strip()
        for c, v in words.cfg.items():
            if said and any(said == w or said == w.rstrip("色的") for w in v.get("words", [])):
                return c
    return None


def plan(intent: dict, st: dict, words, max_count: int, text: str = "") -> Action:
    from agentic_sim.task.ledger import Ledger
    k = intent.get("intent")
    reply = (intent.get("reply") or "").strip()[:200]
    led = st.get("ledger") or {}
    target = dict(led.get("target") or {})
    scene = set(st.get("colors_in_scene") or [])
    q = st.get("question")
    ask = lambda t, i=k: Action("say", t, intent=i, reply="ask")          # noqa: E731
    rej = lambda t, i=k: Action("say", t, intent=i, reply="reject")       # noqa: E731
    info = lambda t, i=k: Action("say", t, intent=i, reply="info")        # noqa: E731
    if k not in INTENTS:
        return Action("say", "抱歉，我沒有理解，請換個說法。", intent=str(k), reply="error")
    if k in ("order_total", "order_add", "order_set"):
        guard = ambiguity(text, intent, st, words)
        if guard:
            return ask(guard, "clarify(guard)")
        items = _items(intent)
        if not items:
            return ask(reply or "要哪些顏色、各幾顆？")
        if any(n < 0 or n > max_count for n in items.values()):
            return rej(f"一個顏色最多 {max_count} 顆。")
        if k == "order_add":
            items = {c: n for c, n in items.items() if n > 0}
            if not items:
                return ask("要追加哪個顏色、幾顆？")
            mode, confirm = "add", f"好的，追加 {words.counts(items)}。"
        elif k == "order_set":   # only the named colours: sent as mode set, so an order arriving first is kept
            mode = "set"
        else:
            items = {c: n for c, n in items.items() if n > 0}
            mode = "total"
        new = Ledger.compose(target, items, mode)
        over = {c: n for c, n in new.items() if n > max_count}
        if over:
            c = next(iter(over))
            return rej(f"一個顏色最多 {max_count} 顆：{words.color(c)}目前要 {target.get(c, 0)} 顆，"
                       f"這樣會變成 {over[c]} 顆，需求沒有改。")
        if mode != "add" and not new:
            return ask("這樣需求就是空的了。要哪些顏色、各幾顆？")
        first = first_color(text, words) if mode == "total" else None
        if first in items:      # placement order follows the ledger's colour order: the named colour goes first
            items = {first: items[first], **{c: n for c, n in items.items() if c != first}}
            new = Ledger.compose(target, items, mode)
        if mode == "total":
            confirm = f"好的，需求改為 {words.counts(new)}" + (f"，{words.color(first)}先放。" if first in items else "。")
        elif mode == "set":   # only the change: the order it applies to may differ from what was read (reported later)
            parts = [f"{words.color(c)}改為 {n} 顆" if n else f"不要{words.color(c)}" for c, n in items.items()]
            confirm = f"好的，{'、'.join(parts)}，其他顏色不變。"
        missing = [c for c, n in items.items() if n > 0 and c not in scene]
        if missing:
            confirm += f"（場景中沒有{'、'.join(words.color(c) for c in missing)}積木，系統會回報缺料。）"
        return Action("send", confirm, "task_order", {"counts": items, "mode": mode}, k)
    if k == "order_pick":
        return plan_pick(intent, st, words, max_count, target, scene)
    if k == "answer":
        if not q:
            return info("目前沒有待回答的問題。")
        qid = intent.get("question_id")
        if qid is not None and int(qid) != int(q["id"]):
            return ask(f"問題 {qid} 已經過期，目前的問題是：{words.question(q)}")
        choice = intent.get("choice")
        if choice not in q["options"]:
            return ask("沒有這個選項。" + words.question(q))
        args = {"id": int(q["id"]), "choice": choice}
        if choice == "substitute":
            c = intent.get("substitute_color")
            if not c:
                return Action("say", "要換成哪個顏色？", intent=k, reply="ask", followup="color")
            if c not in scene:
                return rej(f"場景中沒有{words.color(c)}積木，請換別的顏色。")
            args["color"] = c
        if choice == "place_at":
            xy = intent.get("xy")
            if not xy or len(xy) != 2:
                return Action("say", "要放到哪裡？請給桌面座標（公尺），例如 0.30 0.05。", intent=k, reply="ask",
                              followup="xy")
            args["xy"] = [float(xy[0]), float(xy[1])]
        return Action("send", f"好的，{words.choice(q, choice, args.get('color'))}。", "task_answer", args, k)
    if k == "status":
        return info(words.status(st))
    if k == "stop":
        return info("聊天不負責停止手臂。要立刻停下請按模擬器面板的 Pause（或 P 鍵），"
                    "要結束任務請執行 tools/task.py stop。")
    if k == "clarify":
        return ask(reply or "可以再說清楚一點嗎？")
    return rej(reply or "這個要求我做不到。")


def plan_pick(intent: dict, st: dict, words, max_count: int, target: dict, scene: set) -> Action:
    """One block named by position. Answered early from the task's block list when the task exposes one (missing
    colour, out of reach, words that fit two blocks); the executor checks again when it picks (pick_rejected)."""
    from agentic_sim.task.ledger import Ledger
    k = "order_pick"
    items = {c: n for c, n in _items(intent).items() if n > 0}
    ref = intent.get("pick_ref")
    if len(items) != 1 or ref not in REFS:
        return Action("say", "要哪個顏色、哪個位置的那一顆？例如「左上角那顆紅色」。", intent=k, reply="ask")
    c = next(iter(items))
    name, where = words.color(c), REF_ZH[ref]
    if c not in scene:
        return Action("say", f"場景裡沒有{name}積木。", intent=k, reply="reject")
    if "storage_blocks" in st:
        blocks = [b for b in st["storage_blocks"] if b["color"] == c]
        if not blocks:
            return Action("say", f"備料盤裡已經沒有{name}積木了。", intent=k, reply="reject")
        best, tie = resolve(blocks, ref, st.get("pick_frame"), st.get("pan_xy") or (0.0, 0.0))
        if tie:
            return Action("say", f"有不只一顆{name}在{where}附近，說清楚一點好嗎？例如「最左邊那顆{name}」。",
                          intent=k, reply="ask")
        if not best["reachable"]:
            ok = sum(1 for b in blocks if b["reachable"])
            more = f"另外有 {ok} 顆{name}夾得到，要改夾其中一顆嗎？" if ok else f"其他{name}也都夾不到。"
            return Action("say", f"{where}那顆{name}在手臂範圍外，夾不到。{more}", intent=k,
                          reply="ask" if ok else "reject")
    new = Ledger.compose(target, {c: 1}, "add")
    if new.get(c, 0) > max_count:
        return Action("say", f"一個顏色最多 {max_count} 顆，{name}已經要 {target.get(c, 0)} 顆，需求沒有改。",
                      intent=k, reply="reject")
    return Action("send", f"好的，夾{where}那顆{name}。", "task_order",
                  {"counts": {c: 1}, "mode": "add", "pick": {"color": c, "ref": ref}}, k)
