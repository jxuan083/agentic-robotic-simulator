"""Desktop controls for the single-arm collection session."""

import json
import queue
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import webbrowser

from agentic_sim.real_episode import Episode
from agentic_sim.real_review import _nearest, native_depth, save_depth_comparison


PHASES = {"align": "正在對齊標準起姿", "recording": "採集中，可操作 leader",
          "replay": "follower 正在重播", "return": "任務結束，準備復位"}
COLOR = {"red": "紅", "blue": "藍", "yellow": "黃"}
TRAY = {"pink": "粉盤", "beige": "米盤"}


class WindowOperator:
    def __init__(self, events):
        self.events = events
        self.end = threading.Event()
        self.abort = threading.Event()
        self.manual = threading.Event()
        self.cancelled = False

    def status(self, state):
        self.events.put(("status", PHASES[state]))

    def halted(self):
        return self.abort.is_set()

    def finished(self, _frame):
        return self.end.is_set()

    def takeover(self, _frame):
        return self.manual.is_set()

    def tick(self, _leader, _frame):
        pass

    def ask(self, kind, text):
        response = queue.Queue(maxsize=1)
        self.events.put(("prompt", (kind, text, response)))
        while not self.halted():
            try:
                return response.get(timeout=.1)
            except queue.Empty:
                pass
        raise RuntimeError("operator stopped; check follower support and torque on site")

    def finish(self):
        outcome = self.ask("outcome", None)
        return outcome | {"cancelled": self.cancelled}

    def support(self, support):
        return self.ask("support", support)


def _thumbnail(parent, path):
    image = tk.PhotoImage(master=parent, file=str(path))
    factor = max(1, (image.width() + 319) // 320)
    return image.subsample(factor)


class Window:
    def __init__(self, session):
        self.session = session
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            raise RuntimeError("open the collection GUI from the machine's desktop terminal") from error
        self.root.title("SO-101 右臂採集")
        self.root.geometry("900x630")
        self.root.minsize(740, 560)
        self.root.option_add("*Font", "sans 11")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.events = queue.Queue()
        self.worker = None
        self.operator = None
        self.history = []
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        self.card_text = tk.StringVar()
        self.progress_text = tk.StringVar()
        self.status_text = tk.StringVar(value="就緒")
        ttk.Label(frame, textvariable=self.card_text, font="sans 15", wraplength=830).pack(anchor="w")
        ttk.Label(frame, textvariable=self.progress_text).pack(anchor="w", pady=(4, 10))
        self.actions = ttk.Frame(frame)
        self.actions.pack(fill="x")
        self.record_button = ttk.Button(self.actions, text="開始採集", command=self.record)
        self.record_button.pack(side="left")
        self.preflight_button = ttk.Button(self.actions, text="空桌動作驗證", command=self.preflight)
        self.preflight_button.pack(side="left", padx=5)
        self.review_button = ttk.Button(self.actions, text="讀回與標幀", command=self.review)
        self.review_button.pack(side="left")
        self.replay_button = ttk.Button(self.actions, text="實機重播", command=self.replay)
        self.replay_button.pack(side="left", padx=5)
        self.retry_button = ttk.Button(self.actions, text="保留原檔，重錄此卡", command=self.retry)
        self.retry_button.pack(side="left")
        if self.session.backend.source == "fake":
            self.dagger_button = ttk.Button(self.actions, text="Fake DAgger", command=self.dagger)
            self.dagger_button.pack(side="left", padx=5)
        self.history_box = tk.Listbox(frame, height=11, exportselection=False)
        self.history_box.pack(fill="both", expand=True, pady=12)
        self.history_box.bind("<<ListboxSelect>>", lambda _: self.buttons())
        self.controls = ttk.Frame(frame)
        self.controls.pack(fill="x")
        self.finish_button = ttk.Button(self.controls, text="結束任務", command=self.finish)
        self.finish_button.pack(side="left")
        self.cancel_button = ttk.Button(self.controls, text="取消並受控停放", command=lambda: self.finish(cancel=True))
        self.cancel_button.pack(side="left", padx=5)
        self.halt_button = ttk.Button(self.controls, text="立即停止送令", command=self.halt)
        self.halt_button.pack(side="left")
        self.recover_button = ttk.Button(self.controls, text="已到場確認支撐與扭力", command=self.recover)
        self.recover_button.pack(side="left", padx=5)
        ttk.Label(frame, textvariable=self.status_text, wraplength=820).pack(anchor="w", pady=12)
        self.refresh()
        self.root.after(50, self.poll)

    def refresh(self):
        count, counts = self.session.progress()
        card, pending = self.session.current()
        source = self.session.backend.source
        if card:
            x, y, yaw = card["pose"]
            kind = "小偏差" if card["kind"] == "small_offset" else "正常"
            region = "近" if card["region"] == "near" else "遠"
            cue = "　張爪停穩後，保留一次局部對準。" if card["kind"] == "small_offset" else ""
            self.card_text.set(f"{source}  {card['id']}  {COLOR[card['color']]}塊  "
                               f"{TRAY[card['source_tray']]} → {TRAY[card['destination']]}　{region}區／{kind}\n"
                               f"盤相對座標 x {x:g} mm，y {y:g} mm，yaw {yaw:g}°{cue}")
        else:
            self.card_text.set(f"{source}  批次已完成")
        quotas = "　".join(
            TRAY[tray] + " " + " ".join(f"{COLOR[color]} {counts[tray, color]}/20" for color in COLOR)
            for tray in TRAY
        )
        self.progress_text.set(f"有效示範 {count}/120　{quotas}")
        self.history_box.delete(0, "end")
        self.history = list(reversed(self.session.state["attempts"]))
        names = {"accepted": "合格", "needs_review": "待標幀／確認", "needs_replay": "首筆待重播",
                 "retry": "待重錄", "interrupted": "中斷", "running": "執行中"}
        for entry in self.history:
            runs = [r for r in self.session.state["replays"] if r["source_episode"] == entry["episode"]]
            latest = ("，通過" if runs[-1].get("passed") else "，未通過") if runs else ""
            self.history_box.insert("end", f"{entry['episode']}　{names[entry['status']]}　重播 {len(runs)} 次{latest}")
        if pending and pending in self.history:
            self.history_box.selection_set(self.history.index(pending))
        self.buttons()

    def selected(self):
        selection = self.history_box.curselection()
        return self.history[selection[0]] if selection else None

    def buttons(self):
        busy = self.worker is not None
        card, pending = self.session.current()
        recover = self.session.state["operator_recovery_required"]
        entry = self.selected()
        valid = entry and entry.get("readback_ok")
        pending_check = pending and pending["status"] in ("needs_review", "needs_replay")
        reviewable = valid and entry["status"] not in ("interrupted", "retry")
        for button, enabled in (
            (self.record_button, not busy and not recover and card and not pending_check),
            (self.preflight_button, not busy and not recover),
            (self.review_button, not busy and valid),
            (self.replay_button, not busy and not recover and reviewable),
            (self.retry_button, not busy and reviewable and entry is pending and pending_check),
            (self.finish_button, busy and self.operator is not None and not self.operator.end.is_set()),
            (self.cancel_button, busy and self.operator is not None and not self.operator.end.is_set()),
            (self.halt_button, busy and self.operator is not None),
            (self.recover_button, not busy and recover),
        ):
            button.configure(state="normal" if enabled else "disabled")
        if hasattr(self, "dagger_button"):
            self.dagger_button.configure(state="normal" if not busy and not recover else "disabled")

    def run(self, task, *, color="red", replay=False, dagger=False):
        if self.worker is not None:
            return
        fake = self.session.backend.source == "fake"
        if fake:
            from agentic_sim.real_collection_fake import FakeOperator

            operator = FakeOperator(color, replay=replay, dagger=dagger)
        else:
            operator = WindowOperator(self.events)
        self.operator = None if fake else operator

        def call():
            try:
                result = task(operator)
            except (Exception, KeyboardInterrupt) as error:
                self.events.put(("error", str(error)))
            else:
                self.events.put(("done", result))

        self.worker = threading.Thread(target=call, name="collection-motion")
        self.worker.start()
        self.status_text.set("開始檢查")
        self.buttons()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status_text.set(value)
                elif kind == "prompt":
                    self.prompt(*value)
                else:
                    self.worker = None
                    self.operator = None
                    if kind == "error":
                        self.status_text.set(value)
                        messagebox.showerror("需要處理", value, parent=self.root)
                    else:
                        self.status_text.set(f"已存檔　{value.get('status', '')}")
                    self.refresh()
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def prompt(self, kind, text, response):
        if kind == "support":
            self.status_text.set("已抵達停放位並停穩")
            response.put(messagebox.askyesno("確認停放支撐", f"請確認已接觸支撐且手臂可卸力。\n{text}",
                                             parent=self.root))
            return
        self.status_text.set("follower 保持當下姿態，等待確認")
        dialog = tk.Toplevel(self.root)
        dialog.title("查看抓放結果")
        dialog.transient(self.root)
        holder = ttk.Frame(dialog, padding=15)
        holder.pack()
        ttk.Label(holder, text="先確認盤面與退場路徑淨空、夾爪已空。").pack()
        placed = tk.BooleanVar(value=False)
        color = tk.StringVar(value="")
        ttk.Checkbutton(holder, text="方塊已放到指定目的盤", variable=placed).pack(anchor="w")
        for value, label in (*COLOR.items(), ("unknown", "抓空／不明")):
            ttk.Radiobutton(holder, text=label, variable=color, value=value).pack(anchor="w")

        def done(empty):
            response.put({"empty": empty, "color": color.get(), "placed": placed.get()})
            dialog.destroy()

        ttk.Button(holder, text="確認空爪、退場路徑淨空", command=lambda: done(True)).pack(fill="x")
        ttk.Button(holder, text="待現場處理，保持扭力", command=lambda: done(False)).pack(fill="x")
        dialog.protocol("WM_DELETE_WINDOW", lambda: done(False))
        dialog.grab_set()

    def _placement(self, card):
        if self.session.backend.source == "fake":
            return card["pose"]
        choice = messagebox.askyesnocancel("確認當前擺位", "已按任務卡擺好方塊、手已離開，工作區淨空？\n"
                                           "選擇「否」可登記實際盤座標。", parent=self.root)
        if choice is True:
            return card["pose"]
        if choice is False:
            raw = simpledialog.askstring("實際位置", "x, y, yaw（mm, mm, °）。留空將保留影格待確認。",
                                         parent=self.root)
            if raw is None:
                return False
            if messagebox.askyesno("開始採集", "確認工作區淨空，手已離開？", parent=self.root):
                return [float(value) for value in raw.split(",")] if raw.strip() else None
        return False

    def record(self):
        card, _ = self.session.current()
        try:
            placement = self._placement(card)
        except ValueError:
            messagebox.showerror("位置格式", "請以逗號分隔 x、y、yaw。", parent=self.root)
            return
        if placement is not False:
            self.run(lambda operator: self.session.record(operator, placement), color=card["color"])

    def preflight(self):
        if self.session.backend.source == "real" and not messagebox.askyesno(
            "空桌驗證", "確認右臂起始支撐、桌面淨空，已準備少量移動並測停止？", parent=self.root
        ):
            return
        self.run(self.session.preflight)

    def dagger(self):
        from agentic_sim.real_collection_fake import fake_policy

        self.run(lambda operator: self.session.dagger(operator, fake_policy), dagger=True)

    def finish(self, cancel=False):
        if self.operator:
            self.operator.cancelled = cancel
            self.operator.end.set()
            self.buttons()

    def halt(self):
        if self.operator:
            self.operator.abort.set()

    def recover(self):
        if messagebox.askyesno("現場已處理", "已確認 follower 在安全支撐位、扭力全關且通訊狀態可重試？",
                               parent=self.root):
            self.session.clear_recovery()
            self.refresh()

    def retry(self):
        self.session.retry(self.selected()["episode"])
        self.refresh()

    def replay(self):
        entry = self.selected()
        card = self.session.card(entry["card"])
        dialog = tk.Toplevel(self.root)
        dialog.title("實機重播前確認盤面")
        dialog.transient(self.root)
        path = self.session.root / entry["episode"] / entry["opening_global"]
        dialog.image = _thumbnail(dialog, path)
        ttk.Label(dialog, image=dialog.image).pack(padx=12, pady=12)
        ttk.Label(dialog, text="依開場圖還原方塊位置、擺角及其他物件，再確認手已離開。").pack(padx=12)

        def start():
            dialog.destroy()
            self.run(lambda op: self.session.replay(entry["episode"], op, layout_ready=True),
                     color=card["color"], replay=True)

        ttk.Button(dialog, text="盤面就緒，開始實機重播", command=start).pack(pady=12)
        dialog.grab_set()

    def review(self):
        entry = self.selected()
        path = self.session.root / entry["episode"]
        report = json.loads((path / "readback.json").read_text(encoding="utf-8"))
        episode = Episode(path)
        dialog = tk.Toplevel(self.root)
        dialog.title("關鍵幀讀回")
        dialog.transient(self.root)
        hints = report["candidates"]
        marks = {key: value["frame"] for key, value in hints.items()}
        if (path / "review.json").exists():
            marks.update({key: value["frame"] for key, value in
                          json.loads((path / "review.json").read_text(encoding="utf-8"))["frames"].items()})
        frame = tk.IntVar(value=marks.get("handoff", 0))
        text = tk.StringVar()
        strip = ttk.Frame(dialog)
        strip.pack()
        panels = [ttk.Label(strip), ttk.Label(strip)]
        for panel in panels:
            panel.pack(side="left", padx=4)

        def update(*_):
            index = frame.get()
            stamp = episode.rows[index]["timestamp_ns"]
            for camera, panel in zip(("global", "wrist"), panels):
                image = _nearest(episode.frames[camera], stamp)
                panel.image = _thumbnail(panel, path / camera / image["rgb"])
                panel.configure(image=panel.image)
            text.set(f"控制幀 {index}　{stamp} ns　已標 {marks}")

        slider = tk.Scale(dialog, from_=0, to=len(episode.rows) - 1, orient="horizontal", variable=frame,
                          command=update, length=650)
        slider.pack()
        ttk.Label(dialog, textvariable=text).pack()
        for reason in report["review_reasons"]:
            ttk.Label(dialog, text=reason).pack(anchor="w", padx=12)
        buttons = ttk.Frame(dialog)
        buttons.pack(pady=6)
        outcome = report["operator_result"]
        placed = tk.BooleanVar(value=outcome.get("placed") is True)
        color = tk.StringVar(value=outcome.get("color", ""))
        result_controls = ttk.Frame(dialog)
        result_controls.pack(pady=6)
        for code, label in (*COLOR.items(), ("unknown", "不明／抓空")):
            ttk.Radiobutton(result_controls, text=label, variable=color, value=code).pack(side="left")
        ttk.Checkbutton(result_controls, text="確認落入目的盤", variable=placed).pack(side="left")

        def mark(key):
            marks[key] = frame.get()
            update()

        loop_token = [0]

        def loop(center):
            loop_token[0] += 1
            token = loop_token[0]
            lo, hi = max(0, center - 3), min(len(episode.rows) - 1, center + 3)
            frame.set(lo)
            update()

            def step():
                if token == loop_token[0] and dialog.winfo_exists():
                    frame.set(lo if frame.get() >= hi else frame.get() + 1)
                    update()
                    dialog.after(160, step)

            dialog.after(160, step)

        for key, label in (("coarse", "粗定位"), ("handoff", "接手"), ("close", "閉爪"), ("lift", "抬穩")):
            ttk.Button(buttons, text=f"標 {label}", command=lambda key=key: mark(key)).pack(side="left")
        ttk.Button(buttons, text="循環前後幀", command=lambda: loop(frame.get())).pack(side="left")
        ttk.Button(buttons, text="停止循環", command=lambda: loop_token.__setitem__(0, loop_token[0] + 1)).pack(side="left")
        for index in hints.get("handoff", {}).get("alternatives", []):
            ttk.Button(dialog, text=f"接手候選 {index}", command=lambda index=index: loop(index)).pack(side="left")

        def save():
            try:
                self.session.approve(entry["episode"], marks, visual={"color": color.get(), "placed": placed.get()})
            except (ValueError, KeyError) as error:
                messagebox.showerror("還需確認", str(error), parent=dialog)
            else:
                dialog.destroy()
                self.refresh()

        def layout():
            raw = simpledialog.askstring("核對實際布局", "輸入 x, y, yaw（mm, mm, °）", parent=dialog)
            if raw:
                try:
                    self.session.verify_layout(entry["episode"], [float(value) for value in raw.split(",")])
                except ValueError as error:
                    messagebox.showerror("暫不能確認布局", str(error), parent=dialog)

        ttk.Button(dialog, text="儲存確認幀", command=save).pack(side="right")
        ttk.Button(dialog, text="修正實際布局", command=layout).pack(side="right")
        if report["depth_summary"]["frames"]:
            ttk.Button(dialog, text="深度取樣", command=lambda: self.depth(dialog, episode, frame.get())).pack(side="right")
        ttk.Button(dialog, text="開啟曲線與雙相機",
                   command=lambda: webbrowser.open((path / "replay.html").resolve().as_uri())).pack(side="right")
        update()

    def depth(self, parent, episode, index):
        import cv2
        import numpy as np

        try:
            raw, _, frame = native_depth(episode, index)
        except (OSError, ValueError) as error:
            messagebox.showerror("深度取樣", str(error), parent=parent)
            return
        valid = raw[raw > 0]
        if not valid.size:
            messagebox.showerror("深度取樣", "該幀沒有有效深度。", parent=parent)
            return
        lo, hi = np.percentile(valid, [3, 97])
        mapped = np.clip((raw.astype(float) - lo) * 255 / max(1, hi - lo), 0, 255).astype(np.uint8)
        preview = cv2.applyColorMap(mapped, cv2.COLORMAP_VIRIDIS)
        preview[raw == 0] = 0
        ok, encoded = cv2.imencode(".png", preview)
        if not ok:
            raise OSError("could not encode depth preview")
        dialog = tk.Toplevel(parent)
        dialog.title(f"原生深度幀 {frame['frame']}")
        dialog.image = tk.PhotoImage(master=dialog, data=encoded.tobytes(), format="png")
        image = ttk.Label(dialog, image=dialog.image, padding=0)
        image.pack()
        caption = ttk.Label(dialog, text="點選方塊頂面的內部像素。", padding=8)
        caption.pack()
        picks = []

        def pick(event):
            picks.append((event.x, event.y))
            if len(picks) == 1:
                caption.configure(text="點選可見盤底的內部像素。")
                return
            try:
                result = save_depth_comparison(episode, index, *picks)
            except (OSError, ValueError) as error:
                caption.configure(text=f"{error}。重新選擇方塊頂面。")
            else:
                distance = result["range_difference_m"] * 1000
                caption.configure(text=f"已儲存沿視線深度差 {distance:.1f} mm。可重新取樣。")
            picks.clear()

        image.bind("<Button-1>", pick)

    def close(self):
        if self.worker is not None:
            self.status_text.set("請先結束或停止目前動作並確認實機狀態。")
            return
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()
