#!/usr/bin/env python3
"""The real SO-101 behind the simulators' task protocol: the chat page / CLI orders, the task executor decides, the
arm runs the routines that work on the hardware (agentic_sim.real_task).

  tools/lab_day.sh task-server          # lerobot env: follower + D455; this terminal is the operator console
  FAKE=1 tools/lab_day.sh task-server   # simulated table (no hardware): the whole chat flow at the desk
  tools/lab_day.sh chat-web             # model service + chat page (http://127.0.0.1:8765) on --backend real

Per block (S1, the program grasp of tools/finals_reach.py): D455 look at park -> fixed route to the start pose ->
grasp (hover, lower, close, wait for the gripper, lift; a miss opens and goes home, the executor retries once) ->
carry into the next free beige slot P1-P4 at 11 cm, open, rise -> start pose -> park -> D455 look verifies the
landing. The arm parks after every delivery, so the trays are always seen without the arm in the way and an operator
can take a block away there (the finals shortage step).

Operator console (this terminal): Esc pauses the arm where it is (the menu continues / reverses); the chat page's
pause button does the same and its resume continues. Ctrl-C at the prompt or the `shutdown` command parks and
releases torque (after the support is confirmed).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from agentic_sim.config import load  # noqa: E402
from agentic_sim.real_task.session import RealHandler, RealSession  # noqa: E402
from agentic_sim.server.core import SimServer  # noqa: E402

HZ = 30


class ServerTerminal:
    """The operator console plus the chat page's pause: key() reports a pause request as Esc once; a menu shown
    while paused from the page waits for resume and continues (g / c / t / r) without a key press."""

    CONTINUE = ("g", "c", "t", "r", "s")

    def __init__(self, term):
        self.term = term
        self.page_pause = threading.Event()
        self._esc_due = False

    def pause(self):
        self.page_pause.set()
        self._esc_due = True

    def resume(self):
        self.page_pause.clear()

    def key(self, wait=0.0):
        if self._esc_due:
            self._esc_due = False
            return "esc"
        return self.term.key(wait)

    def choose(self, prompt, options):
        if self.page_pause.is_set() or self._esc_due:
            self.say(f"{prompt}\n（聊天頁暫停中：按「繼續」後自動繼續；或在這裡選）")
            while self.page_pause.is_set():
                key = self.term.key(0.2)
                if key in options:
                    self.page_pause.clear()
                    return key
            pick = next((k for k in self.CONTINUE if k in options), None)
            if pick:
                self.say(f"→ {options[pick]}（聊天頁繼續）")
                return pick
        return self.term.choose(prompt, options)

    def __getattr__(self, name):
        return getattr(self.term, name)


class RealArm:
    """scan / pick / place / home on the follower (see agentic_sim.real_task.arm). Runs in the executor's worker
    thread; the driver is used by one routine at a time."""

    def __init__(self, args, follower, refs, hz, term):
        from auto_collect import Episodes, look  # noqa: F401  (look: D455 at park)
        from finals_reach import in_zone, safe_zone
        from real_route import check_calibration, run_dir
        from agentic_sim.real_kinematics import ArmModel, find_calibration
        from agentic_sim.real_place import Placer, beige_slots, camera_from_pose, make_corr
        from agentic_sim.real_route import Driver, Track, load_route

        self.args, self.term = args, term
        self.data, self.route = load_route(Path(args.route))
        check_calibration(self.data, refs, leader=False)
        self.park, self.channel = self.data["park"]["pose"], self.data["channel"]["command"]
        self.support = self.data["park_support"]
        self.quads = json.loads(Path(args.trays).read_text())["quads"]
        self.cam = camera_from_pose(args.pose)
        self.model = ArmModel.from_file(find_calibration(args.follower_id), ROOT)
        self.corr = make_corr(args.corr)
        self.zone = safe_zone(self.model, self.corr, args.follower_id, args.corr, ROOT / "outputs/d455")
        self.in_zone = in_zone
        self.slot_poses = beige_slots(self.cam, self.quads)          # [(xy m, tray yaw)] P1-P4
        self.slots = [tuple(float(v) for v in p) for p, _ in self.slot_poses]
        self.driver = Driver(follower, term, run_dir(Path(args.route), "task"), hz=hz, drive_leader=False)
        self.driver.gripper_speed = 150.0
        self.placer = Placer(self.driver, self.model, self.corr, speed=75.0)
        self.eps = Episodes(self.driver, self.placer, np.random.default_rng(0))
        self.Track = Track
        self.home_xyz = self.model.ee(self.park)
        self.pan_xy = tuple(self.model.pan_origin[:2])
        self.where = "park"
        self.driver.arm_at(self.park)

    # ---- moves ------------------------------------------------------------------------------------
    def _to_start(self):
        from agentic_sim.real_route import follow
        if self.where == "park":
            follow(self.driver, self.route, ("停放位", "起姿"))
            self.where = "start"

    def home(self, open_gripper=True, phase=None):
        from agentic_sim.real_grasp import lift_and_return
        from agentic_sim.real_place import OPEN_PCT
        from agentic_sim.real_route import follow
        if self.where == "park":
            return
        if open_gripper:
            self.placer.set_grip(OPEN_PCT, "張開")
        if self.where != "start":
            back = lift_and_return(self.driver.target, self.channel, self.args.follower_id)
            if back is not None:
                follow(self.driver, back, ("目前位置", "起姿"))
            else:
                from agentic_sim.real_route import retimed, pose
                follow(self.driver, retimed([dict(self.driver.target),
                                             pose(self.channel) | {"gripper": self.driver.target["gripper"]}]),
                       ("目前位置", "起姿"))
        self.driver.target = self.driver.target | {"gripper": self.channel["gripper"]}
        follow(self.driver, self.route.reverse(), ("起姿", "停放位"))
        self.where = "park"

    def scan(self):
        from auto_collect import snapshot
        from agentic_sim.real_d455 import detect, open_aligned
        for attempt in range(2):
            try:
                read, _, stop = open_aligned(self.args.serial)
                try:
                    blocks = snapshot(read, self.cam, self.quads)
                    bgr, depth = read()
                    info = detect(bgr, depth, self.cam, self.quads)[2]
                finally:
                    stop()
                break
            except RuntimeError:
                if attempt:
                    raise
                time.sleep(2.0)
        occluded = bool(info.get("occluded"))
        return blocks, not occluded, "托盤上方有手或手臂" if occluded else ""

    def pick(self, block, phase=None):
        from agentic_sim.real_place import HELD_PCT, OPEN_PCT
        phase and phase("start")
        self._to_start()
        phase and phase("grasp")
        held, err = self.eps.grasp(block)
        self.where = "lift"
        if err or held is None or held < HELD_PCT:
            phase and phase("retreat")
            self.placer.set_grip(OPEN_PCT, "張開")
            self.home(open_gripper=True)
            return False, {"reason": err or "missed", "gripper_pct": held}
        return True, {"gripper_pct": round(float(held), 1)}

    def place(self, xy, phase=None):
        from agentic_sim.real_place import CLOSED_PCT, OPEN_PCT, hold_while
        k = int(np.argmin([math.hypot(xy[0] - s[0], xy[1] - s[1]) for s in self.slots]))
        slot_xy, yaw = self.slot_poses[k]
        phase and phase("transport")
        current = dict(self.driver.target)
        plan = hold_while(self.driver, lambda: self.placer.plan_place(slot_xy, yaw, yaw, CLOSED_PCT, current))
        if isinstance(plan, str):
            self.term.choose(f"放置 P{k + 1} 沒有執行（{plan}）。扶住夾爪裡的方塊後按 o 張爪，方塊拿走", {"o": "張爪"})
            self.placer.set_grip(OPEN_PCT, "張開")
            self.home(open_gripper=True)
            return f"P{k + 1}: {plan}（方塊已由人取下）"
        self.placer.execute_place(plan, f"米盤 P{k + 1}")
        self.where = "slot"
        phase and phase("home")
        self.home(open_gripper=True)
        return None

    # ---- state ----------------------------------------------------------------------------------------
    def at_park(self) -> bool:
        return self.where == "park"

    def ee(self):
        p = self.model.ee(self.driver.target)
        return p - self.corr(p)

    def gripper_open(self) -> bool:
        return float(self.driver.target.get("gripper", 0.0)) >= 25.0

    def reachable(self, xy) -> bool:
        from agentic_sim.real_place import GRASP_M, HOVER_M, OPEN_PCT
        if not self.in_zone(self.zone, xy[0], xy[1]):
            return False
        c = np.array([xy[0], xy[1], 0.0295 / 2])
        return all(self.placer.pose_for(c + [0, 0, h], 0.0, self.channel, ph, OPEN_PCT) is not None
                   for h, ph in ((HOVER_M, "hover"), (GRASP_M, "grasp")))

    def pause(self):
        self.term.pause()

    def resume(self):
        self.term.resume()

    def close(self):
        from agentic_sim.real_route import park_and_release
        if self.where != "park":
            self.home(open_gripper=False)
        park_and_release(self.driver, self.Track([dict(self.driver.target), self.park], [0.0]), self.park,
                         self.support)


def serve(session, handler, port, log=print, term=None):
    server = SimServer(handler, port=port, backend="real", log=log)
    log(f"real task server on 127.0.0.1:{server.address[1]}  (chat: --backend real)")
    period = 1.0 / HZ
    told = False
    try:
        while not server.stop_requested:
            began = time.monotonic()
            server.poll(0.0)
            session.step()
            ex = session.task.ex if session.task is not None else None
            waiting = ex is not None and ex.waiting_operator and ex.skill is None
            if waiting and not told:
                log("待命：手臂在停放位。操作者處理完（例如取走缺料的方塊）後，在這裡按 c 繼續。")
                told = True
            if waiting and term is not None and term.key(0.0) == "c":
                ex.operator_continue()
                log("繼續：重新觀測粉盤。")
            told = told and waiting
            time.sleep(max(0.0, period - (time.monotonic() - began)))
    finally:
        server.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", default=None, help="simulated table: colours in the pink tray, e.g. blue,blue,red,yellow")
    ap.add_argument("--config", default=str(ROOT / "outputs/single.local.yaml"))
    ap.add_argument("--route", default=str(ROOT / "outputs/routes/right-park-channel-261125.json"))
    ap.add_argument("--trays", default=str(ROOT / "outputs/261125-trays/trays.json"))
    ap.add_argument("--pose", default=str(ROOT / "outputs/d455/pose_261125.json"))
    ap.add_argument("--corr", default="21,-1.3,20")
    ap.add_argument("--serial", default="038122250316")
    ap.add_argument("--follower-id", default="my_awesome_follower_arm")
    ap.add_argument("--standby-after", type=int, default=None,
                    help="finals: after this many deliveries wait at park until the operator presses c (e.g. 2)")
    ap.add_argument("--events", default=str(ROOT / f"outputs/task/real-{time.strftime('%y%m%d-%H%M%S')}.jsonl"))
    args = ap.parse_args(argv)
    port = int(load(ROOT / "configs/sim_server.yaml")["servers"]["real"]["port"])

    if args.fake:
        from agentic_sim.real_task.arm import FakeArm
        rng = np.random.default_rng(0)
        from agentic_sim.real_task.arm import FAKE_PLUSH_AT
        colors = [c for c in args.fake.split(",") if c != "plush"]
        xs = [(0.20 + 0.05 * (i % 3), 0.10 + 0.05 * (i // 3)) for i in range(len(colors))]
        items = [{"color": c, "x": x, "y": y, "yaw": float(rng.uniform(0, 1.5))} for c, (x, y) in zip(colors, xs)]
        if "plush" in args.fake.split(","):          # FAKE_TABLE=red,red,blue,yellow,plush: the toy on the table
            items.append({"color": "plush", "x": FAKE_PLUSH_AT[0], "y": FAKE_PLUSH_AT[1], "yaw": 0.4})
        arm = FakeArm(items, seconds=1.0)
        session = RealSession(arm, event_log=args.events, standby_after=args.standby_after)
        serve(session, RealHandler(session, fake=True), port)
        return 0

    from real_route import Terminal, real_devices
    with Terminal() as raw:
        term = ServerTerminal(raw)
        term.say("實機任務伺服器：聊天下單 → 執行器 → 手臂。開扭力前確認：follower 在停放位、扭力關；"
                 "粉盤方塊在橘色弧線內；米盤 P1-P4 清空；工作區淨空。Esc 原地暫停。")
        if term.choose("y 開始（開扭力）／n 離開", {"y": "開始", "n": "離開"}) != "y":
            return 0
        with real_devices(args.config, "right", leader=False) as (_, follower, refs, hz):
            arm = RealArm(args, follower, refs, hz, term)
            try:
                session = RealSession(arm, log=term.say, event_log=args.events, standby_after=args.standby_after)
                serve(session, RealHandler(session), port, log=term.say, term=term)
            finally:
                arm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
