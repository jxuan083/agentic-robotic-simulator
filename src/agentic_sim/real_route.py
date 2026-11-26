"""Park <-> channel route for one SO-101 follower: teach once with the leader, then replay.

Teaching: the follower starts torque-off on its park support. It then follows the leader
directly (absolute joint mapping, as in the original LeRobot teleop), rate limited and soft
clipped to the calibrated range. Hitting a range edge only clips the command; it never stops
the loop. The commands sent between park and the operator's "c" key become the route.

Replay: going out follows the recorded commands; coming back follows them in reverse. The
playback clock only advances while the follower keeps up, so a slow axis slows the route
instead of aborting it. A persistent lag (obstruction) or Esc/Ctrl-C holds the measured pose
with torque on and opens a menu: continue, go back the way it came, or emergency release.

Torque is released only near the park pose, after the operator confirms the support. The
only exception is the emergency release, which requires typing OFF.
"""

from datetime import datetime
import json
from pathlib import Path
from time import monotonic_ns, sleep

from agentic_sim.real_control import JOINTS, UNITS, pose
from agentic_sim.real_episode import write_json
from agentic_sim.real_motion import CARRY_SQUEEZE, OFF, ON


FORMAT = "so101-park-channel-route/1"
BODY = JOINTS[:-1]

TELEOP_SPEED = dict.fromkeys(BODY, 90.0) | {"gripper": 200.0}  # command slew, unit/s
ENGAGE_SPEED = 20.0          # slew right after torque-on, ramps to TELEOP_SPEED
ENGAGE_SECONDS = 1.5
ENGAGE_GAP = dict.fromkeys(BODY, 12.0) | {"gripper": 40.0}  # leader-follower gap to engage
PLAY_SPEED = 45.0            # body joint cap during playback, deg/s (11/26: 30 felt slow)
GRIPPER_PLAY_SPEED = 60.0    # percent/s
PUSH_LAG = 8.0               # between GATE_LAG and this, playback creeps on at PUSH_RATE (gravity sag) ...
PUSH_RATE = 0.3              # ... beyond it the clock stops and a lasting lag pauses as an obstruction
GATE_LAG = 4.0               # playback clock pauses while body lag exceeds this
STALL_LAG = 15.0             # teleop: lag treated as obstruction ...
STALL_SECONDS = 0.8          # ... when it persists this long (playback: gated time)
ENDPOINT = 3.0               # follower must be this near park/channel to start a route
RESEAT_MAX = 15.0            # torque-off follower this near park (deg) may be moved back to it under torque
RESEAT_SPEED = 10.0          # deg/s for that move
RELEASE_LAG = 4.0            # body distance from park allowed for a normal torque release
STILL = 0.3                  # stationary window spread
STILL_SECONDS = 0.3
LEADER_SPEED = dict.fromkeys(BODY, 20.0) | {"gripper": 40.0}  # driven leader slew, unit/s
LEADER_STALL_LAG = 8.0       # driven leader this far behind its goal ...
LEADER_STALL_S = 0.5         # ... for this long is blocked: stop it before it loads the bus
LEADER_ROUTE_NEAR = 25.0     # leader within this of the route's park pose follows the route out of the fold
LEADER_ARRIVED = dict.fromkeys(BODY, 3.0) | {"gripper": 8.0}  # driven leader counts as at the follower pose
LEADER_SETTLE_S = 3.0        # after the goal reaches the follower pose, wait this long for the leader to arrive
HANDOVER_COUNTDOWN_S = 3     # operator takes the leader; its torque releases when this runs out
HANDOVER_GUARD_S = 0.8       # follower holds after the leader release; leader sag in this window is ignored
PATH_ERROR = 0.35            # simplification tolerance, body joints deg
GRIPPER_PATH_ERROR = 2.0
MAX_IDLE_GAP_S = 0.05        # stationary stretches shrink to this in the recorded timing


class Paused(Exception):
    """Motion stopped; the follower holds its measured pose with torque on."""

    def __init__(self, reason, track=None, index=0):
        super().__init__(reason)
        self.track = track
        self.index = index


class Released(Exception):
    """Torque was released on operator request outside the normal park flow."""


def body_gap(a, b):
    return max(abs(a[j] - b[j]) for j in BODY)


def lerp(a, b, fraction):
    return {j: a[j] + (b[j] - a[j]) * fraction for j in JOINTS}


def calibrated_limits(follower):
    if hasattr(follower, "limits"):
        return {j: tuple(follower.limits[j]) for j in JOINTS}
    limits = {}
    for joint in BODY:
        cal = follower.bus.calibration[joint]
        half = (cal.range_max - cal.range_min) * 180 / follower.resolution
        limits[joint] = (-half, half)
    limits["gripper"] = (0.0, 100.0)
    return limits


def now_iso():
    return datetime.now().astimezone().isoformat()


class Track:
    """Piecewise-linear joint path; seconds[i] is the recorded time from points[i] to points[i+1]."""

    def __init__(self, points, seconds):
        if len(points) < 2 or len(seconds) != len(points) - 1:
            raise ValueError("a track needs at least two points and one duration per segment")
        self.points = [pose(p) for p in points]
        self.seconds = [float(s) for s in seconds]

    def reverse(self):
        return Track(self.points[::-1], self.seconds[::-1])

    def tail(self, index, current):
        """Continue toward the end from somewhere between points[index] and points[index + 1]."""
        return Track([current, *self.points[index + 1:]], [self.seconds[index], *self.seconds[index + 1:]])

    def head_back(self, index, current):
        """Go back toward points[0] from somewhere between points[index] and points[index + 1]."""
        return Track([current, *self.points[index::-1]], [self.seconds[index], *self.seconds[:index][::-1]])

    def to_json(self):
        return {"waypoints": self.points, "segment_seconds": self.seconds}


def compress(samples):
    """Turn (time_ns, command) samples into a simplified Track with moving-time durations."""
    if len(samples) < 2:
        raise ValueError("not enough samples to build a route")
    timeline = [(0.0, samples[0][1])]
    for (t0, a), (t1, b) in zip(samples, samples[1:]):
        dt = (t1 - t0) / 1e9
        moving = body_gap(a, b) > 0.02 or abs(a["gripper"] - b["gripper"]) > 0.2
        elapsed = timeline[-1][0] + (dt if moving else min(dt, MAX_IDLE_GAP_S))
        if moving:
            timeline.append((elapsed, b))
        else:
            timeline[-1] = (elapsed, timeline[-1][1])
    if timeline[-1][1] is not samples[-1][1]:
        timeline.append((timeline[-1][0] + 1e-3, samples[-1][1]))
    points = [p for _, p in timeline]
    keep = sorted(_simplify(points))
    kept = [timeline[i] for i in keep]
    return Track([p for _, p in kept], [max(1e-3, b[0] - a[0]) for a, b in zip(kept, kept[1:])])


def _simplify(points, body_error=PATH_ERROR):
    scale = dict.fromkeys(BODY, 1 / body_error) | {"gripper": 1 / GRIPPER_PATH_ERROR}
    keep = {0, len(points) - 1}
    segments = [(0, len(points) - 1)]
    while segments:
        left, right = segments.pop()
        start, end = points[left], points[right]
        delta = {j: (end[j] - start[j]) * scale[j] for j in JOINTS}
        norm = sum(d * d for d in delta.values())
        worst, split = 1.0, None
        for index in range(left + 1, right):
            row = {j: (points[index][j] - start[j]) * scale[j] for j in JOINTS}
            frac = max(0.0, min(1.0, sum(row[j] * delta[j] for j in JOINTS) / norm)) if norm else 0.0
            error = max(abs(row[j] - frac * delta[j]) for j in JOINTS)
            if error > worst:
                worst, split = error, index
        if split is not None:
            keep.add(split)
            segments.extend(((left, split), (split, right)))
    return keep


class Driver:
    """Owns one armed follower: logging, clipping, teleop, playback, settle and release."""

    def __init__(self, follower, term, run_dir, *, hz=60, rate=1.0, relative=False, drive_leader=True,
                 clock=monotonic_ns, wait=sleep):
        self.f = follower
        self.term = term
        self.hz = hz
        self.rate = rate
        self.play_speed, self.gripper_speed = PLAY_SPEED, GRIPPER_PLAY_SPEED   # caps for play(); reach raises them
        self.relative = relative  # fallback when leader/follower calibration zeros disagree
        self.drive_leader = drive_leader  # hand-over: drive the leader to the follower pose, then release it
        self.clutch = None        # leader pose at a driven hand-over: teleop maps displacement from here
        self.leader_route = None  # park -> start Track: a folded leader is driven along it, not straight
        self.leader_rest = None   # the leader's own folded pose, read before it is driven out (its way home ends here)
        self.on_send = None       # episode recorder hook: (command, readback) after each follower command
        self.clock = clock
        self.wait = wait
        self.limits = calibrated_limits(follower)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.control = (self.run_dir / "control.jsonl").open("x", buffering=1)
        self.events = (self.run_dir / "events.jsonl").open("x", buffering=1)
        self.origin = clock()  # monotonic: other processes (e.g. a D455 recorder) align their stamps with it
        self.phase = "idle"
        self.last = None
        self.target = None
        self.armed = False
        self.state = {"joint_units": UNITS, "stage": "opened", "started_at": now_iso(),
                      "clock_origin_monotonic_ns": self.origin}
        self.stage("opened")

    # -- bookkeeping -------------------------------------------------------
    def t(self):
        return self.clock() - self.origin

    def event(self, kind, **fields):
        self.events.write(json.dumps({"event": kind, "phase": self.phase, "t_ns": self.t(), **fields},
                                     ensure_ascii=False, allow_nan=False) + "\n")

    def stage(self, name, **fields):
        self.state.update(stage=name, updated_at=now_iso(), **fields)
        write_json(self.run_dir / "status.json", self.state)

    def close_logs(self):
        self.control.close()
        self.events.close()

    # -- device ------------------------------------------------------------
    def read(self):
        self.last = pose(self.f.read())
        return self.last

    def torque(self):
        return self.f.torque_states()

    def clip(self, target):
        clipped, edges = {}, []
        for joint in JOINTS:
            low, high = self.limits[joint]
            clipped[joint] = min(high, max(low, target[joint]))
            if joint in BODY and (target[joint] <= low + 0.5 or target[joint] >= high - 0.5):
                edges.append(joint)
        return clipped, edges

    def send(self, target, leader=None):
        clipped, edges = self.clip(pose(target))
        sent = pose(self.f.send(clipped))
        command_ns = self.t()
        actual = self.read()
        self.target = clipped
        row = {"phase": self.phase, "command_ns": command_ns, "readback_ns": self.t(),
               "command": sent, "readback": actual}
        if leader is not None:
            row["leader"] = leader
        if edges:
            row["range_edge"] = edges
        self.control.write(json.dumps(row, allow_nan=False) + "\n")
        if self.on_send is not None:
            self.on_send(sent, actual)
        return sent, actual, edges

    def hold(self, reason):
        actual = self.read()
        self.send(actual)
        self.event("hold", reason=reason, readback=actual)
        return actual

    def tick(self, began_ns):
        self.wait(max(0.0, began_ns / 1e9 + 1 / self.hz - self.clock() / 1e9))

    def stop_requested(self):
        return self.term.key() == "esc"

    def arm_at(self, park):
        actual = self.read()
        if self.torque() != OFF:
            raise RuntimeError("follower torque must be confirmed off before arming at park")
        gap = body_gap(actual, park)
        if gap > RESEAT_MAX:
            raise RuntimeError(f"follower is {gap:.1f} deg from the taught park pose; support and reposition it")
        reseat = gap > ENDPOINT
        if reseat:
            off = "、".join(f"{j} {actual[j] - park[j]:+.1f}°" for j in BODY if abs(actual[j] - park[j]) > ENDPOINT)
            if self.term.choose(f"follower 離停放位 {gap:.1f}°（{off}）。s 開扭力後以 {RESEAT_SPEED:.0f}°/s 移回停放位"
                                "／q 離開（人工擺回）", {"s": "自動復位", "q": "離開"}) != "s":
                raise RuntimeError(f"follower is {gap:.1f} deg from the taught park pose; reposition it by hand")
        self.event("start_measured", readback=actual, park=park)
        self.armed = True
        self.f.arm()
        if self.torque() != ON:
            raise RuntimeError("could not confirm torque on for all six axes")
        self.target = self.clip(self.read())[0]
        self.event("torque_on", readback=self.last)
        if reseat:
            home = park | {"gripper": self.target["gripper"]}
            follow(self, Track([dict(self.target), home], [gap / RESEAT_SPEED]), ("目前位置", "停放位"))
            self.event("reseated_at_park", gap_deg=gap, readback=self.last)
        self.stage("armed", pose=self.last)

    # -- motions -----------------------------------------------------------
    def teleop(self, leader, exit_keys, record=None):
        """Follower follows the leader until a key in exit_keys. Returns the key."""
        self.phase = "teleop"
        target = dict(self.target)
        offset = dict.fromkeys(JOINTS, 0.0)
        if self.relative or self.clutch is not None:
            # Driven hand-over: map displacement from the released leader pose, so gravity sag right after the
            # release never moves the follower (same clutch as the LeHome DAgger recorder).
            first = self.clutch if self.clutch is not None else pose(leader.read())
            self.clutch = None
            offset = {j: target[j] - first[j] for j in JOINTS}
            self.event("relative_offset", offset=offset)
        begin = self.clock()
        stall_since = None
        next_status = 0
        while True:
            began = self.clock()
            key = self.term.key()
            if key == "esc":
                self.hold("operator stop")
                raise Paused("操作者暫停")
            if key in exit_keys:
                return key
            lead = pose(leader.read())
            elapsed = (began - begin) / 1e9
            for joint in JOINTS:
                speed = TELEOP_SPEED[joint]
                if elapsed < ENGAGE_SECONDS:
                    speed = min(speed, ENGAGE_SPEED + (speed - ENGAGE_SPEED) * elapsed / ENGAGE_SECONDS)
                step = speed / self.hz
                target[joint] += max(-step, min(step, lead[joint] + offset[joint] - target[joint]))
            target = self.clip(target)[0]
            sent, actual, edges = self.send(target, leader=lead)
            if record is not None:
                record.append((self.clock(), dict(target)))
            lag = body_gap(sent, actual)
            if lag > STALL_LAG:
                stall_since = stall_since or began
                if began - stall_since > STALL_SECONDS * 1e9:
                    self.hold(f"stall lag {lag:.1f}")
                    raise Paused(f"follower 跟不上 leader（差 {lag:.0f}°），可能受阻")
            else:
                stall_since = None
            if began >= next_status:
                note = ("  到範圍邊界：" + "、".join(edges)) if edges else ""
                self.term.status(f"跟隨中  最大落後 {lag:4.1f}°{note}")
                next_status = began + 200_000_000
            self.tick(began)

    def play(self, track, label):
        """Follow track from the current command. Paused carries the segment index reached."""
        self.phase = label
        points = [dict(self.target), *track.points[1:]]
        work = Track(points, track.seconds)
        durations = []
        for i, recorded in enumerate(work.seconds):
            a, b = work.points[i], work.points[i + 1]
            durations.append(max(recorded / self.rate, body_gap(a, b) / self.play_speed,
                                 abs(a["gripper"] - b["gripper"]) / self.gripper_speed, 1 / self.hz))
        total = sum(durations)
        done = 0.0
        segment, tau = 0, 0.0
        stalled = 0.0
        sent = dict(self.target)
        next_status = 0
        while True:
            began = self.clock()
            if self.stop_requested():
                self.hold("operator stop")
                raise Paused("操作者暫停", work, segment)
            actual = self.last if self.last is not None else self.read()
            lag = body_gap(sent, actual)
            if lag <= GATE_LAG:
                tau += 1 / self.hz
                stalled = 0.0
            elif lag <= PUSH_LAG:
                # Heavy, stretched poses sag a steady few degrees under gravity: keep advancing slowly, the
                # growing command error adds servo torque. An obstacle drives the lag past PUSH_LAG instead.
                tau += PUSH_RATE / self.hz
                stalled = 0.0
            else:
                stalled += 1 / self.hz
                if stalled > STALL_SECONDS * 1.25:
                    self.hold(f"stall lag {lag:.1f}")
                    raise Paused(f"follower 停滯（差 {lag:.0f}°），可能受阻", work, segment)
            while segment < len(durations) and tau >= durations[segment]:
                tau -= durations[segment]
                done += durations[segment]
                segment += 1
            if segment >= len(durations):
                self.send(work.points[-1])
                return
            goal = lerp(work.points[segment], work.points[segment + 1], tau / durations[segment])
            sent, actual, _ = self.send(goal)
            if began >= next_status:
                self.term.status(f"{label}  {100 * (done + tau) / total:5.1f}%  剩 {total - done - tau:4.1f} s"
                                 f"  落後 {lag:4.1f}°")
                next_status = began + 200_000_000
            self.tick(began)

    def settle(self, seconds=3.0):
        """Hold the last command until the follower is stationary. Returns (readback, still)."""
        window = []
        deadline = self.clock() + seconds * 1e9
        while self.clock() < deadline:
            began = self.clock()
            if self.stop_requested():
                self.hold("operator stop")
                raise Paused("操作者暫停")
            _, actual, _ = self.send(self.target)
            window.append((began, actual))
            while len(window) > 1 and began - window[1][0] >= STILL_SECONDS * 1e9:
                window.pop(0)
            if began - window[0][0] >= STILL_SECONDS * 1e9 and all(
                    body_gap(window[0][1], a) < STILL for _, a in window):
                return actual, True
            self.tick(began)
        return self.last, False

    def release(self, park, support):
        """Normal torque release: only near park, stationary, after the support is confirmed."""
        self.phase = "park"
        try:
            actual, still = self.settle()
        except Paused as pause:
            self.term.say(f"已暫停並原地保持：{pause}")
            return False
        gap = body_gap(actual, park)
        if not still or gap > RELEASE_LAG:
            self.term.say(f"未在停放位停穩（距停放位 {gap:.1f}°，停穩={still}），不解扭力。")
            return False
        self.stage("awaiting_support", pose=actual, park_gap_deg=gap)
        if self.term.choose(f"距停放位 {gap:.1f}°。目視確認前臂已接觸停放支撐（{support}）、手與相機淨空，"
                            "y 解扭力／n 保持", {"y": "解扭力", "n": "保持"}) != "y":
            return False
        self._torque_off("park", readback=actual, park_gap_deg=gap)
        return True

    def menu(self, prompt, options):
        """Ask until a non-emergency choice; 'x' releases in place only after typing OFF."""
        while True:
            choice = self.term.choose(prompt, options)
            if choice != "x":
                return choice
            self.emergency_release()

    def emergency_release(self):
        if not self.term.word("緊急原地解扭力：須有人扶住或已有支撐。輸入 OFF 再按 Enter，其他取消", "OFF"):
            return False
        self._torque_off("emergency", readback=self.read())
        raise Released("已依操作者要求原地解扭力")

    def _torque_off(self, kind, **fields):
        self.f.close()
        torque = self.torque()
        if torque != OFF:
            self.event("torque_off_failed", torque=torque)
            raise RuntimeError("torque off was not verified on every joint")
        self.armed = False
        after = self.read()
        self.event("torque_off", release=kind, torque=torque, after=after, **fields)
        self.stage("released", release=kind, torque=torque, pose=after)

    # -- leader engagement --------------------------------------------------
    def align_leader(self, leader, reference):
        """Engage the leader at the follower pose: driven hand-over when the leader supports it, else manual."""
        self.clutch = None
        if self.drive_leader and not self.relative and hasattr(leader, "drive_on"):
            return self._handover(leader, reference)
        return self._manual_align(leader, reference)

    def _handover(self, leader, reference):
        """Drive the leader to the follower pose, count down, release the leader while the follower holds."""
        while True:
            choice = self.term.choose("leader 自動移到 follower 的姿勢，手先離開 leader。"
                                      "a 開始／m 改手動對齊／q 取消", {"a": "自動對齊", "m": "手動", "q": "取消"})
            if choice == "q":
                return False
            if choice == "m":
                return self._manual_align(leader, reference)
            self._leader = leader
            start = pose(leader.read())
            route = self.leader_route
            if route is not None and body_gap(start, route.points[0]) <= LEADER_ROUTE_NEAR:
                self.leader_rest = start
            leader.drive_on()
            self.event("leader_torque_on", leader=pose(leader.read()), follower=reference)
            try:
                outcome = self._drive_leader(leader, reference)
                if outcome == "arrived":
                    outcome = self._countdown(leader)
            except BaseException as error:
                self.event("leader_error", error=repr(error))
                self.term.say(f"leader 交接出錯：{error}")
                try:
                    self._release_leader("error")
                except Exception as release_error:  # noqa: BLE001 - keep the original failure
                    self.event("leader_release_failed", error=repr(release_error))
                    self.term.say(f"leader 解扭力也未確認：{release_error}")
                raise
            if outcome == "released":
                return self._after_release(leader, reference)
            # stopped with the leader holding: the operator supports it before its torque goes off
            notes = {"esc": "已暫停", "stall": "leader 被擋住或卡住", "short": "leader 到不了 follower 的姿勢"}
            self.term.say(f"{notes[outcome]}，leader 原地保持中。")
            choice = self.term.choose("r 重新自動對齊／m 扶住 leader 後解扭力，改手動對齊／q 扶住 leader 後解扭力並取消",
                                      {"r": "重試", "m": "手動", "q": "取消"})
            if choice == "r":
                continue  # leader stays torque-on; drive_on re-holds its current pose
            self._release_leader_supported()
            return self._manual_align(leader, reference) if choice == "m" else False

    def _leader_goals(self, start, reference):
        """Waypoints for the driven leader. A leader resting folded near park follows the taught route out of the
        fold (a straight joint move made its gripper rub on the way up); the gripper keeps its opening until the
        body has arrived and opens last."""
        hold = {"gripper": start["gripper"]}
        goals = []
        route = self.leader_route
        if (route is not None and body_gap(start, route.points[0]) <= LEADER_ROUTE_NEAR
                and body_gap(route.points[-1], reference) <= LEADER_ROUTE_NEAR):
            goals = [pose(p) | hold for p in route.points[1:]]
        goals.append(pose(reference) | hold)
        goals.append(pose(reference))
        return goals

    def hold_leader(self, leader):
        """Torque the leader on where it is (end of a demonstration): the operator lets go and it does not drop."""
        self._leader = leader
        leader.drive_on()
        self.event("leader_hold", leader=pose(leader.read()))

    def park_leader(self, leader, goals):
        """Drive a held leader along goals (the last one: its folded rest pose) and release it there. Returns
        True when it arrived; otherwise it holds until the operator supports it, then releases."""
        self._leader = leader
        outcome = self._drive_leader(leader, goals[-1], goals=[dict(g) for g in goals])
        if outcome == "arrived":
            self._release_leader("rest")
            return True
        notes = {"esc": "已暫停", "stall": "leader 被擋住或卡住", "short": "leader 沒回到收折位"}
        self.term.say(f"{notes.get(outcome, outcome)}，leader 原地保持中。")
        self._release_leader_supported()
        return False

    def _drive_leader(self, leader, reference, goals=None):
        goal = pose(leader.read())
        goals = goals or self._leader_goals(goal, reference)
        self.event("leader_drive_plan", waypoints=len(goals), via_route=len(goals) > 2)
        stall_since = reached_at = None
        next_status = 0
        while True:
            began = self.clock()
            if self.term.key() == "esc":
                leader.drive(pose(leader.read()))
                return "esc"
            aim = goals[0]
            for joint in JOINTS:
                step = LEADER_SPEED[joint] / self.hz
                goal[joint] += max(-step, min(step, aim[joint] - goal[joint]))
            if len(goals) > 1 and all(goal[j] == aim[j] for j in JOINTS):
                goals.pop(0)
            leader.drive(goal)
            now = pose(leader.read())
            lag = max(body_gap(goal, now), abs(goal["gripper"] - now["gripper"]) / 3)
            if lag > LEADER_STALL_LAG:
                stall_since = stall_since or began
                if began - stall_since > LEADER_STALL_S * 1e9:
                    leader.drive(now)
                    self.event("leader_stall", goal=goal, leader=now)
                    return "stall"
            else:
                stall_since = None
            if all(goal[j] == reference[j] for j in JOINTS):
                reached_at = reached_at or began
                if all(abs(now[j] - reference[j]) <= LEADER_ARRIVED[j] for j in JOINTS):
                    self.event("leader_arrived", leader=now, follower=reference)
                    return "arrived"
                if began - reached_at > LEADER_SETTLE_S * 1e9:
                    self.event("leader_short", leader=now, follower=reference)
                    return "short"
            if began >= next_status:
                self.term.status(f"leader 移動中，距 follower 姿勢 {body_gap(now, reference):4.1f}°（Esc 暫停）")
                next_status = began + 200_000_000
            self.tick(began)

    def _countdown(self, leader):
        end = self.clock() + HANDOVER_COUNTDOWN_S * 1e9
        self.term.say("leader 已就位。現在握住 leader，倒數結束後 leader 解扭力、由你接手。")
        while True:
            began = self.clock()
            left = (end - began) / 1e9
            if left <= 0:
                break
            if self.term.key() == "esc":
                return "esc"
            self.term.status(f"握住 leader：{int(left) + 1} 秒後解 leader 扭力（Esc 取消）")
            self.tick(began)
        self._release_leader("handover")
        return "released"

    def _after_release(self, leader, reference):
        end = self.clock() + HANDOVER_GUARD_S * 1e9
        while (began := self.clock()) < end:  # follower keeps holding; leader movement here is ignored
            self.term.status("leader 已解扭力，接手中…")
            self.tick(began)
        after = pose(leader.read())
        sag = body_gap(after, reference)
        if sag > max(ENGAGE_GAP[j] for j in BODY):
            self.event("leader_dropped", leader=after, follower=reference, gap_deg=sag)
            self.term.say(f"leader 解扭力後偏離 {sag:.0f}°，沒有扶穩。改用手動對齊。")
            return self._manual_align(leader, reference)
        self.clutch = after
        self.event("leader_engaged", leader=after, follower=reference, handover="driven", sag_deg=sag)
        self.term.say("已交接，follower 跟隨 leader 的位移。")
        return True

    def _release_leader(self, kind):
        self._leader.release()
        self.event("leader_torque_off", release=kind, leader=pose(self._leader.read()))

    def _release_leader_supported(self):
        while self.term.key(0.2) != "enter":
            self.term.status("扶住 leader 後按 Enter，解 leader 扭力")
        self._release_leader("supported")

    def _manual_align(self, leader, reference):
        """Show the leader-follower gap until it is small and the operator presses Enter."""
        if self.relative:
            self.term.say("相對映射：follower 會從目前姿勢跟隨 leader 的位移。握好 leader 後按 Enter；Esc 取消。")
        else:
            self.term.say("把 leader 擺到和 follower 相近的姿勢，差距都在門檻內後按 Enter 開始跟隨；Esc 取消。")
        while True:
            lead = pose(leader.read())
            over = [] if self.relative else [f"{j} {lead[j] - reference[j]:+.0f}°" for j in JOINTS
                                             if abs(lead[j] - reference[j]) > ENGAGE_GAP[j]]
            self.term.status("leader 對齊完成，按 Enter" if not over else "尚需對齊：" + "  ".join(over))
            key = self.term.key(0.2)
            if key == "esc":
                return False
            if key == "enter":
                if not over:
                    self.event("leader_engaged", leader=lead, follower=reference)
                    return True
                self.term.say("差距仍超過門檻，請再調整 leader。")


# -- route file ----------------------------------------------------------------
def load_route(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    track = Track(data["waypoints"], data["segment_seconds"])
    return data, track


def save_route(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, data)


RETRACE_PATH_ERROR = 2.0    # body deg: reversed task path back to the channel
TUNE_PATH_ERROR = 1.0      # body deg: drops hand jitter from a taught route, keeps its shape
OPEN_AFTER_LIFT = 45.0     # gripper stays at park opening until shoulder_lift has risen this far from park


def tune_route(data, *, grip, body_error=TUNE_PATH_ERROR, open_after_lift=OPEN_AFTER_LIFT, straight_from=None,
               hold_roll=False):
    """Route without the taught pauses: speed-capped timing, jitter removed, fixed gripper opening.

    The gripper keeps the park opening through the folded area and switches to `grip` once shoulder_lift has
    risen `open_after_lift` deg from park (reverse: closes there before entering the folded area).
    straight_from: waypoint index after which the final adjustments become one straight joint move to the channel.
    hold_roll: keep wrist_roll at the park value along the whole route (no wrist turn; protects the wrist camera mount).
    """
    if not 0.0 <= grip <= 100.0:
        raise ValueError("gripper opening must be within 0..100 %")
    points = [pose(p) for p in data["waypoints"]]
    if straight_from is not None:
        if not 0 <= straight_from < len(points) - 1:
            raise ValueError(f"straight_from must be within 0..{len(points) - 2}")
        points = points[:straight_from + 1] + points[-1:]
    park_grip = data["park"]["pose"]["gripper"]
    lift0 = points[0]["shoulder_lift"]
    switch = next((i for i, p in enumerate(points) if p["shoulder_lift"] - lift0 >= open_after_lift), None)
    if switch is None:
        raise ValueError(f"shoulder_lift never rises {open_after_lift} deg from park along this route")
    points = [p | {"gripper": park_grip if i < switch else float(grip)} for i, p in enumerate(points)]
    if hold_roll:
        points = [p | {"wrist_roll": data["park"]["pose"]["wrist_roll"]} for p in points]
    points = [points[i] for i in sorted(_simplify(points, body_error))]
    seconds = [max(1e-3, body_gap(a, b) / PLAY_SPEED, abs(a["gripper"] - b["gripper"]) / GRIPPER_PLAY_SPEED)
               for a, b in zip(points, points[1:])]
    out = dict(data)
    out["waypoints"], out["segment_seconds"] = points, seconds
    # The old readback belongs to the old channel pose; keep it only as history until the next verified run.
    out["channel"] = dict(data["channel"]) | {"command": dict(points[-1]), "readback": dict(points[-1]),
                                              "readback_before_tune": data["channel"]["readback"]}
    out["verified"] = {"reverse_to_park": None, "forward_to_channel": None}
    out["tuned"] = {"at": datetime.now().astimezone().isoformat(), "gripper_open_pct": float(grip),
                    "gripper_open_after_lift_deg": open_after_lift, "path_tolerance_deg": body_error,
                    "timing": f"speed cap {PLAY_SPEED} deg/s, {GRIPPER_PLAY_SPEED} %/s; taught pauses removed",
                    "straight_from": straight_from, "hold_roll": hold_roll, "waypoints_before": len(data["waypoints"]), "waypoints_after": len(points),
                    "open_from_waypoint": next(i for i, p in enumerate(points) if p["gripper"] == float(grip))}
    return out


def archive_existing(path):
    path = Path(path)
    if not path.exists():
        return None
    index = 1
    while (old := path.with_name(f"{path.stem}.previous-{index:02}{path.suffix}")).exists():
        index += 1
    path.rename(old)
    return old


def sample_park(follower, wait, count=6):
    samples = []
    for i in range(count):
        if i:
            wait(0.05)
        samples.append(pose(follower.read()))
    if follower.torque_states() != OFF:
        raise RuntimeError("all follower axes must be torque-off while sampling the park pose")
    if any(body_gap(s, samples[0]) > 0.15 for s in samples):
        raise RuntimeError("follower is still moving on its support; wait until it is stable")
    return samples


# -- flows ---------------------------------------------------------------------
def follow(driver, track, ends):
    """Play track from ends[0] to ends[1] with the pause menu. Returns the endpoint reached."""
    while True:
        try:
            driver.stage(f"to_{ends[1]}")
            try:
                driver.play(track, f"往{ends[1]}")
                driver.settle()
            except Paused as pause:
                if pause.track is None:  # paused while settling at the end of the track
                    pause.track, pause.index = track, len(track.points) - 2
                raise
            driver.event("reached", endpoint=ends[1], readback=driver.last)
            driver.term.say(f"已到{ends[1]}。")
            return ends[1]
        except Paused as pause:
            driver.stage("paused", reason=str(pause), pose=driver.last)
            driver.term.say(f"已暫停並原地保持：{pause}")
            choice = driver.menu(
                f"g 繼續往{ends[1]}／b 原路退回{ends[0]}／x 緊急原地解扭力",
                {"g": "繼續", "b": "退回", "x": "緊急解力"})
            if choice == "b":
                track = pause.track.head_back(pause.index, driver.target)
                ends = (ends[1], ends[0])
            else:
                track = pause.track.tail(pause.index, driver.target)


def park_and_release(driver, track_to_park, park, support):
    """Reverse along track_to_park, confirm the support, release torque at park."""
    track = track_to_park
    while True:
        if follow(driver, track, ("出發點", "停放位")) != "停放位":
            driver.menu("已退回出發點並保持。r 再次沿路線回停放位／x 緊急原地解扭力",
                        {"r": "回停放位", "x": "緊急解力"})
            track = track_to_park
            continue
        while not driver.release(park, support):
            driver.menu("s 重新對位停放位再試／x 緊急原地解扭力", {"s": "再對位", "x": "緊急解力"})
            gap = body_gap(driver.read(), park)
            if gap > 3 * RELEASE_LAG:
                driver.term.say(f"距停放位 {gap:.1f}°，太遠，不做直線對位。")
                continue
            seat = Track([driver.target, park | {"gripper": driver.target["gripper"]}], [0.0])
            follow(driver, seat, ("目前位置", "停放位"))
        return True


def _resume_teleop(driver, leader, samples):
    """After a pause: realign the leader. False when the operator cancels."""
    driver.target = driver.clip(driver.read())[0]
    if not driver.align_leader(leader, driver.last):
        return False
    samples.append((driver.clock(), dict(driver.target)))
    return True


def retrace_home(samples, channel):
    """Back to the channel along the path just driven, reversed: it is known to clear the trays and the arm never
    swings back over itself (raising shoulder_lift alone with a straight elbow did). Hand jitter and pauses are
    dropped and it plays at the route speed caps; the gripper keeps its current opening."""
    points = [p for _, p in samples]
    points = [points[i] for i in sorted(_simplify(points, RETRACE_PATH_ERROR))][::-1]
    grip = points[0]["gripper"]
    return retimed([*(p | {"gripper": grip} for p in points[:-1]), pose(channel) | {"gripper": grip}])


def lift_home(current, channel):
    """Back to the channel without retracing: raise shoulder_lift to the channel value first (the upper arm tilts
    back, the gripper rises away from the trays), then move the other joints straight to the channel pose."""
    current = pose(current)
    points = [current]
    if current["shoulder_lift"] > channel["shoulder_lift"]:
        points.append(current | {"shoulder_lift": channel["shoulder_lift"]})
    points.append(pose(channel) | {"gripper": current["gripper"]})
    return retimed(points)


def task_session(driver, leader, channel_command):
    """Leader teleop from the channel; returns along the measured task path to the channel."""
    if not driver.align_leader(leader, driver.read()):
        return
    task = [(driver.clock(), dict(driver.target))]
    while True:
        try:
            driver.stage("task_teleop")
            driver.term.say("leader 操作中。c 沿本次實走路徑退回通道位；Esc 暫停。")
            driver.teleop(leader, {"c"}, task)
            break
        except Paused as pause:
            driver.term.say(f"已暫停並原地保持：{pause}")
            choice = driver.menu("t 重新對齊 leader 繼續／c 沿實走路徑退回通道位／x 緊急原地解扭力",
                                 {"t": "繼續", "c": "回通道", "x": "緊急解力"})
            if choice == "c" or not _resume_teleop(driver, leader, task):
                break
    task.append((driver.clock(), dict(driver.target)))
    follow(driver, retrace_home(task, channel_command), ("任務位", "通道位"))


def record(driver, leader, route_path, meta, park_samples):
    """Teach park -> channel with the leader, save the route, then verify the reverse."""
    park = park_samples[-1]
    driver.term.say("扭力開啟前：follower 留在停放支撐上。")
    if not driver.align_leader(leader, park):
        driver.stage("cancelled")
        return None
    driver.arm_at(park)
    samples = [(driver.clock(), dict(driver.target))]
    while True:
        try:
            driver.stage("recording")
            driver.term.say("開始跟隨。用 leader 把手臂從停放位帶到通道位，停穩後按 c 存通道位；Esc 暫停。")
            driver.teleop(leader, {"c"}, samples)
            break
        except Paused as pause:
            driver.term.say(f"已暫停並原地保持：{pause}")
            choice = None
            while choice != "t":
                choice = driver.menu(
                    "t 重新對齊 leader 繼續錄／r 沿已錄路徑退回停放位並解扭力（不存路線）／x 緊急原地解扭力",
                    {"t": "繼續", "r": "退回", "x": "緊急解力"})
                if choice == "r":
                    samples.append((driver.clock(), dict(driver.target)))
                    park_and_release(driver, compress(samples).reverse(), park, meta["park_support"])
                    driver.stage("released_without_route")
                    return None
                if choice == "t" and not _resume_teleop(driver, leader, samples):
                    choice = None
    channel_readback, still = driver.settle()
    samples.append((driver.clock(), dict(driver.target)))
    track = compress(samples)
    track.points[0] = dict(park)  # the path starts exactly at the supported park reading
    data = {
        "format": FORMAT, **meta, "joint_units": UNITS, "created_at": now_iso(),
        "limits": {j: list(v) for j, v in driver.limits.items()},
        "park": {"pose": park, "samples": park_samples, "torque": OFF},
        "channel": {"command": dict(driver.target), "readback": channel_readback, "stationary": still},
        **track.to_json(),
        "recorded": {"run": str(driver.run_dir), "raw_samples": len(samples),
                     "moving_seconds": round(sum(track.seconds), 3),
                     "path_tolerance_deg": PATH_ERROR, "gripper_path_tolerance_pct": GRIPPER_PATH_ERROR},
        "verified": {"reverse_to_park": None, "forward_to_channel": None},
    }
    save_route(route_path, data)
    driver.event("route_saved", path=str(route_path), waypoints=len(track.points))
    driver.term.say(f"通道位已保存：{len(track.points)} 個路徑點，移動時間 {sum(track.seconds):.1f} s → {route_path}")
    channel_hold(driver, leader, data, track, route_path)
    return data


def channel_hold(driver, leader, data, track, route_path):
    """Menu at the channel: leader task test, or reverse to park and release."""
    park = data["park"]["pose"]
    while True:
        driver.stage("held_channel", pose=driver.last)
        options = {"r": "回停放位並解扭力", "x": "緊急解力"}
        if leader is not None:
            options["t"] = "leader 操作"
        prompt = "已在通道位保持。r 原路反向回停放位並解扭力" + ("／t 用 leader 操作再回通道" if leader else "") \
                 + "／x 緊急原地解扭力"
        choice = driver.menu(prompt, options)
        if choice == "t":
            task_session(driver, leader, data["channel"]["command"])
        else:
            if body_gap(driver.read(), data["channel"]["command"]) > ENDPOINT:
                driver.term.say("目前不在通道位附近，無法沿路線退回。")
                continue
            park_and_release(driver, track.reverse(), park, data["park_support"])
            data["verified"]["reverse_to_park"] = str(driver.run_dir)
            save_route(route_path, data)
            driver.term.say("已回停放位並解扭力，六軸讀回為 0。")
            return


RECOVER_GAP = 8.0  # body distance to the nearest route point to allow following it home


def recover(driver, data, track):
    """Torque already on (interrupted session): go back along the route to park and release."""
    driver.f.resume()
    driver.armed = True
    driver.phase = "recover"
    current = driver.read()
    driver.target = driver.clip(current)[0]
    index = min(range(len(track.points)), key=lambda i: body_gap(current, track.points[i]))
    gap = body_gap(current, track.points[index])
    driver.event("recover_measured", readback=current, nearest_index=index, gap_deg=gap)
    driver.stage("recover_measured", pose=current, nearest_index=index, gap_deg=gap)
    if gap > RECOVER_GAP:
        driver.term.say(f"follower 離路線最近點仍有 {gap:.1f}°（>{RECOVER_GAP:.0f}°），不自動沿路線移動。")
        driver.menu("q 保持扭力並離開（由現場處理）／x 緊急原地解扭力", {"q": "離開", "x": "緊急解力"})
        return False
    driver.term.say(f"目前在路線第 {index}/{len(track.points) - 1} 點附近（差 {gap:.1f}°）。")
    if driver.menu("確認空爪、回程淨空。r 沿路線退回停放位並解扭力／x 緊急原地解扭力",
                   {"r": "回停放位", "x": "緊急解力"}) != "r":
        return False
    home = Track([driver.target, *track.points[index::-1]], [0.0, *track.seconds[:index][::-1]])
    return park_and_release(driver, home, data["park"]["pose"], data["park_support"])


def roundtrip(driver, leader, route_path, data, track):
    """From park: forward to the channel, hold, then reverse back and release."""
    park = data["park"]["pose"]
    driver.arm_at(park)
    reached = follow(driver, track, ("停放位", "通道位"))
    if reached == "停放位":
        park_and_release(driver, Track([driver.target, park], [0.0]), park, data["park_support"])
        return
    data["verified"]["forward_to_channel"] = str(driver.run_dir)
    save_route(route_path, data)
    channel_hold(driver, leader, data, track, route_path)


DROP_FORMAT = "so101-drop-route/1"


def record_drop(driver, leader, data, track, drop_path, tray, release):
    """From park: go to the channel, teach channel -> drop pose over a tray with the leader, save it,
    then reverse back to the channel and home to park."""
    park = data["park"]["pose"]
    driver.arm_at(park)
    if follow(driver, track, ("停放位", "通道位")) != "通道位":
        park_and_release(driver, Track([driver.target, park], [0.0]), park, data["park_support"])
        return None
    channel = dict(driver.target)
    if not driver.align_leader(leader, driver.read()):
        park_and_release(driver, track.reverse(), park, data["park_support"])
        return None
    samples = [(driver.clock(), dict(driver.target))]
    while True:
        try:
            driver.stage("recording_drop", tray=tray)
            driver.term.say(f"用 leader 從通道位移到{tray}盤上方的放開位置（方塊離盤面數公分），停穩後按 c；Esc 暫停。")
            driver.teleop(leader, {"c"}, samples)
            break
        except Paused as pause:
            driver.term.say(f"已暫停並原地保持：{pause}")
            choice = driver.menu("t 重新對齊 leader 繼續／r 沿已錄路徑退回通道位再回停放位（不存）／x 緊急原地解扭力",
                                 {"t": "繼續", "r": "退回", "x": "緊急解力"})
            if choice == "r" or not _resume_teleop(driver, leader, samples):
                samples.append((driver.clock(), dict(driver.target)))
                follow(driver, compress(samples).reverse(), ("中途", "通道位"))
                park_and_release(driver, track.reverse(), park, data["park_support"])
                return None
    readback, still = driver.settle()
    samples.append((driver.clock(), dict(driver.target)))
    drop = compress(samples)
    drop.points[0] = channel
    drop = retimed([drop.points[i] for i in sorted(_simplify(drop.points, TUNE_PATH_ERROR))])
    out = {"format": DROP_FORMAT, "tray": tray, "arm": data["arm"], "layout_id": data["layout_id"],
           "calibrations": data["calibrations"], "base_route": data.get("created_at"),
           "release_gripper": release, "drop_readback": readback, "stationary": still,
           "created_at": now_iso(), "recorded": {"run": str(driver.run_dir), "raw_samples": len(samples)},
           **drop.to_json()}
    save_route(drop_path, out)
    driver.event("drop_saved", path=str(drop_path), tray=tray, waypoints=len(drop.points))
    driver.term.say(f"{tray} 放開路線已保存：{len(drop.points)} 個路徑點 → {drop_path}")
    if driver.menu("確認空爪、回程淨空。r 倒放回通道位再回停放位並解扭力／x 緊急原地解扭力",
                   {"r": "回停放位", "x": "緊急解力"}) == "r":
        back_to_channel(driver, drop.reverse(), tray)
        park_and_release(driver, track.reverse(), park, data["park_support"])
    return out


def back_to_channel(driver, back, tray):
    """Reverse a drop route to the channel; if the operator stepped back to the drop pose, offer it again."""
    while follow(driver, back, (f"{tray}放開位", "通道位")) != "通道位":
        driver.menu(f"已退回{tray}放開位並保持。r 再次沿放開路線倒放回通道位／x 緊急原地解扭力",
                    {"r": "回通道位", "x": "緊急解力"})
        start = dict(driver.target)
        back = retimed([start, *back.points[1:]])


def retimed(points):
    """Track over points at the playback speed caps (no taught pauses)."""
    return Track(points, [max(1e-3, body_gap(a, b) / PLAY_SPEED, abs(a["gripper"] - b["gripper"]) / GRIPPER_PLAY_SPEED)
                          for a, b in zip(points, points[1:])])


def drop_test(driver, data, track, drop):
    """Without the leader: park -> channel -> taught drop pose (gripper held, optionally on a block) -> open to the
    release opening -> reverse to the channel -> park and release. Same order as the collection's auto placement."""
    park, tray = data["park"]["pose"], drop["tray"]
    if drop["calibrations"] != data["calibrations"] or drop["layout_id"] != data["layout_id"]:
        raise ValueError(f"drop route {tray} was taught with another calibration or layout")
    if body_gap(drop["waypoints"][0], data["channel"]["command"]) > ENDPOINT:
        raise ValueError(f"drop route {tray} starts at an older channel pose; teach it again (REPLACE=1)")
    driver.arm_at(park)
    if follow(driver, track, ("停放位", "通道位")) != "通道位":
        park_and_release(driver, Track([driver.target, park], [0.0]), park, data["park_support"])
        return None
    choice = driver.menu(f"在通道位。g 空爪測試{tray}盤放開位／b 把方塊放進爪間後按 b，夾爪閉合夾住再測／"
                         "q 不測，回停放位", {"g": "空爪", "b": "夾方塊", "q": "回停放位"})
    if choice == "q":
        park_and_release(driver, track.reverse(), park, data["park_support"])
        return None
    if choice == "b":
        follow(driver, Track([driver.target, driver.target | {"gripper": 0.0}], [0.0]), ("張爪", "閉爪"))
        driver.target = driver.target | {"gripper": max(0.0, driver.read()["gripper"] - CARRY_SQUEEZE)}
    place_from_channel(driver, drop)
    park_and_release(driver, track.reverse(), park, data["park_support"])
    driver.term.say("已回停放位並解扭力。")
    return True


def place_from_channel(driver, drop):
    """At the channel holding the block: follow the drop route, open, confirm, reverse back to the channel."""
    tray = drop["tray"]
    grip = driver.target["gripper"]
    out = retimed([dict(driver.target), *[pose(p) | {"gripper": grip} for p in drop["waypoints"][1:]]])
    if follow(driver, out, ("通道位", f"{tray}放開位")) == f"{tray}放開位":
        opened = max(grip, drop["release_gripper"])
        follow(driver, Track([driver.target, driver.target | {"gripper": opened}], [0.0]), ("夾住", "放開"))
        driver.event("drop_test_released", tray=tray, readback=driver.last, gripper=opened)
        driver.menu(f"已在{tray}盤放開位張爪到 {opened:.0f}%。確認方塊落進盤內、回程淨空後按 r 倒放回通道位",
                    {"r": "回通道位"})
        back = retimed([dict(driver.target), *[pose(p) | {"gripper": opened} for p in drop["waypoints"][-2::-1]]])
        back_to_channel(driver, back, tray)


def build_setup(data, track, drops, *, version, workspace, cards, speed=20.0, trays=None):
    """Collection setup JSON (configs/real_collection.setup.example.json shape) from taught routes."""
    limits = {j: list(v) for j, v in data["limits"].items()}
    clip = lambda p: {j: min(limits[j][1], max(limits[j][0], p[j])) for j in JOINTS}  # noqa: E731
    motion = {
        "arm": data["arm"], "version": version, "layout": data["layout_id"], "workspace": workspace,
        "calibrations": data["calibrations"], "park_support": data["park_support"],
        "park": clip(data["park"]["pose"]), "via": [clip(p) for p in track.points[1:-1]],
        "start": clip(data["channel"]["command"]), "limits": limits,
        "speed": dict.fromkeys(BODY, float(speed)) | {"gripper": 25.0},
        "rgb_depth": {} if not trays else {"tray_quads": trays["quads"], "tray_reference": trays["reference"]},
        "drops": {}, "routes": {"park_channel": data.get("created_at")},
    }
    for drop in drops:
        if drop["calibrations"] != data["calibrations"] or drop["layout_id"] != data["layout_id"]:
            raise ValueError(f"drop route {drop['tray']} was taught with another calibration or layout")
        points = [clip(p) for p in drop["waypoints"]]
        if body_gap(points[0], motion["start"]) > ENDPOINT:
            raise ValueError(f"drop route {drop['tray']} does not start at this channel pose")
        motion["drops"][drop["tray"]] = {"waypoints": points, "release_gripper": drop["release_gripper"]}
    return {"motion": motion, "cards": cards}
