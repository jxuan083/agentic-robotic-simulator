"""Real-arm skills with the task executor's Skill interface (agentic_sim.task.skills).

Each skill runs one blocking arm routine (agentic_sim.real_task.arm) in a worker thread; step() is called every
control frame by the executor and only watches the thread, reports phase changes and builds the SkillResult.

Cancel: a skill that has not started moving is cancelled at once; once its routine runs it finishes (the hardware
routines stop only through the operator pause). The executor treats that like a phase past the point of no return:
it keeps the skill and handles the new order afterwards (a held block the order no longer needs -> its question).
Hold accepts a cancel at any time (it never moves).
"""

from __future__ import annotations

import threading
import time

from agentic_sim.task.skills import SkillResult


class RealSkill:
    name = "real"

    def __init__(self, ctx, **params):
        self.ctx, self.params = ctx, params
        self.arm, self.tracker = ctx.arm, ctx.tracker
        self.status = "running"
        self.phase = ""
        self.cancel_reason: str | None = None
        self.revision = ctx.ledger.revision
        self.result: SkillResult | None = None
        self.holding = params.get("holding")
        self.evidence: dict = {}
        self.obs = None
        self.t0 = None
        self.phase_t = None
        self.phase_time: dict = {}
        self._thread: threading.Thread | None = None
        self._out = None
        self._phase_seen = ""

    # ---- executor interface ------------------------------------------------------------------
    @property
    def cancellable(self) -> bool:
        return self.status == "running" and self._thread is None

    def cancel(self, reason: str, revision: int) -> bool:
        if not self.cancellable:
            return False
        self.cancel_reason, self.revision = reason, revision
        return True

    def step(self, obs):
        self.obs = obs
        if self.status != "running":
            return
        if self.t0 is None:
            self.t0 = obs.t
        if self._thread is None:
            if self.cancel_reason is not None:
                self.ctx.emit("cancel_taken", skill=self.name, phase=self.phase, reason=self.cancel_reason,
                              block=self.params.get("block"))
                return self._finish("cancelled", self.cancel_reason)
            self._thread = threading.Thread(target=self._work, name=f"skill-{self.name}", daemon=True)
            self._thread.start()
        if self.phase != self._phase_seen:          # phases set by the worker are reported from this thread
            self._report_phase()
        if not self._thread.is_alive():
            kind, value = self._out or ("error", RuntimeError("no result"))
            if kind == "error":
                self.ctx.log(f"[real skill {self.name}] {type(value).__name__}: {value}")
                return self._finish("failed", f"{type(value).__name__}: {value}")
            self.finish(value)

    # ---- worker ---------------------------------------------------------------------------------
    def _work(self):
        try:
            self._out = ("ok", self.run())
        except BaseException as error:  # noqa: BLE001 - reported as a failed skill on the executor's thread
            self._out = ("error", error)

    def set_phase(self, phase: str):
        """Called from the worker; reported on the next executor frame."""
        self.phase = phase

    def _report_phase(self):
        now = self.obs.t
        if self._phase_seen:
            self.phase_time[self._phase_seen] = self.phase_time.get(self._phase_seen, 0.0) + (now - self.phase_t)
        self._phase_seen, self.phase_t = self.phase, now
        self.ctx.emit("phase", skill=self.name, phase=self.phase, block=self.params.get("block"),
                      color=self.params.get("color"), nth=self.params.get("nth"))

    def run(self):
        raise NotImplementedError

    def finish(self, value):
        self._finish("succeeded")

    def _finish(self, status: str, reason: str = ""):
        if self._phase_seen and self.phase_t is not None:
            self.phase_time[self._phase_seen] = self.phase_time.get(self._phase_seen, 0.0) + (self.obs.t - self.phase_t)
        self.status = status
        self.result = SkillResult(self.name, dict(self.params), status, reason, self.obs.t - (self.t0 or self.obs.t),
                                  dict(self.phase_time), dict(self.evidence), self.holding)


class RealObserve(RealSkill):
    """D455 look at the storage tray from park (the arm folded out of the view). Evidence as TaskCamera's Observe,
    each item with its tracker ID ("block")."""
    name = "observe"

    def run(self):
        self.set_phase("look")
        if not self.arm.at_park():
            if self.ctx.holding is not None:
                raise RuntimeError("asked to look while holding a block away from park")
            self.set_phase("park")
            self.arm.home()
            self.set_phase("look")
        return self.arm.scan()

    def finish(self, value):
        blocks, clear, reason = value
        held = () if self.ctx.holding is None else (self.ctx.holding,)
        matched = self.tracker.update(blocks, exclude=held)
        items = [{"block": b, "color": d["color"], "xy": [float(d["x"]), float(d["y"])],
                  "yaw": float(d.get("yaw", 0.0)), "fit": 1.0, "source": "d455"}
                 for b, d in matched if self.tracker.tray[b] == "storage"]
        self.evidence = {"items": items, "clear": bool(clear), "reason_unclear": reason or "",
                         "views": [{"camera": "d455", "blocks": len(blocks)}]}
        self._finish("succeeded")


class RealPick(RealSkill):
    """Park -> start pose -> program grasp of the block -> lifted, holding. A miss opens, goes back to park and
    fails ("missed"); the executor retries (task.yaml retry.pick_attempts) or skips the block."""
    name = "pick"

    def run(self):
        b = self.params["block"]
        target = self.params.get("target") or {}
        xy = target.get("xy") or [float(self.tracker.pos[b][0]), float(self.tracker.pos[b][1])]
        yaw = float(target.get("yaw", self.tracker.yaw[b]))
        self.set_phase("approach")
        held, info = self.arm.pick({"color": self.tracker.colors[b], "x": xy[0], "y": xy[1],
                                    "z_top": float(self.tracker.pos[b][2]) + 0.0295 / 2, "yaw": yaw},
                                   phase=self.set_phase)
        return held, info

    def finish(self, value):
        held, info = value
        b = self.params["block"]
        self.evidence = dict(info or {})
        if held:
            self.holding = b
            self.tracker.remove(b)          # off the table, in the gripper
            return self._finish("succeeded")
        self.holding = None
        self._finish("failed", (info or {}).get("reason", "missed"))


class RealPlace(RealSkill):
    """Holding at the lift: carry into the slot, open, rise, back to park; the D455 at park verifies the landing
    (a block of that colour in the placement tray near the slot). Not cancellable once started."""
    name = "place"
    VERIFY_M = 0.04

    def run(self):
        xy = self.params["xy"]
        self.set_phase("transport")
        err = self.arm.place(xy, phase=self.set_phase)
        if err:
            return {"error": err}
        self.set_phase("verify")
        blocks, clear, reason = self.arm.scan()
        return {"blocks": blocks, "clear": clear}

    def finish(self, value):
        b = self.params["block"]
        if value.get("error"):
            self.evidence = {"error": value["error"]}
            self.holding = b                    # nothing moved: still holding (the arm raises at its lift)
            return self._finish("failed", value["error"])
        xy = self.params["xy"]
        near = [d for d in value["blocks"] if d.get("tray") == "beige" and d["color"] == self.tracker.colors[b]
                and ((d["x"] - xy[0]) ** 2 + (d["y"] - xy[1]) ** 2) ** .5 < self.VERIFY_M]
        self.holding = None
        if near:
            d = min(near, key=lambda d: (d["x"] - xy[0]) ** 2 + (d["y"] - xy[1]) ** 2)
            self.tracker.move(b, (d["x"], d["y"]), "placement")
            self.evidence = {"landed_xy": [round(d["x"], 4), round(d["y"], 4)],
                             "slot_offset_mm": round(1000 * ((d["x"] - xy[0]) ** 2 + (d["y"] - xy[1]) ** 2) ** .5, 1)}
            return self._finish("succeeded")
        self.tracker.move(b, xy, None)
        self.evidence = {"seen_in_placement": [d for d in value["blocks"] if d.get("tray") == "beige"]}
        self._finish("uncertain", "not_verified")


class RealHome(RealSkill):
    """Open the gripper (empty hand) and go back to park: standby, or the end of the task."""
    name = "home"

    def run(self):
        self.set_phase("home")
        self.arm.home(open_gripper=self.ctx.holding is None)


class RealHold(RealSkill):
    """Keep the block where it is while a question is open (the servos hold the last command)."""
    name = "hold"

    def step(self, obs):
        self.obs = obs
        if self.status != "running":
            return
        if self.t0 is None:
            self.t0 = obs.t
            self.phase = "hold"
            self._report_phase()
        if self.cancel_reason is not None:
            self._finish("cancelled", self.cancel_reason)

    @property
    def cancellable(self) -> bool:
        return self.status == "running"

    def cancel(self, reason: str, revision: int) -> bool:
        if self.status != "running":
            return False
        self.cancel_reason, self.revision = reason, revision
        return True


SKILLS = {"pick": RealPick, "place": RealPlace, "home": RealHome, "hold": RealHold, "observe": RealObserve}


def wait_for(skill, ctx, obs_fn, timeout=30.0, dt=0.01):
    """Test helper: step a skill until it finishes."""
    end = time.monotonic() + timeout
    while skill.status == "running" and time.monotonic() < end:
        skill.step(obs_fn())
        time.sleep(dt)
    return skill.result
