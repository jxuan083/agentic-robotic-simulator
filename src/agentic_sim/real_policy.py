"""A LeRobot ACT checkpoint driving the follower (the plush toy's grasp; later the blocks' S3), in the same
observation layout the converter wrote (tools/real_to_lerobot.py, agentic_sim.real_dataset):

  observation.state                follower readback, LeRobot degrees / gripper %   (JOINTS order)
  observation.images.<camera>      RGB, the recorded resolution; the global one with the target marker drawn
                                   (real_dataset.draw_markers at the target's pixel) when the dataset had --markers
  observation.environment_state    the target condition (real_dataset.condition: x, y mm, sin / cos fold*yaw)
  action                           follower command

The loop (run): every control period, readback -> newest camera frames -> one action -> Driver.send (the driver
clips to the joint limits). Watch holds the per-step rules, shared with the MuJoCo evaluation
(tools/sim_plush_eval.py):
- an action more than STEP_LIMIT_DEG from the last command on a body joint is approached at that rate (ACT's first
  action and the first of each new chunk sit 3-7 deg off: the 11/28 sim study); one more than STOP_JUMP_DEG away is
  out of bounds (docs/real-dagger.md: ACT 越界) and stops the loop;
- the grasp is done when the gripper command has closed 10 % below its widest so far (with open_pct: only after
  it reached open_pct, so a low first gripper action is no close), the grasp centre is LIFT_M above its lowest point
  and the commands have settled for STILL_S.
It also stops on Esc and on a time-out; then the follower holds where it is.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from agentic_sim.real_control import JOINTS

BODY = JOINTS[:-1]
STEP_LIMIT_DEG = 6.0   # per control step: a larger jump is approached at this rate
STOP_JUMP_DEG = 15.0   # out of bounds: stop
CLOSE_PCT = 10.0       # closed: the gripper command this far below its widest so far
PLUSH_OPEN_PCT = 42.0  # the plush grasp opens to >= 46 % (docs/plush.md), the start pose holds ~36 %
TIMEOUT_S = 20.0
LIFT_M = 0.04          # done: the gripper has closed and the grasp centre is this far above its lowest point ...
STILL_S = 0.3          # ... and the commands have settled for this long
HELD_GAP_PCT = 4.0     # held: the gripper reads this much more open than commanded (something between the jaws)


class ActPolicy:
    """ACT checkpoint (…/pretrained_model) with its saved pre / post processors."""

    def __init__(self, checkpoint: str):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.policy = ACTPolicy.from_pretrained(checkpoint)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint)
        self.inputs = set(self.policy.config.input_features)
        self.device = next(self.policy.parameters()).device

    def reset(self):
        self.policy.reset()

    def __call__(self, state, images_rgb: dict, condition=None) -> list:
        import torch

        obs = {"observation.state": torch.tensor([state], dtype=torch.float32)}
        for name, img in images_rgb.items():
            key = f"observation.images.{name}"
            if key in self.inputs:
                obs[key] = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0)[None]
        if "observation.environment_state" in self.inputs:
            obs["observation.environment_state"] = torch.tensor([condition], dtype=torch.float32)
        obs["task"] = [""]
        with torch.no_grad():
            act = self.post(self.policy.select_action(self.pre(obs)))
        return [float(x) for x in act.squeeze(0).cpu()]


class LatestFrames:
    """Each camera read in its own thread; frames() returns the newest BGR frame of each (cameras keep their own
    rate, the control loop never waits on a camera)."""

    def __init__(self, readers: dict):
        self.readers, self.latest, self.error = readers, {}, None
        self.stop = threading.Event()
        self.threads = [threading.Thread(target=self._run, args=(n, r), daemon=True) for n, r in readers.items()]
        for t in self.threads:
            t.start()

    def _run(self, name, read):
        try:
            while not self.stop.is_set():
                color = read()[0]
                self.latest[name] = color
        except Exception as error:  # noqa: BLE001 - reported to the control loop
            self.error = f"{name}: {error}"

    def frames(self, wait_s=3.0) -> dict:
        end = time.monotonic() + wait_s
        while len(self.latest) < len(self.readers) and self.error is None and time.monotonic() < end:
            time.sleep(0.01)
        if self.error or len(self.latest) < len(self.readers):
            raise RuntimeError(self.error or "no camera frame")
        return dict(self.latest)

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=1.0)


def to_inputs(frames_bgr: dict, marks: dict | None) -> dict:
    """BGR camera frames -> the RGB images the dataset holds (marker drawn on the global camera)."""
    from PIL import Image

    from agentic_sim.real_dataset import MARKER_CAMERA, draw_markers
    out = {}
    for name, bgr in frames_bgr.items():
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        if marks and name == MARKER_CAMERA:
            rgb = np.asarray(draw_markers(Image.fromarray(rgb), marks))
        out[name] = rgb
    return out


class Watch:
    """run()'s per-step rules (the module doc), from the command the follower holds at the start (joint -> value)."""

    def __init__(self, start: dict, open_pct=None):
        self.last = dict(start)
        self.peak = float(start["gripper"])
        self.open_pct = open_pct
        self.opened = open_pct is None or self.peak >= open_pct
        self.low_z, self.closed_at, self.still_since = math.inf, None, None
        self.jump, self.ramped = 0.0, 0

    def command(self, action: dict) -> dict | None:
        """The command to send for this action (ramped), or None when it is out of bounds (self.jump says how far)."""
        self.jump = max(abs(action[j] - self.last[j]) for j in BODY)
        if self.jump > STOP_JUMP_DEG:
            return None
        if self.jump > STEP_LIMIT_DEG:
            self.ramped += 1
            action = action | {j: self.last[j] + float(np.clip(action[j] - self.last[j], -STEP_LIMIT_DEG,
                                                                STEP_LIMIT_DEG)) for j in BODY}
        self.step = max(abs(action[j] - self.last[j]) for j in BODY)
        self.last = dict(action)
        return action

    def done(self, t: float, z) -> bool:
        """After sending: t seconds since the start, z the grasp-centre height of the command (None: no lift test)."""
        g = float(self.last["gripper"])
        if self.closed_at is None:
            if z is not None:
                self.low_z = min(self.low_z, z)
            self.opened |= self.open_pct is not None and g >= self.open_pct
            if self.opened and g < self.peak - CLOSE_PCT:
                self.closed_at = t
            self.peak = max(self.peak, g)
        moving = self.step > 0.3
        self.still_since = None if moving else (self.still_since if self.still_since is not None else t)
        lifted = z is None or z >= self.low_z + LIFT_M
        return (self.closed_at is not None and lifted and self.still_since is not None
                and t - self.still_since >= STILL_S)


def run(driver, policy, frames, *, condition=None, marks=None, ee_z=None, timeout_s=TIMEOUT_S, phase=None,
        open_pct=None) -> dict:
    """Run the policy on the follower until the grasp is done (see the module doc). frames() -> {camera: BGR};
    ee_z(pose) -> grasp-centre height (m) for the lift test; open_pct: see Watch. Returns {ok, why, steps, seconds,
    grasp_z, gripper_pct, held, ramped}; the follower holds its last command at the end."""
    from agentic_sim.real_route import Paused

    policy.reset()
    began = driver.clock()
    w = Watch(driver.target, open_pct)
    steps, why = 0, "timeout"
    while True:
        t0 = driver.clock()
        if driver.stop_requested():
            driver.hold("operator stop")
            raise Paused("操作者暫停")
        state = driver.read()
        action = policy([float(state[j]) for j in JOINTS], to_inputs(frames(), marks), condition)
        target = w.command(dict(zip(JOINTS, action)))
        if target is None:
            driver.hold(f"policy step {w.jump:.1f} deg")
            why = f"step {w.jump:.1f} deg"
            break
        driver.send(target)
        steps += 1
        elapsed = (driver.clock() - began) / 1e9
        if phase and steps == 1:
            phase("policy")
        if w.done(elapsed, ee_z(driver.target) if ee_z else None):
            why = "done"
            break
        if elapsed > timeout_s:
            driver.hold("policy timeout")
            break
        driver.tick(t0)
    reading = driver.read()
    grip = float(reading["gripper"])
    return {"ok": why == "done", "why": why, "steps": steps, "seconds": round((driver.clock() - began) / 1e9, 2),
            "grasp_z": None if math.isinf(w.low_z) else round(w.low_z, 4), "gripper_pct": round(grip, 1),
            "held": grip >= float(driver.target["gripper"]) + HELD_GAP_PCT, "ramped": w.ramped}
