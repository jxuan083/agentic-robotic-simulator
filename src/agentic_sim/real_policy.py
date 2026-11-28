"""A LeRobot ACT checkpoint driving the follower (the plush toy's grasp; later the blocks' S3), in the same
observation layout the converter wrote (tools/real_to_lerobot.py, agentic_sim.real_dataset):

  observation.state                follower readback, LeRobot degrees / gripper %   (JOINTS order)
  observation.images.<camera>      RGB, the recorded resolution; the global one with the target marker drawn
                                   (real_dataset.draw_markers at the target's pixel) when the dataset had --markers
  observation.environment_state    the target condition (real_dataset.condition: x, y mm, sin / cos fold*yaw)
  action                           follower command

The loop (run): every control period, readback -> newest camera frames -> one action -> Driver.send (the driver
clips to the joint limits). It stops when done() says the grasp is finished, on Esc, on a time-out, or when an action
jumps more than STEP_LIMIT_DEG on a body joint from the last command (docs/real-dagger.md: ACT 越界); then the
follower holds where it is.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from agentic_sim.real_control import JOINTS

BODY = JOINTS[:-1]
STEP_LIMIT_DEG = 6.0
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


def run(driver, policy, frames, *, condition=None, marks=None, ee_z=None, timeout_s=TIMEOUT_S, phase=None) -> dict:
    """Run the policy on the follower until the grasp is done (see the module doc). frames() -> {camera: BGR};
    ee_z(pose) -> grasp-centre height (m) for the lift test. Returns {ok, why, steps, seconds, grasp_z,
    gripper_pct, held}; the follower holds its last command at the end."""
    from agentic_sim.real_route import Paused

    policy.reset()
    began = driver.clock()
    low_z, closed_at, still_since, last = math.inf, None, None, dict(driver.target)
    steps, why = 0, "timeout"
    open_cmd = float(driver.target["gripper"])
    while True:
        t0 = driver.clock()
        if driver.stop_requested():
            driver.hold("operator stop")
            raise Paused("操作者暫停")
        state = driver.read()
        action = policy([float(state[j]) for j in JOINTS], to_inputs(frames(), marks), condition)
        target = dict(zip(JOINTS, action))
        jump = max(abs(target[j] - last[j]) for j in BODY)
        if jump > STEP_LIMIT_DEG:
            driver.hold(f"policy step {jump:.1f} deg")
            why = f"step {jump:.1f} deg"
            break
        driver.send(target)
        steps += 1
        elapsed = (driver.clock() - began) / 1e9
        if phase and steps == 1:
            phase("policy")
        z = ee_z(driver.target) if ee_z else None
        if z is not None and closed_at is None:
            low_z = min(low_z, z)
        if closed_at is None and target["gripper"] < open_cmd - 10.0:
            closed_at = elapsed
        moving = jump > 0.3
        still_since = None if moving else (still_since if still_since is not None else elapsed)
        lifted = z is None or z >= low_z + LIFT_M
        if closed_at is not None and lifted and still_since is not None and elapsed - still_since >= STILL_S:
            why = "done"
            break
        if elapsed > timeout_s:
            driver.hold("policy timeout")
            break
        last = target
        driver.tick(t0)
    reading = driver.read()
    grip = float(reading["gripper"])
    return {"ok": why == "done", "why": why, "steps": steps, "seconds": round((driver.clock() - began) / 1e9, 2),
            "grasp_z": None if math.isinf(low_z) else round(low_z, 4), "gripper_pct": round(grip, 1),
            "held": grip >= float(driver.target["gripper"]) + HELD_GAP_PCT}
