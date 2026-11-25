"""LeRobot/Feetech and camera adapters. Driver imports happen at device open."""

from contextlib import contextmanager
from hashlib import sha256
from math import isfinite
import os
from pathlib import Path
from time import monotonic_ns, sleep

from agentic_sim.real_control import JOINTS, pose
from agentic_sim.real_episode import stream_name


GRIPPER_CLOSING_GAP_PCT = 3


def load_profile(path):
    import yaml

    profile = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError("expected a mapping in the local profile")
    arms = profile["arms"]
    cameras = profile["cameras"]
    hz = profile["control_hz"]
    if not 1 <= len(arms) <= 2 or len(cameras) < 2:
        raise ValueError("profile needs one or two arms and at least two cameras")
    if type(hz) not in (int, float) or not isfinite(hz) or hz <= 0:
        raise ValueError("control_hz must be positive and finite")
    for name, pair in arms.items():
        stream_name(name)
        if set(pair) != {"leader", "follower"}:
            raise ValueError(f"{name}: specify leader and follower")
        for spec in pair.values():
            ident = spec["id"]
            if not ident or Path(ident).name != ident or ident in (".", ".."):
                raise ValueError(f"{name}: expected a local calibration ID")
            if not Path(spec["port"]).expanduser().is_absolute():
                raise ValueError(f"{name}: expected an absolute serial device path")
    for name, spec in cameras.items():
        stream_name(name)
        if spec["backend"] not in ("v4l2", "realsense") or not str(spec["device"]).strip():
            raise ValueError(f"{name}: expected a V4L2 path or RealSense serial")
        if spec["backend"] == "v4l2" and not Path(spec["device"]).expanduser().is_absolute():
            raise ValueError(f"{name}: expected an absolute V4L2 device path")
        if spec["backend"] == "v4l2" and spec.get("depth"):
            raise ValueError(f"{name}: depth requires the RealSense backend")
        if any(type(spec[field]) != int or spec[field] <= 0 for field in ("width", "height", "fps")):
            raise ValueError(f"{name}: invalid camera size or FPS")
        if not all(isinstance(spec["calibration"].get(key), str) and spec["calibration"][key].strip()
                   for key in ("id", "version")):
            raise ValueError(f"{name}: camera needs a calibration ID and version")
    return profile


def calibration_path(role, spec):
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    lerobot = Path(os.environ.get("HF_LEROBOT_HOME", hf / "lerobot"))
    root = Path(os.environ.get("HF_LEROBOT_CALIBRATION", lerobot / "calibration"))
    kind = "teleoperators/so_leader" if role == "leader" else "robots/so_follower"
    folder = Path(spec.get("calibration_dir", root / kind)).expanduser()
    return folder / (spec["id"] + ".json")


def calibration_ref(role, spec):
    return {"id": spec["id"], "version": "sha256:" + sha256(calibration_path(role, spec).read_bytes()).hexdigest()}


def check_devices(profile):
    """Validate the local paths without opening serial devices or camera streams."""
    from importlib.util import find_spec

    paths = [Path(spec["port"]).expanduser() for pair in profile["arms"].values() for spec in pair.values()]
    paths += [Path(spec["device"]).expanduser() for spec in profile["cameras"].values() if spec["backend"] == "v4l2"]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("serial and V4L2 devices must be distinct")
    serials = [spec["device"] for spec in profile["cameras"].values() if spec["backend"] == "realsense"]
    if len(set(serials)) != len(serials):
        raise ValueError("RealSense serials must be distinct")
    for path in paths:
        if not path.is_char_device() or not os.access(path, os.R_OK | os.W_OK):
            raise ValueError(f"device missing or inaccessible: {path}")
    dependencies = ["cv2", "lerobot", "scservo_sdk"] + (["pyrealsense2"] if serials else [])
    for module in dependencies:
        if find_spec(module) is None:
            raise RuntimeError(f"install {module} in the hardware environment")
    for pair in profile["arms"].values():
        for role, spec in pair.items():
            calibration_ref(role, spec)


def _device(role, spec):
    # The existing LeRobot hardware env imports processor before concrete devices.
    import lerobot.processor  # noqa: F401

    options = dict(port=str(Path(spec["port"]).expanduser()), id=spec["id"], use_degrees=True,
                   calibration_dir=calibration_path(role, spec).parent)
    if not calibration_path(role, spec).is_file():
        raise FileNotFoundError(calibration_path(role, spec))
    if role == "leader":
        from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
        from lerobot.teleoperators.so_leader.so_leader import SOLeader

        return SOLeader(SOLeaderTeleopConfig(**options))
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    return SOFollower(SOFollowerRobotConfig(**options))


class PositionBus:
    def __init__(self, bus):
        self.bus = bus
        self.last_read = None

    def read(self):
        self.last_read = pose(self.bus.sync_read("Present_Position", num_retry=1))
        return dict(self.last_read)


class Leader(PositionBus):
    """Leader arm. Torque stays off except while it is driven to the follower pose for a hand-over."""

    def __init__(self, bus):
        super().__init__(bus)
        self.driven = False
        self.resolution = bus.model_resolution_table["sts3215"] - 1

    def _torque_states(self):
        return Follower._torque_states(self)

    def drive_on(self):
        """Hold the current pose with torque on: the goal is written and verified before torque enable."""
        if set(self.bus.sync_read("Operating_Mode", normalize=False, num_retry=2).values()) != {0}:
            raise RuntimeError("leader must be in position mode to be driven")
        raw = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
        self.driven = True
        self.bus.sync_write("Goal_Position", raw, normalize=False, num_retry=2)
        goals = {j: self.bus.read("Goal_Position", j, normalize=False, num_retry=2) for j in JOINTS}
        if any(abs(goals[j] - raw[j]) > 2 for j in JOINTS):
            self.release()
            raise RuntimeError("could not verify the leader's stationary goal")
        self.bus.sync_write("Torque_Enable", dict.fromkeys(JOINTS, 1), normalize=False, num_retry=2)
        if self._torque_states() != dict.fromkeys(JOINTS, 1):
            self.release()
            raise RuntimeError("not all leader motors acknowledged torque enable")

    def drive(self, target):
        if not self.driven:
            raise RuntimeError("leader torque is off")
        target = pose(target)
        for joint in JOINTS[:-1]:
            c = self.bus.calibration[joint]
            limit = (c.range_max - c.range_min) * 180 / self.resolution
            target[joint] = max(-limit, min(limit, target[joint]))
        target["gripper"] = max(0.0, min(100.0, target["gripper"]))
        self.bus.sync_write("Goal_Position", target, num_retry=3)

    def read(self):
        self.last_read = pose(self.bus.sync_read("Present_Position", num_retry=3))
        return dict(self.last_read)

    def release(self):
        off = dict.fromkeys(JOINTS, 0)
        for _ in range(6):
            try:
                self.bus.sync_write("Torque_Enable", off, normalize=False, num_retry=3)
                if self.bus.sync_read("Torque_Enable", normalize=False, num_retry=3) == off:
                    self.driven = False
                    return
            except (ConnectionError, RuntimeError, OSError):
                pass
            sleep(.05)
        raise RuntimeError("could not verify leader torque off")


class Follower(PositionBus):
    def __init__(self, bus):
        super().__init__(bus)
        self.started = False
        self.next_health_check = 0
        self.resolution = bus.model_resolution_table["sts3215"] - 1

    def arm(self):
        raw = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
        seated = {}
        for joint in JOINTS:
            calibration = self.bus.calibration[joint]
            seated[joint] = min(calibration.range_max, max(calibration.range_min, raw[joint]))
            if abs(seated[joint] - raw[joint]) > 4:
                raise RuntimeError(f"{joint}: startup position lies outside the calibrated range")
        # Writing a new Goal_Position can energize these motors. Register cleanup first.
        self.started = True
        self.bus.sync_write("Goal_Position", seated, normalize=False, num_retry=2)
        goals = {j: self.bus.read("Goal_Position", j, normalize=False, num_retry=2) for j in JOINTS}
        if goals != seated:
            raise RuntimeError("could not verify the stationary startup target")
        self.bus.sync_write("Torque_Enable", dict.fromkeys(JOINTS, 1), normalize=False, num_retry=2)
        if self._torque_states() != dict.fromkeys(JOINTS, 1):
            raise RuntimeError("not all follower motors acknowledged torque enable")
        self.read()

    def resume(self):
        """Take over a follower left holding with torque on (e.g. after an interrupted session)."""
        if self._torque_states() != dict.fromkeys(JOINTS, 1):
            raise RuntimeError("resume requires torque confirmed on for all six axes")
        self.started = True
        current = self.read()
        self.bus.sync_write("Goal_Position", current, num_retry=2)

    def send(self, target):
        if not self.started:
            raise RuntimeError("follower has not been armed")
        now = monotonic_ns()
        if now >= self.next_health_check:
            if self.bus.read("Torque_Enable", "gripper", normalize=False, num_retry=1) != 1:
                raise RuntimeError("gripper torque was lost")
            self.next_health_check = now + 200_000_000
        target = pose(target)
        for joint in JOINTS[:-1]:
            c = self.bus.calibration[joint]
            limit = (c.range_max - c.range_min) * 180 / self.resolution
            target[joint] = max(-limit, min(limit, target[joint]))
        # Long holds against a 10-point closing error overloaded the lab gripper.
        target["gripper"] = max(target["gripper"], self.last_read["gripper"] - GRIPPER_CLOSING_GAP_PCT)
        self.bus.sync_write("Goal_Position", target, num_retry=1)
        return target

    def _torque_states(self):
        states = {}
        for joint, motor in self.bus.motors.items():
            address, _ = self.bus.model_ctrl_table[motor.model]["Torque_Enable"]
            value, comm, _ = self.bus.packet_handler.read1ByteTxRx(self.bus.port_handler, motor.id, address)
            states[joint] = value if self.bus._is_comm_success(comm) else None
        return states

    def torque_states(self):
        return self._torque_states()

    def close(self):
        if not self.started:
            return
        disabled = dict.fromkeys(JOINTS, 0)
        for _ in range(2):
            self.bus.sync_write("Torque_Enable", disabled, normalize=False, num_retry=2)
            sleep(.1)
            if self._torque_states() == disabled:
                self.started = False
                return
        raise RuntimeError("could not verify torque off for all follower joints")


@contextmanager
def _open_arm(name, role, spec, *, managed=False, resume=False):
    # Keep the LeRobot object alive while using its calibrated motor bus.
    device = _device(role, spec)
    bus = device.bus
    arm = Leader(bus) if role == "leader" else Follower(bus)
    try:
        bus.connect()
        if not bus.is_calibrated:
            raise RuntimeError(f"{name} {role}: calibration disagrees with the device")
        # resume: a follower already holding with torque on, taken over by Follower.resume().
        if not (resume and role == "follower") and any(
                bus.sync_read("Torque_Enable", normalize=False, num_retry=2).values()):
            raise RuntimeError(f"{name} {role}: expected torque off before connecting")
        if role == "follower":
            from lerobot.motors.feetech.feetech import OperatingMode

            if set(bus.sync_read("Operating_Mode", normalize=False, num_retry=2).values()) != {OperatingMode.POSITION.value}:
                raise RuntimeError(f"{name}: follower must already be in position mode")
        try:
            yield arm
        finally:
            if role == "follower" and not managed:
                arm.close()
            if role == "leader" and arm.driven:
                arm.release()  # never leave the leader stiff after the program ends
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def open_leader(name, spec):
    return _open_arm(name, "leader", spec)


def open_follower(name, spec, *, managed=False, resume=False):
    return _open_arm(name, "follower", spec, managed=managed, resume=resume)


def encode_png(image):
    import cv2

    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise OSError("failed to encode a camera frame")
    return data.tobytes()


@contextmanager
def open_camera(spec):
    if spec["backend"] == "v4l2":
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import ColorMode, OpenCVCameraConfig

        camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=Path(spec["device"]).expanduser(),
                              width=spec["width"], height=spec["height"], fps=spec["fps"],
                              fourcc=spec.get("fourcc"), color_mode=ColorMode.BGR))
        try:
            camera.connect()
            yield lambda: (camera.read(), None, None), {}
        finally:
            if camera.is_connected:
                camera.disconnect()
        return

    import numpy as np
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(spec["device"]))
    size = spec["width"], spec["height"]
    config.enable_stream(rs.stream.color, *size, rs.format.bgr8, spec["fps"])
    if spec.get("depth"):
        config.enable_stream(rs.stream.depth, *size, rs.format.z16, spec["fps"])
    active = pipeline.start(config)
    try:
        color_stream = active.get_stream(rs.stream.color)
        info = {"color_intrinsics": _intrinsics(color_stream)}
        if spec.get("depth"):
            depth_stream = active.get_stream(rs.stream.depth)
            transform = depth_stream.get_extrinsics_to(color_stream)
            info["depth"] = {
                "units_m": active.get_device().first_depth_sensor().get_depth_scale(),
                "alignment": "native_depth", "intrinsics": _intrinsics(depth_stream),
                "depth_to_color": {"rotation_column_major": list(transform.rotation),
                                   "translation_m": list(transform.translation)},
            }

        def read():
            frames = pipeline.wait_for_frames(500)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame() if spec.get("depth") else None
            if not color or (spec.get("depth") and not depth):
                raise RuntimeError("incomplete RealSense frameset")
            timing = {"color": _timing(color)}
            if depth is not None:
                timing["depth"] = _timing(depth)
            return (np.asanyarray(color.get_data()).copy(),
                    None if depth is None else np.asanyarray(depth.get_data()).copy(), timing)

        yield read, info
    finally:
        pipeline.stop()


def _intrinsics(stream):
    intr = stream.as_video_stream_profile().get_intrinsics()
    return {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
            "cx": intr.ppx, "cy": intr.ppy, "distortion": str(intr.model), "coefficients": list(intr.coeffs)}


def _timing(frame):
    return {"number": frame.get_frame_number(), "timestamp_ms": frame.get_timestamp(),
            "clock": str(frame.get_frame_timestamp_domain())}
