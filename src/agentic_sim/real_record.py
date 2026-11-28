"""Live SO-101 capture with independent camera workers."""

from contextlib import ExitStack
from threading import Event, Thread
from time import monotonic_ns, sleep

from agentic_sim.real_control import ControlRouter, open_pairs
from agentic_sim.real_episode import EpisodeWriter
from agentic_sim.real_hardware import (
    GRIPPER_CLOSING_GAP_PCT, calibration_ref, encode_png, open_camera, open_follower, open_leader,
)


class CameraWorker:
    def __init__(self, name, spec, read, writer, stop):
        self.name = name
        self.spec = spec
        self.read = read
        self.writer = writer
        self.stop = stop
        self.last_frame = None
        self.error = None
        self.thread = Thread(target=self._run, name=f"camera-{name}")

    def _run(self):
        sequence = None
        try:
            while not self.stop.is_set():
                color, depth, timing = self.read()
                stamp = self.writer.clock()
                if timing is not None:
                    number = timing["color"]["number"]
                    if number == sequence:
                        continue
                    sequence = number
                shape = self.spec["height"], self.spec["width"]
                if color.shape != (*shape, 3) or color.dtype.name != "uint8":
                    raise ValueError("unexpected RGB dimensions or pixel format")
                if depth is not None and (depth.shape != shape or depth.dtype.name != "uint16"):
                    raise ValueError("unexpected native depth dimensions or pixel format")
                self.writer.add_image(self.name, encode_png(color), stamp,
                                      depth=encode_png(depth) if depth is not None else None,
                                      device_timing=timing)
                self.last_frame = stamp
        except Exception as error:
            self.error = error

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"{self.name}: {self.error}") from self.error
        if self.last_frame is None or self.writer.clock() - self.last_frame > 500_000_000:
            raise RuntimeError(f"{self.name}: no recent camera frame")


def record_live(output, profile, *, provenance, layout, stop_requested):
    arms = {name: {role: calibration_ref(role, spec) for role, spec in pair.items()}
            for name, pair in profile["arms"].items()}
    writer = None
    error = None
    try:
        with ExitStack() as stack:
            cameras = {}
            readers = {}
            for name, spec in profile["cameras"].items():
                readers[name], info = stack.enter_context(open_camera(spec))
                cameras[name] = {k: v for k, v in spec.items() if k != "device" and k != "depth"} | info
            control = {"leader_mapping": "relative_clutch", "gripper_closing_gap_pct": GRIPPER_CLOSING_GAP_PCT}
            writer = EpisodeWriter(output, source="real", arms=arms, cameras=cameras, provenance=provenance | control,
                                   hz=profile["control_hz"], layout=layout)
            stopped = Event()
            workers = [CameraWorker(name, spec, readers[name], writer, stopped) for name, spec in cameras.items()]
            started = []
            try:
                for worker in workers:
                    worker.thread.start()
                    started.append(worker)
                deadline = monotonic_ns() + 8_000_000_000
                while any(w.last_frame is None for w in workers) and monotonic_ns() < deadline:
                    if stop_requested():
                        raise RuntimeError("stopped before the first camera sample")
                    if any(w.error is not None for w in workers):
                        break
                    sleep(.02)
                for worker in workers:
                    worker.check()
                if stop_requested():
                    raise RuntimeError("stopped before arming followers")
                with open_pairs(profile["arms"], open_leader, open_follower) as pairs:
                    router = ControlRouter(pairs)
                    interval = 1 / profile["control_hz"]
                    print("RECORDING. q: finish after supporting the followers. Esc/Ctrl-C: interrupt.", flush=True)
                    while not stop_requested():
                        start = monotonic_ns()
                        for worker in workers:
                            worker.check()
                        writer.add_control(router.step(writer.clock))
                        sleep(max(0, interval - (monotonic_ns() - start) / 1e9))
            finally:
                stopped.set()
                for worker in started:
                    worker.thread.join()
            for worker in workers:
                worker.check()
            if not writer.count:
                raise RuntimeError("stopped before the first control sample")
    except BaseException as failure:
        error = failure
        raise
    finally:
        if writer is not None:
            writer.close(error=error)
