"""Capture and follower-command replay using the same managed motion lifecycle."""

from contextlib import ExitStack, contextmanager
from importlib.metadata import version
import json
from pathlib import Path
from threading import Event
from time import monotonic_ns, sleep

from agentic_sim.real_control import ControlRouter
from agentic_sim.real_episode import EpisodeWriter, write_json
from agentic_sim.real_motion import Motion, TRACKING, near


class LiveBackend:
    source = "real"
    clock = staticmethod(monotonic_ns)
    wait = staticmethod(sleep)

    def __init__(self, profile, arm, provenance):
        from agentic_sim.real_hardware import calibration_ref

        self.arm = arm
        self.pair = profile["arms"][arm]
        self.calibrations = {role: calibration_ref(role, spec) for role, spec in self.pair.items()}
        self.cameras = profile["cameras"]
        self.hz = profile["control_hz"]
        self.provenance = provenance | {"lerobot": version("lerobot")}

    @contextmanager
    def devices(self, *, leader):
        from agentic_sim.real_hardware import open_follower, open_leader

        with ExitStack() as stack:
            hand = stack.enter_context(open_leader(self.arm, self.pair["leader"])) if leader else None
            yield hand, stack.enter_context(open_follower(self.arm, self.pair["follower"], managed=True))

    @contextmanager
    def capture(self, root, plan):
        from agentic_sim.real_hardware import open_camera
        from agentic_sim.real_record import CameraWorker

        with ExitStack() as stack:
            streams = {}
            cameras = {}
            for name, spec in self.cameras.items():
                streams[name], info = stack.enter_context(open_camera(spec))
                cameras[name] = {k: v for k, v in spec.items() if k not in ("device", "depth")} | info
            writer = EpisodeWriter(root, source=self.source, arms={self.arm: self.calibrations},
                                   cameras=cameras, provenance=self.provenance, hz=self.hz, layout=plan.layout)
            stopped = Event()
            workers = [CameraWorker(name, spec, streams[name], writer, stopped) for name, spec in cameras.items()]
            started = []
            error = None

            def check():
                for worker in workers:
                    worker.check()

            try:
                for worker in workers:
                    worker.thread.start()
                    started.append(worker)
                deadline = self.clock() + 8_000_000_000
                while any(w.last_frame is None and w.error is None for w in workers) and self.clock() < deadline:
                    self.wait(.02)
                check()
                yield writer, check, lambda _row: None
                check()
            except BaseException as failure:
                error = failure
                raise
            finally:
                stopped.set()
                for worker in started:
                    worker.thread.join()
                writer.close(error=error)


def validate_setup(backend, plan):
    if plan.arm != backend.arm or plan.calibrations != backend.calibrations:
        raise ValueError("motion setup arm or calibration differs from the connected profile")


@contextmanager
def managed_motion(backend, plan, path, operator, *, leader, episode_clock=None):
    validate_setup(backend, plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    origin = backend.clock()
    with path.open("x", encoding="utf-8", buffering=1) as log:
        def emit(event):
            log.write(json.dumps(event, allow_nan=False) + "\n")

        with backend.devices(leader=leader) as (hand, device):
            clock = episode_clock if episode_clock is not None else lambda: backend.clock() - origin
            motion = Motion(device, plan, emit, clock=clock,
                            wait=backend.wait, halted=operator.halted)
            try:
                operator.status("align")
                start = motion.align()
                yield hand, motion, start
            except BaseException as error:
                motion.fault(error)
                raise


def capture_round(root, backend, plan, operator, *, policy=None, card=None):
    """One demonstrator or DAgger round; only task commands are in control.jsonl."""
    validate_setup(backend, plan)
    root = Path(root)
    with backend.capture(root, plan) as (writer, check_images, save_images):
        writer.manifest.update(motion_setup=plan.reference(), mode="dagger" if policy else "demonstration",
                               task_card=card)
        with managed_motion(backend, plan, root / "motion.jsonl", operator, leader=True,
                            episode_clock=writer.clock) as (leader, motion, start):
            if not policy and hasattr(leader, "drive_on"):
                # Bring the leader to the follower's start pose and hand it over before recording starts.
                from agentic_sim.real_leader import handover

                handover(leader, start, route=list(plan.route), clock=backend.clock, wait=backend.wait,
                         hz=backend.hz, say=getattr(operator, "say", lambda _text: None), halted=operator.halted,
                         event=lambda kind, **fields: motion.event(kind, **fields))
            router = ControlRouter({plan.arm: (leader, motion)}, start="policy" if policy else "manual")
            if policy:
                router.enqueue_policy([{plan.arm: start}], generation=router.generation)
            operator.status("recording")
            try:
                while not operator.finished(writer.count):
                    began = backend.clock()
                    check_images()
                    operator.tick(leader, writer.count)
                    if router.source == "policy":
                        if operator.takeover(writer.count):
                            router.take_over(writer.clock())
                            motion.event("takeover", torque_preserved=True)
                        else:
                            chunk = policy(writer.count)
                            if chunk:
                                router.enqueue_policy(chunk, generation=router.generation)
                    row = router.step(writer.clock)
                    writer.add_control(row)
                    save_images(row)
                    backend.wait(max(0, 1 / backend.hz - (backend.clock() - began) / 1e9))
            finally:
                discarded = router.stop()
            motion.event("task_ended", frames=writer.count, discarded_policy_actions=discarded)
            destination = (card or {}).get("destination")
            if (destination in (plan.drops or {}) and writer.count and not getattr(operator, "cancelled", False)
                    and operator.carry(destination)):
                # The block is lifted: the taught drop route places it, outside the training segment.
                operator.status("place")
                motion.place(destination)
                writer.manifest["placement"] = {"tray": destination, "mode": "taught_drop_route"}
            operator.status("return")
            result = {}

            def empty():
                result.update(operator.finish())
                return result.get("empty") is True

            motion.park(empty, operator.support)
            if not writer.count:
                result["cancelled"] = True
            writer.manifest["operator_result"] = result
    return result


def replay_steps(episode, backend, plan):
    validate_setup(backend, plan)
    manifest = episode.manifest
    if manifest["source"] != backend.source:
        raise ValueError(f"{backend.source} replay requires a {backend.source} capture")
    if plan.arm not in manifest["arms"] or manifest["arms"][plan.arm]["follower"] != plan.calibrations["follower"]:
        raise ValueError("selected follower or calibration differs from the episode")
    if manifest.get("motion_setup") != plan.reference() or manifest.get("layout_id") != plan.layout:
        raise ValueError("episode start setup differs; handle legacy or other-layout recordings separately")
    if not episode.validate()["ok"]:
        raise ValueError("replay requires a complete, valid episode")
    steps = [row["arms"][plan.arm] for row in episode.rows]
    if not near(steps[0]["readback"], plan.start) or not near(steps[0]["command"], plan.start):
        raise ValueError("recorded first frame differs from the standard start")
    for step in steps:
        plan.in_workspace(step["command"])
    if any(b["command_ns"] - a["command_ns"] < 1_000_000 for a, b in zip(steps, steps[1:])):
        raise ValueError("recorded command intervals are too short")
    return steps


def replay_round(episode, backend, plan, root, operator):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    result = {"source_episode_id": episode.manifest["episode_id"], "arm": plan.arm,
              "source": backend.source, "frames_sent": 0, "status": "running"}
    write_json(root / "result.json", result)
    try:
        steps = replay_steps(episode, backend, plan)
        with managed_motion(backend, plan, root / "motion.jsonl", operator, leader=False) as (_, motion, start):
            if not near(start, steps[0]["readback"]):
                raise RuntimeError("aligned follower differs from recorded first-frame readback")
            operator.status("replay")
            replay_start = backend.clock()
            for i, step in enumerate(steps):
                due = replay_start + step["command_ns"] - steps[0]["command_ns"]
                while (remaining := due - backend.clock()) > 0 and not operator.finished(i):
                    motion.ensure_running()
                    backend.wait(min(remaining / 1e9, .05))
                if operator.finished(i):
                    break
                if backend.clock() - due > 100_000_000:
                    raise RuntimeError(f"replay could not keep recorded timing at frame {i}")
                actual = motion.read()
                sent = motion.send(step["command"])
                after = motion.read()
                motion.event("replay_frame", frame=i, recorded_command_ns=step["command_ns"],
                             command=sent, readback=after, before=actual)
                result["frames_sent"] += 1
                if not near(sent, step["command"]):
                    raise RuntimeError(f"replay target was clipped at frame {i}")
                if not near(after, sent, TRACKING):
                    raise RuntimeError(f"replay tracking error at frame {i}")
            complete = result["frames_sent"] == len(steps)
            destination = (episode.manifest.get("task_card") or {}).get("destination")
            if complete and destination in (plan.drops or {}) and operator.carry(destination):
                operator.status("place")
                motion.place(destination)
                result["placement"] = {"tray": destination, "mode": "taught_drop_route"}
            operator.status("return")
            outcome = {}

            def empty():
                outcome.update(operator.finish())
                return outcome.get("empty") is True

            motion.park(empty, operator.support)
            result.update(status="complete" if complete else "cancelled", outcome=outcome)
    except BaseException as error:
        result.update(status="interrupted", error=str(error))
        raise
    finally:
        write_json(root / "result.json", result)
    return result
