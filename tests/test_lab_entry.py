"""PR #12 review 3: tools/lab.py stops only the services it created (recorded before they start), reuses services
that exist even while they still warm up, and names the component that failed. Pure Python: the control scripts,
docker and the readiness probes are replaced by fakes."""
import os
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import lab  # noqa: E402


class World:
    """Fake services: which exist, which answer, and every control-script call."""
    def __init__(self, model="none", sim="none", up_hangs=()):
        self.model, self.sim = model, sim            # none | starting | ready
        self.up_hangs = set(up_hangs)
        self.calls = []

    def run(self, cmd, timeout=None):
        script, verb = cmd[0], cmd[-1]
        self.calls.append((script, verb))
        kind = "model" if "serve.sh" in script else "sim"
        if verb == "up":
            if kind in self.up_hangs:
                setattr(self, kind, "starting")          # created, never ready
                raise subprocess.TimeoutExpired(cmd, timeout)
            setattr(self, kind, "ready")
        elif verb == "down":
            setattr(self, kind, "none")
        return types.SimpleNamespace(returncode=0)


@pytest.fixture
def fake(monkeypatch):
    def make(**kw):
        w = World(**kw)
        monkeypatch.setattr(lab, "container_exists", lambda name: w.model != "none")
        monkeypatch.setattr(lab, "model_up", lambda url: w.model == "ready")
        monkeypatch.setattr(lab, "sim_exists", lambda backend: w.sim != "none")
        monkeypatch.setattr(lab, "probe", lambda backend, timeout=1.0: {"ok": 1} if w.sim == "ready" else None)
        monkeypatch.setattr(lab.Lab, "run", lambda self, cmd, timeout=None: w.run(cmd, timeout))
        monkeypatch.setattr(lab.time, "sleep", lambda s: None)

        class Client:
            def __init__(self, *a, **k):
                pass

            def connect(self):
                pass

            def call(self, *a, **k):
                w.calls.append(("load_experiment", "call"))

            def close(self):
                pass
        monkeypatch.setattr(lab.SimClient, "for_backend", staticmethod(lambda *a, **k: Client()))
        return w
    return make


def _args(**kw):
    d = dict(backend="mujoco", model="q4", experiment="lab_20261120", blocks=None, block_colors=None,
             wrist_lens=None, no_chat=False, start_timeout=None, type=None, type_reopen=None)
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_model_still_warming_up_is_reused_and_kept(fake, monkeypatch):
    w = fake(model="starting")
    ticks = iter([False, False, True])
    monkeypatch.setattr(lab, "model_up", lambda url: w.model == "ready" or next(ticks, True))
    L = lab.Lab(_args())
    L.model()
    L.simulator()
    L.stop()
    assert ("docker/llamacpp/serve.sh", "up") not in w.calls          # not started again
    assert ("docker/llamacpp/serve.sh", "down") not in w.calls        # not ours: kept
    assert ("tools/mujoco_server.sh", "down") in w.calls               # the simulator was ours


def test_new_service_that_times_out_is_cleaned_and_named(fake, capsys):
    w = fake(up_hangs={"model"})
    L = lab.Lab(_args(start_timeout=2))
    with pytest.raises(TimeoutError, match="模型服務 q4"):
        L.model()
    L.stop()
    assert w.calls == [("docker/llamacpp/serve.sh", "up"), ("docker/llamacpp/serve.sh", "down")]
    assert w.model == "none"                                           # the next start can create it again
    L2 = lab.Lab(_args())
    w.up_hangs.clear()
    L2.model()
    assert w.model == "ready"


def test_existing_ready_model_kept_new_simulator_stopped(fake):
    w = fake(model="ready")
    L = lab.Lab(_args())
    L.model()
    L.simulator()
    L.stop()
    assert w.model == "ready" and w.sim == "none"
    assert [c for c in w.calls if c[1] in ("up", "down")] == [("tools/mujoco_server.sh", "up"),
                                                             ("tools/mujoco_server.sh", "down")]


def test_main_returns_failure_and_cleans_up(fake, monkeypatch):
    w = fake(up_hangs={"sim"})
    rc = lab.main(["--backend", "mujoco", "--no-chat", "--start-timeout", "1"])
    assert rc == 1 and w.sim == "none" and ("tools/mujoco_server.sh", "down") in w.calls
