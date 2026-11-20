"""PR #12 reviews 3 and 2-1: tools/lab.py stops only the instance it created (the container ID / PID the control
script wrote to ARS_CREATED_FILE), reuses services that exist even while they still warm up or were created by
another entry in between, and names the component that failed.

The first group replaces the control scripts with an in-process fake. The second group runs the real
docker/llamacpp/serve.sh with stub `docker`, `curl` and `sha256sum` commands on PATH (no containers, no network): the
stub can hold a `docker run` at a gate, so two entries interleave in a fixed order instead of by timing."""
import os
import subprocess
import sys
import threading
import time
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import lab  # noqa: E402


def _args(**kw):
    d = dict(backend="mujoco", model="q4", experiment="lab_20261120", blocks=None, block_colors=None,
             wrist_lens=None, no_chat=False, start_timeout=None, type=None, type_reopen=None)
    d.update(kw)
    return types.SimpleNamespace(**d)


def _no_load(monkeypatch, calls):
    class Client:
        def __init__(self, *a, **k):
            pass

        def connect(self):
            pass

        def call(self, *a, **k):
            calls.append(("load_experiment", "call"))

        def close(self):
            pass
    monkeypatch.setattr(lab.SimClient, "for_backend", staticmethod(lambda *a, **k: Client()))


# ---- fake control scripts ----------------------------------------------------------------------------------------
class World:
    """Fake services: state and instance ID of each, and every control-script call."""
    def __init__(self, model="none", sim="none", up_hangs=()):
        self.state = {"model": model, "sim": sim}               # none | starting | ready
        self.ident = {k: (f"{k}-pre" if v != "none" else None) for k, v in self.state.items()}
        self.up_hangs = set(up_hangs)
        self.calls = []
        self.n = 0

    def run(self, cmd, timeout=None, env=None):
        script, verb = cmd[0], cmd[1]
        self.calls.append((script, verb))
        kind = "model" if "serve.sh" in script else "sim"
        if verb == "up":
            if self.state[kind] != "none":
                return types.SimpleNamespace(returncode=0)        # "already running": nothing written
            self.n += 1
            self.ident[kind] = f"{kind}-{self.n}"
            with open(env["ARS_CREATED_FILE"], "w") as f:
                f.write(self.ident[kind])
            if kind in self.up_hangs:
                self.state[kind] = "starting"                     # created, never ready
                raise subprocess.TimeoutExpired(cmd, timeout)
            self.state[kind] = "ready"
        elif verb == "down":
            if cmd[2] == self.ident[kind]:
                self.state[kind], self.ident[kind] = "none", None
        return types.SimpleNamespace(returncode=0)


@pytest.fixture
def fake(monkeypatch):
    def make(**kw):
        w = World(**kw)
        monkeypatch.setattr(lab, "script_info", lambda script, env: {
            "name": "ars-llm-q4", "url": "http://127.0.0.1:8001/v1", "model": "Qwen3.5-9B-Q4_K_M"})
        monkeypatch.setattr(lab, "model_up", lambda url: w.state["model"] == "ready")
        monkeypatch.setattr(lab, "probe", lambda backend, timeout=1.0: {"ok": 1} if w.state["sim"] == "ready" else None)
        monkeypatch.setattr(lab.Lab, "run", lambda self, cmd, timeout=None, env=None: w.run(cmd, timeout, env))
        monkeypatch.setattr(lab.time, "sleep", lambda s: None)
        _no_load(monkeypatch, w.calls)
        return w
    return make


def test_model_still_warming_up_is_reused_and_kept(fake, monkeypatch):
    w = fake(model="starting")
    ticks = iter([False, False, True])
    monkeypatch.setattr(lab, "model_up", lambda url: w.state["model"] == "ready" or next(ticks, True))
    L = lab.Lab(_args())
    L.model()
    L.simulator()
    L.stop()
    assert ("docker/llamacpp/serve.sh", "down") not in w.calls        # not ours: kept
    assert w.ident["model"] == "model-pre" and w.state["model"] != "none"
    assert ("tools/mujoco_server.sh", "down") in w.calls               # the simulator was ours
    assert w.state["sim"] == "none"


def test_new_service_that_times_out_is_cleaned_and_named(fake, capsys):
    w = fake(up_hangs={"model"})
    L = lab.Lab(_args(start_timeout=2))
    with pytest.raises(TimeoutError, match="模型服務 q4"):
        L.model()
    L.stop()
    assert w.calls == [("docker/llamacpp/serve.sh", "up"), ("docker/llamacpp/serve.sh", "down")]
    assert w.state["model"] == "none"                                  # the next start can create it again
    L2 = lab.Lab(_args())
    w.up_hangs.clear()
    L2.model()
    assert w.state["model"] == "ready"


def test_existing_ready_model_kept_new_simulator_stopped(fake):
    w = fake(model="ready")
    L = lab.Lab(_args())
    L.model()
    L.simulator()
    L.stop()
    assert w.state["model"] == "ready" and w.state["sim"] == "none"
    assert [c for c in w.calls if c[1] in ("up", "down")] == [("tools/mujoco_server.sh", "up"),
                                                             ("tools/mujoco_server.sh", "down")]


def test_down_names_the_created_instance(fake):
    w = fake()
    L = lab.Lab(_args())
    L.simulator()
    assert L.owned == [("mujoco 模擬伺服器", ["tools/mujoco_server.sh", "down", "sim-1"])]


def test_main_returns_failure_and_cleans_up(fake, monkeypatch):
    w = fake(up_hangs={"sim"})
    rc = lab.main(["--backend", "mujoco", "--no-chat", "--start-timeout", "1"])
    assert rc == 1 and w.state["sim"] == "none" and ("tools/mujoco_server.sh", "down") in w.calls


# ---- the real serve.sh with stub docker / curl -----------------------------------------------------------------
DOCKER = r'''#!/usr/bin/env bash
# stub docker: containers are "<name> <id>" lines in $FAKE_DOCKER_STATE
S="$FAKE_DOCKER_STATE"; touch "$S"
case "$1" in
  ps) cut -d' ' -f1 "$S" ;;
  run)
    shift; name=; cid=
    while [ $# -gt 0 ]; do case "$1" in --name) name=$2; shift ;; --cidfile) cid=$2; shift ;; esac; shift; done
    if [ -n "${FAKE_DOCKER_GATE:-}" ]; then          # hold this run until the test opens the gate
      touch "$FAKE_DOCKER_GATE.reached"; while [ -e "$FAKE_DOCKER_GATE" ]; do sleep 0.02; done; fi
    if grep -q "^$name " "$S"; then echo "Conflict. The container name \"/$name\" is already in use" >&2; exit 125; fi
    id=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
    echo "$name $id" >> "$S"
    [ -z "$cid" ] || printf '%s\n' "$id" > "$cid"
    echo "$id" ;;
  stop)
    grep -q -e "^$2 " -e " $2\$" "$S" || exit 1
    grep -v -e "^$2 " -e " $2\$" "$S" > "$S.t" || true; mv "$S.t" "$S"; echo "$2" ;;
  *) exit 0 ;;
esac
'''
CURL = '''#!/usr/bin/env bash
[ -e "$FAKE_READY" ] && [ -s "$FAKE_DOCKER_STATE" ]
'''


@pytest.fixture
def stub(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in (("docker", DOCKER), ("curl", CURL), ("sha256sum", "#!/usr/bin/env bash\ncat > /dev/null\n")):
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
    models = tmp_path / "models"
    models.mkdir()
    (models / "Qwen3.5-9B-Q4_K_M.gguf").write_text("stub")
    st = types.SimpleNamespace(state=tmp_path / "containers", ready=tmp_path / "ready", tmp=tmp_path)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(st.state))
    monkeypatch.setenv("FAKE_READY", str(st.ready))
    monkeypatch.setenv("LLM_MODEL_DIR", str(models))
    monkeypatch.setattr(lab, "model_up", lambda url: st.ready.exists() and st.state.exists() and st.state.read_text() != "")
    st.containers = lambda: [tuple(l.split()) for l in st.state.read_text().splitlines()] if st.state.exists() else []
    return st


def test_two_entries_interleaved_the_second_reuses_the_first_ones_container(stub, capfd):
    stub.ready.touch()                                   # the service answers as soon as its container exists
    gate = stub.tmp / "gate"
    gate.touch()
    A, B = lab.Lab(_args()), lab.Lab(_args())
    A.env["FAKE_DOCKER_GATE"] = str(gate)
    errors = []

    def run_a():
        try:
            A.model()
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    t = threading.Thread(target=run_a)
    t.start()
    for _ in range(500):                                 # A found no container and is held right before creating
        if (stub.tmp / "gate.reached").exists():
            break
        time.sleep(0.01)
    assert (stub.tmp / "gate.reached").exists() and stub.containers() == []
    B.model()                                            # B creates the container
    [(name, cid)] = stub.containers()
    os.unlink(gate)                                      # A resumes: loses the name race, reuses B's container
    t.join(10)
    assert not errors and A.owned == [] and B.owned[0][1][-1] == cid
    assert "already running (started by another command)" in capfd.readouterr().out   # the name race path
    A.stop()                                             # A exits first: the container stays
    assert stub.containers() == [(name, cid)] and lab.model_up("")
    B.stop()                                             # B exits: now it is stopped
    assert stub.containers() == []


@pytest.mark.parametrize("ready_later", [True, False])
def test_custom_container_name_still_warming_up_is_kept(stub, monkeypatch, ready_later):
    monkeypatch.setenv("LLM_CONTAINER_NAME", "team-q4")
    stub.state.write_text("team-q4 pre0001\n")           # started earlier with the same environment; HTTP not ready
    L = lab.Lab(_args(start_timeout=3))
    assert lab.script_info(lab.MODELS["q4"][0], L.env)["name"] == "team-q4"
    if ready_later:
        threading.Timer(0.5, stub.ready.touch).start()
        L.model()
    else:
        with pytest.raises(TimeoutError, match="模型服務 q4"):
            L.model()
    L.stop()
    assert L.owned == [] and stub.containers() == [("team-q4", "pre0001")]


def test_created_then_timed_out_is_cleaned_and_restarts(stub):
    L = lab.Lab(_args(start_timeout=1.5))                # never ready: serve.sh up waits, lab gives up
    with pytest.raises(TimeoutError, match="模型服務 q4"):
        L.model()
    [(name, cid)] = stub.containers()
    assert L.owned[0][1][-2:] == ["down", cid]
    L.stop()
    assert stub.containers() == []
    stub.ready.touch()
    L2 = lab.Lab(_args())
    L2.model()                                           # the next start creates it again
    assert len(stub.containers()) == 1 and stub.containers()[0][1] != cid
    L2.stop()
    assert stub.containers() == []
