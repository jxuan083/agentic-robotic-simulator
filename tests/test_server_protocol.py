"""Protocol encode/decode and a real localhost round trip through SimServer.poll() + SimClient."""
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from agentic_sim.server import protocol as P  # noqa: E402
from agentic_sim.server.client import RemoteError, SimClient  # noqa: E402
from agentic_sim.server.core import SimServer  # noqa: E402


def test_encode_decode_roundtrip():
    r = P.decode_request(P.encode_request(3, "load_experiment", name="minimal_pick_place", options={"focus": True}))
    assert (r.id, r.cmd, r.args) == (3, "load_experiment", {"name": "minimal_pick_place", "options": {"focus": True}})


@pytest.mark.parametrize("line,code", [
    (b"not json", P.BAD_REQUEST),
    (b"[1, 2]", P.BAD_REQUEST),
    (b'{"id": 1}', P.BAD_REQUEST),
    (b'{"id": 1, "cmd": "fly"}', P.UNKNOWN_COMMAND),
    (b'{"id": 1, "cmd": "load_experiment"}', P.BAD_ARGS),
    (b'{"id": 1, "cmd": "load_experiment", "args": {"name": 5}}', P.BAD_ARGS),
    (b'{"id": 1, "cmd": "reset", "args": {"hard": true}}', P.BAD_ARGS),
    (b'{"id": 1, "cmd": "hello", "args": {"protocol_version": true}}', P.BAD_ARGS),
])
def test_decode_rejects(line, code):
    with pytest.raises(P.ProtocolError) as ei:
        P.decode_request(line)
    assert ei.value.code == code


class FakeSim:
    """Stands in for a simulator main loop: handlers run on the polling thread."""

    def __init__(self):
        self.experiment = None
        self.calls = []

    def cmd_status(self):
        return {"experiment": self.experiment, "thread": threading.current_thread().name}

    def cmd_load_experiment(self, name, options=None):
        self.calls.append(("load", name, options))
        self.experiment = name
        return {"loaded": name}

    def cmd_reset(self):
        raise RuntimeError("boom")


@pytest.fixture
def server():
    sim = FakeSim()
    srv = SimServer(sim, port=0, backend="fake", log=lambda *a: None)
    stop = threading.Event()

    def loop():  # the "main loop": poll once per frame
        while not stop.is_set() and not srv.stop_requested:
            srv.poll(timeout=0.01)

    t = threading.Thread(target=loop, name="sim-main")
    t.start()
    yield srv, sim
    stop.set()
    t.join(2)
    srv.close()


def test_round_trip_and_errors(server):
    srv, sim = server
    host, port = srv.address
    with SimClient(host, port, timeout=2) as c:
        assert c.server_info == {"protocol_version": P.PROTOCOL_VERSION, "backend": "fake"}
        assert c.call("load_experiment", name="m", options={"focus": True}) == {"loaded": "m"}
        st = c.call("status")
        assert st == {"experiment": "m", "thread": "sim-main"}  # handlers ran on the main-loop thread
        with pytest.raises(RemoteError) as ei:
            c.call("reset")
        assert ei.value.code == P.FAILED and "boom" in ei.value.message
        with pytest.raises(RemoteError) as ei:
            c.call("pause")
        assert ei.value.code == P.NOT_SUPPORTED
        assert c.call("status")["experiment"] == "m"  # still serving after errors
    # the server outlives the client: a new client sees the same state
    with SimClient(host, port, timeout=2) as c2:
        assert c2.call("status")["experiment"] == "m"
        assert c2.call("shutdown") == {"stopping": True}
    assert sim.calls == [("load", "m", {"focus": True})]


def test_raw_bad_lines_get_error_responses(server):
    import json
    import socket
    srv, _ = server
    s = socket.create_connection(srv.address, timeout=2)
    s.sendall(b"garbage\n" + P.encode_request(9, "status"))
    data = b""
    while data.count(b"\n") < 2:
        data += s.recv(4096)
    first, second = [json.loads(x) for x in data.split(b"\n")[:2]]
    assert first["ok"] is False and first["error"]["code"] == P.BAD_REQUEST
    assert second["ok"] is True and second["id"] == 9
    s.close()


def test_localhost_only():
    with pytest.raises(ValueError):
        SimServer(FakeSim(), host="0.0.0.0", port=0)
