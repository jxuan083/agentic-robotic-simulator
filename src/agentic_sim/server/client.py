"""Blocking client for the simulator control server (used by the menu, scripts and agents)."""

from __future__ import annotations

import socket

from agentic_sim.config import load
from agentic_sim.server import protocol as P

SERVER_CONFIG = "configs/sim_server.yaml"


class RemoteError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def server_address(backend: str, config: str = SERVER_CONFIG) -> tuple:
    s = load(config)["servers"][backend]
    return s["host"], int(s["port"])


class SimClient:
    def __init__(self, host: str, port: int, timeout: float = 5.0, name: str = "client"):
        self.host, self.port, self.timeout, self.name = host, port, timeout, name
        self._sock = None
        self._buf = b""
        self._next_id = 0
        self.server_info: dict = {}

    @classmethod
    def for_backend(cls, backend: str, **kw) -> "SimClient":
        host, port = server_address(backend)
        return cls(host, port, **kw)

    def connect(self) -> dict:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.server_info = self.call("hello", client=self.name, protocol_version=P.PROTOCOL_VERSION)
        if self.server_info.get("protocol_version") != P.PROTOCOL_VERSION:
            self.close()
            raise RemoteError(P.BAD_REQUEST, f"server speaks protocol {self.server_info.get('protocol_version')}, "
                                             f"client {P.PROTOCOL_VERSION}")
        return self.server_info

    def call(self, cmd: str, **args):
        """Send one request and wait for its response; raise RemoteError when the server answers ok=false."""
        if self._sock is None:
            raise ConnectionError("not connected")
        self._next_id += 1
        rid = self._next_id
        self._sock.sendall(P.encode_request(rid, cmd, **args))
        while True:
            while b"\n" not in self._buf:
                chunk = self._sock.recv(65536)
                if not chunk:
                    self.close()
                    raise ConnectionError("server closed the connection")
                self._buf += chunk
            line, self._buf = self._buf.split(b"\n", 1)
            resp = P.decode_response(line)
            if resp.get("id") != rid:
                continue  # stale response (e.g. after a client-side timeout); skip it
            if resp["ok"]:
                return resp["result"]
            err = resp.get("error") or {}
            raise RemoteError(err.get("code", "?"), err.get("message", ""))

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self):
        if self._sock is None:
            self.connect()
        return self

    def __exit__(self, *exc):
        self.close()


def probe(backend: str, timeout: float = 0.5) -> dict | None:
    """Status of the backend's server, or None if nothing is listening."""
    try:
        with SimClient.for_backend(backend, timeout=timeout, name="probe") as c:
            info = dict(c.server_info)
            try:
                info["status"] = c.call("status")
            except RemoteError as e:
                info["status_error"] = str(e)
            return info
    except (OSError, ConnectionError, RemoteError, ValueError):
        return None
