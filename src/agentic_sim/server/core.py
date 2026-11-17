"""Non-blocking control server, polled from the simulator's own main loop.

Simulators (Kit in particular) want their API called from the main thread, so the server never starts a thread:
the main loop calls `poll()` once per frame, and handlers run right there between frames. A handler is any object
with `cmd_<name>(**args)` methods; a missing method answers `not_supported`, an exception answers `failed`.
"""

from __future__ import annotations

import selectors
import socket
import traceback

from agentic_sim.server import protocol as P


class SimServer:
    def __init__(self, handler, host: str = "127.0.0.1", port: int = 0, backend: str = "unknown", log=print):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the control server listens on localhost only")
        self.handler = handler
        self.backend = backend
        self._log = log
        self._sel = selectors.DefaultSelector()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(8)
        self._sock.setblocking(False)
        self._sel.register(self._sock, selectors.EVENT_READ, None)
        self._bufs: dict = {}
        self.address = self._sock.getsockname()[:2]
        self.stop_requested = False

    @property
    def clients(self) -> int:
        return len(self._bufs)

    def poll(self, timeout: float = 0.0, max_requests: int = 32) -> int:
        """Accept connections and serve pending requests; returns how many requests were handled."""
        handled = 0
        for key, _ in self._sel.select(timeout):
            if key.data is None:
                conn, _ = self._sock.accept()
                conn.setblocking(False)
                self._sel.register(conn, selectors.EVENT_READ, "client")
                self._bufs[conn] = b""
                continue
            conn = key.fileobj
            try:
                data = conn.recv(65536)
            except (ConnectionError, OSError):
                data = b""
            if not data:
                self._drop(conn)
                continue
            buf = self._bufs[conn] + data
            while b"\n" in buf and handled < max_requests:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    self._send(conn, self._handle(line))
                    handled += 1
            if len(buf) > P.MAX_LINE:
                self._send(conn, P.error(None, P.BAD_REQUEST, f"line longer than {P.MAX_LINE} bytes"))
                buf = b""
            if conn in self._bufs:
                self._bufs[conn] = buf
        return handled

    def _handle(self, line: bytes) -> bytes:
        try:
            req = P.decode_request(line)
        except P.ProtocolError as e:
            return P.error(e.req_id, e.code, e.message)
        if req.cmd == "hello":
            return P.ok(req.id, {"protocol_version": P.PROTOCOL_VERSION, "backend": self.backend})
        fn = getattr(self.handler, f"cmd_{req.cmd}", None)
        if fn is None and req.cmd == "shutdown":
            self.stop_requested = True  # built in: the owner's main loop sees stop_requested and exits
            return P.ok(req.id, {"stopping": True})
        if fn is None:
            return P.error(req.id, P.NOT_SUPPORTED, f"{self.backend} server does not support {req.cmd!r} yet")
        try:
            result = fn(**req.args)
        except Exception as e:  # report to the client, keep serving
            self._log(f"[server] {req.cmd} failed:\n{traceback.format_exc()}")
            return P.error(req.id, P.FAILED, f"{type(e).__name__}: {e}")
        if req.cmd == "shutdown":
            self.stop_requested = True
        return P.ok(req.id, result)

    def _send(self, conn, data: bytes):
        try:
            conn.setblocking(True)  # responses are small; a short blocking send keeps ordering simple
            conn.settimeout(1.0)
            conn.sendall(data)
            conn.setblocking(False)
        except OSError:
            self._drop(conn)

    def _drop(self, conn):
        if conn in self._bufs:
            del self._bufs[conn]
            try:
                self._sel.unregister(conn)
            except (KeyError, ValueError):
                pass
            conn.close()

    def close(self):
        for conn in list(self._bufs):
            self._drop(conn)
        try:
            self._sel.unregister(self._sock)
        except (KeyError, ValueError):
            pass
        self._sock.close()
        self._sel.close()
