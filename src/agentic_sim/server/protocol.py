"""Simulator control protocol: one JSON object per line over TCP on 127.0.0.1. Engine independent.

Request:   {"id": 7, "cmd": "load_experiment", "args": {"name": "minimal_pick_place"}}
Response:  {"id": 7, "ok": true, "result": {...}}
           {"id": 7, "ok": false, "error": {"code": "bad_args", "message": "..."}}

The first request of a connection should be `hello`; its result carries the protocol version and the backend, so a
client can refuse a server it does not understand. Long operations (e.g. run_acceptance) return at once and report
progress through `status`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

PROTOCOL_VERSION = 1
MAX_LINE = 1 << 20  # bytes; longer lines are rejected

# command -> {arg: (type, required)}
COMMANDS = {
    "hello": {"client": (str, False), "protocol_version": (int, False)},
    "status": {},
    "list_experiments": {},
    "load_experiment": {"name": (str, True), "options": (dict, False)},
    "reset": {},
    "pause": {},
    "resume": {},
    "get_state": {},
    "run_acceptance": {"scenario": (str, True)},
    "shutdown": {},
    # task executor (issue #5): orders, answers, state; the executor takes the arm on the first order
    "task_order": {"counts": (dict, True), "mode": (str, False)},
    "task_answer": {"id": (int, True), "choice": (str, True), "color": (str, False), "xy": (list, False)},
    "task_state": {},
    "task_stop": {},
}

# error codes
BAD_REQUEST = "bad_request"          # not JSON / not an object / line too long
UNKNOWN_COMMAND = "unknown_command"
BAD_ARGS = "bad_args"
NOT_SUPPORTED = "not_supported"      # valid command the backend does not implement (yet)
FAILED = "failed"                    # the handler raised


class ProtocolError(Exception):
    def __init__(self, code: str, message: str, req_id=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.req_id = req_id


@dataclass
class Request:
    id: int | str | None
    cmd: str
    args: dict = field(default_factory=dict)


def _check_args(cmd: str, args: dict, req_id):
    spec = COMMANDS[cmd]
    for k in args:
        if k not in spec:
            raise ProtocolError(BAD_ARGS, f"{cmd}: unknown argument {k!r} (allowed: {sorted(spec)})", req_id)
    for k, (typ, required) in spec.items():
        if k not in args:
            if required:
                raise ProtocolError(BAD_ARGS, f"{cmd}: missing argument {k!r}", req_id)
            continue
        val = args[k]
        if not isinstance(val, typ) or (typ is int and isinstance(val, bool)):
            raise ProtocolError(BAD_ARGS, f"{cmd}: argument {k!r} must be {typ.__name__}", req_id)


def encode_request(req_id, cmd: str, **args) -> bytes:
    if cmd not in COMMANDS:
        raise ProtocolError(UNKNOWN_COMMAND, f"unknown command {cmd!r}", req_id)
    _check_args(cmd, args, req_id)
    return (json.dumps({"id": req_id, "cmd": cmd, "args": args}, ensure_ascii=False) + "\n").encode()


def decode_request(line: bytes) -> Request:
    if len(line) > MAX_LINE:
        raise ProtocolError(BAD_REQUEST, f"line longer than {MAX_LINE} bytes")
    try:
        d = json.loads(line)
    except (ValueError, UnicodeDecodeError) as e:
        raise ProtocolError(BAD_REQUEST, f"invalid JSON: {e}") from None
    if not isinstance(d, dict):
        raise ProtocolError(BAD_REQUEST, "request must be a JSON object")
    req_id = d.get("id")
    cmd = d.get("cmd")
    if not isinstance(cmd, str):
        raise ProtocolError(BAD_REQUEST, "missing 'cmd'", req_id)
    if cmd not in COMMANDS:
        raise ProtocolError(UNKNOWN_COMMAND, f"unknown command {cmd!r} (known: {sorted(COMMANDS)})", req_id)
    args = d.get("args") or {}
    if not isinstance(args, dict):
        raise ProtocolError(BAD_ARGS, "'args' must be an object", req_id)
    _check_args(cmd, args, req_id)
    return Request(req_id, cmd, args)


def _plain(o):
    """json.dumps fallback for numpy scalars / arrays (handlers often return simulator values)."""
    if hasattr(o, "tolist"):  # numpy arrays and scalars (bool_, float32, ...)
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def ok(req_id, result=None) -> bytes:
    return (json.dumps({"id": req_id, "ok": True, "result": result if result is not None else {}},
                       ensure_ascii=False, default=_plain) + "\n").encode()


def error(req_id, code: str, message: str) -> bytes:
    return (json.dumps({"id": req_id, "ok": False, "error": {"code": code, "message": message}},
                       ensure_ascii=False) + "\n").encode()


def decode_response(line: bytes) -> dict:
    d = json.loads(line)
    if not isinstance(d, dict) or "ok" not in d:
        raise ProtocolError(BAD_REQUEST, f"malformed response: {line[:200]!r}")
    return d
