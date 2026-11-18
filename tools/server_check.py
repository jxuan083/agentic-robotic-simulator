#!/usr/bin/env python3
"""Check a running simulator server (issue #3, P2): connection, commands, reconnect, repeated experiment loads.

  uv run tools/server_check.py [--backend isaac] [--loads 10] [--out outputs/server_check.json]

Writes one JSON with every timing and memory reading; exits 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agentic_sim.server.client import RemoteError, SimClient  # noqa: E402


def wait_frames(c, n=30, timeout=10.0):
    f0 = c.call("status").get("frames", 0)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = c.call("status")
        if st.get("frames", 0) >= f0 + n:
            return st
        time.sleep(0.05)
    raise TimeoutError("session did not advance")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="isaac")
    ap.add_argument("--experiment", default="minimal_pick_place")
    ap.add_argument("--loads", type=int, default=10)
    ap.add_argument("--out", default="outputs/server_check.json")
    a = ap.parse_args()
    checks, report = [], {"backend": a.backend, "experiment": a.experiment}

    def check(label, ok, detail=None):
        checks.append({"label": label, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}: {label}" + (f"  ({detail})" if detail is not None else ""), flush=True)

    t0 = time.monotonic()
    c = SimClient.for_backend(a.backend, timeout=60.0, name="server_check")
    info = c.connect()
    report["connect_s"] = round(time.monotonic() - t0, 3)
    check("hello: protocol and backend", info.get("backend") == a.backend, info)

    t0 = time.monotonic()
    r = c.call("load_experiment", name=a.experiment, options={})
    report["first_load_s"] = round(time.monotonic() - t0, 3)
    wait_frames(c)
    st = c.call("get_state")
    check("get_state after load: session running", st.get("frames", 0) > 0 and "ee" in st, st.get("frames"))

    # reset / pause / resume act at the next frame like the panel buttons
    s0 = st["session"]
    c.call("reset")
    wait_frames(c, 5)
    check("reset starts a new session", c.call("get_state")["session"] == s0 + 1)
    c.call("pause")
    wait_frames(c, 5)
    st1 = c.call("get_state")
    wait_frames(c, 20)
    st2 = c.call("get_state")
    check("pause: paused, simulated time stands still", st1["paused"] and st2["t_sim"] == st1["t_sim"],
          (st1["t_sim"], st2["t_sim"]))
    c.call("pause")  # pause while paused is a no-op
    wait_frames(c, 5)
    check("pause while paused keeps it paused", c.call("get_state")["paused"])
    c.call("resume")
    wait_frames(c, 20)
    st3 = c.call("get_state")
    check("resume: running again", (not st3["paused"]) and st3["t_sim"] > st2["t_sim"])

    # errors keep the server serving
    try:
        c.call("load_experiment", name="does_not_exist")
        check("unknown experiment rejected", False)
    except RemoteError as e:
        check("unknown experiment rejected", e.code == "failed", e.message[:80])

    # the server outlives the client
    c.close()
    t0 = time.monotonic()
    c = SimClient.for_backend(a.backend, timeout=60.0, name="server_check")
    c.connect()
    st = c.call("get_state")
    report["reconnect_get_state_s"] = round(time.monotonic() - t0, 3)
    check("reconnect: same experiment and session, still stepping",
          st["experiment"] == a.experiment and st["session"] == st3["session"], st["session"])

    # repeated experiment loads (scene rebuild inside the same Kit process)
    loads = []
    for i in range(a.loads):
        t0 = time.monotonic()
        r = c.call("load_experiment", name=a.experiment, options={"focus": i % 2 == 1})
        wall = time.monotonic() - t0
        wait_frames(c, 30)
        st = c.call("status")
        g = c.call("get_state")
        loads.append({"i": i, "server_s": r["seconds"], "client_s": round(wall, 3), "rss_mb": st["rss_mb"],
                      "rss_growth_mb": st["rss_growth_mb"], "focus": i % 2 == 1,
                      "block_z": g["block"][2], "frames": g["frames"]})
        print(f"  load {i}: {wall:.2f}s rss {st['rss_mb']} MB", flush=True)
    report["loads"] = loads
    check(f"{a.loads} loads, each under 3 s (client side)", all(x["client_s"] < 3.0 for x in loads),
          max(x["client_s"] for x in loads))
    check("every rebuilt scene steps and the block rests in the tray",
          all(x["frames"] >= 30 and abs(x["block_z"] - 0.01675) < 0.003 for x in loads))
    growth = loads[-1]["rss_mb"] - loads[0]["rss_mb"]
    report["rss_growth_per_load_mb"] = round(growth / max(1, len(loads) - 1), 1)
    print(f"  rss growth {growth} MB over {len(loads) - 1} loads", flush=True)
    c.call("load_experiment", name=a.experiment, options={})
    c.close()

    report["checks"] = checks
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(report, f, indent=2)
    n = sum(x["ok"] for x in checks)
    print(f"{n}/{len(checks)} checks passed; wrote {a.out}")
    return 0 if n == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
