#!/usr/bin/env python3
"""Host side of the simulator's right-column screen video (agentic_sim.screen_video).

  uv run tools/screen_video.py agent outputs/screen_video/control/<run id>   # next to an Isaac command run by hand
  uv run tools/screen_video.py status outputs/screen_video/control/<run id>

tools/sim.py starts the agent by itself; the MuJoCo script starts one in its own process. The agent records only the
window the simulator names (tools/record_isaac_window.sh, by X window id) and writes outputs/screen_video/*.mp4.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from agentic_sim.screen_video import ScreenVideoAgent, _abs, _read_json  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["agent", "status"])
    ap.add_argument("control_dir")
    ap.add_argument("--parent-pid", type=int, default=None, help="exit (after saving) when this process is gone")
    a = ap.parse_args(argv)
    if a.action == "status":
        print(json.dumps(_read_json(os.path.join(_abs(a.control_dir), "status.json")), indent=2, ensure_ascii=False))
        return 0
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    agent = ScreenVideoAgent(a.control_dir, log=lambda m: print(f"[screen_video] {m}", flush=True))
    saved = agent.run(stop, parent_pid=a.parent_pid)
    for p in saved:
        print(f"[screen_video] saved {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
