"""Command-line options of live perception (issue #14), with no numerical dependency.

Kept outside the agentic_sim.perception package on purpose: importing that package (its __init__) loads NumPy and
the detector stack, and the launchers (tools/sim.py, tools/lab.py) must parse --help or print a command with
PyYAML only. The perception code is imported only when live perception actually runs."""

from __future__ import annotations

DEFAULTS = {"live_perception": False, "perception_camera": "both", "perception_hz": 2.0, "perception_json": None}


def add_arguments(parser):
    parser.add_argument('--live-perception', action='store_true', help='show live RGB block corners and coordinates')
    parser.add_argument('--perception-camera', choices=('both', 'wrist', 'global'), default=DEFAULTS['perception_camera'])
    parser.add_argument('--perception-hz', type=float, default=DEFAULTS['perception_hz'],
                        help='maximum inference updates per second (default: 2)')
    parser.add_argument('--perception-json', default=DEFAULTS['perception_json'],
                        help='optional path periodically overwritten with latest full estimates')


def options(ns) -> dict:
    """Perception options of a parsed namespace for the server ({} when off). A namespace built without these
    options (another caller, an older settings object) takes the command-line defaults, i.e. perception off."""
    get = lambda k: getattr(ns, k, DEFAULTS[k])  # noqa: E731
    if not get("live_perception"):
        return {}
    return {"live_perception": True, "perception_camera": get("perception_camera"),
            "perception_hz": get("perception_hz"), "perception_json": get("perception_json")}
