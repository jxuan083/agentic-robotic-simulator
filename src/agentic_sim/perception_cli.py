"""Command-line options of live perception (issue #14), with no numerical dependency.

Kept outside the agentic_sim.perception package on purpose: importing that package (its __init__) loads NumPy and
the detector stack, and the launchers (tools/sim.py, tools/lab.py) must parse --help or print a command with
PyYAML only. The perception code is imported only when live perception actually runs."""

from __future__ import annotations

DEFAULTS = {"live_perception": False, "perception_camera": "both", "perception_hz": 2.0, "perception_json": None,
            "perception_stage": "estimate", "perception_delay": 0.0, "annotations": "on"}
EXTRA = ("perception_stage", "perception_delay", "annotations")   # passed on only when not the default


def add_arguments(parser):
    parser.add_argument('--live-perception', action='store_true', help='show live RGB block corners and coordinates')
    parser.add_argument('--perception-camera', choices=('both', 'wrist', 'global'), default=DEFAULTS['perception_camera'])
    parser.add_argument('--perception-hz', type=float, default=DEFAULTS['perception_hz'],
                        help='maximum inference updates per second (default: 2)')
    parser.add_argument('--perception-json', default=DEFAULTS['perception_json'],
                        help='optional path periodically overwritten with latest full estimates')
    parser.add_argument('--perception-stage', choices=('estimate', 'capture'), default=DEFAULTS['perception_stage'],
                        help='estimate (default): capture and estimate; capture: capture only, no estimation '
                             '(benchmarks the capture cost)')
    parser.add_argument('--perception-delay', type=float, default=DEFAULTS['perception_delay'],
                        help='test only: extra seconds before each estimate completes (late results must show as '
                             'dated snapshots, never on newer frames)')
    parser.add_argument('--annotations', choices=('on', 'off'), default=DEFAULTS['annotations'],
                        help='camera cells start with the annotated snapshot (on) or the raw image (off); toggled '
                             'per camera in the window. Independent of --live-perception')


def options(ns) -> dict:
    """Perception options of a parsed namespace for the server ({} when off). A namespace built without these
    options (another caller, an older settings object) takes the command-line defaults, i.e. perception off."""
    get = lambda k: getattr(ns, k, DEFAULTS[k])  # noqa: E731
    if not get("live_perception"):
        return {}
    out = {"live_perception": True, "perception_camera": get("perception_camera"),
           "perception_hz": get("perception_hz"), "perception_json": get("perception_json")}
    out.update({k: get(k) for k in EXTRA if get(k) != DEFAULTS[k]})
    return out


def args(opts) -> list:
    """Command-line arguments of the perception options of a RunOptions-like object (only what is on)."""
    if not getattr(opts, "live_perception", False):
        return []
    out = ["--live-perception", "--perception-camera", opts.perception_camera, "--perception-hz", str(opts.perception_hz)]
    if opts.perception_json:
        out += ["--perception-json", opts.perception_json]
    if getattr(opts, "perception_stage", "estimate") != "estimate":
        out += ["--perception-stage", opts.perception_stage]
    if getattr(opts, "perception_delay", 0.0):
        out += ["--perception-delay", str(opts.perception_delay)]
    if getattr(opts, "annotations", "on") != "on":
        out += ["--annotations", opts.annotations]
    return out
