"""Experiment registry: one YAML per experiment in configs/experiments/, referencing existing configs only.

An experiment names the scene, robots, teleop / colour-DR configs and acceptance scenarios, and the backends that
support it. It never repeats their values, so a scene change is made in one place. Pure Python (PyYAML only): the
menu, the server and tests load it without Isaac Sim or MuJoCo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from agentic_sim.config import REPO_ROOT, load

EXPERIMENT_DIR = "configs/experiments"
ACCEPTANCE_DIR = "configs/acceptance"
BACKENDS = ("isaac", "mujoco")
ROBOT_MODELS = ("so101",)
RUN_OPTIONS = ("focus", "color_dr", "color_dr_seed", "blocks")

_REQUIRED = ("name", "description", "backends", "scene", "robots", "teleop")
_OPTIONAL = ("color_dr", "cameras", "acceptance")


class ExperimentError(ValueError):
    """Invalid experiment file; the message lists every problem found."""


@dataclass(frozen=True)
class Robot:
    id: str
    model: str


@dataclass(frozen=True)
class AcceptanceEntry:
    scenario: str
    options: dict = field(default_factory=dict)  # run options forced for this scenario (e.g. color_dr)


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    backends: tuple
    scene: str
    robots: tuple
    teleop: str
    color_dr: str | None
    cameras: tuple
    acceptance: tuple
    path: str

    def supports(self, backend: str) -> bool:
        return backend in self.backends

    def scenario(self, name: str) -> AcceptanceEntry:
        for a in self.acceptance:
            if a.scenario == name:
                return a
        raise KeyError(f"experiment {self.name!r} has no acceptance scenario {name!r} "
                       f"(available: {[a.scenario for a in self.acceptance]})")


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _exists(path) -> bool:
    return isinstance(path, str) and os.path.isfile(_abs(path))


def parse(raw: dict, path: str) -> Experiment:
    """Validate a loaded experiment dict; raise ExperimentError listing all problems."""
    errs = []
    if not isinstance(raw, dict):
        raise ExperimentError(f"{path}: expected a mapping")
    for k in _REQUIRED:
        if k not in raw:
            errs.append(f"missing key {k!r}")
    for k in raw:
        if k not in _REQUIRED + _OPTIONAL:
            errs.append(f"unknown key {k!r}")
    stem = os.path.splitext(os.path.basename(path))[0]
    if raw.get("name") != stem:
        errs.append(f"name {raw.get('name')!r} must match the file name {stem!r}")

    backends = raw.get("backends") or []
    if not isinstance(backends, list) or not backends:
        errs.append("backends must be a non-empty list")
        backends = []
    for b in backends:
        if b not in BACKENDS:
            errs.append(f"unknown backend {b!r} (known: {list(BACKENDS)})")

    for k in ("scene", "teleop"):
        if k in raw and not _exists(raw[k]):
            errs.append(f"{k}: file not found: {raw[k]}")
    if raw.get("color_dr") is not None and not _exists(raw["color_dr"]):
        errs.append(f"color_dr: file not found: {raw['color_dr']}")

    robots = []
    rl = raw.get("robots") or []
    if not isinstance(rl, list) or len(rl) != 1:
        errs.append("robots must list exactly one arm (multi-arm setups are not supported yet)")
    for r in rl if isinstance(rl, list) else []:
        if not isinstance(r, dict) or set(r) != {"id", "model"}:
            errs.append(f"robot entry must be {{id, model}} (the mount pose comes from the scene): {r!r}")
            continue
        if r["model"] not in ROBOT_MODELS:
            errs.append(f"unknown robot model {r['model']!r} (known: {list(ROBOT_MODELS)})")
        robots.append(Robot(str(r["id"]), str(r["model"])))

    cameras = raw.get("cameras") or []
    if cameras:
        errs.append("cameras are not supported yet (leave the list empty)")

    acceptance = []
    for a in raw.get("acceptance") or []:
        if isinstance(a, str):
            a = {"scenario": a}
        if not isinstance(a, dict) or "scenario" not in a or set(a) - {"scenario", "options"}:
            errs.append(f"acceptance entry must be a name or {{scenario, options}}: {a!r}")
            continue
        name, opts = str(a["scenario"]), dict(a.get("options") or {})
        if not _exists(os.path.join(ACCEPTANCE_DIR, name + ".yaml")):
            errs.append(f"acceptance scenario not found: {ACCEPTANCE_DIR}/{name}.yaml")
        for o in opts:
            if o not in RUN_OPTIONS:
                errs.append(f"acceptance {name}: unknown option {o!r} (known: {list(RUN_OPTIONS)})")
        acceptance.append(AcceptanceEntry(name, opts))
    names = [a.scenario for a in acceptance]
    if len(set(names)) != len(names):
        errs.append("acceptance scenarios must be unique")

    if errs:
        raise ExperimentError(f"{path}:\n  - " + "\n  - ".join(errs))
    return Experiment(
        name=raw["name"], description=str(raw["description"]), backends=tuple(backends), scene=raw["scene"],
        robots=tuple(robots), teleop=raw["teleop"], color_dr=raw.get("color_dr"), cameras=tuple(cameras),
        acceptance=tuple(acceptance), path=path)


def load_experiment(name_or_path: str, directory: str = EXPERIMENT_DIR) -> Experiment:
    """Load by name (configs/experiments/<name>.yaml) or by path."""
    path = name_or_path if name_or_path.endswith((".yaml", ".yml")) else os.path.join(directory, name_or_path + ".yaml")
    if not os.path.isfile(_abs(path)):
        raise ExperimentError(f"experiment not found: {path} (available: {[e.name for e in list_experiments(directory)]})")
    return parse(load(path), path)


def list_experiments(directory: str = EXPERIMENT_DIR, backend: str | None = None) -> list:
    """All experiments, sorted by name; optionally only those supporting `backend`. Invalid files raise."""
    d = _abs(directory)
    out = []
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith((".yaml", ".yml")):
            e = load_experiment(os.path.join(directory, f))
            if backend is None or e.supports(backend):
                out.append(e)
    return out
