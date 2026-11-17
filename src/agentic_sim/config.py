"""Minimal YAML config loading. Values may be plain or {value, source, note} entries (SceneSpec provenance)."""

from __future__ import annotations

import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path: str) -> dict:
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    with open(path) as f:
        return yaml.safe_load(f)


def v(entry):
    """Return the value of a provenance entry, or the entry itself."""
    return entry["value"] if isinstance(entry, dict) and "value" in entry else entry


def provisional_fields(d: dict, prefix: str = "") -> list:
    """Dotted keys whose source is provisional (for logs and reports)."""
    out = []
    for k, e in d.items():
        key = f"{prefix}{k}"
        if isinstance(e, dict) and "value" in e:
            if e.get("source") == "provisional":
                out.append(key)
        elif isinstance(e, dict):
            out += provisional_fields(e, key + ".")
    return out
