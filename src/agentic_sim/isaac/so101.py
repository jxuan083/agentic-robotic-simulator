"""SO-101 articulation config reused from the NVIDIA Sim-to-Real SO-101 Workshop (Apache-2.0).

The workshop checkout is not vendored; it is mounted at $SO101_WORKSHOP_DIR (pinned commit in
docker/isaac/fetch_assets.sh). Actuator gains and the USD come from the workshop unchanged.
"""

from __future__ import annotations

import importlib.util
import os

WORKSHOP_COMMIT = "ce807d9"
_ASSET_REL = "source/sim_to_real_so101/assets"


def workshop_dir() -> str:
    return os.environ.get("SO101_WORKSHOP_DIR", os.path.expanduser("~/ref/Sim-to-Real-SO-101-Workshop"))


def _load_workshop_module():
    path = os.path.join(workshop_dir(), _ASSET_REL, "so101.py")
    spec = importlib.util.spec_from_file_location("so101_workshop_asset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def so101_cfg(prim_path: str, pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)):
    """Workshop SO-101 (no-camera USD) placed with its base frame at pos/rot (w, x, y, z)."""
    mod = _load_workshop_module()
    cfg = mod.SO101_CFG.replace(prim_path=prim_path)
    cfg.spawn = cfg.spawn.replace(usd_path=os.path.join(workshop_dir(), _ASSET_REL, "usd/SO-ARM101-USD-NO-CAMERA.usd"))
    cfg.init_state = cfg.init_state.replace(pos=tuple(pos), rot=tuple(rot))
    return cfg
