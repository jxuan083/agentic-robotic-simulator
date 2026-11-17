#!/usr/bin/env bash
# Fetch the NVIDIA Sim-to-Real SO-101 Workshop (Apache-2.0) at the pinned commit for the SO-101 USD asset.
set -euo pipefail
WORKSHOP_DIR="${SO101_WORKSHOP_DIR:-$HOME/ref/Sim-to-Real-SO-101-Workshop}"
COMMIT="ce807d9"
if [[ ! -d "$WORKSHOP_DIR/.git" ]]; then
  git clone https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop.git "$WORKSHOP_DIR"
fi
git -C "$WORKSHOP_DIR" fetch --quiet origin
git -C "$WORKSHOP_DIR" checkout --quiet "$COMMIT"
git -C "$WORKSHOP_DIR" lfs pull
echo "SO-101 workshop at $(git -C "$WORKSHOP_DIR" rev-parse --short HEAD): $WORKSHOP_DIR"
