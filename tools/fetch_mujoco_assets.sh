#!/usr/bin/env bash
# Fetch the SO-101 MJCF from MuJoCo Menagerie (Apache-2.0) at the pinned commit; only robotstudio_so101/ is checked out.
set -euo pipefail
MENAGERIE_DIR="${MENAGERIE_DIR:-$HOME/ref/mujoco_menagerie}"
COMMIT="3abca2e302625cec9429c49806121ea2855cef81"   # 2026-09-28, "Fix SO101 wrist_roll joint range upper limit"
if [[ ! -d "$MENAGERIE_DIR/.git" ]]; then
  git clone --quiet --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$MENAGERIE_DIR"
  git -C "$MENAGERIE_DIR" sparse-checkout set robotstudio_so101
fi
git -C "$MENAGERIE_DIR" fetch --quiet origin
git -C "$MENAGERIE_DIR" checkout --quiet "$COMMIT"
echo "SO-101 MJCF at $(git -C "$MENAGERIE_DIR" rev-parse --short HEAD): $MENAGERIE_DIR/robotstudio_so101"
