#!/usr/bin/env bash
# Fetch the SiT model and sampler definitions from Diffusion-wo-CFG (Model Guidance).
# They are not redistributed here; this script places them in third_party/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/third_party/Diffusion-wo-CFG"
COMMIT=53c386f4f12d2cc6fdfa6584f3e0d107380f0003
if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/tzco/Diffusion-wo-CFG.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$COMMIT" || true
git -C "$DEST" checkout --quiet "$COMMIT"
echo "Diffusion-wo-CFG at $COMMIT in $DEST"
echo "Add it to PYTHONPATH: export PYTHONPATH=$HERE:$DEST:\${PYTHONPATH:-}"
