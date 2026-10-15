#!/usr/bin/env bash
# Fetch NVIDIA EDM2 (toy_example.py, dnnlib, torch_utils; CC BY-NC-SA 4.0) at the pinned
# commit, apply patches/toy_example.patch, and download the released Autoguidance toy models.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/third_party/edm2"
COMMIT=4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc
: "${GUIDANCENFT_ASSETS:?set GUIDANCENFT_ASSETS to the asset root}"
if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/NVlabs/edm2.git "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
if git -C "$DEST" apply --check "$HERE/patches/toy_example.patch" 2>/dev/null; then
  git -C "$DEST" apply "$HERE/patches/toy_example.patch"
fi
MODELS="$GUIDANCENFT_ASSETS/models/autoguidance-toy"
mkdir -p "$MODELS"
for name in clsA-layers04-dim64/iter4096 clsAB-layers04-dim32/iter0512; do
  out="$MODELS/${name%/*}-${name#*/}.pkl"
  [[ -s "$out" ]] || curl -fL "https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/$name.pkl" -o "$out"
done
echo "EDM2 at $COMMIT in $DEST; toy models in $MODELS"
