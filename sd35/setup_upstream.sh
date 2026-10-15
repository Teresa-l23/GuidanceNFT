#!/usr/bin/env bash
# Fetch NVIDIA DiffusionNFT (Apache-2.0) at the pinned commit into
# baselines/diffusion_nft/upstream and apply patches/diffusion_nft_upstream.patch
# (offline reward loading, CFG objective modes, the learned-center extension hooks).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/baselines/diffusion_nft/upstream"
COMMIT="$(cat "$HERE/baselines/diffusion_nft/UPSTREAM_COMMIT.txt")"
if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/NVlabs/DiffusionNFT.git "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
if git -C "$DEST" apply --check "$HERE/patches/diffusion_nft_upstream.patch" 2>/dev/null; then
  git -C "$DEST" apply "$HERE/patches/diffusion_nft_upstream.patch"
  echo "Applied patches/diffusion_nft_upstream.patch"
else
  git -C "$DEST" apply --reverse --check "$HERE/patches/diffusion_nft_upstream.patch" \
    && echo "patches/diffusion_nft_upstream.patch already applied"
fi
echo "export PYTHONPATH=$HERE:\${PYTHONPATH:-}"
