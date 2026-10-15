#!/usr/bin/env bash
# Fetch CEDR (Reasoning with Continuous Latent Diffusion, MIT license) at the pinned
# commit and apply the NFT / GuidanceNFT changes (patches/cedr.patch).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/third_party/CEDR"
COMMIT=538198dce91e5df119edf119df916b2333fafa4c
if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/chengxiang/CEDR.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$COMMIT" || true
git -C "$DEST" checkout --quiet "$COMMIT"
if git -C "$DEST" apply --check "$HERE/patches/cedr.patch" 2>/dev/null; then
  git -C "$DEST" apply "$HERE/patches/cedr.patch"
  echo "Applied patches/cedr.patch"
else
  git -C "$DEST" apply --reverse --check "$HERE/patches/cedr.patch" \
    && echo "patches/cedr.patch already applied"
fi
echo "Install with: pip install -e '$DEST[data,math]'"
