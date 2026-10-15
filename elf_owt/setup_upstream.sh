#!/usr/bin/env bash
# Fetch ELF (Embedded Language Flows, MIT license) at the pinned commit and apply
# the small src/ changes that elf_nft needs (patches/elf_src.patch).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/third_party/ELF"
COMMIT=5098bf28b5e9b52c329970a7e4e1cc28251c76e6
if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/lillian039/ELF.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$COMMIT" || true
git -C "$DEST" checkout --quiet "$COMMIT"
if git -C "$DEST" apply --check "$HERE/patches/elf_src.patch" 2>/dev/null; then
  git -C "$DEST" apply "$HERE/patches/elf_src.patch"
  echo "Applied patches/elf_src.patch"
else
  git -C "$DEST" apply --reverse --check "$HERE/patches/elf_src.patch" \
    && echo "patches/elf_src.patch already applied"
fi
echo "export ELF_REPO_ROOT=$DEST"
echo "export PYTHONPATH=$DEST/src:$HERE:\${PYTHONPATH:-}"
