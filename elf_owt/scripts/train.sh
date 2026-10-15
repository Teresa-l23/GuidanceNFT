#!/usr/bin/env bash
# Usage: scripts/train.sh <config.json|yml> [extra trainer args...]
# Trains ELF-B on OpenWebText with NFT or GuidanceNFT on all visible GPUs (JAX pmap).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
: "${GUIDANCENFT_ASSETS:?set GUIDANCENFT_ASSETS to the asset root (outputs go to runs/elf_nft/)}"
export ELF_REPO_ROOT="${ELF_REPO_ROOT:-$HERE/third_party/ELF}"
export PYTHONPATH="$ELF_REPO_ROOT/src:$HERE:${PYTHONPATH:-}"
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
cd "$HERE"
python -m baselines.elf_nft.scripts.train_nft_elf --owt --config "$CONFIG" "$@" --validate_only
python -m baselines.elf_nft.scripts.train_nft_elf --owt --config "$CONFIG" "$@"
