#!/usr/bin/env bash
# Usage: scripts/train.sh {nft|guidancenft} [RUN_NAME] [extra `cedr train nft` args...]
# NFT or GuidanceNFT from the CEDR gsm8k-b-pre-nft package on all visible GPUs.
# Resumes from RUN_DIR/latest.pt when it exists.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GUIDANCENFT_ASSETS:?set GUIDANCENFT_ASSETS to the asset root}"
ARM=${1:?nft or guidancenft}; shift
RUN=${1:-$ARM}; shift || true
A="$GUIDANCENFT_ASSETS/gsm8k"
RECIPE=${RECIPE:-$HERE/third_party/CEDR/configs/gsm8k_b_$ARM.yaml}
OUTPUT="$A/runs/$RUN"
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
mkdir -p "$OUTPUT"
resume=()
[[ -f "$OUTPUT/latest.pt" ]] && resume=(--resume "$OUTPUT/latest.pt")
exec torchrun --standalone --nproc-per-node="$GPUS" -m cedr train nft \
  --config "$RECIPE" --data "$A/reward-data" --init "$A/packages/gsm8k-b-pre-nft" \
  --micro-batch "${MICRO:-24}" --output "$OUTPUT" "${resume[@]}" "$@"
