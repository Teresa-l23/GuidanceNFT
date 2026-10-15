#!/usr/bin/env bash
# Usage: scripts/train.sh <config.json> <run-name> [extra --config.* overrides]
# Trains SD3.5-Medium with LoRA under NFT (cfg1 / cfg1_dual / cfg4_rollout) or
# GuidanceNFT (v5flow learned center) on NGPU GPUs (default 2, as in the paper).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
RUN_NAME="$2"; shift 2
: "${GUIDANCENFT_ASSETS:?set GUIDANCENFT_ASSETS to the asset root}"
set -a
source "$HERE/baselines/diffusion_nft/scripts/assets_env.sh"
set +a
export DIFFUSION_NFT_OFFLINE="${DIFFUSION_NFT_OFFLINE:-1}" WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false
export DIFFUSION_NFT_CONFIG_SHA256="$(sha256sum "$CONFIG" | cut -d" " -f1)"  # recorded in center checkpoints
export PYTHONPATH="$DIFFUSION_NFT_UPSTREAM_ROOT:$HERE:${PYTHONPATH:-}"
RUN_DIR="$GUIDANCENFT_ASSETS/runs/diffusion_nft_sd35_v5/$RUN_NAME"
mkdir -p "$RUN_DIR"
cd "$DIFFUSION_NFT_UPSTREAM_ROOT"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NGPU:-2}" \
  scripts/train_nft_sd3.py \
  --config "$HERE/baselines/diffusion_nft_sd35_v5/config/v5flow.py:$CONFIG" \
  --config.pretrained.model="$SD35_MODEL_DIR" \
  --config.pretrained.local_files_only=True \
  --config.save_dir="$RUN_DIR/train" \
  --config.logdir="$RUN_DIR/wandb" \
  --config.tensorboard_dir="$RUN_DIR/tensorboard" \
  --config.inline_preview_dir="$RUN_DIR/previews" \
  --config.run_name="$RUN_NAME" "$@"
