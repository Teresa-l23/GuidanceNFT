#!/usr/bin/env bash
# Usage: scripts/train.sh <config.json> [extra trainer args...]
# Runs the SiT explicit-CFG trainer for configs/sit_xl_*.json and the
# Model Guidance trainer for configs/mg_*.json on all visible GPUs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"; shift
export PYTHONPATH="$HERE:$HERE/third_party/Diffusion-wo-CFG:${PYTHONPATH:-}"
: "${GUIDANCENFT_ASSETS:?set GUIDANCENFT_ASSETS to the asset root (checkpoints/, runs/)}"
case "$(basename "$CONFIG")" in
  sit_xl_*) TRAINER="$HERE/train_sit_xl_cfg_nft.py" ;;
  mg_*) TRAINER="$HERE/train_diffusion_nft.py" ;;
  *) echo "Unknown config family: $CONFIG" >&2; exit 2 ;;
esac
NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
python "$TRAINER" --config "$CONFIG" "$@" --validate-only
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NGPU" \
  "$TRAINER" --config "$CONFIG" "$@"
