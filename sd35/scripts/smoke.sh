#!/usr/bin/env bash
# Usage: scripts/smoke.sh {nft|guidancenft} [task]   (task: pickscore, geneval, ocr, learned3)
# Two policy updates with 8 prompt groups; GuidanceNFT also fits its center for one epoch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${2:-pickscore}"
case "$1" in
  nft) PATTERN="cfg_${TASK}_b24_g48_steps10_cfg1_u*" ;;
  guidancenft) PATTERN="cfg_${TASK}_b24_g48_steps10_cfg4_v5flow_pretrain100_u*" ;;
  *) echo "Usage: $0 {nft|guidancenft} [task]" >&2; exit 2 ;;
esac
SOURCE="$(ls "$HERE"/baselines/diffusion_nft_sd35_v5/config/$PATTERN.json | head -1)"
RUN_NAME="$1-$TASK-smoke-$(date +%Y%m%d-%H%M%S)"
SMOKE_CONFIG="$GUIDANCENFT_ASSETS/runs/diffusion_nft_sd35_v5/$RUN_NAME.smoke_config.json"
mkdir -p "$(dirname "$SMOKE_CONFIG")"
python "$HERE/scripts/make_smoke_config.py" "$SOURCE" "$SMOKE_CONFIG"
echo "smoke of $(basename "$SOURCE")"
exec "$HERE/scripts/train.sh" "$SMOKE_CONFIG" "$RUN_NAME"
