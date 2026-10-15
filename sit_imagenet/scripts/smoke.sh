#!/usr/bin/env bash
# Usage: scripts/smoke.sh {sit_nft|sit_guidancenft|mg_nft|mg_guidancenft}
# A tiny run (1-2 updates, few samples) that checks the full pipeline end to end.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$1"
STAMP="$(date +%Y%m%d-%H%M%S)"
case "$NAME" in
  sit_nft|sit_guidancenft)
    CONFIG="$HERE/configs/sit_xl_${NAME#sit_}$([[ $NAME == sit_nft ]] && echo _cfg1 || echo _cfg1p5).json"
    PRETRAIN=$([[ $NAME == sit_guidancenft ]] && echo 1 || echo 0)
    ARGS=(--smoke --run-name "${NAME}-smoke-$STAMP"
          --output-root "$GUIDANCENFT_ASSETS/runs/sit_xl_cfg_nft/smokes"
          --max-outer-steps 1 --groups-global 2 --samples-per-class 24
          --timestep-fraction 0.1 --center-timestep-fraction 0.1
          --center-pretrain-updates "$PRETRAIN"
          --center-pretrain-timesteps-per-endpoint 1 --checkpoint-keep 1) ;;
  mg_nft|mg_guidancenft)
    CONFIG="$HERE/configs/${NAME}.json"
    ARGS=(--run-name "${NAME}-smoke-$STAMP"
          --output-root "$GUIDANCENFT_ASSETS/runs/model_guidance_nft/smokes"
          --max-outer-steps 2 --sampling-steps 2 --groups-global 2
          --samples-per-class 2 --rollout-batch-size 2
          --timestep-fraction 0.5 --center-timestep-fraction 0.5
          --checkpoint-save-every 0) ;;
  *) echo "Usage: $0 {sit_nft|sit_guidancenft|mg_nft|mg_guidancenft}" >&2; exit 2 ;;
esac
exec "$HERE/scripts/train.sh" "$CONFIG" "${ARGS[@]}"
