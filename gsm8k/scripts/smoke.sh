#!/usr/bin/env bash
# Usage: scripts/smoke.sh {nft|guidancenft}
# Two center-pretraining updates (GuidanceNFT) and two policy updates on 4 groups per GPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GUIDANCENFT_ASSETS:?set GUIDANCENFT_ASSETS to the asset root}"
ARM=${1:?nft or guidancenft}
RUN="smoke-$ARM-$(date +%Y%m%d-%H%M%S)"
OUTPUT="$GUIDANCENFT_ASSETS/gsm8k/runs/$RUN"
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
mkdir -p "$OUTPUT"
python - "$HERE/third_party/CEDR/configs/gsm8k_b_$ARM.yaml" "$OUTPUT/smoke_recipe.yaml" "$GPUS" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["nft"].update(prompt_groups=4 * int(sys.argv[3]), save_every=1000, eval_every=0)
if cfg["nft"].get("center") == "learned":
    cfg["nft"].update(center_pretrain_updates=2, center_pretrain_save_every=1000)
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
PY
RECIPE="$OUTPUT/smoke_recipe.yaml" exec "$HERE/scripts/train.sh" "$ARM" "$RUN" --max-updates 2 --no-compile
