#!/usr/bin/env bash
# Usage: scripts/prepare_data.sh
# Downloads the CEDR GSM8K-B packages and prepares the GSM8K train reward population
# (7,473 rows) and the 1,319-question test cohort under $GUIDANCENFT_ASSETS/gsm8k.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GUIDANCENFT_ASSETS:?set GUIDANCENFT_ASSETS to the asset root}"
A="$GUIDANCENFT_ASSETS/gsm8k"
CEDR="$HERE/third_party/CEDR"
REVISION=740312add88f781978c0658806c59bc2815b9866  # openai/gsm8k, as pinned by CEDR
mkdir -p "$A/packages"
for model in gsm8k-b-pre-nft gsm8k-b-post-nft; do
  [[ -f "$A/packages/$model/manifest.json" ]] || cedr download --model "$model" --output "$A/packages/$model"
done
TOKENIZER="$A/packages/gsm8k-b-pre-nft/shared/tokenizer"
cedr prepare-benchmark --benchmark gsm8k --checkpoint "$A/packages/gsm8k-b-post-nft" --output "$A/test"
cedr fetch --dataset openai/gsm8k --subset main --revision "$REVISION" --split train \
  --prompt-field question --answer-field answer --output "$A/train.jsonl"
cedr prepare --config "$CEDR/configs/gsm8k_b_guidancenft.yaml" --input "$A/train.jsonl" \
  --tokenizer "$TOKENIZER" --output "$A/reward-data"
