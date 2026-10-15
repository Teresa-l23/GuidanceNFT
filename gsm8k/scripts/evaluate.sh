#!/usr/bin/env bash
# Usage: scripts/evaluate.sh PACKAGE OUTPUT_DIR
# Pass@1 on the 1,319 GSM8K test questions with the package's own inference settings
# (64 steps, asynchronous clocks (2.5, 2, 1.5), SCCFG 2), one directory per seed.
# SEEDS defaults to the paper's eight evaluation seeds.
set -euo pipefail
: "${GUIDANCENFT_ASSETS:?set GUIDANCENFT_ASSETS to the asset root}"
PACKAGE=${1:?package directory}
OUTPUT=${2:?output directory}
DATA="$GUIDANCENFT_ASSETS/gsm8k/test"
for seed in ${SEEDS:-42 123 456 789 2026 31415 27182 16180}; do
  out="$OUTPUT/seed$seed"
  [[ -f "$out/scoring/scoring.json" ]] && continue
  cedr generate --config "$PACKAGE/config.json" --checkpoint "$PACKAGE" \
    --data "$DATA" --seed "$seed" --output "$out"
  cedr score --config "$PACKAGE/config.json" --data "$DATA" \
    --predictions "$out/predictions.jsonl" --workers 8 --output "$out/scoring"
  echo "seed $seed: $(cat "$out/scoring/scoring.json")"
done
