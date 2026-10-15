#!/usr/bin/env bash
# Reproduces the toy figures of the paper on one GPU (about 15 minutes):
#   1. NFT on the 2D tree toy with three objectives (Figure 1 samplers);
#   2. oracle iterated NFT updates under CFG 4 for each loss center (reference-error fields);
#   3. the figure panels, including the method figure and the appendix bias fields.
# Usage: scripts/reproduce_paper_figures.sh [output dir under $GUIDANCENFT_ASSETS]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GUIDANCENFT_ASSETS:?set GUIDANCENFT_ASSETS to the asset root}"
OUT="${1:-$GUIDANCENFT_ASSETS/runs/toy/$(date +%Y%m%d-%H%M%S)}"
MODELS="$GUIDANCENFT_ASSETS/models/autoguidance-toy"
A64="$MODELS/clsA-layers04-dim64-iter4096.pkl"
AB32="$MODELS/clsAB-layers04-dim32-iter0512.pkl"
DEVICE="${DEVICE:-cuda}"
export PYTHONPATH="$HERE:$HERE/third_party/edm2:${PYTHONPATH:-}"
cd "$HERE"

train() {  # arm config good bad
  python toy_nft_guidance.py run --config "configs/$2.json" --output-dir "$OUT/training/$1" \
    --good-model "$3" --bad-model "$4" --device "$DEVICE" --no-figure
}
train a64_nft trunk3_a64_separate_w3_10step "$A64" "$AB32"
train ab32_nft trunk3_ab32_separate_w3_10step "$AB32" "$AB32"
train joint_guided_field trunk3_joint_guided_field_w3_10step "$A64" "$AB32"
train joint_paired_branches trunk3_joint_paired_branches_w3_10step "$A64" "$AB32"
python toy_nft_guided_vs_separate_comparison.py \
  --joint-run-dir "$OUT/training/joint_guided_field" --separate-root "$OUT" \
  --paired-run-dir "$OUT/training/joint_paired_branches" \
  --output-dir "$OUT/evaluation" --device "$DEVICE"

for arm in ideal std cond std_bias cond_bias; do
  python toy_cfg4_iterated_push.py --output "$OUT/reference_fields" --arm "$arm" --snapshot-rounds 0 5 10
done
# Draw all arms together: the colour scales are shared across rows.
mkdir -p "$OUT/figures"
python plot_toy_cfg4_bias_only_fields.py --run "$OUT/reference_fields" \
  --output "$OUT/figures/toy_reference_fields" --figure1-data "$OUT/evaluation"
python plot_figure1_samplers.py "$OUT/evaluation" "$OUT/figures" thin   # Figure 1
echo "Figures in $OUT/figures"
