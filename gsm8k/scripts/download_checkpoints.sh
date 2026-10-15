#!/usr/bin/env bash
# Usage: scripts/download_checkpoints.sh
# Downloads the paper's GSM8K checkpoints from jiayin-l23/GuidanceNFT into
# $GUIDANCENFT_ASSETS/gsm8k/checkpoints and completes each package with the shared
# tokenizer and embedding of the CEDR packages (run scripts/prepare_data.sh first).
set -euo pipefail
: "${GUIDANCENFT_ASSETS:?set GUIDANCENFT_ASSETS to the asset root}"
A="$GUIDANCENFT_ASSETS/gsm8k"
SHARED="$A/packages/gsm8k-b-post-nft/shared"
[[ -d "$SHARED" ]] || { echo "missing $SHARED; run scripts/prepare_data.sh" >&2; exit 1; }
hf download jiayin-l23/GuidanceNFT --include 'gsm8k_*/*' --local-dir "$A/checkpoints"
for package in "$A"/checkpoints/gsm8k_*/; do
  [[ -d "$package/shared" ]] || cp -r "$SHARED" "$package/shared"
  python -c "import sys; from cedr.packages import read_package; read_package(sys.argv[1])" "$package"
  echo "verified $package"
done
