#!/usr/bin/env bash
# Usage: scripts/smoke.sh {nft|guidancenft}
# Two policy updates with 32 rollouts and a 32-sample final evaluation.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$1" in
  nft) BASE="owt_w1_conditional_only_standard_100_cfg_sweep" ;;
  guidancenft) BASE="owt_w3_conditional_only_shifted_center_100" ;;
  *) echo "Usage: $0 {nft|guidancenft}" >&2; exit 2 ;;
esac
RUN_DIR="$GUIDANCENFT_ASSETS/runs/elf_nft/smokes/$1-smoke-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
python - "$HERE/baselines/elf_nft/configs/$BASE.json" "$RUN_DIR/smoke_config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
config["formal_experiment"] = False
config["evaluation"].update(num_samples=32, at_start=False)
json.dump(config, open(sys.argv[2], "w"), indent=2)
PY
exec "$HERE/scripts/train.sh" "$RUN_DIR/smoke_config.json" --output_dir "$RUN_DIR" \
  --max_outer_steps 2 --samples_per_prompt 32 --rollout_generation_batch_size 8
