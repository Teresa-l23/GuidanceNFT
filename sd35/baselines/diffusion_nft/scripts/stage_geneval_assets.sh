#!/usr/bin/env bash

# Stage the gated SD3.5 Medium base model and the two GenEval reward assets.
# Run this once in a network-enabled Slurm compute allocation. Formal training
# and evaluation use only these local assets with Hugging Face offline mode.

set -euo pipefail
set +x  # Never expose HF_TOKEN even if the caller enabled shell tracing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/assets_env.sh"

usage() {
  cat <<'EOF'
Usage:
  SD35_MODEL_REVISION=<40-hex-HF-commit> HF_TOKEN=<read-token> \
    bash baselines/diffusion_nft/scripts/stage_geneval_assets.sh [--stage]
  SD35_MODEL_REVISION=<40-hex-HF-commit> \
    bash baselines/diffusion_nft/scripts/stage_geneval_assets.sh --check

The SD3.5 Medium repository is gated. Before staging, accept its license at
https://huggingface.co/stabilityai/stable-diffusion-3.5-medium and supply a
read-capable HF_TOKEN through the environment. The token is neither printed nor
written to the asset manifest. A moving revision such as `main` is rejected.
EOF
}

MODE="stage"
case "${1:-}" in
  ""|--stage) ;;
  --check) MODE="check" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

diffusion_nft_require_compute_allocation
diffusion_nft_require_sd35_revision

verify_file() {
  local path="$1"
  local expected_sha="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing staged asset: $path" >&2
    return 1
  fi
  local actual_sha
  actual_sha="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "Checksum mismatch for $path" >&2
    echo "expected=$expected_sha" >&2
    echo "actual=$actual_sha" >&2
    return 1
  fi
}

check_staged_assets() {
  [[ -f "$SD35_MODEL_DIR/model_index.json" ]] || {
    echo "Missing staged SD3.5 model_index.json under $SD35_MODEL_DIR" >&2
    return 1
  }
  [[ -f "$SD35_MODEL_DIR/.diffusion_nft_revision" ]] || {
    echo "Missing SD3.5 revision marker under $SD35_MODEL_DIR" >&2
    return 1
  }
  [[ -f "$SD35_MODEL_DIR/.diffusion_nft_complete" ]] || {
    echo "Missing SD3.5 completion marker under $SD35_MODEL_DIR" >&2
    return 1
  }
  [[ "$(tr -d '[:space:]' < "$SD35_MODEL_DIR/.diffusion_nft_revision")" == "$SD35_MODEL_REVISION" ]] || {
    echo "The staged SD3.5 revision marker does not match SD35_MODEL_REVISION." >&2
    return 1
  }
  verify_file "$GENEVAL_DETECTOR_PATH" "$GENEVAL_DETECTOR_SHA256"
  verify_file "$GENEVAL_OPENAI_CLIP_PATH" "$GENEVAL_OPENAI_CLIP_SHA256"
  [[ -f "$DIFFUSION_NFT_STAGE_MANIFEST" ]] || {
    echo "Missing staging manifest: $DIFFUSION_NFT_STAGE_MANIFEST" >&2
    return 1
  }
  echo "All pinned GenEval assets are staged and checksum-valid."
  echo "SD35_MODEL_DIR=$SD35_MODEL_DIR"
  echo "GENEVAL_DETECTOR_PATH=$GENEVAL_DETECTOR_PATH"
  echo "GENEVAL_OPENAI_CLIP_PATH=$GENEVAL_OPENAI_CLIP_PATH"
}

if [[ "$MODE" == "check" ]]; then
  check_staged_assets
  exit 0
fi

if [[ -z "$DIFFUSION_NFT_CONDA_EXE" || ! -x "$DIFFUSION_NFT_CONDA_EXE" ]]; then
  echo "Could not find an executable Conda; set DIFFUSION_NFT_CONDA_EXE." >&2
  exit 1
fi
if ! "$DIFFUSION_NFT_CONDA_EXE" env list \
  | awk -v wanted="$DIFFUSION_NFT_CONDA_ENV" '$1 == wanted { found = 1 } END { exit !found }'; then
  echo "Missing staging environment: $DIFFUSION_NFT_CONDA_ENV" >&2
  echo "Run setup_environment.sh --create inside a GPU allocation first." >&2
  exit 1
fi

mkdir -p \
  "$SD35_MODEL_DIR" \
  "$DIFFUSION_NFT_REWARD_CKPT_DIR" \
  "$HF_HUB_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME"

download_checked_file() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"

  if [[ -f "$destination" ]]; then
    if verify_file "$destination" "$expected_sha"; then
      echo "Reusing checksum-valid asset: $destination"
      return 0
    fi
    echo "Refusing to overwrite the checksum-invalid asset above." >&2
    echo "Move it aside explicitly, then rerun staging." >&2
    return 1
  fi

  command -v curl >/dev/null || {
    echo "curl is required to stage GenEval assets." >&2
    return 1
  }
  local temporary_path
  temporary_path="$(mktemp "${destination}.partial.XXXXXX")"
  if ! curl --fail --location --retry 5 --retry-all-errors --output "$temporary_path" "$url"; then
    rm -f "$temporary_path"
    echo "Download failed: $url" >&2
    return 1
  fi
  if ! verify_file "$temporary_path" "$expected_sha"; then
    rm -f "$temporary_path"
    return 1
  fi
  chmod 0444 "$temporary_path"
  mv "$temporary_path" "$destination"
}

if [[ ! -f "$SD35_MODEL_DIR/.diffusion_nft_complete" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required to stage gated $SD35_MODEL_REPO." >&2
    echo "Accept the model license and export a read-capable token; it will not be logged." >&2
    exit 1
  fi

  export HF_HUB_OFFLINE=0
  export TRANSFORMERS_OFFLINE=0
  export HF_HUB_DISABLE_TELEMETRY=1
  echo "Staging gated $SD35_MODEL_REPO at immutable revision $SD35_MODEL_REVISION ..."
  if ! "$DIFFUSION_NFT_CONDA_EXE" run --no-capture-output -n "$DIFFUSION_NFT_CONDA_ENV" \
    huggingface-cli download "$SD35_MODEL_REPO" \
      --repo-type model \
      --revision "$SD35_MODEL_REVISION" \
      --local-dir "$SD35_MODEL_DIR" \
      --local-dir-use-symlinks False; then
    echo "SD3.5 Medium staging failed." >&2
    echo "Confirm that the license was accepted, HF_TOKEN has read access, the revision exists," >&2
    echo "and this compute allocation has network access. Partial files were retained for resume." >&2
    exit 1
  fi
  [[ -f "$SD35_MODEL_DIR/model_index.json" ]] || {
    echo "The Hugging Face download completed without model_index.json; refusing to mark it complete." >&2
    exit 1
  }
  printf '%s\n' "$SD35_MODEL_REVISION" > "$SD35_MODEL_DIR/.diffusion_nft_revision"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$SD35_MODEL_DIR/.diffusion_nft_complete"
else
  echo "Reusing completed SD3.5 staging directory: $SD35_MODEL_DIR"
fi

download_checked_file "$GENEVAL_DETECTOR_URL" "$GENEVAL_DETECTOR_PATH" "$GENEVAL_DETECTOR_SHA256"
download_checked_file "$GENEVAL_OPENAI_CLIP_URL" "$GENEVAL_OPENAI_CLIP_PATH" "$GENEVAL_OPENAI_CLIP_SHA256"

manifest_tmp="$(mktemp "${DIFFUSION_NFT_STAGE_MANIFEST}.partial.XXXXXX")"
{
  printf 'STAGED_AT_UTC=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'DIFFUSION_NFT_UPSTREAM_COMMIT=%q\n' "$DIFFUSION_NFT_UPSTREAM_COMMIT"
  printf 'SD35_MODEL_REPO=%q\n' "$SD35_MODEL_REPO"
  printf 'SD35_MODEL_REVISION=%q\n' "$SD35_MODEL_REVISION"
  printf 'SD35_MODEL_DIR=%q\n' "$SD35_MODEL_DIR"
  printf 'SD35_MODEL_FILE_COUNT=%q\n' "$(find "$SD35_MODEL_DIR" -type f | wc -l)"
  printf 'SD35_MODEL_BYTES=%q\n' "$(du -sb "$SD35_MODEL_DIR" | awk '{print $1}')"
  printf 'GENEVAL_DETECTOR_PATH=%q\n' "$GENEVAL_DETECTOR_PATH"
  printf 'GENEVAL_DETECTOR_SHA256=%q\n' "$GENEVAL_DETECTOR_SHA256"
  printf 'GENEVAL_OPENAI_CLIP_PATH=%q\n' "$GENEVAL_OPENAI_CLIP_PATH"
  printf 'GENEVAL_OPENAI_CLIP_SHA256=%q\n' "$GENEVAL_OPENAI_CLIP_SHA256"
  printf 'DIFFUSION_NFT_MMCV_REVISION=%q\n' "$DIFFUSION_NFT_MMCV_REVISION"
  printf 'DIFFUSION_NFT_MMDET_REVISION=%q\n' "$DIFFUSION_NFT_MMDET_REVISION"
  printf 'DIFFUSION_NFT_OPEN_CLIP_REVISION=%q\n' "$DIFFUSION_NFT_OPEN_CLIP_REVISION"
  printf 'DIFFUSION_NFT_CLIP_BENCHMARK_REVISION=%q\n' "$DIFFUSION_NFT_CLIP_BENCHMARK_REVISION"
} > "$manifest_tmp"
chmod 0444 "$manifest_tmp"
mv "$manifest_tmp" "$DIFFUSION_NFT_STAGE_MANIFEST"

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
check_staged_assets
echo "Staging manifest: $DIFFUSION_NFT_STAGE_MANIFEST"
