#!/usr/bin/env bash

# Stage the optional DiffusionNFT reward models and construct a dedicated
# dependency environment. Persistent weights, caches, wheels, source locks,
# manifests, and the cloned Conda environment all remain in group storage.

set -euo pipefail
set +x
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/assets_env.sh"

usage() {
  cat <<'EOF'
Usage:
  bash baselines/diffusion_nft/scripts/stage_optional_reward_assets.sh --prepare
  bash baselines/diffusion_nft/scripts/stage_optional_reward_assets.sh --check

Both modes require a Slurm compute allocation. --prepare downloads pinned
CLIPScore, HPSv2.1, Aesthetic, ImageReward, and BERT-tokenizer assets; clones
sit-nft-diffusionnft into sit-nft-diffusionnft-rewards; installs the pinned
optional packages; and verifies both environments and all static assets on CPU.
--check is network-free and read-only. GPU scorer forwards are intentionally
deferred to the separate training capacity smoke.

UnifiedReward is intentionally excluded: the repository calls a separately
deployed sglang HTTP service at 127.0.0.1:17140, so it is not a static scorer
that can be made runnable by this environment-and-checkpoint setup.
EOF
}

case "${1:-}" in
  --prepare) MODE=prepare ;;
  --check) MODE=check ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# != 1 )); then
  usage >&2
  exit 2
fi

diffusion_nft_require_compute_allocation

if [[ -z "$DIFFUSION_NFT_CONDA_EXE" || ! -x "$DIFFUSION_NFT_CONDA_EXE" ]]; then
  echo "Could not find an executable Conda; set DIFFUSION_NFT_CONDA_EXE." >&2
  exit 1
fi
command -v sha256sum >/dev/null || { echo "sha256sum is required." >&2; exit 1; }

SOURCE_ENV="$DIFFUSION_NFT_CONDA_ENV"
TARGET_ENV="$DIFFUSION_NFT_OPTIONAL_REWARD_CONDA_ENV"
CONSTRAINTS="$SCRIPT_DIR/optional_reward_constraints.txt"
STATE_DIR="$DIFFUSION_NFT_ASSET_ROOT/environment/optional-rewards"
OWNER_MARKER="$STATE_DIR/owned-target.env"
MANIFEST="$STATE_DIR/manifest.sha256"
WHEELHOUSE="$DIFFUSION_NFT_ASSET_ROOT/wheelhouse/optional-rewards"
SOURCE_LOCK_ROOT="$DIFFUSION_NFT_SOURCE_ROOT/optional-rewards"
OPENAI_CLIP_REVISION=d50d76daa670286dd6cacf3bcd80b5e4823fc8e1
OPENAI_CLIP_SOURCE="$SOURCE_LOCK_ROOT/openai-clip-$OPENAI_CLIP_REVISION"
OPENAI_CLIP_BUILD_ROOT=""
OPENAI_CLIP_BUILD_SOURCE=""
IMAGEREWARD_PATCH_ROOT=""
IMAGEREWARD_INSTALL_PACKAGE=""
HPS_PACKAGE="$WHEELHOUSE/hpsv2x-1.2.0.tar.gz"
HPS_PACKAGE_SHA256=66220d040a34183e6cb53b22023a034278126f052506e54a9d89608d9e766f39
HPS_PACKAGE_URL=https://files.pythonhosted.org/packages/4e/87/7ff2e74350a04827802239e2146c38a27067ffcd8f73399131fa2fb5466e/hpsv2x-1.2.0.tar.gz
IMAGEREWARD_PACKAGE="$WHEELHOUSE/image_reward-1.5-py3-none-any.whl"
IMAGEREWARD_PACKAGE_SHA256=b4917a680c3caa68a1ce3847cebab3ae6d5b4c409d7f8804f537b9d42662cf12
IMAGEREWARD_PACKAGE_URL=https://files.pythonhosted.org/packages/ea/df/b2e66a6f93494ac43d2cfb77475e0321e062b765ec35a368a07decf22a1d/image_reward-1.5-py3-none-any.whl

SOURCE_RUN=("$DIFFUSION_NFT_CONDA_EXE" run --no-capture-output -n "$SOURCE_ENV")
TARGET_RUN=("$DIFFUSION_NFT_CONDA_EXE" run --no-capture-output -n "$TARGET_ENV")

cleanup_openai_clip_build_source() {
  [[ -n "$OPENAI_CLIP_BUILD_ROOT" && -d "$OPENAI_CLIP_BUILD_ROOT" ]] || return 0
  [[ "$(dirname "$OPENAI_CLIP_BUILD_ROOT")" == "${SLURM_TMPDIR:-/tmp}" ]] || {
    echo "Refusing to clean unexpected OpenAI CLIP build root: $OPENAI_CLIP_BUILD_ROOT" >&2
    return 1
  }
  [[ "$(basename "$OPENAI_CLIP_BUILD_ROOT")" == diffnft-openai-clip-build.* ]] || {
    echo "Refusing to clean unexpected OpenAI CLIP build root: $OPENAI_CLIP_BUILD_ROOT" >&2
    return 1
  }
  rm -rf -- "$OPENAI_CLIP_BUILD_ROOT"
  OPENAI_CLIP_BUILD_ROOT=""
  OPENAI_CLIP_BUILD_SOURCE=""
}

cleanup_imagereward_patch() {
  [[ -n "$IMAGEREWARD_PATCH_ROOT" && -d "$IMAGEREWARD_PATCH_ROOT" ]] || return 0
  [[ "$(dirname "$IMAGEREWARD_PATCH_ROOT")" == "${SLURM_TMPDIR:-/tmp}" ]] || {
    echo "Refusing to clean unexpected ImageReward patch root: $IMAGEREWARD_PATCH_ROOT" >&2
    return 1
  }
  [[ "$(basename "$IMAGEREWARD_PATCH_ROOT")" == diffnft-imagereward-wheel.* ]] || {
    echo "Refusing to clean unexpected ImageReward patch root: $IMAGEREWARD_PATCH_ROOT" >&2
    return 1
  }
  rm -rf -- "$IMAGEREWARD_PATCH_ROOT"
  IMAGEREWARD_PATCH_ROOT=""
  IMAGEREWARD_INSTALL_PACKAGE=""
}

cleanup_temporary_build_inputs() {
  local status=0
  cleanup_openai_clip_build_source || status=$?
  cleanup_imagereward_patch || status=$?
  return "$status"
}
trap cleanup_temporary_build_inputs EXIT

conda_env_exists() {
  "$DIFFUSION_NFT_CONDA_EXE" env list \
    | awk -v wanted="$1" '$1 == wanted { found = 1 } END { exit !found }'
}

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  [[ -s "$path" ]] || { echo "Missing or empty file: $path" >&2; return 1; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "Checksum mismatch: expected=$expected actual=$actual path=$path" >&2
    return 1
  }
}

download_exact() {
  local url="$1"
  local destination="$2"
  local expected_sha="$3"
  local temporary_file
  mkdir -p "$(dirname "$destination")"
  if [[ -s "$destination" ]]; then
    if [[ -n "$expected_sha" ]]; then
      verify_sha256 "$destination" "$expected_sha"
    fi
    return 0
  fi
  temporary_file="$(mktemp "${destination}.partial.XXXXXX")"
  if ! curl --fail --location --retry 5 --retry-all-errors \
    --output "$temporary_file" "$url"; then
    rm -f -- "$temporary_file"
    return 1
  fi
  if [[ -n "$expected_sha" ]] && ! verify_sha256 "$temporary_file" "$expected_sha"; then
    rm -f -- "$temporary_file"
    return 1
  fi
  mv "$temporary_file" "$destination"
}

checkout_openai_clip() {
  if [[ -d "$OPENAI_CLIP_SOURCE/.git" ]]; then
    local actual_revision
    actual_revision="$(git -C "$OPENAI_CLIP_SOURCE" rev-parse HEAD)"
    [[ "$actual_revision" == "$OPENAI_CLIP_REVISION" ]] || {
      echo "OpenAI CLIP source revision mismatch: $OPENAI_CLIP_SOURCE" >&2
      return 1
    }
    return 0
  fi
  [[ ! -e "$OPENAI_CLIP_SOURCE" ]] || {
    echo "Refusing non-Git OpenAI CLIP source path: $OPENAI_CLIP_SOURCE" >&2
    return 1
  }
  mkdir -p "$SOURCE_LOCK_ROOT"
  local temporary_root
  temporary_root="$(mktemp -d "$SOURCE_LOCK_ROOT/.openai-clip.XXXXXX")"
  if ! git clone --no-checkout https://github.com/openai/CLIP.git "$temporary_root/repo" \
    || ! git -C "$temporary_root/repo" checkout --detach "$OPENAI_CLIP_REVISION"; then
    rm -rf -- "$temporary_root"
    return 1
  fi
  [[ "$(git -C "$temporary_root/repo" rev-parse HEAD)" == "$OPENAI_CLIP_REVISION" ]] || {
    rm -rf -- "$temporary_root"
    echo "OpenAI CLIP checkout did not resolve to the requested revision." >&2
    return 1
  }
  mv "$temporary_root/repo" "$OPENAI_CLIP_SOURCE"
  rmdir "$temporary_root"
}

prepare_openai_clip_build_source() {
  cleanup_openai_clip_build_source
  OPENAI_CLIP_BUILD_ROOT="$(
    mktemp -d "${SLURM_TMPDIR:-/tmp}/diffnft-openai-clip-build.XXXXXX"
  )"
  OPENAI_CLIP_BUILD_SOURCE="$OPENAI_CLIP_BUILD_ROOT/source"
  mkdir -p "$OPENAI_CLIP_BUILD_SOURCE"
  git -C "$OPENAI_CLIP_SOURCE" archive --format=tar "$OPENAI_CLIP_REVISION" \
    | tar -xf - -C "$OPENAI_CLIP_BUILD_SOURCE"
  [[ -s "$OPENAI_CLIP_BUILD_SOURCE/setup.py" ]] || {
    echo "Pristine OpenAI CLIP archive is missing setup.py." >&2
    return 1
  }
}

prepare_imagereward_install_wheel() {
  cleanup_imagereward_patch
  IMAGEREWARD_PATCH_ROOT="$(
    mktemp -d "${SLURM_TMPDIR:-/tmp}/diffnft-imagereward-wheel.XXXXXX"
  )"
  IMAGEREWARD_INSTALL_PACKAGE="$IMAGEREWARD_PATCH_ROOT/$(basename "$IMAGEREWARD_PACKAGE")"
  "${SOURCE_RUN[@]}" python "$SCRIPT_DIR/rewrite_imagereward_wheel.py" \
    "$IMAGEREWARD_PACKAGE" "$IMAGEREWARD_INSTALL_PACKAGE"
  [[ -s "$IMAGEREWARD_INSTALL_PACKAGE" ]] || {
    echo "ImageReward compatibility wheel was not created." >&2
    return 1
  }
}

print_environment() {
  local label="$1"
  shift
  echo "$label"
  "$@" python -c \
    'import sys, torch, transformers, huggingface_hub; print(f"python={sys.executable}"); print(f"torch={torch.__version__} cuda={torch.version.cuda}"); print(f"transformers={transformers.__version__}"); print(f"huggingface_hub={huggingface_hub.__version__}")'
  "$@" python -m pip check
}

check_static_assets() {
  diffusion_nft_require_pickscore_assets
  diffusion_nft_require_clipscore_assets
  diffusion_nft_require_hpsv2_assets
  diffusion_nft_require_aesthetic_assets
  diffusion_nft_require_imagereward_assets
  verify_sha256 "$DIFFUSION_NFT_CLIPSCORE_MODEL_DIR/model.safetensors" \
    "$DIFFUSION_NFT_CLIPSCORE_MODEL_SHA256"
  [[ -s "$MANIFEST" ]] || { echo "Missing optional reward manifest: $MANIFEST" >&2; return 1; }
  sha256sum --check "$MANIFEST"
}

verify_optional_package_contract() {
  "${TARGET_RUN[@]}" python - <<'PY'
import importlib.metadata as metadata

import ImageReward
import clip
import hpsv2
import timm
from packaging.requirements import Requirement

image_reward_requirements = [
    Requirement(value) for value in metadata.requires("image-reward") or []
]
timm_requirements = [
    str(requirement.specifier)
    for requirement in image_reward_requirements
    if requirement.name.lower().replace("_", "-") == "timm"
]
if timm_requirements != [">=1.0.17"]:
    raise SystemExit(
        "Unexpected ImageReward timm metadata: " + repr(timm_requirements)
    )
if timm.__version__ != "1.0.17":
    raise SystemExit(f"Expected timm 1.0.17, got {timm.__version__}")
print(f"ImageReward={ImageReward.__file__}")
print(f"clip={clip.__file__}")
print(f"hpsv2={hpsv2.__file__}")
print(f"timm={timm.__version__} ImageReward-timm={timm_requirements[0]}")
PY
}

check_optional_environment() {
  [[ -s "$DIFFUSION_NFT_OPTIONAL_REWARD_ENV_MARKER" ]] || {
    echo "Missing ready marker: $DIFFUSION_NFT_OPTIONAL_REWARD_ENV_MARKER" >&2
    return 1
  }
  conda_env_exists "$TARGET_ENV" || {
    echo "Missing Conda environment: $TARGET_ENV" >&2
    return 1
  }
  print_environment "Optional reward environment ($TARGET_ENV)" "${TARGET_RUN[@]}"
  verify_optional_package_contract
}

conda_env_exists "$SOURCE_ENV" || {
  echo "Source Conda environment does not exist: $SOURCE_ENV" >&2
  exit 1
}
print_environment "Validated source environment before setup ($SOURCE_ENV)" "${SOURCE_RUN[@]}"

if [[ "$MODE" == "check" ]]; then
  check_static_assets
  check_optional_environment
  print_environment "Validated source environment after check ($SOURCE_ENV)" "${SOURCE_RUN[@]}"
  echo "Optional reward assets and environment passed read-only checks."
  exit 0
fi

command -v curl >/dev/null || { echo "curl is required for --prepare." >&2; exit 1; }
command -v git >/dev/null || { echo "git is required for --prepare." >&2; exit 1; }
command -v tar >/dev/null || { echo "tar is required for --prepare." >&2; exit 1; }
[[ -r "$CONSTRAINTS" ]] || { echo "Missing constraints file: $CONSTRAINTS" >&2; exit 1; }
mkdir -p "$DIFFUSION_NFT_REWARD_CKPT_DIR" "$WHEELHOUSE" "$STATE_DIR" \
  "$DIFFUSION_NFT_CLIPSCORE_CACHE_ROOT" "$HF_HUB_CACHE"
rm -f -- "$DIFFUSION_NFT_OPTIONAL_REWARD_ENV_MARKER"

SOURCE_FREEZE_BEFORE="$("${SOURCE_RUN[@]}" python -m pip freeze | sha256sum | awk '{print $1}')"

env HF_HOME="$DIFFUSION_NFT_SHARED_HF_ROOT" \
  HF_HUB_CACHE="$DIFFUSION_NFT_SHARED_HF_ROOT/hub" \
  "${SOURCE_RUN[@]}" huggingface-cli download openai/clip-vit-large-patch14 \
    config.json merges.txt model.safetensors preprocessor_config.json \
    special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json \
    --revision "$DIFFUSION_NFT_CLIPSCORE_REVISION" \
    --cache-dir "$DIFFUSION_NFT_SHARED_HF_ROOT/hub"
verify_sha256 "$DIFFUSION_NFT_CLIPSCORE_MODEL_DIR/model.safetensors" \
  "$DIFFUSION_NFT_CLIPSCORE_MODEL_SHA256"
mkdir -p "$DIFFUSION_NFT_CLIPSCORE_CACHE_ROOT/refs"
printf '%s\n' "$DIFFUSION_NFT_CLIPSCORE_REVISION" \
  > "$DIFFUSION_NFT_CLIPSCORE_CACHE_ROOT/refs/main"

BASELINE_CLIP_CACHE_ROOT="$HF_HUB_CACHE/models--openai--clip-vit-large-patch14"
if [[ -L "$BASELINE_CLIP_CACHE_ROOT" ]]; then
  [[ "$(realpath -e "$BASELINE_CLIP_CACHE_ROOT")" == "$(realpath -e "$DIFFUSION_NFT_CLIPSCORE_CACHE_ROOT")" ]] || {
    echo "Unexpected existing CLIP cache symlink: $BASELINE_CLIP_CACHE_ROOT" >&2
    exit 1
  }
elif [[ -e "$BASELINE_CLIP_CACHE_ROOT" ]]; then
  echo "Refusing to replace existing non-symlink CLIP cache: $BASELINE_CLIP_CACHE_ROOT" >&2
  exit 1
else
  ln -s "$DIFFUSION_NFT_CLIPSCORE_CACHE_ROOT" "$BASELINE_CLIP_CACHE_ROOT"
fi

env HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" \
  "${SOURCE_RUN[@]}" huggingface-cli download bert-base-uncased \
    config.json tokenizer.json tokenizer_config.json vocab.txt \
    --revision "$DIFFUSION_NFT_BERT_TOKENIZER_REVISION" \
    --cache-dir "$HF_HUB_CACHE"
mkdir -p "$DIFFUSION_NFT_BERT_TOKENIZER_CACHE_ROOT/refs"
printf '%s\n' "$DIFFUSION_NFT_BERT_TOKENIZER_REVISION" \
  > "$DIFFUSION_NFT_BERT_TOKENIZER_CACHE_ROOT/refs/main"

download_exact \
  "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/$DIFFUSION_NFT_HPS_BACKBONE_REVISION/open_clip_pytorch_model.bin?download=true" \
  "$DIFFUSION_NFT_HPS_BACKBONE_PATH" "$DIFFUSION_NFT_HPS_BACKBONE_SHA256"
download_exact \
  "https://huggingface.co/xswu/HPSv2/resolve/$DIFFUSION_NFT_HPSV21_REVISION/HPS_v2.1_compressed.pt?download=true" \
  "$DIFFUSION_NFT_HPSV21_PATH" "$DIFFUSION_NFT_HPSV21_SHA256"
download_exact \
  "https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/$DIFFUSION_NFT_AESTHETIC_REVISION/sac%2Blogos%2Bava1-l14-linearMSE.pth" \
  "$DIFFUSION_NFT_AESTHETIC_PATH" ""
download_exact \
  "https://huggingface.co/THUDM/ImageReward/resolve/$DIFFUSION_NFT_IMAGEREWARD_REVISION/ImageReward.pt?download=true" \
  "$DIFFUSION_NFT_IMAGEREWARD_PATH" "$DIFFUSION_NFT_IMAGEREWARD_SHA256"
download_exact \
  "https://huggingface.co/THUDM/ImageReward/resolve/$DIFFUSION_NFT_IMAGEREWARD_REVISION/med_config.json?download=true" \
  "$DIFFUSION_NFT_IMAGEREWARD_CONFIG_PATH" ""
download_exact "$HPS_PACKAGE_URL" "$HPS_PACKAGE" "$HPS_PACKAGE_SHA256"
download_exact "$IMAGEREWARD_PACKAGE_URL" "$IMAGEREWARD_PACKAGE" "$IMAGEREWARD_PACKAGE_SHA256"
checkout_openai_clip
prepare_openai_clip_build_source
prepare_imagereward_install_wheel

if ! conda_env_exists "$TARGET_ENV"; then
  printf 'SOURCE_ENV=%q\nTARGET_ENV=%q\nCREATED_BY=%q\n' \
    "$SOURCE_ENV" "$TARGET_ENV" "$0" > "$OWNER_MARKER"
  "$DIFFUSION_NFT_CONDA_EXE" create --yes --name "$TARGET_ENV" --clone "$SOURCE_ENV"
elif [[ ! -s "$OWNER_MARKER" ]]; then
  echo "Refusing to mutate pre-existing unowned Conda environment: $TARGET_ENV" >&2
  exit 1
fi

# OpenAI CLIP's pinned setup.py imports pkg_resources while pip is collecting
# build requirements.  Recent isolated setuptools build environments no longer
# guarantee that module, whereas the cloned target environment contains the
# pinned setuptools/pkg_resources pair.  Keep the entire constrained install in
# that environment.  Build from a fresh archive of the pinned revision so
# setup.py cannot reuse build/ or egg-info artifacts left in persistent group
# storage by an interrupted earlier installation. ImageReward 1.5 pins the
# obsolete timm==0.6.13 even though its imported compatibility modules remain
# available in timm 1.0.17. First force-install the exact-source wheel whose sole
# metadata rewrite is audited by rewrite_imagereward_wheel.py, then resolve the
# remaining dependencies against the project pin.
"${TARGET_RUN[@]}" python -m pip install --force-reinstall --no-deps \
  "$IMAGEREWARD_INSTALL_PACKAGE"
"${TARGET_RUN[@]}" python -m pip install --no-build-isolation \
  --constraint "$CONSTRAINTS" \
  "$HPS_PACKAGE" fairscale==0.4.13 "$OPENAI_CLIP_BUILD_SOURCE"
cleanup_openai_clip_build_source
cleanup_imagereward_patch
"${TARGET_RUN[@]}" python -m pip check
verify_optional_package_contract

SOURCE_FREEZE_AFTER="$("${SOURCE_RUN[@]}" python -m pip freeze | sha256sum | awk '{print $1}')"
[[ "$SOURCE_FREEZE_AFTER" == "$SOURCE_FREEZE_BEFORE" ]] || {
  echo "Source environment changed unexpectedly during isolated setup." >&2
  exit 1
}

"${TARGET_RUN[@]}" python -m pip freeze > "$STATE_DIR/target-pip-freeze.txt"
sha256sum \
  "$DIFFUSION_NFT_CLIPSCORE_MODEL_DIR/model.safetensors" \
  "$DIFFUSION_NFT_HPS_BACKBONE_PATH" \
  "$DIFFUSION_NFT_HPSV21_PATH" \
  "$DIFFUSION_NFT_AESTHETIC_PATH" \
  "$DIFFUSION_NFT_IMAGEREWARD_PATH" \
  "$DIFFUSION_NFT_IMAGEREWARD_CONFIG_PATH" \
  "$HPS_PACKAGE" "$IMAGEREWARD_PACKAGE" \
  "$STATE_DIR/target-pip-freeze.txt" > "$MANIFEST"

check_static_assets
print_environment "Optional reward environment after CPU setup ($TARGET_ENV)" "${TARGET_RUN[@]}"
print_environment "Validated source environment after setup ($SOURCE_ENV)" "${SOURCE_RUN[@]}"
printf 'READY_AT_UTC=%q\nSETUP_SCOPE=%q\nGPU_FORWARD_VALIDATION=%q\nIMAGEREWARD_TIMM_REQUIREMENT=%q\nSOURCE_ENV=%q\nTARGET_ENV=%q\nSOURCE_FREEZE_SHA256=%q\nOPENAI_CLIP_REVISION=%q\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" environment_and_static_assets deferred_to_capacity_smoke \
  'timm>=1.0.17' "$SOURCE_ENV" "$TARGET_ENV" \
  "$SOURCE_FREEZE_AFTER" "$OPENAI_CLIP_REVISION" \
  > "$DIFFUSION_NFT_OPTIONAL_REWARD_ENV_MARKER"
check_optional_environment
echo "Optional reward CPU setup completed without modifying $SOURCE_ENV."
