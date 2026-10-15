#!/usr/bin/env bash

# Build an isolated environment for the official DiffusionNFT SD3.5/GenEval
# baseline. The official pins conflict with the repository's `sit-nft`
# environment, so this script refuses to modify that environment in place.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/assets_env.sh"

usage() {
  cat <<'EOF'
Usage:
  bash baselines/diffusion_nft/scripts/setup_environment.sh --check
  bash baselines/diffusion_nft/scripts/setup_environment.sh --create

Both modes must run inside a Slurm compute allocation. `--check` is read-only
and reports official-pin conflicts plus the state of the isolated environment.
`--create` additionally requires at least one full H200 and a CUDA 12.4 nvcc.

The default target is `sit-nft-diffusionnft`; override it with
DIFFUSION_NFT_CONDA_ENV. The existing `sit-nft` environment is never changed.
The cluster CUDA toolkit defaults to `/opt/apps/rhel9/cuda-12.4`; override it
with DIFFUSION_NFT_CUDA_HOME when the site installation moves.

The mmcv, mmdetection, open_clip, and CLIP_benchmark commits are exact
2026-08-13 reproduction locks, not revisions reported by the paper authors.
EOF
}

case "${1:-}" in
  --check) MODE="check" ;;
  --create) MODE="create" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

diffusion_nft_require_compute_allocation

if [[ -z "$DIFFUSION_NFT_CONDA_EXE" || ! -x "$DIFFUSION_NFT_CONDA_EXE" ]]; then
  echo "Could not find an executable Conda; set DIFFUSION_NFT_CONDA_EXE." >&2
  exit 1
fi

conda_env_exists() {
  "$DIFFUSION_NFT_CONDA_EXE" env list \
    | awk -v wanted="$1" '$1 == wanted { found = 1 } END { exit !found }'
}

print_pin_comparison() {
  local env_name="$1"
  local label="$2"
  local conda_listing
  conda_listing="$("$DIFFUSION_NFT_CONDA_EXE" list -n "$env_name" 2>/dev/null)"
  echo
  echo "$label ($env_name)"
  printf '%-20s %-18s %-18s %s\n' "package" "installed" "official pin" "state"
  local row package expected installed state
  local rows=(
    "python|3.10.16"
    "torch|2.6.0"
    "torchvision|0.21.0"
    "transformers|4.40.0"
    "accelerate|1.4.0"
    "diffusers|0.33.1"
    "huggingface-hub|0.29.1"
    "numpy|1.26.4"
    "pillow|10.4.0"
    "wandb|0.18.7"
    "peft|0.10.0"
    "bitsandbytes|0.45.3"
    "deepspeed|0.16.4"
    "flash-attn|2.7.4.post1"
  )
  for row in "${rows[@]}"; do
    IFS='|' read -r package expected <<< "$row"
    installed="$(awk -v wanted="$package" '$1 == wanted { value = $2 } END { print value }' <<< "$conda_listing")"
    if [[ -z "$installed" && "$package" == "torch" ]]; then
      installed="$(awk '$1 == "pytorch" { value = $2 } END { print value }' <<< "$conda_listing")"
    fi
    if [[ -z "$installed" ]]; then
      installed="<missing>"
      state="MISSING"
    elif [[ "$installed" == "$expected" || "$installed" == "$expected"+* ]]; then
      state="OK"
    else
      state="CONFLICT"
    fi
    printf '%-20s %-18s %-18s %s\n' "$package" "$installed" "$expected" "$state"
  done
}

echo "Official DiffusionNFT environment conflicts with the project environment:"
echo "  official: torch=2.6.0, torchvision=0.21.0, diffusers=0.33.1"
echo "  project:  torch=2.5.1, torchvision=0.20.1, diffusers>=0.30,<1"
echo "Target isolation environment: $DIFFUSION_NFT_CONDA_ENV"
echo 'No command in this script updates or removes the `sit-nft` environment.'

if conda_env_exists sit-nft; then
  print_pin_comparison sit-nft "Existing project environment (informational only)"
else
  echo 'The project environment `sit-nft` is not present in this Conda installation.'
fi

if conda_env_exists "$DIFFUSION_NFT_CONDA_ENV"; then
  print_pin_comparison "$DIFFUSION_NFT_CONDA_ENV" "Isolated DiffusionNFT environment"
else
  echo
  echo "Isolated environment is not present: $DIFFUSION_NFT_CONDA_ENV"
fi

echo
echo "Pinned GenEval source revisions (local 2026-08-13 lock):"
echo "  mmcv=$DIFFUSION_NFT_MMCV_REVISION"
echo "  mmdetection=$DIFFUSION_NFT_MMDET_REVISION"
echo "  open_clip=$DIFFUSION_NFT_OPEN_CLIP_REVISION"
echo "  CLIP_benchmark=$DIFFUSION_NFT_CLIP_BENCHMARK_REVISION"

if [[ "$MODE" == "check" ]]; then
  exit 0
fi

if [[ "$DIFFUSION_NFT_CONDA_ENV" == "sit-nft" ]]; then
  echo "Refusing --create with DIFFUSION_NFT_CONDA_ENV=sit-nft." >&2
  echo "The official pins are isolated intentionally; choose another environment name." >&2
  exit 1
fi
if conda_env_exists "$DIFFUSION_NFT_CONDA_ENV"; then
  echo "Refusing to mutate the existing environment $DIFFUSION_NFT_CONDA_ENV." >&2
  echo "Use --check to inspect it, or choose a new DIFFUSION_NFT_CONDA_ENV name." >&2
  exit 1
fi

command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is required for --create." >&2
  exit 1
}
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if (( ${#gpu_names[@]} < 1 )); then
  echo "--create requires a GPU allocation; no allocated GPU was detected." >&2
  exit 1
fi
for gpu_name in "${gpu_names[@]}"; do
  if [[ "$gpu_name" != *H200* ]]; then
    echo "--create requires full H200 GPUs; detected: $gpu_name" >&2
    exit 1
  fi
done

DIFFUSION_NFT_CUDA_HOME="${DIFFUSION_NFT_CUDA_HOME:-/opt/apps/rhel9/cuda-12.4}"
if ! command -v nvcc >/dev/null && [[ -x "$DIFFUSION_NFT_CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME="$DIFFUSION_NFT_CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
command -v nvcc >/dev/null || {
  echo "nvcc is required to compile the pinned mmcv CUDA ops and flash-attn." >&2
  echo "Expected it under $DIFFUSION_NFT_CUDA_HOME; set DIFFUSION_NFT_CUDA_HOME if needed." >&2
  exit 1
}
nvcc_release="$(nvcc --version | awk '/release/ {gsub(/,/, "", $6); sub(/^V/, "", $6); print $6; exit}')"
if [[ "$nvcc_release" != 12.4* ]]; then
  echo "This cluster reproduction uses the PyTorch cu124 wheel, but nvcc reports $nvcc_release." >&2
  echo "Refusing a potentially ABI-incompatible mmcv build." >&2
  exit 1
fi

mkdir -p "$DIFFUSION_NFT_SOURCE_ROOT" "$DIFFUSION_NFT_ASSET_ROOT/environment"

checkout_exact_source() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local destination="$DIFFUSION_NFT_SOURCE_ROOT/${name}-${revision}"

  if [[ -d "$destination/.git" ]]; then
    local actual_revision
    actual_revision="$(git -C "$destination" rev-parse HEAD)"
    if [[ "$actual_revision" != "$revision" ]]; then
      echo "Pinned source directory has unexpected HEAD: $destination" >&2
      echo "expected=$revision actual=$actual_revision" >&2
      return 1
    fi
    printf '%s\n' "$destination"
    return 0
  fi
  if [[ -e "$destination" ]]; then
    echo "Refusing to overwrite non-Git path: $destination" >&2
    return 1
  fi

  local temporary_root
  temporary_root="$(mktemp -d "$DIFFUSION_NFT_SOURCE_ROOT/.${name}.XXXXXX")"
  if ! git clone --no-checkout "$url" "$temporary_root/repo"; then
    rm -rf "$temporary_root"
    return 1
  fi
  if ! git -C "$temporary_root/repo" checkout --detach "$revision"; then
    rm -rf "$temporary_root"
    return 1
  fi
  if [[ "$(git -C "$temporary_root/repo" rev-parse HEAD)" != "$revision" ]]; then
    rm -rf "$temporary_root"
    echo "Git checkout did not resolve to the requested revision for $name." >&2
    return 1
  fi
  mv "$temporary_root/repo" "$destination"
  rmdir "$temporary_root"
  printf '%s\n' "$destination"
}

echo "Creating isolated environment $DIFFUSION_NFT_CONDA_ENV ..."
"$DIFFUSION_NFT_CONDA_EXE" create --yes -n "$DIFFUSION_NFT_CONDA_ENV" python=3.10.16 pip

run_python() {
  PYTHONPATH="$DIFFUSION_NFT_UPSTREAM_ROOT" \
    "$DIFFUSION_NFT_CONDA_EXE" run --no-capture-output -n "$DIFFUSION_NFT_CONDA_ENV" python "$@"
}

run_python -m pip install pip==24.3.1
run_python -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple \
  torch==2.6.0 torchvision==0.21.0

runtime_pins=(
  transformers==4.40.0
  accelerate==1.4.0
  diffusers==0.33.1
  numpy==1.26.4
  pandas==2.2.3
  scipy==1.15.2
  scikit-learn==1.6.1
  scikit-image==0.25.2
  albumentations==1.4.10
  opencv-python==4.11.0.86
  pillow==10.4.0
  tqdm==4.67.1
  wandb==0.18.7
  pydantic==2.10.6
  requests==2.32.3
  matplotlib==3.10.0
  deepspeed==0.16.4
  peft==0.10.0
  bitsandbytes==0.45.3
  aiohttp==3.11.13
  fastapi==0.115.11
  uvicorn==0.34.0
  huggingface-hub==0.29.1
  safetensors==0.5.2
  datasets==3.3.2
  fsspec[http]==2024.12.0
  tokenizers==0.19.1
  einops==0.8.1
  nvidia-ml-py==12.570.86
  absl-py==2.1.0
  ml-collections==1.0.0
  sentencepiece==0.2.0
  timm==1.0.17
  ftfy==6.3.1
  regex==2024.11.6
  pycocotools==2.0.8
  pycocoevalcap==1.2
  webdataset==0.2.111
  braceexpand==0.1.7
  protobuf==5.29.3
  six==1.17.0
  terminaltables==3.1.10
  addict==2.4.0
  yapf==0.40.2
  pyyaml==6.0.2
  packaging==24.2
  openmim==0.3.9
  mmengine==0.10.7
  ninja==1.11.1.3
  wheel==0.45.1
  setuptools==75.8.0
  pdm-backend==2.4.3
)
run_python -m pip install "${runtime_pins[@]}"

# xformers is unpinned in the official setup.py. This exact wheel is our local
# compatibility lock for torch 2.6.0; it is recorded separately from author pins.
run_python -m pip install --no-deps xformers==0.0.29.post3

mmcv_source="$(checkout_exact_source mmcv https://github.com/open-mmlab/mmcv.git "$DIFFUSION_NFT_MMCV_REVISION")"
mmdet_source="$(checkout_exact_source mmdetection https://github.com/open-mmlab/mmdetection.git "$DIFFUSION_NFT_MMDET_REVISION")"
open_clip_source="$(checkout_exact_source open_clip https://github.com/mlfoundations/open_clip.git "$DIFFUSION_NFT_OPEN_CLIP_REVISION")"
clip_benchmark_source="$(checkout_exact_source CLIP_benchmark https://github.com/LAION-AI/CLIP_benchmark.git "$DIFFUSION_NFT_CLIP_BENCHMARK_REVISION")"

export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}"

run_python -m pip install --no-build-isolation flash-attn==2.7.4.post1
run_python -m pip install --no-build-isolation --no-deps -e "$mmcv_source"
run_python -m pip install --no-deps -e "$mmdet_source"
run_python -m pip install --no-build-isolation --no-deps "$open_clip_source"
run_python -m pip install --no-deps "$clip_benchmark_source"
run_python -m pip install --no-deps -e "$DIFFUSION_NFT_UPSTREAM_ROOT"
run_python -m pip check

run_python - <<'PY'
from pathlib import Path

import clip_benchmark
import diffusers
import flow_grpo
import mmcv
import mmdet
import open_clip
import peft
import torch
import torchvision
import transformers

geneval_config = (
    Path(mmdet.__file__).resolve().parent.parent
    / "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
)
if not geneval_config.is_file():
    raise SystemExit(f"editable mmdetection install does not expose GenEval config: {geneval_config}")
print(
    {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "mmcv": mmcv.__version__,
        "mmdet": mmdet.__version__,
        "geneval_config": str(geneval_config),
    }
)
PY

lock_prefix="$DIFFUSION_NFT_ASSET_ROOT/environment/$DIFFUSION_NFT_CONDA_ENV"
{
  printf 'CREATED_AT_UTC=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'DIFFUSION_NFT_UPSTREAM_COMMIT=%q\n' "$DIFFUSION_NFT_UPSTREAM_COMMIT"
  printf 'DIFFUSION_NFT_CONDA_ENV=%q\n' "$DIFFUSION_NFT_CONDA_ENV"
  printf 'CUDA_HOME=%q\n' "$CUDA_HOME"
  printf 'NVCC_RELEASE=%q\n' "$nvcc_release"
  printf 'PYTORCH_CUDA_WHEEL=%q\n' "cu124"
  printf 'MMCV_REVISION=%q\n' "$DIFFUSION_NFT_MMCV_REVISION"
  printf 'MMDET_REVISION=%q\n' "$DIFFUSION_NFT_MMDET_REVISION"
  printf 'OPEN_CLIP_REVISION=%q\n' "$DIFFUSION_NFT_OPEN_CLIP_REVISION"
  printf 'CLIP_BENCHMARK_REVISION=%q\n' "$DIFFUSION_NFT_CLIP_BENCHMARK_REVISION"
  printf 'XFORMERS_LOCAL_LOCK=%q\n' "0.0.29.post3"
} > "${lock_prefix}.setup.env"
"$DIFFUSION_NFT_CONDA_EXE" list -n "$DIFFUSION_NFT_CONDA_ENV" --explicit > "${lock_prefix}.conda-explicit.txt"
run_python -m pip freeze --all > "${lock_prefix}.pip-freeze.txt"

echo "Created isolated environment: $DIFFUSION_NFT_CONDA_ENV"
echo "Environment provenance: ${lock_prefix}.setup.env"
echo "Run --check in a compute allocation to review the installed pins."
