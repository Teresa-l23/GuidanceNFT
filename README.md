# GuidanceNFT

Code for **GuidanceNFT: NFT-Based Reward Fine-Tuning of Guided Diffusion Models**.

DiffusionNFT (NFT) fine-tunes a diffusion model by regressing its velocity toward a
reward-weighted displacement around a reference velocity. When rollouts are drawn from a
guided sampler (explicit CFG, or guidance built into the model), the reference used by
NFT no longer matches the score of the rollout distribution, and the update is biased.
GuidanceNFT fits the matching reference ("center") online by flow matching on the guided
samples and uses it in both regression terms of the NFT loss.

## Repository layout

| Directory | Paper section | What it contains |
|---|---|---|
| [`toy/`](toy) | Fig. 1, method figure, appendix reference-error fields | 2D tree toy: NFT with three objectives, and oracle iterated NFT updates under CFG 4 for each loss center |
| [`sd35/`](sd35) | §4.1, explicit CFG on SD3.5-Medium | LoRA fine-tuning with NFT and GuidanceNFT on GenEval, OCR, PickScore and Learned3; evaluation matrix |
| [`sit_imagenet/`](sit_imagenet) | §4.1 SiT-XL/2 (explicit CFG); §4.2 Model Guidance (built-in guidance) | Class-conditional ImageNet 256 with a class-name PickScore reward |
| [`elf_owt/`](elf_owt) | §4.2, ELF-B on OpenWebText | Native self-conditioning CFG (SCCFG) in a continuous diffusion language model |

GSM8K reasoning experiments (§4.2) will be released separately.

Each directory is self-contained: it has its own `setup_upstream.sh`, which fetches the
third-party code it builds on at a pinned commit and applies our patch, and a `scripts/`
folder with training and smoke-test launchers. Third-party code is not redistributed
except where its license allows; see [THIRD_PARTY.md](THIRD_PARTY.md).

## Common setup

All scripts read and write large files under one asset root:

```bash
export GUIDANCENFT_ASSETS=/path/to/assets   # checkpoints/, models/, runs/, caches
```

Pretrained weights are expected at the following paths (each subdirectory's README
gives download commands and checksums):

| Model | Path under `$GUIDANCENFT_ASSETS` |
|---|---|
| SiT-XL/2 256 | `checkpoints/sit/SiT-XL-2-256.pt` |
| SiT-XL/2 with Model Guidance | `checkpoints/model_guidance/SiT-XL-2-MG.pt` |
| SD3.5-Medium (`b940f67`) | `diffusion_nft/<commit>/models/stable-diffusion-3.5-medium/<revision>` |
| Autoguidance toy models | `models/autoguidance-toy/*.pkl` (downloaded by `toy/setup_upstream.sh`) |
| ELF-B (OpenWebText) | Hugging Face `embedded-language-flows/ELF-B-owt` |

The paper's runs used two H200 GPUs per job (one for the toy).

## Quick start

```bash
# Toy figures, about 15 minutes on one GPU
cd toy && ./setup_upstream.sh && scripts/reproduce_paper_figures.sh

# SiT-XL/2 (explicit CFG) and Model Guidance (built-in guidance)
cd sit_imagenet && ./setup_upstream.sh
scripts/smoke.sh sit_guidancenft                         # 2-GPU pipeline check
scripts/train.sh configs/sit_xl_guidancenft_cfg1p5.json  # paper run, 400 updates

# SD3.5-Medium
cd sd35 && ./setup_upstream.sh
scripts/smoke.sh guidancenft pickscore
scripts/train.sh baselines/diffusion_nft_sd35_v5/config/cfg_learned3_b24_g48_steps10_cfg4_v5flow_pretrain100_u200_eval25.json learned3-guidancenft

# ELF-B on OpenWebText
cd elf_owt && ./setup_upstream.sh
scripts/smoke.sh guidancenft
scripts/train.sh baselines/elf_nft/configs/owt_w3_conditional_only_shifted_center_100.json
```

## Checkpoints

The fine-tuned checkpoints reported in the paper are on the Hugging Face Hub at
[jiayin-l23/GuidanceNFT](https://huggingface.co/jiayin-l23/GuidanceNFT): SiT-XL/2 and Model
Guidance weights, SD3.5-Medium LoRA adapters and ELF-B checkpoints, each folder with the
SHA-256 of its files. `sit_imagenet/evaluate_released.py` evaluates the ImageNet weights with
the paper protocol.

## License

Our code is released under the MIT License ([LICENSE](LICENSE)). Patches to third-party
files keep the license of the file they modify; see [THIRD_PARTY.md](THIRD_PARTY.md).
