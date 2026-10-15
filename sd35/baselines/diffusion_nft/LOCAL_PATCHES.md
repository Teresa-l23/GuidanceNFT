# Changes to DiffusionNFT

`upstream/` is [NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) at the commit
in `UPSTREAM_COMMIT.txt` (Apache-2.0), with `../../patches/diffusion_nft_upstream.patch`
applied. With a config that enables none of the options below, the trainer keeps the
released NFT loss, reward definitions and arithmetic order.

## Offline reward models

`flow_grpo/reward_ckpt_path.py`, `aesthetic_scorer.py`, `clip_scorer.py`, `gen_eval.py`,
`hpsv2_scorer.py`, `imagereward_scorer.py`, `pickscore_scorer.py`, `ocr.py`

* `DIFFUSION_NFT_REWARD_CKPT_DIR` points to locally staged reward checkpoints; a missing
  file raises before the reward model is built.
* With `DIFFUSION_NFT_OFFLINE` (or `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`) set, every
  reward model loads from local files only, including ImageReward's BLIP tokenizer.
* GenEval loads OpenAI's ViT-L/14 TorchScript archive by extracting its state dict, which
  PyTorch 2.6 otherwise rejects, and runs its colour classifier without DataLoader workers
  because rewards are computed in a worker thread during training.

## Base model loading and bounded training

`config/base.py`, `config/nft.py`, `scripts/train_nft_sd3.py`

* Local, revision-pinned SD3.5 snapshots (`pretrained.local_files_only`).
* An optional cap on optimizer steps (`train.max_train_steps`) and an optional check of the
  distributed world size used to derive the batch geometry.
* Complete-checkpoint markers, rolling checkpoints and exact resume (RNG state per rank).
* TensorBoard and JSONL scalar logging (`flow_grpo/metric_logging.py`, new file).

## CFG objectives and paired inline evaluation

`scripts/train_nft_sd3.py`, `flow_grpo/diffusers_patch/pipeline_with_logprob.py`,
`flow_grpo/diffusers_patch/solver.py`

* `cfg_mode` selects the rollout guidance scale and the loss center of the compared
  methods (`cfg1`, `cfg1_dual`, `cfg1_dropout`, `cfg4_rollout`, `cfg4_shifted_center`,
  `cfg4_full`, `cfg4_dual_shifted_center`, `cfg4_alternating`); the tensor algebra lives in
  `../diffusion_nft_sd35_v5/cfg_modes.py`.
* `cfg_unconditional_source="pretrained_base"` routes the unconditional CFG branch to the
  base model (`../diffusion_nft_sd35_v5/pretrained_null_cfg.py`).
* Inline evaluation at several CFG scales with paired initial noise, and per-step preview
  sheets (`../diffusion_nft_sd35_v5/inline_preview.py`).

## Learned center (GuidanceNFT)

`scripts/train_nft_sd3.py`

When `v5flow.enabled` is set, the trainer loads `../diffusion_nft_sd35_v5/v5flow.py`, adds
a LoRA adapter for the center (named `v5`), fits it by flow matching on the guided rollout
endpoints before each policy update (stage 1), and uses it as the center of the NFT loss
(stage 2). Center initialization, refresh interval and EMA are configured in the
`v5flow` section of the JSON config.

## Deterministic evaluation

`scripts/evaluation.py`

Non-padding distributed sharding with a check for missing or duplicate sample IDs,
per-sample generator seeds `(seed + sample_id) mod (2^63 - 1)`, resumable rank-local
results, and an optional deterministic DPM solver matching the training rollouts.

## Environment

`scripts/setup_environment.sh` builds the environment with the authors' package pins,
using the `cu124` PyTorch wheels (`torch==2.6.0`, `torchvision==0.21.0`) and compiling
flash-attn and MMCV against CUDA 12.4.
