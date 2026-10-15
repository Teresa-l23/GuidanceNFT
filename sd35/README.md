# SD3.5-Medium with explicit CFG

LoRA fine-tuning of Stable Diffusion 3.5 Medium with NFT and GuidanceNFT, built on
NVIDIA's DiffusionNFT code (`baselines/diffusion_nft/upstream`, fetched and patched by
`setup_upstream.sh`). Our method lives in `baselines/diffusion_nft_sd35_v5/`:

* `v5flow.py`: the learned center ("v5"), fitted by flow matching on CFG-4 rollouts, with
  its initialization (100 fitting epochs with the policy frozen) and online refresh;
* `cfg_modes.py`: the rollout law and loss center of every compared method
  (`cfg1`, `cfg1_dual`, `cfg4_rollout`, `cfg4_shifted_center`, ...);
* `pretrained_null_cfg.py`: the pretrained-empty-branch control (inference only);
* `config/v5flow.py`: turns a JSON config into the trainer's `ml_collections` config.

## Setup

```bash
./setup_upstream.sh            # NVlabs/DiffusionNFT @ cbb14f8 + patches/diffusion_nft_upstream.patch
export GUIDANCENFT_ASSETS=/path/to/assets
bash baselines/diffusion_nft/scripts/setup_environment.sh --create   # conda env, pinned wheels
bash baselines/diffusion_nft/scripts/stage_geneval_assets.sh         # SD3.5-M, GenEval detector
bash baselines/diffusion_nft/scripts/stage_optional_reward_assets.sh # PickScore, HPS, CLIP, aesthetic, ImageReward, OCR
```

`baselines/diffusion_nft/scripts/assets_env.sh` lists every asset path and pinned
revision; SD3.5-Medium is pinned to revision `b940f670f0eda2d07fbb75229e779da1ad11eb80`.

## Configs (`baselines/diffusion_nft_sd35_v5/config/`)

Names encode the task, the per-GPU batch (`b24`), the number of prompt groups (`g48`),
the rollout steps, the CFG mode and the update budget.

| Paper | NFT (CFG 1) | NFT, both branches (CFG 1) | GuidanceNFT (CFG 4) |
|---|---|---|---|
| GenEval, update 200 | `cfg_geneval_b24_g48_steps10_cfg1_u200_eval50` | `cfg_geneval_b12_g48_steps10_cfg1_dual_u200_resume100` | `cfg_geneval_b24_g48_steps10_cfg4_v5flow_pretrain100_u200_eval100` |
| Learned3, update 200 | `cfg_learned3_b24_g48_steps10_cfg1_u200_eval25` | `cfg_learned3_b12_g48_steps10_cfg1_dual_u200_eval25` | `cfg_learned3_b24_g48_steps10_cfg4_v5flow_pretrain100_u200_eval25` |
| Learned3, update 300 | `cfg_learned3_b24_g48_steps10_cfg1_u400_eval25` | | `cfg_learned3_b24_g48_steps10_cfg4_v5flow_pretrain100_u400_eval25` |
| PickScore, update 100 | `cfg_pickscore_b24_g48_steps10_cfg1_u100_eval50` | `cfg_pickscore_b12_g48_steps10_cfg1_dual_u100_eval50` | `cfg_pickscore_b24_g48_steps10_cfg4_v5flow_pretrain100_u100_eval50` |
| PickScore, update 200 | `cfg_pickscore_b24_g48_steps10_cfg1_u200` | | `cfg_pickscore_b24_g48_steps10_cfg4_v5flow_pretrain100_u200` |
| OCR, update 50 | `cfg_ocr_b24_g48_steps10_cfg1_u100_eval50` | `cfg_ocr_b12_g48_steps10_cfg1_dual_u100_eval50` | `cfg_ocr_b24_g48_steps10_cfg4_v5flow_pretrain100_u100_eval50` |
| OCR, update 200 | `cfg_ocr_b24_g48_steps10_cfg1_u200` | | `cfg_ocr_b24_g48_steps10_cfg4_v5flow_pretrain100_u200` |

These are byte-identical to the configs of the paper runs (checked against the SHA-256
recorded by the evaluation manifests), except that absolute asset paths were replaced by
`${GUIDANCENFT_ASSETS}`.

Other configs:

* Center comparison under CFG-4 rollouts with five rollout steps (Figure 3):
  `cfg_geneval_b24_g48_steps5_{cfg1,cfg4_rollout,cfg4_shifted_center,cfg4_v5flow}_u200_eval100`
  (NFT at CFG 1, NFT with the old center, the analytic center, GuidanceNFT). The run
  directories of this comparison were not kept, so these four configs are matched to the
  paper by name only.
* Center refresh schedule (appendix): `cfg_pickscore_b24_g12_steps10_cfg4_v5flow_pretrained100_center{1x1,10x5,10x10,fixed}_u50_eval25`.
  They start from the PickScore GuidanceNFT run after center initialization
  (`pickscore_steps10_u100_54872044/train/checkpoints/checkpoint-0`, released with the
  checkpoints).
* Evaluation manifests: `four_family_three_seed_seven_reward_eval.json` (main table, three
  evaluation seeds, seven rewards), `step100_drawbench_paper_rewards.json` and
  `current_geneval200_learned3_drawbench_offline_rewards.json` (DrawBench).

## Run

```bash
scripts/smoke.sh guidancenft pickscore      # two updates, 8 prompt groups
scripts/train.sh baselines/diffusion_nft_sd35_v5/config/cfg_learned3_b24_g48_steps10_cfg4_v5flow_pretrain100_u200_eval25.json learned3-guidancenft
```

`scripts/train.sh` runs `torchrun` on two GPUs (set `NGPU` to change it; the configs
declare `world_size: 2`) and writes to `$GUIDANCENFT_ASSETS/runs/diffusion_nft_sd35_v5/<run>`.

The evaluation matrix of the main table is driven by
`baselines/diffusion_nft_sd35_v5/tools/run_three_seed_seven_reward_matrix.py`, one
coordinate (family, policy, CFG, seed) per call, with
`baselines/diffusion_nft/upstream/scripts/evaluation.py` as the evaluator.

Unit tests:

```bash
export PYTHONPATH=$PWD:$PWD/baselines/diffusion_nft/upstream
(cd baselines/diffusion_nft_sd35_v5/tests && python -m unittest discover)
```
