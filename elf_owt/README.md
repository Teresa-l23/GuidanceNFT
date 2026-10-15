# ELF-B on OpenWebText (built-in SCCFG)

NFT and GuidanceNFT for ELF-B, a continuous diffusion language model whose guidance is
built in through self-conditioning CFG (SCCFG). The reward is the GPT-2 Large
log-likelihood with an adaptive entropy floor. Implemented in JAX/Flax.

## Setup

```bash
./setup_upstream.sh   # lillian039/ELF @ 5098bf2 + patches/elf_src.patch
export GUIDANCENFT_ASSETS=/path/to/assets
export ELF_REPO_ROOT=$PWD/third_party/ELF
export PYTHONPATH=$ELF_REPO_ROOT/src:$PWD
```

The pretrained model `embedded-language-flows/ELF-B-owt` and GPT-2 Large are loaded from
the Hugging Face Hub.

## Configs (`baselines/elf_nft/configs/`)

| Config | Paper | Method |
|---|---|---|
| `owt_w1_conditional_only_standard_100_cfg_sweep.json` | SCCFG sweep figure, NFT | NFT, SCCFG-1 rollouts |
| `owt_w3_conditional_only_shifted_center_100.json` | SCCFG sweep figure, GuidanceNFT | GuidanceNFT, SCCFG-3 rollouts |
| `owt_sccfg1_randomreconstruct_100_cfg_sweep.json` | SCCFG sweep figure, LFRM-style | reconstruction of the base prediction, SCCFG-1 rollouts |
| `owt_w3_bernoulli_shifted_center_100_cfg_sweep.json`, `owt_w3_paired_shifted_center_100.json` | appendix, center branches | GuidanceNFT with other self-conditioning modes |
| `owt_w1_paired_standard_100.json`, `owt_w1_bernoulli_standard_100.json` | appendix, NFT controls | NFT with other self-conditioning modes |
| `owt_cfg3_recovered_center_w1_rollout_dual_100.json`, `owt_cfg3_recovered_center_constructed_rollout_dual_100.json` | appendix, recovered center | analytic recovered center |
| `owt_sccfg_adaptive_entropy_standard.json`, `owt_sccfg_adaptive_entropy_shifted_center.json` | repetition diagnostic | NFT vs GuidanceNFT at SCCFG 3 |

Each run uses 384 endpoints per update, eight time draws per endpoint, 100 updates,
learning rate 3e-5, β = 1, reference coefficient 0.1 and decoder cross-entropy
coefficient 0.25. The configs are the exact ones of the paper runs, with asset paths
replaced by `${GUIDANCENFT_ASSETS}`.

## Run

```bash
scripts/smoke.sh guidancenft      # or nft; two updates and a 32-sample evaluation
scripts/train.sh baselines/elf_nft/configs/owt_w3_conditional_only_shifted_center_100.json
```

The SCCFG sweep (1,024 samples per point, 32-step SDE, γ = 1.5) is
`baselines/elf_nft/scripts/evaluate_owt_cfg_sweep.py`, and the figure is drawn by
`baselines/elf_nft/scripts/plot_owt_cfg_sweep.py`.

The trainer also contains the WMT14 and XSum code paths of our development work; they are
not used in the paper and have no configs here.

Unit tests: `python -m unittest discover -s baselines/elf_nft/tests -t .`
