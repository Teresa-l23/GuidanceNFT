# GSM8K reasoning with ELF-B (built-in SCCFG)

NFT and GuidanceNFT for the Qwen-based ELF-B reasoning model of
[CEDR](https://github.com/chengxiang/CEDR) (Reasoning with Continuous Latent Diffusion).
Guidance is built into the model through self-conditioning CFG (SCCFG); rollouts, the
loss and evaluation all use SCCFG 2. The reward is exact match of the final numerical
answer on the GSM8K training questions.

## What is here

The code is CEDR at a pinned commit plus `patches/cedr.patch`, which adds:

- generated-only NFT: endpoints are the model's own rollouts, groups with both correct and
  incorrect answers are kept, and training times are drawn from the rollout time grid;
- the GuidanceNFT center (`nft.center: learned`): a copy of the policy fitted online by
  flow matching on the SCCFG-2 rollouts, after 100 pretraining updates, and used as the
  center of both NFT terms;
- `LearnedEmbeddingPromptEncoder`, the prompt encoder of the released checkpoints, so that
  `cedr generate` can load them;
- the recipes `configs/gsm8k_b_nft.yaml` and `configs/gsm8k_b_guidancenft.yaml`.

**Provenance of the paper numbers.** The paper's runs were trained with the original ELF-B
training code, before CEDR was released, from an epoch-18 ELF-B GSM8K checkpoint whose
prompt encoder learns its own token embedding. The public CEDR package `gsm8k-b-pre-nft` is
a different epoch-18 checkpoint of the same model, with a frozen Qwen embedding. The
released fine-tuned weights below are exactly those of the paper runs. The trainer in this
directory implements the same objectives in CEDR and starts from `gsm8k-b-pre-nft`, so
retraining with it starts from a different model and does not reproduce the paper's
numbers exactly. The Pretrained and LFRM rows of the paper's GSM8K table are the public
`gsm8k-b-pre-nft` and `gsm8k-b-post-nft` packages.

## Setup

```bash
./setup_upstream.sh                        # chengxiang/CEDR @ 538198d + patches/cedr.patch
pip install -e 'third_party/CEDR[data,math]'
export GUIDANCENFT_ASSETS=/path/to/assets  # outputs go to $GUIDANCENFT_ASSETS/gsm8k
scripts/prepare_data.sh
```

`prepare_data.sh` downloads `gsm8k-b-pre-nft` and `gsm8k-b-post-nft` from
[xc91/CEDR](https://huggingface.co/xc91/CEDR), the GSM8K train split (7,473 questions,
`openai/gsm8k` revision `740312a`) as the reward population, and the 1,319 test questions.

## Recipes (`third_party/CEDR/configs/`)

| Recipe | Paper row | Center | Time draws per endpoint |
|---|---|---|---|
| `gsm8k_b_nft.yaml` | NFT | old policy | all 31 interior rollout times |
| `gsm8k_b_guidancenft.yaml` | GuidanceNFT | learned center, 100 pretraining updates, EMA 0.9 | 16 of 31 |

Both use 24 prompts per update with 15 rollouts each (32 steps, synchronous clocks,
SCCFG 2), generated endpoints only, learning rate 1e-4, β = 1, reference coefficient 0.1,
an old policy that tracks the policy with EMA decay min(0.001u, 0.5) at update u and
0.4 from update 400, and 500 updates. The paper's
GuidanceNFT run chose the 16 times so that two consecutive updates covered all 31; the CEDR
recipe draws them uniformly for each endpoint.

## Run

```bash
scripts/smoke.sh guidancenft    # or nft; two updates on 4 groups per GPU, no evaluation
scripts/train.sh guidancenft    # 500 updates on all visible GPUs; resumes from latest.pt
```

Every 50 updates the trainer writes `eval_update_XXXX.pt`, the EMA 0.99 policy and prompt
encoder, which `cedr generate` loads directly.

## Evaluation

```bash
scripts/evaluate.sh $GUIDANCENFT_ASSETS/gsm8k/checkpoints/gsm8k_guidancenft_step500 out/guidancenft
```

This scores the 1,319 test questions with the package's inference settings (64 steps,
asynchronous clocks (2.5, 2, 1.5), SCCFG 2, answers capped at 767 tokens) for the paper's
eight seeds. Pass@k over the seeds follows the paper's unbiased estimator.

## Checkpoints

The two fine-tuned models of the paper's GSM8K table are CEDR inference packages in
[jiayin-l23/GuidanceNFT](https://huggingface.co/jiayin-l23/GuidanceNFT) (ELF and prompt
encoder EMA 0.99). `scripts/download_checkpoints.sh` fetches them into
`$GUIDANCENFT_ASSETS/gsm8k/checkpoints` and adds the shared tokenizer files from the CEDR
packages.

| Folder | Method | Update | Pass@1 | Pass@2 | Pass@4 | Pass@8 |
|---|---|---|---|---|---|---|
| `gsm8k_nft_step200` | NFT | 200 | 41.39 ± 1.07 | 51.48 | 59.95 | 67.17 |
| `gsm8k_guidancenft_step500` | GuidanceNFT | 500 | 43.04 ± 1.10 | 52.75 | 61.31 | 68.61 |

The values are `scripts/evaluate.sh` results over the eight seeds, with problem-bootstrap
std (4,000 resamples). The paper reports 41.68 / 51.53 / 59.78 / 67.02 for NFT and
43.03 / 52.66 / 61.24 / 68.61 for GuidanceNFT. GuidanceNFT matches the paper on seven of
the eight seeds exactly. The paper's NFT numbers come from the evaluation built into the
original training code, a separate implementation of the same protocol, so per-seed
counts differ while the mean Pass@1 is within 0.3 points.
