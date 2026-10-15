# SiT-XL/2 and Model Guidance on ImageNet 256

NFT and GuidanceNFT for class-conditional SiT-XL/2 with a class-name PickScore reward
(prompt `an image of {class_name}`).

* **Explicit CFG (§4.1).** The pretrained SiT-XL/2. NFT trains on CFG-1 rollouts;
  GuidanceNFT first fits its center for 100 updates and then trains on CFG-1.5 rollouts.
* **Built-in guidance (§4.2).** SiT-XL/2 trained with Model Guidance, sampled with
  external CFG 1. Both methods train on the model's native guided field.

## Setup

```bash
./setup_upstream.sh      # SiT model and sampler from tzco/Diffusion-wo-CFG @ 53c386f
export GUIDANCENFT_ASSETS=/path/to/assets
export PYTHONPATH=$PWD:$PWD/third_party/Diffusion-wo-CFG
```

Pretrained checkpoints (the configs check the SiT-XL/2 hash):

| File under `$GUIDANCENFT_ASSETS` | Source | SHA-256 |
|---|---|---|
| `checkpoints/sit/SiT-XL-2-256.pt` | [SiT release](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) | `e645b5cd82d1d645cfd9b7c96041b56b2a0aa61faed90c0869e79be02ab55aca` |
| `checkpoints/model_guidance/SiT-XL-2-MG.pt` | `third_party/Diffusion-wo-CFG/download.sh` (concatenates `SiT-XL-2-MG-{a..f}`) | `527ee98e5decd8f8cb099df1348195e50ff82106c13d350033d8f339b4a1e8cf` |

The PickScore model (`yuvalkirstain/PickScore_v1`), its CLIP-H processor and the
`stabilityai/sd-vae-ft-ema` VAE are loaded from the Hugging Face Hub at the revisions
pinned in `sit_cfg_nft/xl_cfg_nft.py` and the configs.

## Configs

| Config | Paper | Method |
|---|---|---|
| `configs/sit_xl_nft_cfg1.json` | Table "SiT-XL/2 PickScore", NFT rows | NFT, CFG-1 rollouts |
| `configs/sit_xl_guidancenft_cfg1p5.json` | same table, GuidanceNFT rows | GuidanceNFT, 100 center updates, then CFG-1.5 rollouts |
| `configs/mg_nft.json` | §4.2 Model Guidance, NFT | NFT on the built-in guided field |
| `configs/mg_guidancenft.json` | §4.2 Model Guidance, GuidanceNFT | GuidanceNFT on the built-in guided field |

All four use 1,152 rollouts per update (48 classes × 24 samples), 10-step SDE rollouts,
nine training time draws per endpoint, learning rate 3e-6, β = 1, reference coefficient
1e-4, and 400 policy updates. These are the exact configs of the paper runs; only the run
name and asset paths differ. The Model Guidance runs in the paper were trained to update
200 and then resumed to update 400 with identical settings.

## Run

```bash
scripts/smoke.sh sit_guidancenft          # sit_nft, sit_guidancenft, mg_nft, mg_guidancenft
scripts/train.sh configs/sit_xl_guidancenft_cfg1p5.json
```

`scripts/train.sh` validates the config and then launches `torchrun` on all visible GPUs
(the paper used two H200s). Evaluation of a saved checkpoint:

```bash
python evaluate_sit_xl_cfg_nft.py --config configs/sit_xl_guidancenft_cfg1p5.json \
  --checkpoint <run>/checkpoints/outer_000400.pt --output-dir <dir> --expected-step 400
python evaluate_diffusion_nft.py --config configs/mg_guidancenft.json \
  --checkpoint <run>/checkpoints/outer_000400.pt \
  --init-checkpoint $GUIDANCENFT_ASSETS/checkpoints/model_guidance/SiT-XL-2-MG.pt \
  --output-dir <dir> --num-classes 1000 --samples-per-class 2
```

Unit tests: `python -m unittest discover -s tests`.
