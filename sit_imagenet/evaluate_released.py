#!/usr/bin/env python3
"""Evaluate a released checkpoint with the paper protocol.

Three evaluation seeds, each covering all 1,000 ImageNet classes with two images per class,
ten-step SDE sampling, class-name PickScore (unscaled). SiT-XL/2 checkpoints are evaluated
at CFG 1 and 1.5; Model Guidance checkpoints at external CFG 1.

Usage:
  python evaluate_released.py --config configs/sit_xl_guidancenft_cfg1p5.json \
      --weights <release>/sit_guidancenft_cfg1p5_step400/model.safetensors --output-dir <dir>
"""
import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file

import evaluate_diffusion_nft as mg
import evaluate_sit_xl_cfg_nft as sit

SEEDS = (20240602507, 20240602508, 20240602509)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="configs/sit_xl_*.json or configs/mg_*.json")
    parser.add_argument("--weights", required=True, help="model.safetensors of a released checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    is_mg = Path(args.config).name.startswith("mg_")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    state = load_file(args.weights)
    if is_mg:
        config = mg.load_config(args.config)
        model = mg.build_model(config, state).to(device).eval().requires_grad_(False)
        reward = mg.build_reward(config.reward, device)
        cfg_scales = (1.0,)
    else:
        config = sit.load_xl_cfg_nft_config(args.config)
        model = sit.build_model(config, state, device=device)
        reward = sit.PickScoreImageNetReward(
            device=device, dtype=torch.float32, processor_name=config.reward.processor_name,
            model_name=config.reward.model_name, prompt_template=config.reward.prompt_template,
            divide_by_26=True, cache_text_embeddings=True, return_raw=False, batch_size=8,
            local_files_only=args.local_files_only)
        cfg_scales = (1.0, 1.5)
    del state
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-ema", local_files_only=args.local_files_only
    ).to(device).eval().requires_grad_(False)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        classes = mg.fixed_class_ids(count=1000, num_classes=1000, seed=seed)
        labels = [c for c in classes for _ in range(2)]
        for cfg in cfg_scales:
            directory = output / f"seed_{seed}_cfg_{cfg}"
            directory.mkdir(exist_ok=True)
            if is_mg:
                eval_args = SimpleNamespace(seed=seed, batch_size=8, sampler="sde", sampling_steps=10, preview_images=8)
                scores = mg.evaluate_policy(name="current", model=model, config=config, vae=vae, reward_fn=reward,
                                            labels_global=labels, args=eval_args, device=device, output_dir=directory)
            else:
                scores = sit.evaluate_policy(name="current", model=model, config=config, vae=vae, reward_fn=reward,
                                             labels_global=labels, seed=seed, cfg_scale=cfg, device=device,
                                             output_dir=directory)
            rows.append(dict(seed=seed, cfg=cfg, pickscore=scores["raw"]["mean"], num_samples=scores["raw"]["num_samples"]))
            print(json.dumps(rows[-1]), flush=True)

    summary = {}
    for cfg in cfg_scales:
        means = [row["pickscore"] for row in rows if row["cfg"] == cfg]
        summary[str(cfg)] = dict(mean=statistics.mean(means), sample_std=statistics.stdev(means), seed_means=means)
    result = dict(config=str(args.config), weights=str(args.weights), seeds=list(SEEDS), rows=rows, pickscore=summary)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
