#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torchvision
from diffusers import AutoencoderKL

from diffusion_nft.config import NFTConfig, load_config
from diffusion_nft.rewards import build_reward
from sampler import euler_maruyama_sampler, euler_sampler
from train_diffusion_nft import _unwrap_checkpoint, autocast_context, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-seed PickScore evaluation for a Model-Guidance DiffusionNFT checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=48)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--sampler", choices=("sde", "ode"), default="sde")
    parser.add_argument("--seed", type=int, default=20240602507)
    parser.add_argument("--preview-images", type=int, default=8)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def select_policy_states(
    payload: Any,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], int]:
    if not isinstance(payload, Mapping):
        raise TypeError("DiffusionNFT checkpoint must be a mapping")
    current = payload.get("model_current")
    old = payload.get("model_old")
    for name, state in (("model_current", current), ("model_old", old)):
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"DiffusionNFT checkpoint has no nonempty {name} state")
        if not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in state.items()
        ):
            raise TypeError(f"{name} must be a tensor state dict")
    step = payload.get("completed_outer_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("DiffusionNFT checkpoint has an invalid completed_outer_step")
    return current, old, step


def fixed_class_ids(*, count: int, num_classes: int, seed: int) -> list[int]:
    if count <= 0 or count > num_classes:
        raise ValueError("num_classes must lie within the model label range")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(num_classes, generator=generator)[:count].tolist()


def seeded_latents(
    seeds: list[int], *, latent_size: int, device: torch.device
) -> torch.Tensor:
    values = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed % (1 << 32))
        values.append(
            torch.randn((4, latent_size, latent_size), generator=generator, device=device)
        )
    return torch.stack(values)


def summarize(values: torch.Tensor) -> dict[str, float | int]:
    scores = values.detach().double().cpu()
    return {
        "mean": float(scores.mean().item()),
        "std": float(scores.std(unbiased=False).item()),
        "min": float(scores.min().item()),
        "max": float(scores.max().item()),
        "num_samples": int(scores.numel()),
    }


def load_policy_states(
    config: NFTConfig,
    *,
    checkpoint_path: Path,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], int]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    current_state, old_state, outer_step = select_policy_states(payload)
    stored_config = payload.get("config")
    if not isinstance(stored_config, Mapping):
        raise ValueError("DiffusionNFT checkpoint has no stored config")
    for name in ("method", "model_key", "image_size", "num_classes", "vae"):
        if stored_config.get(name) != getattr(config, name):
            raise ValueError(f"checkpoint/config mismatch for {name}")
    return current_state, old_state, outer_step


@torch.no_grad()
def evaluate_policy(
    *,
    name: str,
    model: torch.nn.Module,
    config: NFTConfig,
    vae: AutoencoderKL,
    reward_fn: Any,
    labels_global: list[int],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    scores: dict[str, list[torch.Tensor]] = {}
    previews: list[torch.Tensor] = []
    latent_size = config.image_size // 8
    sampler_fn = euler_maruyama_sampler if args.sampler == "sde" else euler_sampler
    for start in range(0, len(labels_global), args.batch_size):
        batch_ids = labels_global[start : start + args.batch_size]
        seeds = [
            args.seed + 1_000_000 + index
            for index in range(start, start + len(batch_ids))
        ]
        latents = seeded_latents(seeds, latent_size=latent_size, device=device)
        labels = torch.tensor(batch_ids, dtype=torch.long, device=device)
        # Reset sampler noise per batch so all policies use paired SDE noise.
        torch.manual_seed((args.seed + 2_000_000 + start) % (1 << 32))
        with autocast_context(config, device):
            endpoints = sampler_fn(
                model,
                latents,
                labels,
                num_steps=args.sampling_steps,
                cfg_scale=1.0,
            )
            decoded = vae.decode(endpoints.float() / 0.18215).sample
        images = (decoded.float().clamp(-1, 1) + 1) / 2
        reward = reward_fn(images, labels)
        for key, value in reward.items():
            scores.setdefault(key, []).append(value.detach().cpu())
        if sum(item.shape[0] for item in previews) < args.preview_images:
            previews.append(images.detach().cpu())
    if args.preview_images > 0:
        torchvision.utils.save_image(
            torch.cat(previews)[: args.preview_images],
            output_dir / f"{name}-preview.png",
            nrow=min(4, args.preview_images),
        )
    return {key: summarize(torch.cat(parts)) for key, parts in scores.items()}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples_per_class <= 0 or args.batch_size <= 0 or args.sampling_steps <= 0:
        raise ValueError("sample, batch, and sampler counts must be positive")
    if args.preview_images < 0:
        raise ValueError("preview-images must be nonnegative")
    if args.expected_step is not None and args.expected_step < 0:
        raise ValueError("expected-step must be nonnegative")
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    init_checkpoint_path = Path(args.init_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint_path.is_file() or not init_checkpoint_path.is_file():
        raise FileNotFoundError("checkpoint or initial checkpoint is missing")
    if output_dir == Path(output_dir.anchor):
        raise ValueError("output-dir cannot be a filesystem root")
    output_dir.mkdir(parents=True, exist_ok=True)

    current_state, old_state, outer_step = load_policy_states(
        config,
        checkpoint_path=checkpoint_path,
    )
    if args.expected_step is not None and outer_step != args.expected_step:
        raise ValueError(
            f"checkpoint completed_outer_step={outer_step}, expected {args.expected_step}"
        )
    initial_payload = torch.load(
        init_checkpoint_path, map_location="cpu", weights_only=False
    )
    initial_state = _unwrap_checkpoint(initial_payload)
    del initial_payload
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{config.vae}",
        local_files_only=args.local_files_only,
    ).to(device).eval().requires_grad_(False)
    reward_fn = build_reward(config.reward, device)
    class_ids = fixed_class_ids(
        count=args.num_classes, num_classes=config.num_classes, seed=args.seed
    )
    labels_global = [
        class_id for class_id in class_ids for _ in range(args.samples_per_class)
    ]
    policies = {}
    policy_states = {
        "pretrained": initial_state,
        "current": current_state,
        "old_policy_ema": old_state,
    }
    for name in ("pretrained", "current", "old_policy_ema"):
        state = policy_states.pop(name)
        model = build_model(config, state).to(device).eval().requires_grad_(False)
        del state
        try:
            policies[name] = evaluate_policy(
                name=name,
                model=model,
                config=config,
                vae=vae,
                reward_fn=reward_fn,
                labels_global=labels_global,
                args=args,
                device=device,
                output_dir=output_dir,
            )
        finally:
            del model
            torch.cuda.empty_cache()
    result = {
        "schema_version": 1,
        "kind": "model_guidance_diffusion_nft_pickscore_eval",
        "class_coverage": (
            "full" if args.num_classes == config.num_classes else "subset"
        ),
        "checkpoint": str(checkpoint_path),
        "completed_outer_step": outer_step,
        "method": config.method,
        "seed": args.seed,
        "num_classes": args.num_classes,
        "samples_per_class": args.samples_per_class,
        "sampling_steps": args.sampling_steps,
        "sampler": args.sampler,
        "policies": policies,
    }
    temporary = output_dir / f".results.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "results.json")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
