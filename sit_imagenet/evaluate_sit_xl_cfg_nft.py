#!/usr/bin/env python3
"""Three-seed, dual-CFG PickScore evaluation for a SiT-XL checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid

NFT_ROOT = Path(__file__).resolve().parents[1]
if str(NFT_ROOT) not in sys.path:
    sys.path.insert(0, str(NFT_ROOT))

import torch
import torchvision
from diffusers import AutoencoderKL
from huggingface_hub import snapshot_download

from sit_cfg_nft.pickscore_reward import PickScoreImageNetReward
from sit_cfg_nft.utils import (
    build_sit_model,
    freeze_for_eval,
    infer_learn_sigma_from_state_dict,
    unwrap_checkpoint,
)
from sit_cfg_nft.xl_cfg_nft import (
    ASSET_ROOT,
    XLCFGNFTConfig,
    euler_maruyama_sit_sampler,
    load_xl_cfg_nft_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(20240602507, 20240602508, 20240602509),
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        action="append",
        dest="cfg_scales",
        help=(
            "Repeat for each evaluation-only CFG scale; defaults to paired "
            "unguided/guided scales 1.0 and 1.5."
        ),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("pretrained", "current", "old_policy_ema"),
        default=("current",),
        help="Checkpoint policies to evaluate, in output order.",
    )
    return parser.parse_args()


def resolve_evaluation_protocol(
    config: XLCFGNFTConfig,
    *,
    cfg_scale: float | None,
    policies: list[str] | tuple[str, ...],
) -> tuple[float, tuple[str, ...]]:
    """Validate evaluation-only overrides without mutating the training config."""

    resolved_cfg = config.rollout.cfg_scale if cfg_scale is None else cfg_scale
    if not math.isfinite(resolved_cfg) or resolved_cfg < 1.0:
        raise ValueError("evaluation CFG scale must be finite and at least one")
    resolved_policies = tuple(policies)
    if not resolved_policies:
        raise ValueError("at least one evaluation policy is required")
    if len(set(resolved_policies)) != len(resolved_policies):
        raise ValueError("evaluation policies must be unique")
    allowed = {"pretrained", "current", "old_policy_ema"}
    if not set(resolved_policies).issubset(allowed):
        raise ValueError("unsupported evaluation policy")
    return float(resolved_cfg), resolved_policies


def config_fingerprint(config: XLCFGNFTConfig) -> str:
    encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_state(payload: Any) -> dict[str, torch.Tensor]:
    state = unwrap_checkpoint(payload)
    if not isinstance(state, Mapping):
        raise TypeError("Model checkpoint does not contain a state mapping")
    normalized = {
        str(name).removeprefix("module."): value
        for name, value in state.items()
        if isinstance(name, str) and torch.is_tensor(value)
    }
    if not normalized:
        raise ValueError("Model checkpoint contains no tensor state")
    return normalized


def validate_training_checkpoint(
    payload: Any,
    *,
    config: XLCFGNFTConfig,
    expected_step: int,
) -> dict[str, Mapping[str, torch.Tensor]]:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported SiT-XL CFG training checkpoint")
    if payload.get("config") != config.to_dict():
        raise RuntimeError("Training checkpoint config differs from evaluation config")
    if payload.get("config_fingerprint") != config_fingerprint(config):
        raise RuntimeError("Training checkpoint config fingerprint differs")
    if payload.get("completed_outer_step") != expected_step:
        raise RuntimeError("Training checkpoint outer step differs")
    if payload.get("completed_pretrain_updates") != config.center.pretrain_updates:
        raise RuntimeError("Training checkpoint center-pretrain progress differs")
    states: dict[str, Mapping[str, torch.Tensor]] = {}
    for name, key in (("current", "model_current"), ("old_policy_ema", "model_old")):
        state = payload.get(key)
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(f"Training checkpoint lacks {key}")
        if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
            raise TypeError(f"Training checkpoint {key} is not a tensor state dict")
        states[name] = state
    return states


def build_model(
    config: XLCFGNFTConfig,
    state: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
):
    learn_sigma = infer_learn_sigma_from_state_dict(dict(state), config.model_key)
    model = build_sit_model(
        model_name=config.model_key,
        image_size=config.image_size,
        num_classes=config.num_classes,
        learn_sigma=learn_sigma,
        device=torch.device("cpu"),
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    freeze_for_eval(model)
    return model


def fixed_class_ids(*, count: int, num_classes: int, seed: int) -> list[int]:
    if count < 1 or count > num_classes:
        raise ValueError("Evaluation class count is outside the model label range")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(num_classes, generator=generator)[:count].tolist()


def summarize(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().double().cpu()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "num_samples": int(values.numel()),
    }


def autocast_context(config: XLCFGNFTConfig):
    if config.mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate_policy(
    *,
    name: str,
    model,
    config: XLCFGNFTConfig,
    labels_global: list[int],
    vae,
    reward_fn,
    device: torch.device,
    seed: int,
    cfg_scale: float,
    output_dir: Path,
) -> dict[str, Any]:
    score_parts: dict[str, list[torch.Tensor]] = {}
    preview_parts: list[torch.Tensor] = []
    latent_size = config.image_size // 8
    for start in range(0, len(labels_global), config.evaluation.batch_size):
        batch_ids = labels_global[start : start + config.evaluation.batch_size]
        labels = torch.tensor(batch_ids, dtype=torch.long, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed((seed + start * 1_000_003) % (1 << 63))
        latents = torch.randn(
            len(batch_ids),
            4,
            latent_size,
            latent_size,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        endpoints = euler_maruyama_sit_sampler(
            model,
            latents,
            labels,
            num_steps=config.rollout.sampling_steps,
            cfg_scale=cfg_scale,
            num_classes=config.num_classes,
            generator=generator,
            autocast=lambda: autocast_context(config),
        )
        with autocast_context(config):
            decoded = vae.decode(endpoints / 0.18215).sample
        images = (decoded.float().clamp(-1.0, 1.0) + 1.0) / 2.0
        scores = reward_fn(images, labels)
        for key in ("pickscore", "raw"):
            score_parts.setdefault(key, []).append(scores[key].detach().cpu())
        if sum(part.shape[0] for part in preview_parts) < config.evaluation.preview_samples:
            preview_parts.append(images.detach().cpu())
    if config.evaluation.preview_samples:
        torchvision.utils.save_image(
            torch.cat(preview_parts)[: config.evaluation.preview_samples],
            output_dir / f"{name}-preview.png",
            nrow=min(4, config.evaluation.preview_samples),
        )
    return {key: summarize(torch.cat(parts)) for key, parts in score_parts.items()}


def _validate_evaluation_axes(
    seeds: list[int] | tuple[int, ...],
    cfg_scales: list[float] | tuple[float, ...] | None,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    resolved_seeds = tuple(seeds)
    resolved_cfgs = tuple(cfg_scales or (1.0, 1.5))
    if len(resolved_seeds) != 3 or len(set(resolved_seeds)) != 3:
        raise ValueError("Evaluation requires exactly three distinct seeds")
    if any(seed < 0 for seed in resolved_seeds):
        raise ValueError("Evaluation seeds must be non-negative")
    if resolved_cfgs != (1.0, 1.5):
        raise ValueError("Evaluation CFG scales must be exactly 1.0 then 1.5")
    return resolved_seeds, resolved_cfgs


def _coordinate_key(seed: int, cfg_scale: float) -> str:
    cfg_token = str(cfg_scale).replace(".", "p")
    return f"seed_{seed}_cfg_{cfg_token}"


def _aggregate_coordinates(coordinates: list[Mapping[str, Any]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for cfg_scale in (1.0, 1.5):
        selected = [
            row for row in coordinates if float(row["evaluation_cfg_scale"]) == cfg_scale
        ]
        if len(selected) != 3:
            raise RuntimeError(f"CFG {cfg_scale} does not contain exactly three seeds")
        entry: dict[str, Any] = {
            "evaluation_cfg_scale": cfg_scale,
            "seeds": [int(row["seed"]) for row in selected],
        }
        for metric in ("pickscore", "raw"):
            seed_means = [float(row["policies"]["current"][metric]["mean"]) for row in selected]
            sample_counts = [
                int(row["policies"]["current"][metric]["num_samples"])
                for row in selected
            ]
            if sample_counts != [2000, 2000, 2000]:
                raise RuntimeError(
                    f"CFG {cfg_scale} {metric} sample counts differ: {sample_counts}"
                )
            tensor = torch.tensor(seed_means, dtype=torch.float64)
            entry[metric] = {
                "mean_across_seeds": float(tensor.mean()),
                "std_across_seed_means": float(tensor.std(unbiased=False)),
                "seed_means": seed_means,
                "num_seeds": 3,
                "num_samples_total": 6000,
            }
        aggregated[str(cfg_scale)] = entry
    return aggregated


def _reuse_complete_result(
    path: Path,
    *,
    method: str,
    step: int,
    seeds: tuple[int, ...],
    cfg_scales: tuple[float, ...],
) -> bool:
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "method": method,
        "completed_outer_step": step,
        "seeds": list(seeds),
        "cfg_scales": list(cfg_scales),
        "num_classes": 1000,
        "samples_per_class": 2,
    }
    differences = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if differences:
        raise RuntimeError(
            "Existing evaluation result identity differs: "
            + json.dumps(differences, sort_keys=True)
        )
    if len(result.get("coordinates", [])) != 6:
        raise RuntimeError("Existing evaluation result lacks six coordinates")
    print(
        "EVALUATION_REUSED "
        + json.dumps(
            {
                "completed_outer_step": step,
                "result": str(path),
                "aggregates": result.get("aggregates"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return True


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Final evaluation requires exactly one visible CUDA GPU")
    config = load_xl_cfg_nft_config(args.config)
    seeds, cfg_scales = _validate_evaluation_axes(args.seeds, args.cfg_scales)
    _, evaluation_policies = resolve_evaluation_protocol(
        config, cfg_scale=1.0, policies=args.policies
    )
    if evaluation_policies != ("current",):
        raise ValueError("The checkpoint trajectory evaluates only the current policy")
    if (
        args.expected_step < 100
        or args.expected_step > config.max_outer_steps
        or args.expected_step % config.evaluation.every
    ):
        raise ValueError("--expected-step is not a configured 50-step evaluation boundary")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise FileNotFoundError("Final training checkpoint is missing or is a symlink")
    try:
        output_dir.relative_to(ASSET_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"--output-dir must remain below {ASSET_ROOT}") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    if _reuse_complete_result(
        result_path,
        method=config.method,
        step=args.expected_step,
        seeds=seeds,
        cfg_scales=cfg_scales,
    ):
        return

    payload = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    states = validate_training_checkpoint(
        payload, config=config, expected_step=args.expected_step
    )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    vae_snapshot = snapshot_download(
        config.vae_name, revision=config.vae_revision, local_files_only=True
    )
    processor_snapshot = snapshot_download(
        config.reward.processor_name,
        revision=config.reward.processor_revision,
        local_files_only=True,
    )
    reward_snapshot = snapshot_download(
        config.reward.model_name,
        revision=config.reward.model_revision,
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(
        vae_snapshot, local_files_only=True
    ).to(device)
    freeze_for_eval(vae)
    reward_fn = PickScoreImageNetReward(
        device=device,
        dtype=torch.float32,
        processor_name=processor_snapshot,
        model_name=reward_snapshot,
        prompt_template=config.reward.prompt_template,
        divide_by_26=config.reward.divide_by_26,
        cache_text_embeddings=True,
        return_raw=False,
        batch_size=config.reward.batch_size,
        local_files_only=True,
    )
    state = states.pop("current")
    model = build_model(config, state, device=device)
    del state, payload, states
    coordinates: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            class_ids = fixed_class_ids(
                count=config.evaluation.num_classes,
                num_classes=config.num_classes,
                seed=seed,
            )
            labels_global = [
                class_id
                for class_id in class_ids
                for _ in range(config.evaluation.samples_per_class)
            ]
            for cfg_scale in cfg_scales:
                key = _coordinate_key(seed, cfg_scale)
                coordinate_dir = output_dir / key
                coordinate_dir.mkdir(parents=True, exist_ok=True)
                policies = {
                    "current": evaluate_policy(
                        name="current",
                        model=model,
                        config=config,
                        labels_global=labels_global,
                        vae=vae,
                        reward_fn=reward_fn,
                        device=device,
                        seed=seed,
                        cfg_scale=cfg_scale,
                        output_dir=coordinate_dir,
                    )
                }
                coordinate = {
                    "seed": seed,
                    "evaluation_cfg_scale": cfg_scale,
                    "class_ids": class_ids,
                    "policies": policies,
                }
                atomic_write_json(coordinate_dir / "results.json", coordinate)
                coordinates.append(coordinate)
                print(
                    "EVAL_RESULT "
                    + json.dumps(
                        {
                            "completed_outer_step": args.expected_step,
                            "seed": seed,
                            "evaluation_cfg_scale": cfg_scale,
                            "normalized_pickscore_mean": policies["current"]["pickscore"]["mean"],
                            "raw_pickscore_mean": policies["current"]["raw"]["mean"],
                            "num_samples": policies["current"]["raw"]["num_samples"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        del model
        torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "kind": "sit_xl_cfg_nft_checkpoint_pickscore_three_seed_dual_cfg",
        "method": config.method,
        "rollout_cfg_scale": config.rollout.cfg_scale,
        "evaluation_policies": list(evaluation_policies),
        "checkpoint": str(checkpoint_path),
        "completed_outer_step": args.expected_step,
        "seeds": list(seeds),
        "cfg_scales": list(cfg_scales),
        "num_classes": config.evaluation.num_classes,
        "samples_per_class": config.evaluation.samples_per_class,
        "sampling_steps": config.rollout.sampling_steps,
        "sampler": config.rollout.sampler,
        "coordinates": coordinates,
        "aggregates": _aggregate_coordinates(coordinates),
    }
    atomic_write_json(result_path, result)
    print(
        "EVAL_AGGREGATE "
        + json.dumps(
            {
                "completed_outer_step": args.expected_step,
                "aggregates": result["aggregates"],
                "result": str(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
