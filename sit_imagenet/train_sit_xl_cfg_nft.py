#!/usr/bin/env python3
"""Train the ordinary SiT-XL/2 Standard, Dual, and ShiftCenter comparison."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any, Callable, Mapping
import uuid

NFT_ROOT = Path(__file__).resolve().parents[1]
if str(NFT_ROOT) not in sys.path:
    sys.path.insert(0, str(NFT_ROOT))

import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
from huggingface_hub import snapshot_download
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision

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
    cfg_variant_nft_loss,
    euler_maruyama_sit_sampler,
    evaluate_sit_branches,
    group_normalize_rewards,
    load_xl_cfg_nft_config,
    make_sit_flow_matching_batch,
    old_policy_decay,
    sample_nft_timesteps,
    sit_sde_training_timesteps,
    validate_world_size_geometry,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Permit only the sealed one-update smoke-test overrides.",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--output-root")
    parser.add_argument("--execution-output-root", help="Storage relocation only; preserves checkpoint config identity.")
    parser.add_argument("--max-outer-steps", type=int)
    parser.add_argument("--groups-global", type=int)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--timestep-fraction", type=float)
    parser.add_argument("--center-timestep-fraction", type=float)
    parser.add_argument("--center-pretrain-updates", type=int)
    parser.add_argument("--center-pretrain-timesteps-per-endpoint", type=int)
    parser.add_argument("--checkpoint-keep", type=int)
    parser.add_argument(
        "--training-microbatch", type=int,
        help="Per-rank policy execution batch; preserves checkpoint config and total records.",
    )
    parser.add_argument(
        "--center-microbatch", type=int,
        help="Per-rank center execution batch, including unfinished center pretraining.",
    )
    parser.add_argument("--stop-request-file")
    parser.add_argument(
        "--stop-after-outer-step",
        type=int,
        help=(
            "Operational segment boundary. The trainer saves the ordinary rolling "
            "checkpoint and exits after this completed outer step without changing "
            "the u400 scientific config."
        ),
    )
    parser.add_argument("--continue-from-checkpoint")
    parser.add_argument("--continue-from-checkpoint-sha256")
    parser.add_argument("--continue-from-step", type=int)
    return parser.parse_args()


def apply_runtime_overrides(config: XLCFGNFTConfig, args: argparse.Namespace) -> XLCFGNFTConfig:
    override_names = (
        "run_name",
        "output_root",
        "max_outer_steps",
        "groups_global",
        "samples_per_class",
        "timestep_fraction",
        "center_timestep_fraction",
        "center_pretrain_updates",
        "center_pretrain_timesteps_per_endpoint",
        "checkpoint_keep",
    )
    requested_overrides = [
        name for name in override_names if getattr(args, name) is not None
    ]
    if requested_overrides and not args.smoke:
        raise ValueError(
            "Scientific/runtime overrides are restricted to --smoke: "
            + ", ".join(requested_overrides)
        )
    if args.smoke and set(requested_overrides) != set(override_names):
        missing = sorted(set(override_names) - set(requested_overrides))
        raise ValueError("--smoke requires every sealed override; missing=" + str(missing))
    if args.run_name is not None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", args.run_name) is None:
            raise ValueError("--run-name must be one safe path component")
        config = replace(config, run_name=args.run_name)
    if args.output_root is not None:
        output_root = Path(args.output_root).expanduser().resolve()
        try:
            output_root.relative_to(ASSET_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"--output-root must remain below {ASSET_ROOT}") from exc
        config = replace(config, output_root=str(output_root))
    if args.max_outer_steps is not None:
        if args.max_outer_steps < 1:
            raise ValueError("--max-outer-steps must be positive")
        config = replace(
            config,
            max_outer_steps=args.max_outer_steps,
            evaluation=replace(config.evaluation, every=args.max_outer_steps),
        )
    if args.groups_global is not None:
        if args.groups_global < 1:
            raise ValueError("--groups-global must be positive")
        config = replace(
            config, rollout=replace(config.rollout, groups_global=args.groups_global)
        )
    if args.samples_per_class is not None:
        if args.samples_per_class < 2:
            raise ValueError("--samples-per-class must be at least two")
        config = replace(
            config,
            rollout=replace(
                config.rollout, samples_per_class=args.samples_per_class
            ),
        )
    if args.timestep_fraction is not None:
        if not 0.0 < args.timestep_fraction <= 1.0:
            raise ValueError("--timestep-fraction must lie in (0, 1]")
        config = replace(
            config,
            training=replace(
                config.training, timestep_fraction=args.timestep_fraction
            ),
        )
    if args.center_timestep_fraction is not None:
        if not 0.0 < args.center_timestep_fraction <= 1.0:
            raise ValueError("--center-timestep-fraction must lie in (0, 1]")
        config = replace(
            config,
            center=replace(
                config.center, timestep_fraction=args.center_timestep_fraction
            ),
        )
    if args.center_pretrain_updates is not None:
        if args.center_pretrain_updates < 0:
            raise ValueError("--center-pretrain-updates must be non-negative")
        if config.method != "shiftcenter" and args.center_pretrain_updates != 0:
            raise ValueError("Only shiftcenter may run center pretraining")
        config = replace(
            config,
            center=replace(
                config.center, pretrain_updates=args.center_pretrain_updates
            ),
        )
    if args.center_pretrain_timesteps_per_endpoint is not None:
        if args.center_pretrain_timesteps_per_endpoint < 1:
            raise ValueError(
                "--center-pretrain-timesteps-per-endpoint must be positive"
            )
        config = replace(
            config,
            center=replace(
                config.center,
                pretrain_timesteps_per_endpoint=(
                    args.center_pretrain_timesteps_per_endpoint
                ),
            ),
        )
    if args.checkpoint_keep is not None:
        if args.checkpoint_keep < 1:
            raise ValueError("--checkpoint-keep must be positive")
        config = replace(
            config,
            checkpoint=replace(config.checkpoint, keep=args.checkpoint_keep),
        )
    if args.smoke:
        expected_pretrain = 1 if config.method == "shiftcenter" else 0
        actual = {
            "max_outer_steps": config.max_outer_steps,
            "groups_global": config.rollout.groups_global,
            "samples_per_class": config.rollout.samples_per_class,
            "timestep_fraction": config.training.timestep_fraction,
            "center_timestep_fraction": config.center.timestep_fraction,
            "center_pretrain_updates": config.center.pretrain_updates,
            "center_pretrain_timesteps_per_endpoint": (
                config.center.pretrain_timesteps_per_endpoint
            ),
            "checkpoint_keep": config.checkpoint.keep,
            "output_root_name": Path(config.output_root).name,
            "run_name_has_smoke": "smoke" in config.run_name,
        }
        expected = {
            "max_outer_steps": 1,
            "groups_global": 2,
            "samples_per_class": 24,
            "timestep_fraction": 0.1,
            "center_timestep_fraction": 0.1,
            "center_pretrain_updates": expected_pretrain,
            "center_pretrain_timesteps_per_endpoint": 1,
            "checkpoint_keep": 1,
            "output_root_name": "smokes",
            "run_name_has_smoke": True,
        }
        if actual != expected:
            raise ValueError(
                "--smoke overrides differ from the sealed contract: "
                + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
            )
    return config


def setup_distributed() -> tuple[torch.device, int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("SiT-XL comparison training requires CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return device, rank, local_rank, world_size


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier(device_ids=[torch.cuda.current_device()])


def rank0_print(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def wrap_ddp(model, *, device: torch.device, local_rank: int, world_size: int):
    if world_size == 1:
        return model
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )


def autocast_context(config: XLCFGNFTConfig, device: torch.device):
    if config.mixed_precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_geometry(
    config: XLCFGNFTConfig,
    *,
    world_size: int,
    training_microbatch: int | None = None,
    center_microbatch: int | None = None,
) -> dict[str, int]:
    """Validate execution-only splits without changing the checkpoint config."""
    policy_batch = config.training.batch_size if training_microbatch is None else training_microbatch
    center_batch = config.center.batch_size if center_microbatch is None else center_microbatch
    for name, value in (("training_microbatch", policy_batch), ("center_microbatch", center_batch)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    execution_config = replace(
        config,
        training=replace(config.training, batch_size=policy_batch),
        center=replace(config.center, batch_size=center_batch),
    )
    geometry = validate_world_size_geometry(execution_config, world_size=world_size)
    geometry.update(
        training_microbatch=policy_batch,
        center_microbatch=center_batch,
        online_center_microbatches_per_rank=(
            geometry["online_center_records_global"] // world_size // center_batch
        ),
    )
    return geometry


def config_fingerprint(config: XLCFGNFTConfig) -> str:
    return config_mapping_fingerprint(config.to_dict())


def config_mapping_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _flatten_config(
    value: Mapping[str, Any], *, prefix: str = ""
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        item = value[key]
        if isinstance(item, Mapping):
            flattened.update(_flatten_config(item, prefix=path))
        else:
            flattened[path] = item
    return flattened


def continuation_config_changes(
    parent_config: Mapping[str, Any],
    target_config: XLCFGNFTConfig,
    *,
    completed_step: int,
) -> dict[str, dict[str, Any]]:
    """Validate a terminal u100 checkpoint extension into a new u400 run.

    Every scientific and optimizer field must remain JSON-equivalent.  The
    continuation may relocate storage, switch the execution geometry from the
    two-GPU reference to one GPU without changing per-rank microbatches, extend
    the horizon, retain every 50-step checkpoint, and expand the fixed
    evaluation from 48 to all 1,000 classes.
    """

    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise TypeError("continuation completed step must be an integer")
    if completed_step < 0:
        raise ValueError("continuation completed step must be non-negative")
    parent = _flatten_config(parent_config)
    target = _flatten_config(target_config.to_dict())
    if set(parent) != set(target):
        raise RuntimeError("Continuation checkpoint config keys differ")
    if parent.get("max_outer_steps") != completed_step:
        raise RuntimeError(
            "Continuation checkpoint must be the completed parent run final step"
        )
    if target_config.max_outer_steps <= completed_step:
        raise RuntimeError(
            "Continuation target max_outer_steps must exceed the parent step"
        )
    if parent.get("run_name") == target.get("run_name"):
        raise RuntimeError("Continuation must use a new run_name")

    allowed = {
        "checkpoint.keep",
        "checkpoint.save_every",
        "evaluation.every",
        "evaluation.num_classes",
        "expected_gpus",
        "init_checkpoint",
        "max_outer_steps",
        "output_root",
        "run_name",
    }
    incompatible = {
        key: {"parent": parent[key], "target": target[key]}
        for key in sorted(parent)
        if key not in allowed and parent[key] != target[key]
    }
    if incompatible:
        raise RuntimeError(
            "Continuation changes scientific or optimizer config: "
            + json.dumps(incompatible, sort_keys=True)
        )
    changes = {
        key: {"parent": parent[key], "target": target[key]}
        for key in sorted(allowed)
        if parent[key] != target[key]
    }
    required = {
        "checkpoint.keep",
        "checkpoint.save_every",
        "evaluation.every",
        "evaluation.num_classes",
        "expected_gpus",
        "max_outer_steps",
        "run_name",
    }
    if not required.issubset(changes):
        raise RuntimeError(
            "Continuation does not declare every required u400 execution, "
            "checkpoint, and evaluation change"
        )
    return changes


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def truncate_jsonl(path: Path, *, key: str, completed: int) -> None:
    if not path.exists():
        return
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if int(record[key]) <= completed:
            kept.append(json.dumps(record, sort_keys=True, allow_nan=False))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    os.replace(temporary, path)


def prepare_run_directory(
    config: XLCFGNFTConfig, *, rank: int, world_size: int,
    execution_output_root: str | None = None,
) -> tuple[Path, str]:
    run_dir = (
        Path(execution_output_root) / config.run_name
        if execution_output_root is not None else config.run_dir
    ).resolve()
    try:
        run_dir.relative_to(ASSET_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Run directory must remain below {ASSET_ROOT}") from exc
    fingerprint = config_fingerprint(config)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints", "pretrain", "previews"):
            (run_dir / name).mkdir(exist_ok=True)
        effective = {"fingerprint": fingerprint, "config": config.to_dict()}
        effective_path = run_dir / "effective_config.json"
        if effective_path.exists():
            if json.loads(effective_path.read_text(encoding="utf-8")) != effective:
                raise RuntimeError(
                    f"Run directory already contains a different effective config: {run_dir}"
                )
        else:
            atomic_write_json(effective_path, effective)
    barrier(world_size)
    return run_dir, fingerprint


def normalize_checkpoint_state(payload: Any) -> Mapping[str, torch.Tensor]:
    state = unwrap_checkpoint(payload)
    if not isinstance(state, Mapping):
        raise TypeError("Initial checkpoint does not contain a state mapping")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        name = key.removeprefix("module.")
        if name in normalized:
            raise ValueError(f"Duplicate normalized checkpoint key: {name}")
        normalized[name] = value
    if not normalized:
        raise ValueError("Initial checkpoint contains no tensor parameters")
    return normalized


def build_model_from_state(
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
    return model.to(device)


@torch.no_grad()
def update_ema(target, source, decay: float) -> None:
    source = unwrap_model(source)
    target_params = dict(target.named_parameters())
    source_params = dict(source.named_parameters())
    if target_params.keys() != source_params.keys():
        raise RuntimeError("EMA parameter structures differ")
    for name, value in source_params.items():
        target_params[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())
    if target_buffers.keys() != source_buffers.keys():
        raise RuntimeError("EMA buffer structures differ")
    for name, value in source_buffers.items():
        target_buffers[name].copy_(value)


def all_ranks_finite(value: torch.Tensor, world_size: int) -> bool:
    flag = torch.tensor(
        int(bool(torch.isfinite(value.detach()).all().item())),
        device=value.device,
        dtype=torch.int32,
    )
    if world_size > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def reduce_mean_metrics(
    metrics: Mapping[str, float], *, device: torch.device, world_size: int
) -> dict[str, float]:
    names = sorted(metrics)
    values = torch.tensor([metrics[name] for name in names], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world_size
    return {name: float(value) for name, value in zip(names, values.cpu().tolist())}


def synchronize_stop_requested(
    requested: bool,
    *,
    request_file: Path | None,
    device: torch.device,
    world_size: int,
) -> bool:
    local_requested = requested or (
        request_file is not None and request_file.is_file()
    )
    flag = torch.tensor(int(local_requested), device=device, dtype=torch.int32)
    if world_size > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def choose_local_classes(
    config: XLCFGNFTConfig, *, round_index: int, rank: int, world_size: int, seed_offset: int
) -> torch.LongTensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + seed_offset + round_index * 100_003)
    selected = torch.randperm(config.num_classes, generator=generator)[
        : config.rollout.groups_global
    ]
    groups_local = config.rollout.groups_global // world_size
    return selected[rank * groups_local : (rank + 1) * groups_local]


def validate_endpoint_dataset(
    dataset: Mapping[str, torch.Tensor],
    *,
    config: XLCFGNFTConfig,
    world_size: int,
) -> dict[str, torch.Tensor]:
    if set(dataset) != {"endpoints", "labels", "advantages"}:
        raise ValueError("Endpoint dataset keys differ from the sealed contract")
    result = dict(dataset)
    expected = (
        config.rollout.groups_global
        * config.rollout.samples_per_class
        // world_size
    )
    latent_size = config.image_size // 8
    if result["endpoints"].shape != (expected, 4, latent_size, latent_size):
        raise ValueError("Endpoint dataset latent shape differs")
    if result["labels"].shape != (expected,) or result["advantages"].shape != (expected,):
        raise ValueError("Endpoint dataset label or advantage shape differs")
    if result["labels"].dtype != torch.long:
        raise ValueError("Endpoint labels must have dtype torch.long")
    if bool(((result["labels"] < 0) | (result["labels"] >= config.num_classes)).any()):
        raise ValueError("Endpoint labels are outside the ImageNet class range")
    grouped_labels = result["labels"].reshape(
        config.rollout.groups_global // world_size,
        config.rollout.samples_per_class,
    )
    if not bool((grouped_labels == grouped_labels[:, :1]).all()):
        raise ValueError("Endpoint labels no longer form contiguous class groups")
    if torch.unique(grouped_labels[:, 0]).numel() != grouped_labels.shape[0]:
        raise ValueError("Endpoint class groups must be unique within a rank")
    if not bool(torch.isfinite(result["endpoints"]).all()):
        raise ValueError("Endpoint dataset contains non-finite latents")
    if not bool(torch.isfinite(result["advantages"]).all()):
        raise ValueError("Endpoint dataset contains non-finite advantages")
    return result


def sample_endpoint_batches(
    *,
    model_old,
    config: XLCFGNFTConfig,
    round_index: int,
    rank: int,
    world_size: int,
    device: torch.device,
    seed_offset: int,
):
    classes = choose_local_classes(
        config,
        round_index=round_index,
        rank=rank,
        world_size=world_size,
        seed_offset=seed_offset,
    )
    labels_cpu = classes.repeat_interleave(config.rollout.samples_per_class)
    generator = torch.Generator(device=device)
    generator.manual_seed(
        config.seed + seed_offset + round_index * 1_000_003 + rank * 100_000_007
    )
    latent_size = config.image_size // 8
    for start in range(0, labels_cpu.numel(), config.rollout.batch_size):
        labels = labels_cpu[start : start + config.rollout.batch_size].to(
            device=device, dtype=torch.long
        )
        latents = torch.randn(
            labels.shape[0],
            4,
            latent_size,
            latent_size,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        endpoints = euler_maruyama_sit_sampler(
            model_old,
            latents,
            labels,
            num_steps=config.rollout.sampling_steps,
            cfg_scale=config.rollout.cfg_scale,
            num_classes=config.num_classes,
            generator=generator,
            autocast=lambda: autocast_context(config, device),
        )
        yield labels, endpoints


@torch.no_grad()
def collect_pretrain_endpoints(
    *,
    model_old,
    config: XLCFGNFTConfig,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    endpoint_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for labels, endpoints in sample_endpoint_batches(
        model_old=model_old,
        config=config,
        round_index=0,
        rank=rank,
        world_size=world_size,
        device=device,
        seed_offset=2_000_000_011,
    ):
        endpoint_parts.append(endpoints.cpu())
        label_parts.append(labels.cpu())
    return validate_endpoint_dataset({
        "endpoints": torch.cat(endpoint_parts),
        "labels": torch.cat(label_parts),
        "advantages": torch.zeros(sum(x.shape[0] for x in label_parts)),
    }, config=config, world_size=world_size)


def load_or_collect_pretrain_dataset(
    *,
    model_old,
    config: XLCFGNFTConfig,
    run_dir: Path,
    fingerprint: str,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    path = run_dir / "pretrain" / f"rank_{rank:05d}.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "schema_version": 1,
            "config_fingerprint": fingerprint,
            "rank": rank,
            "world_size": world_size,
        }
        if not isinstance(payload, Mapping) or any(payload.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"Pretrain endpoint shard identity differs: {path}")
        dataset = payload.get("dataset")
        if not isinstance(dataset, Mapping):
            raise RuntimeError(f"Pretrain endpoint shard has no dataset: {path}")
        return validate_endpoint_dataset(
            dataset, config=config, world_size=world_size
        )
    dataset = collect_pretrain_endpoints(
        model_old=model_old,
        config=config,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema_version": 1,
            "config_fingerprint": fingerprint,
            "rank": rank,
            "world_size": world_size,
            "dataset": dataset,
        },
        temporary,
    )
    os.replace(temporary, path)
    return dataset


@torch.no_grad()
def collect_reward_rollout(
    *,
    model_old,
    vae,
    reward_fn,
    config: XLCFGNFTConfig,
    outer_step: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float], torch.Tensor]:
    endpoint_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    raw_parts: list[torch.Tensor] = []
    preview_parts: list[torch.Tensor] = []
    preview_remaining = config.rollout.preview_samples
    for labels, endpoints in sample_endpoint_batches(
        model_old=model_old,
        config=config,
        round_index=outer_step,
        rank=rank,
        world_size=world_size,
        device=device,
        seed_offset=0,
    ):
        with autocast_context(config, device):
            decoded = vae.decode(endpoints / 0.18215).sample
        images = (decoded.float().clamp(-1.0, 1.0) + 1.0) / 2.0
        reward_output = reward_fn(images, labels)
        rewards = reward_output["avg"].detach().float()
        raw = reward_output.get("raw", rewards).detach().float()
        endpoint_parts.append(endpoints.cpu())
        label_parts.append(labels.cpu())
        reward_parts.append(rewards.cpu())
        raw_parts.append(raw.cpu())
        if preview_remaining:
            take = min(preview_remaining, images.shape[0])
            preview_parts.append(images[:take].cpu())
            preview_remaining -= take

    endpoints = torch.cat(endpoint_parts)
    labels = torch.cat(label_parts)
    rewards = torch.cat(reward_parts)
    raw_rewards = torch.cat(raw_parts)
    groups_local = config.rollout.groups_global // world_size
    advantages = group_normalize_rewards(
        rewards.reshape(groups_local, config.rollout.samples_per_class),
        epsilon=config.reward.advantage_epsilon,
    ).reshape(-1)
    dataset = {"endpoints": endpoints, "labels": labels, "advantages": advantages}
    metrics = {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std(unbiased=False)),
        "reward_raw_mean": float(raw_rewards.mean()),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std(unbiased=False)),
        "advantage_clip_fraction": float(
            (advantages.abs() >= config.training.advantage_clip).float().mean()
        ),
    }
    return (
        validate_endpoint_dataset(dataset, config=config, world_size=world_size),
        metrics,
        torch.cat(preview_parts),
    )


def phase_batches(
    *, num_examples: int, batch_size: int, inner_epochs: int, time_count: int, seed: int
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for inner_epoch in range(inner_epochs):
        for time_draw in range(time_count):
            order = torch.randperm(num_examples, generator=generator)
            for start in range(0, num_examples, batch_size):
                yield inner_epoch, time_draw, order[start : start + batch_size]


def run_accumulated_update(
    *,
    model_train,
    optimizer,
    dataset: Mapping[str, torch.Tensor],
    time_table: torch.Tensor,
    config: XLCFGNFTConfig,
    phase_seed: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    loss_builder: Callable[
        [torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, Mapping[str, torch.Tensor]],
    ],
) -> dict[str, float]:
    num_examples = int(dataset["endpoints"].shape[0])
    if time_table.ndim != 3 or time_table.shape[1] != num_examples:
        raise ValueError("time_table must have shape [inner_epochs, endpoints, times]")
    inner_epochs, _, time_count = time_table.shape
    total_records = num_examples * inner_epochs * time_count
    total_microbatches = inner_epochs * time_count * math.ceil(num_examples / batch_size)
    cuda_generator = torch.Generator(device=device)
    cuda_generator.manual_seed(phase_seed + rank * 1_000_003)
    optimizer.zero_grad(set_to_none=True)
    model_train.train()
    sums: dict[str, float] = {}
    records = 0
    for micro_index, (inner_epoch, time_draw, indices) in enumerate(
        phase_batches(
            num_examples=num_examples,
            batch_size=batch_size,
            inner_epochs=inner_epochs,
            time_count=time_count,
            seed=phase_seed + rank * 10_007,
        )
    ):
        endpoints = dataset["endpoints"][indices].to(device, non_blocking=True)
        labels = dataset["labels"][indices].to(device, non_blocking=True).long()
        advantages = dataset["advantages"][indices].to(device, non_blocking=True).float()
        device_indices = indices.to(device)
        t = time_table[inner_epoch, device_indices, time_draw]
        noise = torch.randn(
            endpoints.shape,
            generator=cuda_generator,
            device=device,
            dtype=endpoints.dtype,
        )
        bridge = make_sit_flow_matching_batch(endpoints, t, noise)
        is_last = micro_index + 1 == total_microbatches
        sync = (
            nullcontext()
            if is_last or not isinstance(model_train, DDP)
            else model_train.no_sync()
        )
        with sync:
            with autocast_context(config, device):
                loss, metrics = loss_builder(
                    bridge.state, labels, t, bridge.target_velocity, advantages
                )
                scaled_loss = loss * (endpoints.shape[0] / total_records)
            if not all_ranks_finite(loss, world_size):
                raise FloatingPointError("Non-finite loss in accumulated update")
            scaled_loss.backward()
        count = int(endpoints.shape[0])
        records += count
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().float()) * count

    if records != total_records:
        raise RuntimeError(f"Accumulated {records} records, expected {total_records}")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        unwrap_model(model_train).parameters(), config.training.max_grad_norm
    )
    if not all_ranks_finite(torch.as_tensor(grad_norm, device=device), world_size):
        raise FloatingPointError("Non-finite gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    result = {name: value / total_records for name, value in sums.items()}
    result.update(
        grad_norm=float(torch.as_tensor(grad_norm).float()),
        records=float(total_records),
        microbatches=float(total_microbatches),
    )
    return result


def rollout_time_table(
    *,
    config: XLCFGNFTConfig,
    num_examples: int,
    inner_epochs: int,
    fraction: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return sample_nft_timesteps(
        sit_sde_training_timesteps(config.rollout.sampling_steps).to(device),
        num_examples=num_examples,
        inner_epochs=inner_epochs,
        timestep_fraction=fraction,
        generator=generator,
    )


def continuous_time_table(
    *, num_examples: int, time_count: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand(
        1, num_examples, time_count, generator=generator, device=device
    )


def run_center_update(
    *,
    model_center,
    model_center_ema,
    optimizer_center,
    dataset: Mapping[str, torch.Tensor],
    time_table: torch.Tensor,
    config: XLCFGNFTConfig,
    phase_seed: int,
    rank: int,
    world_size: int,
    device: torch.device,
    microbatch: int | None = None,
) -> dict[str, float]:
    def loss_builder(state, labels, t, target, advantages):
        del advantages
        prediction = model_center(state, t, labels)
        per_example = (prediction.float() - target.float()).square().flatten(1).mean(1)
        loss = per_example.mean()
        return loss, {"loss": loss.detach()}

    metrics = run_accumulated_update(
        model_train=model_center,
        optimizer=optimizer_center,
        dataset=dataset,
        time_table=time_table,
        config=config,
        phase_seed=phase_seed,
        batch_size=config.center.batch_size if microbatch is None else microbatch,
        rank=rank,
        world_size=world_size,
        device=device,
        loss_builder=loss_builder,
    )
    update_ema(model_center_ema, model_center, config.center.ema_decay)
    freeze_for_eval(model_center_ema)
    return {f"center_{key}": value for key, value in metrics.items()}


def run_policy_update(
    *,
    model_current,
    model_old,
    model_reference,
    model_center_snapshot,
    optimizer_current,
    dataset: Mapping[str, torch.Tensor],
    config: XLCFGNFTConfig,
    outer_step: int,
    rank: int,
    world_size: int,
    device: torch.device,
    microbatch: int | None = None,
) -> dict[str, float]:
    time_table = rollout_time_table(
        config=config,
        num_examples=int(dataset["endpoints"].shape[0]),
        inner_epochs=config.training.inner_epochs,
        fraction=config.training.timestep_fraction,
        seed=config.seed + outer_step * 10_000_019 + 400_009 + rank * 17,
        device=device,
    )
    model_old.eval()
    model_reference.eval()
    if model_center_snapshot is not None:
        model_center_snapshot.eval()

    def loss_builder(state, labels, t, target, advantages):
        if config.method in {"standard", "shiftcenter"}:
            with torch.no_grad():
                old_cond = model_old(state, t, labels).detach()
                ref_cond = model_reference(state, t, labels).detach()
                center = (
                    model_center_snapshot(state, t, labels).detach()
                    if config.method == "shiftcenter" else None
                )
            current_cond = model_current(state, t, labels)
            return cfg_variant_nft_loss(
                method=config.method,
                current_conditional=current_cond,
                old_conditional=old_cond,
                reference_conditional=ref_cond,
                target_velocity=target,
                advantages=advantages,
                beta=config.training.beta,
                advantage_clip=config.training.advantage_clip,
                kl_beta=config.training.kl_beta,
                cfg_scale=config.rollout.cfg_scale,
                center_prediction=center,
            )

        with torch.no_grad():
            old_cond, old_null = evaluate_sit_branches(
                model_old, state, t, labels, num_classes=config.num_classes
            )
            ref_cond, ref_null = evaluate_sit_branches(
                model_reference, state, t, labels, num_classes=config.num_classes
            )
        current_cond, current_null = evaluate_sit_branches(
            model_current, state, t, labels, num_classes=config.num_classes
        )
        return cfg_variant_nft_loss(
            method=config.method,
            current_conditional=current_cond,
            current_unconditional=current_null,
            old_conditional=old_cond.detach(),
            old_unconditional=old_null.detach(),
            reference_conditional=ref_cond.detach(),
            reference_unconditional=ref_null.detach(),
            target_velocity=target,
            advantages=advantages,
            beta=config.training.beta,
            advantage_clip=config.training.advantage_clip,
            kl_beta=config.training.kl_beta,
            cfg_scale=config.rollout.cfg_scale,
        )

    metrics = run_accumulated_update(
        model_train=model_current,
        optimizer=optimizer_current,
        dataset=dataset,
        time_table=time_table,
        config=config,
        phase_seed=config.seed + outer_step * 100_003 + 400_009,
        batch_size=config.training.batch_size if microbatch is None else microbatch,
        rank=rank,
        world_size=world_size,
        device=device,
        loss_builder=loss_builder,
    )
    return {f"policy_{key}": value for key, value in metrics.items()}


def checkpoint_payload(
    *,
    config: XLCFGNFTConfig,
    fingerprint: str,
    completed_pretrain_updates: int,
    completed_outer_step: int,
    model_current,
    model_old,
    optimizer_current,
    model_center,
    model_center_ema,
    optimizer_center,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_fingerprint": fingerprint,
        "config": config.to_dict(),
        "completed_pretrain_updates": completed_pretrain_updates,
        "completed_outer_step": completed_outer_step,
        "model_current": unwrap_model(model_current).state_dict(),
        "model_old": model_old.state_dict(),
        "optimizer_current": optimizer_current.state_dict(),
        "model_center": (
            unwrap_model(model_center).state_dict() if model_center is not None else None
        ),
        "model_center_ema": (
            model_center_ema.state_dict() if model_center_ema is not None else None
        ),
        "optimizer_center": (
            optimizer_center.state_dict() if optimizer_center is not None else None
        ),
    }


def save_checkpoint(
    *,
    run_dir: Path,
    payload: Mapping[str, Any],
    label: str,
    keep: int,
) -> Path:
    directory = run_dir / "checkpoints"
    latest = directory / "latest.pt"
    temporary = directory / f".latest.{os.getpid()}.tmp"
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, latest)
    milestone = directory / f"{label}.pt"
    milestone_tmp = directory / f".{label}.{os.getpid()}.tmp"
    try:
        os.link(latest, milestone_tmp)
    except OSError:
        with milestone_tmp.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(milestone_tmp, milestone)
    milestones = sorted(directory.glob("pretrain_*.pt")) + sorted(
        directory.glob("outer_*.pt")
    )
    for stale in milestones[:-keep]:
        stale.unlink()
    return latest


def _restore_checkpoint_state(
    *,
    payload: Mapping[str, Any],
    model_current,
    model_old,
    optimizer_current,
    model_center,
    model_center_ema,
    optimizer_center,
) -> tuple[int, int]:
    unwrap_model(model_current).load_state_dict(payload["model_current"], strict=True)
    model_old.load_state_dict(payload["model_old"], strict=True)
    optimizer_current.load_state_dict(payload["optimizer_current"])
    if model_center is not None:
        if any(payload.get(name) is None for name in (
            "model_center", "model_center_ema", "optimizer_center"
        )):
            raise RuntimeError("ShiftCenter checkpoint lacks center state")
        unwrap_model(model_center).load_state_dict(payload["model_center"], strict=True)
        model_center_ema.load_state_dict(payload["model_center_ema"], strict=True)
        optimizer_center.load_state_dict(payload["optimizer_center"])
    elif any(payload.get(name) is not None for name in (
        "model_center", "model_center_ema", "optimizer_center"
    )):
        raise RuntimeError("Non-shift checkpoint unexpectedly contains center state")
    pretrain = int(payload.get("completed_pretrain_updates", 0))
    outer = int(payload.get("completed_outer_step", 0))
    if pretrain < 0 or outer < 0:
        raise RuntimeError("Checkpoint progress counters are invalid")
    return pretrain, outer


def restore_checkpoint(
    *,
    path: Path,
    fingerprint: str,
    model_current,
    model_old,
    optimizer_current,
    model_center,
    model_center_ema,
    optimizer_center,
) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported comparison checkpoint")
    if payload.get("config_fingerprint") != fingerprint:
        raise RuntimeError("Resume checkpoint configuration fingerprint differs")
    return _restore_checkpoint_state(
        payload=payload,
        model_current=model_current,
        model_old=model_old,
        optimizer_current=optimizer_current,
        model_center=model_center,
        model_center_ema=model_center_ema,
        optimizer_center=optimizer_center,
    )


def load_continuation_checkpoint(
    *,
    path: Path,
    expected_sha256: str,
    expected_step: int,
    target_config: XLCFGNFTConfig,
    mmap: bool = False,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Authenticate one terminal parent checkpoint for an immutable extension."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Continuation checkpoint is not a regular file: {path}")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("Continuation checkpoint SHA-256 is invalid")
    if isinstance(expected_step, bool) or not isinstance(expected_step, int):
        raise TypeError("Continuation expected step must be an integer")
    if expected_step < 0:
        raise ValueError("Continuation expected step must be non-negative")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Continuation checkpoint SHA-256 differs: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=mmap
    )
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Continuation checkpoint is not a mapping")
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("Continuation checkpoint schema_version differs")
    completed_step = checkpoint.get("completed_outer_step")
    if completed_step != expected_step:
        raise RuntimeError(
            "Continuation checkpoint step differs: "
            f"expected={expected_step} actual={completed_step}"
        )
    completed_pretrain = checkpoint.get("completed_pretrain_updates")
    if completed_pretrain != target_config.center.pretrain_updates:
        raise RuntimeError(
            "Continuation checkpoint center pretrain progress differs: "
            f"expected={target_config.center.pretrain_updates} "
            f"actual={completed_pretrain}"
        )
    parent_config = checkpoint.get("config")
    if not isinstance(parent_config, Mapping):
        raise RuntimeError("Continuation checkpoint has no config mapping")
    parent_fingerprint = config_mapping_fingerprint(parent_config)
    if checkpoint.get("config_fingerprint") != parent_fingerprint:
        raise RuntimeError("Continuation checkpoint stored config fingerprint differs")
    changes = continuation_config_changes(
        parent_config,
        target_config,
        completed_step=expected_step,
    )
    metadata = {
        "schema_version": 1,
        "kind": "ordinary_sit_xl_cfg_nft_continuation",
        "parent_checkpoint": str(path),
        "parent_checkpoint_bytes": path.stat().st_size,
        "parent_checkpoint_sha256": actual_sha256,
        "parent_config_fingerprint": parent_fingerprint,
        "parent_completed_pretrain_updates": completed_pretrain,
        "parent_completed_outer_step": expected_step,
        "target_config_fingerprint": config_fingerprint(target_config),
        "target_max_outer_steps": target_config.max_outer_steps,
        "allowed_config_changes": changes,
    }
    return checkpoint, metadata


def restore_continuation_checkpoint(
    *,
    path: Path,
    expected_sha256: str,
    expected_step: int,
    target_config: XLCFGNFTConfig,
    model_current,
    model_old,
    optimizer_current,
    model_center,
    model_center_ema,
    optimizer_center,
) -> tuple[int, int, dict[str, Any]]:
    checkpoint, metadata = load_continuation_checkpoint(
        path=path,
        expected_sha256=expected_sha256,
        expected_step=expected_step,
        target_config=target_config,
    )
    completed_pretrain, completed_outer = _restore_checkpoint_state(
        payload=checkpoint,
        model_current=model_current,
        model_old=model_old,
        optimizer_current=optimizer_current,
        model_center=model_center,
        model_center_ema=model_center_ema,
        optimizer_center=optimizer_center,
    )
    if (
        completed_pretrain != target_config.center.pretrain_updates
        or completed_outer != expected_step
    ):
        raise RuntimeError("Restored continuation progress differs from its identity")
    return completed_pretrain, completed_outer, metadata


def maybe_save_checkpoint(
    *,
    force: bool,
    progress: int,
    phase: str,
    rank: int,
    world_size: int,
    run_dir: Path,
    config: XLCFGNFTConfig,
    fingerprint: str,
    completed_pretrain_updates: int,
    completed_outer_step: int,
    model_current,
    model_old,
    optimizer_current,
    model_center,
    model_center_ema,
    optimizer_center,
) -> None:
    due = force or progress % config.checkpoint.save_every == 0
    if not due:
        return
    barrier(world_size)
    if rank == 0:
        payload = checkpoint_payload(
            config=config,
            fingerprint=fingerprint,
            completed_pretrain_updates=completed_pretrain_updates,
            completed_outer_step=completed_outer_step,
            model_current=model_current,
            model_old=model_old,
            optimizer_current=optimizer_current,
            model_center=model_center,
            model_center_ema=model_center_ema,
            optimizer_center=optimizer_center,
        )
        label = f"{phase}_{progress:06d}"
        latest = save_checkpoint(
            run_dir=run_dir, payload=payload, label=label, keep=config.checkpoint.keep
        )
        print(f"Saved {phase} checkpoint: {latest}", flush=True)
    barrier(world_size)


def preflight_assets(config: XLCFGNFTConfig) -> dict[str, Any]:
    checkpoint = Path(config.init_checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(f"Official SiT checkpoint is not a regular file: {checkpoint}")
    actual_hash = sha256_file(checkpoint)
    if actual_hash != config.init_checkpoint_sha256:
        raise RuntimeError(
            f"SiT checkpoint SHA256 differs: expected={config.init_checkpoint_sha256} "
            f"actual={actual_hash}"
        )
    snapshots = {
        "vae": snapshot_download(
            config.vae_name,
            revision=config.vae_revision,
            local_files_only=True,
        ),
        "pickscore_model": snapshot_download(
            config.reward.model_name,
            revision=config.reward.model_revision,
            local_files_only=True,
        ),
        "pickscore_processor": snapshot_download(
            config.reward.processor_name,
            revision=config.reward.processor_revision,
            local_files_only=True,
        ),
    }
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = normalize_checkpoint_state(payload)
    rows = int(state["final_layer.linear.weight"].shape[0])
    if rows not in {16, 32}:
        raise RuntimeError(f"Unexpected SiT-XL/2 final head rows: {rows}")
    model = build_model_from_state(config, state, device=torch.device("cpu"))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model, payload, state
    return {
        "checkpoint_sha256": actual_hash,
        "snapshots": snapshots,
        "head_rows": rows,
        "strict_model_load": True,
        "parameter_count": parameter_count,
    }


def main() -> None:
    args = parse_args()
    config = apply_runtime_overrides(load_xl_cfg_nft_config(args.config), args)
    geometry = execution_geometry(
        config,
        world_size=config.expected_gpus,
        training_microbatch=args.training_microbatch,
        center_microbatch=args.center_microbatch,
    )
    continuation_values = (
        args.continue_from_checkpoint,
        args.continue_from_checkpoint_sha256,
        args.continue_from_step,
    )
    continuation_present = tuple(value is not None for value in continuation_values)
    if any(continuation_present) and not all(continuation_present):
        raise ValueError("Continuation checkpoint arguments must be provided together")
    if args.smoke and any(continuation_present):
        raise ValueError("Smoke overrides cannot be combined with continuation")
    continuation_path = None
    if all(continuation_present):
        continuation_path = Path(args.continue_from_checkpoint).expanduser().resolve()
        try:
            continuation_path.relative_to(ASSET_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"--continue-from-checkpoint must remain below {ASSET_ROOT}"
            ) from exc
    if args.stop_after_outer_step is not None:
        if not 0 <= args.stop_after_outer_step <= config.max_outer_steps:
            raise ValueError(
                "--stop-after-outer-step must lie in [0, max_outer_steps]"
            )
        if args.stop_after_outer_step % config.evaluation.every:
            raise ValueError(
                "--stop-after-outer-step must lie on an evaluation boundary"
            )
    if args.validate_only:
        result: dict[str, Any] = {"config": config.to_dict(), "geometry": geometry}
        if continuation_path is not None:
            checkpoint, continuation_metadata = load_continuation_checkpoint(
                path=continuation_path,
                expected_sha256=args.continue_from_checkpoint_sha256,
                expected_step=args.continue_from_step,
                target_config=config,
                mmap=True,
            )
            del checkpoint
            result["continuation"] = continuation_metadata
        print(json.dumps(result, indent=2))
        return
    if args.preflight:
        print(json.dumps(preflight_assets(config), indent=2))
        return

    device, rank, local_rank, world_size = setup_distributed()
    stop_requested = False

    def request_stop(signum, frame):
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    try:
        geometry = execution_geometry(
            config,
            world_size=world_size,
            training_microbatch=args.training_microbatch,
            center_microbatch=args.center_microbatch,
        )
        run_dir, fingerprint = prepare_run_directory(
            config, rank=rank, world_size=world_size,
            execution_output_root=args.execution_output_root,
        )
        stop_request_file = None
        if args.stop_request_file is not None:
            stop_request_file = Path(args.stop_request_file).expanduser().resolve()
            if stop_request_file.parent != run_dir.resolve():
                raise ValueError("--stop-request-file must be directly inside run_dir")
        asset_identity = None
        preflight_error = None
        if rank == 0:
            try:
                asset_identity = preflight_assets(config)
            except Exception as exc:  # synchronize a rank-zero preflight failure
                preflight_error = f"{type(exc).__name__}: {exc}"
        preflight_ok = torch.tensor(
            int(rank != 0 or preflight_error is None), device=device, dtype=torch.int32
        )
        if world_size > 1:
            dist.broadcast(preflight_ok, src=0)
        if not bool(preflight_ok.item()):
            raise RuntimeError(preflight_error or "Rank-zero asset preflight failed")
        if rank == 0:
            assert asset_identity is not None
            atomic_write_json(run_dir / "asset_identity.json", asset_identity)
        barrier(world_size)
        rank0_print(rank, "comparison_geometry=" + json.dumps(geometry, sort_keys=True))

        checkpoint_payload_initial = torch.load(
            config.init_checkpoint, map_location="cpu", mmap=True, weights_only=False
        )
        initial_state = normalize_checkpoint_state(checkpoint_payload_initial)
        base_model = build_model_from_state(config, initial_state, device=device)
        del checkpoint_payload_initial, initial_state

        model_current_base = base_model
        model_old = copy.deepcopy(base_model)
        model_reference = copy.deepcopy(base_model)
        freeze_for_eval(model_old)
        freeze_for_eval(model_reference)
        model_current = wrap_ddp(
            model_current_base,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
        )
        optimizer_current = torch.optim.AdamW(
            model_current_base.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        model_center = model_center_ema = optimizer_center = None
        if config.method == "shiftcenter":
            model_center_base = copy.deepcopy(base_model)
            model_center_ema = copy.deepcopy(base_model)
            freeze_for_eval(model_center_ema)
            model_center = wrap_ddp(
                model_center_base,
                device=device,
                local_rank=local_rank,
                world_size=world_size,
            )
            optimizer_center = torch.optim.AdamW(
                model_center_base.parameters(),
                lr=config.center.learning_rate,
                weight_decay=config.center.weight_decay,
            )

        completed_pretrain = 0
        completed_outer = 0
        latest_path = run_dir / "checkpoints" / "latest.pt"
        continuation_record_path = run_dir / "continuation.json"
        if latest_path.exists():
            completed_pretrain, completed_outer = restore_checkpoint(
                path=latest_path,
                fingerprint=fingerprint,
                model_current=model_current,
                model_old=model_old,
                optimizer_current=optimizer_current,
                model_center=model_center,
                model_center_ema=model_center_ema,
                optimizer_center=optimizer_center,
            )
            rank0_print(
                rank,
                f"Resumed pretrain={completed_pretrain}, outer={completed_outer} from {latest_path}",
            )
            if continuation_path is not None:
                if not continuation_record_path.is_file():
                    raise RuntimeError(
                        "Continued run has a local checkpoint but no continuation record"
                    )
                continuation_record = json.loads(
                    continuation_record_path.read_text(encoding="utf-8")
                )
                expected_identity = {
                    "parent_checkpoint": str(continuation_path),
                    "parent_checkpoint_sha256": args.continue_from_checkpoint_sha256,
                    "parent_completed_outer_step": args.continue_from_step,
                    "target_config_fingerprint": fingerprint,
                }
                differences = {
                    name: {
                        "expected": value,
                        "actual": continuation_record.get(name),
                    }
                    for name, value in expected_identity.items()
                    if continuation_record.get(name) != value
                }
                if differences:
                    raise RuntimeError(
                        "Continuation record differs on resume: "
                        + json.dumps(differences, sort_keys=True)
                    )
                if completed_outer < args.continue_from_step:
                    raise RuntimeError(
                        "Local continuation checkpoint predates the parent checkpoint"
                    )
            elif continuation_record_path.exists():
                raise RuntimeError(
                    "Continued run resume requires the original continuation arguments"
                )
        elif continuation_path is not None:
            (
                completed_pretrain,
                completed_outer,
                continuation_record,
            ) = restore_continuation_checkpoint(
                path=continuation_path,
                expected_sha256=args.continue_from_checkpoint_sha256,
                expected_step=args.continue_from_step,
                target_config=config,
                model_current=model_current,
                model_old=model_old,
                optimizer_current=optimizer_current,
                model_center=model_center,
                model_center_ema=model_center_ema,
                optimizer_center=optimizer_center,
            )
            if rank == 0:
                atomic_write_json(continuation_record_path, continuation_record)
            maybe_save_checkpoint(
                force=True,
                progress=completed_outer,
                phase="outer",
                rank=rank,
                world_size=world_size,
                run_dir=run_dir,
                config=config,
                fingerprint=fingerprint,
                completed_pretrain_updates=completed_pretrain,
                completed_outer_step=completed_outer,
                model_current=model_current,
                model_old=model_old,
                optimizer_current=optimizer_current,
                model_center=model_center,
                model_center_ema=model_center_ema,
                optimizer_center=optimizer_center,
            )
            rank0_print(
                rank,
                f"Continued pretrain={completed_pretrain}, outer={completed_outer} "
                f"from {continuation_path}",
            )
        if completed_pretrain > config.center.pretrain_updates:
            raise RuntimeError("Checkpoint pretrain progress exceeds this config")
        if completed_outer > config.max_outer_steps:
            raise RuntimeError("Checkpoint outer progress exceeds this config")
        if rank == 0:
            append_jsonl(run_dir / "execution_history.jsonl", {
                "timestamp_unix": time.time(),
                "config_fingerprint": fingerprint,
                "completed_pretrain_updates": completed_pretrain,
                "completed_outer_step": completed_outer,
                "geometry": geometry,
                "gpu_name_rank0": torch.cuda.get_device_name(device),
            })
            truncate_jsonl(
                run_dir / "pretrain_metrics.jsonl",
                key="pretrain_update",
                completed=completed_pretrain,
            )
            truncate_jsonl(
                run_dir / "metrics.jsonl",
                key="outer_step",
                completed=completed_outer,
            )
        barrier(world_size)
        stop_requested = synchronize_stop_requested(
            stop_requested,
            request_file=stop_request_file,
            device=device,
            world_size=world_size,
        )

        segment_boundary_reached = (
            args.stop_after_outer_step is not None
            and completed_outer >= args.stop_after_outer_step
        )

        if (
            not stop_requested
            and not segment_boundary_reached
            and config.method == "shiftcenter"
            and completed_pretrain < config.center.pretrain_updates
        ):
            pretrain_dataset = load_or_collect_pretrain_dataset(
                model_old=model_old,
                config=config,
                run_dir=run_dir,
                fingerprint=fingerprint,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            for update in range(completed_pretrain + 1, config.center.pretrain_updates + 1):
                started = time.time()
                times = continuous_time_table(
                    num_examples=int(pretrain_dataset["endpoints"].shape[0]),
                    time_count=config.center.pretrain_timesteps_per_endpoint,
                    seed=config.seed + update * 10_000_019 + 2_000_003 + rank * 17,
                    device=device,
                )
                metrics = run_center_update(
                    model_center=model_center,
                    model_center_ema=model_center_ema,
                    optimizer_center=optimizer_center,
                    dataset=pretrain_dataset,
                    time_table=times,
                    config=config,
                    phase_seed=config.seed + update * 100_003 + 2_000_003,
                    microbatch=geometry["center_microbatch"],
                    rank=rank,
                    world_size=world_size,
                    device=device,
                )
                reduced = reduce_mean_metrics(metrics, device=device, world_size=world_size)
                record = {
                    "pretrain_update": update,
                    "optimizer_updates": 1,
                    "timesteps_per_endpoint": config.center.pretrain_timesteps_per_endpoint,
                    "records_global": int(reduced["center_records"] * world_size),
                    "seconds": time.time() - started,
                    **reduced,
                }
                if rank == 0:
                    append_jsonl(run_dir / "pretrain_metrics.jsonl", record)
                    print("PRETRAIN_METRICS " + json.dumps(record, sort_keys=True), flush=True)
                completed_pretrain = update
                stop_requested = synchronize_stop_requested(
                    stop_requested,
                    request_file=stop_request_file,
                    device=device,
                    world_size=world_size,
                )
                maybe_save_checkpoint(
                    force=stop_requested or update == config.center.pretrain_updates,
                    progress=update,
                    phase="pretrain",
                    rank=rank,
                    world_size=world_size,
                    run_dir=run_dir,
                    config=config,
                    fingerprint=fingerprint,
                    completed_pretrain_updates=completed_pretrain,
                    completed_outer_step=completed_outer,
                    model_current=model_current,
                    model_old=model_old,
                    optimizer_current=optimizer_current,
                    model_center=model_center,
                    model_center_ema=model_center_ema,
                    optimizer_center=optimizer_center,
                )
                if stop_requested:
                    break
            del pretrain_dataset

        stop_requested = synchronize_stop_requested(
            stop_requested,
            request_file=stop_request_file,
            device=device,
            world_size=world_size,
        )
        if not stop_requested and not segment_boundary_reached:
            vae_snapshot = snapshot_download(
                config.vae_name,
                revision=config.vae_revision,
                local_files_only=True,
            )
            pickscore_model_snapshot = snapshot_download(
                config.reward.model_name,
                revision=config.reward.model_revision,
                local_files_only=True,
            )
            pickscore_processor_snapshot = snapshot_download(
                config.reward.processor_name,
                revision=config.reward.processor_revision,
                local_files_only=True,
            )
            vae = AutoencoderKL.from_pretrained(
                vae_snapshot, local_files_only=True
            ).to(device)
            freeze_for_eval(vae)
            reward_fn = PickScoreImageNetReward(
                device=device,
                dtype=torch.float32,
                processor_name=pickscore_processor_snapshot,
                model_name=pickscore_model_snapshot,
                prompt_template=config.reward.prompt_template,
                divide_by_26=config.reward.divide_by_26,
                cache_text_embeddings=True,
                return_raw=False,
                batch_size=config.reward.batch_size,
                local_files_only=True,
            )

            for outer_step in range(completed_outer + 1, config.max_outer_steps + 1):
                outer_started = time.time()
                torch.cuda.reset_peak_memory_stats(device)
                dataset, rollout_metrics, preview = collect_reward_rollout(
                    model_old=model_old,
                    vae=vae,
                    reward_fn=reward_fn,
                    config=config,
                    outer_step=outer_step,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                )
                if rank == 0 and outer_step == completed_outer + 1:
                    torchvision.utils.save_image(
                        preview,
                        run_dir / "previews" / f"outer_{outer_step:06d}.png",
                        nrow=config.rollout.samples_per_class,
                    )
                del preview

                center_metrics = {
                    "center_loss": 0.0,
                    "center_grad_norm": 0.0,
                    "center_records": 0.0,
                    "center_microbatches": 0.0,
                }
                center_snapshot = None
                if config.method == "shiftcenter":
                    center_times = rollout_time_table(
                        config=config,
                        num_examples=int(dataset["endpoints"].shape[0]),
                        inner_epochs=config.center.inner_epochs,
                        fraction=config.center.timestep_fraction,
                        seed=config.seed + outer_step * 10_000_019 + 200_003 + rank * 17,
                        device=device,
                    )
                    center_metrics = run_center_update(
                        model_center=model_center,
                        model_center_ema=model_center_ema,
                        optimizer_center=optimizer_center,
                        dataset=dataset,
                        time_table=center_times,
                        config=config,
                        phase_seed=config.seed + outer_step * 100_003 + 200_003,
                        microbatch=geometry["center_microbatch"],
                        rank=rank,
                        world_size=world_size,
                        device=device,
                    )
                    center_snapshot = model_center_ema

                policy_metrics = run_policy_update(
                    model_current=model_current,
                    model_old=model_old,
                    model_reference=model_reference,
                    model_center_snapshot=center_snapshot,
                    optimizer_current=optimizer_current,
                    dataset=dataset,
                    config=config,
                    outer_step=outer_step,
                    microbatch=geometry["training_microbatch"],
                    rank=rank,
                    world_size=world_size,
                    device=device,
                )
                decay = old_policy_decay(
                    outer_step, config.training.old_policy_decay_type
                )
                update_ema(model_old, model_current, decay)
                freeze_for_eval(model_old)
                del dataset

                local = {**rollout_metrics, **center_metrics, **policy_metrics}
                reduced = reduce_mean_metrics(local, device=device, world_size=world_size)
                elapsed = torch.tensor(time.time() - outer_started, device=device, dtype=torch.float64)
                peak = torch.tensor(
                    torch.cuda.max_memory_allocated(device), device=device, dtype=torch.float64
                )
                if world_size > 1:
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                record = {
                    "outer_step": outer_step,
                    "method": config.method,
                    "rollout_cfg_scale": config.rollout.cfg_scale,
                    "rollout_sampling_steps": config.rollout.sampling_steps,
                    "rollout_samples_global": geometry["endpoints_global"],
                    "class_groups_global": config.rollout.groups_global,
                    "samples_per_class": config.rollout.samples_per_class,
                    "train_timesteps_per_endpoint": geometry[
                        "policy_timesteps_per_endpoint"
                    ],
                    "policy_updates": 1,
                    "current_branches_updated": (
                        2 if config.method == "dual" else 1
                    ),
                    "center_updates": int(config.method == "shiftcenter"),
                    "completed_center_pretrain_updates": completed_pretrain,
                    "old_policy_decay": decay,
                    "world_size": world_size,
                    "policy_records_global": int(
                        reduced["policy_records"] * world_size
                    ),
                    "center_records_global": int(
                        reduced["center_records"] * world_size
                    ),
                    "outer_seconds": float(elapsed),
                    "peak_memory_allocated_bytes": int(peak),
                    **reduced,
                }
                if not all(
                    math.isfinite(float(value))
                    for value in record.values()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ):
                    raise FloatingPointError(f"Non-finite outer metrics: {record}")
                if rank == 0:
                    append_jsonl(run_dir / "metrics.jsonl", record)
                    print("OUTER_METRICS " + json.dumps(record, sort_keys=True), flush=True)
                completed_outer = outer_step
                stop_requested = synchronize_stop_requested(
                    stop_requested,
                    request_file=stop_request_file,
                    device=device,
                    world_size=world_size,
                )
                maybe_save_checkpoint(
                    force=stop_requested or outer_step == config.max_outer_steps,
                    progress=outer_step,
                    phase="outer",
                    rank=rank,
                    world_size=world_size,
                    run_dir=run_dir,
                    config=config,
                    fingerprint=fingerprint,
                    completed_pretrain_updates=completed_pretrain,
                    completed_outer_step=completed_outer,
                    model_current=model_current,
                    model_old=model_old,
                    optimizer_current=optimizer_current,
                    model_center=model_center,
                    model_center_ema=model_center_ema,
                    optimizer_center=optimizer_center,
                )
                if stop_requested:
                    break
                if (
                    args.stop_after_outer_step is not None
                    and outer_step >= args.stop_after_outer_step
                ):
                    segment_boundary_reached = True
                    break

        barrier(world_size)
        if rank == 0:
            atomic_write_json(
                run_dir / "run_summary.json",
                {
                    "method": config.method,
                    "completed_pretrain_updates": completed_pretrain,
                    "requested_pretrain_updates": config.center.pretrain_updates,
                    "completed_outer_step": completed_outer,
                    "requested_outer_steps": config.max_outer_steps,
                    "stopped_on_signal": stop_requested,
                    "segment_boundary_reached": segment_boundary_reached,
                    "stop_after_outer_step": args.stop_after_outer_step,
                },
            )
        rank0_print(rank, f"SiT XL CFG comparison stopped at {run_dir}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
