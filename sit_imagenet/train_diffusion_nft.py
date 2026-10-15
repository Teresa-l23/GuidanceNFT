from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any, Callable, Collection, Mapping
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from diffusers import AutoencoderKL

from diffusion_nft.config import NFTConfig, load_config
from diffusion_nft.objective import (
    diffusion_nft_loss,
    group_normalize_rewards,
    make_flow_matching_batch,
    mse_per_example,
    nft_training_timestep_count,
    old_policy_decay,
    sample_nft_timesteps,
    sampler_timesteps,
)
from diffusion_nft.rewards import build_reward
from models import SiT_models
from sampler import euler_maruyama_sampler, euler_sampler


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiffusionNFT for model-guided SiT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--output-root")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--max-outer-steps", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--groups-global", type=int)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--rollout-batch-size", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--center-batch-size", type=int)
    parser.add_argument("--timestep-fraction", type=float)
    parser.add_argument("--center-timestep-fraction", type=float)
    parser.add_argument("--checkpoint-save-every", type=int)
    parser.add_argument("--checkpoint-keep", type=int)
    parser.add_argument(
        "--checkpoint-milestones",
        help="Comma-separated immutable outer steps; the configured final step is always added.",
    )
    parser.add_argument("--durable-checkpoint-dir")
    parser.add_argument("--durable-run-name")
    parser.add_argument("--durable-source-id")
    parser.add_argument("--continue-from-checkpoint")
    parser.add_argument("--continue-from-checkpoint-sha256")
    parser.add_argument("--continue-from-step", type=int)
    parser.add_argument(
        "--allow-interrupted-checkpoint-relocation",
        action="store_true",
        help=(
            "Relocate a non-terminal rolling checkpoint without changing its "
            "scientific, optimizer, or target-step configuration."
        ),
    )
    parser.add_argument(
        "--inherit-run-relocation-marker",
        help=(
            "Pipeline marker for resuming a same-disk checkpoint under a new "
            "run root; only the output root and init-checkpoint location may change."
        ),
    )
    parser.add_argument("--reward-name", choices=["pickscore_imagenet", "constant"])
    return parser.parse_args()


def config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    possible = {
        "run_name": args.run_name,
        "output_root": args.output_root,
        "init_checkpoint": args.init_checkpoint,
        "max_outer_steps": args.max_outer_steps,
        "rollout.sampling_steps": args.sampling_steps,
        "rollout.groups_global": args.groups_global,
        "rollout.samples_per_class": args.samples_per_class,
        "rollout.batch_size": args.rollout_batch_size,
        "training.batch_size": args.train_batch_size,
        "center.batch_size": args.center_batch_size,
        "training.timestep_fraction": args.timestep_fraction,
        "center.timestep_fraction": args.center_timestep_fraction,
        "checkpoint.save_every": args.checkpoint_save_every,
        "checkpoint.keep": args.checkpoint_keep,
        "reward.name": args.reward_name,
    }
    return {key: value for key, value in possible.items() if value is not None}


def setup_distributed() -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("DiffusionNFT training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", timeout=torch.distributed.constants.default_pg_timeout)
    return device, rank, local_rank, world_size


def rank0_print(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier(device_ids=[torch.cuda.current_device()])


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def autocast_context(config: NFTConfig, device: torch.device):
    if config.mixed_precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _unwrap_checkpoint(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a state-dict mapping")
    for key in ("ema", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break
    result: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        normalized = key.removeprefix("module.")
        result[normalized] = value
    if not result:
        raise ValueError("Checkpoint contains no tensor state dict")
    return result


def build_model(config: NFTConfig, state: Mapping[str, torch.Tensor]):
    if config.model_key not in SiT_models:
        raise KeyError(f"Unknown model_key {config.model_key!r}; choices={sorted(SiT_models)}")
    model = SiT_models[config.model_key](
        input_size=config.image_size // 8,
        class_dropout_prob=0.0,
        learn_sigma=False,
        num_classes=config.num_classes,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Initial checkpoint does not exactly match {config.model_key}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model


def freeze_for_eval(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def update_ema(target, source, decay: float) -> None:
    source = unwrap_model(source)
    target_parameters = dict(target.named_parameters())
    source_parameters = dict(source.named_parameters())
    if target_parameters.keys() != source_parameters.keys():
        raise RuntimeError("EMA model parameter names differ")
    for name, parameter in source_parameters.items():
        target_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())
    if target_buffers.keys() != source_buffers.keys():
        raise RuntimeError("EMA model buffer names differ")
    for name, value in source_buffers.items():
        target_buffers[name].copy_(value)


@torch.no_grad()
def update_old_policy(old_model, current_model, decay: float) -> None:
    update_ema(old_model, current_model, decay)
    freeze_for_eval(old_model)


def wrap_ddp(model, device: torch.device, local_rank: int, world_size: int):
    if world_size == 1:
        return model
    return DDP(model, device_ids=[local_rank], output_device=local_rank)


def config_fingerprint(config: NFTConfig) -> str:
    return config_mapping_fingerprint(config.to_dict())


def config_mapping_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def continuation_config_changes(
    parent_config: Mapping[str, Any], target_config: NFTConfig, *, completed_step: int
) -> dict[str, dict[str, Any]]:
    """Validate an exact terminal-checkpoint continuation into a new run.

    Scientific and optimizer settings must be byte-for-byte JSON equivalent.  A
    continuation may only move the run directory and increase max_outer_steps.
    """

    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise TypeError("continuation completed step must be an integer")
    if completed_step < 0:
        raise ValueError("continuation completed step must be non-negative")
    parent = dict(parent_config)
    target = target_config.to_dict()
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
    if parent.get("output_root") == target.get("output_root"):
        raise RuntimeError("Continuation must use a new output_root")

    allowed = {"output_root", "max_outer_steps"}
    incompatible = {
        key: {"parent": parent.get(key), "target": target.get(key)}
        for key in sorted(parent)
        if key not in allowed and parent.get(key) != target.get(key)
    }
    if incompatible:
        raise RuntimeError(
            "Continuation changes scientific or optimizer config: "
            + json.dumps(incompatible, sort_keys=True)
        )
    return {
        key: {"parent": parent[key], "target": target[key]}
        for key in sorted(allowed)
        if parent[key] != target[key]
    }


def interrupted_relocation_config_changes(
    parent_config: Mapping[str, Any],
    target_config: NFTConfig,
    *,
    completed_step: int,
) -> dict[str, dict[str, Any]]:
    """Validate relocation of an authenticated non-terminal rolling checkpoint."""

    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise TypeError("interrupted checkpoint completed step must be an integer")
    if completed_step < 0:
        raise ValueError("interrupted checkpoint completed step must be non-negative")
    parent = dict(parent_config)
    target = target_config.to_dict()
    if set(parent) != set(target):
        raise RuntimeError("Interrupted checkpoint config keys differ")
    parent_max = parent.get("max_outer_steps")
    if isinstance(parent_max, bool) or not isinstance(parent_max, int):
        raise RuntimeError("Interrupted checkpoint max_outer_steps is invalid")
    if completed_step >= parent_max:
        raise RuntimeError(
            "Interrupted checkpoint must be strictly before its configured final step"
        )
    if target_config.max_outer_steps != parent_max:
        raise RuntimeError(
            "Interrupted checkpoint relocation cannot change max_outer_steps"
        )
    if parent.get("output_root") == target.get("output_root"):
        raise RuntimeError("Interrupted checkpoint relocation requires a new output_root")

    allowed = {"init_checkpoint", "output_root"}
    incompatible = {
        key: {"parent": parent.get(key), "target": target.get(key)}
        for key in sorted(parent)
        if key not in allowed and parent.get(key) != target.get(key)
    }
    if incompatible:
        raise RuntimeError(
            "Interrupted checkpoint relocation changes scientific, optimizer, or "
            "target-step config: "
            + json.dumps(incompatible, sort_keys=True)
        )
    return {
        key: {"parent": parent[key], "target": target[key]}
        for key in sorted(allowed)
        if parent[key] != target[key]
    }


def inherited_run_relocation_config_changes(
    parent_config: Mapping[str, Any],
    target_config: NFTConfig,
    *,
    completed_step: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate a same-disk retry under a new immutable run name."""

    parent = dict(parent_config)
    target = target_config.to_dict()
    if set(parent) != set(target):
        raise RuntimeError("Inherited checkpoint config keys differ")
    if parent.get("output_root") == target.get("output_root"):
        raise RuntimeError("Inherited checkpoint requires a new output_root")
    if parent.get("max_outer_steps") != target.get("max_outer_steps"):
        raise RuntimeError("Inherited checkpoint cannot change max_outer_steps")
    allowed = {"init_checkpoint", "output_root"}
    incompatible = {
        key: {"parent": parent.get(key), "target": target.get(key)}
        for key in sorted(parent)
        if key not in allowed and parent.get(key) != target.get(key)
    }
    if incompatible:
        raise RuntimeError(
            "Inherited checkpoint changes scientific or optimizer config: "
            + json.dumps(incompatible, sort_keys=True)
        )
    if completed_step is not None:
        if isinstance(completed_step, bool) or not isinstance(completed_step, int):
            raise TypeError("Inherited checkpoint completed step must be an integer")
        maximum = target_config.max_outer_steps
        if completed_step < 0 or completed_step > maximum:
            raise RuntimeError(
                "Inherited checkpoint completed step is outside the target range"
            )
    return {
        key: {"parent": parent[key], "target": target[key]}
        for key in sorted(allowed)
        if parent[key] != target[key]
    }


def load_inherited_run_relocation(
    marker_path: str | Path, target_config: NFTConfig
) -> dict[str, Any]:
    """Authenticate the old/new system-run roots and copied effective config."""

    marker = Path(marker_path).expanduser().resolve()
    destination_root = Path(target_config.output_root).expanduser().resolve()
    expected_marker = destination_root / ".pipeline-inherited.json"
    if marker != expected_marker or not marker.is_file() or marker.is_symlink():
        raise RuntimeError("Inherited-run marker is outside the target output root")
    identity = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(identity, Mapping) or set(identity) != {
        "schema_version", "source_run_name", "destination_run_name"
    } or identity.get("schema_version") != 1:
        raise RuntimeError("Inherited-run marker has an unsupported schema")
    if identity.get("destination_run_name") != destination_root.name:
        raise RuntimeError("Inherited-run destination name differs")
    source_name = identity.get("source_run_name")
    if (
        not isinstance(source_name, str)
        or not source_name
        or Path(source_name).name != source_name
    ):
        raise RuntimeError("Inherited-run source name is unsafe")
    source_root = (destination_root.parent / source_name).resolve()
    if source_root.parent != destination_root.parent or not source_root.is_dir():
        raise RuntimeError("Inherited-run source root is missing")
    source_effective_path = source_root / target_config.run_name / "effective_config.json"
    destination_effective_path = (
        destination_root / target_config.run_name / "effective_config.json"
    )
    for path in (source_effective_path, destination_effective_path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Inherited effective config is missing or unsafe: {path}")
    source_effective = json.loads(source_effective_path.read_text(encoding="utf-8"))
    destination_effective = json.loads(
        destination_effective_path.read_text(encoding="utf-8")
    )
    if not isinstance(source_effective, Mapping):
        raise RuntimeError("Inherited source effective config is invalid")
    parent_config = source_effective.get("config")
    if not isinstance(parent_config, Mapping):
        raise RuntimeError("Inherited source config is missing")
    parent_fingerprint = config_mapping_fingerprint(parent_config)
    if source_effective.get("fingerprint") != parent_fingerprint:
        raise RuntimeError("Inherited source config fingerprint differs")
    inherited_run_relocation_config_changes(parent_config, target_config)
    target_effective = {
        "fingerprint": config_fingerprint(target_config),
        "config": target_config.to_dict(),
    }
    if destination_effective not in (source_effective, target_effective):
        raise RuntimeError("Inherited destination effective config differs")
    source_latest = source_root / "checkpoints" / "latest.pt"
    destination_latest = destination_root / "checkpoints" / "latest.pt"
    for path in (source_latest, destination_latest):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Inherited rolling checkpoint is missing or unsafe: {path}")
    return {
        "identity": dict(identity),
        "parent_config": dict(parent_config),
        "parent_fingerprint": parent_fingerprint,
        "source_latest": source_latest,
        "destination_latest": destination_latest,
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def parse_checkpoint_milestones(
    value: str | None, *, final_step: int
) -> frozenset[int] | None:
    """Parse an optional immutable milestone set and always include final_step."""

    if value is None:
        return None
    if final_step < 1:
        raise ValueError("final_step must be positive")
    pieces = value.split(",")
    if not pieces or any(not piece or not piece.isdigit() for piece in pieces):
        raise ValueError("checkpoint milestones must be comma-separated integers")
    steps = [int(piece) for piece in pieces]
    if len(set(steps)) != len(steps):
        raise ValueError("checkpoint milestones must not contain duplicates")
    if any(step < 0 or step > final_step for step in steps):
        raise ValueError(f"checkpoint milestones must be in [0, {final_step}]")
    return frozenset((*steps, final_step))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_checkpoint_immutable(source: Path, destination: Path) -> tuple[int, str]:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"checkpoint source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError(f"checkpoint destination is unsafe: {destination}")
    source_bytes = source.stat().st_size
    if source_bytes <= 0:
        raise ValueError(f"checkpoint source is empty: {source}")
    if destination.is_file():
        source_sha256 = _sha256_file(source)
        if (
            destination.stat().st_size != source_bytes
            or _sha256_file(destination) != source_sha256
        ):
            raise ValueError(f"immutable durable checkpoint differs: {destination}")
        return source_bytes, source_sha256

    filesystem = os.statvfs(destination.parent)
    available_bytes = filesystem.f_bavail * filesystem.f_frsize
    required_bytes = source_bytes + 1024**3
    if available_bytes < required_bytes:
        raise OSError(
            f"insufficient durable-checkpoint space at {destination.parent}: "
            f"available={available_bytes} required={required_bytes}"
        )
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while chunk := reader.read(8 * 1024 * 1024):
                writer.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if copied != source_bytes:
            raise OSError(
                f"checkpoint changed while copying: expected {source_bytes}, copied {copied}"
            )
        os.replace(temporary, destination)
        destination.chmod(0o440)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return copied, digest.hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def publish_durable_checkpoint(
    source: Path,
    destination_dir: Path,
    *,
    step: int,
    expected_fingerprint: str,
    run_name: str,
    source_id: str,
) -> dict[str, Any]:
    """Atomically publish and validate one immutable checkpoint on durable storage."""

    if step < 0:
        raise ValueError("durable checkpoint step must be non-negative")
    if not run_name or not source_id:
        raise ValueError("run_name and source_id are required")
    if destination_dir.is_symlink():
        raise ValueError(f"durable checkpoint directory must not be a symlink: {destination_dir}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload is not a mapping: {source}")
    if payload.get("config_fingerprint") != expected_fingerprint:
        raise ValueError(f"checkpoint fingerprint mismatch: {source}")
    if payload.get("completed_outer_step") != step:
        raise ValueError(f"checkpoint outer step mismatch: {source}")
    filename = f"outer_{step:06d}.pt"
    checkpoint_bytes, checkpoint_sha256 = _copy_checkpoint_immutable(
        source, destination_dir / filename
    )
    del payload
    return {
        "step": step,
        "filename": filename,
        "bytes": checkpoint_bytes,
        "sha256": checkpoint_sha256,
    }


def publish_durable_checkpoints(
    source_dir: Path,
    destination_dir: Path,
    *,
    steps: Collection[int],
    expected_fingerprint: str,
    run_name: str,
    source_id: str,
) -> dict[str, Any]:
    """Copy an exact, validated checkpoint set to persistent storage."""

    selected = tuple(sorted(set(steps)))
    if not selected or selected[0] < 0:
        raise ValueError("durable checkpoint steps must be non-empty and non-negative")
    if not run_name or not source_id:
        raise ValueError("run_name and source_id are required")
    if destination_dir.is_symlink():
        raise ValueError(f"durable checkpoint directory must not be a symlink: {destination_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for step in selected:
        filename = f"outer_{step:06d}.pt"
        records.append(
            publish_durable_checkpoint(
                source_dir / filename,
                destination_dir,
                step=step,
                expected_fingerprint=expected_fingerprint,
                run_name=run_name,
                source_id=source_id,
            )
        )
    manifest = {
        "schema_version": 1,
        "status": "verified",
        "kind": "model_guidance_durable_checkpoints",
        "run_name": run_name,
        "source_id": source_id,
        "config_fingerprint": expected_fingerprint,
        "steps": list(selected),
        "final_step": selected[-1],
        "checkpoints": records,
    }
    manifest_path = destination_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest_path.chmod(0o440)
    return manifest


def prepare_run_directory(
    config: NFTConfig,
    rank: int,
    world_size: int,
    *,
    inherited_relocation: Mapping[str, Any] | None = None,
) -> tuple[Path, str]:
    run_dir = config.run_dir.resolve()
    fingerprint = config_fingerprint(config)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "previews").mkdir(exist_ok=True)
        effective_path = run_dir / "effective_config.json"
        effective = {"fingerprint": fingerprint, "config": config.to_dict()}
        if effective_path.exists():
            existing = json.loads(effective_path.read_text(encoding="utf-8"))
            if existing != effective:
                parent_config = (
                    inherited_relocation.get("parent_config")
                    if inherited_relocation is not None
                    else None
                )
                parent_effective = (
                    {
                        "fingerprint": inherited_relocation.get("parent_fingerprint"),
                        "config": parent_config,
                    }
                    if isinstance(parent_config, Mapping)
                    else None
                )
                if existing != parent_effective:
                    raise RuntimeError(
                        f"Run directory already has a different effective config: {run_dir}"
                    )
                atomic_write_json(effective_path, effective)
        else:
            atomic_write_json(effective_path, effective)
    barrier(world_size)
    return run_dir, fingerprint


def reduce_mean_metrics(
    metrics: Mapping[str, float], device: torch.device, world_size: int
) -> dict[str, float]:
    names = sorted(metrics)
    values = torch.tensor([metrics[name] for name in names], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world_size
    return {name: float(value) for name, value in zip(names, values.cpu().tolist())}


def all_ranks_finite(value: torch.Tensor, world_size: int) -> bool:
    flag = torch.tensor(
        int(torch.isfinite(value.detach()).all().item()),
        device=value.device,
        dtype=torch.int32,
    )
    if world_size > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


@torch.no_grad()
def collect_rollout(
    *,
    model_old,
    vae,
    reward_fn,
    config: NFTConfig,
    outer_step: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float], torch.Tensor]:
    groups_global = config.rollout.groups_global
    repeats = config.rollout.samples_per_class
    if groups_global % world_size != 0:
        raise ValueError("rollout.groups_global must be divisible by world_size")
    groups = groups_global // world_size
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(config.seed + outer_step * 100_003)
    classes_global = torch.randperm(config.num_classes, generator=cpu_generator)[:groups_global]
    classes = classes_global[rank * groups : (rank + 1) * groups]
    labels_cpu = classes.repeat_interleave(repeats)

    sample_generator = torch.Generator(device=device)
    sample_generator.manual_seed(config.seed + outer_step * 1_000_003 + rank * 100_000_007)
    latent_size = config.image_size // 8
    endpoint_parts: list[torch.Tensor] = []
    reward_parts: list[torch.Tensor] = []
    raw_reward_parts: list[torch.Tensor] = []
    preview_parts: list[torch.Tensor] = []
    preview_remaining = config.rollout.preview_samples
    for start in range(0, labels_cpu.numel(), config.rollout.batch_size):
        end = min(start + config.rollout.batch_size, labels_cpu.numel())
        labels = labels_cpu[start:end].to(device=device, dtype=torch.long)
        noise = torch.randn(
            labels.shape[0],
            4,
            latent_size,
            latent_size,
            generator=sample_generator,
            device=device,
            dtype=torch.float32,
        )
        with autocast_context(config, device):
            if config.rollout.sampler == "sde":
                endpoints = euler_maruyama_sampler(
                    model_old,
                    noise,
                    labels,
                    num_steps=config.rollout.sampling_steps,
                    cfg_scale=1.0,
                )
            else:
                endpoints = euler_sampler(
                    model_old,
                    noise,
                    labels,
                    num_steps=config.rollout.sampling_steps,
                    cfg_scale=1.0,
                )
            endpoints = endpoints.float()
            decoded = vae.decode(endpoints / 0.18215).sample
        images_01 = (decoded.float().clamp(-1.0, 1.0) + 1.0) / 2.0
        reward_outputs = reward_fn(images_01, labels)
        rewards_chunk = reward_outputs["avg"].detach().float()
        raw_chunk = reward_outputs.get("raw", rewards_chunk).detach().float()
        endpoint_parts.append(endpoints.detach().cpu())
        reward_parts.append(rewards_chunk.cpu())
        raw_reward_parts.append(raw_chunk.cpu())
        if preview_remaining > 0:
            take = min(preview_remaining, images_01.shape[0])
            preview_parts.append(images_01[:take].detach().cpu())
            preview_remaining -= take

    endpoints_cpu = torch.cat(endpoint_parts)
    rewards = torch.cat(reward_parts)
    raw_rewards = torch.cat(raw_reward_parts)
    preview = torch.cat(preview_parts)
    advantages = group_normalize_rewards(
        rewards.reshape(groups, repeats), epsilon=config.reward.advantage_epsilon
    ).reshape(-1)
    metrics = {
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "reward_raw_mean": float(raw_rewards.mean().item()),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()),
        "advantage_clip_fraction": float(
            (advantages.abs() >= config.training.advantage_clip).float().mean().item()
        ),
    }
    dataset = {
        "endpoints": endpoints_cpu,
        "labels": labels_cpu,
        "advantages": advantages,
        "rewards": rewards,
    }
    return dataset, metrics, preview


def _phase_batches(
    *,
    num_examples: int,
    batch_size: int,
    inner_epochs: int,
    timestep_count: int,
    seed: int,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for inner_epoch in range(inner_epochs):
        for time_draw in range(timestep_count):
            order = torch.randperm(num_examples, generator=generator)
            for start in range(0, num_examples, batch_size):
                yield inner_epoch, time_draw, order[start : start + batch_size]


def _fresh_bridge(
    endpoints: torch.Tensor,
    *,
    t: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if t.shape != (endpoints.shape[0],):
        raise ValueError(f"t must have shape [{endpoints.shape[0]}]")
    noise = torch.randn(
        endpoints.shape,
        generator=generator,
        device=endpoints.device,
        dtype=endpoints.dtype,
    )
    batch = make_flow_matching_batch(endpoints, t, noise)
    return t, batch.state, batch.target_velocity


def run_accumulated_phase(
    *,
    model_train,
    optimizer,
    dataset: Mapping[str, torch.Tensor],
    config: NFTConfig,
    outer_step: int,
    rank: int,
    world_size: int,
    device: torch.device,
    batch_size: int,
    inner_epochs: int,
    timestep_fraction: float,
    phase_seed_offset: int,
    max_grad_norm: float,
    loss_builder: Callable[
        [torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, Mapping[str, torch.Tensor]],
    ],
) -> tuple[dict[str, float], int]:
    num_examples = int(dataset["endpoints"].shape[0])
    time_grid = sampler_timesteps(
        config.rollout.sampler, config.rollout.sampling_steps
    ).to(device=device)
    timestep_count = nft_training_timestep_count(
        config.rollout.sampling_steps, timestep_fraction
    )
    total_records = num_examples * inner_epochs * timestep_count
    if total_records <= 0:
        raise ValueError("A training phase must contain records")
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    record_count = 0
    microbatches = 0
    cuda_generator = torch.Generator(device=device)
    cuda_generator.manual_seed(
        config.seed + outer_step * 10_000_019 + rank * 1_000_003 + phase_seed_offset
    )
    timestep_table = sample_nft_timesteps(
        time_grid,
        num_examples=num_examples,
        inner_epochs=inner_epochs,
        timestep_fraction=timestep_fraction,
        generator=cuda_generator,
    )
    model_train.train()
    schedule = _phase_batches(
        num_examples=num_examples,
        batch_size=batch_size,
        inner_epochs=inner_epochs,
        timestep_count=timestep_count,
        seed=config.seed + outer_step * 100_003 + rank * 10_007 + phase_seed_offset,
    )
    total_microbatches = inner_epochs * timestep_count * math.ceil(num_examples / batch_size)
    for microbatch_index, (inner_epoch, time_draw, indices) in enumerate(schedule):
        endpoints = dataset["endpoints"][indices].to(device=device, non_blocking=True)
        labels = dataset["labels"][indices].to(device=device, non_blocking=True)
        advantages = dataset["advantages"][indices].to(device=device, non_blocking=True)
        timestep_indices = indices.to(device=device, non_blocking=True)
        t = timestep_table[inner_epoch, timestep_indices, time_draw]
        t, state, target = _fresh_bridge(endpoints, t=t, generator=cuda_generator)
        is_last_microbatch = microbatch_index + 1 == total_microbatches
        sync_context = (
            nullcontext()
            if is_last_microbatch or not isinstance(model_train, DDP)
            else model_train.no_sync()
        )
        with sync_context:
            with autocast_context(config, device):
                loss, batch_metrics = loss_builder(state, labels, t, target, advantages)
                scaled_loss = loss * (endpoints.shape[0] / total_records)
            if not all_ranks_finite(loss, world_size):
                raise FloatingPointError("Non-finite loss in accumulated training phase")
            scaled_loss.backward()
        batch_records = int(endpoints.shape[0])
        record_count += batch_records
        microbatches += 1
        for name, value in batch_metrics.items():
            scalar = float(value.detach().float().item())
            sums[name] = sums.get(name, 0.0) + scalar * batch_records

    if record_count != total_records:
        raise RuntimeError(f"Accumulated {record_count} records, expected {total_records}")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        unwrap_model(model_train).parameters(), max_grad_norm
    )
    if not all_ranks_finite(torch.as_tensor(grad_norm, device=device), world_size):
        raise FloatingPointError("Non-finite gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metrics = {name: value / total_records for name, value in sums.items()}
    metrics["grad_norm"] = float(torch.as_tensor(grad_norm).float().item())
    metrics["records"] = float(total_records)
    metrics["microbatches"] = float(microbatches)
    return metrics, total_records


def run_center_update(
    *,
    model_center,
    model_center_ema,
    optimizer,
    dataset: Mapping[str, torch.Tensor],
    config: NFTConfig,
    outer_step: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, float]:
    def loss_builder(state, labels, t, target, advantages):
        del advantages
        prediction = model_center(state, t, labels)
        per_example = mse_per_example(prediction, target)
        loss = per_example.mean()
        return loss, {"loss": loss.detach()}

    metrics, _ = run_accumulated_phase(
        model_train=model_center,
        optimizer=optimizer,
        dataset=dataset,
        config=config,
        outer_step=outer_step,
        rank=rank,
        world_size=world_size,
        device=device,
        batch_size=config.center.batch_size,
        inner_epochs=config.center.inner_epochs,
        timestep_fraction=config.center.timestep_fraction,
        phase_seed_offset=200_003,
        max_grad_norm=config.training.max_grad_norm,
        loss_builder=loss_builder,
    )
    update_ema(model_center_ema, model_center, config.center.ema_decay)
    freeze_for_eval(model_center_ema)
    return {f"center_{name}": value for name, value in metrics.items()}


def run_policy_update(
    *,
    model_current,
    model_old,
    model_reference,
    model_center_snapshot,
    optimizer,
    dataset: Mapping[str, torch.Tensor],
    config: NFTConfig,
    outer_step: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, float]:
    model_old.eval()
    model_reference.eval()
    if model_center_snapshot is not None:
        model_center_snapshot.eval()

    def loss_builder(state, labels, t, target, advantages):
        with torch.no_grad():
            old_prediction = model_old(state, t, labels).detach()
            reference_prediction = model_reference(state, t, labels).detach()
            center_prediction = (
                model_center_snapshot(state, t, labels).detach()
                if model_center_snapshot is not None
                else None
            )
        current_prediction = model_current(state, t, labels)
        return diffusion_nft_loss(
            current_prediction=current_prediction,
            old_prediction=old_prediction,
            reference_prediction=reference_prediction,
            target_velocity=target,
            advantages=advantages,
            beta=config.training.beta,
            advantage_clip=config.training.advantage_clip,
            kl_beta=config.training.kl_beta,
            center_prediction=center_prediction,
        )

    metrics, _ = run_accumulated_phase(
        model_train=model_current,
        optimizer=optimizer,
        dataset=dataset,
        config=config,
        outer_step=outer_step,
        rank=rank,
        world_size=world_size,
        device=device,
        batch_size=config.training.batch_size,
        inner_epochs=config.training.inner_epochs,
        timestep_fraction=config.training.timestep_fraction,
        phase_seed_offset=400_009,
        max_grad_norm=config.training.max_grad_norm,
        loss_builder=loss_builder,
    )
    return {f"policy_{name}": value for name, value in metrics.items()}


def checkpoint_payload(
    *,
    config: NFTConfig,
    fingerprint: str,
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
    completed_outer_step: int,
    keep: int,
    save_milestone: bool,
    preserve_milestone_steps: Collection[int] | None = None,
) -> tuple[Path, Path | None]:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (".latest.*.tmp", ".outer_*.tmp"):
        for stale in checkpoint_dir.glob(pattern):
            stale.unlink(missing_ok=True)
    latest = checkpoint_dir / "latest.pt"
    latest_tmp = checkpoint_dir / f".latest.{os.getpid()}.tmp"
    with latest_tmp.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(latest_tmp, latest)

    step_path: Path | None = None
    if save_milestone:
        step_path = checkpoint_dir / f"outer_{completed_outer_step:06d}.pt"
        step_tmp = checkpoint_dir / f".outer_{completed_outer_step:06d}.{os.getpid()}.tmp"
        step_tmp.unlink(missing_ok=True)
        try:
            os.link(latest, step_tmp)
        except OSError:
            with step_tmp.open("wb") as handle:
                torch.save(dict(payload), handle)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(step_tmp, step_path)
        milestones = sorted(checkpoint_dir.glob("outer_*.pt"))
        if preserve_milestone_steps is None:
            stale_milestones = milestones[:-keep]
        else:
            preserved = set(preserve_milestone_steps)
            stale_milestones = [
                path
                for path in milestones
                if int(path.stem.removeprefix("outer_")) not in preserved
            ]
        for stale in stale_milestones:
            stale.unlink()

    directory_fd = os.open(checkpoint_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return latest, step_path


def _restore_checkpoint_state(
    checkpoint: Mapping[str, Any],
    *,
    model_current,
    model_old,
    optimizer_current,
    model_center,
    model_center_ema,
    optimizer_center,
) -> None:
    unwrap_model(model_current).load_state_dict(checkpoint["model_current"], strict=True)
    model_old.load_state_dict(checkpoint["model_old"], strict=True)
    optimizer_current.load_state_dict(checkpoint["optimizer_current"])
    if model_center is not None:
        if checkpoint.get("model_center") is None or checkpoint.get("model_center_ema") is None:
            raise RuntimeError("ShiftCenter checkpoint is missing center state")
        unwrap_model(model_center).load_state_dict(checkpoint["model_center"], strict=True)
        model_center_ema.load_state_dict(checkpoint["model_center_ema"], strict=True)
        optimizer_center.load_state_dict(checkpoint["optimizer_center"])
    elif any(
        checkpoint.get(name) is not None
        for name in ("model_center", "model_center_ema", "optimizer_center")
    ):
        raise RuntimeError("Original-center checkpoint unexpectedly contains center state")


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
    target_config: NFTConfig | None = None,
    inherited_relocation: Mapping[str, Any] | None = None,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Resume checkpoint is not a mapping")
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("Resume checkpoint schema_version differs")
    completed_step = checkpoint.get("completed_outer_step")
    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise RuntimeError("Resume checkpoint completed_outer_step is invalid")
    checkpoint_fingerprint = checkpoint.get("config_fingerprint")
    if checkpoint_fingerprint != fingerprint:
        if inherited_relocation is None or target_config is None:
            raise RuntimeError("Resume checkpoint configuration fingerprint differs")
        parent_config = inherited_relocation.get("parent_config")
        parent_fingerprint = inherited_relocation.get("parent_fingerprint")
        if (
            not isinstance(parent_config, Mapping)
            or checkpoint_fingerprint != parent_fingerprint
            or checkpoint.get("config") != parent_config
        ):
            raise RuntimeError("Inherited checkpoint configuration identity differs")
        source_latest = inherited_relocation.get("source_latest")
        if (
            not isinstance(source_latest, Path)
            or not source_latest.is_file()
            or not os.path.samefile(path, source_latest)
        ):
            raise RuntimeError("Inherited checkpoint is not the source system-disk file")
        inherited_run_relocation_config_changes(
            parent_config,
            target_config,
            completed_step=completed_step,
        )
    _restore_checkpoint_state(
        checkpoint,
        model_current=model_current,
        model_old=model_old,
        optimizer_current=optimizer_current,
        model_center=model_center,
        model_center_ema=model_center_ema,
        optimizer_center=optimizer_center,
    )
    return completed_step


def load_continuation_checkpoint(
    *,
    path: Path,
    expected_sha256: str,
    expected_step: int,
    target_config: NFTConfig,
    mmap: bool = False,
    allow_interrupted_relocation: bool = False,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Load and authenticate one external parent checkpoint for continuation."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Continuation checkpoint is not a regular file: {path}")
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("Continuation checkpoint SHA-256 is invalid")
    actual_sha256 = _sha256_file(path)
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
    parent_config = checkpoint.get("config")
    if not isinstance(parent_config, Mapping):
        raise RuntimeError("Continuation checkpoint has no config mapping")
    parent_fingerprint = config_mapping_fingerprint(parent_config)
    if checkpoint.get("config_fingerprint") != parent_fingerprint:
        raise RuntimeError("Continuation checkpoint stored config fingerprint differs")
    if allow_interrupted_relocation:
        changes = interrupted_relocation_config_changes(
            parent_config, target_config, completed_step=expected_step
        )
        continuation_mode = "interrupted_relocation"
    else:
        changes = continuation_config_changes(
            parent_config, target_config, completed_step=expected_step
        )
        continuation_mode = "terminal_extension"
    metadata = {
        "schema_version": 1,
        "kind": "model_guidance_diffusion_nft_continuation",
        "continuation_mode": continuation_mode,
        "parent_checkpoint": str(path),
        "parent_checkpoint_bytes": path.stat().st_size,
        "parent_checkpoint_sha256": actual_sha256,
        "parent_config_fingerprint": parent_fingerprint,
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
    target_config: NFTConfig,
    model_current,
    model_old,
    optimizer_current,
    model_center,
    model_center_ema,
    optimizer_center,
    allow_interrupted_relocation: bool = False,
) -> tuple[int, dict[str, Any]]:
    checkpoint, metadata = load_continuation_checkpoint(
        path=path,
        expected_sha256=expected_sha256,
        expected_step=expected_step,
        target_config=target_config,
        allow_interrupted_relocation=allow_interrupted_relocation,
    )
    _restore_checkpoint_state(
        checkpoint,
        model_current=model_current,
        model_old=model_old,
        optimizer_current=optimizer_current,
        model_center=model_center,
        model_center_ema=model_center_ema,
        optimizer_center=optimizer_center,
    )
    return expected_step, metadata


def append_metrics(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def truncate_metrics_after_step(path: Path, completed_outer_step: int) -> int:
    """Discard metric rows that were written after the rolling checkpoint."""

    if completed_outer_step < 0:
        raise ValueError("completed_outer_step must be non-negative")
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    removed = 0
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
            outer_step = int(record["outer_step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            if line_number == len(lines):
                removed += 1
                continue
            raise RuntimeError(
                f"Invalid metrics row {line_number} in {path}: {error}"
            ) from error
        if outer_step <= completed_outer_step:
            kept.append(json.dumps(record, sort_keys=True))
        else:
            removed += 1
    if removed:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            if kept:
                handle.write("\n".join(kept) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    return removed


def checkpoint_save_policy(
    config: NFTConfig,
    outer_step: int,
    stop_requested: bool,
    milestone_steps: Collection[int] | None = None,
) -> tuple[bool, bool]:
    """Return whether to write the rolling checkpoint and a milestone."""

    if outer_step < 1:
        raise ValueError("outer_step must be positive")
    rolling = config.checkpoint.save_every > 0
    if not rolling:
        return False, False
    scheduled = (
        outer_step % config.checkpoint.save_every == 0
        if milestone_steps is None
        else outer_step in milestone_steps
    )
    milestone = rolling and (
        scheduled or outer_step == config.max_outer_steps or stop_requested
    )
    return rolling, milestone


def main() -> None:
    args = parse_args()
    config = load_config(args.config, config_overrides(args))
    inherited_relocation = (
        load_inherited_run_relocation(
            args.inherit_run_relocation_marker, config
        )
        if args.inherit_run_relocation_marker
        else None
    )
    milestone_steps = parse_checkpoint_milestones(
        args.checkpoint_milestones, final_step=config.max_outer_steps
    )
    durable_values = (
        args.durable_checkpoint_dir,
        args.durable_run_name,
        args.durable_source_id,
    )
    if any(durable_values) and not all(durable_values):
        raise ValueError("durable checkpoint arguments must be provided together")
    if args.durable_checkpoint_dir and milestone_steps is None:
        raise ValueError("durable checkpoint publication requires --checkpoint-milestones")
    durable_checkpoint_dir = (
        Path(args.durable_checkpoint_dir).expanduser().resolve()
        if args.durable_checkpoint_dir
        else None
    )
    continuation_values = (
        args.continue_from_checkpoint,
        args.continue_from_checkpoint_sha256,
        args.continue_from_step,
    )
    if any(value is not None for value in continuation_values) and not all(
        value is not None for value in continuation_values
    ):
        raise ValueError("continuation checkpoint arguments must be provided together")
    continuation_path = (
        Path(args.continue_from_checkpoint).expanduser().resolve()
        if args.continue_from_checkpoint
        else None
    )
    if args.allow_interrupted_checkpoint_relocation and continuation_path is None:
        raise ValueError(
            "--allow-interrupted-checkpoint-relocation requires all continuation "
            "checkpoint arguments"
        )
    if args.continue_from_step is not None and args.continue_from_step < 0:
        raise ValueError("--continue-from-step must be non-negative")
    init_path = Path(config.init_checkpoint).expanduser().resolve()
    if not init_path.is_file():
        raise FileNotFoundError(f"Initial checkpoint is missing: {init_path}")
    if config.model_key not in SiT_models:
        raise KeyError(f"Unknown model_key: {config.model_key}")
    if args.validate_only:
        continuation_metadata = None
        if continuation_path is not None:
            checkpoint, continuation_metadata = load_continuation_checkpoint(
                path=continuation_path,
                expected_sha256=args.continue_from_checkpoint_sha256,
                expected_step=args.continue_from_step,
                target_config=config,
                mmap=True,
                allow_interrupted_relocation=(
                    args.allow_interrupted_checkpoint_relocation
                ),
            )
            del checkpoint
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        if continuation_metadata is not None:
            print(
                "VALID_CONTINUATION "
                + json.dumps(continuation_metadata, sort_keys=True)
            )
        print(f"VALID_CONFIG init_checkpoint={init_path} run_dir={config.run_dir.resolve()}")
        return

    device, rank, local_rank, world_size = setup_distributed()
    try:
        if config.rollout.groups_global % world_size != 0:
            raise ValueError(
                "rollout.groups_global must be divisible by the distributed world size"
            )
        seed = config.seed + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        run_dir, fingerprint = prepare_run_directory(
            config,
            rank,
            world_size,
            inherited_relocation=inherited_relocation,
        )
        rank0_print(
            rank,
            f"method={config.method} world_size={world_size} device={device} run_dir={run_dir}",
        )
        rank0_print(
            rank,
            "nft_scale="
            + json.dumps(
                {
                    "groups_global": config.rollout.groups_global,
                    "samples_per_group": config.rollout.samples_per_class,
                    "endpoints_global": (
                        config.rollout.groups_global * config.rollout.samples_per_class
                    ),
                    "timesteps_per_endpoint": nft_training_timestep_count(
                        config.rollout.sampling_steps,
                        config.training.timestep_fraction,
                    ),
                    "records_per_policy_update_global": (
                        config.rollout.groups_global
                        * config.rollout.samples_per_class
                        * config.training.inner_epochs
                        * nft_training_timestep_count(
                            config.rollout.sampling_steps,
                            config.training.timestep_fraction,
                        )
                    ),
                    "microbatch_per_gpu": config.training.batch_size,
                },
                sort_keys=True,
            ),
        )

        checkpoint_object = torch.load(init_path, map_location="cpu", weights_only=False)
        initial_state = _unwrap_checkpoint(checkpoint_object)
        base_model = build_model(config, initial_state)
        del checkpoint_object, initial_state

        model_current_base = base_model.to(device)
        model_old = copy.deepcopy(base_model).to(device)
        model_reference = copy.deepcopy(base_model).to(device)
        freeze_for_eval(model_old)
        freeze_for_eval(model_reference)
        model_current = wrap_ddp(model_current_base, device, local_rank, world_size)
        optimizer_current = torch.optim.AdamW(
            model_current_base.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        model_center_base = model_center = model_center_ema = optimizer_center = None
        if config.method == "shiftcenter":
            model_center_base = copy.deepcopy(base_model).to(device)
            model_center_ema = copy.deepcopy(base_model).to(device)
            freeze_for_eval(model_center_ema)
            model_center = wrap_ddp(model_center_base, device, local_rank, world_size)
            optimizer_center = torch.optim.AdamW(
                model_center_base.parameters(),
                lr=config.center.learning_rate,
                weight_decay=config.center.weight_decay,
            )
        del base_model

        offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{config.vae}", local_files_only=offline
        ).to(device)
        freeze_for_eval(vae)
        reward_fn = build_reward(config.reward, device)

        start_step = 0
        latest_checkpoint = run_dir / "checkpoints" / "latest.pt"
        if latest_checkpoint.exists():
            start_step = restore_checkpoint(
                path=latest_checkpoint,
                fingerprint=fingerprint,
                model_current=model_current,
                model_old=model_old,
                optimizer_current=optimizer_current,
                model_center=model_center,
                model_center_ema=model_center_ema,
                optimizer_center=optimizer_center,
                target_config=config,
                inherited_relocation=inherited_relocation,
            )
            rank0_print(rank, f"Resumed from {latest_checkpoint} at outer step {start_step}")
            if continuation_path is not None:
                continuation_record_path = run_dir / "continuation.json"
                if not continuation_record_path.is_file():
                    raise RuntimeError(
                        "Continued run has a local checkpoint but no continuation record"
                    )
                expected_identity = {
                    "parent_checkpoint": str(continuation_path),
                    "parent_checkpoint_sha256": args.continue_from_checkpoint_sha256,
                    "parent_completed_outer_step": args.continue_from_step,
                    "target_config_fingerprint": fingerprint,
                }
                if args.allow_interrupted_checkpoint_relocation:
                    expected_identity["continuation_mode"] = (
                        "interrupted_relocation"
                    )
                if inherited_relocation is not None:
                    if rank == 0:
                        inherited_record = json.loads(
                            continuation_record_path.read_text(encoding="utf-8")
                        )
                        inherited_expected = dict(expected_identity)
                        recorded_fingerprint = inherited_record.get(
                            "target_config_fingerprint"
                        )
                        if recorded_fingerprint == inherited_relocation[
                            "parent_fingerprint"
                        ]:
                            inherited_expected["target_config_fingerprint"] = (
                                recorded_fingerprint
                            )
                        inherited_differences = {
                            name: {
                                "expected": value,
                                "actual": inherited_record.get(name),
                            }
                            for name, value in inherited_expected.items()
                            if inherited_record.get(name) != value
                        }
                        if inherited_differences:
                            raise RuntimeError(
                                "Inherited continuation record differs: "
                                + json.dumps(inherited_differences, sort_keys=True)
                            )
                        if recorded_fingerprint != fingerprint:
                            inherited_record["target_config_fingerprint"] = fingerprint
                            atomic_write_json(
                                continuation_record_path, inherited_record
                            )
                    barrier(world_size)
                continuation_record = json.loads(
                    continuation_record_path.read_text(encoding="utf-8")
                )
                differences = {
                    name: {"expected": value, "actual": continuation_record.get(name)}
                    for name, value in expected_identity.items()
                    if continuation_record.get(name) != value
                }
                if differences:
                    raise RuntimeError(
                        "Continuation record differs on resume: "
                        + json.dumps(differences, sort_keys=True)
                    )
                if start_step < args.continue_from_step:
                    raise RuntimeError(
                        "Local continuation checkpoint predates the parent checkpoint"
                    )
        elif continuation_path is not None:
            start_step, continuation_record = restore_continuation_checkpoint(
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
                allow_interrupted_relocation=(
                    args.allow_interrupted_checkpoint_relocation
                ),
            )
            if rank == 0:
                atomic_write_json(run_dir / "continuation.json", continuation_record)
                save_checkpoint(
                    run_dir=run_dir,
                    payload=checkpoint_payload(
                        config=config,
                        fingerprint=fingerprint,
                        completed_outer_step=start_step,
                        model_current=model_current,
                        model_old=model_old,
                        optimizer_current=optimizer_current,
                        model_center=model_center,
                        model_center_ema=model_center_ema,
                        optimizer_center=optimizer_center,
                    ),
                    completed_outer_step=start_step,
                    keep=config.checkpoint.keep,
                    save_milestone=False,
                    preserve_milestone_steps=milestone_steps,
                )
            barrier(world_size)
            rank0_print(
                rank,
                f"Continued from {continuation_path} at outer step {start_step}",
            )

        stop_requested = False

        def request_stop(signum, frame):
            del frame
            nonlocal stop_requested
            stop_requested = True
            rank0_print(rank, f"Received signal {signum}; stopping after the current outer round")

        signal.signal(signal.SIGUSR1, request_stop)

        metrics_path = run_dir / "metrics.jsonl"
        if rank == 0:
            removed_metrics = truncate_metrics_after_step(metrics_path, start_step)
            if removed_metrics:
                print(
                    f"Removed {removed_metrics} metric rows newer than checkpoint step "
                    f"{start_step}",
                    flush=True,
                )
        barrier(world_size)
        if durable_checkpoint_dir is not None and start_step > 0:
            if rank == 0:
                assert milestone_steps is not None
                for durable_step in sorted(
                    step for step in milestone_steps if step <= start_step
                ):
                    local_milestone = (
                        run_dir / "checkpoints" / f"outer_{durable_step:06d}.pt"
                    )
                    durable_milestone = (
                        durable_checkpoint_dir / f"outer_{durable_step:06d}.pt"
                    )
                    if local_milestone.is_file():
                        publish_durable_checkpoint(
                            local_milestone,
                            durable_checkpoint_dir,
                            step=durable_step,
                            expected_fingerprint=fingerprint,
                            run_name=args.durable_run_name,
                            source_id=args.durable_source_id,
                        )
                        local_milestone.unlink()
                    elif not durable_milestone.is_file():
                        raise FileNotFoundError(
                            f"durable milestone is missing after resume: {durable_milestone}"
                        )
            barrier(world_size)
        if start_step == 0 and milestone_steps is not None and 0 in milestone_steps:
            if rank == 0:
                _, milestone_path = save_checkpoint(
                    run_dir=run_dir,
                    payload=checkpoint_payload(
                        config=config,
                        fingerprint=fingerprint,
                        completed_outer_step=0,
                        model_current=model_current,
                        model_old=model_old,
                        optimizer_current=optimizer_current,
                        model_center=model_center,
                        model_center_ema=model_center_ema,
                        optimizer_center=optimizer_center,
                    ),
                    completed_outer_step=0,
                    keep=config.checkpoint.keep,
                    save_milestone=True,
                    preserve_milestone_steps=milestone_steps,
                )
                assert milestone_path is not None
                if durable_checkpoint_dir is not None:
                    publish_durable_checkpoint(
                        milestone_path,
                        durable_checkpoint_dir,
                        step=0,
                        expected_fingerprint=fingerprint,
                        run_name=args.durable_run_name,
                        source_id=args.durable_source_id,
                    )
                    milestone_path.unlink()
                print("Saved initial outer checkpoint milestone: outer_000000.pt", flush=True)
            barrier(world_size)
        for outer_step in range(start_step + 1, config.max_outer_steps + 1):
            outer_start = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            dataset, rollout_metrics, preview = collect_rollout(
                model_old=model_old,
                vae=vae,
                reward_fn=reward_fn,
                config=config,
                outer_step=outer_step,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            if rank == 0 and outer_step == start_step + 1:
                torchvision.utils.save_image(
                    preview,
                    run_dir / "previews" / f"outer_{outer_step:06d}.png",
                    nrow=config.rollout.samples_per_class,
                )
            del preview

            center_metrics: dict[str, float] = {
                "center_loss": 0.0,
                "center_grad_norm": 0.0,
                "center_records": 0.0,
                "center_microbatches": 0.0,
            }
            center_snapshot = None
            if config.method == "shiftcenter":
                center_metrics = run_center_update(
                    model_center=model_center,
                    model_center_ema=model_center_ema,
                    optimizer=optimizer_center,
                    dataset=dataset,
                    config=config,
                    outer_step=outer_step,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                )
                center_snapshot = (
                    model_center_ema
                    if config.center.snapshot == "ema"
                    else unwrap_model(model_center)
                )
                center_snapshot.eval()

            policy_metrics = run_policy_update(
                model_current=model_current,
                model_old=model_old,
                model_reference=model_reference,
                model_center_snapshot=center_snapshot,
                optimizer=optimizer_current,
                dataset=dataset,
                config=config,
                outer_step=outer_step,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            decay = old_policy_decay(outer_step, config.training.old_policy_decay_type)
            update_old_policy(model_old, model_current, decay)

            local_metrics = {**rollout_metrics, **center_metrics, **policy_metrics}
            reduced = reduce_mean_metrics(local_metrics, device, world_size)
            elapsed = time.time() - outer_start
            max_elapsed = torch.tensor(elapsed, device=device, dtype=torch.float64)
            max_memory = torch.tensor(
                torch.cuda.max_memory_allocated(device), device=device, dtype=torch.float64
            )
            if world_size > 1:
                dist.all_reduce(max_elapsed, op=dist.ReduceOp.MAX)
                dist.all_reduce(max_memory, op=dist.ReduceOp.MAX)
            record: dict[str, Any] = {
                "outer_step": outer_step,
                "method": config.method,
                "model_key": config.model_key,
                "intrinsic_guidance_rollout": True,
                "external_cfg_scale": 1.0,
                "reward_name": config.reward.name,
                "rollout_sampler": config.rollout.sampler,
                "rollout_sampling_steps": config.rollout.sampling_steps,
                "rollout_samples_global": (
                    config.rollout.groups_global * config.rollout.samples_per_class
                ),
                "class_groups_global": config.rollout.groups_global,
                "samples_per_class": config.rollout.samples_per_class,
                "train_timesteps_per_endpoint": nft_training_timestep_count(
                    config.rollout.sampling_steps, config.training.timestep_fraction
                ),
                "center_timesteps_per_endpoint": (
                    nft_training_timestep_count(
                        config.rollout.sampling_steps, config.center.timestep_fraction
                    )
                    if config.method == "shiftcenter"
                    else 0
                ),
                "policy_updates": 1,
                "center_updates": int(config.method == "shiftcenter"),
                "center_snapshot": (
                    config.center.snapshot if config.method == "shiftcenter" else "old"
                ),
                "old_policy_decay": decay,
                "world_size": world_size,
                "policy_records_global": int(reduced["policy_records"] * world_size),
                "center_records_global": int(reduced["center_records"] * world_size),
                "policy_effective_batch_records_global": int(
                    reduced["policy_records"] * world_size
                ),
                "outer_seconds": float(max_elapsed.item()),
                "peak_memory_allocated_bytes": int(max_memory.item()),
                **reduced,
            }
            finite_values = [
                value
                for value in record.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if not all(math.isfinite(float(value)) for value in finite_values):
                raise FloatingPointError(f"Non-finite metric record: {record}")
            if rank == 0:
                append_metrics(metrics_path, record)
                print("OUTER_METRICS " + json.dumps(record, sort_keys=True), flush=True)

            should_save_rolling, should_save_milestone = checkpoint_save_policy(
                config, outer_step, stop_requested, milestone_steps
            )
            if should_save_rolling:
                barrier(world_size)
                if rank == 0:
                    payload = checkpoint_payload(
                        config=config,
                        fingerprint=fingerprint,
                        completed_outer_step=outer_step,
                        model_current=model_current,
                        model_old=model_old,
                        optimizer_current=optimizer_current,
                        model_center=model_center,
                        model_center_ema=model_center_ema,
                        optimizer_center=optimizer_center,
                    )
                    latest_path, milestone_path = save_checkpoint(
                        run_dir=run_dir,
                        payload=payload,
                        completed_outer_step=outer_step,
                        keep=config.checkpoint.keep,
                        save_milestone=should_save_milestone,
                        preserve_milestone_steps=milestone_steps,
                    )
                    message = f"Saved rolling checkpoint: {latest_path}"
                    if milestone_path is not None:
                        if (
                            durable_checkpoint_dir is not None
                            and milestone_steps is not None
                            and outer_step in milestone_steps
                        ):
                            durable = publish_durable_checkpoint(
                                milestone_path,
                                durable_checkpoint_dir,
                                step=outer_step,
                                expected_fingerprint=fingerprint,
                                run_name=args.durable_run_name,
                                source_id=args.durable_source_id,
                            )
                            milestone_path.unlink()
                            message += (
                                "; durable milestone: "
                                f"{durable_checkpoint_dir / durable['filename']}"
                            )
                        else:
                            message += f"; milestone: {milestone_path}"
                    print(message, flush=True)
                barrier(world_size)
            if stop_requested:
                break

        barrier(world_size)
        if rank == 0:
            atomic_write_json(
                run_dir / "run_summary.json",
                {
                    "method": config.method,
                    "start_outer_step": (
                        args.continue_from_step
                        if args.continue_from_step is not None
                        else 0
                    ),
                    "completed_outer_step": outer_step if "outer_step" in locals() else start_step,
                    "requested_outer_steps": config.max_outer_steps,
                    "stopped_on_signal": stop_requested,
                    "metrics_path": str(metrics_path),
                },
            )
        rank0_print(rank, f"DiffusionNFT completed: {run_dir}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
