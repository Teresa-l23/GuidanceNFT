"""Persistent v5-flow utilities for the SD3.5 DiffusionNFT trainer.

The functions in this module stay in SD3's native rectified-flow coordinate:
``t=0`` is clean data and ``t=1`` is Gaussian noise.  They intentionally do
not import the ImageNet/SiT implementation, whose time coordinate and model
calling convention are different.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file as load_safetensors

from baselines.diffusion_nft_sd35_v5.cfg_modes import (
    rollout_guidance_scale as cfg_rollout_guidance_scale,
    validate_cfg_mode,
)


V5_ADAPTER_NAME = "v5"
V5FLOW_STATE_FILENAME = "v5flow_state.pt"
V5FLOW_INITIALIZER_STATE_FILENAME = "v5flow_initializer.pt"
V5FLOW_INITIALIZER_MARKER_FILENAME = ".v5flow_initializer_complete.json"
V5FLOW_PRETRAIN_STATE_FILENAME = "v5flow_pretrain_state.pt"
V5FLOW_PRETRAIN_MARKER_FILENAME = ".v5flow_pretrain_complete.json"
V5FLOW_PRETRAIN_RANK_STATE_SCHEMA_VERSION = 1
POLICY_STATE_FILENAME = "policy_state.pt"
REQUEUE_RANK_STATE_SCHEMA_VERSION = 1
REQUEUE_PHASES = frozenset({"sampling", "stage1", "stage2", "outer_boundary"})
CHECKPOINT_COMPLETE_MARKER_FILENAME = ".checkpoint_complete.json"
CHECKPOINT_CFG_MARKER_KEYS = frozenset({"cfg_mode", "rollout_guidance_scale"})
# Keep the historical marker scale explicit even though cfg1_dual is again a
# supported train-time mode; immutable snapshots and current runs share it.
LEGACY_CHECKPOINT_CFG_GUIDANCE_SCALES = {"cfg1_dual": 1.0}
V5FLOW_PROMOTION_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "cfg_mode",
        "checkpoint_kind",
        "checkpoint_step",
        "resume_epoch",
        "rollout_guidance_scale",
        "v5_optimization_step",
    }
)
V5FLOW_STATE_KEYS = frozenset(
    {
        "schema_version",
        "adapter_default_raw",
        "adapter_old",
        "adapter_v5_raw",
        "optimizer_v5",
        "v5_scaler",
        "v5_ema",
        "theta_ema",
        "v5_optimization_step",
    }
)
ROLLING_LATEST_FILENAME = "latest.json"
_ROLLING_CHECKPOINT_NAME = re.compile(r"checkpoint-(0|[1-9][0-9]*)\Z")
EXPECTED_SOURCE_TARGET_MODULES = frozenset(
    {
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    }
)


def validate_checkpoint_cfg_marker(
    marker: dict[str, Any],
    *,
    cfg1_only: bool = False,
) -> tuple[str, float]:
    """Validate optional completion-marker CFG metadata.

    Markers written before the CFG experiments omit both fields and are
    interpreted as ``cfg1`` at scale 1.  New markers must carry both fields;
    accepting just one would make a resume silently ambiguous.
    """

    if not isinstance(marker, dict):
        raise ValueError("Checkpoint completion marker must be a JSON object")
    present_keys = set(marker) & CHECKPOINT_CFG_MARKER_KEYS
    if not present_keys:
        cfg_mode = "cfg1"
        guidance_scale = 1.0
    else:
        if present_keys != CHECKPOINT_CFG_MARKER_KEYS:
            raise ValueError("Checkpoint completion marker has a partial CFG contract")
        raw_mode = marker["cfg_mode"]
        if raw_mode in LEGACY_CHECKPOINT_CFG_GUIDANCE_SCALES:
            cfg_mode = raw_mode
            expected_scale = LEGACY_CHECKPOINT_CFG_GUIDANCE_SCALES[cfg_mode]
        else:
            try:
                cfg_mode = validate_cfg_mode(raw_mode)
            except ValueError as error:
                raise ValueError(
                    f"Checkpoint completion marker has invalid cfg_mode: {raw_mode!r}"
                ) from error
            expected_scale = cfg_rollout_guidance_scale(cfg_mode)
        raw_scale = marker["rollout_guidance_scale"]
        if (
            isinstance(raw_scale, bool)
            or not isinstance(raw_scale, (int, float))
            or float(raw_scale) != expected_scale
        ):
            raise ValueError(
                "Checkpoint completion marker rollout guidance disagrees with cfg_mode: "
                f"mode={cfg_mode!r}, scale={raw_scale!r}"
            )
        guidance_scale = float(raw_scale)
    if cfg1_only and (cfg_mode != "cfg1" or guidance_scale != 1.0):
        raise ValueError(
            "This checkpoint method only supports cfg1 with rollout_guidance_scale=1.0; "
            f"got mode={cfg_mode!r}, scale={guidance_scale!r}"
        )
    return cfg_mode, guidance_scale


@dataclass(frozen=True)
class SD3EndpointFlowMatchingBatch:
    """Fresh forward-process sample in SD3's noise-time coordinate."""

    state: torch.Tensor
    target_velocity: torch.Tensor
    noise: torch.Tensor


def v5flow_enabled(config) -> bool:
    """Return the master ablation switch, defaulting to the official path."""

    section = getattr(config, "v5flow", None)
    return bool(getattr(section, "enabled", False)) if section is not None else False


def v5flow_warmup_enabled(config) -> bool:
    """Return the opt-in pre-outer warm-up switch."""

    if not v5flow_enabled(config):
        return False
    warmup = getattr(config.v5flow, "warmup", None)
    return bool(getattr(warmup, "enabled", False)) if warmup is not None else False


def v5flow_initializer_path(config) -> str:
    """Return the optional one-time v5 initializer directory."""

    if not v5flow_enabled(config):
        return ""
    return str(getattr(config.v5flow, "initializer_path", "")).strip()


def v5flow_pretrained_checkpoint(config) -> str:
    """Return the optional completed formal checkpoint used as V5 initialization."""

    if not v5flow_enabled(config):
        return ""
    return str(getattr(config.v5flow, "pretrained_checkpoint", "")).strip()


def v5flow_pretrained_state_sha256(config) -> str:
    """Return the pinned state digest for :func:`v5flow_pretrained_checkpoint`."""

    if not v5flow_enabled(config):
        return ""
    return str(getattr(config.v5flow, "pretrained_state_sha256", "")).strip()


def v5flow_online_update_due(*, next_policy_step: int, update_interval: int) -> bool:
    """Return whether Stage 1 must refresh before ``next_policy_step``.

    ``update_interval=0`` freezes the loaded center.  A positive interval runs
    Stage 1 before policy steps divisible by that interval, so interval 1
    preserves the historical update-before-every-policy-step behavior.
    """

    if (
        isinstance(next_policy_step, bool)
        or not isinstance(next_policy_step, int)
        or next_policy_step <= 0
    ):
        raise ValueError("next_policy_step must be a positive integer")
    if (
        isinstance(update_interval, bool)
        or not isinstance(update_interval, int)
        or update_interval < 0
    ):
        raise ValueError("update_interval must be a non-negative integer")
    return update_interval > 0 and next_policy_step % update_interval == 0


def fixed_distributed_k_repeat_batches(
    *,
    dataset_size: int,
    batch_size: int,
    k: int,
    num_replicas: int,
    rank: int,
    num_batches: int,
    seed: int,
) -> list[list[int]]:
    """Build deterministic, globally disjoint prompt batches for warm-up.

    Each global batch contains ``num_replicas * batch_size / k`` unique prompt
    groups, and every selected prompt occurs exactly ``k`` times across ranks.
    A single seeded permutation supplies all batches, so no prompt can cross
    the train/holdout boundary when the caller splits the returned list.
    """

    values = {
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "k": k,
        "num_replicas": num_replicas,
        "num_batches": num_batches,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 <= rank < num_replicas
    ):
        raise ValueError("rank must be an integer in [0, num_replicas)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    global_batch_size = num_replicas * batch_size
    if global_batch_size % k:
        raise ValueError("k must divide num_replicas * batch_size")
    groups_per_batch = global_batch_size // k
    required_groups = groups_per_batch * num_batches
    if required_groups > dataset_size:
        raise ValueError(
            "Warm-up requires more disjoint prompt groups than the dataset contains: "
            f"required={required_groups}, available={dataset_size}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    unique_indices = torch.randperm(dataset_size, generator=generator)[
        :required_groups
    ].tolist()
    rank_batches = []
    rank_start = rank * batch_size
    rank_end = rank_start + batch_size
    for batch_index in range(num_batches):
        group_start = batch_index * groups_per_batch
        group_end = group_start + groups_per_batch
        repeated = [
            index
            for index in unique_indices[group_start:group_end]
            for _ in range(k)
        ]
        shuffle = torch.randperm(global_batch_size, generator=generator).tolist()
        shuffled = [repeated[index] for index in shuffle]
        rank_batches.append(shuffled[rank_start:rank_end])
    return rank_batches


def resolve_marked_resume_position(
    *,
    checkpoint_step: int,
    resume_epoch: int,
    checkpoint_kind: str,
    theta_updates_per_epoch: int,
) -> tuple[int, bool]:
    """Resolve the next outer epoch and whether its policy update is pending.

    ``outer_boundary`` checkpoints are written before an outer epoch starts, so
    they resume that epoch directly. ``final`` checkpoints are written after
    the last theta update but before the end-of-epoch old-policy update. A
    boundary-aligned final checkpoint therefore resumes at the next epoch only
    after the caller applies that pending update.
    """

    for name, value in {
        "checkpoint_step": checkpoint_step,
        "resume_epoch": resume_epoch,
        "theta_updates_per_epoch": theta_updates_per_epoch,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if theta_updates_per_epoch == 0:
        raise ValueError("theta_updates_per_epoch must be positive")
    if checkpoint_kind == "outer_boundary":
        expected_step = resume_epoch * theta_updates_per_epoch
        if checkpoint_step != expected_step:
            raise ValueError(
                "outer-boundary checkpoint step does not match resume_epoch: "
                f"step={checkpoint_step}, expected={expected_step}"
            )
        return resume_epoch, False
    if checkpoint_kind == "final":
        expected_step = (resume_epoch + 1) * theta_updates_per_epoch
        if checkpoint_step != expected_step:
            raise ValueError(
                "final checkpoint is not aligned to a completed outer epoch: "
                f"step={checkpoint_step}, expected={expected_step}"
            )
        return resume_epoch + 1, True
    raise ValueError(f"Unsupported checkpoint_kind: {checkpoint_kind!r}")


def validate_v5flow_config(config) -> bool:
    """Validate the SD3-only v5-flow contract and return whether it is enabled."""

    if not v5flow_enabled(config):
        return False
    v5 = config.v5flow
    if not bool(config.use_lora):
        raise ValueError("v5flow requires config.use_lora=True")
    cfg_mode = validate_cfg_mode(str(getattr(config, "cfg_mode", "cfg1")))
    cfg_unconditional_source = str(
        getattr(config, "cfg_unconditional_source", "same_model")
    )
    if cfg_unconditional_source not in {"same_model", "pretrained_base"}:
        raise ValueError(
            "v5flow cfg_unconditional_source must be same_model or pretrained_base"
        )
    if cfg_mode not in {
        "cfg1",
        "cfg4_rollout",
        "cfg4_dual_shifted_center",
        "cfg4_full",
        "cfg4_alternating",
    }:
        raise ValueError(
            "v5flow supports only cfg1, cfg4_rollout, "
            "cfg4_dual_shifted_center, or cfg4_full/cfg4_alternating; its learned "
            "v5 center must not be combined with an analytic old-CFG center or "
            "the cfg1_dual/cfg1_dropout objectives"
        )
    expected_guidance_scale = cfg_rollout_guidance_scale(cfg_mode)
    if float(config.sample.guidance_scale) != expected_guidance_scale:
        raise ValueError(
            f"v5flow cfg_mode={cfg_mode!r} requires sample.guidance_scale="
            f"{expected_guidance_scale}"
        )
    if (
        cfg_unconditional_source == "pretrained_base"
        and cfg_mode != "cfg4_rollout"
    ):
        raise ValueError(
            "v5flow pretrained_base unconditional routing requires cfg4_rollout"
        )
    if str(v5.center) != "v5":
        raise ValueError("v5flow.center must be 'v5'; disable v5flow for the old-center control")
    if str(v5.init) != "old_once":
        raise ValueError("v5flow.init must be 'old_once' so the adapter and optimizer persist")
    if str(v5.timestep_sampling) not in {"rollout_schedule", "continuous_uniform"}:
        raise ValueError(
            "v5flow.timestep_sampling must be 'rollout_schedule' or 'continuous_uniform'"
        )
    if str(v5.snapshot) not in {"raw", "ema"}:
        raise ValueError("v5flow.snapshot must be 'raw' or 'ema'")
    if str(v5.snapshot) == "ema" and not bool(v5.use_ema):
        raise ValueError("v5flow.snapshot='ema' requires v5flow.use_ema=True")
    if str(v5.theta_lifecycle) not in {"persistent", "reset"}:
        raise ValueError("v5flow.theta_lifecycle must be 'persistent' or 'reset'")
    if str(v5.theta_lifecycle) == "reset" and not bool(config.train.ema):
        raise ValueError("v5flow.theta_lifecycle='reset' requires train.ema=True")
    if str(v5.loss_space) not in {"velocity", "reconstruction"}:
        raise ValueError("v5flow.loss_space must be 'velocity' or 'reconstruction'")
    if not isinstance(v5.normalize_branch_mae, bool):
        raise ValueError("v5flow.normalize_branch_mae must be boolean")
    if int(v5.num_inner_epochs) <= 0:
        raise ValueError("v5flow.num_inner_epochs must be positive")
    update_interval = getattr(v5, "update_interval", 1)
    if (
        isinstance(update_interval, bool)
        or not isinstance(update_interval, int)
        or update_interval < 0
    ):
        raise ValueError("v5flow.update_interval must be a non-negative integer")
    if not math.isfinite(float(v5.learning_rate)) or float(v5.learning_rate) <= 0.0:
        raise ValueError("v5flow.learning_rate must be positive and finite")
    if not 0.0 <= float(v5.ema_decay) < 1.0:
        raise ValueError("v5flow.ema_decay must lie in [0, 1)")
    if not 0.0 <= float(v5.theta_ema_decay) < 1.0:
        raise ValueError("v5flow.theta_ema_decay must lie in [0, 1)")
    if not math.isfinite(float(v5.max_grad_norm)) or float(v5.max_grad_norm) <= 0.0:
        raise ValueError("v5flow.max_grad_norm must be positive and finite")
    if not 0.0 < float(config.train.timestep_fraction) <= 1.0:
        raise ValueError("train.timestep_fraction must lie in (0, 1]")
    initializer_path = v5flow_initializer_path(config)
    if initializer_path and not os.path.isabs(initializer_path):
        raise ValueError("v5flow.initializer_path must be absolute")
    pretrained_checkpoint = v5flow_pretrained_checkpoint(config)
    pretrained_digest = v5flow_pretrained_state_sha256(config)
    if bool(pretrained_checkpoint) != bool(pretrained_digest):
        raise ValueError(
            "v5flow.pretrained_checkpoint and pretrained_state_sha256 must be "
            "set together"
        )
    if pretrained_checkpoint and not os.path.isabs(pretrained_checkpoint):
        raise ValueError("v5flow.pretrained_checkpoint must be absolute")
    if pretrained_digest and re.fullmatch(r"[0-9a-f]{64}", pretrained_digest) is None:
        raise ValueError("v5flow.pretrained_state_sha256 must be lowercase hex")
    if cfg_mode == "cfg4_alternating":
        updates_per_epoch = (int(config.sample.num_batches_per_epoch)
                             * int(config.sample.train_batch_size)
                             // int(config.train.batch_size)
                             // int(config.train.gradient_accumulation_steps))
        if (not pretrained_checkpoint or str(v5.theta_lifecycle) != "persistent"
                or update_interval <= 0 or int(config.train.num_inner_epochs) != 1
                or updates_per_epoch != 1 or cfg_unconditional_source != "same_model"):
            raise ValueError("cfg4_alternating requires persistent pretrained V5 and one policy update per epoch")
    if cfg_mode == "cfg4_dual_shifted_center" and not pretrained_checkpoint:
        raise ValueError(
            "cfg4_dual_shifted_center requires a pinned pretrained V5 checkpoint"
        )
    if pretrained_checkpoint and cfg_mode not in {
        "cfg4_rollout",
        "cfg4_dual_shifted_center",
        "cfg4_alternating",
    }:
        raise ValueError(
            "v5flow.pretrained_checkpoint is reserved for "
            "cfg4_rollout, cfg4_dual_shifted_center, or cfg4_alternating"
        )
    warmup = getattr(v5, "warmup", None)
    if cfg_mode in {"cfg4_rollout", "cfg4_full"} and initializer_path:
        raise ValueError(
            f"{cfg_mode} v5flow cannot load a reusable CFG1 initializer; use "
            "a same-run CFG4 pre-outer warm-up so v5 is fitted from CFG4 endpoints"
        )
    if pretrained_checkpoint and initializer_path:
        raise ValueError(
            "v5flow pretrained checkpoint and reusable initializer are mutually exclusive"
        )
    if (
        pretrained_checkpoint
        and warmup is not None
        and bool(getattr(warmup, "enabled", False))
    ):
        raise ValueError(
            "v5flow pretrained checkpoint and same-run warm-up are mutually exclusive"
        )
    if warmup is not None and bool(getattr(warmup, "enabled", False)):
        for field in (
            "seed",
            "num_rollout_batches",
            "num_holdout_batches",
            "num_inner_epochs",
            "num_train_timesteps",
            "gradient_accumulation_steps",
            "num_refreshes",
            "keep_checkpoints",
        ):
            value = getattr(warmup, field)
            minimum = 0 if field == "seed" else 2 if field == "keep_checkpoints" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"v5flow.warmup.{field} must be an integer >= {minimum}"
                )
        if str(warmup.timestep_sampling) not in {
            "rollout_schedule",
            "continuous_uniform",
        }:
            raise ValueError(
                "v5flow.warmup.timestep_sampling must be rollout_schedule or "
                "continuous_uniform"
            )
        if str(warmup.prompt_pool) not in {"train", "test"}:
            raise ValueError("v5flow.warmup.prompt_pool must be train or test")
        if str(getattr(warmup, "parameter_dtype", "model")) not in {
            "model",
            "float32",
        }:
            raise ValueError(
                "v5flow.warmup.parameter_dtype must be model or float32"
            )
        if int(warmup.num_rollout_batches) % int(
            warmup.gradient_accumulation_steps
        ):
            raise ValueError(
                "v5flow.warmup.num_rollout_batches must divide by "
                "gradient_accumulation_steps"
            )
        if (
            str(warmup.timestep_sampling) == "rollout_schedule"
            and int(warmup.num_train_timesteps) > int(config.sample.num_steps)
        ):
            raise ValueError(
                "v5flow.warmup.num_train_timesteps exceeds the rollout schedule"
            )
        if initializer_path:
            raise ValueError(
                "v5flow initializer and warm-up cannot be enabled together"
            )
        export_only = bool(getattr(warmup, "export_only", False))
        output_path = str(getattr(warmup, "output_path", "")).strip()
        checkpoint_root = str(getattr(warmup, "checkpoint_root", "")).strip()
        if export_only:
            if not output_path or not os.path.isabs(output_path):
                raise ValueError(
                    "export-only v5flow warm-up requires an absolute output_path"
                )
            if int(config.max_train_steps) != 0:
                raise ValueError(
                    "export-only v5flow warm-up requires max_train_steps=0"
                )
            if int(warmup.num_refreshes) > 1 and (
                not checkpoint_root or not os.path.isabs(checkpoint_root)
            ):
                raise ValueError(
                    "multi-refresh v5flow warm-up requires an absolute checkpoint_root"
                )
        elif output_path:
            raise ValueError(
                "v5flow warm-up output_path requires export_only=true"
            )
        elif checkpoint_root:
            raise ValueError(
                "v5flow warm-up checkpoint_root requires export_only=true"
            )
        if not export_only and int(warmup.num_refreshes) != 1:
            raise ValueError(
                "multi-refresh v5flow warm-up requires export_only=true"
            )
    elif int(config.max_train_steps) == 0:
        raise ValueError("max_train_steps=0 requires export-only v5flow warm-up")
    return True


def sample_sd3_training_timesteps(
    *,
    stored_timesteps: torch.Tensor,
    num_samples: int,
    mode: str,
) -> torch.Tensor:
    """Return one training-time matrix per endpoint.

    ``rollout_schedule`` independently permutes each endpoint's stored solver
    labels and selects ``num_samples`` labels. ``continuous_uniform`` ignores
    the discrete labels' values (but uses their batch/device) and draws fresh
    transformer timesteps from ``1000 * U[0, 1)``. The returned shape is
    ``[batch, num_samples]``.
    """

    if stored_timesteps.ndim != 2 or stored_timesteps.shape[0] == 0:
        raise ValueError("stored_timesteps must be a nonempty rank-2 tensor")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    batch_size, schedule_size = stored_timesteps.shape
    if mode == "rollout_schedule":
        if num_samples > schedule_size:
            raise ValueError("num_samples exceeds the stored rollout schedule")
        permutations = torch.stack(
            [
                torch.randperm(schedule_size, device=stored_timesteps.device)
                for _ in range(batch_size)
            ]
        )
        selected = permutations[:, :num_samples]
        return stored_timesteps[
            torch.arange(batch_size, device=stored_timesteps.device)[:, None],
            selected,
        ]
    if mode == "continuous_uniform":
        return torch.rand(
            (batch_size, num_samples),
            device=stored_timesteps.device,
            dtype=torch.float32,
        ) * 1000.0
    raise ValueError(f"Unsupported timestep sampling mode: {mode!r}")


def make_sd3_endpoint_flow_matching_batch(
    *,
    clean_latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> SD3EndpointFlowMatchingBatch:
    """Construct ``x_t=(1-t)x_0+t*eps`` and target ``eps-x_0``.

    ``timesteps`` use the SD3 transformer convention ``[0, 1000]``.  The
    Gaussian is fresh for every call; rollout trajectory states are never used.
    """

    if clean_latents.ndim < 2 or clean_latents.shape[0] == 0:
        raise ValueError("clean_latents must be a nonempty batched tensor")
    if not clean_latents.is_floating_point():
        raise ValueError("clean_latents must be floating point")
    if timesteps.ndim != 1 or timesteps.shape[0] != clean_latents.shape[0]:
        raise ValueError(
            f"timesteps must have shape [{clean_latents.shape[0]}], got {tuple(timesteps.shape)}"
        )
    if not bool(torch.isfinite(timesteps).all().item()):
        raise ValueError("timesteps must be finite")
    if bool(((timesteps < 0) | (timesteps > 1000)).any().item()):
        raise ValueError("SD3 timesteps must lie in [0, 1000]")
    if noise is None:
        noise = torch.randn_like(clean_latents.float())
    elif noise.shape != clean_latents.shape:
        raise ValueError("noise and clean_latents must have identical shapes")
    noise = noise.to(device=clean_latents.device, dtype=torch.float32)
    clean_float = clean_latents.float()
    t = timesteps.to(device=clean_latents.device, dtype=torch.float32) / 1000.0
    t_view = t.view(-1, *([1] * (clean_latents.ndim - 1)))
    return SD3EndpointFlowMatchingBatch(
        state=(1.0 - t_view) * clean_float + t_view * noise,
        target_velocity=noise - clean_float,
        noise=noise,
    )


def compose_shifted_center_predictions(
    *,
    current_prediction: torch.Tensor,
    old_prediction: torch.Tensor,
    center_prediction: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``v5 +/- beta * (v_theta - v_old)`` with frozen anchors."""

    if not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be positive and finite")
    if (
        current_prediction.shape != old_prediction.shape
        or current_prediction.shape != center_prediction.shape
    ):
        raise ValueError("current, old, and center predictions must have identical shapes")
    old = old_prediction.detach()
    center = center_prediction.detach()
    displacement = current_prediction - old
    return center + float(beta) * displacement, center - float(beta) * displacement


def prediction_loss_per_example(
    *,
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    t: torch.Tensor,
    loss_space: str,
    normalize_residual_mae: bool,
    mae_epsilon: float = 1e-5,
) -> torch.Tensor:
    """Return selectable SD3 velocity/reconstruction branch losses.

    SD3 uses ``x_t=(1-t)x_0+t*eps`` and ``W=eps-x_0``.  Direct velocity loss
    therefore uses ``prediction-W``.  Reconstruction loss uses the equivalent
    residual ``t*(prediction-W)`` because ``x0_hat=x_t-t*prediction``.  Optional
    MAE normalization divides each example by its own detached mean absolute
    residual, matching the released DiffusionNFT adaptive scale.
    """

    if prediction.shape != target_velocity.shape:
        raise ValueError("prediction and target_velocity must have identical shapes")
    if prediction.ndim < 2 or prediction.shape[0] == 0:
        raise ValueError("prediction must be a nonempty batched tensor")
    if t.ndim != 1 or t.shape[0] != prediction.shape[0]:
        raise ValueError(f"t must have shape [{prediction.shape[0]}]")
    if loss_space not in {"velocity", "reconstruction"}:
        raise ValueError("loss_space must be 'velocity' or 'reconstruction'")
    if not isinstance(normalize_residual_mae, bool):
        raise ValueError("normalize_residual_mae must be boolean")
    if not math.isfinite(float(mae_epsilon)) or float(mae_epsilon) <= 0.0:
        raise ValueError("mae_epsilon must be positive and finite")

    residual = prediction - target_velocity
    if loss_space == "reconstruction":
        t_view = t.to(device=residual.device, dtype=residual.dtype).view(
            -1, *([1] * (residual.ndim - 1))
        )
        residual = t_view * residual
    reduction_dims = tuple(range(1, residual.ndim))
    if normalize_residual_mae:
        with torch.no_grad():
            denominator = (
                residual.detach()
                .double()
                .abs()
                .mean(dim=reduction_dims, keepdim=True)
                .clamp(min=float(mae_epsilon))
            )
        return (residual.square() / denominator).mean(dim=reduction_dims)
    return residual.square().mean(dim=reduction_dims)


def _adapter_marker(adapter_name: str) -> str:
    if not adapter_name or "." in adapter_name:
        raise ValueError(f"invalid adapter name: {adapter_name!r}")
    return f".{adapter_name}."


def named_adapter_parameters(model, adapter_name: str) -> dict[str, torch.nn.Parameter]:
    """Return one PEFT adapter keyed independently of its concrete name."""

    marker = _adapter_marker(adapter_name)
    result = {
        name.replace(marker, ".<adapter>."): parameter
        for name, parameter in model.named_parameters()
        if marker in name
    }
    if not result:
        raise ValueError(f"No parameters found for PEFT adapter {adapter_name!r}")
    return dict(sorted(result.items()))


def adapter_parameters(model, adapter_name: str) -> list[torch.nn.Parameter]:
    """Return adapter parameters in a deterministic, cross-adapter order."""

    return list(named_adapter_parameters(model, adapter_name).values())


@torch.no_grad()
def copy_parameter_values_(
    source: Iterable[torch.Tensor], target: Iterable[torch.nn.Parameter]
) -> None:
    """Copy an ordered parameter sequence after exact count/shape validation."""

    source_values = list(source)
    target_parameters = list(target)
    if len(source_values) != len(target_parameters):
        raise ValueError(
            "Parameter sequence lengths differ: "
            f"source={len(source_values)}, target={len(target_parameters)}"
        )
    for index, (source_value, target_parameter) in enumerate(
        zip(source_values, target_parameters, strict=True)
    ):
        if source_value.shape != target_parameter.shape:
            raise ValueError(f"Parameter shape mismatch at index {index}")
        target_parameter.copy_(
            source_value.detach().to(
                device=target_parameter.device,
                dtype=target_parameter.dtype,
            )
        )


@torch.no_grad()
def copy_adapter_parameters_(model, *, source: str, target: str) -> None:
    """Copy one LoRA adapter into another after exact structural validation."""

    source_parameters = named_adapter_parameters(model, source)
    target_parameters = named_adapter_parameters(model, target)
    if source_parameters.keys() != target_parameters.keys():
        raise ValueError(f"Adapter structures differ: source={source!r}, target={target!r}")
    for key in source_parameters:
        target_parameters[key].copy_(source_parameters[key].detach())


@torch.no_grad()
def copy_adapter_parameters_preserve_dtype_(model, *, source: str, target: str) -> None:
    """Copy an adapter while promoting target storage to the source dtype."""

    source_parameters = named_adapter_parameters(model, source)
    target_parameters = named_adapter_parameters(model, target)
    if source_parameters.keys() != target_parameters.keys():
        raise ValueError(f"Adapter structures differ: source={source!r}, target={target!r}")
    for key, source_parameter in source_parameters.items():
        target_parameter = target_parameters[key]
        if not source_parameter.is_floating_point() or not target_parameter.is_floating_point():
            raise ValueError(f"LoRA adapter parameter must be floating point: {key}")
        if target_parameter.dtype != source_parameter.dtype:
            target_parameter.data = target_parameter.data.to(dtype=source_parameter.dtype)
        target_parameter.copy_(source_parameter.detach())


@torch.no_grad()
def set_adapter_storage_dtype_(
    model,
    *,
    adapter_names: Iterable[str],
    dtype: torch.dtype,
) -> None:
    """Set adapter parameter storage dtype before DDP/optimizer construction.

    This only casts the currently initialized values. A self-contained formal
    checkpoint resume subsequently replaces those values with its raw adapter
    tensors, while retaining the matching parameter storage required by the
    restored optimizer and EMA states.
    """

    names = tuple(adapter_names)
    if not names:
        raise ValueError("adapter_names must be nonempty")
    for adapter_name in names:
        for key, parameter in named_adapter_parameters(model, adapter_name).items():
            if not parameter.is_floating_point():
                raise ValueError(f"LoRA adapter parameter must be floating point: {key}")
            if parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype=dtype)


def sha256_file(path: str | Path) -> str:
    """Return the stable SHA256 digest of one regular file."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def restore_peft_adapter_storage_from_safetensors_(
    model,
    *,
    adapter_name: str,
    adapter_dir: str | Path,
    expected_dtype: torch.dtype,
) -> dict[str, Any]:
    """Restore exact PEFT safetensors values without base-dtype downcasting.

    PEFT serialization removes the adapter-name segment from LoRA keys. This
    function maps those keys back to the canonical named-parameter structure,
    requires an exact key set and dtype, and changes ``Parameter.data`` before
    DDP/optimizer construction.
    """

    adapter_dir = Path(adapter_dir).expanduser().resolve()
    state_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not state_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Incomplete PEFT adapter directory: {adapter_dir}")
    with config_path.open("r", encoding="utf-8") as config_file:
        adapter_config = json.load(config_file)
    expected_scalars = {
        "peft_type": "LORA",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "bias": "none",
    }
    mismatches = {
        key: {"actual": adapter_config.get(key), "expected": expected}
        for key, expected in expected_scalars.items()
        if adapter_config.get(key) != expected
    }
    target_modules = adapter_config.get("target_modules")
    if not isinstance(target_modules, list) or set(target_modules) != EXPECTED_SOURCE_TARGET_MODULES:
        mismatches["target_modules"] = {
            "actual": target_modules,
            "expected": sorted(EXPECTED_SOURCE_TARGET_MODULES),
        }
    if mismatches:
        raise ValueError(
            "Source adapter config changes the pinned LoRA forward semantics: "
            f"{mismatches}"
        )
    saved_state = load_safetensors(state_path, device="cpu")
    parameters = named_adapter_parameters(model, adapter_name)
    saved_by_canonical = {}
    for saved_key, value in saved_state.items():
        canonical_key = None
        for factor in ("lora_A", "lora_B"):
            suffix = f"{factor}.weight"
            if saved_key.endswith(suffix):
                canonical_key = (
                    f"{saved_key[:-len(suffix)]}{factor}.<adapter>.weight"
                )
                break
        if canonical_key is None or canonical_key in saved_by_canonical:
            raise ValueError(f"Unsupported or duplicate PEFT tensor key: {saved_key}")
        saved_by_canonical[canonical_key] = value
    if saved_by_canonical.keys() != parameters.keys():
        raise ValueError(
            "PEFT safetensors structure does not exactly match the loaded adapter: "
            f"missing={sorted(parameters.keys() - saved_by_canonical.keys())}, "
            f"extra={sorted(saved_by_canonical.keys() - parameters.keys())}"
        )
    for key, parameter in parameters.items():
        value = saved_by_canonical[key]
        if value.shape != parameter.shape:
            raise ValueError(f"PEFT safetensors shape mismatch for {key}")
        if value.dtype != expected_dtype:
            raise ValueError(
                f"PEFT safetensors dtype mismatch for {key}: "
                f"{value.dtype} != {expected_dtype}"
            )
        parameter.data = value.to(device=parameter.device).clone()
    dtype_names = sorted({str(parameter.dtype) for parameter in parameters.values()})
    return {
        "adapter_dir": os.fspath(adapter_dir),
        "adapter_model_path": os.fspath(state_path),
        "adapter_model_sha256": sha256_file(state_path),
        "adapter_config_path": os.fspath(config_path),
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_config_contract": {
            **expected_scalars,
            "target_modules": sorted(EXPECTED_SOURCE_TARGET_MODULES),
        },
        "adapter_storage_dtypes": dtype_names,
        "adapter_tensor_count": len(parameters),
    }


def v5flow_initializer_storage_dtype(initializer_dir: str | Path) -> str:
    """Read the declared adapter storage dtype without loading tensor state."""

    if not initializer_dir:
        return "model"
    marker_path = (
        Path(initializer_dir).expanduser().resolve()
        / V5FLOW_INITIALIZER_MARKER_FILENAME
    )
    with marker_path.open("r", encoding="utf-8") as marker_file:
        marker = json.load(marker_file)
    metadata = marker.get("metadata") if isinstance(marker, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid v5flow initializer metadata: {marker_path}")
    dtype = str(metadata.get("adapter_storage_dtype", "model"))
    if dtype not in {"model", "float32"}:
        raise ValueError(f"Unsupported v5flow initializer adapter_storage_dtype: {dtype}")
    return dtype


def v5flow_checkpoint_storage_dtype(checkpoint_dir: str | Path) -> str:
    """Inspect a formal v5flow resume before DDP/optimizer construction."""

    state_path = (
        Path(checkpoint_dir).expanduser().resolve() / V5FLOW_STATE_FILENAME
    )
    if not state_path.is_file():
        raise FileNotFoundError(f"v5flow resume state is missing: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    adapter_states = [
        state.get("adapter_default_raw"),
        state.get("adapter_old"),
        state.get("adapter_v5_raw"),
    ]
    if any(not isinstance(adapter_state, dict) or not adapter_state for adapter_state in adapter_states):
        raise ValueError(f"v5flow resume has invalid adapter state: {state_path}")
    dtypes = {
        value.dtype
        for adapter_state in adapter_states
        for value in adapter_state.values()
        if isinstance(value, torch.Tensor)
    }
    tensor_count = sum(len(adapter_state) for adapter_state in adapter_states)
    if tensor_count == 0 or any(
        not isinstance(value, torch.Tensor)
        for adapter_state in adapter_states
        for value in adapter_state.values()
    ):
        raise ValueError(f"v5flow resume has invalid adapter tensors: {state_path}")
    if dtypes == {torch.float32}:
        return "float32"
    if len(dtypes) != 1:
        raise ValueError(f"v5flow resume mixes adapter storage dtypes: {sorted(map(str, dtypes))}")
    return "model"


@torch.no_grad()
def load_v5flow_pretrained_formal_checkpoint(
    *,
    checkpoint_dir: str | Path,
    model,
    optimizer_v5: torch.optim.Optimizer,
    v5_scaler,
    v5_ema,
    theta_parameters: Iterable[torch.nn.Parameter],
    theta_ema,
    map_location: torch.device,
    expected_state_sha256: str,
) -> tuple[int, dict[str, Any]]:
    """Seed a fresh formal run from a completed step-zero V5 warm-up.

    Unlike a formal resume, this restores only the three adapter snapshots and
    the persistent V5 optimizer/scaler/EMA state.  It deliberately leaves the
    theta optimizer, theta scaler, global step, formal resume position, and RNG
    streams fresh.  The source must be a completed ``cfg4_rollout``
    outer-boundary checkpoint at step zero, and its state file is content-pinned
    by ``expected_state_sha256``.
    """

    checkpoint_path = Path(checkpoint_dir).expanduser()
    if not checkpoint_path.is_absolute():
        raise ValueError("Pretrained V5 checkpoint must be absolute")
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Pretrained V5 checkpoint directory is missing: {checkpoint_path}"
        )
    if re.fullmatch(r"[0-9a-f]{64}", expected_state_sha256) is None:
        raise ValueError("Expected pretrained V5 state SHA256 must be lowercase hex")

    marker_path = checkpoint_path / CHECKPOINT_COMPLETE_MARKER_FILENAME
    state_path = checkpoint_path / V5FLOW_STATE_FILENAME
    if not marker_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"Incomplete pretrained V5 checkpoint: {checkpoint_path}"
        )
    with marker_path.open("r", encoding="utf-8") as marker_file:
        marker = json.load(marker_file)
    if not isinstance(marker, dict) or set(marker) != V5FLOW_PROMOTION_MARKER_KEYS:
        actual_keys = set(marker) if isinstance(marker, dict) else set()
        raise ValueError(
            "Unexpected pretrained V5 marker keys: "
            f"missing={sorted(V5FLOW_PROMOTION_MARKER_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - V5FLOW_PROMOTION_MARKER_KEYS)}"
        )
    expected_marker_values = {
        "schema_version": 1,
        "cfg_mode": "cfg4_rollout",
        "checkpoint_kind": "outer_boundary",
        "checkpoint_step": 0,
        "resume_epoch": 0,
        "rollout_guidance_scale": 4.0,
    }
    for key, expected_value in expected_marker_values.items():
        if marker[key] != expected_value:
            raise ValueError(
                f"Pretrained V5 marker {key} mismatch: "
                f"{marker[key]!r} != {expected_value!r}"
            )
    marker_v5_step = marker["v5_optimization_step"]
    if (
        not isinstance(marker_v5_step, int)
        or isinstance(marker_v5_step, bool)
        or marker_v5_step <= 0
    ):
        raise ValueError("Pretrained V5 checkpoint requires positive optimizer steps")

    actual_state_sha256 = sha256_file(state_path)
    if actual_state_sha256 != expected_state_sha256:
        raise ValueError(
            "Pretrained V5 state SHA256 mismatch: "
            f"{actual_state_sha256} != {expected_state_sha256}"
        )
    state = torch.load(state_path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict) or set(state) != V5FLOW_STATE_KEYS:
        actual_keys = set(state) if isinstance(state, dict) else set()
        raise ValueError(
            "Unexpected pretrained V5 state keys: "
            f"missing={sorted(V5FLOW_STATE_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - V5FLOW_STATE_KEYS)}"
        )
    if state["schema_version"] != 1:
        raise ValueError(
            f"Unsupported pretrained V5 state schema: {state['schema_version']!r}"
        )
    if state["v5_optimization_step"] != marker_v5_step:
        raise ValueError("Pretrained V5 marker and state optimizer steps disagree")

    adapter_states = (
        state["adapter_default_raw"],
        state["adapter_old"],
        state["adapter_v5_raw"],
    )
    if any(not isinstance(value, dict) or not value for value in adapter_states):
        raise ValueError("Pretrained V5 checkpoint has an invalid adapter state")
    canonical_keys = tuple(adapter_states[0])
    if canonical_keys != tuple(sorted(canonical_keys)) or any(
        tuple(value) != canonical_keys for value in adapter_states[1:]
    ):
        raise ValueError(
            "Pretrained V5 adapter states must share canonical sorted key order"
        )
    for key in canonical_keys:
        default_value = state["adapter_default_raw"][key]
        old_value = state["adapter_old"][key]
        v5_value = state["adapter_v5_raw"][key]
        if any(
            not isinstance(value, torch.Tensor)
            or value.shape != default_value.shape
            or value.dtype != default_value.dtype
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
            for value in (default_value, old_value, v5_value)
        ):
            raise ValueError(f"Invalid pretrained V5 adapter tensor: {key}")
        if not torch.equal(default_value, old_value):
            raise ValueError(
                f"Pretrained V5 checkpoint is not an untouched step-zero policy: {key}"
            )

    current_source = _adapter_cpu_state(model, "default")
    saved_source = state["adapter_default_raw"]
    if tuple(current_source) != canonical_keys:
        raise ValueError("Loaded policy adapter structure differs from V5 checkpoint")
    for key, current_value in current_source.items():
        saved_value = saved_source[key]
        if (
            current_value.dtype != saved_value.dtype
            or current_value.shape != saved_value.shape
            or not torch.equal(current_value, saved_value.cpu())
        ):
            raise ValueError(
                f"Loaded initial LoRA is not the pretrained source witness: {key}"
            )

    storage_dtype = (
        "float32"
        if all(
            value.dtype == torch.float32
            for adapter_state in adapter_states
            for value in adapter_state.values()
        )
        else "model"
    )
    storage_metadata = {"adapter_storage_dtype": storage_dtype}
    _validate_declared_adapter_storage_dtype(
        metadata=storage_metadata,
        adapter_states=adapter_states,
        v5_state=state["adapter_v5_raw"],
        optimizer_state=state["optimizer_v5"],
        ema_state=state["v5_ema"],
        artifact="pretrained V5 formal checkpoint",
    )

    for adapter_name, adapter_state in (
        ("default", state["adapter_default_raw"]),
        ("old", state["adapter_old"]),
        (V5_ADAPTER_NAME, state["adapter_v5_raw"]),
    ):
        _load_adapter_cpu_state(
            model,
            adapter_name,
            adapter_state,
            require_dtype_match=True,
        )
    optimizer_v5.load_state_dict(state["optimizer_v5"])
    if (v5_scaler is None) != (state["v5_scaler"] is None):
        raise ValueError("Pretrained V5 GradScaler contract mismatch")
    if v5_scaler is not None:
        v5_scaler.load_state_dict(state["v5_scaler"])
    if (v5_ema is None) != (state["v5_ema"] is None):
        raise ValueError("Pretrained V5 EMA contract mismatch")
    if v5_ema is not None:
        v5_ema.load_state_dict(state["v5_ema"])

    theta_parameter_list = list(theta_parameters)
    current_parameters = named_adapter_parameters(model, "default")
    if {id(parameter) for parameter in theta_parameter_list} != {
        id(parameter) for parameter in current_parameters.values()
    }:
        raise ValueError("theta_parameters do not exactly name the default adapter")
    if theta_ema is None or not hasattr(theta_ema, "sync_with_model"):
        raise ValueError("Pretrained V5 initialization requires a fresh theta EMA")
    theta_ema.sync_with_model(theta_parameter_list)
    _assert_loaded_float32_training_state(
        model=model,
        optimizer=optimizer_v5,
        ema=v5_ema,
        metadata=storage_metadata,
        artifact="pretrained V5 formal checkpoint",
    )
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")
    return int(marker_v5_step), {
        "checkpoint_dir": os.fspath(checkpoint_path),
        "state_path": os.fspath(state_path),
        "state_sha256": actual_state_sha256,
        "source_cfg_mode": "cfg4_rollout",
        "source_checkpoint_step": 0,
        "source_v5_optimization_step": int(marker_v5_step),
        "restored_default_old_v5": True,
        "restored_v5_optimizer_scaler_ema": True,
        "synced_fresh_theta_ema": True,
        "formal_step_optimizer_scaler_rng_restored": False,
    }


def _all_reduce_gradients(parameters: Iterable[torch.nn.Parameter], world_size: int) -> None:
    """Average auxiliary-adapter gradients not registered with the main DDP reducer."""

    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("A v5 adapter parameter did not receive a gradient")
        if world_size > 1:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def _globally_has_nonfinite_gradient(
    parameters: Iterable[torch.nn.Parameter], world_size: int
) -> bool:
    """Return one synchronized non-finite decision after reduce/unscale/clip."""

    parameters = list(parameters)
    if not parameters:
        raise ValueError("v5 adapter parameter list cannot be empty")
    found = torch.zeros((), dtype=torch.int32, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("A v5 adapter parameter did not receive a gradient")
        nonfinite = torch.logical_not(torch.isfinite(parameter.grad)).any().to(
            dtype=found.dtype,
            device=found.device,
        )
        found.copy_(torch.maximum(found, nonfinite))
    if world_size > 1:
        dist.all_reduce(found, op=dist.ReduceOp.MAX)
    return bool(found.item())


@torch.no_grad()
def evaluate_v5flow_fm_loss(
    *,
    model,
    samples: dict[str, torch.Tensor],
    num_batches: int,
    num_train_timesteps: int,
    mixed_precision_dtype: torch.dtype | None,
    world_size: int,
    timestep_sampling: str,
    probe_rng_state: dict[str, Any] | None = None,
) -> float:
    """Measure a fixed endpoint probe without changing model or RNG state.

    The center is always the single prompt-conditioned field fitted from the
    current rollout endpoints; a dual theta objective does not create an empty-
    conditioned V5 branch. Each call samples fresh forward-process times/noise inside a saved RNG
    scope. Calling it before and after warm-up from the same outer RNG point
    therefore evaluates both models on exactly the same stochastic probe.
    The returned scalar is averaged across batches, timesteps, and DDP ranks.
    """

    required = {"latents_clean", "timesteps", "prompt_embeds", "pooled_prompt_embeds"}
    missing = sorted(required - set(samples))
    if missing:
        raise ValueError(f"v5flow probe samples are missing keys: {missing}")
    if num_batches <= 0 or num_train_timesteps <= 0:
        raise ValueError("v5flow probe loop counts must be positive")
    total_examples = int(samples["latents_clean"].shape[0])
    if total_examples == 0 or total_examples % num_batches != 0:
        raise ValueError("v5flow probe samples must divide evenly into num_batches")
    schedule_timesteps = int(samples["timesteps"].shape[1])
    if timestep_sampling == "rollout_schedule" and num_train_timesteps > schedule_timesteps:
        raise ValueError("num_train_timesteps exceeds the stored rollout schedule")
    if timestep_sampling not in {"rollout_schedule", "continuous_uniform"}:
        raise ValueError(f"Unsupported timestep sampling mode: {timestep_sampling!r}")

    device = samples["latents_clean"].device
    batch_size = total_examples // num_batches
    enable_amp = mixed_precision_dtype is not None
    rng_state = capture_rng_state(device)
    was_training = bool(model.training)
    loss_sum = 0.0
    loss_count = 0
    try:
        if probe_rng_state is not None:
            restore_rng_state(probe_rng_state, device)
        model.eval()
        model.set_adapter(V5_ADAPTER_NAME)
        training_timesteps = sample_sd3_training_timesteps(
            stored_timesteps=samples["timesteps"],
            num_samples=num_train_timesteps,
            mode=timestep_sampling,
        )
        for batch_index in range(num_batches):
            start = batch_index * batch_size
            end = start + batch_size
            clean = samples["latents_clean"][start:end]
            for timestep_index in range(num_train_timesteps):
                timesteps = training_timesteps[start:end, timestep_index]
                fm_batch = make_sd3_endpoint_flow_matching_batch(
                    clean_latents=clean,
                    timesteps=timesteps,
                )
                with torch.autocast(
                    device_type=device.type,
                    enabled=enable_amp,
                    dtype=mixed_precision_dtype,
                ):
                    prediction = model(
                        hidden_states=fm_batch.state,
                        timestep=timesteps,
                        encoder_hidden_states=samples["prompt_embeds"][start:end],
                        pooled_projections=samples["pooled_prompt_embeds"][start:end],
                        return_dict=False,
                    )[0]
                    loss = (
                        prediction.float() - fm_batch.target_velocity
                    ).square().mean()
                loss_sum += float(loss.item())
                loss_count += 1
    finally:
        restore_rng_state(rng_state, device)
        model.set_adapter("default")
        model.train(was_training)

    aggregate = torch.tensor(
        [loss_sum, float(loss_count)],
        device=device,
        dtype=torch.float64,
    )
    if world_size > 1:
        dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
    if aggregate[1].item() <= 0.0:
        raise RuntimeError("v5flow probe produced no loss values")
    return float((aggregate[0] / aggregate[1]).item())


def train_v5flow_epoch(
    *,
    model,
    samples: dict[str, torch.Tensor],
    num_batches: int,
    num_train_timesteps: int,
    num_inner_epochs: int,
    gradient_accumulation_steps: int,
    optimizer: torch.optim.Optimizer,
    scaler,
    mixed_precision_dtype: torch.dtype | None,
    parameters: list[torch.nn.Parameter],
    ema,
    optimization_step: int,
    max_grad_norm: float,
    world_size: int,
    timestep_sampling: str,
) -> tuple[int, dict[str, float]]:
    """Fit the persistent v5 adapter with unweighted endpoint flow matching.

    The same clean latents, prompt conditioning, and configured time distribution
    as Stage 2 are used. This remains a single shared-center fit even when Stage 2
    trains both conditional and empty theta displacements. Endpoint order and training times are resampled each inner
    epoch, while every source gets fresh Gaussian noise. Since the main DDP
    wrapper only registers the deployed adapter, this auxiliary adapter's
    gradients are explicitly averaged across ranks before each optimizer step.
    """

    if num_batches <= 0 or num_train_timesteps <= 0 or num_inner_epochs <= 0:
        raise ValueError("v5flow loop counts must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    total_examples = int(samples["latents_clean"].shape[0])
    schedule_timesteps = int(samples["timesteps"].shape[1])
    if timestep_sampling == "rollout_schedule" and num_train_timesteps > schedule_timesteps:
        raise ValueError("num_train_timesteps exceeds the stored rollout schedule")
    if total_examples % num_batches != 0:
        raise ValueError("v5flow samples must divide evenly into num_batches")
    if num_batches % gradient_accumulation_steps != 0:
        raise ValueError("v5flow num_batches must divide by gradient_accumulation_steps")
    batch_size = total_examples // num_batches
    effective_accumulation = gradient_accumulation_steps * num_train_timesteps
    enable_amp = mixed_precision_dtype is not None
    loss_sum = 0.0
    backward_count = 0
    optimizer_updates = 0
    optimizer.zero_grad()
    model.train()
    model.set_adapter(V5_ADAPTER_NAME)
    cpu_rng_state = torch.random.get_rng_state()
    cuda_device = samples["latents_clean"].device
    cuda_rng_state = (
        torch.cuda.get_rng_state(cuda_device)
        if cuda_device.type == "cuda"
        else None
    )
    try:
        if timestep_sampling == "continuous_uniform":
            # Keep Stage 2's global RNG stream unchanged while preventing the
            # auxiliary Stage 1 from replaying the exact same permutation,
            # time, and Gaussian draws after the state is restored below.
            torch.rand((), device=cuda_device)
        for _ in range(num_inner_epochs):
            permutation = torch.randperm(total_examples, device=samples["latents_clean"].device)
            shuffled = {key: value[permutation] for key, value in samples.items()}
            training_timesteps = sample_sd3_training_timesteps(
                stored_timesteps=shuffled["timesteps"],
                num_samples=num_train_timesteps,
                mode=timestep_sampling,
            )
            for batch_index in range(num_batches):
                start = batch_index * batch_size
                end = start + batch_size
                for timestep_index in range(num_train_timesteps):
                    clean = shuffled["latents_clean"][start:end]
                    timesteps = training_timesteps[start:end, timestep_index]
                    fm_batch = make_sd3_endpoint_flow_matching_batch(
                        clean_latents=clean,
                        timesteps=timesteps,
                    )
                    with torch.autocast(
                        device_type="cuda",
                        enabled=enable_amp,
                        dtype=mixed_precision_dtype,
                    ):
                        prediction = model(
                            hidden_states=fm_batch.state,
                            timestep=timesteps,
                            encoder_hidden_states=shuffled["prompt_embeds"][start:end],
                            pooled_projections=shuffled["pooled_prompt_embeds"][start:end],
                            return_dict=False,
                        )[0]
                        loss = (prediction.float() - fm_batch.target_velocity).square().mean()
                    scaled_loss = loss / effective_accumulation
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    backward_count += 1
                    loss_sum += float(loss.detach().item())
                    if backward_count % effective_accumulation == 0:
                        # GradScaler records found_inf during unscale_.  Reduce
                        # the still-scaled gradients first so one rank's
                        # non-finite value is visible to every rank before that
                        # decision; otherwise some ranks could step while the
                        # offending rank skips and immediately diverge.
                        _all_reduce_gradients(parameters, world_size)
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                        if _globally_has_nonfinite_gradient(parameters, world_size):
                            optimizer.zero_grad()
                            raise FloatingPointError(
                                "v5flow found a non-finite globally reduced gradient "
                                "after unscale/clip; refusing to step any rank"
                            )
                        if mixed_precision_dtype == torch.float16:
                            scale_before = float(scaler.get_scale())
                            scaler.step(optimizer)
                            scaler.update()
                            scale_after = float(scaler.get_scale())
                            if scale_after < scale_before:
                                optimizer.zero_grad()
                                raise FloatingPointError(
                                    "v5flow GradScaler skipped an optimizer step "
                                    f"at optimization_step={optimization_step} "
                                    f"(scale {scale_before} -> {scale_after}); "
                                    "refusing to advance the step or EMA"
                                )
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        optimization_step += 1
                        optimizer_updates += 1
                        if ema is not None:
                            ema.step(parameters, optimization_step)
        if backward_count % effective_accumulation != 0:
            raise RuntimeError("v5flow ended with a partial gradient-accumulation window")
    finally:
        # Stage 1 is an auxiliary fit. Preserve the released theta/rollout RNG
        # stream so a master-off ablation does not change later permutations,
        # FM noise, or rollout seeds merely by skipping this stage.
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, cuda_device)
        model.set_adapter("default")
    return optimization_step, {
        "v5flow/fm_loss": loss_sum / max(backward_count, 1),
        "v5flow/optimizer_updates": float(optimizer_updates),
        "v5flow/optimization_step": float(optimization_step),
    }


def _adapter_cpu_state(model, adapter_name: str) -> dict[str, torch.Tensor]:
    return {
        key: parameter.detach().cpu().clone()
        for key, parameter in named_adapter_parameters(model, adapter_name).items()
    }


@torch.no_grad()
def _load_adapter_cpu_state(
    model,
    adapter_name: str,
    state: dict[str, torch.Tensor],
    *,
    require_dtype_match: bool = False,
) -> None:
    parameters = named_adapter_parameters(model, adapter_name)
    if parameters.keys() != state.keys():
        raise ValueError(f"Checkpoint structure does not match adapter {adapter_name!r}")
    for key, parameter in parameters.items():
        value = state[key]
        if value.shape != parameter.shape:
            raise ValueError(f"Checkpoint tensor shape mismatch for {key}")
        if require_dtype_match and value.dtype != parameter.dtype:
            raise ValueError(
                f"Checkpoint tensor dtype mismatch for {key}: "
                f"{value.dtype} != {parameter.dtype}"
            )
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


@torch.no_grad()
def promote_v5flow_ema_to_policy_(
    *,
    checkpoint_dir: str | Path,
    model,
    theta_parameters: Iterable[torch.nn.Parameter],
    theta_ema,
    expected_state_sha256: str,
) -> dict[str, Any]:
    """Initialize a fresh old-center policy from a pre-outer V5 EMA.

    The source must be a completed CFG4-rollout checkpoint at formal step zero:
    only the V5 warm-up may have advanced.  The currently loaded ``default``
    adapter is first checked bit-for-bit against the checkpoint's source-policy
    witness.  The saved V5 EMA is then copied into ``default`` and ``old``, and
    the freshly constructed theta EMA is synchronized to the same values.

    No optimizer, scaler, formal step, RNG, raw theta, or V5 optimizer state is
    restored.  Callers therefore continue from global step zero with a fresh
    reward-finetuning optimizer.
    """

    checkpoint_path = Path(checkpoint_dir).expanduser()
    if not checkpoint_path.is_absolute():
        raise ValueError("V5 promotion checkpoint must be absolute")
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"V5 promotion checkpoint directory is missing: {checkpoint_path}"
        )
    if re.fullmatch(r"[0-9a-f]{64}", expected_state_sha256) is None:
        raise ValueError("Expected V5 promotion state SHA256 must be lowercase hex")

    marker_path = checkpoint_path / CHECKPOINT_COMPLETE_MARKER_FILENAME
    state_path = checkpoint_path / V5FLOW_STATE_FILENAME
    if not marker_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"Incomplete V5 promotion checkpoint: {checkpoint_path}"
        )
    with marker_path.open("r", encoding="utf-8") as marker_file:
        marker = json.load(marker_file)
    if not isinstance(marker, dict) or set(marker) != V5FLOW_PROMOTION_MARKER_KEYS:
        actual_keys = set(marker) if isinstance(marker, dict) else set()
        raise ValueError(
            "Unexpected V5 promotion marker keys: "
            f"missing={sorted(V5FLOW_PROMOTION_MARKER_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - V5FLOW_PROMOTION_MARKER_KEYS)}"
        )
    expected_marker_values = {
        "schema_version": 1,
        "cfg_mode": "cfg4_rollout",
        "checkpoint_kind": "outer_boundary",
        "checkpoint_step": 0,
        "resume_epoch": 0,
        "rollout_guidance_scale": 4.0,
    }
    for key, expected_value in expected_marker_values.items():
        if marker[key] != expected_value:
            raise ValueError(
                f"V5 promotion marker {key} mismatch: "
                f"{marker[key]!r} != {expected_value!r}"
            )
    marker_v5_step = marker["v5_optimization_step"]
    if (
        not isinstance(marker_v5_step, int)
        or isinstance(marker_v5_step, bool)
        or marker_v5_step <= 0
    ):
        raise ValueError("V5 promotion requires a positive warm-up optimizer step")

    actual_state_sha256 = sha256_file(state_path)
    if actual_state_sha256 != expected_state_sha256:
        raise ValueError(
            "V5 promotion state SHA256 mismatch: "
            f"{actual_state_sha256} != {expected_state_sha256}"
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or set(state) != V5FLOW_STATE_KEYS:
        actual_keys = set(state) if isinstance(state, dict) else set()
        raise ValueError(
            "Unexpected V5 promotion state keys: "
            f"missing={sorted(V5FLOW_STATE_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - V5FLOW_STATE_KEYS)}"
        )
    if state["schema_version"] != 1:
        raise ValueError(
            f"Unsupported V5 promotion state schema: {state['schema_version']!r}"
        )
    if state["v5_optimization_step"] != marker_v5_step:
        raise ValueError("V5 promotion marker and state optimization steps disagree")

    adapter_fields = (
        "adapter_default_raw",
        "adapter_old",
        "adapter_v5_raw",
    )
    adapter_states = [state[field] for field in adapter_fields]
    if any(not isinstance(value, dict) or not value for value in adapter_states):
        raise ValueError("V5 promotion checkpoint contains an invalid adapter state")
    canonical_keys = tuple(adapter_states[0])
    if canonical_keys != tuple(sorted(canonical_keys)) or any(
        tuple(value) != canonical_keys for value in adapter_states[1:]
    ):
        raise ValueError(
            "V5 promotion adapter states must share canonical sorted key order"
        )

    v5_ema_state = state["v5_ema"]
    if (
        not isinstance(v5_ema_state, dict)
        or set(v5_ema_state) != {"decay", "ema_parameters"}
        or not isinstance(v5_ema_state["ema_parameters"], list)
    ):
        raise ValueError("V5 promotion checkpoint does not contain a usable V5 EMA")
    v5_ema_decay = v5_ema_state["decay"]
    if (
        isinstance(v5_ema_decay, bool)
        or not isinstance(v5_ema_decay, (int, float))
        or not math.isfinite(float(v5_ema_decay))
        or not 0.0 <= float(v5_ema_decay) < 1.0
    ):
        raise ValueError("V5 promotion checkpoint has an invalid EMA decay")
    ema_parameters = v5_ema_state["ema_parameters"]
    if len(ema_parameters) != len(canonical_keys):
        raise ValueError("V5 promotion EMA tensor count does not match the adapter")

    promoted_state: dict[str, torch.Tensor] = {}
    for index, (key, ema_value) in enumerate(
        zip(canonical_keys, ema_parameters, strict=True)
    ):
        source_value = state["adapter_default_raw"][key]
        old_value = state["adapter_old"][key]
        raw_v5_value = state["adapter_v5_raw"][key]
        values = (source_value, old_value, raw_v5_value, ema_value)
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise ValueError(f"Invalid V5 promotion tensor at index {index}: {key}")
        if any(value.shape != source_value.shape for value in values[1:]):
            raise ValueError(f"V5 promotion tensor shape mismatch: {key}")
        if any(value.dtype != source_value.dtype for value in values[1:]):
            raise ValueError(f"V5 promotion tensor dtype mismatch: {key}")
        if ema_value.dtype != raw_v5_value.dtype:
            raise ValueError(f"V5 promotion raw/EMA dtype mismatch: {key}")
        if any(
            not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
            for value in values
        ):
            raise ValueError(f"V5 promotion tensor is invalid: {key}")
        promoted_state[key] = ema_value

    current_source = named_adapter_parameters(model, "default")
    old_target = named_adapter_parameters(model, "old")
    if tuple(current_source) != canonical_keys or tuple(old_target) != canonical_keys:
        raise ValueError("Loaded policy adapter structure differs from V5 checkpoint")
    for key, parameter in current_source.items():
        saved_source = state["adapter_default_raw"][key]
        actual_source = parameter.detach().cpu()
        if (
            actual_source.dtype != saved_source.dtype
            or actual_source.shape != saved_source.shape
            or not torch.equal(actual_source, saved_source)
        ):
            raise ValueError(
                f"Loaded initial LoRA is not the checkpoint source witness: {key}"
            )
        old_parameter = old_target[key]
        if (
            old_parameter.shape != promoted_state[key].shape
            or old_parameter.dtype != promoted_state[key].dtype
        ):
            raise ValueError(f"Old-policy promotion target is incompatible: {key}")

    theta_parameter_list = list(theta_parameters)
    if {id(parameter) for parameter in theta_parameter_list} != {
        id(parameter) for parameter in current_source.values()
    }:
        raise ValueError("theta_parameters do not exactly name the default adapter")
    if (
        theta_ema is None
        or not hasattr(theta_ema, "sync_with_model")
        or not isinstance(getattr(theta_ema, "ema_parameters", None), list)
        or len(theta_ema.ema_parameters) != len(theta_parameter_list)
    ):
        raise ValueError("V5 promotion requires a fresh theta EMA")
    for index, (ema_parameter, theta_parameter) in enumerate(
        zip(theta_ema.ema_parameters, theta_parameter_list, strict=True)
    ):
        if (
            not isinstance(ema_parameter, torch.Tensor)
            or ema_parameter.shape != theta_parameter.shape
            or ema_parameter.dtype != theta_parameter.dtype
        ):
            raise ValueError(
                f"Fresh theta EMA is incompatible at parameter index {index}"
            )

    _load_adapter_cpu_state(
        model,
        "default",
        promoted_state,
        require_dtype_match=True,
    )
    copy_adapter_parameters_(model, source="default", target="old")
    theta_ema.sync_with_model(theta_parameter_list)
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")

    return {
        "checkpoint_dir": os.fspath(checkpoint_path),
        "state_path": os.fspath(state_path),
        "state_sha256": actual_state_sha256,
        "source_checkpoint_step": 0,
        "source_v5_optimization_step": marker_v5_step,
        "source_cfg_mode": "cfg4_rollout",
        "source_rollout_guidance_scale": 4.0,
        "promoted_snapshot": "v5_ema",
        "promoted_tensor_count": len(promoted_state),
        "optimizer_restored": False,
        "scaler_restored": False,
        "formal_step_restored": False,
    }


def _validate_declared_adapter_storage_dtype(
    *,
    metadata: dict[str, Any],
    adapter_states: Iterable[dict[str, torch.Tensor]],
    v5_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    ema_state: dict[str, Any] | None,
    artifact: str,
) -> None:
    """Enforce raw, AdamW, and EMA dtypes before any loader can cast them."""

    declared = metadata.get("adapter_storage_dtype")
    if declared is None or declared == "model":
        return
    if declared != "float32":
        raise ValueError(f"Unsupported {artifact} adapter_storage_dtype: {declared!r}")
    for state in adapter_states:
        for key, value in state.items():
            if value.dtype != torch.float32:
                raise ValueError(
                    f"{artifact} declares float32 storage but {key} is {value.dtype}"
                )
    expected_shapes = [value.shape for value in v5_state.values()]
    param_ids = [
        parameter_id
        for group in optimizer_state.get("param_groups", [])
        for parameter_id in group.get("params", [])
    ]
    optimizer_slots = optimizer_state.get("state", {})
    if len(param_ids) != len(expected_shapes) or set(param_ids) != set(optimizer_slots):
        raise ValueError(
            f"{artifact} AdamW parameter-state structure does not match v5"
        )
    for expected_shape, parameter_id in zip(expected_shapes, param_ids, strict=True):
        parameter_slots = optimizer_slots[parameter_id]
        if not isinstance(parameter_slots, dict):
            raise ValueError(f"{artifact} AdamW state is invalid for {parameter_id}")
        for slot_name in ("exp_avg", "exp_avg_sq"):
            slot = parameter_slots.get(slot_name)
            if (
                not isinstance(slot, torch.Tensor)
                or slot.dtype != torch.float32
                or slot.shape != expected_shape
            ):
                raise ValueError(
                    f"{artifact} AdamW {slot_name} must be float32 with shape "
                    f"{tuple(expected_shape)} for parameter {parameter_id}"
                )
    if not isinstance(ema_state, dict):
        raise ValueError(f"{artifact} float32 contract requires v5 EMA state")
    ema_parameters = ema_state.get("ema_parameters")
    if not isinstance(ema_parameters, list) or len(ema_parameters) != len(expected_shapes):
        raise ValueError(f"{artifact} v5 EMA structure does not match v5")
    for index, (ema_parameter, expected_shape) in enumerate(
        zip(ema_parameters, expected_shapes, strict=True)
    ):
        if (
            not isinstance(ema_parameter, torch.Tensor)
            or ema_parameter.dtype != torch.float32
            or ema_parameter.shape != expected_shape
        ):
            raise ValueError(
                f"{artifact} v5 EMA tensor {index} must be float32 with shape "
                f"{tuple(expected_shape)}"
            )


def _assert_loaded_float32_training_state(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    ema,
    metadata: dict[str, Any],
    artifact: str,
) -> None:
    """Verify PyTorch loaders did not silently cast a declared F32 artifact."""

    if metadata.get("adapter_storage_dtype") != "float32":
        return
    parameters = adapter_parameters(model, V5_ADAPTER_NAME)
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError(f"{artifact} loaded v5 parameters are not all float32")
    for index, parameter in enumerate(parameters):
        slots = optimizer.state.get(parameter)
        if not isinstance(slots, dict):
            raise RuntimeError(f"{artifact} loaded AdamW state {index} is missing")
        for slot_name in ("exp_avg", "exp_avg_sq"):
            slot = slots.get(slot_name)
            if (
                not isinstance(slot, torch.Tensor)
                or slot.dtype != torch.float32
                or slot.shape != parameter.shape
            ):
                raise RuntimeError(
                    f"{artifact} loaded AdamW {slot_name} {index} violates F32"
                )
    ema_parameters = getattr(ema, "ema_parameters", None)
    if not isinstance(ema_parameters, list) or len(ema_parameters) != len(parameters):
        raise RuntimeError(f"{artifact} loaded v5 EMA structure is invalid")
    for index, (ema_parameter, parameter) in enumerate(
        zip(ema_parameters, parameters, strict=True)
    ):
        if ema_parameter.dtype != torch.float32 or ema_parameter.shape != parameter.shape:
            raise RuntimeError(f"{artifact} loaded v5 EMA tensor {index} violates F32")


def _validate_v5flow_initializer_state(
    state: Any,
    *,
    path: Path,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "source_adapter_default",
        "adapter_v5_raw",
        "optimizer_v5",
        "v5_scaler",
        "v5_ema",
        "v5_optimization_step",
        "metadata",
    }
    if not isinstance(state, dict) or set(state) != expected_keys:
        actual_keys = set(state) if isinstance(state, dict) else set()
        raise ValueError(
            f"Unexpected v5flow initializer keys in {path}: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    if state["schema_version"] != 1:
        raise ValueError(
            f"Unsupported v5flow initializer schema in {path}: "
            f"{state['schema_version']!r}"
        )
    optimization_step = state["v5_optimization_step"]
    if (
        not isinstance(optimization_step, int)
        or isinstance(optimization_step, bool)
        or optimization_step <= 0
    ):
        raise ValueError("v5flow initializer optimization step must be positive")
    source_state = state["source_adapter_default"]
    v5_state = state["adapter_v5_raw"]
    if (
        not isinstance(source_state, dict)
        or not source_state
        or not isinstance(v5_state, dict)
        or source_state.keys() != v5_state.keys()
    ):
        raise ValueError("v5flow initializer adapters are empty or structurally different")
    for key, source_value in source_state.items():
        v5_value = v5_state[key]
        if (
            not isinstance(source_value, torch.Tensor)
            or not isinstance(v5_value, torch.Tensor)
            or source_value.shape != v5_value.shape
        ):
            raise ValueError(f"Invalid v5flow initializer adapter tensor: {key}")
    optimizer_state = state["optimizer_v5"]
    if (
        not isinstance(optimizer_state, dict)
        or not isinstance(optimizer_state.get("param_groups"), list)
        or not optimizer_state.get("state")
    ):
        raise ValueError("v5flow initializer is missing trained optimizer state")
    if not isinstance(state["metadata"], dict):
        raise ValueError("v5flow initializer metadata must be a mapping")
    _validate_declared_adapter_storage_dtype(
        metadata=state["metadata"],
        adapter_states=(source_state, v5_state),
        v5_state=v5_state,
        optimizer_state=optimizer_state,
        ema_state=state["v5_ema"],
        artifact="v5flow initializer",
    )
    return state


def save_v5flow_initializer(
    *,
    output_dir: str | Path,
    model,
    optimizer_v5: torch.optim.Optimizer,
    v5_scaler,
    v5_ema,
    v5_optimization_step: int,
    metadata: dict[str, Any],
) -> Path:
    """Atomically publish a reusable v5-only initializer directory.

    The source policy adapter is stored only as a compatibility witness.  A
    consumer restores the raw v5 adapter, its AdamW state, optional GradScaler,
    and v5 EMA; it never restores theta, old, theta EMA, or the formal step.
    Existing destinations are never overwritten.
    """

    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Refusing to overwrite v5flow initializer: {output_dir}")
    if not isinstance(metadata, dict):
        raise ValueError("v5flow initializer metadata must be a mapping")
    try:
        json.dumps(metadata, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("v5flow initializer metadata must be JSON serializable") from error

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    publication_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    staging_dir = output_dir.with_name(
        f".{output_dir.name}.publishing-{publication_id}-{os.getpid()}"
    )
    if os.path.lexists(staging_dir):
        raise FileExistsError(f"Refusing to reuse v5flow staging directory: {staging_dir}")

    state = {
        "schema_version": 1,
        "source_adapter_default": _adapter_cpu_state(model, "default"),
        "adapter_v5_raw": _adapter_cpu_state(model, V5_ADAPTER_NAME),
        "optimizer_v5": optimizer_v5.state_dict(),
        "v5_scaler": v5_scaler.state_dict() if v5_scaler is not None else None,
        "v5_ema": v5_ema.state_dict() if v5_ema is not None else None,
        "v5_optimization_step": int(v5_optimization_step),
        "metadata": dict(metadata),
    }
    _validate_v5flow_initializer_state(state, path=staging_dir)
    try:
        staging_dir.mkdir()
        state_path = staging_dir / V5FLOW_INITIALIZER_STATE_FILENAME
        with state_path.open("xb") as state_file:
            torch.save(state, state_file)
            state_file.flush()
            os.fsync(state_file.fileno())
        marker = {
            "schema_version": 1,
            "state_filename": V5FLOW_INITIALIZER_STATE_FILENAME,
            "state_sha256": sha256_file(state_path),
            "v5_optimization_step": int(v5_optimization_step),
            "metadata": dict(metadata),
        }
        marker_path = staging_dir / V5FLOW_INITIALIZER_MARKER_FILENAME
        with marker_path.open("x", encoding="utf-8") as marker_file:
            json.dump(marker, marker_file, indent=2, sort_keys=True)
            marker_file.write("\n")
            marker_file.flush()
            os.fsync(marker_file.fileno())
        os.replace(staging_dir, output_dir)
    finally:
        if staging_dir.is_dir():
            shutil.rmtree(staging_dir)
    return output_dir


def load_v5flow_initializer(
    *,
    initializer_dir: str | Path,
    model,
    optimizer_v5: torch.optim.Optimizer,
    v5_scaler,
    v5_ema,
    map_location: torch.device,
    expected_metadata: dict[str, Any],
) -> int:
    """Load only reusable v5 state after exact source-policy validation."""

    initializer_dir = Path(initializer_dir).expanduser().resolve()
    marker_path = initializer_dir / V5FLOW_INITIALIZER_MARKER_FILENAME
    state_path = initializer_dir / V5FLOW_INITIALIZER_STATE_FILENAME
    if not marker_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"Incomplete v5flow initializer directory: {initializer_dir}"
        )
    with marker_path.open("r", encoding="utf-8") as marker_file:
        marker = json.load(marker_file)
    legacy_marker_keys = {
        "schema_version",
        "state_filename",
        "v5_optimization_step",
        "metadata",
    }
    hashed_marker_keys = legacy_marker_keys | {"state_sha256"}
    if (
        not isinstance(marker, dict)
        or frozenset(marker) not in {
            frozenset(legacy_marker_keys),
            frozenset(hashed_marker_keys),
        }
        or marker.get("schema_version") != 1
        or marker.get("state_filename") != V5FLOW_INITIALIZER_STATE_FILENAME
    ):
        raise ValueError(f"Invalid v5flow initializer marker: {marker_path}")
    if "state_sha256" in marker:
        digest = marker["state_sha256"]
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or sha256_file(state_path) != digest
        ):
            raise ValueError(f"v5flow initializer state digest mismatch: {state_path}")
    state = torch.load(state_path, map_location=map_location, weights_only=False)
    state = _validate_v5flow_initializer_state(state, path=state_path)
    if (
        marker["v5_optimization_step"] != state["v5_optimization_step"]
        or marker["metadata"] != state["metadata"]
    ):
        raise ValueError("v5flow initializer marker and state disagree")
    if not isinstance(expected_metadata, dict):
        raise ValueError("expected v5flow initializer metadata must be a mapping")
    for key, expected_value in expected_metadata.items():
        if state["metadata"].get(key) != expected_value:
            raise ValueError(
                f"v5flow initializer metadata mismatch for {key}: "
                f"{state['metadata'].get(key)!r} != {expected_value!r}"
            )

    current_source = _adapter_cpu_state(model, "default")
    saved_source = state["source_adapter_default"]
    if current_source.keys() != saved_source.keys():
        raise ValueError("v5flow initializer source adapter structure mismatch")
    for key, current_value in current_source.items():
        saved_value = saved_source[key]
        if (
            current_value.dtype != saved_value.dtype
            or current_value.shape != saved_value.shape
            or not torch.equal(current_value, saved_value.cpu())
        ):
            raise ValueError(
                f"v5flow initializer was trained from a different source policy: {key}"
            )

    _load_adapter_cpu_state(
        model,
        V5_ADAPTER_NAME,
        state["adapter_v5_raw"],
        require_dtype_match=True,
    )
    optimizer_v5.load_state_dict(state["optimizer_v5"])
    if (v5_scaler is None) != (state["v5_scaler"] is None):
        raise ValueError("v5flow initializer GradScaler contract mismatch")
    if v5_scaler is not None:
        v5_scaler.load_state_dict(state["v5_scaler"])
    if (v5_ema is None) != (state["v5_ema"] is None):
        raise ValueError("v5flow initializer EMA contract mismatch")
    if v5_ema is not None:
        v5_ema.load_state_dict(state["v5_ema"])
    _assert_loaded_float32_training_state(
        model=model,
        optimizer=optimizer_v5,
        ema=v5_ema,
        metadata=state["metadata"],
        artifact="v5flow initializer",
    )
    return int(state["v5_optimization_step"])


def v5flow_pretrain_rank_state_filename(rank: int) -> str:
    """Return the rank-local RNG filename used by a flow-pretrain checkpoint."""

    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    return f"rank-{rank:05d}-rng.pt"


def _validate_v5flow_pretrain_state(state: Any, *, path: Path) -> dict[str, Any]:
    """Validate the common, rank-independent part of a pretrain checkpoint."""

    expected_keys = {
        "schema_version",
        "source_adapter_default",
        "adapter_v5_raw",
        "optimizer_v5",
        "v5_scaler",
        "v5_ema",
        "v5_optimization_step",
        "completed_refreshes",
        "metadata",
    }
    if not isinstance(state, dict) or set(state) != expected_keys:
        actual_keys = set(state) if isinstance(state, dict) else set()
        raise ValueError(
            f"Unexpected v5flow pretrain keys in {path}: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    if state["schema_version"] != 1:
        raise ValueError(
            f"Unsupported v5flow pretrain schema in {path}: "
            f"{state['schema_version']!r}"
        )
    for field in ("v5_optimization_step", "completed_refreshes"):
        value = state[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"v5flow pretrain {field} must be a positive integer")
    source_state = state["source_adapter_default"]
    v5_state = state["adapter_v5_raw"]
    if (
        not isinstance(source_state, dict)
        or not source_state
        or not isinstance(v5_state, dict)
        or source_state.keys() != v5_state.keys()
    ):
        raise ValueError("v5flow pretrain adapters are empty or structurally different")
    for key, source_value in source_state.items():
        v5_value = v5_state[key]
        if (
            not isinstance(source_value, torch.Tensor)
            or not isinstance(v5_value, torch.Tensor)
            or source_value.shape != v5_value.shape
            or not bool(torch.isfinite(source_value).all().item())
            or not bool(torch.isfinite(v5_value).all().item())
        ):
            raise ValueError(f"Invalid v5flow pretrain adapter tensor: {key}")
    optimizer_state = state["optimizer_v5"]
    if (
        not isinstance(optimizer_state, dict)
        or not isinstance(optimizer_state.get("param_groups"), list)
        or not optimizer_state.get("state")
    ):
        raise ValueError("v5flow pretrain checkpoint is missing trained optimizer state")
    if not isinstance(state["metadata"], dict):
        raise ValueError("v5flow pretrain metadata must be a mapping")
    _validate_declared_adapter_storage_dtype(
        metadata=state["metadata"],
        adapter_states=(source_state, v5_state),
        v5_state=v5_state,
        optimizer_state=optimizer_state,
        ema_state=state["v5_ema"],
        artifact="v5flow pretrain checkpoint",
    )
    return state


def _validate_v5flow_pretrain_rank_state(
    state: Any,
    *,
    path: Path,
    rank: int,
    world_size: int,
    v5_optimization_step: int,
    completed_refreshes: int,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "rank",
        "world_size",
        "v5_optimization_step",
        "completed_refreshes",
        "rng_state",
    }
    if not isinstance(state, dict) or set(state) != expected_keys:
        raise ValueError(f"Unexpected v5flow pretrain rank-state keys in {path}")
    expected_values = {
        "schema_version": V5FLOW_PRETRAIN_RANK_STATE_SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "v5_optimization_step": v5_optimization_step,
        "completed_refreshes": completed_refreshes,
    }
    for key, expected in expected_values.items():
        if state[key] != expected:
            raise ValueError(
                f"v5flow pretrain rank state {key} mismatch in {path}: "
                f"{state[key]!r} != {expected!r}"
            )
    rng_state = state["rng_state"]
    if not isinstance(rng_state, dict) or set(rng_state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError(f"Invalid v5flow pretrain RNG state in {path}")
    return state


def _load_v5flow_pretrain_marker(
    checkpoint_dir: Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    marker_path = checkpoint_dir / V5FLOW_PRETRAIN_MARKER_FILENAME
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError(f"v5flow pretrain checkpoint must be a real directory: {checkpoint_dir}")
    match = _ROLLING_CHECKPOINT_NAME.fullmatch(checkpoint_dir.name)
    if match is None:
        raise ValueError(f"Invalid v5flow pretrain checkpoint name: {checkpoint_dir.name}")
    if not marker_path.is_file():
        raise FileNotFoundError(f"Missing v5flow pretrain marker: {marker_path}")
    with marker_path.open("r", encoding="utf-8") as marker_file:
        marker = json.load(marker_file)
    expected_keys = {
        "schema_version",
        "state_filename",
        "v5_optimization_step",
        "completed_refreshes",
        "world_size",
        "rank_rng_files",
        "state_sha256",
        "rank_rng_sha256",
        "metadata",
    }
    if (
        not isinstance(marker, dict)
        or set(marker) != expected_keys
        or marker.get("schema_version") != 1
        or marker.get("state_filename") != V5FLOW_PRETRAIN_STATE_FILENAME
    ):
        raise ValueError(f"Invalid v5flow pretrain marker: {marker_path}")
    for field in ("v5_optimization_step", "completed_refreshes", "world_size"):
        value = marker[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"v5flow pretrain marker {field} must be positive")
    if int(match.group(1)) != marker["v5_optimization_step"]:
        raise ValueError(f"v5flow pretrain marker step disagrees with directory: {marker_path}")
    expected_rank_files = [
        v5flow_pretrain_rank_state_filename(rank)
        for rank in range(marker["world_size"])
    ]
    if marker["rank_rng_files"] != expected_rank_files:
        raise ValueError(f"v5flow pretrain marker rank files are invalid: {marker_path}")
    digest_pattern = re.compile(r"[0-9a-f]{64}\Z")
    if not isinstance(marker["state_sha256"], str) or not digest_pattern.fullmatch(
        marker["state_sha256"]
    ):
        raise ValueError(f"v5flow pretrain marker state digest is invalid: {marker_path}")
    expected_rank_digest_keys = set(expected_rank_files)
    rank_digests = marker["rank_rng_sha256"]
    if (
        not isinstance(rank_digests, dict)
        or set(rank_digests) != expected_rank_digest_keys
        or any(
            not isinstance(digest, str) or not digest_pattern.fullmatch(digest)
            for digest in rank_digests.values()
        )
    ):
        raise ValueError(f"v5flow pretrain marker rank digests are invalid: {marker_path}")
    required_files = [
        checkpoint_dir / V5FLOW_PRETRAIN_STATE_FILENAME,
        *(checkpoint_dir / filename for filename in expected_rank_files),
    ]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required_files):
        raise ValueError(f"v5flow pretrain checkpoint is incomplete: {checkpoint_dir}")
    if verify_hashes:
        state_path = checkpoint_dir / V5FLOW_PRETRAIN_STATE_FILENAME
        if sha256_file(state_path) != marker["state_sha256"]:
            raise ValueError(f"v5flow pretrain common-state digest mismatch: {state_path}")
        for filename in expected_rank_files:
            rank_path = checkpoint_dir / filename
            if sha256_file(rank_path) != rank_digests[filename]:
                raise ValueError(f"v5flow pretrain rank-state digest mismatch: {rank_path}")
    if not isinstance(marker["metadata"], dict):
        raise ValueError(f"v5flow pretrain marker metadata is invalid: {marker_path}")
    return marker


def find_latest_v5flow_pretrain_checkpoint(
    checkpoint_root: str | Path,
) -> Path | None:
    """Return the newest complete refresh checkpoint, ignoring partial writes."""

    checkpoint_root = Path(checkpoint_root).expanduser().resolve(strict=False)
    if not checkpoint_root.exists():
        return None
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ValueError(f"v5flow pretrain checkpoint_root must be a real directory: {checkpoint_root}")
    complete: list[tuple[int, Path]] = []
    for child in checkpoint_root.iterdir():
        try:
            marker = _load_v5flow_pretrain_marker(child, verify_hashes=False)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            continue
        complete.append((int(marker["v5_optimization_step"]), child.resolve()))
    if not complete:
        return None
    complete.sort(key=lambda item: item[0], reverse=True)
    for _, candidate in complete:
        try:
            _load_v5flow_pretrain_marker(candidate, verify_hashes=True)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            continue
        return candidate
    return None


def save_v5flow_pretrain_checkpoint(
    *,
    output_dir: str | Path,
    model,
    optimizer_v5: torch.optim.Optimizer,
    v5_scaler,
    v5_ema,
    v5_optimization_step: int,
    completed_refreshes: int,
    rng_state: dict[str, Any],
    rank: int,
    world_size: int,
    metadata: dict[str, Any],
) -> Path:
    """Atomically publish one refresh-boundary v5 pretrain checkpoint.

    Rank 0 writes the adapter/optimizer/scaler/EMA state and every rank writes
    its own RNG stream.  The directory becomes discoverable only after all
    files are complete and rank 0 atomically renames the staging directory.
    """

    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if rank >= world_size:
        raise ValueError("rank must be smaller than world_size")
    for field, value in {
        "v5_optimization_step": v5_optimization_step,
        "completed_refreshes": completed_refreshes,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(metadata, dict):
        raise ValueError("v5flow pretrain metadata must be a mapping")
    try:
        json.dumps(metadata, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("v5flow pretrain metadata must be JSON serializable") from error
    if world_size > 1 and (not dist.is_available() or not dist.is_initialized()):
        raise RuntimeError("distributed v5flow pretrain checkpoint requires initialized torch.distributed")

    publication_id = os.environ.get("SLURM_JOB_ID", "local")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", publication_id):
        raise ValueError(f"Unsafe checkpoint publication id: {publication_id!r}")
    staging_dir = output_dir.with_name(
        f".{output_dir.name}.publishing-{publication_id}"
    )
    if rank == 0:
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Refusing to overwrite v5flow pretrain checkpoint: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(staging_dir):
            if staging_dir.is_symlink() or not staging_dir.is_dir():
                raise ValueError(f"Unsafe stale v5flow pretrain staging path: {staging_dir}")
            shutil.rmtree(staging_dir)
        staging_dir.mkdir()
    if world_size > 1:
        dist.barrier()

    rank_state = {
        "schema_version": V5FLOW_PRETRAIN_RANK_STATE_SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "v5_optimization_step": v5_optimization_step,
        "completed_refreshes": completed_refreshes,
        "rng_state": _cpu_tensor_tree(rng_state),
    }
    rank_path = staging_dir / v5flow_pretrain_rank_state_filename(rank)
    _validate_v5flow_pretrain_rank_state(
        rank_state,
        path=rank_path,
        rank=rank,
        world_size=world_size,
        v5_optimization_step=v5_optimization_step,
        completed_refreshes=completed_refreshes,
    )
    rank_temporary = rank_path.with_name(f".{rank_path.name}.{os.getpid()}.tmp")
    with rank_temporary.open("xb") as rank_file:
        torch.save(rank_state, rank_file)
        rank_file.flush()
        os.fsync(rank_file.fileno())
    os.replace(rank_temporary, rank_path)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        state = {
            "schema_version": 1,
            "source_adapter_default": _adapter_cpu_state(model, "default"),
            "adapter_v5_raw": _adapter_cpu_state(model, V5_ADAPTER_NAME),
            "optimizer_v5": optimizer_v5.state_dict(),
            "v5_scaler": v5_scaler.state_dict() if v5_scaler is not None else None,
            "v5_ema": v5_ema.state_dict() if v5_ema is not None else None,
            "v5_optimization_step": v5_optimization_step,
            "completed_refreshes": completed_refreshes,
            "metadata": dict(metadata),
        }
        state_path = staging_dir / V5FLOW_PRETRAIN_STATE_FILENAME
        _validate_v5flow_pretrain_state(state, path=state_path)
        with state_path.open("xb") as state_file:
            torch.save(state, state_file)
            state_file.flush()
            os.fsync(state_file.fileno())
        marker = {
            "schema_version": 1,
            "state_filename": V5FLOW_PRETRAIN_STATE_FILENAME,
            "v5_optimization_step": v5_optimization_step,
            "completed_refreshes": completed_refreshes,
            "world_size": world_size,
            "rank_rng_files": [
                v5flow_pretrain_rank_state_filename(state_rank)
                for state_rank in range(world_size)
            ],
            "state_sha256": sha256_file(state_path),
            "rank_rng_sha256": {
                v5flow_pretrain_rank_state_filename(state_rank): sha256_file(
                    staging_dir / v5flow_pretrain_rank_state_filename(state_rank)
                )
                for state_rank in range(world_size)
            },
            "metadata": dict(metadata),
        }
        marker_path = staging_dir / V5FLOW_PRETRAIN_MARKER_FILENAME
        with marker_path.open("x", encoding="utf-8") as marker_file:
            json.dump(marker, marker_file, indent=2, sort_keys=True)
            marker_file.write("\n")
            marker_file.flush()
            os.fsync(marker_file.fileno())
        os.replace(staging_dir, output_dir)
    if world_size > 1:
        dist.barrier()
    return output_dir


def load_v5flow_pretrain_checkpoint(
    *,
    checkpoint_dir: str | Path,
    model,
    optimizer_v5: torch.optim.Optimizer,
    v5_scaler,
    v5_ema,
    map_location: torch.device,
    rank: int,
    world_size: int,
    expected_metadata: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    """Restore v5 pretrain state plus this rank's exact refresh-boundary RNG."""

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    marker = _load_v5flow_pretrain_marker(checkpoint_dir)
    if marker["world_size"] != world_size:
        raise ValueError(
            "v5flow pretrain world size mismatch: "
            f"{marker['world_size']} != {world_size}"
        )
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
        raise ValueError("rank must be an integer in [0, world_size)")
    if marker["metadata"] != expected_metadata:
        raise ValueError("v5flow pretrain checkpoint metadata does not match this run")
    state_path = checkpoint_dir / V5FLOW_PRETRAIN_STATE_FILENAME
    state = torch.load(state_path, map_location=map_location, weights_only=False)
    state = _validate_v5flow_pretrain_state(state, path=state_path)
    if (
        state["v5_optimization_step"] != marker["v5_optimization_step"]
        or state["completed_refreshes"] != marker["completed_refreshes"]
        or state["metadata"] != marker["metadata"]
    ):
        raise ValueError("v5flow pretrain marker and common state disagree")

    current_source = _adapter_cpu_state(model, "default")
    saved_source = state["source_adapter_default"]
    if current_source.keys() != saved_source.keys():
        raise ValueError("v5flow pretrain source adapter structure mismatch")
    for key, current_value in current_source.items():
        saved_value = saved_source[key]
        if (
            current_value.dtype != saved_value.dtype
            or current_value.shape != saved_value.shape
            or not torch.equal(current_value, saved_value.cpu())
        ):
            raise ValueError(
                f"v5flow pretrain checkpoint used a different source policy: {key}"
            )
    _load_adapter_cpu_state(
        model,
        V5_ADAPTER_NAME,
        state["adapter_v5_raw"],
        require_dtype_match=True,
    )
    optimizer_v5.load_state_dict(state["optimizer_v5"])
    if (v5_scaler is None) != (state["v5_scaler"] is None):
        raise ValueError("v5flow pretrain GradScaler contract mismatch")
    if v5_scaler is not None:
        v5_scaler.load_state_dict(state["v5_scaler"])
    if (v5_ema is None) != (state["v5_ema"] is None):
        raise ValueError("v5flow pretrain EMA contract mismatch")
    if v5_ema is not None:
        v5_ema.load_state_dict(state["v5_ema"])
    _assert_loaded_float32_training_state(
        model=model,
        optimizer=optimizer_v5,
        ema=v5_ema,
        metadata=state["metadata"],
        artifact="v5flow pretrain checkpoint",
    )

    rank_path = checkpoint_dir / v5flow_pretrain_rank_state_filename(rank)
    rank_state = torch.load(rank_path, map_location="cpu", weights_only=False)
    rank_state = _validate_v5flow_pretrain_rank_state(
        rank_state,
        path=rank_path,
        rank=rank,
        world_size=world_size,
        v5_optimization_step=state["v5_optimization_step"],
        completed_refreshes=state["completed_refreshes"],
    )
    return (
        int(state["v5_optimization_step"]),
        int(state["completed_refreshes"]),
        rank_state["rng_state"],
    )


def prune_v5flow_pretrain_checkpoints(
    checkpoint_root: str | Path,
    *,
    keep: int = 2,
) -> list[Path]:
    """Remove older complete refresh checkpoints while retaining safe fallbacks."""

    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 2:
        raise ValueError("keep must be an integer >= 2")
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    complete: list[tuple[int, Path]] = []
    for child in checkpoint_root.iterdir():
        try:
            marker = _load_v5flow_pretrain_marker(child)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            continue
        complete.append((int(marker["v5_optimization_step"]), child.resolve()))
    complete.sort(key=lambda item: item[0], reverse=True)
    removed = []
    for _, stale in complete[keep:]:
        if stale.parent != checkpoint_root or stale.is_symlink():
            raise ValueError(f"Refusing to prune unsafe v5flow pretrain path: {stale}")
        shutil.rmtree(stale)
        removed.append(stale)
    return removed


def save_policy_state(
    *,
    checkpoint_dir: str | Path,
    model,
    theta_ema,
) -> Path:
    """Save the raw theta/old adapters and theta EMA for an old-center run."""

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / POLICY_STATE_FILENAME
    torch.save(
        {
            "schema_version": 1,
            "adapter_default_raw": _adapter_cpu_state(model, "default"),
            "adapter_old": _adapter_cpu_state(model, "old"),
            "theta_ema": theta_ema.state_dict() if theta_ema is not None else None,
        },
        path,
    )
    return path


def load_policy_state(
    *,
    checkpoint_dir: str | Path,
    model,
    theta_ema,
    map_location: torch.device,
) -> None:
    """Restore the raw theta/old adapters and configured theta EMA."""

    path = Path(checkpoint_dir) / POLICY_STATE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"old-center resume requires policy state: {path}")
    state = torch.load(path, map_location=map_location, weights_only=False)
    expected_keys = {"schema_version", "adapter_default_raw", "adapter_old", "theta_ema"}
    if set(state) != expected_keys or state.get("schema_version") != 1:
        raise ValueError(f"Unsupported policy checkpoint state in {path}")
    _load_adapter_cpu_state(model, "default", state["adapter_default_raw"])
    _load_adapter_cpu_state(model, "old", state["adapter_old"])
    if theta_ema is not None:
        if state["theta_ema"] is None:
            raise ValueError("policy checkpoint is missing the configured theta EMA")
        theta_ema.load_state_dict(state["theta_ema"])


def _cpu_tensor_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tensor_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tensor_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tensor_tree(child) for child in value)
    return value


def move_tensor_tree(value: Any, device: torch.device) -> Any:
    """Move every tensor in a nested checkpoint payload to ``device``."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: move_tensor_tree(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [move_tensor_tree(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(move_tensor_tree(child, device) for child in value)
    return value


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU, and this rank's CUDA RNG streams."""

    cuda_state = torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().cpu(),
        "torch_cuda": cuda_state,
    }


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    """Restore RNG streams captured by :func:`capture_rng_state`."""

    expected_keys = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != expected_keys:
        raise ValueError("exact-resume RNG state has unexpected keys")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        if state["torch_cuda"] is None:
            raise ValueError("CUDA resume requires a saved per-rank CUDA RNG state")
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def requeue_rank_state_filename(rank: int) -> str:
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    return f"rank-{rank:05d}.pt"


def save_requeue_rank_state(
    *,
    checkpoint_dir: str | Path,
    rank: int,
    world_size: int,
    checkpoint_step: int,
    resume_epoch: int,
    resume_phase: str,
    next_sampling_batch: int,
    rng_state: dict[str, Any],
    payload: Any,
) -> Path:
    """Atomically save one rank's exact resume cursor, RNG, and payload on CPU."""

    if resume_phase not in REQUEUE_PHASES:
        raise ValueError(f"Unsupported exact-resume phase: {resume_phase!r}")
    for name, value in {
        "rank": rank,
        "checkpoint_step": checkpoint_step,
        "resume_epoch": resume_epoch,
        "next_sampling_batch": next_sampling_batch,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if rank >= world_size:
        raise ValueError("rank must be smaller than world_size")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / requeue_rank_state_filename(rank)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    state = {
        "schema_version": REQUEUE_RANK_STATE_SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "checkpoint_step": checkpoint_step,
        "resume_epoch": resume_epoch,
        "resume_phase": resume_phase,
        "next_sampling_batch": next_sampling_batch,
        "rng_state": _cpu_tensor_tree(rng_state),
        "payload": _cpu_tensor_tree(payload),
    }
    try:
        with temporary.open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def load_requeue_rank_state(
    *,
    checkpoint_dir: str | Path,
    rank: int,
    world_size: int,
    checkpoint_step: int,
    resume_epoch: int,
    resume_phase: str,
) -> dict[str, Any]:
    """Load and cross-check one rank's exact resume payload."""

    path = Path(checkpoint_dir) / requeue_rank_state_filename(rank)
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-rank exact resume state: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected_keys = {
        "schema_version",
        "rank",
        "world_size",
        "checkpoint_step",
        "resume_epoch",
        "resume_phase",
        "next_sampling_batch",
        "rng_state",
        "payload",
    }
    if set(state) != expected_keys:
        raise ValueError(f"Unexpected per-rank exact resume state keys in {path}")
    expected_values = {
        "schema_version": REQUEUE_RANK_STATE_SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "checkpoint_step": checkpoint_step,
        "resume_epoch": resume_epoch,
        "resume_phase": resume_phase,
    }
    for key, expected in expected_values.items():
        if state[key] != expected:
            raise ValueError(
                f"Per-rank exact resume state {key} mismatch: {state[key]!r} != {expected!r}"
            )
    if not isinstance(state["next_sampling_batch"], int) or state["next_sampling_batch"] < 0:
        raise ValueError("Per-rank next_sampling_batch must be non-negative")
    return state


def _load_completed_rolling_marker(checkpoint_dir: Path) -> dict[str, Any]:
    """Load the commit marker for one managed rolling checkpoint generation."""

    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError(f"Rolling checkpoint must be a real directory: {checkpoint_dir}")
    match = _ROLLING_CHECKPOINT_NAME.fullmatch(checkpoint_dir.name)
    if match is None:
        raise ValueError(f"Invalid rolling checkpoint directory name: {checkpoint_dir.name}")
    checkpoint_step = int(match.group(1))
    marker_path = checkpoint_dir / CHECKPOINT_COMPLETE_MARKER_FILENAME
    if not marker_path.is_file():
        raise FileNotFoundError(f"Missing rolling checkpoint completion marker: {marker_path}")
    with marker_path.open("r", encoding="utf-8") as marker_file:
        marker = json.load(marker_file)
    marker_cfg_mode, marker_guidance_scale = validate_checkpoint_cfg_marker(marker)
    expected_keys = {
        "schema_version",
        "checkpoint_step",
        "v5_optimization_step",
        "resume_epoch",
        "checkpoint_kind",
        "v5flow_enabled",
        "resume_phase",
        "next_sampling_batch",
        "world_size",
        "rank_state_files",
    }
    if set(marker) not in (
        expected_keys,
        expected_keys | CHECKPOINT_CFG_MARKER_KEYS,
    ):
        raise ValueError(f"Unexpected rolling completion marker keys: {marker_path}")
    if marker["schema_version"] != 2 or marker["checkpoint_kind"] != "rolling":
        raise ValueError(f"Unsupported rolling completion marker: {marker_path}")
    if marker["checkpoint_step"] != checkpoint_step:
        raise ValueError(f"Rolling marker step does not match directory: {marker_path}")
    next_sampling_batch = marker["next_sampling_batch"]
    if (
        marker["resume_phase"] != "outer_boundary"
        or not isinstance(next_sampling_batch, int)
        or isinstance(next_sampling_batch, bool)
        or next_sampling_batch != 0
    ):
        raise ValueError(f"Rolling checkpoint is not an outer-boundary state: {marker_path}")
    for field in ("checkpoint_step", "v5_optimization_step", "resume_epoch"):
        value = marker[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Rolling marker {field} must be non-negative: {marker_path}")
    if not isinstance(marker["v5flow_enabled"], bool):
        raise ValueError(f"Rolling marker v5flow_enabled must be boolean: {marker_path}")
    if marker["v5flow_enabled"] and marker_cfg_mode not in {
        "cfg1",
        "cfg4_rollout",
        "cfg4_dual_shifted_center",
        "cfg4_full",
        "cfg4_alternating",
    }:
        raise ValueError(
            "v5flow rolling checkpoints support only cfg1, cfg4_rollout, "
            "cfg4_dual_shifted_center, or cfg4_full/cfg4_alternating; "
            f"got mode={marker_cfg_mode!r}, scale={marker_guidance_scale}: "
            f"{marker_path}"
        )
    world_size = marker["world_size"]
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError(f"Rolling marker world_size must be positive: {marker_path}")
    expected_rank_files = [requeue_rank_state_filename(rank) for rank in range(world_size)]
    if marker["rank_state_files"] != expected_rank_files:
        raise ValueError(f"Rolling marker rank files do not match world_size: {marker_path}")
    required_files = [
        checkpoint_dir / "optimizer.pt",
        checkpoint_dir / "scaler.pt",
        checkpoint_dir / "lora" / "adapter_model.safetensors",
        checkpoint_dir / "lora" / "adapter_config.json",
        checkpoint_dir
        / (V5FLOW_STATE_FILENAME if marker["v5flow_enabled"] else POLICY_STATE_FILENAME),
        *(checkpoint_dir / filename for filename in expected_rank_files),
    ]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required_files):
        raise ValueError(f"Rolling checkpoint has a missing or empty state file: {checkpoint_dir}")
    return marker


def publish_rolling_checkpoint(
    checkpoint_dir: str | Path,
    *,
    keep_generations: int = 2,
) -> Path:
    """Atomically publish one complete generation and prune older complete ones.

    The checkpoint itself remains a directory because a resumable SD3.5 state
    consists of LoRA, two optimizer/scaler families, EMA state, and one RNG file
    per rank. ``latest.json`` is only the atomic pointer/commit index. Keeping two
    complete directories guarantees a fallback if the next generation is killed
    before its completion marker is published.
    """

    if (
        not isinstance(keep_generations, int)
        or isinstance(keep_generations, bool)
        or keep_generations < 2
    ):
        raise ValueError("keep_generations must be an integer >= 2")
    checkpoint_dir = Path(checkpoint_dir).resolve()
    rolling_root = checkpoint_dir.parent
    if rolling_root.name != "rolling":
        raise ValueError(f"Rolling checkpoint parent must be named 'rolling': {rolling_root}")
    current_marker = _load_completed_rolling_marker(checkpoint_dir)

    completed: list[tuple[int, int, Path]] = []
    for child in rolling_root.iterdir():
        try:
            marker = _load_completed_rolling_marker(child)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            continue
        marker_mtime_ns = (child / CHECKPOINT_COMPLETE_MARKER_FILENAME).stat().st_mtime_ns
        completed.append((int(marker["checkpoint_step"]), marker_mtime_ns, child.resolve()))
    completed.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if any(step > int(current_marker["checkpoint_step"]) for step, _, _ in completed):
        raise ValueError("Refusing to move the rolling latest pointer to an older step")

    pointer_path = rolling_root / ROLLING_LATEST_FILENAME
    pointer_payload = {
        "schema_version": 1,
        "checkpoint_dir": checkpoint_dir.name,
        "checkpoint_step": int(current_marker["checkpoint_step"]),
        "resume_epoch": int(current_marker["resume_epoch"]),
        "completion_marker": CHECKPOINT_COMPLETE_MARKER_FILENAME,
    }
    pointer_temporary = pointer_path.with_name(f".{pointer_path.name}.{os.getpid()}.tmp")
    try:
        with pointer_temporary.open("x", encoding="utf-8") as pointer_file:
            json.dump(pointer_payload, pointer_file, indent=2, sort_keys=True)
            pointer_file.write("\n")
            pointer_file.flush()
            os.fsync(pointer_file.fileno())
        os.replace(pointer_temporary, pointer_path)
    finally:
        try:
            pointer_temporary.unlink()
        except FileNotFoundError:
            pass

    for _, _, stale_dir in completed[keep_generations:]:
        if stale_dir.parent != rolling_root or stale_dir.is_symlink():
            raise ValueError(f"Refusing to prune unsafe rolling path: {stale_dir}")
        shutil.rmtree(stale_dir)
    return pointer_path


def save_v5flow_state(
    *,
    checkpoint_dir: str | Path,
    model,
    optimizer_v5: torch.optim.Optimizer,
    v5_scaler,
    v5_ema,
    theta_ema,
    v5_optimization_step: int,
) -> Path:
    """Save raw theta/old/v5 adapters and both EMA/optimizer states."""

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / V5FLOW_STATE_FILENAME
    torch.save(
        {
            "schema_version": 1,
            "adapter_default_raw": _adapter_cpu_state(model, "default"),
            "adapter_old": _adapter_cpu_state(model, "old"),
            "adapter_v5_raw": _adapter_cpu_state(model, V5_ADAPTER_NAME),
            "optimizer_v5": optimizer_v5.state_dict(),
            "v5_scaler": v5_scaler.state_dict() if v5_scaler is not None else None,
            "v5_ema": v5_ema.state_dict() if v5_ema is not None else None,
            "theta_ema": theta_ema.state_dict() if theta_ema is not None else None,
            "v5_optimization_step": int(v5_optimization_step),
        },
        path,
    )
    return path


def load_v5flow_state(
    *,
    checkpoint_dir: str | Path,
    model,
    optimizer_v5: torch.optim.Optimizer,
    v5_scaler,
    v5_ema,
    theta_ema,
    map_location: torch.device,
) -> int:
    """Restore extension state and return the cumulative v5 optimizer step."""

    path = Path(checkpoint_dir) / V5FLOW_STATE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"v5flow resume requires the extension checkpoint state: {path}"
        )
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get("schema_version") != 1:
        raise ValueError(f"Unsupported v5flow checkpoint schema: {state.get('schema_version')!r}")
    resume_adapter_states = (
        state["adapter_default_raw"],
        state["adapter_old"],
        state["adapter_v5_raw"],
    )
    resume_dtypes = {
        tensor.dtype
        for adapter_state in resume_adapter_states
        for tensor in adapter_state.values()
    }
    if len(resume_dtypes) != 1:
        raise ValueError("v5flow checkpoint mixes adapter storage dtypes")
    resume_metadata = {
        "adapter_storage_dtype": (
            "float32" if resume_dtypes == {torch.float32} else "model"
        )
    }
    _validate_declared_adapter_storage_dtype(
        metadata=resume_metadata,
        adapter_states=resume_adapter_states,
        v5_state=state["adapter_v5_raw"],
        optimizer_state=state["optimizer_v5"],
        ema_state=state["v5_ema"],
        artifact="v5flow formal checkpoint",
    )
    _load_adapter_cpu_state(
        model, "default", state["adapter_default_raw"], require_dtype_match=True
    )
    _load_adapter_cpu_state(
        model, "old", state["adapter_old"], require_dtype_match=True
    )
    _load_adapter_cpu_state(
        model,
        V5_ADAPTER_NAME,
        state["adapter_v5_raw"],
        require_dtype_match=True,
    )
    optimizer_v5.load_state_dict(state["optimizer_v5"])
    if v5_scaler is not None:
        if state["v5_scaler"] is None:
            raise ValueError("v5flow checkpoint is missing the configured v5 GradScaler")
        v5_scaler.load_state_dict(state["v5_scaler"])
    if v5_ema is not None:
        if state["v5_ema"] is None:
            raise ValueError("v5flow checkpoint is missing the configured v5 EMA")
        v5_ema.load_state_dict(state["v5_ema"])
    if theta_ema is not None:
        if state["theta_ema"] is None:
            raise ValueError("v5flow checkpoint is missing the configured theta EMA")
        theta_ema.load_state_dict(state["theta_ema"])
    _assert_loaded_float32_training_state(
        model=model,
        optimizer=optimizer_v5,
        ema=v5_ema,
        metadata=resume_metadata,
        artifact="v5flow formal checkpoint",
    )
    return int(state["v5_optimization_step"])
