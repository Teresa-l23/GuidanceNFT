"""Strict SiT-XL/2 DiffusionNFT comparison contracts.

This module keeps the scientific parts of the ordinary-SiT comparison small and
testable: exact JSON parsing, rollout/train record accounting, the historical
three-channel SiT CFG composition, the three policy losses, and the 10-step
Euler--Maruyama sampler used by the Model Guidance comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, NamedTuple

import torch


ASSET_ROOT = Path(
    os.environ.get(
        "GUIDANCENFT_ASSETS",
        str(Path(__file__).resolve().parents[2] / "assets"),
    )
)
os.environ.setdefault("GUIDANCENFT_ASSETS", str(ASSET_ROOT))
OFFICIAL_SIT_XL_SHA256 = (
    "e645b5cd82d1d645cfd9b7c96041b56b2a0aa61faed90c0869e79be02ab55aca"
)
PICKSCORE_REVISION = "a4e4367c6dfa7288a00c550414478f865b875800"
PICKSCORE_PROCESSOR_REVISION = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
VAE_REVISION = "f04b2c4b98319346dad8c65879f680b1997b204a"


@dataclass(frozen=True)
class RolloutConfig:
    sampler: str
    sampling_steps: int
    cfg_scale: float
    groups_global: int
    samples_per_class: int
    batch_size: int
    preview_samples: int


@dataclass(frozen=True)
class RewardConfig:
    name: str
    processor_name: str
    processor_revision: str
    model_name: str
    model_revision: str
    prompt_template: str
    batch_size: int
    dtype: str
    divide_by_26: bool
    advantage_epsilon: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    timestep_fraction: float
    inner_epochs: int
    learning_rate: float
    weight_decay: float
    beta: float
    kl_beta: float
    advantage_clip: float
    max_grad_norm: float
    old_policy_decay_type: int


@dataclass(frozen=True)
class CenterConfig:
    batch_size: int
    timestep_fraction: float
    inner_epochs: int
    learning_rate: float
    weight_decay: float
    ema_decay: float
    snapshot: str
    pretrain_updates: int
    pretrain_timesteps_per_endpoint: int


@dataclass(frozen=True)
class CheckpointConfig:
    save_every: int
    keep: int


@dataclass(frozen=True)
class EvaluationConfig:
    every: int
    at_start: bool
    at_end: bool
    num_classes: int
    samples_per_class: int
    batch_size: int
    preview_samples: int


@dataclass(frozen=True)
class XLCFGNFTConfig:
    schema_version: int
    method: str
    run_name: str
    output_root: str
    init_checkpoint: str
    init_checkpoint_sha256: str
    model_key: str
    image_size: int
    num_classes: int
    vae_name: str
    vae_revision: str
    seed: int
    mixed_precision: str
    expected_gpus: int
    batch_reference_gpus: int
    max_outer_steps: int
    rollout: RolloutConfig
    reward: RewardConfig
    training: TrainingConfig
    center: CenterConfig
    checkpoint: CheckpointConfig
    evaluation: EvaluationConfig

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root).expanduser() / self.run_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ROOT_KEYS = {
    "schema_version",
    "method",
    "run_name",
    "output_root",
    "init_checkpoint",
    "init_checkpoint_sha256",
    "model_key",
    "image_size",
    "num_classes",
    "vae_name",
    "vae_revision",
    "seed",
    "mixed_precision",
    "expected_gpus",
    "batch_reference_gpus",
    "max_outer_steps",
    "rollout",
    "reward",
    "training",
    "center",
    "checkpoint",
    "evaluation",
}
_ROLLOUT_KEYS = {
    "sampler",
    "sampling_steps",
    "cfg_scale",
    "groups_global",
    "samples_per_class",
    "batch_size",
    "preview_samples",
}
_REWARD_KEYS = {
    "name",
    "processor_name",
    "processor_revision",
    "model_name",
    "model_revision",
    "prompt_template",
    "batch_size",
    "dtype",
    "divide_by_26",
    "advantage_epsilon",
}
_TRAINING_KEYS = {
    "batch_size",
    "timestep_fraction",
    "inner_epochs",
    "learning_rate",
    "weight_decay",
    "beta",
    "kl_beta",
    "advantage_clip",
    "max_grad_norm",
    "old_policy_decay_type",
}
_CENTER_KEYS = {
    "batch_size",
    "timestep_fraction",
    "inner_epochs",
    "learning_rate",
    "weight_decay",
    "ema_decay",
    "snapshot",
    "pretrain_updates",
    "pretrain_timesteps_per_endpoint",
}
_EVALUATION_KEYS = {
    "every",
    "at_start",
    "at_end",
    "num_classes",
    "samples_per_class",
    "batch_size",
    "preview_samples",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(
    value: Any, name: str, minimum: float, *, strict: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    invalid = result <= minimum if strict else result < minimum
    if not math.isfinite(result) or invalid:
        operator = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {operator} {minimum}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _revision(value: Any, name: str) -> str:
    result = str(value)
    if re.fullmatch(r"[0-9a-f]{40}", result) is None:
        raise ValueError(f"{name} must be a pinned lowercase 40-character revision")
    return result


def _sha256(value: Any, name: str) -> str:
    result = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return result


def _parse(raw: Mapping[str, Any]) -> XLCFGNFTConfig:
    _exact_keys(raw, _ROOT_KEYS, "root")
    if raw["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    method = str(raw["method"])
    if method not in {"standard", "dual", "shiftcenter"}:
        raise ValueError("method must be standard, dual, or shiftcenter")

    rollout = _mapping(raw["rollout"], "rollout")
    reward = _mapping(raw["reward"], "reward")
    training = _mapping(raw["training"], "training")
    center = _mapping(raw["center"], "center")
    checkpoint = _mapping(raw["checkpoint"], "checkpoint")
    evaluation = _mapping(raw["evaluation"], "evaluation")
    _exact_keys(rollout, _ROLLOUT_KEYS, "rollout")
    _exact_keys(reward, _REWARD_KEYS, "reward")
    _exact_keys(training, _TRAINING_KEYS, "training")
    _exact_keys(center, _CENTER_KEYS, "center")
    _exact_keys(checkpoint, {"save_every", "keep"}, "checkpoint")
    _exact_keys(evaluation, _EVALUATION_KEYS, "evaluation")

    run_name = str(raw["run_name"])
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_name) is None:
        raise ValueError("run_name must be one safe path component of at most 100 chars")
    output_root = Path(os.path.expandvars(str(raw["output_root"]))).expanduser()
    init_checkpoint = Path(os.path.expandvars(str(raw["init_checkpoint"]))).expanduser()
    if not output_root.is_absolute() or not init_checkpoint.is_absolute():
        raise ValueError("output_root and init_checkpoint must be absolute")
    try:
        output_root.resolve(strict=False).relative_to(ASSET_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"output_root must be below {ASSET_ROOT}") from exc

    model_key = str(raw["model_key"])
    image_size = _integer(raw["image_size"], "image_size", 8)
    num_classes = _integer(raw["num_classes"], "num_classes", 2)
    if (model_key, image_size, num_classes) != ("SiT-XL/2", 256, 1000):
        raise ValueError("This comparison is locked to ordinary SiT-XL/2 ImageNet-256")
    if str(raw["vae_name"]) != "stabilityai/sd-vae-ft-ema":
        raise ValueError("vae_name must be stabilityai/sd-vae-ft-ema")
    if _revision(raw["vae_revision"], "vae_revision") != VAE_REVISION:
        raise ValueError("vae_revision differs from the pinned comparison snapshot")

    sampler = str(rollout["sampler"])
    sampling_steps = _integer(rollout["sampling_steps"], "rollout.sampling_steps", 2)
    if (sampler, sampling_steps) != ("sde", 10):
        raise ValueError("The comparison requires the 10-step SDE rollout")
    cfg_scale = _finite_float(rollout["cfg_scale"], "rollout.cfg_scale", 1.0)
    expected_cfg = 1.5 if method == "shiftcenter" else 1.0
    if cfg_scale != expected_cfg:
        raise ValueError(f"{method} requires rollout.cfg_scale={expected_cfg}")

    groups_global = _integer(rollout["groups_global"], "rollout.groups_global", 1)
    samples_per_class = _integer(
        rollout["samples_per_class"], "rollout.samples_per_class", 2
    )
    if (groups_global, samples_per_class) != (48, 24):
        raise ValueError("The formal comparison requires 48 groups x 24 samples")

    timestep_fraction = _finite_float(
        training["timestep_fraction"],
        "training.timestep_fraction",
        0.0,
        strict=True,
    )
    center_fraction = _finite_float(
        center["timestep_fraction"], "center.timestep_fraction", 0.0, strict=True
    )
    if timestep_fraction != 0.99 or center_fraction != 0.99:
        raise ValueError("Formal policy and online-center timestep_fraction must be 0.99")
    if nft_training_timestep_count(sampling_steps, timestep_fraction) != 9:
        raise AssertionError("10 * 0.99 must select exactly nine timestamps")

    pretrain_updates = _integer(
        center["pretrain_updates"], "center.pretrain_updates", 0
    )
    expected_pretrain = 100 if method == "shiftcenter" else 0
    if pretrain_updates != expected_pretrain:
        raise ValueError(f"{method} requires center.pretrain_updates={expected_pretrain}")
    pretrain_times = _integer(
        center["pretrain_timesteps_per_endpoint"],
        "center.pretrain_timesteps_per_endpoint",
        1,
    )
    if pretrain_times != 4:
        raise ValueError("Center pretraining requires four continuous times per endpoint")
    center_snapshot = str(center["snapshot"])
    if center_snapshot != "ema":
        raise ValueError("ShiftCenter comparison uses the EMA center snapshot")

    reward_name = str(reward["name"])
    if reward_name != "pickscore_imagenet":
        raise ValueError("The formal comparison requires pickscore_imagenet")
    if str(reward["prompt_template"]) != "an image of {class_name}":
        raise ValueError("PickScore prompt_template differs from the comparison contract")
    if str(reward["dtype"]) != "fp32" or reward["divide_by_26"] is not True:
        raise ValueError("PickScore must run in fp32 with divide_by_26=true")
    if str(reward["model_name"]) != "yuvalkirstain/PickScore_v1":
        raise ValueError("reward.model_name differs from PickScore_v1")
    if str(reward["processor_name"]) != "laion/CLIP-ViT-H-14-laion2B-s32B-b79K":
        raise ValueError("reward.processor_name differs from the PickScore processor")
    if _revision(reward["model_revision"], "reward.model_revision") != PICKSCORE_REVISION:
        raise ValueError("reward.model_revision differs from the pinned snapshot")
    if (
        _revision(reward["processor_revision"], "reward.processor_revision")
        != PICKSCORE_PROCESSOR_REVISION
    ):
        raise ValueError("reward.processor_revision differs from the pinned snapshot")
    if _sha256(raw["init_checkpoint_sha256"], "init_checkpoint_sha256") != OFFICIAL_SIT_XL_SHA256:
        raise ValueError("init_checkpoint_sha256 differs from the official SiT-XL/2 asset")

    mixed_precision = str(raw["mixed_precision"])
    if mixed_precision not in {"no", "bf16"}:
        raise ValueError("mixed_precision must be no or bf16")
    expected_gpus = _integer(raw["expected_gpus"], "expected_gpus", 1)
    reference_gpus = _integer(raw["batch_reference_gpus"], "batch_reference_gpus", 1)
    if expected_gpus not in {1, 2} or reference_gpus != 2:
        raise ValueError(
            "Formal continuation configs require expected_gpus in {1, 2} and "
            "batch_reference_gpus=2"
        )
    max_outer_steps = _integer(raw["max_outer_steps"], "max_outer_steps", 1)

    evaluation_every = _integer(evaluation["every"], "evaluation.every", 1)
    evaluation_at_start = _boolean(
        evaluation["at_start"], "evaluation.at_start"
    )
    evaluation_at_end = _boolean(evaluation["at_end"], "evaluation.at_end")
    evaluation_num_classes = _integer(
        evaluation["num_classes"], "evaluation.num_classes", 1
    )
    evaluation_samples_per_class = _integer(
        evaluation["samples_per_class"], "evaluation.samples_per_class", 1
    )
    evaluation_batch_size = _integer(
        evaluation["batch_size"], "evaluation.batch_size", 1
    )
    evaluation_preview_samples = _integer(
        evaluation["preview_samples"], "evaluation.preview_samples", 0
    )
    if max_outer_steps == 400:
        if evaluation_every != 50:
            raise ValueError("The u400 continuation evaluates every 50 outer steps")
    elif evaluation_every != max_outer_steps:
        raise ValueError("A non-u400 comparison evaluates only its final outer step")
    if evaluation_at_start or not evaluation_at_end:
        raise ValueError("Evaluation must use at_start=false and at_end=true")
    expected_eval_classes = 1000 if max_outer_steps == 400 else 48
    if (evaluation_num_classes, evaluation_samples_per_class) != (
        expected_eval_classes,
        2,
    ):
        raise ValueError(
            f"Evaluation requires {expected_eval_classes} fixed classes x 2 samples"
        )
    if evaluation_batch_size != 8 or evaluation_preview_samples != 8:
        raise ValueError("Final evaluation requires batch_size=8 and preview_samples=8")

    decay_type = _integer(
        training["old_policy_decay_type"], "training.old_policy_decay_type", 0
    )
    if decay_type not in {0, 1, 2}:
        raise ValueError("training.old_policy_decay_type must be 0, 1, or 2")
    ema_decay = _finite_float(center["ema_decay"], "center.ema_decay", 0.0)
    if ema_decay >= 1.0:
        raise ValueError("center.ema_decay must be less than 1")

    return XLCFGNFTConfig(
        schema_version=1,
        method=method,
        run_name=run_name,
        output_root=str(output_root),
        init_checkpoint=str(init_checkpoint),
        init_checkpoint_sha256=_sha256(
            raw["init_checkpoint_sha256"], "init_checkpoint_sha256"
        ),
        model_key=model_key,
        image_size=image_size,
        num_classes=num_classes,
        vae_name=str(raw["vae_name"]),
        vae_revision=VAE_REVISION,
        seed=_integer(raw["seed"], "seed", 0),
        mixed_precision=mixed_precision,
        expected_gpus=expected_gpus,
        batch_reference_gpus=reference_gpus,
        max_outer_steps=max_outer_steps,
        rollout=RolloutConfig(
            sampler=sampler,
            sampling_steps=sampling_steps,
            cfg_scale=cfg_scale,
            groups_global=groups_global,
            samples_per_class=samples_per_class,
            batch_size=_integer(rollout["batch_size"], "rollout.batch_size", 1),
            preview_samples=_integer(
                rollout["preview_samples"], "rollout.preview_samples", 1
            ),
        ),
        reward=RewardConfig(
            name=reward_name,
            processor_name=str(reward["processor_name"]),
            processor_revision=PICKSCORE_PROCESSOR_REVISION,
            model_name=str(reward["model_name"]),
            model_revision=PICKSCORE_REVISION,
            prompt_template=str(reward["prompt_template"]),
            batch_size=_integer(reward["batch_size"], "reward.batch_size", 1),
            dtype=str(reward["dtype"]),
            divide_by_26=True,
            advantage_epsilon=_finite_float(
                reward["advantage_epsilon"],
                "reward.advantage_epsilon",
                0.0,
                strict=True,
            ),
        ),
        training=TrainingConfig(
            batch_size=_integer(training["batch_size"], "training.batch_size", 1),
            timestep_fraction=timestep_fraction,
            inner_epochs=_integer(training["inner_epochs"], "training.inner_epochs", 1),
            learning_rate=_finite_float(
                training["learning_rate"], "training.learning_rate", 0.0, strict=True
            ),
            weight_decay=_finite_float(
                training["weight_decay"], "training.weight_decay", 0.0
            ),
            beta=_finite_float(training["beta"], "training.beta", 0.0, strict=True),
            kl_beta=_finite_float(training["kl_beta"], "training.kl_beta", 0.0),
            advantage_clip=_finite_float(
                training["advantage_clip"],
                "training.advantage_clip",
                0.0,
                strict=True,
            ),
            max_grad_norm=_finite_float(
                training["max_grad_norm"],
                "training.max_grad_norm",
                0.0,
                strict=True,
            ),
            old_policy_decay_type=decay_type,
        ),
        center=CenterConfig(
            batch_size=_integer(center["batch_size"], "center.batch_size", 1),
            timestep_fraction=center_fraction,
            inner_epochs=_integer(center["inner_epochs"], "center.inner_epochs", 1),
            learning_rate=_finite_float(
                center["learning_rate"], "center.learning_rate", 0.0, strict=True
            ),
            weight_decay=_finite_float(
                center["weight_decay"], "center.weight_decay", 0.0
            ),
            ema_decay=ema_decay,
            snapshot=center_snapshot,
            pretrain_updates=pretrain_updates,
            pretrain_timesteps_per_endpoint=pretrain_times,
        ),
        checkpoint=CheckpointConfig(
            save_every=_integer(checkpoint["save_every"], "checkpoint.save_every", 1),
            keep=_integer(checkpoint["keep"], "checkpoint.keep", 1),
        ),
        evaluation=EvaluationConfig(
            every=evaluation_every,
            at_start=evaluation_at_start,
            at_end=evaluation_at_end,
            num_classes=evaluation_num_classes,
            samples_per_class=evaluation_samples_per_class,
            batch_size=evaluation_batch_size,
            preview_samples=evaluation_preview_samples,
        ),
    )


def load_xl_cfg_nft_config(path: str | Path) -> XLCFGNFTConfig:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("Experiment config root must be a JSON object")
    return _parse(raw)


def validate_world_size_geometry(
    config: XLCFGNFTConfig, *, world_size: int
) -> dict[str, int]:
    if world_size != config.expected_gpus:
        raise ValueError(
            f"Runtime world_size={world_size} differs from expected_gpus={config.expected_gpus}"
        )
    if config.rollout.groups_global % world_size:
        raise ValueError("rollout.groups_global must divide evenly across ranks")
    endpoints_global = config.rollout.groups_global * config.rollout.samples_per_class
    endpoints_local = endpoints_global // world_size
    for name, value in {
        "rollout.batch_size": config.rollout.batch_size,
        "training.batch_size": config.training.batch_size,
        "center.batch_size": config.center.batch_size,
    }.items():
        if endpoints_local % value:
            raise ValueError(f"{name} must divide the {endpoints_local} local endpoints")
    train_times = nft_training_timestep_count(
        config.rollout.sampling_steps, config.training.timestep_fraction
    )
    center_times = nft_training_timestep_count(
        config.rollout.sampling_steps, config.center.timestep_fraction
    )
    return {
        "world_size": world_size,
        "endpoints_global": endpoints_global,
        "endpoints_local": endpoints_local,
        "class_groups_local": config.rollout.groups_global // world_size,
        "policy_timesteps_per_endpoint": train_times,
        "policy_records_global": (
            endpoints_global * config.training.inner_epochs * train_times
        ),
        "policy_microbatches_per_rank": (
            config.training.inner_epochs
            * train_times
            * (endpoints_local // config.training.batch_size)
        ),
        "online_center_records_global": (
            endpoints_global * config.center.inner_epochs * center_times
            if config.method == "shiftcenter"
            else 0
        ),
        "center_pretrain_updates": config.center.pretrain_updates,
        "center_pretrain_records_per_update_global": (
            endpoints_global * config.center.pretrain_timesteps_per_endpoint
            if config.method == "shiftcenter"
            else 0
        ),
    }


class FlowMatchingBatch(NamedTuple):
    state: torch.Tensor
    target_velocity: torch.Tensor
    noise: torch.Tensor


def group_normalize_rewards(rewards: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    """Normalize each ``[class, samples]`` row independently."""

    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [groups, samples_per_group>=2]")
    if not bool(torch.isfinite(rewards).all().item()):
        raise ValueError("rewards must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    return centered / (rewards.std(dim=1, unbiased=False, keepdim=True) + epsilon)


def nft_training_timestep_count(sampling_steps: int, fraction: float) -> int:
    if sampling_steps < 2:
        raise ValueError("sampling_steps must be at least two")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("timestep fraction must lie in (0, 1]")
    count = int(sampling_steps * fraction)
    if count < 1:
        raise ValueError("timestep fraction selects no time")
    return count


def sit_sde_training_timesteps(sampling_steps: int) -> torch.Tensor:
    """Return native SiT times corresponding to the MG SDE model-input grid."""

    if sampling_steps < 2:
        raise ValueError("sampling_steps must be at least two")
    mg_times = torch.linspace(1.0, 0.04, sampling_steps, dtype=torch.float32)
    return 1.0 - mg_times


def sample_nft_timesteps(
    time_grid: torch.Tensor,
    *,
    num_examples: int,
    inner_epochs: int,
    timestep_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Independently shuffle the rollout grid per endpoint and keep its prefix."""

    if time_grid.ndim != 1 or time_grid.numel() < 2:
        raise ValueError("time_grid must be one-dimensional with at least two values")
    if num_examples < 1 or inner_epochs < 1:
        raise ValueError("num_examples and inner_epochs must be positive")
    count = nft_training_timestep_count(time_grid.numel(), timestep_fraction)
    keys = torch.rand(
        inner_epochs,
        num_examples,
        time_grid.numel(),
        generator=generator,
        device=time_grid.device,
    )
    indices = keys.argsort(dim=-1)[..., :count]
    expanded = time_grid.reshape(1, 1, -1).expand(inner_epochs, num_examples, -1)
    return torch.gather(expanded, -1, indices)


def make_sit_flow_matching_batch(
    endpoints: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> FlowMatchingBatch:
    """Create native SiT ``x_t=(1-t)z+t*x`` records and ``x-z`` targets."""

    if endpoints.ndim < 2 or endpoints.shape[0] == 0:
        raise ValueError("endpoints must be a non-empty batch")
    if t.shape != (endpoints.shape[0],):
        raise ValueError(f"t must have shape [{endpoints.shape[0]}]")
    if not bool(((t >= 0.0) & (t <= 1.0)).all().item()):
        raise ValueError("native SiT times must lie in [0, 1]")
    if noise is None:
        noise = torch.randn_like(endpoints)
    if noise.shape != endpoints.shape:
        raise ValueError("noise must match endpoints")
    expanded_t = t.view(-1, *([1] * (endpoints.ndim - 1))).to(endpoints)
    noise = noise.to(endpoints)
    return FlowMatchingBatch(
        state=(1.0 - expanded_t) * noise + expanded_t * endpoints,
        target_velocity=endpoints - noise,
        noise=noise,
    )


def compose_sit_cfg_velocity(
    conditional: torch.Tensor, unconditional: torch.Tensor, cfg_scale: float
) -> torch.Tensor:
    """Apply historical SiT CFG to channels 0:3 and retain conditional channel 3+."""

    if conditional.shape != unconditional.shape or conditional.ndim < 2:
        raise ValueError("conditional and unconditional velocities must match")
    if conditional.shape[1] < 3:
        raise ValueError("historical SiT CFG requires at least three channels")
    if not math.isfinite(cfg_scale) or cfg_scale < 1.0:
        raise ValueError("cfg_scale must be finite and at least one")
    guided = unconditional[:, :3] + cfg_scale * (
        conditional[:, :3] - unconditional[:, :3]
    )
    return torch.cat([guided, conditional[:, 3:]], dim=1)


def evaluate_sit_branches(
    model,
    state: torch.Tensor,
    t: torch.Tensor,
    labels: torch.LongTensor,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate conditional and null branches together at an identical state/time."""

    expected_batch_shape = (state.shape[0],)
    if (
        state.shape[0] == 0
        or t.shape != expected_batch_shape
        or labels.shape != expected_batch_shape
    ):
        raise ValueError("state, t, and labels batch shapes differ")
    if labels.dtype != torch.long:
        raise ValueError("labels must have dtype torch.long")
    paired = model(
        torch.cat([state, state]),
        torch.cat([t, t]),
        torch.cat([labels, torch.full_like(labels, num_classes)]),
    )
    if paired.shape[0] != 2 * state.shape[0] or paired.shape[1:] != state.shape[1:]:
        raise ValueError("SiT branch model returned an incompatible shape")
    return paired.chunk(2, dim=0)


def _mse_per_example(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must match")
    return (prediction.float() - target.float()).square().flatten(1).mean(dim=1)


def _base_nft_loss(
    *,
    current: torch.Tensor,
    old: torch.Tensor,
    reference: torch.Tensor,
    target: torch.Tensor,
    advantages: torch.Tensor,
    beta: float,
    advantage_clip: float,
    kl_beta: float,
    center: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if current.shape != old.shape or current.shape != reference.shape:
        raise ValueError("current, old, and reference predictions must match")
    if target.shape != current.shape or advantages.shape != (current.shape[0],):
        raise ValueError("target or advantage shape differs from predictions")
    if beta <= 0 or advantage_clip <= 0 or kl_beta < 0:
        raise ValueError("invalid beta, advantage_clip, or kl_beta")
    selected_center = old if center is None else center
    if selected_center.shape != current.shape:
        raise ValueError("center prediction must match current prediction")
    displacement = current - old
    positive = selected_center + beta * displacement
    negative = selected_center - beta * displacement
    positive_loss = _mse_per_example(positive, target)
    negative_loss = _mse_per_example(negative, target)
    clipped = advantages.clamp(-advantage_clip, advantage_clip)
    reward_weight = ((clipped / advantage_clip) / 2.0 + 0.5).clamp(0.0, 1.0)
    unscaled = (reward_weight * positive_loss + (1.0 - reward_weight) * negative_loss) / beta
    policy = advantage_clip * unscaled.mean()
    kl = _mse_per_example(current, reference).mean()
    total = policy + kl_beta * kl
    return total, {
        "total_loss": total.detach(),
        "policy_loss": policy.detach(),
        "kl_loss": kl.detach(),
        "positive_loss": positive_loss.mean().detach(),
        "negative_loss": negative_loss.mean().detach(),
        "reward_weight_mean": reward_weight.mean().detach(),
        "old_deviation": _mse_per_example(current, old).mean().detach(),
        "center_deviation": _mse_per_example(selected_center, old).mean().detach(),
    }


def cfg_variant_nft_loss(
    *,
    method: str,
    current_conditional: torch.Tensor,
    old_conditional: torch.Tensor,
    reference_conditional: torch.Tensor,
    target_velocity: torch.Tensor,
    advantages: torch.Tensor,
    beta: float,
    advantage_clip: float,
    kl_beta: float,
    cfg_scale: float,
    current_unconditional: torch.Tensor | None = None,
    old_unconditional: torch.Tensor | None = None,
    reference_unconditional: torch.Tensor | None = None,
    center_prediction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Standard, Dual, or ShiftCenter with explicit gradient routing.

    Standard uses conditional current/old/reference predictions at CFG 1.
    Dual concatenates independent conditional and null objectives with equal
    weight. ShiftCenter uses only the conditional displacement and substitutes
    the fitted center for the old branch center. All variants use velocity MSE,
    matching the local Model Guidance objective.
    """

    branches = (current_unconditional, old_unconditional, reference_unconditional)
    if method not in {"standard", "dual", "shiftcenter"}:
        raise ValueError(f"Unsupported method: {method}")
    expected_cfg = 1.5 if method == "shiftcenter" else 1.0
    if cfg_scale != expected_cfg:
        raise ValueError(f"{method} requires cfg_scale={expected_cfg}")
    if method != "shiftcenter" and center_prediction is not None:
        raise ValueError(f"{method} uses its old-policy center")
    if method in {"standard", "shiftcenter"}:
        if method == "shiftcenter" and center_prediction is None:
            raise ValueError("shiftcenter requires center_prediction")
        if any(value is not None for value in branches):
            raise ValueError(f"{method} updates only the conditional branch")
        return _base_nft_loss(
            current=current_conditional,
            old=old_conditional.detach(),
            reference=reference_conditional.detach(),
            target=target_velocity,
            advantages=advantages,
            beta=beta,
            advantage_clip=advantage_clip,
            kl_beta=kl_beta,
            center=center_prediction.detach() if center_prediction is not None else None,
        )
    if method == "dual":
        if any(value is None for value in branches):
            raise ValueError("dual requires all unconditional predictions")
        return _base_nft_loss(
            current=torch.cat([current_conditional, current_unconditional]),
            old=torch.cat([old_conditional.detach(), old_unconditional.detach()]),
            reference=torch.cat(
                [reference_conditional.detach(), reference_unconditional.detach()]
            ),
            target=torch.cat([target_velocity, target_velocity]),
            advantages=torch.cat([advantages, advantages]),
            beta=beta,
            advantage_clip=advantage_clip,
            kl_beta=kl_beta,
            center=None,
        )
    raise ValueError(f"Unsupported method: {method}")


def old_policy_decay(step: int, decay_type: int) -> float:
    if step < 0:
        raise ValueError("step must be non-negative")
    schedules = {0: (0, 0.0, 0.0), 1: (0, 0.001, 0.5), 2: (75, 0.0075, 0.999)}
    if decay_type not in schedules:
        raise ValueError("decay_type must be 0, 1, or 2")
    flat, rate, maximum = schedules[decay_type]
    return 0.0 if step < flat else min((step - flat) * rate, maximum)


def _score_from_mg_velocity(
    velocity: torch.Tensor, state: torch.Tensor, mg_t: torch.Tensor
) -> torch.Tensor:
    expanded = mg_t.view(-1, *([1] * (state.ndim - 1)))
    reverse_alpha_ratio = -(1.0 - expanded)
    variance = expanded
    return (reverse_alpha_ratio * velocity - state) / variance


@torch.no_grad()
def euler_maruyama_sit_sampler(
    model,
    latents: torch.Tensor,
    labels: torch.LongTensor,
    *,
    num_steps: int,
    cfg_scale: float,
    num_classes: int,
    generator: torch.Generator,
    autocast: Callable[[], Any],
) -> torch.Tensor:
    """Run the Model Guidance 10-step EM protocol through the ordinary-SiT clock."""

    if latents.shape[0] == 0 or labels.shape != (latents.shape[0],):
        raise ValueError("latents and labels batch shapes differ")
    mg_grid = torch.linspace(1.0, 0.04, num_steps, dtype=torch.float64)
    mg_grid = torch.cat([mg_grid, torch.zeros(1, dtype=torch.float64)])
    state = latents.to(torch.float64)
    model_dtype = latents.dtype

    def drift(current: torch.Tensor, mg_scalar: torch.Tensor) -> torch.Tensor:
        mg_time = torch.full(
            (current.shape[0],),
            float(mg_scalar),
            device=current.device,
            dtype=model_dtype,
        )
        sit_time = 1.0 - mg_time
        with autocast():
            model_input = current.to(model_dtype)
            if cfg_scale == 1.0:
                sit_velocity = model(model_input, sit_time, labels)
                if sit_velocity.shape != model_input.shape:
                    raise ValueError("SiT model returned an incompatible shape")
            else:
                conditional, unconditional = evaluate_sit_branches(
                    model,
                    model_input,
                    sit_time,
                    labels,
                    num_classes=num_classes,
                )
                sit_velocity = compose_sit_cfg_velocity(
                    conditional, unconditional, cfg_scale
                )
        mg_velocity = -sit_velocity.to(torch.float64)
        score = _score_from_mg_velocity(
            mg_velocity,
            current,
            mg_time.to(torch.float64),
        )
        diffusion = 2.0 * float(mg_scalar)
        return mg_velocity - 0.5 * diffusion * score

    for current_t, next_t in zip(mg_grid[:-2], mg_grid[1:-1]):
        dt = float(next_t - current_t)
        current = state
        current_drift = drift(current, current_t)
        epsilon = torch.randn(
            current.shape,
            generator=generator,
            device=current.device,
            dtype=torch.float64,
        )
        state = (
            current
            + current_drift * dt
            + math.sqrt(2.0 * float(current_t)) * epsilon * math.sqrt(abs(dt))
        )

    current_t, next_t = mg_grid[-2], mg_grid[-1]
    state = state + drift(state, current_t) * float(next_t - current_t)
    return state.to(torch.float32)
