from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault(
    "GUIDANCENFT_ASSETS", str(Path(__file__).resolve().parents[2] / "assets")
)


@dataclass(frozen=True)
class RolloutConfig:
    sampler: str
    sampling_steps: int
    groups_global: int
    samples_per_class: int
    batch_size: int
    preview_samples: int


@dataclass(frozen=True)
class RewardConfig:
    name: str
    processor_name: str
    model_name: str
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


@dataclass(frozen=True)
class CheckpointConfig:
    save_every: int
    keep: int


@dataclass(frozen=True)
class NFTConfig:
    schema_version: int
    method: str
    run_name: str
    output_root: str
    init_checkpoint: str
    model_key: str
    image_size: int
    num_classes: int
    vae: str
    seed: int
    mixed_precision: str
    max_outer_steps: int
    rollout: RolloutConfig
    reward: RewardConfig
    training: TrainingConfig
    center: CenterConfig
    checkpoint: CheckpointConfig

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
    "model_key",
    "image_size",
    "num_classes",
    "vae",
    "seed",
    "mixed_precision",
    "max_outer_steps",
    "rollout",
    "reward",
    "training",
    "center",
    "checkpoint",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys differ: missing={missing}, extra={extra}")


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, name: str, minimum: float, *, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= minimum if strict else result < minimum):
        operator = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {operator} {minimum}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _parse(raw: Mapping[str, Any]) -> NFTConfig:
    _exact_keys(raw, _ROOT_KEYS, "root")
    if raw["schema_version"] != 2:
        raise ValueError("schema_version must be 2")

    method = str(raw["method"])
    if method == "standard":
        method = "original"
    if method not in {"original", "shiftcenter"}:
        raise ValueError("method must be 'original' or 'shiftcenter'")

    rollout = _mapping(raw["rollout"], "rollout")
    _exact_keys(
        rollout,
        {
            "sampler",
            "sampling_steps",
            "groups_global",
            "samples_per_class",
            "batch_size",
            "preview_samples",
        },
        "rollout",
    )
    sampler = str(rollout["sampler"])
    if sampler not in {"sde", "ode"}:
        raise ValueError("rollout.sampler must be 'sde' or 'ode'")

    reward = _mapping(raw["reward"], "reward")
    _exact_keys(
        reward,
        {
            "name",
            "processor_name",
            "model_name",
            "prompt_template",
            "batch_size",
            "dtype",
            "divide_by_26",
            "advantage_epsilon",
        },
        "reward",
    )
    reward_name = str(reward["name"])
    if reward_name not in {"pickscore_imagenet", "constant"}:
        raise ValueError("reward.name must be 'pickscore_imagenet' or 'constant'")
    reward_dtype = str(reward["dtype"])
    if reward_dtype not in {"fp32", "bf16"}:
        raise ValueError("reward.dtype must be 'fp32' or 'bf16'")

    training = _mapping(raw["training"], "training")
    _exact_keys(
        training,
        {
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
        },
        "training",
    )
    decay_type = _integer(training["old_policy_decay_type"], "training.old_policy_decay_type", 0)
    if decay_type not in {0, 1, 2}:
        raise ValueError("training.old_policy_decay_type must be 0, 1, or 2")

    center = _mapping(raw["center"], "center")
    _exact_keys(
        center,
        {
            "batch_size",
            "timestep_fraction",
            "inner_epochs",
            "learning_rate",
            "weight_decay",
            "ema_decay",
            "snapshot",
        },
        "center",
    )
    center_snapshot = str(center["snapshot"])
    if center_snapshot not in {"raw", "ema"}:
        raise ValueError("center.snapshot must be 'raw' or 'ema'")

    checkpoint = _mapping(raw["checkpoint"], "checkpoint")
    _exact_keys(checkpoint, {"save_every", "keep"}, "checkpoint")

    run_name = str(raw["run_name"])
    if not run_name or "/" in run_name or run_name in {".", ".."} or ".." in run_name:
        raise ValueError("run_name must be a safe single path component")
    image_size = _integer(raw["image_size"], "image_size", 8)
    if image_size not in {256, 512}:
        raise ValueError("image_size must be 256 or 512")
    mixed_precision = str(raw["mixed_precision"])
    if mixed_precision not in {"no", "bf16"}:
        raise ValueError("mixed_precision must be 'no' or 'bf16'")

    ema_decay = _finite_float(center["ema_decay"], "center.ema_decay", 0.0)
    if ema_decay >= 1.0:
        raise ValueError("center.ema_decay must be less than 1")

    groups_global = _integer(rollout["groups_global"], "rollout.groups_global", 1)
    if groups_global > _integer(raw["num_classes"], "num_classes", 2):
        raise ValueError("rollout.groups_global cannot exceed num_classes")
    timestep_fraction = _finite_float(
        training["timestep_fraction"], "training.timestep_fraction", 0.0, strict=True
    )
    center_timestep_fraction = _finite_float(
        center["timestep_fraction"], "center.timestep_fraction", 0.0, strict=True
    )
    if timestep_fraction > 1.0 or center_timestep_fraction > 1.0:
        raise ValueError("timestep fractions must be at most 1")
    sampling_steps = _integer(rollout["sampling_steps"], "rollout.sampling_steps", 2)
    if int(sampling_steps * timestep_fraction) < 1:
        raise ValueError("training.timestep_fraction selects no sampling timesteps")
    if int(sampling_steps * center_timestep_fraction) < 1:
        raise ValueError("center.timestep_fraction selects no sampling timesteps")

    return NFTConfig(
        schema_version=2,
        method=method,
        run_name=run_name,
        output_root=os.path.expandvars(str(raw["output_root"])),
        init_checkpoint=os.path.expandvars(str(raw["init_checkpoint"])),
        model_key=str(raw["model_key"]),
        image_size=image_size,
        num_classes=_integer(raw["num_classes"], "num_classes", 2),
        vae=str(raw["vae"]),
        seed=_integer(raw["seed"], "seed", 0),
        mixed_precision=mixed_precision,
        max_outer_steps=_integer(raw["max_outer_steps"], "max_outer_steps", 1),
        rollout=RolloutConfig(
            sampler=sampler,
            sampling_steps=sampling_steps,
            groups_global=groups_global,
            samples_per_class=_integer(
                rollout["samples_per_class"], "rollout.samples_per_class", 2
            ),
            batch_size=_integer(rollout["batch_size"], "rollout.batch_size", 1),
            preview_samples=_integer(
                rollout["preview_samples"], "rollout.preview_samples", 1
            ),
        ),
        reward=RewardConfig(
            name=reward_name,
            processor_name=str(reward["processor_name"]),
            model_name=str(reward["model_name"]),
            prompt_template=str(reward["prompt_template"]),
            batch_size=_integer(reward["batch_size"], "reward.batch_size", 1),
            dtype=reward_dtype,
            divide_by_26=_boolean(reward["divide_by_26"], "reward.divide_by_26"),
            advantage_epsilon=_finite_float(
                reward["advantage_epsilon"], "reward.advantage_epsilon", 0.0, strict=True
            ),
        ),
        training=TrainingConfig(
            batch_size=_integer(training["batch_size"], "training.batch_size", 1),
            timestep_fraction=timestep_fraction,
            inner_epochs=_integer(training["inner_epochs"], "training.inner_epochs", 1),
            learning_rate=_finite_float(
                training["learning_rate"], "training.learning_rate", 0.0, strict=True
            ),
            weight_decay=_finite_float(training["weight_decay"], "training.weight_decay", 0.0),
            beta=_finite_float(training["beta"], "training.beta", 0.0, strict=True),
            kl_beta=_finite_float(training["kl_beta"], "training.kl_beta", 0.0),
            advantage_clip=_finite_float(
                training["advantage_clip"], "training.advantage_clip", 0.0, strict=True
            ),
            max_grad_norm=_finite_float(
                training["max_grad_norm"], "training.max_grad_norm", 0.0, strict=True
            ),
            old_policy_decay_type=decay_type,
        ),
        center=CenterConfig(
            batch_size=_integer(center["batch_size"], "center.batch_size", 1),
            timestep_fraction=center_timestep_fraction,
            inner_epochs=_integer(center["inner_epochs"], "center.inner_epochs", 1),
            learning_rate=_finite_float(
                center["learning_rate"], "center.learning_rate", 0.0, strict=True
            ),
            weight_decay=_finite_float(center["weight_decay"], "center.weight_decay", 0.0),
            ema_decay=ema_decay,
            snapshot=center_snapshot,
        ),
        checkpoint=CheckpointConfig(
            save_every=_integer(checkpoint["save_every"], "checkpoint.save_every", 0),
            keep=_integer(checkpoint["keep"], "checkpoint.keep", 1),
        ),
    )


def _set_override(raw: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = raw
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise KeyError(f"Cannot override unknown mapping {'.'.join(parts[:-1])}")
        target = child
    if parts[-1] not in target:
        raise KeyError(f"Cannot override unknown key {dotted_key}")
    target[parts[-1]] = value


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> NFTConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("The experiment config root must be a JSON object")
    for key, value in (overrides or {}).items():
        _set_override(raw, key, value)
    return _parse(raw)


def validate_world_size_geometry(
    config: NFTConfig, *, num_gpus: int, reference_gpus: int
) -> dict[str, int]:
    """Validate a GPU override while keeping the global rollout fixed."""

    if num_gpus <= 0 or reference_gpus <= 0:
        raise ValueError("GPU counts must be positive")
    if reference_gpus != 2:
        raise ValueError(
            "Model Guidance uses the measured two-GPU batch reference"
        )
    if config.rollout.groups_global % num_gpus:
        raise ValueError(
            "rollout.groups_global must divide evenly across requested GPUs"
        )

    measured_microbatch = {"original": 288, "shiftcenter": 252}[config.method]
    if config.training.batch_size != measured_microbatch:
        raise ValueError(
            "training microbatch differs from the measured H200 value"
        )
    if config.center.batch_size != measured_microbatch:
        raise ValueError(
            "center microbatch differs from the measured H200 value"
        )

    global_endpoints = (
        config.rollout.groups_global * config.rollout.samples_per_class
    )
    local_endpoints = global_endpoints // num_gpus
    training_times = int(
        config.rollout.sampling_steps * config.training.timestep_fraction
    )
    center_times = int(
        config.rollout.sampling_steps * config.center.timestep_fraction
    )
    return {
        "num_gpus": num_gpus,
        "reference_gpus": reference_gpus,
        "global_endpoints": global_endpoints,
        "local_endpoints": local_endpoints,
        "training_microbatch_per_gpu": config.training.batch_size,
        "training_backward_microbatches_per_outer": (
            config.training.inner_epochs
            * training_times
            * math.ceil(local_endpoints / config.training.batch_size)
        ),
        "center_microbatch_per_gpu": config.center.batch_size,
        "center_backward_microbatches_per_outer": (
            config.center.inner_epochs
            * center_times
            * math.ceil(local_endpoints / config.center.batch_size)
        ),
    }
