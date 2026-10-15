"""Strict configuration loader for the independent ELF NFT trainer."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from configs.config import Config, load_config_from_yaml

from .reward import (
    ADAPTIVE_ENTROPY_FLOOR,
    FIXED_JOINT,
    OWT_REWARD_MODES,
    PPL_ONLY,
)


ELF_NFT_ASSET_ROOT = Path(
    os.environ.get(
        "GUIDANCENFT_ASSETS",
        str(Path(__file__).resolve().parents[3] / "assets"),
    )
).expanduser()
os.environ.setdefault("GUIDANCENFT_ASSETS", str(ELF_NFT_ASSET_ROOT))
# Root of the patched ELF checkout created by setup_upstream.sh.
REPO_ROOT = Path(
    os.environ.get(
        "ELF_REPO_ROOT",
        str(Path(__file__).resolve().parents[2] / "third_party" / "ELF"),
    )
).resolve()
DEFAULT_CHECKPOINT_KEEP_EVERY = 25


@dataclass(frozen=True)
class SCCFGEvaluationCase:
    """Separate the network's CFG input from two-pass external guidance."""

    name: str
    model_cfg_scale: float
    external_cfg_scale: float


@dataclass(frozen=True)
class NFTConfig:
    """Fully validated NFT-only settings.

    ELF architecture/data settings continue to come from the unmodified base
    training YAML named by ``base_config``.
    """

    config_path: str
    config_schema_version: int
    task: str
    base_config: str
    pretrained_checkpoint: str
    pretrained_checkpoint_revision: str | None
    parameterization: str
    method: str
    output_dir: str
    seed: int
    rollout_sampling_method: str
    rollout_steps: int
    rollout_time_schedule: str
    rollout_sde_gamma: float
    rollout_cfg_scale: float
    self_cond_cfg_scale: float
    rollout_self_cond_cfg_scale: float
    rollout_field_mode: str
    reward_type: str
    hypothesis_max_tokens: int
    reward_model_name: str
    reward_batch_size: int
    reward_max_length: int
    empty_text_reward: float
    reward_combination_mode: str
    entropy_target: float
    initial_entropy_weight: float
    max_entropy_weight: float
    entropy_ema_decay: float
    entropy_dual_learning_rate: float
    prompt_groups_per_outer: int
    samples_per_prompt: int
    max_outer_steps: int
    rollout_generation_batch_size: int
    train_batch_size: int
    num_inner_epochs: int
    learning_rate: float
    weight_decay: float
    theta_ema_decay: float
    beta: float
    kl_beta: float
    decoder_ce_weight: float
    adv_clip_max: float
    max_grad_norm: float
    old_policy_decay_type: int
    self_cond_update_probability: float
    training_self_cond_cfg_distribution: str
    training_self_cond_cfg_min: float
    training_self_cond_cfg_max: float
    training_velocity_reconstruction: bool
    unconditional_update_every: int
    timestep_schedule: str
    timesteps_per_batch: int
    center_learning_rate: float
    center_num_inner_epochs: int
    center_ema_decay: float
    center_snapshot: str
    center_pretrain_steps: int
    center_self_cond_cfg_scale: float
    save_every: int
    keep_checkpoints: int
    keep_every_n_steps: int
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str
    formal_experiment: bool
    self_condition_mode: str = "bernoulli"
    evaluation_cases: tuple[SCCFGEvaluationCase, ...] = ()
    evaluation_num_samples: int = 1024
    evaluation_at_start: bool = False
    evaluation_at_end: bool = False
    midpoint_evaluation: bool = True
    source_cfg_mode: str = "legacy"
    evaluation_input_cfg_scales: tuple[float, ...] = (2.0,)
    evaluation_policy_snapshot: str = "raw"
    checkpoint_rolling_every: int = 0
    evaluation_every_n_steps: int = 0
    evaluation_seeds: tuple[int, ...] = ()

    @property
    def rollout_batch_size(self) -> int:
        return self.prompt_groups_per_outer * self.samples_per_prompt

    @property
    def policy_training_records_per_outer(self) -> int:
        """Endpoint/time-noise records consumed by policy training per round."""

        return (
            self.rollout_batch_size
            * self.num_inner_epochs
            * self.timesteps_per_batch
        )

    @property
    def center_training_records_per_outer(self) -> int:
        """Endpoint/time-noise records consumed by center training per round."""

        return (
            self.rollout_batch_size
            * self.center_num_inner_epochs
            * self.timesteps_per_batch
        )

    @property
    def policy_updates_per_outer(self) -> int:
        """Optimizer updates after all policy records are averaged together."""

        return 1

    @property
    def center_updates_per_outer(self) -> int:
        """Optimizer updates after all center records are averaged together."""

        return 1

    @property
    def policy_backward_microbatches_per_outer(self) -> int:
        return (
            self.rollout_batch_size
            // self.train_batch_size
            * self.num_inner_epochs
            * self.timesteps_per_batch
        )

    @property
    def center_backward_microbatches_per_outer(self) -> int:
        return (
            self.rollout_batch_size
            // self.train_batch_size
            * self.center_num_inner_epochs
            * self.timesteps_per_batch
        )

    def with_overrides(
        self,
        *,
        method: str | None = None,
        output_dir: str | None = None,
        max_outer_steps: int | None = None,
        samples_per_prompt: int | None = None,
        rollout_generation_batch_size: int | None = None,
        train_batch_size: int | None = None,
        reward_batch_size: int | None = None,
        self_condition_mode: str | None = None,
    ) -> "NFTConfig":
        updated = replace(
            self,
            method=self.method if method is None else method,
            output_dir=self.output_dir if output_dir is None else output_dir,
            max_outer_steps=(
                self.max_outer_steps if max_outer_steps is None else max_outer_steps
            ),
            samples_per_prompt=(
                self.samples_per_prompt
                if samples_per_prompt is None else samples_per_prompt
            ),
            rollout_generation_batch_size=(
                self.rollout_generation_batch_size
                if rollout_generation_batch_size is None
                else rollout_generation_batch_size
            ),
            train_batch_size=(
                self.train_batch_size
                if train_batch_size is None else train_batch_size
            ),
            reward_batch_size=(
                self.reward_batch_size
                if reward_batch_size is None else reward_batch_size
            ),
            self_condition_mode=(
                self.self_condition_mode
                if self_condition_mode is None else self_condition_mode
            ),
        )
        validate_nft_config(updated)
        return updated


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a configuration mapping")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, name: str, minimum: float | None = None) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _resolve_repo_path(path: str, name: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{name} does not exist: {candidate}")
    return candidate


def _require_group_output(path: str) -> str:
    candidate = Path(os.path.expandvars(path)).expanduser().resolve()
    try:
        candidate.relative_to(ELF_NFT_ASSET_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"output_dir must be under {ELF_NFT_ASSET_ROOT}, got {candidate}"
        ) from error
    if candidate == ELF_NFT_ASSET_ROOT.resolve():
        raise ValueError("output_dir must be a child of the group asset root")
    return str(candidate)


def validate_nft_config(config: NFTConfig) -> None:
    if config.task not in {"owt", "wmt", "xsum"}:
        raise ValueError("task must be 'owt', 'wmt', or 'xsum'")
    conditional_task = config.task in {"wmt", "xsum"}
    if config.parameterization != "full_shared":
        raise ValueError("parameterization must be 'full_shared'")
    if config.config_schema_version >= 11 and (
        not isinstance(config.pretrained_checkpoint_revision, str)
        or re.fullmatch(
            r"[0-9a-f]{40}", config.pretrained_checkpoint_revision
        )
        is None
    ):
        raise ValueError(
            "schema >=11 requires a full-SHA pretrained_checkpoint_revision"
        )
    if config.method not in {"standard", "shifted_center", "recovered_center"}:
        raise ValueError(
            "method must be standard, shifted_center, or recovered_center"
        )
    source_modes = {"cfg1": "standard", "cfg2_shifted_center": "shifted_center",
                    "cfg1_dual": "standard"}
    if config.source_cfg_mode not in {"legacy", *source_modes}:
        raise ValueError("unsupported source_cfg_mode")
    if config.source_cfg_mode != "legacy":
        if not conditional_task or config.config_schema_version < 9:
            raise ValueError("classic source CFG requires conditional schema >= 9")
        if config.method != source_modes[config.source_cfg_mode]:
            raise ValueError("method does not match source_cfg_mode")
        if (config.self_cond_cfg_scale != 1.0 or config.rollout_self_cond_cfg_scale != 1.0
                or config.rollout_field_mode != "native"):
            raise ValueError("classic source CFG requires native SCCFG=1 in training and rollout")
        if config.self_condition_mode != "conditional_only":
            raise ValueError("classic source CFG requires conditional_only SC routing")
        if config.source_cfg_mode == "cfg2_shifted_center" and config.rollout_cfg_scale != 2.0:
            raise ValueError("cfg2_shifted_center requires source rollout CFG=2")
        if config.midpoint_evaluation:
            raise ValueError(
                "conditional classic CFG uses scheduled evaluation, not OWT midpoint"
            )
    if (
        isinstance(config.center_pretrain_steps, bool)
        or not isinstance(config.center_pretrain_steps, int)
        or config.center_pretrain_steps < 0
    ):
        raise ValueError("center.pretrain_steps must be a non-negative integer")
    if config.center_pretrain_steps and (
        not conditional_task
        or config.method != "shifted_center"
        or config.source_cfg_mode != "cfg2_shifted_center"
        or config.rollout_cfg_scale != 2.0
    ):
        raise ValueError(
            "center pretraining requires conditional shifted_center with CFG2 rollouts"
        )
    conditional_evaluation = conditional_task and config.source_cfg_mode != "legacy"
    for name in ("checkpoint_rolling_every", "evaluation_every_n_steps"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    periodic_owt_evaluation = (
        config.task == "owt" and bool(config.evaluation_cases)
    )
    if config.evaluation_every_n_steps and not (
        conditional_evaluation or periodic_owt_evaluation
    ):
        raise ValueError(
            "periodic evaluation requires conditional source CFG modes or "
            "configured OWT SCCFG cases"
        )
    if config.evaluation_seeds:
        if config.task != "owt":
            raise ValueError("explicit evaluation seeds currently require task=owt")
        if len(set(config.evaluation_seeds)) != len(config.evaluation_seeds):
            raise ValueError("evaluation seeds must be unique")
    if config.rollout_time_schedule not in {"uniform", "logit_normal"}:
        raise ValueError("rollout_time_schedule must be uniform or logit_normal")
    if config.rollout_sampling_method not in {"ode", "sde"}:
        raise ValueError("rollout sampling method must be ode or sde")
    if config.rollout_sampling_method == "ode" and config.rollout_sde_gamma != 0.0:
        raise ValueError("ODE rollout requires sde_gamma=0")
    if config.rollout_sampling_method == "sde" and config.rollout_sde_gamma <= 0.0:
        raise ValueError("SDE rollout requires sde_gamma > 0")
    if config.timestep_schedule not in {"uniform", "logit_normal"}:
        raise ValueError("timestep_schedule must be uniform or logit_normal")
    if config.center_snapshot not in {"raw", "ema"}:
        raise ValueError("center_snapshot must be raw or ema")
    if config.keep_checkpoints < 3:
        raise ValueError("checkpoint.keep must be at least 3")
    if config.rollout_cfg_scale <= 0:
        raise ValueError("rollout_cfg_scale must be positive")
    if config.task == "owt" and config.rollout_cfg_scale != 1.0:
        raise ValueError("OWT has no input condition and requires input CFG scale 1.0")
    if config.self_cond_cfg_scale <= 0:
        raise ValueError("self_cond_cfg_scale must be positive")
    if config.rollout_self_cond_cfg_scale <= 0:
        raise ValueError("rollout_self_cond_cfg_scale must be positive")
    if config.rollout_field_mode not in {"native", "recovered_base"}:
        raise ValueError("rollout_field_mode must be native or recovered_base")
    if config.rollout_field_mode == "recovered_base":
        if config.task != "owt":
            raise ValueError("recovered_base rollout currently requires task=owt")
        if config.rollout_self_cond_cfg_scale <= 1.0:
            raise ValueError("recovered_base rollout requires rollout SC-CFG > 1")
    if not 0.0 <= config.self_cond_update_probability <= 1.0:
        raise ValueError("self_cond_update_probability must lie in [0, 1]")
    if config.training_self_cond_cfg_distribution not in {
        "fixed",
        "log_uniform_one_plus_scale",
    }:
        raise ValueError(
            "training self-cond CFG distribution must be fixed or "
            "log_uniform_one_plus_scale"
        )
    if not (
        math.isfinite(config.training_self_cond_cfg_min)
        and math.isfinite(config.training_self_cond_cfg_max)
        and config.training_self_cond_cfg_min > 0
        and config.training_self_cond_cfg_max > 0
    ):
        raise ValueError("training self-cond CFG bounds must be finite and positive")
    if config.training_self_cond_cfg_distribution == "fixed":
        if not (
            config.training_self_cond_cfg_min
            == config.training_self_cond_cfg_max
            == config.self_cond_cfg_scale
        ):
            raise ValueError(
                "fixed training self-cond CFG must equal sampling.self_cond_cfg_scale"
            )
    elif config.training_self_cond_cfg_min >= config.training_self_cond_cfg_max:
        raise ValueError("sampled training self-cond CFG requires min < max")
    if config.self_condition_mode not in {"bernoulli", "conditional_only", "paired"}:
        raise ValueError("self_condition_mode must be bernoulli, conditional_only, or paired")
    if config.training_velocity_reconstruction:
        if (
            config.task != "owt"
            or config.method not in {"standard", "shifted_center"}
            or config.source_cfg_mode != "legacy"
            or config.self_condition_mode == "paired"
        ):
            raise ValueError(
                "CFG1 velocity reconstruction currently requires an OWT standard "
                "or shifted_center single-branch field"
            )
        if (
            config.rollout_field_mode != "native"
            or config.self_cond_cfg_scale != 1.0
        ):
            raise ValueError(
                "CFG1 velocity reconstruction requires native rollout and "
                "SCCFG1 decoder fields"
            )
    if config.center_self_cond_cfg_scale <= 0:
        raise ValueError("center.self_cond_cfg_scale must be positive")
    if (
        isinstance(config.unconditional_update_every, bool)
        or not isinstance(config.unconditional_update_every, int)
        or config.unconditional_update_every < 0
    ):
        raise ValueError(
            "unconditional_update_every must be a non-negative integer"
        )
    if config.unconditional_update_every and (
        config.task != "owt"
        or config.method != "shifted_center"
        or config.self_condition_mode != "conditional_only"
        or config.training_velocity_reconstruction
    ):
        raise ValueError(
            "periodic unconditional updates require OWT shifted_center with "
            "conditional_only routing and native training fields"
        )
    if config.method == "recovered_center":
        if config.task != "owt":
            raise ValueError("recovered_center currently requires task=owt")
        if config.self_condition_mode != "paired":
            raise ValueError("recovered_center requires self_condition_mode=paired")
        if config.self_cond_cfg_scale <= 1.0:
            raise ValueError("recovered_center requires training SC-CFG > 1")
    if config.evaluation_cases:
        if config.task != "owt":
            raise ValueError("SCCFG evaluation cases currently require task=owt")
        names = [case.name for case in config.evaluation_cases]
        if len(set(names)) != len(names) or any(not name.strip() for name in names):
            raise ValueError("evaluation case names must be nonempty and unique")
        for case in config.evaluation_cases:
            if any(not math.isfinite(scale) or scale <= 0 for scale in (
                case.model_cfg_scale, case.external_cfg_scale
            )):
                raise ValueError("evaluation CFG scales must be finite and positive")
        if (config.rollout_generation_batch_size <= 0 or config.evaluation_num_samples <= 0 or
                config.evaluation_num_samples % config.rollout_generation_batch_size):
            raise ValueError("evaluation num_samples must be positive and divisible by generation batch size")
    elif (
        config.evaluation_at_start or config.evaluation_at_end
    ) and not conditional_evaluation:
        raise ValueError("enabled SCCFG evaluation requires at least one case")
    if conditional_evaluation:
        if not config.evaluation_input_cfg_scales:
            raise ValueError("conditional evaluation input CFG scales must be nonempty")
        if len(set(config.evaluation_input_cfg_scales)) != len(
            config.evaluation_input_cfg_scales
        ):
            raise ValueError("conditional evaluation input CFG scales must be unique")
        if any(
            not math.isfinite(scale) or scale < 1
            for scale in config.evaluation_input_cfg_scales
        ):
            raise ValueError("conditional evaluation input CFG must be finite and >= 1")
        if config.evaluation_policy_snapshot not in {"raw", "theta_ema"}:
            raise ValueError(
                "conditional evaluation policy snapshot must be raw or theta_ema"
            )
        if (
            config.evaluation_policy_snapshot == "theta_ema"
            and config.theta_ema_decay <= 0.0
        ):
            raise ValueError(
                "theta_ema evaluation requires training.theta_ema_decay > 0"
            )
        if (config.evaluation_num_samples <= 0 or config.rollout_generation_batch_size <= 0
                or config.evaluation_num_samples % config.rollout_generation_batch_size):
            raise ValueError(
                "conditional evaluation samples must divide by generation batch size"
            )
    if config.old_policy_decay_type not in {0, 1, 2}:
        raise ValueError("old_policy_decay_type must be 0, 1, or 2")
    if config.beta <= 0 or config.adv_clip_max <= 0:
        raise ValueError("beta and adv_clip_max must be positive")
    if config.kl_beta < 0 or config.decoder_ce_weight < 0 or config.learning_rate <= 0:
        raise ValueError(
            "kl_beta/decoder_ce_weight must be non-negative and learning_rate positive"
        )
    if config.center_learning_rate <= 0 or config.max_grad_norm <= 0:
        raise ValueError("center_learning_rate and max_grad_norm must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0 <= config.theta_ema_decay < 1:
        raise ValueError("theta_ema_decay must lie in [0, 1)")
    if not 0 <= config.center_ema_decay < 1:
        raise ValueError("center_ema_decay must lie in [0, 1)")
    if config.samples_per_prompt < 2:
        raise ValueError("samples_per_prompt must be >= 2 for group advantages")
    if config.task == "owt":
        if config.prompt_groups_per_outer != 1:
            raise ValueError("OWT must use exactly one null-condition reward group")
        if config.reward_type != "causal_lm_loglik":
            raise ValueError("OWT reward type must be causal_lm_loglik")
        if not config.reward_model_name.strip():
            raise ValueError("OWT reward model_name must be nonempty")
        if config.reward_batch_size <= 0 or config.reward_max_length < 2:
            raise ValueError("OWT reward batch_size must be positive and max_length >= 2")
        if config.empty_text_reward >= 0:
            raise ValueError("OWT empty_text_reward must be negative")
        if config.reward_combination_mode not in OWT_REWARD_MODES:
            raise ValueError(
                f"OWT reward combination mode must be one of {sorted(OWT_REWARD_MODES)}"
            )
        if config.reward_combination_mode == PPL_ONLY:
            if any(
                value != 0.0
                for value in (
                    config.entropy_target,
                    config.initial_entropy_weight,
                    config.max_entropy_weight,
                    config.entropy_ema_decay,
                    config.entropy_dual_learning_rate,
                )
            ):
                raise ValueError("ppl_only requires all entropy-control fields to be zero")
        else:
            if config.entropy_target <= 0:
                raise ValueError("joint OWT reward requires entropy_target > 0")
            if not 0.0 <= config.initial_entropy_weight <= 0.5:
                raise ValueError("initial_entropy_weight must lie in [0, 0.5]")
            if not config.initial_entropy_weight <= config.max_entropy_weight <= 0.5:
                raise ValueError(
                    "max_entropy_weight must lie in [initial_entropy_weight, 0.5]"
                )
            if not 0.0 <= config.entropy_ema_decay < 1.0:
                raise ValueError("entropy_ema_decay must lie in [0, 1)")
            if config.reward_combination_mode == ADAPTIVE_ENTROPY_FLOOR:
                if config.entropy_dual_learning_rate <= 0:
                    raise ValueError(
                        "adaptive entropy floor requires dual_learning_rate > 0"
                    )
            elif (
                config.reward_combination_mode == FIXED_JOINT
                and config.entropy_dual_learning_rate != 0.0
            ):
                raise ValueError("fixed_joint requires dual_learning_rate=0")
    else:
        expected_reward = "sentence_bleu" if config.task == "wmt" else "rouge_l"
        if config.reward_type != expected_reward:
            raise ValueError(
                f"{config.task} reward type must be {expected_reward}"
            )
        if config.reward_combination_mode != PPL_ONLY:
            raise ValueError(
                "conditional tasks support only their single configured reward"
            )
    if config.hypothesis_max_tokens < 0:
        raise ValueError("hypothesis_max_tokens cannot be negative")
    if (
        config.config_schema_version >= 11
        and conditional_task
        and config.hypothesis_max_tokens != 64
    ):
        raise ValueError(
            "schema-11 conditional reward/evaluation requires "
            "hypothesis_max_tokens=64"
        )
    if min(
        config.rollout_steps,
        config.prompt_groups_per_outer,
        config.max_outer_steps,
        config.rollout_generation_batch_size,
        config.train_batch_size,
        config.num_inner_epochs,
        config.timesteps_per_batch,
        config.center_num_inner_epochs,
        config.save_every,
        config.keep_checkpoints,
        config.keep_every_n_steps,
    ) <= 0:
        raise ValueError("all count/checkpoint fields must be positive")
    if config.rollout_batch_size % config.train_batch_size:
        raise ValueError("rollout batch must divide evenly into train_batch_size")
    if config.rollout_batch_size % config.rollout_generation_batch_size:
        raise ValueError(
            "rollout batch must divide evenly into rollout generation batch size"
        )
    _require_group_output(config.output_dir)


def load_experiment_config(path: str) -> tuple[Config, NFTConfig, dict[str, Any]]:
    """Load an ELF base YAML plus a strict NFT overlay."""

    config_path = _resolve_repo_path(path, "experiment config")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = (json.load(handle) if config_path.suffix == ".json" else yaml.safe_load(handle)) or {}
    root = _mapping(raw, "root")
    schema_version = _integer(root.get("schema_version"), "schema_version", 1)
    if schema_version not in {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}:
        raise ValueError(
            "schema_version must be 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, or 15"
        )
    _exact_keys(
        root,
        {
            "schema_version", "task", "base_config", "pretrained_checkpoint",
            "parameterization", "method", "output_dir", "seed", "sampling",
            "rollout", "reward", "training",
            "center", "checkpoint", "logging", "formal_experiment",
        } | ({"evaluation"} if schema_version >= 7 else set())
        | (
            {"pretrained_checkpoint_revision"}
            if schema_version >= 11
            else set()
        ),
        "root",
    )

    task = str(root["task"])
    sampling = _mapping(root["sampling"], "sampling")
    rollout = _mapping(root["rollout"], "rollout")
    reward = _mapping(root["reward"], "reward")
    training = _mapping(root["training"], "training")
    center = _mapping(root["center"], "center")
    checkpoint = dict(_mapping(root["checkpoint"], "checkpoint"))
    checkpoint.setdefault("keep_every", DEFAULT_CHECKPOINT_KEEP_EVERY)
    logging = _mapping(root["logging"], "logging")
    sampling_keys = {
        "method", "num_steps", "time_schedule", "sde_gamma", "cfg_scale",
        "self_cond_cfg_scale",
    }
    if schema_version >= 8:
        sampling_keys.update({"rollout_self_cond_cfg_scale", "rollout_field_mode"})
    _exact_keys(sampling, sampling_keys, "sampling")
    rollout_keys = {
        "prompt_groups_per_outer", "samples_per_prompt", "max_outer_steps"
    }
    if schema_version >= 6:
        rollout_keys.add("generation_batch_size")
    _exact_keys(rollout, rollout_keys, "rollout")
    if task == "owt":
        reward_keys = {
            "type", "model_name", "batch_size", "max_length",
            "empty_text_reward",
        }
        if schema_version >= 5:
            reward_keys.add("combination")
        _exact_keys(reward, reward_keys, "reward")
    elif task in {"wmt", "xsum"}:
        _exact_keys(
            reward,
            {"type"}
            | ({"hypothesis_max_tokens"} if schema_version >= 11 else set()),
            "reward",
        )
    else:
        raise ValueError("task must be 'owt', 'wmt', or 'xsum'")
    _exact_keys(
        training,
        {
            "batch_size", "num_inner_epochs", "learning_rate", "weight_decay",
            "beta", "kl_beta", "decoder_ce_weight", "adv_clip_max",
            "max_grad_norm", "old_policy_decay_type",
            "self_cond_update_probability", "timestep_schedule", "timesteps_per_batch",
        } | ({"self_condition_mode"} if schema_version >= 7 else set())
        | ({"source_cfg_mode"} if schema_version >= 9 else set())
        | ({"theta_ema_decay"} if schema_version >= 11 else set())
        | (
            {
                "self_cond_cfg_distribution",
                "self_cond_cfg_min",
                "self_cond_cfg_max",
                "velocity_reconstruction",
            }
            if schema_version >= 12
            else set()
        )
        | ({"unconditional_update_every"} if schema_version >= 15 else set()),
        "training",
    )
    _exact_keys(
        center,
        {"learning_rate", "num_inner_epochs", "ema_decay", "snapshot"}
        | ({"pretrain_steps"} if schema_version >= 10 else set())
        | ({"self_cond_cfg_scale"} if schema_version >= 14 else set()),
        "center",
    )
    _exact_keys(checkpoint, {"save_every", "keep", "keep_every"}
                | ({"rolling_every"} if schema_version >= 9 else set()), "checkpoint")
    _exact_keys(logging, {"use_wandb", "wandb_project", "wandb_run_name"}, "logging")

    evaluation_cases = ()
    evaluation_num_samples = 1024
    evaluation_at_start = evaluation_at_end = False
    midpoint_evaluation = True
    evaluation_input_cfg_scales = (2.0,)
    evaluation_policy_snapshot = "raw"
    evaluation_every_n_steps = 0
    evaluation_seeds = ()
    if schema_version >= 7:
        evaluation = _mapping(root["evaluation"], "evaluation")
        evaluation_keys = {
            "cases", "num_samples", "at_start", "at_end", "midpoint"
        }
        if schema_version >= 11 and task in {"wmt", "xsum"}:
            evaluation_keys.update(
                {"input_cfg_scales", "policy_snapshot", "every"}
            )
        elif schema_version >= 13 and task == "owt":
            evaluation_keys.update({"every", "seeds"})
        elif schema_version >= 9 and task in {"wmt", "xsum"}:
            evaluation_keys.update({"input_cfg_scale", "every"})
        _exact_keys(evaluation, evaluation_keys, "evaluation")
        if schema_version >= 11 and task in {"wmt", "xsum"}:
            if not isinstance(evaluation["input_cfg_scales"], list):
                raise ValueError("evaluation.input_cfg_scales must be a list")
            evaluation_input_cfg_scales = tuple(
                _finite_float(
                    value,
                    f"evaluation.input_cfg_scales[{index}]",
                    1.0,
                )
                for index, value in enumerate(evaluation["input_cfg_scales"])
            )
            evaluation_policy_snapshot = str(evaluation["policy_snapshot"])
            evaluation_every_n_steps = _integer(
                evaluation["every"], "evaluation.every", 0
            )
        elif schema_version >= 13 and task == "owt":
            evaluation_every_n_steps = _integer(
                evaluation["every"], "evaluation.every", 0
            )
            if not isinstance(evaluation["seeds"], list) or not evaluation["seeds"]:
                raise ValueError("evaluation.seeds must be a nonempty list")
            evaluation_seeds = tuple(
                _integer(seed, f"evaluation.seeds[{index}]", 0)
                for index, seed in enumerate(evaluation["seeds"])
            )
        elif schema_version >= 9 and task in {"wmt", "xsum"}:
            evaluation_input_cfg_scales = (_finite_float(
                evaluation["input_cfg_scale"], "evaluation.input_cfg_scale", 1.0
            ),)
            evaluation_every_n_steps = _integer(evaluation["every"], "evaluation.every", 0)
        if not isinstance(evaluation["cases"], list):
            raise ValueError("evaluation.cases must be a list")
        cases = []
        for item in evaluation["cases"]:
            case = _mapping(item, "evaluation case")
            _exact_keys(case, {"name", "model_cfg_scale", "external_cfg_scale"}, "evaluation case")
            cases.append(SCCFGEvaluationCase(
                name=str(case["name"]),
                model_cfg_scale=_finite_float(case["model_cfg_scale"], "model_cfg_scale", 0.0),
                external_cfg_scale=_finite_float(case["external_cfg_scale"], "external_cfg_scale", 0.0),
            ))
        evaluation_cases = tuple(cases)
        evaluation_num_samples = _integer(evaluation["num_samples"], "evaluation.num_samples", 1)
        evaluation_at_start = _boolean(evaluation["at_start"], "evaluation.at_start")
        evaluation_at_end = _boolean(evaluation["at_end"], "evaluation.at_end")
        midpoint_evaluation = _boolean(evaluation["midpoint"], "evaluation.midpoint")

    if task == "owt" and schema_version >= 5:
        combination = _mapping(reward["combination"], "reward.combination")
        _exact_keys(
            combination,
            {
                "mode", "entropy_target", "initial_entropy_weight",
                "max_entropy_weight", "entropy_ema_decay", "dual_learning_rate",
            },
            "reward.combination",
        )
        reward_combination_mode = str(combination["mode"])
        entropy_target = _finite_float(
            combination["entropy_target"], "reward.combination.entropy_target", 0.0
        )
        initial_entropy_weight = _finite_float(
            combination["initial_entropy_weight"],
            "reward.combination.initial_entropy_weight",
            0.0,
        )
        max_entropy_weight = _finite_float(
            combination["max_entropy_weight"],
            "reward.combination.max_entropy_weight",
            0.0,
        )
        entropy_ema_decay = _finite_float(
            combination["entropy_ema_decay"],
            "reward.combination.entropy_ema_decay",
            0.0,
        )
        entropy_dual_learning_rate = _finite_float(
            combination["dual_learning_rate"],
            "reward.combination.dual_learning_rate",
            0.0,
        )
    else:
        reward_combination_mode = PPL_ONLY
        entropy_target = 0.0
        initial_entropy_weight = 0.0
        max_entropy_weight = 0.0
        entropy_ema_decay = 0.0
        entropy_dual_learning_rate = 0.0

    base_config_path = _resolve_repo_path(str(root["base_config"]), "base_config")
    # Upstream ELF training YAMLs intentionally express sampling config paths
    # relative to ``src/`` because its entry points are launched there.
    previous_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT / "src")
        elf_config = load_config_from_yaml(str(base_config_path))
    finally:
        os.chdir(previous_cwd)
    prompt_groups_per_outer = _integer(
        rollout["prompt_groups_per_outer"], "rollout.prompt_groups_per_outer", 1
    )
    samples_per_prompt = _integer(
        rollout["samples_per_prompt"], "rollout.samples_per_prompt", 2
    )
    rollout_batch_size = prompt_groups_per_outer * samples_per_prompt
    rollout_generation_batch_size = (
        _integer(
            rollout["generation_batch_size"],
            "rollout.generation_batch_size",
            1,
        )
        if schema_version >= 6
        else rollout_batch_size
    )

    config = NFTConfig(
        config_path=str(config_path),
        config_schema_version=schema_version,
        task=task,
        base_config=str(base_config_path),
        pretrained_checkpoint=str(root["pretrained_checkpoint"]),
        pretrained_checkpoint_revision=(
            str(root["pretrained_checkpoint_revision"])
            if schema_version >= 11
            else None
        ),
        parameterization=str(root["parameterization"]),
        method=str(root["method"]),
        output_dir=_require_group_output(str(root["output_dir"])),
        seed=_integer(root["seed"], "seed"),
        rollout_sampling_method=str(sampling["method"]),
        rollout_steps=_integer(sampling["num_steps"], "sampling.num_steps", 1),
        rollout_time_schedule=str(sampling["time_schedule"]),
        rollout_sde_gamma=_finite_float(
            sampling["sde_gamma"], "sampling.sde_gamma", 0.0
        ),
        rollout_cfg_scale=_finite_float(sampling["cfg_scale"], "sampling.cfg_scale", 0.0),
        self_cond_cfg_scale=_finite_float(
            sampling["self_cond_cfg_scale"], "sampling.self_cond_cfg_scale", 0.0
        ),
        rollout_self_cond_cfg_scale=_finite_float(
            sampling.get("rollout_self_cond_cfg_scale", sampling["self_cond_cfg_scale"]),
            "sampling.rollout_self_cond_cfg_scale",
            0.0,
        ),
        rollout_field_mode=str(sampling.get("rollout_field_mode", "native")),
        reward_type=str(reward["type"]),
        hypothesis_max_tokens=(
            _integer(
                reward["hypothesis_max_tokens"],
                "reward.hypothesis_max_tokens",
                1,
            )
            if task in {"wmt", "xsum"} and schema_version >= 11
            else 0
        ),
        reward_model_name=str(reward.get("model_name", "")),
        reward_batch_size=(
            _integer(reward["batch_size"], "reward.batch_size", 1)
            if task == "owt" else 0
        ),
        reward_max_length=(
            _integer(reward["max_length"], "reward.max_length", 2)
            if task == "owt" else 0
        ),
        empty_text_reward=(
            _finite_float(reward["empty_text_reward"], "reward.empty_text_reward")
            if task == "owt" else 0.0
        ),
        reward_combination_mode=reward_combination_mode,
        entropy_target=entropy_target,
        initial_entropy_weight=initial_entropy_weight,
        max_entropy_weight=max_entropy_weight,
        entropy_ema_decay=entropy_ema_decay,
        entropy_dual_learning_rate=entropy_dual_learning_rate,
        prompt_groups_per_outer=prompt_groups_per_outer,
        samples_per_prompt=samples_per_prompt,
        max_outer_steps=_integer(rollout["max_outer_steps"], "rollout.max_outer_steps", 1),
        rollout_generation_batch_size=rollout_generation_batch_size,
        train_batch_size=_integer(training["batch_size"], "training.batch_size", 1),
        num_inner_epochs=_integer(
            training["num_inner_epochs"], "training.num_inner_epochs", 1
        ),
        learning_rate=_finite_float(
            training["learning_rate"], "training.learning_rate", 0.0
        ),
        weight_decay=_finite_float(
            training["weight_decay"], "training.weight_decay", 0.0
        ),
        theta_ema_decay=(
            _finite_float(
                training["theta_ema_decay"],
                "training.theta_ema_decay",
                0.0,
            )
            if schema_version >= 11
            else 0.0
        ),
        beta=_finite_float(training["beta"], "training.beta", 0.0),
        kl_beta=_finite_float(training["kl_beta"], "training.kl_beta", 0.0),
        decoder_ce_weight=_finite_float(
            training["decoder_ce_weight"], "training.decoder_ce_weight", 0.0
        ),
        adv_clip_max=_finite_float(
            training["adv_clip_max"], "training.adv_clip_max", 0.0
        ),
        max_grad_norm=_finite_float(
            training["max_grad_norm"], "training.max_grad_norm", 0.0
        ),
        old_policy_decay_type=_integer(
            training["old_policy_decay_type"], "training.old_policy_decay_type"
        ),
        self_cond_update_probability=_finite_float(
            training["self_cond_update_probability"],
            "training.self_cond_update_probability",
            0.0,
        ),
        training_self_cond_cfg_distribution=(
            str(training["self_cond_cfg_distribution"])
            if schema_version >= 12
            else "fixed"
        ),
        training_self_cond_cfg_min=(
            _finite_float(
                training["self_cond_cfg_min"],
                "training.self_cond_cfg_min",
                0.0,
            )
            if schema_version >= 12
            else _finite_float(
                sampling["self_cond_cfg_scale"],
                "sampling.self_cond_cfg_scale",
                0.0,
            )
        ),
        training_self_cond_cfg_max=(
            _finite_float(
                training["self_cond_cfg_max"],
                "training.self_cond_cfg_max",
                0.0,
            )
            if schema_version >= 12
            else _finite_float(
                sampling["self_cond_cfg_scale"],
                "sampling.self_cond_cfg_scale",
                0.0,
            )
        ),
        training_velocity_reconstruction=(
            _boolean(
                training["velocity_reconstruction"],
                "training.velocity_reconstruction",
            )
            if schema_version >= 12
            else False
        ),
        unconditional_update_every=(
            _integer(
                training["unconditional_update_every"],
                "training.unconditional_update_every",
                0,
            )
            if schema_version >= 15
            else 0
        ),
        timestep_schedule=str(training["timestep_schedule"]),
        timesteps_per_batch=_integer(
            training["timesteps_per_batch"], "training.timesteps_per_batch", 1
        ),
        center_learning_rate=_finite_float(
            center["learning_rate"], "center.learning_rate", 0.0
        ),
        center_num_inner_epochs=_integer(
            center["num_inner_epochs"], "center.num_inner_epochs", 1
        ),
        center_ema_decay=_finite_float(
            center["ema_decay"], "center.ema_decay", 0.0
        ),
        center_snapshot=str(center["snapshot"]),
        center_pretrain_steps=(
            _integer(center["pretrain_steps"], "center.pretrain_steps", 0)
            if schema_version >= 10
            else 0
        ),
        center_self_cond_cfg_scale=_finite_float(
            center["self_cond_cfg_scale"]
            if schema_version >= 14
            else sampling["self_cond_cfg_scale"],
            "center.self_cond_cfg_scale",
            0.0,
        ),
        save_every=_integer(checkpoint["save_every"], "checkpoint.save_every", 1),
        keep_checkpoints=_integer(checkpoint["keep"], "checkpoint.keep", 1),
        keep_every_n_steps=_integer(
            checkpoint["keep_every"], "checkpoint.keep_every", 1
        ),
        use_wandb=_boolean(logging["use_wandb"], "logging.use_wandb"),
        wandb_project=str(logging["wandb_project"]),
        wandb_run_name=str(logging["wandb_run_name"]),
        formal_experiment=_boolean(root["formal_experiment"], "formal_experiment"),
        self_condition_mode=str(training.get("self_condition_mode", "bernoulli")),
        evaluation_cases=evaluation_cases,
        evaluation_num_samples=evaluation_num_samples,
        evaluation_at_start=evaluation_at_start,
        evaluation_at_end=evaluation_at_end,
        midpoint_evaluation=midpoint_evaluation,
        source_cfg_mode=str(training.get("source_cfg_mode", "legacy")),
        evaluation_input_cfg_scales=evaluation_input_cfg_scales,
        evaluation_policy_snapshot=evaluation_policy_snapshot,
        evaluation_every_n_steps=evaluation_every_n_steps,
        evaluation_seeds=evaluation_seeds,
        checkpoint_rolling_every=_integer(checkpoint.get("rolling_every", 0), "checkpoint.rolling_every", 0),
    )
    # Checked on the file, not in validate_nft_config: evaluation sweeps replace
    # self_cond_cfg_scale per case and must not trip over the inert center scale.
    if config.center_self_cond_cfg_scale != config.self_cond_cfg_scale and (
        config.task != "owt"
        or config.method != "shifted_center"
        or config.source_cfg_mode != "legacy"
    ):
        raise ValueError(
            "a separate center SC-CFG scale requires OWT shifted_center"
        )
    validate_nft_config(config)

    if config.task == "owt":
        if elf_config.eval_data_path is not None or elf_config.max_input_length is not None:
            raise ValueError("OWT requires an unconditional ELF base config")
    elif elf_config.eval_data_path is None or elf_config.max_input_length is None:
        raise ValueError("conditional tasks require a conditional ELF base config")
    if config.task in {"wmt", "xsum"} and config.config_schema_version >= 11:
        for name in ("tokenizer_revision", "encoder_checkpoint_revision"):
            value = getattr(elf_config, name, None)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{40}", value) is None
            ):
                raise ValueError(
                    f"schema 11 conditional task requires full-SHA base {name}"
                )
    if config.task == "xsum":
        from .xsum_data import xsum_settings_from_config

        xsum_settings_from_config(elf_config, config)
    if elf_config.self_cond_prob <= 0:
        raise ValueError("ELF NFT SC fields require self_cond_prob > 0")
    if elf_config.num_self_cond_cfg_tokens <= 0:
        raise ValueError("ELF NFT requires the pretrained SC-CFG token interface")
    if any(not (
        elf_config.self_cond_cfg_min
        <= scale
        <= elf_config.self_cond_cfg_max
    ) for scale in (
        config.self_cond_cfg_scale,
        config.rollout_self_cond_cfg_scale,
        config.center_self_cond_cfg_scale,
        config.training_self_cond_cfg_min,
        config.training_self_cond_cfg_max,
        *(case.model_cfg_scale for case in config.evaluation_cases),
    )):
        raise ValueError(
            "training, rollout, and evaluation self_cond_cfg_scale must remain "
            "inside the ELF pretraining range "
            f"[{elf_config.self_cond_cfg_min}, {elf_config.self_cond_cfg_max}]"
        )
    return elf_config, config, dict(raw)
