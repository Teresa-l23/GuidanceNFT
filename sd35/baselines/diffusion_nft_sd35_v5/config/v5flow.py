"""Build an SD3.5 persistent-v5 ConfigDict from an editable JSON file.

``ml_collections.DEFINE_config_file`` requires a Python module, so this file is
only a compatibility bridge.  Experiment values live in ``v5flow.json`` (or in
the JSON path passed after the colon in ``--config v5flow.py:path.json``).
"""

from __future__ import annotations

import imp
import json
import math
import os
import re
from pathlib import Path

import ml_collections


_HERE = Path(__file__).resolve().parent
_OFFICIAL_CONFIG = (_HERE / ".." / ".." / "diffusion_nft" / "upstream" / "config" / "nft.py").resolve()
official = imp.load_source("diffusion_nft_official_config", os.fspath(_OFFICIAL_CONFIG))

_TASK_DATASETS = {
    "geneval": "geneval",
    "ocr": "ocr",
    "pickscore": "pickscore",
    "clipscore": "pickscore",
    "hpsv2": "pickscore",
    "learned3": "pickscore",
    "control-learned3": "pickscore",
    "control-geneval": "geneval",
    "control-ocr": "ocr",
    "control-pickscore": "pickscore",
    "control-hpsv2": "pickscore",
    "pickscore-no-v5-ema": "pickscore",
}
_TASK_REWARDS = {
    "geneval": {"geneval"},
    "ocr": {"ocr"},
    "pickscore": {"pickscore"},
    "clipscore": {"clipscore"},
    "hpsv2": {"hpsv2"},
    "learned3": {"pickscore", "clipscore", "hpsv2"},
    "control-learned3": {"pickscore", "clipscore", "hpsv2"},
    "control-geneval": {"geneval"},
    "control-ocr": {"ocr"},
    "control-pickscore": {"pickscore"},
    "control-hpsv2": {"hpsv2"},
    "pickscore-no-v5-ema": {"pickscore"},
}
_TASK_EVAL_DATASETS = {
    "geneval": "geneval",
    "ocr": "ocr",
    "pickscore": "drawbench",
    "clipscore": "drawbench",
    "hpsv2": "drawbench",
    "learned3": "drawbench",
    "control-learned3": "drawbench",
    "control-geneval": "geneval",
    "control-ocr": "ocr",
    "control-pickscore": "drawbench",
    "control-hpsv2": "drawbench",
    "pickscore-no-v5-ema": "drawbench",
}
_EVAL_DATASET_SIZES = {"geneval": 2212, "ocr": 1018, "drawbench": 1000}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "task",
    "world_size",
    "cfg_mode",
    "cfg_unconditional_source",
    "reward_fn",
    "sampling",
    "training",
    "v5flow",
    "evaluation",
}
_OPTIONAL_TOP_LEVEL_KEYS = {
    "cfg_mode",
    "cfg_unconditional_source",
}
_CFG_MODES = {
    "cfg1",
    "cfg1_dual",
    "cfg1_dropout",
    "cfg4_rollout",
    "cfg4_shifted_center",
    "cfg4_dual_shifted_center",
    "cfg4_full",
    "cfg4_alternating",
}
_CFG_UNCONDITIONAL_SOURCES = {"same_model", "pretrained_base"}
_SAMPLING_KEYS = {
    "rollout_steps",
    "eval_steps",
    "resolution",
    "num_image_per_prompt",
    "num_groups",
    "max_train_batch_size",
    "test_batch_size",
}
_TRAINING_KEYS = {
    "gradient_steps_per_epoch",
    "max_train_steps",
    "seed",
    "beta",
    "kl_beta",
    "decay_type",
    "save_freq",
    "latest_save_freq",
    "eval_freq",
    "initial_lora_path",
    "initial_v5flow_checkpoint",
    "initial_v5flow_state_sha256",
    "condition_drop_rate",
}
_OPTIONAL_TRAINING_KEYS = {
    "latest_save_freq",
    "initial_lora_path",
    "initial_v5flow_checkpoint",
    "initial_v5flow_state_sha256",
    "condition_drop_rate",
}
_V5FLOW_KEYS = {
    "enabled",
    "center",
    "init",
    "timestep_sampling",
    "num_inner_epochs",
    "update_interval",
    "learning_rate",
    "max_grad_norm",
    "use_ema",
    "ema_decay",
    "snapshot",
    "theta_lifecycle",
    "theta_ema_decay",
    "loss_space",
    "normalize_branch_mae",
    "initializer_path",
    "pretrained_checkpoint",
    "pretrained_state_sha256",
    "warmup",
}
_OPTIONAL_V5FLOW_KEYS = {
    "initializer_path",
    "pretrained_checkpoint",
    "pretrained_state_sha256",
    "update_interval",
    "warmup",
}
_V5FLOW_WARMUP_KEYS = {
    "enabled",
    "export_only",
    "output_path",
    "seed",
    "num_rollout_batches",
    "num_holdout_batches",
    "num_inner_epochs",
    "num_train_timesteps",
    "gradient_accumulation_steps",
    "timestep_sampling",
    "num_refreshes",
    "checkpoint_root",
    "keep_checkpoints",
    "prompt_pool",
    "parameter_dtype",
}
_OPTIONAL_V5FLOW_WARMUP_KEYS = {
    "num_refreshes",
    "checkpoint_root",
    "keep_checkpoints",
    "prompt_pool",
    "parameter_dtype",
}
_LEGACY_EVALUATION_KEYS = {"auto_submit", "save_images"}
_EXPLICIT_EVALUATION_KEYS = {
    "auto_submit",
    "save_images",
    "dataset",
    "reward_fn",
    "batch_size",
    "expected_samples",
    "guidance_scales",
    "inline_max_samples",
    "inline_at_final",
}
_OPTIONAL_EXPLICIT_EVALUATION_KEYS = {
    "guidance_scales",
    "inline_max_samples",
    "inline_at_final",
}


def _require_mapping(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _require_exact_keys(
    value: dict,
    field: str,
    expected: set[str],
    *,
    optional: set[str] | frozenset[str] = frozenset(),
) -> dict:
    missing = sorted(expected - set(optional) - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"Invalid {field} keys; missing={missing}, unknown={unknown}")
    return value


def _require_int(value, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_finite(value, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def _require_bool(value, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _require_choice(value, field: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}, got {value!r}")
    return value


def _load_settings(config_path: str) -> dict:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = _HERE / path
    path = path.resolve()
    if path.suffix.lower() != ".json":
        raise ValueError(f"v5flow config must be JSON, got: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            settings = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    settings = _require_mapping(settings, "config")
    unknown = sorted(set(settings) - _TOP_LEVEL_KEYS)
    missing = sorted(_TOP_LEVEL_KEYS - _OPTIONAL_TOP_LEVEL_KEYS - set(settings))
    if unknown or missing:
        raise ValueError(f"Invalid top-level JSON keys; missing={missing}, unknown={unknown}")
    if settings["schema_version"] != 2:
        raise ValueError("schema_version must be 2")
    return settings


def get_config(config_path: str = "v5flow.json"):
    """Load JSON, validate its experiment geometry, and return a ConfigDict."""

    settings = _load_settings(config_path)
    task = settings["task"]
    if task not in _TASK_DATASETS:
        raise ValueError(f"task must be one of {sorted(_TASK_DATASETS)}, got {task!r}")
    dataset = _TASK_DATASETS[task]
    world_size = _require_int(settings["world_size"], "world_size")
    cfg_mode_is_explicit = "cfg_mode" in settings
    cfg_mode = _require_choice(
        settings.get("cfg_mode", "cfg1"),
        "cfg_mode",
        _CFG_MODES,
    )
    cfg_unconditional_source = _require_choice(
        settings.get("cfg_unconditional_source", "same_model"),
        "cfg_unconditional_source",
        _CFG_UNCONDITIONAL_SOURCES,
    )

    reward_fn = _require_mapping(settings["reward_fn"], "reward_fn")
    if not reward_fn:
        raise ValueError("reward_fn must not be empty")
    reward_fn = {
        name: _require_finite(weight, f"reward_fn.{name}", positive=True)
        for name, weight in reward_fn.items()
    }
    expected_rewards = _TASK_REWARDS[task]
    if set(reward_fn) != expected_rewards:
        raise ValueError(
            f"task={task!r} requires reward_fn keys {sorted(expected_rewards)}, "
            f"got {sorted(reward_fn)}"
        )

    sampling = _require_exact_keys(
        _require_mapping(settings["sampling"], "sampling"),
        "sampling",
        _SAMPLING_KEYS,
    )
    rollout_steps = _require_int(sampling["rollout_steps"], "sampling.rollout_steps")
    eval_steps = _require_int(sampling["eval_steps"], "sampling.eval_steps")
    resolution = _require_int(sampling["resolution"], "sampling.resolution")
    num_image_per_prompt = _require_int(
        sampling["num_image_per_prompt"], "sampling.num_image_per_prompt", minimum=2
    )
    num_groups = _require_int(sampling["num_groups"], "sampling.num_groups")
    max_train_batch_size = _require_int(
        sampling["max_train_batch_size"], "sampling.max_train_batch_size"
    )
    test_batch_size = _require_int(sampling["test_batch_size"], "sampling.test_batch_size")

    training = _require_exact_keys(
        _require_mapping(settings["training"], "training"),
        "training",
        _TRAINING_KEYS,
        optional=_OPTIONAL_TRAINING_KEYS,
    )
    gradient_steps_per_epoch = _require_int(
        training["gradient_steps_per_epoch"], "training.gradient_steps_per_epoch"
    )
    max_train_steps = _require_int(
        training["max_train_steps"], "training.max_train_steps", minimum=0
    )
    seed = _require_int(training["seed"], "training.seed", minimum=0)
    beta = _require_finite(training["beta"], "training.beta", positive=True)
    kl_beta = _require_finite(training["kl_beta"], "training.kl_beta")
    if kl_beta < 0.0:
        raise ValueError("training.kl_beta must be non-negative")
    condition_drop_rate = _require_finite(
        training.get("condition_drop_rate", 0.0),
        "training.condition_drop_rate",
    )
    if cfg_mode == "cfg1_dropout":
        if not 0.0 < condition_drop_rate < 1.0:
            raise ValueError(
                "cfg1_dropout requires training.condition_drop_rate in (0, 1)"
            )
    elif condition_drop_rate != 0.0:
        raise ValueError(
            "training.condition_drop_rate must be zero unless "
            "cfg_mode='cfg1_dropout'"
        )
    decay_type = _require_int(training["decay_type"], "training.decay_type", minimum=0)
    if decay_type not in {0, 1, 2}:
        raise ValueError("training.decay_type must be 0, 1, or 2")
    save_freq = _require_int(training["save_freq"], "training.save_freq", minimum=0)
    latest_save_freq = _require_int(
        training.get("latest_save_freq", 1),
        "training.latest_save_freq",
        minimum=0,
    )
    eval_freq = _require_int(training["eval_freq"], "training.eval_freq", minimum=0)
    initial_lora_path = training.get("initial_lora_path", "")
    if not isinstance(initial_lora_path, str):
        raise ValueError("training.initial_lora_path must be a string")
    if initial_lora_path:
        initial_lora_path = os.fspath(Path(os.path.expandvars(initial_lora_path)).expanduser())
        if not Path(initial_lora_path).is_absolute():
            raise ValueError("training.initial_lora_path must be absolute when set")
    initial_v5flow_checkpoint = training.get("initial_v5flow_checkpoint", "")
    if not isinstance(initial_v5flow_checkpoint, str):
        raise ValueError("training.initial_v5flow_checkpoint must be a string")
    if initial_v5flow_checkpoint:
        initial_v5flow_checkpoint = os.fspath(
            Path(initial_v5flow_checkpoint).expanduser()
        )
        if not Path(initial_v5flow_checkpoint).is_absolute():
            raise ValueError(
                "training.initial_v5flow_checkpoint must be absolute when set"
            )
    initial_v5flow_state_sha256 = training.get(
        "initial_v5flow_state_sha256", ""
    )
    if not isinstance(initial_v5flow_state_sha256, str):
        raise ValueError("training.initial_v5flow_state_sha256 must be a string")
    has_promotion_checkpoint = bool(initial_v5flow_checkpoint)
    has_promotion_digest = bool(initial_v5flow_state_sha256)
    if has_promotion_checkpoint != has_promotion_digest:
        raise ValueError(
            "training.initial_v5flow_checkpoint and "
            "training.initial_v5flow_state_sha256 must be set together"
        )
    if has_promotion_digest and re.fullmatch(
        r"[0-9a-f]{64}", initial_v5flow_state_sha256
    ) is None:
        raise ValueError(
            "training.initial_v5flow_state_sha256 must be a lowercase SHA256"
        )
    if has_promotion_checkpoint and not initial_lora_path:
        raise ValueError(
            "training.initial_v5flow_checkpoint requires initial_lora_path as "
            "the source-policy structure witness"
        )

    config = official._get_config(
        base_model="sd3",
        n_gpus=world_size,
        gradient_step_per_epoch=gradient_steps_per_epoch,
        dataset=dataset,
        reward_fn=reward_fn,
        name=f"v5flow_{task}_steps{rollout_steps}_{world_size}gpu",
        num_image_per_prompt=num_image_per_prompt,
        num_groups=num_groups,
        max_train_batch_size=max_train_batch_size,
    )
    if int(config.sample.train_batch_size) != max_train_batch_size:
        raise ValueError(
            "sampling.max_train_batch_size must be exactly realizable by the prompt-group "
            f"geometry; requested={max_train_batch_size}, resolved={config.sample.train_batch_size}"
        )
    if max_train_steps % gradient_steps_per_epoch != 0:
        raise ValueError(
            "training.max_train_steps must end on an outer-epoch boundary: "
            f"steps={max_train_steps}, updates_per_epoch={gradient_steps_per_epoch}"
        )
    config.sample.num_steps = rollout_steps
    config.sample.eval_num_steps = eval_steps
    config.sample.guidance_scale = (
        1.0 if cfg_mode in {"cfg1", "cfg1_dual", "cfg1_dropout"} else 4.0
    )
    config.sample.test_batch_size = test_batch_size
    config.resolution = resolution
    config.max_train_steps = max_train_steps
    # Runtime-only boundary for releasing GPU memory to standalone evaluation.
    # The scientific horizon and decay schedule remain max_train_steps.
    config.segment_end_step = 0
    config.seed = seed
    config.beta = beta
    config.train.beta = kl_beta
    config.train.condition_drop_rate = condition_drop_rate
    config.decay_type = decay_type
    config.eval_freq = eval_freq
    config.save_freq = save_freq
    config.latest_save_freq = latest_save_freq
    config.train.lora_path = initial_lora_path
    config.train.initial_v5flow_checkpoint = initial_v5flow_checkpoint
    config.train.initial_v5flow_state_sha256 = initial_v5flow_state_sha256
    config.run_name = f"sd35_v5flow_{task}_steps{rollout_steps}_{world_size}gpu"
    config.save_dir = f"logs/nft_sd35_v5/{task}_steps{rollout_steps}_{world_size}gpu"
    config.reward_fn = ml_collections.ConfigDict(reward_fn)
    config.reward_executor_workers = 1
    # The batch launcher supplies a group-backed run-specific directory.  The
    # defaults keep direct/legacy invocations side-effect free.
    config.inline_preview_dir = ""
    config.inline_preview_max_samples = 15
    config.inline_eval_max_samples = 0
    config.inline_eval_at_final = False
    if cfg_mode_is_explicit:
        # Historical JSONs must continue through the released old-center code
        # path instead of being silently rerouted through the CFG extension.
        config.cfg_mode = cfg_mode
    config.cfg_unconditional_source = cfg_unconditional_source

    v5_settings = _require_exact_keys(
        _require_mapping(settings["v5flow"], "v5flow"),
        "v5flow",
        _V5FLOW_KEYS,
        optional=_OPTIONAL_V5FLOW_KEYS,
    )
    config.v5flow = ml_collections.ConfigDict(v5_settings)
    config.v5flow.enabled = _require_bool(v5_settings["enabled"], "v5flow.enabled")
    config.v5flow.use_ema = _require_bool(v5_settings["use_ema"], "v5flow.use_ema")
    config.v5flow.center = _require_choice(
        v5_settings["center"], "v5flow.center", {"v5", "old"}
    )
    config.v5flow.init = _require_choice(
        v5_settings["init"], "v5flow.init", {"old_once"}
    )
    config.v5flow.timestep_sampling = _require_choice(
        v5_settings["timestep_sampling"],
        "v5flow.timestep_sampling",
        {"rollout_schedule", "continuous_uniform"},
    )
    config.v5flow.snapshot = _require_choice(
        v5_settings["snapshot"], "v5flow.snapshot", {"raw", "ema"}
    )
    config.v5flow.learning_rate = _require_finite(
        v5_settings["learning_rate"], "v5flow.learning_rate", positive=True
    )
    config.v5flow.max_grad_norm = _require_finite(
        v5_settings["max_grad_norm"], "v5flow.max_grad_norm", positive=True
    )
    config.v5flow.num_inner_epochs = _require_int(
        v5_settings["num_inner_epochs"], "v5flow.num_inner_epochs"
    )
    config.v5flow.update_interval = _require_int(
        v5_settings.get("update_interval", 1),
        "v5flow.update_interval",
        minimum=0,
    )
    config.v5flow.ema_decay = _require_finite(v5_settings["ema_decay"], "v5flow.ema_decay")
    config.v5flow.theta_ema_decay = _require_finite(
        v5_settings["theta_ema_decay"], "v5flow.theta_ema_decay"
    )
    if not 0.0 <= config.v5flow.ema_decay < 1.0:
        raise ValueError("v5flow.ema_decay must lie in [0, 1)")
    if not 0.0 <= config.v5flow.theta_ema_decay < 1.0:
        raise ValueError("v5flow.theta_ema_decay must lie in [0, 1)")
    config.v5flow.theta_lifecycle = _require_choice(
        v5_settings["theta_lifecycle"],
        "v5flow.theta_lifecycle",
        {"persistent", "reset"},
    )
    config.v5flow.loss_space = _require_choice(
        v5_settings["loss_space"],
        "v5flow.loss_space",
        {"velocity", "reconstruction"},
    )
    config.v5flow.normalize_branch_mae = _require_bool(
        v5_settings["normalize_branch_mae"],
        "v5flow.normalize_branch_mae",
    )
    initializer_path = v5_settings.get("initializer_path", "")
    if not isinstance(initializer_path, str):
        raise ValueError("v5flow.initializer_path must be a string")
    if initializer_path:
        initializer_path = os.fspath(Path(initializer_path).expanduser())
        if not Path(initializer_path).is_absolute():
            raise ValueError("v5flow.initializer_path must be absolute when set")
    config.v5flow.initializer_path = initializer_path
    if initializer_path and not config.v5flow.enabled:
        raise ValueError("v5flow.initializer_path requires v5flow.enabled=true")

    pretrained_checkpoint = v5_settings.get("pretrained_checkpoint", "")
    if not isinstance(pretrained_checkpoint, str):
        raise ValueError("v5flow.pretrained_checkpoint must be a string")
    if pretrained_checkpoint:
        pretrained_checkpoint = os.fspath(
            Path(os.path.expandvars(pretrained_checkpoint)).expanduser()
        )
        if not Path(pretrained_checkpoint).is_absolute():
            raise ValueError("v5flow.pretrained_checkpoint must be absolute when set")
    pretrained_state_sha256 = v5_settings.get("pretrained_state_sha256", "")
    if not isinstance(pretrained_state_sha256, str):
        raise ValueError("v5flow.pretrained_state_sha256 must be a string")
    if bool(pretrained_checkpoint) != bool(pretrained_state_sha256):
        raise ValueError(
            "v5flow.pretrained_checkpoint and pretrained_state_sha256 must be "
            "set together"
        )
    if pretrained_state_sha256 and re.fullmatch(
        r"[0-9a-f]{64}", pretrained_state_sha256
    ) is None:
        raise ValueError(
            "v5flow.pretrained_state_sha256 must be a lowercase SHA256"
        )
    config.v5flow.pretrained_checkpoint = pretrained_checkpoint
    config.v5flow.pretrained_state_sha256 = pretrained_state_sha256
    if pretrained_checkpoint and not config.v5flow.enabled:
        raise ValueError("v5flow.pretrained_checkpoint requires v5flow.enabled=true")

    warmup_settings = v5_settings.get("warmup")
    if warmup_settings is None:
        warmup_settings = {
            "enabled": False,
            "export_only": False,
            "output_path": "",
            "seed": seed,
            "num_rollout_batches": 1,
            "num_holdout_batches": 1,
            "num_inner_epochs": 1,
            "num_train_timesteps": 1,
            "gradient_accumulation_steps": 1,
            "timestep_sampling": config.v5flow.timestep_sampling,
            "num_refreshes": 1,
            "checkpoint_root": "",
            "keep_checkpoints": 2,
            "prompt_pool": "test",
            "parameter_dtype": "model",
        }
    else:
        warmup_settings = _require_exact_keys(
            _require_mapping(warmup_settings, "v5flow.warmup"),
            "v5flow.warmup",
            _V5FLOW_WARMUP_KEYS,
            optional=_OPTIONAL_V5FLOW_WARMUP_KEYS,
        )
    config.v5flow.warmup = ml_collections.ConfigDict(warmup_settings)
    config.v5flow.warmup.enabled = _require_bool(
        warmup_settings["enabled"], "v5flow.warmup.enabled"
    )
    config.v5flow.warmup.export_only = _require_bool(
        warmup_settings["export_only"], "v5flow.warmup.export_only"
    )
    warmup_output_path = warmup_settings["output_path"]
    if not isinstance(warmup_output_path, str):
        raise ValueError("v5flow.warmup.output_path must be a string")
    if warmup_output_path:
        warmup_output_path = os.fspath(Path(warmup_output_path).expanduser())
        if not Path(warmup_output_path).is_absolute():
            raise ValueError("v5flow.warmup.output_path must be absolute when set")
    config.v5flow.warmup.output_path = warmup_output_path
    config.v5flow.warmup.seed = _require_int(
        warmup_settings["seed"], "v5flow.warmup.seed", minimum=0
    )
    config.v5flow.warmup.num_rollout_batches = _require_int(
        warmup_settings["num_rollout_batches"],
        "v5flow.warmup.num_rollout_batches",
    )
    config.v5flow.warmup.num_holdout_batches = _require_int(
        warmup_settings["num_holdout_batches"],
        "v5flow.warmup.num_holdout_batches",
    )
    config.v5flow.warmup.num_inner_epochs = _require_int(
        warmup_settings["num_inner_epochs"],
        "v5flow.warmup.num_inner_epochs",
    )
    config.v5flow.warmup.num_train_timesteps = _require_int(
        warmup_settings["num_train_timesteps"],
        "v5flow.warmup.num_train_timesteps",
    )
    config.v5flow.warmup.gradient_accumulation_steps = _require_int(
        warmup_settings["gradient_accumulation_steps"],
        "v5flow.warmup.gradient_accumulation_steps",
    )
    config.v5flow.warmup.timestep_sampling = _require_choice(
        warmup_settings["timestep_sampling"],
        "v5flow.warmup.timestep_sampling",
        {"rollout_schedule", "continuous_uniform"},
    )
    config.v5flow.warmup.num_refreshes = _require_int(
        warmup_settings.get("num_refreshes", 1),
        "v5flow.warmup.num_refreshes",
    )
    checkpoint_root = warmup_settings.get("checkpoint_root", "")
    if not isinstance(checkpoint_root, str):
        raise ValueError("v5flow.warmup.checkpoint_root must be a string")
    if checkpoint_root:
        checkpoint_root = os.fspath(Path(os.path.expandvars(checkpoint_root)).expanduser())
        if not Path(checkpoint_root).is_absolute():
            raise ValueError(
                "v5flow.warmup.checkpoint_root must be absolute when set"
            )
    config.v5flow.warmup.checkpoint_root = checkpoint_root
    config.v5flow.warmup.keep_checkpoints = _require_int(
        warmup_settings.get("keep_checkpoints", 2),
        "v5flow.warmup.keep_checkpoints",
        minimum=2,
    )
    config.v5flow.warmup.prompt_pool = _require_choice(
        warmup_settings.get("prompt_pool", "test"),
        "v5flow.warmup.prompt_pool",
        {"train", "test"},
    )
    config.v5flow.warmup.parameter_dtype = _require_choice(
        warmup_settings.get("parameter_dtype", "model"),
        "v5flow.warmup.parameter_dtype",
        {"model", "float32"},
    )
    if config.v5flow.warmup.enabled:
        if not config.v5flow.enabled:
            raise ValueError("v5flow.warmup.enabled requires v5flow.enabled=true")
        if (
            config.v5flow.warmup.num_rollout_batches
            % config.v5flow.warmup.gradient_accumulation_steps
            != 0
        ):
            raise ValueError(
                "v5flow.warmup.num_rollout_batches must divide by "
                "gradient_accumulation_steps"
            )
        if (
            config.v5flow.warmup.timestep_sampling == "rollout_schedule"
            and config.v5flow.warmup.num_train_timesteps > rollout_steps
        ):
            raise ValueError(
                "v5flow.warmup.num_train_timesteps exceeds the rollout schedule"
            )
    if initializer_path and config.v5flow.warmup.enabled:
        raise ValueError(
            "v5flow.initializer_path and v5flow.warmup.enabled are mutually exclusive"
        )
    if pretrained_checkpoint and initializer_path:
        raise ValueError(
            "v5flow pretrained checkpoint and initializer_path are mutually exclusive"
        )
    if pretrained_checkpoint and config.v5flow.warmup.enabled:
        raise ValueError(
            "v5flow pretrained checkpoint and warmup.enabled are mutually exclusive"
        )
    if config.v5flow.warmup.export_only:
        if not config.v5flow.warmup.enabled:
            raise ValueError("v5flow.warmup.export_only requires warmup.enabled=true")
        if not warmup_output_path:
            raise ValueError("v5flow.warmup.export_only requires an absolute output_path")
        if max_train_steps != 0:
            raise ValueError(
                "v5flow warm-up export requires training.max_train_steps=0"
            )
        if config.v5flow.warmup.num_refreshes > 1 and not checkpoint_root:
            raise ValueError(
                "multi-refresh v5flow warm-up requires an absolute checkpoint_root"
            )
    else:
        if warmup_output_path:
            raise ValueError(
                "v5flow.warmup.output_path is only valid for export_only warm-up"
            )
        if max_train_steps == 0:
            raise ValueError(
                "training.max_train_steps=0 is reserved for export-only v5 warm-up"
            )
        if checkpoint_root:
            raise ValueError(
                "v5flow.warmup.checkpoint_root is reserved for export-only warm-up"
            )
        if config.v5flow.warmup.num_refreshes != 1:
            raise ValueError(
                "multi-refresh v5flow warm-up is reserved for export-only pretraining"
            )

    if config.v5flow.enabled and cfg_mode not in {
        "cfg1",
        "cfg4_rollout",
        "cfg4_dual_shifted_center",
        "cfg4_full",
        "cfg4_alternating",
    }:
        raise ValueError(
            "v5flow supports only cfg1, cfg4_rollout, "
            "cfg4_dual_shifted_center, or cfg4_full/cfg4_alternating; analytic "
            "old-CFG-center, cfg1_dual, and cfg1_dropout objectives require "
            "v5flow.enabled=false"
        )
    if cfg_mode == "cfg4_alternating":
        if (not config.v5flow.enabled or not pretrained_checkpoint
                or config.v5flow.theta_lifecycle != "persistent"
                or config.v5flow.update_interval <= 0
                or gradient_steps_per_epoch != 1 or config.train.num_inner_epochs != 1
                or cfg_unconditional_source != "same_model"):
            raise ValueError("cfg4_alternating requires persistent pretrained V5, positive interval, "
                             "same_model, and exactly one conditional update per epoch")
    if cfg_mode == "cfg4_dual_shifted_center" and not pretrained_checkpoint:
        raise ValueError(
            "cfg4_dual_shifted_center requires v5flow.pretrained_checkpoint"
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
    if cfg_unconditional_source == "pretrained_base":
        if cfg_mode != "cfg4_rollout":
            raise ValueError(
                "cfg_unconditional_source='pretrained_base' requires "
                "cfg_mode='cfg4_rollout'"
            )
        if not config.v5flow.enabled:
            raise ValueError(
                "cfg_unconditional_source='pretrained_base' requires "
                "v5flow.enabled=true"
            )
    if initial_v5flow_checkpoint:
        if cfg_mode != "cfg1":
            raise ValueError(
                "V5-to-policy initialization requires cfg_mode='cfg1'"
            )
        if config.v5flow.enabled:
            raise ValueError(
                "V5-to-policy initialization requires v5flow to be disabled "
                "during reward finetuning"
            )
        if not bool(config.train.ema):
            raise ValueError("V5-to-policy initialization requires train.ema=true")
    evaluation = _require_mapping(settings["evaluation"], "evaluation")
    evaluation_keys = set(evaluation)
    if evaluation_keys == _LEGACY_EVALUATION_KEYS:
        evaluation = _require_exact_keys(
            evaluation,
            "evaluation",
            _LEGACY_EVALUATION_KEYS,
        )
    elif (
        _EXPLICIT_EVALUATION_KEYS - _OPTIONAL_EXPLICIT_EVALUATION_KEYS
    ) <= evaluation_keys <= _EXPLICIT_EVALUATION_KEYS:
        evaluation = _require_exact_keys(
            evaluation,
            "evaluation",
            _EXPLICIT_EVALUATION_KEYS,
            optional=_OPTIONAL_EXPLICIT_EVALUATION_KEYS,
        )
        eval_dataset = _require_choice(
            evaluation["dataset"],
            "evaluation.dataset",
            set(_EVAL_DATASET_SIZES),
        )
        expected_eval_dataset = _TASK_EVAL_DATASETS[task]
        if eval_dataset != expected_eval_dataset:
            raise ValueError(
                f"task={task!r} requires evaluation.dataset={expected_eval_dataset!r}, "
                f"got {eval_dataset!r}"
            )
        eval_reward_fn = _require_mapping(evaluation["reward_fn"], "evaluation.reward_fn")
        if not eval_reward_fn:
            raise ValueError("evaluation.reward_fn must not be empty")
        eval_reward_fn = {
            name: _require_finite(
                weight,
                f"evaluation.reward_fn.{name}",
                positive=True,
            )
            for name, weight in eval_reward_fn.items()
        }
        if any(weight != 1.0 for weight in eval_reward_fn.values()):
            raise ValueError(
                "evaluation.reward_fn values are metric selectors and must all equal 1.0"
            )
        if set(eval_reward_fn) != expected_rewards:
            raise ValueError(
                f"task={task!r} requires evaluation.reward_fn keys "
                f"{sorted(expected_rewards)}, got {sorted(eval_reward_fn)}"
            )
        evaluation["reward_fn"] = eval_reward_fn
        evaluation["batch_size"] = _require_int(
            evaluation["batch_size"], "evaluation.batch_size"
        )
        evaluation["expected_samples"] = _require_int(
            evaluation["expected_samples"], "evaluation.expected_samples"
        )
        expected_samples = _EVAL_DATASET_SIZES[eval_dataset]
        if evaluation["expected_samples"] != expected_samples:
            raise ValueError(
                f"evaluation.expected_samples must be {expected_samples} for "
                f"dataset={eval_dataset!r}"
            )
    else:
        missing_legacy = sorted(_LEGACY_EVALUATION_KEYS - evaluation_keys)
        unknown = sorted(evaluation_keys - _EXPLICIT_EVALUATION_KEYS)
        raise ValueError(
            "Invalid evaluation keys; use either the legacy or explicit schema; "
            f"missing_legacy={missing_legacy}, unknown={unknown}"
        )
    if not isinstance(evaluation.get("auto_submit"), bool):
        raise ValueError("evaluation.auto_submit must be boolean")
    if not isinstance(evaluation.get("save_images"), bool):
        raise ValueError("evaluation.save_images must be boolean")
    guidance_scales = evaluation.get("guidance_scales", [1.0])
    if not isinstance(guidance_scales, list) or not guidance_scales:
        raise ValueError("evaluation.guidance_scales must be a nonempty JSON list")
    normalized_guidance_scales = []
    for index, guidance_scale in enumerate(guidance_scales):
        guidance_scale = _require_finite(
            guidance_scale,
            f"evaluation.guidance_scales[{index}]",
            positive=True,
        )
        if guidance_scale not in {1.0, 4.0}:
            raise ValueError("evaluation.guidance_scales entries must be 1.0 or 4.0")
        if guidance_scale in normalized_guidance_scales:
            raise ValueError("evaluation.guidance_scales must not contain duplicates")
        normalized_guidance_scales.append(guidance_scale)
    normalized_guidance_scales.sort()
    if (
        cfg_unconditional_source == "pretrained_base"
        and normalized_guidance_scales != [4.0]
    ):
        raise ValueError(
            "cfg_unconditional_source='pretrained_base' requires "
            "evaluation.guidance_scales=[4.0] so every intermediate and final "
            "evaluation uses the fixed pretrained-empty CFG contract"
        )
    evaluation["guidance_scales"] = normalized_guidance_scales
    config.inline_eval_max_samples = _require_int(
        evaluation.get("inline_max_samples", 0),
        "evaluation.inline_max_samples",
        minimum=0,
    )
    config.inline_eval_at_final = _require_bool(
        evaluation.get("inline_at_final", False),
        "evaluation.inline_at_final",
    )
    if config.inline_eval_at_final:
        if eval_freq <= 0:
            raise ValueError(
                "evaluation.inline_at_final=true requires training.eval_freq > 0"
            )
        if max_train_steps <= 0 or max_train_steps % eval_freq != 0:
            raise ValueError(
                "evaluation.inline_at_final=true requires a positive "
                "training.max_train_steps divisible by training.eval_freq"
            )
        if evaluation["auto_submit"]:
            raise ValueError(
                "evaluation.inline_at_final=true requires "
                "evaluation.auto_submit=false"
            )
    if config.v5flow.warmup.export_only and (
        evaluation["auto_submit"] or evaluation["save_images"]
    ):
        raise ValueError(
            "export-only v5 warm-up cannot auto-submit evaluation or save eval images"
        )
    config.v5flow_evaluation = ml_collections.ConfigDict(evaluation)
    config.eval_guidance_scales = normalized_guidance_scales
    return config
