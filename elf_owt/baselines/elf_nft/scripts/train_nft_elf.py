#!/usr/bin/env python
"""Online SC-CFG DiffusionNFT / flow-center training for ELF text tasks."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import jax_utils
from flax.training.common_utils import shard
from torch.utils.data import Subset
from transformers import AutoTokenizer

from baselines.elf_nft.checkpointing import (
    build_checkpoint_payload,
    load_online_checkpoint,
    save_online_checkpoint,
    validate_resume_metadata,
)
from baselines.elf_nft.config import load_experiment_config
from baselines.elf_nft.full_model import (
    PARAMETERIZATION,
    frozen_decoder_delta_norm,
    parameter_counts,
    restore_frozen_decoder_params,
    validate_full_model_params,
)
from baselines.elf_nft.nft_math import old_policy_decay
from baselines.elf_nft.source_cfg import training_input_cfg_scale
from baselines.elf_nft.wmt_evaluation import evaluate_wmt_policy
from baselines.elf_nft.xsum_data import (
    XSUM_CENTER_PRETRAIN_SELECTION_SEED_TAG,
    load_pinned_xsum_datasets,
    preflight_xsum_datasets,
    save_xsum_preflight,
    xsum_prompt_batch,
    xsum_settings_from_config,
)
from baselines.elf_nft.xsum_evaluation import (
    evaluate_xsum_policy,
    validate_xsum_evaluation_examples,
)
from baselines.elf_nft.reward import (
    CausalLMLogLikelihoodReward,
    entropy_weight_from_state,
    make_entropy_constraint_state,
    update_entropy_constraint_state,
)
from baselines.elf_nft.rollout import (
    collect_old_policy_rollout,
    evaluate_owt_policy,
    make_rollout_functions,
    resolve_rollout_sccfg_settings,
)
from baselines.elf_nft.train_steps import (
    CENTER_ACCUMULATED_METRIC_NAMES,
    POLICY_ACCUMULATED_METRIC_NAMES,
    accumulate_center_gradients,
    accumulate_policy_gradients,
    apply_center_gradients,
    apply_policy_gradients,
    create_center_state,
    create_model_state,
    finalize_gradient_accumulator,
    initialize_gradient_accumulator,
    theta_ema_effective_decay,
    updates_empty_branch,
)
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import load_encoder_checkpoint, load_pretrained_params
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split
from utils.logging_utils import log_for_0


LOGGER = logging.getLogger("elf_nft")
FULL_EVALUATION_SEED_TAG = 0x4556414C
WMT_TRAIN_SELECTION_SEED_TAG = 0x454C46
WMT_CENTER_PRETRAIN_SELECTION_SEED_TAG = 0x43505245


@dataclass(frozen=True)
class FullEvaluationPlan:
    outer_step: int
    num_samples: int
    generation_batch_size: int
    seed: int


def _build_midpoint_full_evaluation_plan(elf_config, nft_config):
    """Return the independent OWT midpoint-evaluation plan, if applicable."""

    if (nft_config.task != "owt" or nft_config.max_outer_steps < 2
            or not nft_config.midpoint_evaluation):
        return None
    num_samples = getattr(elf_config, "num_samples", None)
    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, int)
        or num_samples <= 0
    ):
        raise ValueError("OWT full evaluation requires elf.num_samples > 0")
    generation_batch_size = int(nft_config.rollout_generation_batch_size)
    if num_samples % generation_batch_size:
        raise ValueError(
            "elf.num_samples must be divisible by the rollout generation batch size"
        )
    seed = int(
        np.random.SeedSequence([nft_config.seed, FULL_EVALUATION_SEED_TAG])
        .generate_state(1, dtype=np.uint32)[0]
    )
    return FullEvaluationPlan(
        outer_step=nft_config.max_outer_steps // 2,
        num_samples=num_samples,
        generation_batch_size=generation_batch_size,
        seed=seed,
    )


def _checkpoint_due(completed_outer, nft_config, full_evaluation_plan=None):
    """Return whether this completed outer step must be checkpointed."""

    return (
        (getattr(nft_config, "checkpoint_rolling_every", 0) > 0
         and completed_outer % nft_config.checkpoint_rolling_every == 0)
        or _wmt_evaluation_due(completed_outer, nft_config)
        or _owt_sccfg_evaluation_due(completed_outer, nft_config)
        or completed_outer % nft_config.save_every == 0
        or completed_outer % nft_config.keep_every_n_steps == 0
        or completed_outer == nft_config.max_outer_steps
        or (
            full_evaluation_plan is not None
            and completed_outer == full_evaluation_plan.outer_step
        )
    )


def _wmt_evaluation_due(step, nft_config):
    """Evaluate scheduled conditional-task checkpoints.

    The historical name remains as a compatibility surface for tests and
    callers, while schema-11 XSum uses the same exact 0/50/100 schedule.
    """

    if getattr(nft_config, "task", None) not in {"wmt", "xsum"} or step <= 0:
        return False
    every = getattr(nft_config, "evaluation_every_n_steps", 0)
    return ((every > 0 and step % every == 0)
            or (nft_config.evaluation_at_end and step == nft_config.max_outer_steps))


def _owt_sccfg_evaluation_due(step, nft_config):
    """Return whether an OWT checkpoint needs its SCCFG evaluation suite."""

    if getattr(nft_config, "task", None) != "owt" or step <= 0:
        return False
    every = getattr(nft_config, "evaluation_every_n_steps", 0)
    return ((every > 0 and step % every == 0)
            or (nft_config.evaluation_at_end and step == nft_config.max_outer_steps))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train ELF SC-CFG with standard, learned, or recovered NFT centers"
    )
    parser.add_argument(
        "--config", required=True, help="ELF NFT experiment YAML or JSON"
    )
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument(
        "--owt", dest="task", action="store_const", const="owt",
        help="Run the null-conditioned OpenWebText branch",
    )
    task.add_argument(
        "--wmt", dest="task", action="store_const", const="wmt",
        help="Run the conditional WMT14 De-En branch",
    )
    task.add_argument(
        "--xsum", dest="task", action="store_const", const="xsum",
        help="Run the conditional XSum summarization branch",
    )
    parser.add_argument(
        "--method",
        choices=("standard", "shifted_center", "recovered_center"),
        default=None,
        help="Override the method declared in the experiment configuration",
    )
    parser.add_argument(
        "--self_condition_mode", choices=("bernoulli", "conditional_only", "paired"),
        default=None, help="SC loss routing; paired sums both branch losses before one update",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Override output directory; must remain under the group asset root",
    )
    parser.add_argument(
        "--max_outer_steps", type=int, default=None,
        help="Override the total number of online outer rounds",
    )
    parser.add_argument(
        "--samples_per_prompt", type=int, default=None,
        help="Capacity-test override for samples per rollout group",
    )
    parser.add_argument(
        "--rollout_generation_batch_size", type=int, default=None,
        help="Capacity-test override for each pmapped rollout-generation batch",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=None,
        help="Capacity-test override for the total pmapped training batch",
    )
    parser.add_argument(
        "--reward_batch_size", type=int, default=None,
        help="Capacity-test override for the total causal-LM scoring batch",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Full-model ELF NFT checkpoint file or directory to resume",
    )
    parser.add_argument(
        "--capacity_probe", action="store_true",
        help=(
            "Run a non-formal one-step memory probe without publishing its "
            "throwaway final checkpoint"
        ),
    )
    validation_mode = parser.add_mutually_exclusive_group()
    validation_mode.add_argument(
        "--validate_only", action="store_true",
        help="Validate configuration and device geometry without loading assets",
    )
    validation_mode.add_argument(
        "--data_preflight_only",
        action="store_true",
        help=(
            "Load pinned conditional-task snapshots and validate schema/token "
            "alignment without loading model weights"
        ),
    )
    return parser.parse_args(argv)


def _configure_logging():
    logging.basicConfig(
        format="%(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
        force=True,
    )
    for name in ("absl", "orbax", "tensorstore", "flax.training.checkpoints"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _validate_device_geometry(nft_config):
    devices = jax.local_device_count()
    if devices <= 0:
        raise RuntimeError("JAX reported no local devices")
    values = {
        "rollout_batch_size": nft_config.rollout_batch_size,
        "rollout_generation_batch_size": (
            nft_config.rollout_generation_batch_size
        ),
        "train_batch_size": nft_config.train_batch_size,
    }
    if nft_config.task in {"wmt", "xsum"}:
        values["prompt_groups_per_outer"] = nft_config.prompt_groups_per_outer
    invalid = {name: value for name, value in values.items() if value % devices}
    if invalid:
        raise ValueError(
            f"batch geometry must divide over {devices} local devices: {invalid}"
        )


def _validate_task_selection(selected_task, nft_config):
    if selected_task != nft_config.task:
        raise ValueError(
            f"--{selected_task} does not match config task={nft_config.task!r}"
        )


def _build_model_and_pretrained_params(elf_config, nft_config, tokenizer, rng):
    encoder_config, encoder_model, _ = get_encoder(
        elf_config.encoder_model_name,
        jnp.float32,
        revision=getattr(elf_config, "tokenizer_revision", None),
    )
    rng, init_rng = jax.random.split(rng)
    input_dim = (
        2 * encoder_config.d_model
        if elf_config.self_cond_prob > 0 else encoder_config.d_model
    )
    dummy_x = jnp.ones((1, elf_config.max_length, input_dim), dtype=jnp.float32)
    dummy_t = jnp.ones((1,), dtype=jnp.float32)
    dummy_sc = (
        jnp.ones((1,), dtype=jnp.float32)
        if elf_config.num_self_cond_cfg_tokens > 0 else None
    )
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    model = ELF_models[elf_config.model](
        text_encoder_dim=encoder_config.d_model,
        max_length=elf_config.max_length,
        attn_drop=elf_config.attn_dropout,
        proj_drop=elf_config.proj_dropout,
        num_time_tokens=elf_config.num_time_tokens,
        num_self_cond_cfg_tokens=elf_config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=elf_config.num_model_mode_tokens,
        bottleneck_dim=elf_config.bottleneck_dim,
    )
    initialized = model.init(
        init_rng,
        x=dummy_x,
        t=dummy_t,
        deterministic=True,
        self_cond_cfg_scale=dummy_sc,
    )
    base_params, pretrained_step = load_pretrained_params(
        nft_config.pretrained_checkpoint,
        initialized["params"],
        revision=nft_config.pretrained_checkpoint_revision,
    )
    # ``load_pretrained_params`` selects the released EMA when available.
    del initialized
    return model, encoder_config, encoder_model, base_params, pretrained_step, rng


def _wmt_reference_settings(elf_config):
    """Return and validate pinned tokenized/raw WMT dataset settings."""

    names = (
        "wmt_tokenized_data_revision",
        "wmt_validation_data_revision",
        "wmt_reference_data_path",
        "wmt_reference_data_name",
        "wmt_reference_data_split",
        "wmt_reference_data_revision",
        "wmt_source_language",
        "wmt_target_language",
    )
    settings = {name: getattr(elf_config, name, None) for name in names}
    invalid = [
        name
        for name, value in settings.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid:
        raise ValueError(
            "WMT original-reference configuration is missing nonempty fields: "
            f"{invalid}"
        )
    if settings["wmt_source_language"] == settings["wmt_target_language"]:
        raise ValueError("WMT source and target reference languages must differ")
    return settings


def _wmt_prompt_indices(
    dataset_size,
    *,
    outer_step,
    nft_config,
    selection_seed_tag=WMT_TRAIN_SELECTION_SEED_TAG,
):
    """Return the exact deterministic WMT row selection for one round."""

    if nft_config.prompt_groups_per_outer > dataset_size:
        raise ValueError("prompt_groups_per_outer exceeds the training dataset")
    selection_rng = np.random.default_rng(
        np.random.SeedSequence([nft_config.seed, outer_step, selection_seed_tag])
    )
    return selection_rng.choice(
        dataset_size,
        size=nft_config.prompt_groups_per_outer,
        replace=False,
    ).tolist()


def _wmt_translation_pair(row, *, index, source_language, target_language):
    """Extract one verbatim source/target pair from a raw WMT translation row."""

    if not isinstance(row, Mapping):
        raise ValueError(f"WMT reference row {index} is not a mapping")
    translation = row.get("translation")
    if not isinstance(translation, Mapping):
        raise ValueError(f"WMT reference row {index} has no translation mapping")
    source = translation.get(source_language)
    target = translation.get(target_language)
    if not isinstance(source, str) or not isinstance(target, str):
        raise ValueError(
            f"WMT reference row {index} lacks string languages "
            f"{source_language!r}/{target_language!r}"
        )
    return source, target


def _token_rows(tokenizer, texts):
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    rows = encoded.get("input_ids")
    if rows is None or len(rows) != len(texts):
        raise ValueError("WMT reference tokenizer returned an invalid batch")
    return [[int(token) for token in row] for row in rows]


def _materialize_aligned_wmt_examples(
    dataset,
    reference_dataset,
    indices,
    *,
    tokenizer,
    source_language,
    target_language,
):
    """Attach raw references after proving row-level token alignment.

    Only the selected examples are materialized.  Both sides of each raw WMT
    pair are tokenized without truncation and must exactly match the existing
    pre-tokenized training row before its verbatim target can become ``target``.
    """

    if len(dataset) != len(reference_dataset):
        raise ValueError(
            "WMT tokenized/reference dataset lengths differ: "
            f"{len(dataset)} != {len(reference_dataset)}"
        )
    examples = []
    source_texts = []
    target_texts = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError(f"WMT dataset index must be an integer, got {index!r}")
        index = int(index)
        if not 0 <= index < len(dataset):
            raise IndexError(f"WMT dataset index out of range: {index}")
        examples.append(dict(dataset[index]))
        source, target = _wmt_translation_pair(
            reference_dataset[index],
            index=index,
            source_language=source_language,
            target_language=target_language,
        )
        source_texts.append(source)
        target_texts.append(target)

    encoded_sources = _token_rows(tokenizer, source_texts)
    encoded_targets = _token_rows(tokenizer, target_texts)
    for offset, index in enumerate(indices):
        expected_source = np.asarray(
            examples[offset].get("condition_input_ids")
        )
        expected_target = np.asarray(examples[offset].get("input_ids"))
        if expected_source.ndim != 1 or expected_target.ndim != 1:
            raise ValueError(
                f"WMT tokenized row {int(index)} lacks one-dimensional token ids"
            )
        if expected_source.tolist() != encoded_sources[offset]:
            raise ValueError(
                f"WMT source/reference token alignment failed at row {int(index)}"
            )
        if expected_target.tolist() != encoded_targets[offset]:
            raise ValueError(
                f"WMT target/reference token alignment failed at row {int(index)}"
            )
        examples[offset]["index"] = int(index)
        examples[offset]["input"] = source_texts[offset]
        examples[offset]["target"] = target_texts[offset]
    return examples


def _validate_wmt_reference_dataset(
    dataset,
    reference_dataset,
    *,
    tokenizer,
    nft_config,
    elf_config,
    pad_token_id,
    source_language,
    target_language,
):
    """Prove alignment for anchors and every row this run will select.

    The preflight intentionally evaluates the deterministic selections for all
    configured policy rounds and, when enabled, all center-pretraining rounds.
    It runs before model weights are loaded so a dataset contract failure cannot
    consume an expensive rollout or optimizer step.
    """

    if not len(dataset):
        raise ValueError("WMT training dataset is empty")
    anchors = sorted({0, len(dataset) // 2, len(dataset) - 1})
    phases = [
        (
            "policy",
            nft_config.max_outer_steps,
            WMT_TRAIN_SELECTION_SEED_TAG,
        )
    ]
    if nft_config.center_pretrain_steps:
        phases.append(
            (
                "center_pretrain",
                nft_config.center_pretrain_steps,
                WMT_CENTER_PRETRAIN_SELECTION_SEED_TAG,
            )
        )
    selected = []
    phase_rows = {}
    for phase, rounds, seed_tag in phases:
        phase_indices = []
        for outer_step in range(rounds):
            phase_indices.extend(
                _wmt_prompt_indices(
                    len(dataset),
                    outer_step=outer_step,
                    nft_config=nft_config,
                    selection_seed_tag=seed_tag,
                )
            )
        phase_rows[phase] = len(phase_indices)
        selected.extend(phase_indices)

    checked = sorted(set(anchors + selected))
    chunk_size = max(1, min(512, nft_config.prompt_groups_per_outer * 4))
    for start in range(0, len(checked), chunk_size):
        _materialize_aligned_wmt_examples(
            dataset,
            reference_dataset,
            checked[start : start + chunk_size],
            tokenizer=tokenizer,
            source_language=source_language,
            target_language=target_language,
        )

    collated_phases = []
    for phase, rounds, seed_tag in phases:
        if rounds <= 0:
            continue
        batch, indices = _prompt_batch(
            dataset,
            outer_step=0,
            nft_config=nft_config,
            elf_config=elf_config,
            pad_token_id=pad_token_id,
            reference_dataset=reference_dataset,
            tokenizer=tokenizer,
            source_language=source_language,
            target_language=target_language,
            selection_seed_tag=seed_tag,
        )
        if batch.get("index") != indices or len(batch.get("target", ())) != len(
            indices
        ):
            raise ValueError(
                f"WMT {phase} collator did not preserve aligned index/target rows"
            )
        collated_phases.append(phase)

    report = {
        "schema_version": 1,
        "dataset_rows": len(dataset),
        "reference_rows": len(reference_dataset),
        "anchor_indices": anchors,
        "scheduled_row_checks": sum(phase_rows.values()),
        "unique_aligned_rows": len(checked),
        "phase_row_checks": phase_rows,
        "collated_phases": collated_phases,
        "prompt_groups_per_outer": nft_config.prompt_groups_per_outer,
        "policy_rounds": nft_config.max_outer_steps,
        "center_pretrain_rounds": nft_config.center_pretrain_steps,
    }
    log_for_0(
        "Verified original WMT reference alignment for all scheduled rows: "
        f"{report}"
    )
    return report


def _prompt_batch(
    dataset,
    *,
    outer_step,
    nft_config,
    elf_config,
    pad_token_id,
    reference_dataset=None,
    tokenizer=None,
    source_language=None,
    target_language=None,
    selection_seed_tag=WMT_TRAIN_SELECTION_SEED_TAG,
):
    # Outer-indexed selection makes data choice reproducible across exact resume.
    indices = _wmt_prompt_indices(
        len(dataset),
        outer_step=outer_step,
        nft_config=nft_config,
        selection_seed_tag=selection_seed_tag,
    )
    if reference_dataset is None:
        subset = Subset(dataset, indices)
    else:
        if tokenizer is None or source_language is None or target_language is None:
            raise ValueError(
                "WMT reference injection requires tokenizer and language names"
            )
        subset = _materialize_aligned_wmt_examples(
            dataset,
            reference_dataset,
            indices,
            tokenizer=tokenizer,
            source_language=source_language,
            target_language=target_language,
        )
    loader = get_dataloader(
        subset,
        batch_size=nft_config.prompt_groups_per_outer,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        max_seq_length=elf_config.max_length,
        pad_token_id=pad_token_id,
        max_input_seq_length=elf_config.max_input_length,
        distributed=False,
    )
    return next(iter(loader)), indices


def _task_prompt_batch(
    dataset,
    *,
    outer_step,
    nft_config,
    elf_config,
    pad_token_id,
    reference_dataset=None,
    reference_settings=None,
    tokenizer=None,
    center_pretrain=False,
):
    """Dispatch deterministic conditional prompt collation by task."""

    if nft_config.task == "xsum":
        seed_tag = (
            XSUM_CENTER_PRETRAIN_SELECTION_SEED_TAG
            if center_pretrain
            else None
        )
        kwargs = {}
        if seed_tag is not None:
            kwargs["selection_seed_tag"] = seed_tag
        return xsum_prompt_batch(
            dataset,
            outer_step=outer_step,
            nft_config=nft_config,
            elf_config=elf_config,
            pad_token_id=pad_token_id,
            **kwargs,
        )
    if nft_config.task != "wmt":
        raise ValueError("prompt batches require WMT or XSum")
    if reference_settings is None:
        raise ValueError("WMT prompt batching requires reference settings")
    return _prompt_batch(
        dataset,
        outer_step=outer_step,
        nft_config=nft_config,
        elf_config=elf_config,
        pad_token_id=pad_token_id,
        reference_dataset=reference_dataset,
        tokenizer=tokenizer,
        source_language=reference_settings["wmt_source_language"],
        target_language=reference_settings["wmt_target_language"],
        selection_seed_tag=(
            WMT_CENTER_PRETRAIN_SELECTION_SEED_TAG
            if center_pretrain
            else WMT_TRAIN_SELECTION_SEED_TAG
        ),
    )


def _load_and_preflight_wmt_training_data(
    elf_config,
    nft_config,
    tokenizer,
    *,
    pad_token_id,
):
    """Load both pinned train views and validate their complete run selection."""

    settings = _wmt_reference_settings(elf_config)
    dataset_cache = os.environ.get("HF_DATASETS_CACHE")
    train_dataset = load_dataset_split(
        elf_config.data_path,
        dataset_cache_dir=dataset_cache,
        revision=settings["wmt_tokenized_data_revision"],
    )
    reference_dataset = load_dataset_split(
        settings["wmt_reference_data_path"],
        dataset_cache_dir=dataset_cache,
        name=settings["wmt_reference_data_name"],
        split=settings["wmt_reference_data_split"],
        revision=settings["wmt_reference_data_revision"],
        numpy_format=False,
    )
    report = _validate_wmt_reference_dataset(
        train_dataset,
        reference_dataset,
        tokenizer=tokenizer,
        nft_config=nft_config,
        elf_config=elf_config,
        pad_token_id=pad_token_id,
        source_language=settings["wmt_source_language"],
        target_language=settings["wmt_target_language"],
    )
    report.update(
        {
            "tokenized_training_dataset": {
                "path": elf_config.data_path,
                "revision": settings["wmt_tokenized_data_revision"],
            },
            "tokenized_validation_dataset": {
                "path": elf_config.eval_data_path,
                "revision": settings["wmt_validation_data_revision"],
            },
            "reference_dataset": {
                "path": settings["wmt_reference_data_path"],
                "name": settings["wmt_reference_data_name"],
                "split": settings["wmt_reference_data_split"],
                "revision": settings["wmt_reference_data_revision"],
                "source_language": settings["wmt_source_language"],
                "target_language": settings["wmt_target_language"],
            },
        }
    )
    return train_dataset, reference_dataset, settings, report


def _iterate_minibatches(buffer, batch_size, permutation_rng):
    train_keys = (
        "endpoints", "cond_seq", "cond_seq_mask", "loss_mask",
        "decoder_targets", "advantages",
    )
    total = int(buffer["endpoints"].shape[0])
    if total % batch_size:
        raise ValueError("sealed rollout buffer does not divide into train batches")
    permutation = np.asarray(jax.random.permutation(permutation_rng, total))
    for start in range(0, total, batch_size):
        indices = permutation[start : start + batch_size]
        yield {
            key: jnp.asarray(buffer[key][indices])
            for key in train_keys
        }


def _host_metrics(replicated_metrics):
    metrics = jax.device_get(jax_utils.unreplicate(replicated_metrics))
    return {name: float(value) for name, value in metrics.items()}


def _host_scalar(replicated_value):
    return jax.device_get(jax_utils.unreplicate(replicated_value)).item()


def _validate_accumulated_batch(
    *, stage, actual_records, expected_records, actual_microbatches,
    expected_microbatches,
):
    if actual_records != expected_records:
        raise RuntimeError(
            f"{stage} accumulated {actual_records} records; "
            f"expected {expected_records}"
        )
    if actual_microbatches != expected_microbatches:
        raise RuntimeError(
            f"{stage} ran {actual_microbatches} backward microbatches; "
            f"expected {expected_microbatches}"
        )


def _apply_center_effective_batch(
    *,
    center_state,
    old_params,
    rollout_buffer,
    rng,
    nft_config,
    p_initialize_center_accumulator,
    p_accumulate_center,
    p_finalize_accumulator,
    p_apply_center,
):
    """Apply one accumulated center update and return host-side diagnostics."""

    center_step_before = int(
        jax.device_get(jax_utils.unreplicate(center_state).step)
    )
    center_accumulator = p_initialize_center_accumulator(center_state.params)
    center_backward_microbatches = 0
    for _ in range(nft_config.center_num_inner_epochs):
        rng, permutation_rng = jax.random.split(rng)
        for minibatch in _iterate_minibatches(
            rollout_buffer, nft_config.train_batch_size, permutation_rng
        ):
            for _ in range(nft_config.timesteps_per_batch):
                rng, step_rng = jax.random.split(rng)
                device_rngs = jax.random.split(
                    step_rng, jax.local_device_count()
                )
                center_accumulator = p_accumulate_center(
                    center_accumulator,
                    center_state.params,
                    old_params,
                    shard(minibatch),
                    device_rngs,
                )
                center_backward_microbatches += 1
    (
        center_mean_gradients,
        center_mean_metrics,
        center_gradient_norm,
        center_record_count,
    ) = p_finalize_accumulator(center_accumulator)
    center_record_count = int(_host_scalar(center_record_count))
    _validate_accumulated_batch(
        stage="center",
        actual_records=center_record_count,
        expected_records=nft_config.center_training_records_per_outer,
        actual_microbatches=center_backward_microbatches,
        expected_microbatches=nft_config.center_backward_microbatches_per_outer,
    )
    center_host_metrics = _host_metrics(center_mean_metrics)
    center_host_metrics["center_grad_norm"] = float(
        _host_scalar(center_gradient_norm)
    )
    _require_finite_metrics(
        center_host_metrics, stage="center effective batch"
    )
    center_state, center_update_metrics = p_apply_center(
        center_state, center_mean_gradients
    )
    center_update_metrics = _host_metrics(center_update_metrics)
    _require_finite_metrics(
        center_update_metrics, stage="center optimizer update"
    )
    center_host_metrics.update(center_update_metrics)
    center_step = int(
        jax.device_get(jax_utils.unreplicate(center_state).step)
    )
    if center_step != center_step_before + 1:
        raise RuntimeError(
            "center effective batch must produce exactly one optimizer update: "
            f"{center_step_before} -> {center_step}"
        )
    return (
        center_state,
        rng,
        center_host_metrics,
        center_backward_microbatches,
        center_record_count,
        center_step,
    )


def _require_finite_metrics(metrics, *, stage: str):
    non_finite = {
        name: value for name, value in metrics.items() if not np.isfinite(value)
    }
    if non_finite:
        raise FloatingPointError(
            f"{stage} produced non-finite metrics: {non_finite}"
        )


def _require_frozen_decoder_reference(reference_params, named_params):
    for name, params in named_params.items():
        if params is None:
            continue
        delta = float(
            jax.device_get(frozen_decoder_delta_norm(reference_params, params))
        )
        if delta != 0.0:
            raise ValueError(
                f"{name} decoder-specific parameters differ from pretrained reference: "
                f"delta_norm={delta}"
            )


def _log_device_memory(label: str):
    for device in jax.local_devices():
        try:
            stats = device.memory_stats()
        except Exception as error:
            log_for_0(
                f"{label} device={device.id} memory statistics unavailable: {error}",
                level=logging.WARNING,
            )
            continue
        if not stats:
            continue
        selected = {
            key: int(stats[key])
            for key in (
                "bytes_in_use", "peak_bytes_in_use", "largest_alloc_size",
                "bytes_limit",
            )
            if key in stats
        }
        log_for_0(f"{label} device={device.id} memory_bytes={selected}")


def _append_jsonl(path: Path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _reconcile_jsonl_to_checkpoint(path: Path, completed_outer: int, *, group_by_case=False):
    """Keep checkpoint-covered records, optionally keyed by case and seed.

    Case mode retains pretrained step-zero evaluations and all distinct cases
    at one step. Seeded cases use ``(step, case, seed)`` so a resume preserves
    every completed replicate; historical unseeded rows keep the original
    ``(step, case)`` key. Both modes drop a truncated final line and keep the
    last duplicate, then replace the JSONL atomically; malformed interior lines
    fail.
    """

    if not path.is_file():
        return 0, 0
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    retained = {}
    dropped = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            dropped += 1
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            if any(remaining.strip() for remaining in lines[line_number:]):
                raise ValueError(
                    f"invalid non-terminal JSONL record in {path}:{line_number}"
                ) from error
            dropped += 1
            continue
        outer_step = record.get("outer_step")
        if isinstance(outer_step, bool) or not isinstance(outer_step, int):
            raise ValueError(f"missing integer outer_step in {path}:{line_number}")
        if not (0 if group_by_case else 1) <= outer_step <= completed_outer:
            dropped += 1
            continue
        if group_by_case and not isinstance(record.get("name"), str):
            raise ValueError(f"missing evaluation case name in {path}:{line_number}")
        if group_by_case:
            seed = record.get("seed")
            if seed is not None and (
                isinstance(seed, bool) or not isinstance(seed, int)
            ):
                raise ValueError(
                    f"invalid evaluation seed in {path}:{line_number}"
                )
            key = (
                (outer_step, record["name"])
                if seed is None
                else (outer_step, record["name"], seed)
            )
        else:
            key = outer_step
        if key in retained:
            dropped += 1
        retained[key] = record

    temporary = path.with_name(f".{path.name}.resume.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(retained):
            handle.write(
                json.dumps(
                    retained[key], ensure_ascii=False, sort_keys=True
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return len(retained), dropped


def _reconcile_center_pretrain_jsonl(
    path: Path, completed_pretrain: int
) -> tuple[int, int]:
    """Trim center-pretrain metrics to an atomically published milestone."""

    if not path.is_file():
        return 0, 0
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    retained = {}
    dropped = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            dropped += 1
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            if any(remaining.strip() for remaining in lines[line_number:]):
                raise ValueError(
                    f"invalid non-terminal JSONL record in {path}:{line_number}"
                ) from error
            dropped += 1
            continue
        step = record.get("center_pretrain_step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(
                f"missing integer center_pretrain_step in {path}:{line_number}"
            )
        if not 1 <= step <= completed_pretrain:
            dropped += 1
            continue
        if step in retained:
            dropped += 1
        retained[step] = record
    temporary = path.with_name(f".{path.name}.resume.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for step in sorted(retained):
            handle.write(
                json.dumps(
                    retained[step], ensure_ascii=False, sort_keys=True
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return len(retained), dropped


def _scheduled_full_evaluation_steps(path: Path) -> set[int]:
    """Return scheduled midpoint steps already represented by valid records."""

    if not path.is_file():
        return set()
    scheduled_steps = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid full-evaluation JSONL record in {path}:{line_number}"
                ) from error
            scheduled_outer = record.get(
                "scheduled_outer_step", record.get("outer_step")
            )
            if isinstance(scheduled_outer, bool) or not isinstance(
                scheduled_outer, int
            ):
                raise ValueError(
                    f"missing integer scheduled_outer_step in {path}:{line_number}"
                )
            scheduled_steps.add(scheduled_outer)
    return scheduled_steps


def _run_owt_full_evaluation(
    *,
    plan,
    actual_outer_step,
    trigger,
    functions,
    policy_params_replicated,
    tokenizer,
    reward_scorer,
    embedding_dim,
    elf_config,
    nft_config,
    output_path,
    evaluation_case=None,
    policy_source="current_policy_checkpoint",
):
    """Run and persist one independent fixed-seed current-policy evaluation."""

    evaluation_start = time.time()
    result = evaluate_owt_policy(
        functions=functions,
        policy_params_replicated=policy_params_replicated,
        tokenizer=tokenizer,
        reward_scorer=reward_scorer,
        embedding_dim=embedding_dim,
        rng=jax.random.PRNGKey(plan.seed),
        elf_config=elf_config,
        nft_config=nft_config,
        num_samples=plan.num_samples,
        generation_batch_size=plan.generation_batch_size,
    )
    _require_finite_metrics(result.metrics, stage="OWT full evaluation")
    record = {
        "outer_step": int(actual_outer_step),
        "scheduled_outer_step": int(plan.outer_step),
        "evaluation_type": "midpoint_full" if evaluation_case is None else "sccfg_sweep",
        "trigger": str(trigger),
        "policy_source": policy_source,
        "num_samples": int(plan.num_samples),
        "generation_batch_size": int(plan.generation_batch_size),
        "seed": int(plan.seed),
        "elapsed_seconds": time.time() - evaluation_start,
        **result.metrics,
        "hypothesis_previews": result.hypotheses[:4],
    }
    if evaluation_case is not None:
        record.update(asdict(evaluation_case))
    _append_jsonl(output_path, record)
    log_for_0(
        f"FULL_EVALUATION outer_step={actual_outer_step} "
        f"scheduled_outer_step={plan.outer_step} trigger={trigger} "
        f"policy={policy_source} samples={plan.num_samples} "
        f"case={None if evaluation_case is None else evaluation_case.name} "
        f"seed={plan.seed} ppl={record['ppl']:.6f} "
        f"entropy={record['entropy']:.6f} "
        f"elapsed_seconds={record['elapsed_seconds']:.3f} "
        f"record={output_path}"
    )
    return record


def _run_sccfg_evaluation_suite(
    *, model_apply_fn, policy_params_replicated, outer_step, tokenizer,
    reward_scorer, embedding_dim, elf_config, nft_config, output_path,
):
    """Evaluate all configured fields with common random numbers, resumably.

    Step zero evaluates the immutable pretrained EMA; the final step evaluates
    current policy parameters. Each completed case is appended immediately so
    a requeued job can skip finished cases without changing the training RNG.
    """

    derived_seed = int(
        np.random.SeedSequence([nft_config.seed, FULL_EVALUATION_SEED_TAG])
        .generate_state(1, dtype=np.uint32)[0]
    )
    seeds = tuple(getattr(nft_config, "evaluation_seeds", ())) or (derived_seed,)
    completed = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                completed[
                    (record["outer_step"], record["name"], record["seed"])
                ] = record
    for case in nft_config.evaluation_cases:
        case_config = replace(
            nft_config,
            self_cond_cfg_scale=case.model_cfg_scale,
            rollout_self_cond_cfg_scale=case.model_cfg_scale,
            rollout_field_mode="native",
        )
        functions = make_rollout_functions(
            model_apply_fn=model_apply_fn, elf_config=elf_config, nft_config=case_config,
            external_cfg_scale=case.external_cfg_scale,
        )
        for seed in seeds:
            plan = FullEvaluationPlan(
                outer_step=outer_step,
                num_samples=nft_config.evaluation_num_samples,
                generation_batch_size=nft_config.rollout_generation_batch_size,
                seed=seed,
            )
            previous = completed.get((outer_step, case.name, seed))
            if previous is not None:
                expected = dict(
                    asdict(case),
                    num_samples=plan.num_samples,
                    seed=seed,
                    generation_batch_size=plan.generation_batch_size,
                )
                if any(
                    previous.get(key) != value for key, value in expected.items()
                ):
                    raise ValueError(
                        "saved SCCFG evaluation settings differ for "
                        f"{case.name} seed={seed}"
                    )
                log_for_0(
                    "Skipping completed SCCFG evaluation "
                    f"step={outer_step} case={case.name} seed={seed}"
                )
                continue
            trigger = (
                "before_training"
                if outer_step == 0
                else (
                    "after_training"
                    if outer_step == nft_config.max_outer_steps
                    else "scheduled_periodic"
                )
            )
            _run_owt_full_evaluation(
                plan=plan,
                actual_outer_step=outer_step,
                trigger=trigger,
                functions=functions,
                policy_params_replicated=policy_params_replicated,
                tokenizer=tokenizer,
                reward_scorer=reward_scorer,
                embedding_dim=embedding_dim,
                elf_config=elf_config,
                nft_config=case_config,
                output_path=output_path,
                evaluation_case=case,
                policy_source=(
                    "pretrained_ema"
                    if outer_step == 0
                    else "current_policy_checkpoint"
                ),
            )


def _wandb_full_evaluation_payload(record):
    return {
        f"full_evaluation/{name}": value
        for name, value in record.items()
        if not isinstance(value, bool) and isinstance(value, (int, float))
    }


def _elf_config_dict(elf_config):
    elf_payload = {}
    for key, value in vars(elf_config).items():
        if key.startswith("_"):
            continue
        if isinstance(value, list) and value and hasattr(value[0], "__dict__"):
            value = [vars(item) for item in value]
        elf_payload[key] = value
    return elf_payload


def _run_wmt_evaluation(
    *, model_apply_fn, encoder_apply_fn, policy_params_replicated,
    theta_ema_params_replicated=None,
    encoder_params_replicated, dataset, outer_step, tokenizer,
    elf_config, nft_config, output_path,
):
    """Append a resumable paired-CFG evaluation of the configured snapshot.

    The formal schema-v11 path selects theta EMA after training updates.  At
    step zero, raw theta and theta EMA are both the immutable released EMA, so
    it is evaluated once and recorded as ``pretrained_ema``.
    """

    seed = int(np.random.SeedSequence([nft_config.seed, 0x574D5445])
               .generate_state(1, dtype=np.uint32)[0])
    requested_snapshot = nft_config.evaluation_policy_snapshot
    if outer_step == 0:
        selected_params = (
            theta_ema_params_replicated
            if theta_ema_params_replicated is not None
            else policy_params_replicated
        )
        policy_snapshot = "pretrained_ema"
    elif requested_snapshot == "theta_ema":
        if theta_ema_params_replicated is None:
            raise ValueError("theta-EMA WMT evaluation requires EMA parameters")
        selected_params = theta_ema_params_replicated
        policy_snapshot = "theta_ema"
    else:
        selected_params = policy_params_replicated
        policy_snapshot = "raw"

    theta_ema_step = int(outer_step) if requested_snapshot == "theta_ema" else 0
    effective_decay = (
        float(
            theta_ema_effective_decay(
                theta_ema_step, nft_config.theta_ema_decay
            )
        )
        if theta_ema_step
        else 0.0
    )
    completed = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                previous = json.loads(line)
                completed[(previous.get("outer_step"), previous.get("name"))] = previous

    records = []
    for input_cfg_scale in nft_config.evaluation_input_cfg_scales:
        settings = dict(
            name=(
                f"source_cfg{input_cfg_scale:g}_sccfg1_"
                f"{policy_snapshot}"
            ),
            input_cfg_scale=input_cfg_scale,
            model_cfg_scale=1.0,
            num_samples=nft_config.evaluation_num_samples,
            generation_batch_size=nft_config.rollout_generation_batch_size,
            seed=seed,
            dataset=elf_config.eval_data_path,
            source_cfg_mode=nft_config.source_cfg_mode,
            requested_policy_snapshot=requested_snapshot,
            policy_snapshot=policy_snapshot,
            policy_source=policy_snapshot,
            theta_ema_decay=nft_config.theta_ema_decay,
            theta_ema_effective_decay=effective_decay,
            theta_ema_step=theta_ema_step,
        )
        previous = completed.get((outer_step, settings["name"]))
        if previous is not None:
            if any(previous.get(key) != value for key, value in settings.items()):
                raise ValueError(
                    "saved WMT evaluation settings differ from configuration"
                )
            required_metrics = (
                "reward_mean",
                "mean_sentence_bleu",
                "corpus_bleu",
                "bleu",
                "rouge1",
                "rouge2",
                "rougeL",
            )
            if any(
                not np.isfinite(previous.get(name, float("nan")))
                for name in required_metrics
            ):
                raise ValueError(
                    "saved WMT evaluation lacks finite ELF-B metrics"
                )
            records.append(previous)
            continue

        eval_config = replace(
            nft_config, rollout_cfg_scale=input_cfg_scale
        )
        functions = make_rollout_functions(
            model_apply_fn=model_apply_fn,
            encoder_apply_fn=encoder_apply_fn,
            elf_config=elf_config,
            nft_config=eval_config,
        )
        start = time.time()
        result = evaluate_wmt_policy(
            dataset=dataset,
            functions=functions,
            policy_params_replicated=selected_params,
            encoder_params_replicated=encoder_params_replicated,
            tokenizer=tokenizer,
            rng=jax.random.PRNGKey(seed),
            elf_config=elf_config,
            nft_config=eval_config,
        )
        _require_finite_metrics(
            result.metrics, stage="held-out WMT evaluation"
        )
        record = dict(
            settings,
            outer_step=int(outer_step),
            evaluation_type="wmt_validation",
            elapsed_seconds=time.time() - start,
            **result.metrics,
            hypothesis_previews=result.hypotheses[:4],
            reference_previews=result.references[:4],
        )
        _append_jsonl(output_path, record)
        log_for_0(record)
        records.append(record)
    return records


def _run_xsum_evaluation(
    *,
    model_apply_fn,
    encoder_apply_fn,
    policy_params_replicated,
    theta_ema_params_replicated=None,
    encoder_params_replicated,
    dataset,
    outer_step,
    tokenizer,
    elf_config,
    nft_config,
    output_path,
):
    """Append resumable paired-CFG XSum metrics for the configured snapshot."""

    seed = int(
        np.random.SeedSequence([nft_config.seed, 0x58534556])
        .generate_state(1, dtype=np.uint32)[0]
    )
    requested_snapshot = nft_config.evaluation_policy_snapshot
    if outer_step == 0:
        selected_params = (
            theta_ema_params_replicated
            if theta_ema_params_replicated is not None
            else policy_params_replicated
        )
        policy_snapshot = "pretrained_ema"
    elif requested_snapshot == "theta_ema":
        if theta_ema_params_replicated is None:
            raise ValueError("theta-EMA XSum evaluation requires EMA parameters")
        selected_params = theta_ema_params_replicated
        policy_snapshot = "theta_ema"
    else:
        selected_params = policy_params_replicated
        policy_snapshot = "raw"

    theta_ema_step = int(outer_step) if requested_snapshot == "theta_ema" else 0
    effective_decay = (
        float(
            theta_ema_effective_decay(
                theta_ema_step, nft_config.theta_ema_decay
            )
        )
        if theta_ema_step
        else 0.0
    )
    data_settings = xsum_settings_from_config(elf_config, nft_config)
    completed = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                previous = json.loads(line)
                completed[(previous.get("outer_step"), previous.get("name"))] = (
                    previous
                )

    records = []
    expected_case_indices = None
    for input_cfg_scale in nft_config.evaluation_input_cfg_scales:
        settings = dict(
            name=(
                f"source_cfg{input_cfg_scale:g}_sccfg1_"
                f"{policy_snapshot}"
            ),
            input_cfg_scale=input_cfg_scale,
            model_cfg_scale=1.0,
            num_samples=nft_config.evaluation_num_samples,
            generation_batch_size=nft_config.rollout_generation_batch_size,
            seed=seed,
            dataset=data_settings.validation_repo,
            dataset_revision=data_settings.validation_revision,
            source_cfg_mode=nft_config.source_cfg_mode,
            requested_policy_snapshot=requested_snapshot,
            policy_snapshot=policy_snapshot,
            policy_source=policy_snapshot,
            theta_ema_decay=nft_config.theta_ema_decay,
            theta_ema_effective_decay=effective_decay,
            theta_ema_step=theta_ema_step,
            hypothesis_max_tokens=nft_config.hypothesis_max_tokens,
            reward_metric="rougeL_f1_x100",
        )
        previous = completed.get((outer_step, settings["name"]))
        if previous is not None:
            if any(previous.get(key) != value for key, value in settings.items()):
                raise ValueError(
                    "saved XSum evaluation settings differ from configuration"
                )
            required_metrics = (
                "rouge1",
                "rouge2",
                "rougeL",
                "reward_mean",
                "hypothesis_token_count_mean",
                "empty_hypothesis_fraction",
            )
            if any(
                not np.isfinite(previous.get(name, float("nan")))
                for name in required_metrics
            ):
                raise ValueError("saved XSum evaluation lacks finite metrics")
            case_indices = validate_xsum_evaluation_examples(
                previous.get("examples"),
                expected_count=nft_config.evaluation_num_samples,
                aggregate_metrics=previous,
                max_tokens=nft_config.hypothesis_max_tokens,
            )
            if (
                expected_case_indices is not None
                and case_indices != expected_case_indices
            ):
                raise ValueError("XSum paired-CFG evaluation index order differs")
            expected_case_indices = case_indices
            records.append(previous)
            continue

        eval_config = replace(
            nft_config, rollout_cfg_scale=input_cfg_scale
        )
        functions = make_rollout_functions(
            model_apply_fn=model_apply_fn,
            encoder_apply_fn=encoder_apply_fn,
            elf_config=elf_config,
            nft_config=eval_config,
        )
        start = time.time()
        result = evaluate_xsum_policy(
            dataset=dataset,
            functions=functions,
            policy_params_replicated=selected_params,
            encoder_params_replicated=encoder_params_replicated,
            tokenizer=tokenizer,
            rng=jax.random.PRNGKey(seed),
            elf_config=elf_config,
            nft_config=eval_config,
        )
        _require_finite_metrics(
            result.metrics, stage="held-out XSum evaluation"
        )
        case_indices = validate_xsum_evaluation_examples(
            result.examples,
            expected_count=nft_config.evaluation_num_samples,
            aggregate_metrics=result.metrics,
            max_tokens=nft_config.hypothesis_max_tokens,
        )
        if (
            expected_case_indices is not None
            and case_indices != expected_case_indices
        ):
            raise ValueError("XSum paired-CFG evaluation index order differs")
        expected_case_indices = case_indices
        record = dict(
            settings,
            outer_step=int(outer_step),
            evaluation_type="xsum_validation",
            elapsed_seconds=time.time() - start,
            **result.metrics,
            hypothesis_previews=result.hypotheses[:4],
            reference_previews=result.references[:4],
            examples=result.examples,
        )
        _append_jsonl(output_path, record)
        log_for_0(
            {
                **{
                    key: value
                    for key, value in record.items()
                    if key != "examples"
                },
                "examples_count": len(result.examples),
            }
        )
        records.append(record)
    return records


def _save_effective_config(output_dir: Path, elf_config, nft_config):
    payload = {
        "elf": _elf_config_dict(elf_config),
        "nft": asdict(nft_config),
    }
    with (output_dir / "effective_config.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    with (output_dir / "effective_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_wmt_data_preflight(output_dir: Path, report):
    """Persist one deterministic proof of the WMT data contract.

    Resumes may revisit this function.  An existing byte-identical report is
    accepted, while a different report fails closed instead of silently
    changing the dataset contract inside an existing output directory.
    """

    path = output_dir / "wmt_data_preflight.json"
    payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(
                "WMT data preflight report changed inside the output directory"
            )
        return path
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return path


def _log_header(elf_config, nft_config, full_evaluation_plan=None):
    rollout_model_scale, rollout_external_scale = resolve_rollout_sccfg_settings(
        nft_config
    )
    center_training_records = (
        nft_config.center_training_records_per_outer
        if nft_config.method == "shifted_center" else 0
    )
    center_updates = (
        nft_config.center_updates_per_outer
        if nft_config.method == "shifted_center" else 0
    )
    log_for_0("=" * 72)
    log_for_0("ELF SC-CFG DiffusionNFT center variants")
    log_for_0("=" * 72)
    log_for_0(f"task={nft_config.task}; method={nft_config.method}")
    log_for_0(f"self_condition_mode={nft_config.self_condition_mode}; paired_loss_reduction=sum")
    log_for_0(
        f"source_cfg_mode={nft_config.source_cfg_mode}; "
        f"training-input-CFG={training_input_cfg_scale(nft_config)}; "
        f"source-dual-reduction=mean; evaluation-input-CFGs="
        f"{list(nft_config.evaluation_input_cfg_scales)}; "
        f"evaluation-policy-snapshot={nft_config.evaluation_policy_snapshot}"
    )
    if nft_config.evaluation_cases:
        log_for_0(
            f"sccfg_evaluation_cases={[asdict(case) for case in nft_config.evaluation_cases]}; "
            f"samples_per_case={nft_config.evaluation_num_samples}; "
            f"at_start={nft_config.evaluation_at_start}; at_end={nft_config.evaluation_at_end}"
        )
    log_for_0(f"task_config={nft_config.base_config}")
    log_for_0(
        f"pretrained_checkpoint={nft_config.pretrained_checkpoint}@"
        f"{nft_config.pretrained_checkpoint_revision or 'unpinned'}"
    )
    if nft_config.task == "wmt":
        reference = _wmt_reference_settings(elf_config)
        log_for_0(
            "tokenized-training="
            f"{elf_config.data_path}@{reference['wmt_tokenized_data_revision']}; "
            "tokenized-validation="
            f"{elf_config.eval_data_path}@{reference['wmt_validation_data_revision']}"
        )
        log_for_0(
            "training-reference="
            f"{reference['wmt_reference_data_path']}@"
            f"{reference['wmt_reference_data_revision']}/"
            f"{reference['wmt_reference_data_name']}:"
            f"{reference['wmt_reference_data_split']} "
            f"{reference['wmt_source_language']}->"
            f"{reference['wmt_target_language']}; "
            "reference-mode=verbatim-original-with-token-alignment"
        )
        log_for_0(
            f"tokenizer={elf_config.tokenizer_name or elf_config.encoder_model_name}@"
            f"{getattr(elf_config, 'tokenizer_revision', None) or 'unpinned'}; "
            f"encoder-checkpoint={elf_config.encoder_checkpoint}@"
            f"{getattr(elf_config, 'encoder_checkpoint_revision', None) or 'unpinned'}"
        )
    elif nft_config.task == "xsum":
        settings = xsum_settings_from_config(elf_config, nft_config)
        log_for_0(
            f"tokenized-training={settings.train_repo}@{settings.train_revision} "
            f"rows={settings.train_rows}; tokenized-validation="
            f"{settings.validation_repo}@{settings.validation_revision} "
            f"rows={settings.validation_rows}"
        )
        log_for_0(
            f"raw-xsum-revision={settings.raw_revision}; tokenizer="
            f"{settings.tokenizer_name}@{settings.tokenizer_revision}; "
            f"encoder-checkpoint={elf_config.encoder_checkpoint}@"
            f"{settings.encoder_revision}; target-cap="
            f"{settings.target_max_tokens}"
        )
    log_for_0(
        f"rollout={nft_config.rollout_sampling_method.upper()}/"
        f"{nft_config.rollout_steps} steps, "
        f"sde_gamma={nft_config.rollout_sde_gamma}, "
        f"input-CFG={nft_config.rollout_cfg_scale}, "
        f"rollout-model-SC-CFG={rollout_model_scale}, "
        f"rollout-field={nft_config.rollout_field_mode}, "
        f"rollout-external-SC-scale={rollout_external_scale}"
    )
    log_for_0(
        f"training-field-SC-CFG-distribution="
        f"{nft_config.training_self_cond_cfg_distribution}; "
        f"range=[{nft_config.training_self_cond_cfg_min}, "
        f"{nft_config.training_self_cond_cfg_max}]; "
        f"CFG1-velocity-reconstruction="
        f"{nft_config.training_velocity_reconstruction}; "
        f"decoder-SC-CFG={nft_config.self_cond_cfg_scale}; "
        f"center-SC-CFG={nft_config.center_self_cond_cfg_scale}; "
        f"conditional-center={nft_config.method}"
    )
    log_for_0(
        f"groups={nft_config.prompt_groups_per_outer}, "
        f"samples/group={nft_config.samples_per_prompt}, "
        f"rollout_batch={nft_config.rollout_batch_size}, "
        f"generation_batch={nft_config.rollout_generation_batch_size}"
    )
    log_for_0(
        f"training-records/outer: center={center_training_records}, "
        f"policy={nft_config.policy_training_records_per_outer}"
    )
    log_for_0(
        f"global-train-microbatch={nft_config.train_batch_size}; "
        f"backward-microbatches/outer: center="
        f"{nft_config.center_backward_microbatches_per_outer if nft_config.method == 'shifted_center' else 0}, "
        f"policy={nft_config.policy_backward_microbatches_per_outer}; "
        "gradient-reduction=record-weighted-mean"
    )
    log_for_0(
        f"updates/outer: center={center_updates}, "
        f"policy={nft_config.policy_updates_per_outer}"
    )
    log_for_0(
        f"theta-EMA=SD3-early-ramp, decay-cap="
        f"{nft_config.theta_ema_decay}; evaluation/checkpoint-only=yes"
    )
    log_for_0(
        f"center-pretrain-steps={nft_config.center_pretrain_steps}; "
        "pretrain-rollout-policy=immutable-pretrained-ema"
    )
    log_for_0(
        f"checkpoint-save-every={nft_config.save_every}; "
        f"rolling-every={nft_config.checkpoint_rolling_every}; "
        f"conditional-eval-every={nft_config.evaluation_every_n_steps}; "
        f"keep-latest={nft_config.keep_checkpoints}; "
        f"keep-every={nft_config.keep_every_n_steps}"
    )
    log_for_0(
        f"SC-routing={nft_config.self_condition_mode}, "
        f"Bernoulli-probability={nft_config.self_cond_update_probability if nft_config.self_condition_mode == 'bernoulli' else 'unused'}, "
        f"target=classic-OT-endpoint-minus-noise, beta={nft_config.beta}, "
        f"kl_beta={nft_config.kl_beta}, "
        f"decoder_ce_weight={nft_config.decoder_ce_weight}"
    )
    log_for_0(
        f"reward={nft_config.reward_type}"
        + (
            f" ({nft_config.reward_model_name}, empty={nft_config.empty_text_reward})"
            if nft_config.task == "owt" else ""
        )
    )
    if nft_config.task == "owt":
        log_for_0(
            f"reward_combination={nft_config.reward_combination_mode}, "
            f"entropy_target={nft_config.entropy_target}, "
            f"initial_entropy_weight={nft_config.initial_entropy_weight}, "
            f"max_entropy_weight={nft_config.max_entropy_weight}, "
            f"entropy_ema_decay={nft_config.entropy_ema_decay}, "
            f"dual_learning_rate={nft_config.entropy_dual_learning_rate}"
        )
        log_for_0(
            "online_metrics=every_outer(old_policy_before_update): "
            "ppl, entropy"
        )
        if full_evaluation_plan is not None:
            log_for_0(
                f"midpoint_full_evaluation=outer_step="
                f"{full_evaluation_plan.outer_step}, "
                f"current_policy_checkpoint, samples="
                f"{full_evaluation_plan.num_samples}, "
                f"generation_batch={full_evaluation_plan.generation_batch_size}, "
                f"fixed_seed={full_evaluation_plan.seed}"
            )
    log_for_0(
        f"parameterization={nft_config.parameterization}; "
        "shared_elf_params=TRAINED; decoder_specific_params=FROZEN"
    )
    log_for_0(f"output_dir={nft_config.output_dir}")
    log_for_0(
        f"jax_backend={jax.default_backend()}, local_devices={jax.local_device_count()}, "
        f"global_devices={jax.device_count()}"
    )
    log_for_0("PAPER_COMPARABLE=no (online NFT adaptation of a released ELF sampler)")
    log_for_0(f"FORMAL_EXPERIMENT={'yes' if nft_config.formal_experiment else 'no'}")
    log_for_0("=" * 72)


def run_training(elf_config, nft_config, *, resume=None, capacity_probe=False):
    _validate_device_geometry(nft_config)
    full_evaluation_plan = _build_midpoint_full_evaluation_plan(
        elf_config, nft_config
    )
    _log_header(elf_config, nft_config, full_evaluation_plan)
    _log_device_memory("startup")
    output_dir = Path(nft_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        validate_resume_metadata(
            resume,
            expected_method=nft_config.method,
            expected_base_checkpoint=nft_config.pretrained_checkpoint,
            expected_base_checkpoint_revision=(
                nft_config.pretrained_checkpoint_revision
            ),
            expected_config_path=nft_config.config_path,
            expected_parameterization=nft_config.parameterization,
            expected_self_condition_mode=nft_config.self_condition_mode,
            expected_source_cfg_mode=nft_config.source_cfg_mode,
            expected_theta_ema_decay=nft_config.theta_ema_decay,
        )
    _save_effective_config(output_dir, elf_config, nft_config)

    wandb_run = None
    if nft_config.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=nft_config.wandb_project,
            name=nft_config.wandb_run_name,
            config={"elf": _elf_config_dict(elf_config), "nft": asdict(nft_config)},
            dir=os.environ.get("SLURM_TMPDIR", "/tmp"),
        )

    rng = jax.random.PRNGKey(nft_config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        elf_config.tokenizer_name or elf_config.encoder_model_name,
        revision=getattr(elf_config, "tokenizer_revision", None),
    )
    pad_token_id = get_pad_token_id(tokenizer, elf_config.pad_token)
    train_dataset = None
    reference_dataset = None
    reference_settings = None
    validation_dataset = None
    if nft_config.task == "wmt":
        (
            train_dataset,
            reference_dataset,
            reference_settings,
            preflight_report,
        ) = _load_and_preflight_wmt_training_data(
            elf_config,
            nft_config,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
        )
        _save_wmt_data_preflight(output_dir, preflight_report)
        if (nft_config.evaluation_at_start or nft_config.evaluation_at_end
                or nft_config.evaluation_every_n_steps):
            validation_dataset = load_dataset_split(
                elf_config.eval_data_path,
                dataset_cache_dir=os.environ.get("HF_DATASETS_CACHE"),
                revision=reference_settings["wmt_validation_data_revision"],
            )
            if nft_config.evaluation_num_samples > len(validation_dataset):
                raise ValueError("WMT evaluation count exceeds the validation dataset size")
    elif nft_config.task == "xsum":
        train_dataset, validation_dataset, xsum_settings = (
            load_pinned_xsum_datasets(elf_config, nft_config)
        )
        preflight_report = preflight_xsum_datasets(
            train_dataset,
            validation_dataset,
            tokenizer=tokenizer,
            elf_config=elf_config,
            nft_config=nft_config,
            pad_token_id=pad_token_id,
            settings=xsum_settings,
        )
        save_xsum_preflight(output_dir, preflight_report)
        if nft_config.evaluation_num_samples > len(validation_dataset):
            raise ValueError(
                "XSum evaluation count exceeds the validation dataset size"
            )

    model, encoder_config, encoder_model, base_params, pretrained_step, rng = (
        _build_model_and_pretrained_params(elf_config, nft_config, tokenizer, rng)
    )
    validate_full_model_params(base_params)
    trainable_count, frozen_decoder_count = parameter_counts(base_params)
    total_count = trainable_count + frozen_decoder_count
    log_for_0(
        f"Loaded immutable pretrained EMA at step={pretrained_step}; "
        f"encoder_dim={encoder_config.d_model}"
    )
    log_for_0(
        f"Full ELF parameters={total_count:,}; trainable_shared={trainable_count:,}; "
        f"frozen_decoder_specific={frozen_decoder_count:,}"
    )
    encoder_params = None
    if nft_config.task in {"wmt", "xsum"}:
        encoder_params = load_encoder_checkpoint(
            elf_config.encoder_checkpoint,
            revision=getattr(elf_config, "encoder_checkpoint_revision", None),
        )
    reward_scorer = None
    if nft_config.task == "owt":
        reward_scorer = CausalLMLogLikelihoodReward(
            model_name=nft_config.reward_model_name,
            batch_size=nft_config.reward_batch_size,
            max_length=nft_config.reward_max_length,
            empty_text_reward=nft_config.empty_text_reward,
        )

    current_state = create_model_state(
        model_params=base_params,
        apply_fn=model.apply,
        learning_rate=nft_config.learning_rate,
        max_grad_norm=nft_config.max_grad_norm,
        weight_decay=nft_config.weight_decay,
        theta_ema_decay=nft_config.theta_ema_decay,
    )
    center_state = None
    if nft_config.method == "shifted_center":
        center_state = create_center_state(
            model_params=base_params,
            apply_fn=model.apply,
            learning_rate=nft_config.center_learning_rate,
            max_grad_norm=nft_config.max_grad_norm,
            weight_decay=nft_config.weight_decay,
        )
    old_params = base_params
    reward_state = make_entropy_constraint_state(
        nft_config.initial_entropy_weight
    )
    start_outer = 0
    completed_center_pretrain = 0
    resume_phase = None

    checkpoint_template = build_checkpoint_payload(
        current_state=current_state,
        center_state=center_state,
        old_params=old_params,
        reward_state=reward_state,
        rng=rng,
        outer_step=0,
        parameterization=PARAMETERIZATION,
    )
    if resume:
        restored = load_online_checkpoint(resume, checkpoint_template)
        current_state = restored["current_state"]
        center_state = restored["center_state"]
        old_params = restored["old_params"]
        reward_state = restored["reward_state"]
        rng = restored["rng"]
        resume_phase = str(restored.get("phase", "online"))
        start_outer = int(restored["outer_step"])
        completed_center_pretrain = int(
            restored.get("center_pretrain_step", 0)
        )
        if start_outer > nft_config.max_outer_steps:
            raise ValueError("resume checkpoint is beyond max_outer_steps")
        if resume_phase == "center_pretrain" and start_outer != 0:
            raise ValueError(
                "center-pretrain resume must precede all online outer steps"
            )
        if resume_phase == "online" and (
            completed_center_pretrain != nft_config.center_pretrain_steps
        ):
            raise ValueError(
                "online resume checkpoint has incomplete center pretraining"
            )
        if completed_center_pretrain > nft_config.center_pretrain_steps:
            raise ValueError(
                "resume checkpoint center pretraining exceeds configuration"
            )
        log_for_0(
            f"Resumed full-model state at phase={resume_phase}; "
            f"center_pretrain_step={completed_center_pretrain}; "
            f"completed outer_step={start_outer}"
        )

    _require_frozen_decoder_reference(
        base_params,
        {
            "current": current_state.params,
            "theta_ema": current_state.ema_params,
            "old": old_params,
            "center": None if center_state is None else center_state.params,
            "center_ema": None if center_state is None else center_state.ema_params,
        },
    )

    base_params_replicated = jax_utils.replicate(base_params)
    encoder_params_replicated = (
        None if encoder_params is None else jax_utils.replicate(encoder_params)
    )
    current_state = jax_utils.replicate(current_state)
    if center_state is not None:
        center_state = jax_utils.replicate(center_state)
    old_params = jax_utils.replicate(old_params)

    rollout_functions = make_rollout_functions(
        model_apply_fn=model.apply,
        encoder_apply_fn=(
            encoder_model.apply
            if nft_config.task in {"wmt", "xsum"}
            else None
        ),
        elf_config=elf_config,
        nft_config=nft_config,
    )
    p_initialize_center_accumulator = None
    p_accumulate_center = None
    p_apply_center = None
    if nft_config.method == "shifted_center":
        p_initialize_center_accumulator = jax.pmap(
            partial(
                initialize_gradient_accumulator,
                metric_names=CENTER_ACCUMULATED_METRIC_NAMES,
            )
        )
        p_accumulate_center = jax.pmap(
            partial(
                accumulate_center_gradients,
                model_apply_fn=model.apply,
                elf_config=elf_config,
                nft_config=nft_config,
            ),
            axis_name="batch",
            donate_argnums=(0,),
        )
        p_apply_center = jax.pmap(
            partial(apply_center_gradients, nft_config=nft_config),
            donate_argnums=(0, 1),
        )
    p_initialize_policy_accumulator = jax.pmap(
        partial(
            initialize_gradient_accumulator,
            metric_names=POLICY_ACCUMULATED_METRIC_NAMES,
        )
    )
    p_accumulate_policy = jax.pmap(
        partial(
            accumulate_policy_gradients,
            model_apply_fn=model.apply,
            elf_config=elf_config,
            nft_config=nft_config,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )
    p_accumulate_policy_empty = None
    if nft_config.unconditional_update_every:
        p_accumulate_policy_empty = jax.pmap(
            partial(
                accumulate_policy_gradients,
                model_apply_fn=model.apply,
                elf_config=elf_config,
                nft_config=nft_config,
                update_branch="empty",
            ),
            axis_name="batch",
            donate_argnums=(0,),
        )
    p_finalize_accumulator = jax.pmap(
        finalize_gradient_accumulator,
        donate_argnums=(0,),
    )
    p_apply_policy = jax.pmap(
        partial(
            apply_policy_gradients,
            theta_ema_decay=nft_config.theta_ema_decay,
        ),
        donate_argnums=(0, 1),
    )

    metrics_path = output_dir / "metrics.jsonl"
    center_pretrain_metrics_path = output_dir / "center_pretrain_metrics.jsonl"
    previews_path = output_dir / "rollout_previews.jsonl"
    full_evaluations_path = output_dir / "full_evaluations.jsonl"
    sccfg_evaluations_path = output_dir / "sccfg_evaluations.jsonl"
    wmt_evaluations_path = output_dir / "wmt_evaluations.jsonl"
    xsum_evaluations_path = output_dir / "xsum_evaluations.jsonl"
    if resume:
        for path in (metrics_path, previews_path, full_evaluations_path):
            retained, dropped = _reconcile_jsonl_to_checkpoint(path, start_outer)
            log_for_0(
                f"Reconciled {path.name} to outer_step={start_outer}: "
                f"retained={retained}, dropped={dropped}"
            )
    if sccfg_evaluations_path.exists():
        retained, dropped = _reconcile_jsonl_to_checkpoint(
            sccfg_evaluations_path, start_outer, group_by_case=True
        )
        log_for_0(f"Reconciled SCCFG evaluations: retained={retained}, dropped={dropped}")
    if wmt_evaluations_path.exists():
        _reconcile_jsonl_to_checkpoint(wmt_evaluations_path, start_outer, group_by_case=True)
    if xsum_evaluations_path.exists():
        _reconcile_jsonl_to_checkpoint(
            xsum_evaluations_path, start_outer, group_by_case=True
        )
    if nft_config.center_pretrain_steps:
        if center_state is None:
            raise RuntimeError("center pretraining requires an initialized center state")
        expected_center_step = completed_center_pretrain + start_outer
        if resume:
            actual_center_step = int(
                jax.device_get(jax_utils.unreplicate(center_state).step)
            )
            if actual_center_step != expected_center_step:
                raise ValueError(
                    "resumed center optimizer step does not match phase progress: "
                    f"{actual_center_step} != {expected_center_step}"
                )
            retained, dropped = _reconcile_center_pretrain_jsonl(
                center_pretrain_metrics_path, completed_center_pretrain
            )
            log_for_0(
                "Reconciled center pretraining to published milestone: "
                f"step={completed_center_pretrain}, retained={retained}, "
                f"dropped={dropped}"
            )
        elif center_pretrain_metrics_path.exists():
            raise FileExistsError(
                "fresh center pretraining requires no existing "
                f"{center_pretrain_metrics_path.name}"
            )
        if completed_center_pretrain == nft_config.center_pretrain_steps:
            log_for_0(
                "Skipping completed center pretraining: "
                f"steps={nft_config.center_pretrain_steps}"
            )
        else:
            for pretrain_index in range(
                completed_center_pretrain, nft_config.center_pretrain_steps
            ):
                pretrain_start = time.time()
                prompt_batch, prompt_indices = _task_prompt_batch(
                    train_dataset,
                    outer_step=pretrain_index,
                    nft_config=nft_config,
                    elf_config=elf_config,
                    pad_token_id=pad_token_id,
                    reference_dataset=reference_dataset,
                    reference_settings=reference_settings,
                    tokenizer=tokenizer,
                    center_pretrain=True,
                )
                rng, rollout_rng = jax.random.split(rng)
                pretrain_rollout = collect_old_policy_rollout(
                    functions=rollout_functions,
                    old_params_replicated=old_params,
                    encoder_params_replicated=encoder_params_replicated,
                    prompt_batch=prompt_batch,
                    tokenizer=tokenizer,
                    rng=rollout_rng,
                    elf_config=elf_config,
                    nft_config=nft_config,
                    reward_scorer=reward_scorer,
                    embedding_dim=encoder_config.d_model,
                    entropy_weight=0.0,
                )
                _require_finite_metrics(
                    pretrain_rollout.metrics, stage="center pretrain rollout"
                )
                (
                    center_state,
                    rng,
                    center_host_metrics,
                    center_backward_microbatches,
                    center_record_count,
                    center_step,
                ) = _apply_center_effective_batch(
                    center_state=center_state,
                    old_params=old_params,
                    rollout_buffer=pretrain_rollout.buffer,
                    rng=rng,
                    nft_config=nft_config,
                    p_initialize_center_accumulator=p_initialize_center_accumulator,
                    p_accumulate_center=p_accumulate_center,
                    p_finalize_accumulator=p_finalize_accumulator,
                    p_apply_center=p_apply_center,
                )
                completed_pretrain = pretrain_index + 1
                if center_step != completed_pretrain:
                    raise RuntimeError(
                        "fresh center pretraining optimizer cadence changed: "
                        f"{center_step} != {completed_pretrain}"
                    )
                pretrain_record = {
                    "phase": "center_pretrain",
                    "center_pretrain_step": completed_pretrain,
                    "center_step": center_step,
                    "policy_step": 0,
                    "rollout_policy_source": "immutable_pretrained_ema",
                    "cfg_scale": nft_config.rollout_cfg_scale,
                    "training_input_cfg_scale": training_input_cfg_scale(
                        nft_config
                    ),
                    "source_cfg_mode": nft_config.source_cfg_mode,
                    "prompt_indices": prompt_indices,
                    "center_training_records": center_record_count,
                    "center_backward_microbatches": (
                        center_backward_microbatches
                    ),
                    "elapsed_seconds": time.time() - pretrain_start,
                    **pretrain_rollout.metrics,
                    **center_host_metrics,
                }
                _append_jsonl(center_pretrain_metrics_path, pretrain_record)
                log_for_0(pretrain_record)
                if (
                    completed_pretrain % nft_config.keep_every_n_steps == 0
                    or completed_pretrain == nft_config.center_pretrain_steps
                ):
                    current_host = jax.device_get(
                        jax_utils.unreplicate(current_state)
                    )
                    center_host = jax.device_get(
                        jax_utils.unreplicate(center_state)
                    )
                    old_host = jax.device_get(
                        jax_utils.unreplicate(old_params)
                    )
                    payload = build_checkpoint_payload(
                        current_state=current_host,
                        center_state=center_host,
                        old_params=old_host,
                        reward_state=reward_state,
                        rng=rng,
                        outer_step=0,
                        center_pretrain_step=completed_pretrain,
                        phase="center_pretrain",
                        parameterization=PARAMETERIZATION,
                    )
                    checkpoint_path = save_online_checkpoint(
                        output_dir=str(output_dir),
                        payload=payload,
                        keep=nft_config.keep_checkpoints,
                        method=nft_config.method,
                        base_checkpoint=nft_config.pretrained_checkpoint,
                        base_checkpoint_revision=(
                            nft_config.pretrained_checkpoint_revision
                        ),
                        config_path=nft_config.config_path,
                        keep_every_n_steps=nft_config.keep_every_n_steps,
                        self_condition_mode=nft_config.self_condition_mode,
                        source_cfg_mode=nft_config.source_cfg_mode,
                        theta_ema_decay=nft_config.theta_ema_decay,
                    )
                    log_for_0(
                        "Saved center-pretrain checkpoint: "
                        f"{checkpoint_path}"
                    )
            completed_center_pretrain = nft_config.center_pretrain_steps
    if nft_config.evaluation_at_start:
        if nft_config.task == "wmt":
            _run_wmt_evaluation(
                model_apply_fn=model.apply, encoder_apply_fn=encoder_model.apply,
                policy_params_replicated=base_params_replicated,
                theta_ema_params_replicated=base_params_replicated,
                encoder_params_replicated=encoder_params_replicated, dataset=validation_dataset,
                outer_step=0, tokenizer=tokenizer, elf_config=elf_config, nft_config=nft_config,
                output_path=wmt_evaluations_path,
            )
        elif nft_config.task == "xsum":
            _run_xsum_evaluation(
                model_apply_fn=model.apply,
                encoder_apply_fn=encoder_model.apply,
                policy_params_replicated=base_params_replicated,
                theta_ema_params_replicated=base_params_replicated,
                encoder_params_replicated=encoder_params_replicated,
                dataset=validation_dataset,
                outer_step=0,
                tokenizer=tokenizer,
                elf_config=elf_config,
                nft_config=nft_config,
                output_path=xsum_evaluations_path,
            )
        else:
            _run_sccfg_evaluation_suite(
                model_apply_fn=model.apply, policy_params_replicated=base_params_replicated,
                outer_step=0, tokenizer=tokenizer, reward_scorer=reward_scorer,
                embedding_dim=encoder_config.d_model, elf_config=elf_config, nft_config=nft_config,
                output_path=sccfg_evaluations_path,
            )
    # A restart after checkpoint publication but before evaluation must finish
    # that exact checkpoint's evaluation before collecting another rollout.
    if _wmt_evaluation_due(start_outer, nft_config):
        evaluation_function = (
            _run_wmt_evaluation
            if nft_config.task == "wmt"
            else _run_xsum_evaluation
        )
        evaluation_function(
            model_apply_fn=model.apply,
            encoder_apply_fn=encoder_model.apply,
            policy_params_replicated=current_state.params,
            theta_ema_params_replicated=current_state.ema_params,
            encoder_params_replicated=encoder_params_replicated,
            dataset=validation_dataset,
            outer_step=start_outer,
            tokenizer=tokenizer,
            elf_config=elf_config,
            nft_config=nft_config,
            output_path=(
                wmt_evaluations_path
                if nft_config.task == "wmt"
                else xsum_evaluations_path
            ),
        )
    if _owt_sccfg_evaluation_due(start_outer, nft_config):
        _run_sccfg_evaluation_suite(
            model_apply_fn=model.apply,
            policy_params_replicated=current_state.params,
            outer_step=start_outer,
            tokenizer=tokenizer,
            reward_scorer=reward_scorer,
            embedding_dim=encoder_config.d_model,
            elf_config=elf_config,
            nft_config=nft_config,
            output_path=sccfg_evaluations_path,
        )
    completed_full_evaluations = _scheduled_full_evaluation_steps(
        full_evaluations_path
    )
    if (
        full_evaluation_plan is not None
        and full_evaluation_plan.outer_step not in completed_full_evaluations
        and start_outer >= full_evaluation_plan.outer_step
    ):
        trigger = (
            "resume_at_midpoint"
            if start_outer == full_evaluation_plan.outer_step
            else "resume_catchup_after_midpoint"
        )
        full_record = _run_owt_full_evaluation(
            plan=full_evaluation_plan,
            actual_outer_step=start_outer,
            trigger=trigger,
            functions=rollout_functions,
            policy_params_replicated=current_state.params,
            tokenizer=tokenizer,
            reward_scorer=reward_scorer,
            embedding_dim=encoder_config.d_model,
            elf_config=elf_config,
            nft_config=nft_config,
            output_path=full_evaluations_path,
        )
        completed_full_evaluations.add(full_evaluation_plan.outer_step)
        if wandb_run is not None:
            wandb_run.log(
                _wandb_full_evaluation_payload(full_record), step=start_outer
            )
    for outer_step in range(start_outer, nft_config.max_outer_steps):
        outer_start = time.time()
        if nft_config.task in {"wmt", "xsum"}:
            if train_dataset is None:
                raise RuntimeError(
                    "conditional training dataset was not initialized"
                )
            prompt_batch, prompt_indices = _task_prompt_batch(
                train_dataset,
                outer_step=outer_step,
                nft_config=nft_config,
                elf_config=elf_config,
                pad_token_id=pad_token_id,
                reference_dataset=reference_dataset,
                reference_settings=reference_settings,
                tokenizer=tokenizer,
            )
        else:
            prompt_batch, prompt_indices = None, None
        entropy_weight = entropy_weight_from_state(
            reward_state,
            mode=nft_config.reward_combination_mode,
            fixed_weight=nft_config.initial_entropy_weight,
        )
        rng, rollout_rng = jax.random.split(rng)
        rollout = collect_old_policy_rollout(
            functions=rollout_functions,
            old_params_replicated=old_params,
            encoder_params_replicated=encoder_params_replicated,
            prompt_batch=prompt_batch,
            tokenizer=tokenizer,
            rng=rollout_rng,
            elf_config=elf_config,
            nft_config=nft_config,
            reward_scorer=reward_scorer,
            embedding_dim=encoder_config.d_model,
            entropy_weight=entropy_weight,
        )
        if nft_config.task == "owt":
            reward_state, reward_control_metrics = update_entropy_constraint_state(
                reward_state,
                rollout.metrics["reward_entropy_mean_all"],
                mode=nft_config.reward_combination_mode,
                entropy_target=nft_config.entropy_target,
                ema_decay=nft_config.entropy_ema_decay,
                dual_learning_rate=nft_config.entropy_dual_learning_rate,
                max_entropy_weight=nft_config.max_entropy_weight,
            )
            rollout.metrics.update(reward_control_metrics)
        _require_finite_metrics(rollout.metrics, stage="rollout")
        if nft_config.task == "owt":
            log_for_0(
                f"ONLINE_ROLLOUT outer_step={outer_step + 1}/"
                f"{nft_config.max_outer_steps} "
                f"policy=old_policy_before_update "
                f"samples={nft_config.rollout_batch_size} "
                f"ppl={rollout.metrics['reward_model_ppl']:.6f} "
                f"entropy="
                f"{rollout.metrics['reward_model_unigram_entropy']:.6f}"
            )

        center_host_metrics = {}
        center_backward_microbatches = 0
        center_record_count = 0
        center_step = 0
        if nft_config.method == "shifted_center":
            if (
                center_state is None
                or p_initialize_center_accumulator is None
                or p_accumulate_center is None
                or p_apply_center is None
            ):
                raise RuntimeError("shifted_center state was not initialized")
            (
                center_state,
                rng,
                center_host_metrics,
                center_backward_microbatches,
                center_record_count,
                center_step,
            ) = _apply_center_effective_batch(
                center_state=center_state,
                old_params=old_params,
                rollout_buffer=rollout.buffer,
                rng=rng,
                nft_config=nft_config,
                p_initialize_center_accumulator=p_initialize_center_accumulator,
                p_accumulate_center=p_accumulate_center,
                p_finalize_accumulator=p_finalize_accumulator,
                p_apply_center=p_apply_center,
            )
            expected_center_step = (
                nft_config.center_pretrain_steps + outer_step + 1
            )
            if center_step != expected_center_step:
                raise RuntimeError(
                    "center optimizer step does not match pretrain plus policy "
                    f"rounds: {center_step} != {expected_center_step}"
                )
            center_params = (
                center_state.ema_params
                if nft_config.center_snapshot == "ema" else center_state.params
            )
        else:
            center_params = old_params

        theta_step_before = int(
            jax.device_get(jax_utils.unreplicate(current_state).step)
        )
        policy_accumulator = p_initialize_policy_accumulator(
            current_state.params
        )
        empty_branch_update = updates_empty_branch(nft_config, outer_step + 1)
        accumulate_policy_microbatch = (
            p_accumulate_policy_empty if empty_branch_update else p_accumulate_policy
        )
        policy_backward_microbatches = 0
        for _ in range(nft_config.num_inner_epochs):
            rng, permutation_rng = jax.random.split(rng)
            for minibatch in _iterate_minibatches(
                rollout.buffer, nft_config.train_batch_size, permutation_rng
            ):
                for _ in range(nft_config.timesteps_per_batch):
                    rng, step_rng = jax.random.split(rng)
                    device_rngs = jax.random.split(
                        step_rng, jax.local_device_count()
                    )
                    policy_accumulator = accumulate_policy_microbatch(
                        policy_accumulator,
                        current_state.params,
                        base_params_replicated,
                        old_params,
                        center_params,
                        shard(minibatch),
                        device_rngs,
                    )
                    policy_backward_microbatches += 1
        (
            policy_mean_gradients,
            policy_mean_metrics,
            policy_gradient_norm,
            policy_record_count,
        ) = p_finalize_accumulator(policy_accumulator)
        policy_record_count = int(_host_scalar(policy_record_count))
        _validate_accumulated_batch(
            stage="policy",
            actual_records=policy_record_count,
            expected_records=nft_config.policy_training_records_per_outer,
            actual_microbatches=policy_backward_microbatches,
            expected_microbatches=(
                nft_config.policy_backward_microbatches_per_outer
            ),
        )
        policy_host_metrics = _host_metrics(policy_mean_metrics)
        policy_host_metrics["theta_grad_norm"] = float(
            _host_scalar(policy_gradient_norm)
        )
        _require_finite_metrics(
            policy_host_metrics, stage="policy effective batch"
        )
        current_state, policy_update_metrics = p_apply_policy(
            current_state, policy_mean_gradients
        )
        policy_update_metrics = _host_metrics(policy_update_metrics)
        _require_finite_metrics(
            policy_update_metrics, stage="policy optimizer update"
        )
        policy_host_metrics.update(policy_update_metrics)

        theta_step = int(jax.device_get(jax_utils.unreplicate(current_state).step))
        if theta_step != theta_step_before + 1:
            raise RuntimeError(
                "policy effective batch must produce exactly one optimizer "
                f"update: {theta_step_before} -> {theta_step}"
            )
        decay = old_policy_decay(theta_step, nft_config.old_policy_decay_type)
        old_params = jax.tree_util.tree_map(
            lambda old, current: old * decay + current * (1.0 - decay),
            old_params,
            current_state.params,
        )
        old_params = restore_frozen_decoder_params(
            old_params, base_params_replicated
        )

        completed_outer = outer_step + 1
        record = {
            "outer_step": completed_outer,
            "policy_update_branch": (
                "empty" if empty_branch_update else "configured"
            ),
            "theta_step": theta_step,
            "theta_ema_step": (
                theta_step if current_state.ema_params is not None else 0
            ),
            "theta_ema_decay": nft_config.theta_ema_decay,
            "old_policy_decay": decay,
            "task": nft_config.task,
            "method": nft_config.method,
            "parameterization": nft_config.parameterization,
            "center_step": center_step,
            "center_pretrain_steps": nft_config.center_pretrain_steps,
            "center_updates_per_outer": (
                nft_config.center_updates_per_outer
                if nft_config.method == "shifted_center" else 0
            ),
            "policy_updates_per_outer": nft_config.policy_updates_per_outer,
            "center_training_records_per_outer": (
                nft_config.center_training_records_per_outer
                if nft_config.method == "shifted_center" else 0
            ),
            "center_accumulated_records": center_record_count,
            "policy_training_records_per_outer": (
                nft_config.policy_training_records_per_outer
            ),
            "policy_accumulated_records": policy_record_count,
            "train_batch_size": nft_config.train_batch_size,
            "center_backward_microbatches_per_outer": (
                center_backward_microbatches
            ),
            "policy_backward_microbatches_per_outer": (
                policy_backward_microbatches
            ),
            "gradient_reduction": "record_weighted_mean",
            "optimizer_update_semantics": "one_effective_batch_per_outer",
            "trainable_shared_parameters": trainable_count,
            "frozen_decoder_specific_parameters": frozen_decoder_count,
            "cfg_scale": nft_config.rollout_cfg_scale,
            "training_input_cfg_scale": training_input_cfg_scale(nft_config),
            "source_cfg_mode": nft_config.source_cfg_mode,
            "source_branch_loss_reduction": "mean" if nft_config.source_cfg_mode == "cfg1_dual" else "single",
            "self_cond_cfg_scale": nft_config.self_cond_cfg_scale,
            "center_self_cond_cfg_scale": nft_config.center_self_cond_cfg_scale,
            "training_self_cond_cfg_distribution": (
                nft_config.training_self_cond_cfg_distribution
            ),
            "training_self_cond_cfg_min": nft_config.training_self_cond_cfg_min,
            "training_self_cond_cfg_max": nft_config.training_self_cond_cfg_max,
            "training_velocity_reconstruction": (
                nft_config.training_velocity_reconstruction
            ),
            "unconditional_update_every": nft_config.unconditional_update_every,
            "rollout_self_cond_cfg_scale": nft_config.rollout_self_cond_cfg_scale,
            "rollout_field_mode": nft_config.rollout_field_mode,
            "rollout_external_sc_scale": resolve_rollout_sccfg_settings(
                nft_config
            )[1],
            "self_condition_mode": nft_config.self_condition_mode,
            "branch_loss_reduction": "sum" if nft_config.self_condition_mode == "paired" else "single",
            "rollout_sampling_method": nft_config.rollout_sampling_method,
            "rollout_sde_gamma": nft_config.rollout_sde_gamma,
            "reward_type": nft_config.reward_type,
            "reward_combination_mode": nft_config.reward_combination_mode,
            "self_cond_update_probability": (
                nft_config.self_cond_update_probability
            ),
            "flow_matching_target": "classic_ot_endpoint_minus_noise",
            "decoder_ce_weight": nft_config.decoder_ce_weight,
            "formal_experiment": nft_config.formal_experiment,
            "elapsed_seconds": time.time() - outer_start,
            **rollout.metrics,
            **center_host_metrics,
            **policy_host_metrics,
        }
        if nft_config.task == "owt":
            record.update(
                {
                    "online_ppl": float(
                        rollout.metrics["reward_model_ppl"]
                    ),
                    "online_entropy": float(
                        rollout.metrics["reward_model_unigram_entropy"]
                    ),
                    "online_num_samples": int(nft_config.rollout_batch_size),
                    "online_policy_source": "old_policy_before_update",
                }
            )
        _append_jsonl(metrics_path, record)
        preview_count = min(4, len(rollout.hypotheses))
        _append_jsonl(
            previews_path,
            {
                "outer_step": completed_outer,
                "prompt_indices": prompt_indices,
                "hypotheses": rollout.hypotheses[:preview_count],
                "references": (
                    None
                    if rollout.references is None
                    else rollout.references[:preview_count]
                ),
                "rewards": rollout.buffer["rewards"][:preview_count].tolist(),
                "sample_entropies": (
                    None
                    if "sample_entropies" not in rollout.buffer
                    else rollout.buffer["sample_entropies"][:preview_count].tolist()
                ),
                "entropy_rewards": (
                    None
                    if "entropy_rewards" not in rollout.buffer
                    else rollout.buffer["entropy_rewards"][:preview_count].tolist()
                ),
            },
        )
        log_for_0(record)
        if wandb_run is not None:
            wandb_run.log(record, step=completed_outer)

        if (
            not capacity_probe
            and _checkpoint_due(completed_outer, nft_config, full_evaluation_plan)
        ):
            current_host = jax.device_get(jax_utils.unreplicate(current_state))
            center_host = (
                None
                if center_state is None
                else jax.device_get(jax_utils.unreplicate(center_state))
            )
            old_host = jax.device_get(jax_utils.unreplicate(old_params))
            payload = build_checkpoint_payload(
                current_state=current_host,
                center_state=center_host,
                old_params=old_host,
                reward_state=reward_state,
                rng=rng,
                outer_step=completed_outer,
                center_pretrain_step=completed_center_pretrain,
                phase="online",
                parameterization=PARAMETERIZATION,
            )
            checkpoint_path = save_online_checkpoint(
                output_dir=str(output_dir),
                payload=payload,
                keep=nft_config.keep_checkpoints,
                method=nft_config.method,
                base_checkpoint=nft_config.pretrained_checkpoint,
                base_checkpoint_revision=(
                    nft_config.pretrained_checkpoint_revision
                ),
                config_path=nft_config.config_path,
                keep_every_n_steps=nft_config.keep_every_n_steps,
                self_condition_mode=nft_config.self_condition_mode,
                source_cfg_mode=nft_config.source_cfg_mode,
                theta_ema_decay=nft_config.theta_ema_decay,
            )
            log_for_0(f"Saved full-model checkpoint: {checkpoint_path}")

        if _wmt_evaluation_due(completed_outer, nft_config):
            evaluation_function = (
                _run_wmt_evaluation
                if nft_config.task == "wmt"
                else _run_xsum_evaluation
            )
            evaluation_function(
                model_apply_fn=model.apply,
                encoder_apply_fn=encoder_model.apply,
                policy_params_replicated=current_state.params,
                theta_ema_params_replicated=current_state.ema_params,
                encoder_params_replicated=encoder_params_replicated,
                dataset=validation_dataset,
                outer_step=completed_outer,
                tokenizer=tokenizer,
                elf_config=elf_config,
                nft_config=nft_config,
                output_path=(
                    wmt_evaluations_path
                    if nft_config.task == "wmt"
                    else xsum_evaluations_path
                ),
            )

        if _owt_sccfg_evaluation_due(completed_outer, nft_config):
            _run_sccfg_evaluation_suite(
                model_apply_fn=model.apply,
                policy_params_replicated=current_state.params,
                outer_step=completed_outer,
                tokenizer=tokenizer,
                reward_scorer=reward_scorer,
                embedding_dim=encoder_config.d_model,
                elf_config=elf_config,
                nft_config=nft_config,
                output_path=sccfg_evaluations_path,
            )

        if (
            full_evaluation_plan is not None
            and completed_outer == full_evaluation_plan.outer_step
            and full_evaluation_plan.outer_step
            not in completed_full_evaluations
        ):
            full_record = _run_owt_full_evaluation(
                plan=full_evaluation_plan,
                actual_outer_step=completed_outer,
                trigger="scheduled_midpoint",
                functions=rollout_functions,
                policy_params_replicated=current_state.params,
                tokenizer=tokenizer,
                reward_scorer=reward_scorer,
                embedding_dim=encoder_config.d_model,
                elf_config=elf_config,
                nft_config=nft_config,
                output_path=full_evaluations_path,
            )
            completed_full_evaluations.add(full_evaluation_plan.outer_step)
            if wandb_run is not None:
                wandb_run.log(
                    _wandb_full_evaluation_payload(full_record),
                    step=completed_outer,
                )

    if nft_config.evaluation_at_end and nft_config.task == "owt":
        _run_sccfg_evaluation_suite(
            model_apply_fn=model.apply, policy_params_replicated=current_state.params,
            outer_step=nft_config.max_outer_steps, tokenizer=tokenizer, reward_scorer=reward_scorer,
            embedding_dim=encoder_config.d_model, elf_config=elf_config, nft_config=nft_config,
            output_path=sccfg_evaluations_path,
        )
    if wandb_run is not None:
        wandb_run.finish()
    _log_device_memory("completed")
    log_for_0("ELF NFT training complete")


def main():
    _configure_logging()
    args = parse_args()
    elf_config, nft_config, _ = load_experiment_config(args.config)
    _validate_task_selection(args.task, nft_config)
    nft_config = nft_config.with_overrides(
        method=args.method,
        output_dir=args.output_dir,
        max_outer_steps=args.max_outer_steps,
        samples_per_prompt=args.samples_per_prompt,
        rollout_generation_batch_size=args.rollout_generation_batch_size,
        train_batch_size=args.train_batch_size,
        reward_batch_size=args.reward_batch_size,
        self_condition_mode=args.self_condition_mode,
    )
    _validate_device_geometry(nft_config)
    if args.capacity_probe:
        if args.resume is not None:
            raise ValueError("--capacity_probe cannot resume a checkpoint")
        if (
            nft_config.formal_experiment
            or nft_config.max_outer_steps != 1
            or nft_config.evaluation_cases
            or nft_config.evaluation_at_start
            or nft_config.evaluation_at_end
            or nft_config.evaluation_every_n_steps
            or nft_config.checkpoint_rolling_every
        ):
            raise ValueError(
                "--capacity_probe requires a non-formal one-step run with no "
                "evaluation or rolling checkpoints"
            )
    if args.validate_only:
        full_evaluation_plan = _build_midpoint_full_evaluation_plan(
            elf_config, nft_config
        )
        _log_header(elf_config, nft_config, full_evaluation_plan)
        log_for_0("Configuration validation complete")
        return
    if args.data_preflight_only:
        if nft_config.task not in {"wmt", "xsum"}:
            raise ValueError(
                "--data_preflight_only is valid only for conditional tasks"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            elf_config.tokenizer_name or elf_config.encoder_model_name,
            revision=getattr(elf_config, "tokenizer_revision", None),
        )
        pad_token_id = get_pad_token_id(tokenizer, elf_config.pad_token)
        if nft_config.task == "wmt":
            _, _, _, report = _load_and_preflight_wmt_training_data(
                elf_config,
                nft_config,
                tokenizer,
                pad_token_id=pad_token_id,
            )
        else:
            train_dataset, validation_dataset, settings = (
                load_pinned_xsum_datasets(elf_config, nft_config)
            )
            report = preflight_xsum_datasets(
                train_dataset,
                validation_dataset,
                tokenizer=tokenizer,
                elf_config=elf_config,
                nft_config=nft_config,
                pad_token_id=pad_token_id,
                settings=settings,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        log_for_0(
            f"{nft_config.task.upper()} data preflight complete; "
            "model weights were not loaded"
        )
        return
    run_training(
        elf_config,
        nft_config,
        resume=args.resume,
        capacity_probe=args.capacity_probe,
    )


if __name__ == "__main__":
    main()
