#!/usr/bin/env python
"""Evaluate one released or NFT-adapted OWT policy over native SC-CFG weights."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from transformers import AutoTokenizer

from baselines.elf_nft.checkpointing import (
    MARKER_FILENAME,
    load_policy_params_from_checkpoint,
    resolve_resume_checkpoint,
    sha256_file,
    validate_resume_metadata,
)
from baselines.elf_nft.config import (
    ELF_NFT_ASSET_ROOT,
    SCCFGEvaluationCase,
    load_experiment_config,
    validate_nft_config,
)
from baselines.elf_nft.full_model import validate_full_model_params
from baselines.elf_nft.reward import CausalLMLogLikelihoodReward
from baselines.elf_nft.scripts.train_nft_elf import (
    FULL_EVALUATION_SEED_TAG,
    _build_model_and_pretrained_params,
    _configure_logging,
    _reconcile_jsonl_to_checkpoint,
    _require_frozen_decoder_reference,
    _run_sccfg_evaluation_suite,
)
from utils.data_utils import get_pad_token_id
from utils.logging_utils import log_for_0


DEFAULT_CFG_SCALES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
MANIFEST_FILENAME = "evaluation_manifest.json"
RESULTS_FILENAME = "results.jsonl"
COMPLETE_FILENAME = ".complete"
POLICY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one OWT policy with common random numbers over native "
            "ELF self-conditioning CFG weights"
        )
    )
    parser.add_argument("--config", required=True, help="Original NFT config")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Published online checkpoint; omit to evaluate released pretrained EMA",
    )
    parser.add_argument("--policy-id", required=True, help="Filesystem-safe policy ID")
    parser.add_argument("--display-name", required=True, help="Plot/display label")
    parser.add_argument(
        "--output-dir", required=True, help="Evaluation directory under the group asset root"
    )
    parser.add_argument(
        "--cfg-scales",
        nargs="+",
        type=float,
        default=list(DEFAULT_CFG_SCALES),
        help="Native model SC-CFG values; external extrapolation stays at one",
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate provenance and geometry without loading model assets",
    )
    return parser.parse_args(argv)


def cfg_case_name(scale: float) -> str:
    """Return a stable native-field case name such as ``native_w2p5``."""

    value = float(scale)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("CFG scales must be finite and positive")
    text = format(value, ".12g")
    if "e" in text.lower():
        raise ValueError("CFG scale cannot be represented as a simple decimal label")
    return "native_w" + text.replace(".", "p")


def build_cfg_cases(scales) -> tuple[SCCFGEvaluationCase, ...]:
    values = tuple(float(scale) for scale in scales)
    if not values:
        raise ValueError("at least one CFG scale is required")
    if len(set(values)) != len(values):
        raise ValueError("CFG scales must be unique")
    return tuple(
        SCCFGEvaluationCase(
            name=cfg_case_name(scale),
            model_cfg_scale=scale,
            external_cfg_scale=1.0,
        )
        for scale in values
    )


def evaluation_seed(seed: int) -> int:
    """Match the independent fixed seed used by training-time full evaluation."""

    return int(
        np.random.SeedSequence([seed, FULL_EVALUATION_SEED_TAG])
        .generate_state(1, dtype=np.uint32)[0]
    )


def _require_group_child(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    root = ELF_NFT_ASSET_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"output-dir must be under {ELF_NFT_ASSET_ROOT}, got {candidate}"
        ) from error
    if candidate == root:
        raise ValueError("output-dir must be a child of the group asset root")
    return candidate


def _read_checkpoint_marker(checkpoint_path: str) -> dict:
    resolved = Path(resolve_resume_checkpoint(checkpoint_path)).resolve()
    marker_path = resolved.parent / MARKER_FILENAME
    with marker_path.open("r", encoding="utf-8") as handle:
        marker = json.load(handle)
    published = Path(marker.get("checkpoint_path", "")).expanduser().resolve()
    if published != resolved:
        raise ValueError(
            "evaluation checkpoint is not the checkpoint published by its marker: "
            f"{resolved} != {published}"
        )
    return marker


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _publish_or_validate_manifest(output_dir: Path, manifest: dict) -> None:
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != manifest:
            raise ValueError("existing evaluation manifest differs from this request")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("nonempty evaluation directory has no provenance manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(manifest_path, manifest)


def _load_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid evaluation result at {path}:{line_number}"
                ) from error
    return records


def validate_result_records(
    records: list[dict], *, cases, outer_step: int, num_samples: int,
    generation_batch_size: int, seed: int,
) -> None:
    expected_by_name = {case.name: case for case in cases}
    actual_by_name = {}
    for record in records:
        name = record.get("name")
        if name in actual_by_name:
            raise ValueError(f"duplicate evaluation result for {name!r}")
        actual_by_name[name] = record
    if set(actual_by_name) != set(expected_by_name):
        raise ValueError(
            "evaluation cases differ: "
            f"missing={sorted(set(expected_by_name) - set(actual_by_name))}, "
            f"extra={sorted(set(actual_by_name) - set(expected_by_name))}"
        )
    for name, case in expected_by_name.items():
        record = actual_by_name[name]
        expected = {
            "outer_step": outer_step,
            "model_cfg_scale": case.model_cfg_scale,
            "external_cfg_scale": 1.0,
            "num_samples": num_samples,
            "generation_batch_size": generation_batch_size,
            "seed": seed,
            "evaluation_type": "sccfg_sweep",
        }
        mismatches = {
            key: (record.get(key), value)
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatches:
            raise ValueError(f"saved result settings differ for {name}: {mismatches}")
        ppl = record.get("ppl")
        entropy = record.get("entropy")
        if not isinstance(ppl, (int, float)) or not math.isfinite(ppl) or ppl <= 0:
            raise ValueError(f"invalid PPL for {name}: {ppl!r}")
        if (
            not isinstance(entropy, (int, float))
            or not math.isfinite(entropy)
            or entropy < 0
        ):
            raise ValueError(f"invalid entropy for {name}: {entropy!r}")


def _build_manifest(*, args, nft_config, cases, checkpoint_path, outer_step):
    checkpoint_training = None
    policy_source = "pretrained_ema"
    if checkpoint_path is not None:
        checkpoint_training = {
            "method": nft_config.method,
            "self_condition_mode": nft_config.self_condition_mode,
            "training_self_cond_cfg_scale": nft_config.self_cond_cfg_scale,
        }
        policy_source = "current_policy_checkpoint"
    return {
        "schema_version": 1,
        "policy_id": args.policy_id,
        "display_name": args.display_name,
        "policy_source": policy_source,
        "outer_step": outer_step,
        "checkpoint_path": checkpoint_path,
        "checkpoint_training": checkpoint_training,
        "base_checkpoint": nft_config.pretrained_checkpoint,
        "source_config_path": str(Path(nft_config.config_path).resolve()),
        "source_config_sha256": sha256_file(nft_config.config_path),
        "evaluation": {
            "model_cfg_scales": [case.model_cfg_scale for case in cases],
            "external_cfg_scale": 1.0,
            "field_mode": "native",
            "num_samples": args.num_samples,
            "generation_batch_size": args.generation_batch_size,
            "seed": evaluation_seed(nft_config.seed),
            "sampling_method": nft_config.rollout_sampling_method,
            "num_sampling_steps": nft_config.rollout_steps,
            "time_schedule": nft_config.rollout_time_schedule,
            "sde_gamma": nft_config.rollout_sde_gamma,
            "input_cfg_scale": nft_config.rollout_cfg_scale,
            "reward_model_name": nft_config.reward_model_name,
        },
    }


def _resolve_source(args, nft_config) -> tuple[str | None, int]:
    if args.checkpoint is None:
        return None, 0
    validate_resume_metadata(
        args.checkpoint,
        expected_method=nft_config.method,
        expected_base_checkpoint=nft_config.pretrained_checkpoint,
        expected_config_path=nft_config.config_path,
        expected_parameterization=nft_config.parameterization,
        expected_self_condition_mode=nft_config.self_condition_mode,
    )
    checkpoint_path = str(Path(resolve_resume_checkpoint(args.checkpoint)).resolve())
    marker = _read_checkpoint_marker(checkpoint_path)
    outer_step = int(marker.get("outer_step", 0))
    if outer_step <= 0:
        raise ValueError("checkpoint marker outer_step must be positive")
    return checkpoint_path, outer_step


def _validate_evaluation_geometry(num_samples: int, batch_size: int) -> None:
    if num_samples <= 0 or batch_size <= 0 or num_samples % batch_size:
        raise ValueError(
            "num-samples must be positive and divisible by generation-batch-size"
        )
    devices = jax.local_device_count()
    if devices <= 0 or batch_size % devices:
        raise ValueError(
            f"generation batch {batch_size} must divide across {devices} local devices"
        )


def run_evaluation(args) -> None:
    if not POLICY_ID_PATTERN.fullmatch(args.policy_id):
        raise ValueError("policy-id must be one alphanumeric path component")
    if not args.display_name.strip():
        raise ValueError("display-name must be nonempty")
    output_dir = _require_group_child(args.output_dir)
    elf_config, nft_config, _ = load_experiment_config(args.config)
    if nft_config.task != "owt":
        raise ValueError("SC-CFG field evaluation currently requires an OWT config")
    cases = build_cfg_cases(args.cfg_scales)
    if any(
        not elf_config.self_cond_cfg_min
        <= case.model_cfg_scale
        <= elf_config.self_cond_cfg_max
        for case in cases
    ):
        raise ValueError(
            "CFG sweep must remain inside the ELF pretraining range "
            f"[{elf_config.self_cond_cfg_min}, {elf_config.self_cond_cfg_max}]"
        )
    eval_config = replace(
        nft_config,
        output_dir=str(output_dir),
        rollout_generation_batch_size=args.generation_batch_size,
        evaluation_cases=cases,
        evaluation_num_samples=args.num_samples,
        evaluation_at_start=False,
        evaluation_at_end=True,
        midpoint_evaluation=False,
    )
    validate_nft_config(eval_config)
    _validate_evaluation_geometry(args.num_samples, args.generation_batch_size)
    checkpoint_path, outer_step = _resolve_source(args, nft_config)
    manifest = _build_manifest(
        args=args,
        nft_config=nft_config,
        cases=cases,
        checkpoint_path=checkpoint_path,
        outer_step=outer_step,
    )
    if args.validate_only:
        log_for_0(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        log_for_0("CFG field evaluation validation complete; no assets loaded and no files written")
        return

    _publish_or_validate_manifest(output_dir, manifest)
    results_path = output_dir / RESULTS_FILENAME
    complete_path = output_dir / COMPLETE_FILENAME
    if complete_path.is_file():
        records = _load_results(results_path)
        validate_result_records(
            records,
            cases=cases,
            outer_step=outer_step,
            num_samples=args.num_samples,
            generation_batch_size=args.generation_batch_size,
            seed=evaluation_seed(nft_config.seed),
        )
        log_for_0(f"Skipping completed CFG sweep: {output_dir}")
        return

    kept, dropped = _reconcile_jsonl_to_checkpoint(
        results_path, outer_step, group_by_case=True
    )
    if kept or dropped:
        log_for_0(f"Reconciled partial CFG sweep: kept={kept}, dropped={dropped}")

    rng = jax.random.PRNGKey(nft_config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        elf_config.tokenizer_name or elf_config.encoder_model_name
    )
    get_pad_token_id(tokenizer, elf_config.pad_token)
    model, encoder_config, _, base_params, pretrained_step, _ = (
        _build_model_and_pretrained_params(
            elf_config, nft_config, tokenizer, rng
        )
    )
    validate_full_model_params(base_params)
    policy_params = base_params
    actual_outer_step = 0
    if checkpoint_path is not None:
        policy_params, actual_outer_step = load_policy_params_from_checkpoint(
            checkpoint_path
        )
        policy_params = jax.tree_util.tree_map(jnp.asarray, policy_params)
        validate_full_model_params(policy_params)
        _require_frozen_decoder_reference(
            base_params, {"evaluation_policy": policy_params}
        )
    if actual_outer_step != outer_step:
        raise ValueError(
            f"checkpoint payload outer_step={actual_outer_step} differs from "
            f"published marker outer_step={outer_step}"
        )
    policy_params_replicated = jax_utils.replicate(policy_params)
    reward_scorer = CausalLMLogLikelihoodReward(
        model_name=nft_config.reward_model_name,
        batch_size=nft_config.reward_batch_size,
        max_length=nft_config.reward_max_length,
        empty_text_reward=nft_config.empty_text_reward,
    )
    log_for_0(
        f"Evaluating policy={args.policy_id} source={manifest['policy_source']} "
        f"outer_step={outer_step} native_cfg_scales="
        f"{[case.model_cfg_scale for case in cases]} "
        f"samples={args.num_samples} common_seed={evaluation_seed(nft_config.seed)}"
    )
    _run_sccfg_evaluation_suite(
        model_apply_fn=model.apply,
        policy_params_replicated=policy_params_replicated,
        outer_step=outer_step,
        tokenizer=tokenizer,
        reward_scorer=reward_scorer,
        embedding_dim=encoder_config.d_model,
        elf_config=elf_config,
        nft_config=eval_config,
        output_path=results_path,
    )
    records = _load_results(results_path)
    validate_result_records(
        records,
        cases=cases,
        outer_step=outer_step,
        num_samples=args.num_samples,
        generation_batch_size=args.generation_batch_size,
        seed=evaluation_seed(nft_config.seed),
    )
    by_name = {record["name"]: record for record in records}
    completion = {
        "policy_id": args.policy_id,
        "display_name": args.display_name,
        "policy_source": manifest["policy_source"],
        "outer_step": outer_step,
        "pretrained_checkpoint_step": int(pretrained_step),
        "num_results": len(records),
        "results": [
            {
                "name": case.name,
                "model_cfg_scale": case.model_cfg_scale,
                "ppl": by_name[case.name]["ppl"],
                "entropy": by_name[case.name]["entropy"],
            }
            for case in cases
        ],
    }
    _atomic_write_json(complete_path, completion)
    log_for_0(f"Completed CFG sweep: {output_dir}")


def main():
    _configure_logging()
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
