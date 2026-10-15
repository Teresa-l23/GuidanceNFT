#!/usr/bin/env python
"""Validate SD3.5 reward-evaluation outputs and compatible legacy provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
PARTIAL_RESULTS_RE = re.compile(r"^\.partial-results-rank-([0-9]{5})\.json$")


class EvaluationValidationError(ValueError):
    """Raised when an evaluation output or its provenance is incompatible."""


@dataclass(frozen=True)
class EvaluationContract:
    """Expected immutable inputs and observable output metadata for one evaluation."""

    output_dir: Path
    checkpoint_dir: Path
    base_model: Path
    target_run_id: str
    checkpoint_step: int
    training_method: str
    cfg_mode: str
    dataset: str
    reward_names: tuple[str, ...]
    batch_size_per_rank: int
    expected_samples: int
    save_images: bool
    eval_steps: int
    resolution: int
    seed: int
    guidance_scale: float
    world_size: int = 2
    mixed_precision: str = "fp16"
    sampler: str = "diffusers"
    deterministic: bool = True
    max_sequence_length: int = 256
    noise_level: float | None = None
    cfg_unconditional_source: str = "same_model"


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise EvaluationValidationError(f"missing regular input file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        raise EvaluationValidationError(f"missing {description}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationValidationError(
            f"invalid {description} at {path}: {error}"
        ) from error


def _decode_shell_word(raw_value: str, *, path: Path, line_number: int) -> str:
    try:
        words = shlex.split(raw_value, posix=True)
    except ValueError as error:
        raise EvaluationValidationError(
            f"invalid shell assignment in {path}:{line_number}: {error}"
        ) from error
    if len(words) != 1:
        raise EvaluationValidationError(
            f"shell assignment in {path}:{line_number} must encode one value"
        )
    return words[0]


def _read_shell_assignments(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise EvaluationValidationError(f"missing provenance assignments: {path}")
    assignments: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        if "=" not in raw_line:
            raise EvaluationValidationError(
                f"invalid provenance assignment in {path}:{line_number}"
            )
        key, raw_value = raw_line.split("=", 1)
        if not ENV_KEY_RE.fullmatch(key) or key in assignments:
            raise EvaluationValidationError(
                f"invalid or duplicate provenance key in {path}:{line_number}: {key!r}"
            )
        assignments[key] = _decode_shell_word(
            raw_value, path=path, line_number=line_number
        )
    return assignments


def _read_manifest_digests(path: Path) -> set[str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise EvaluationValidationError(f"missing input digest manifest: {path}")
    digests: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not SHA256_RE.fullmatch(fields[0]):
            raise EvaluationValidationError(
                f"invalid input digest manifest entry in {path}:{line_number}"
            )
        digests.add(fields[0])
    return digests


def _require_assignments(
    actual: dict[str, str], expected: dict[str, str], *, path: Path
) -> None:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            raise EvaluationValidationError(
                f"provenance mismatch for {key} in {path}: "
                f"expected={expected_value!r}, got={actual_value!r}"
            )


def _require_optional_assignment(
    actual: dict[str, str], key: str, expected_value: str, *, path: Path
) -> None:
    if key in actual and actual[key] != expected_value:
        raise EvaluationValidationError(
            f"provenance mismatch for optional {key} in {path}: "
            f"expected={expected_value!r}, got={actual[key]!r}"
        )


def _require_guidance_assignment(
    actual: dict[str, str], expected_value: float, *, path: Path, optional: bool
) -> None:
    if "GUIDANCE_SCALE" not in actual:
        if optional:
            return
        raise EvaluationValidationError(
            f"provenance mismatch for GUIDANCE_SCALE in {path}: "
            f"expected={expected_value!r}, got=None"
        )
    try:
        recorded_guidance = float(actual["GUIDANCE_SCALE"])
    except ValueError as error:
        raise EvaluationValidationError(
            f"invalid GUIDANCE_SCALE in {path}: {actual['GUIDANCE_SCALE']!r}"
        ) from error
    if not math.isclose(
        recorded_guidance,
        expected_value,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        qualifier = "optional " if optional else ""
        raise EvaluationValidationError(
            f"provenance mismatch for {qualifier}GUIDANCE_SCALE in {path}: "
            f"expected={expected_value!r}, got={recorded_guidance!r}"
        )


def _resolve_existing(path: Path, description: str) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise EvaluationValidationError(
            f"cannot resolve {description} {path}: {error}"
        ) from error


def _require_metadata_value(
    metadata: dict[str, Any], key: str, expected_value: Any
) -> None:
    actual_value = metadata.get(key)
    if isinstance(expected_value, float):
        if (
            isinstance(actual_value, bool)
            or not isinstance(actual_value, (int, float))
            or not math.isclose(
                float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise EvaluationValidationError(
                f"metadata mismatch for {key}: "
                f"expected={expected_value!r}, got={actual_value!r}"
            )
    elif actual_value != expected_value:
        raise EvaluationValidationError(
            f"metadata mismatch for {key}: "
            f"expected={expected_value!r}, got={actual_value!r}"
        )


def validate_output(contract: EvaluationContract) -> dict[str, Any]:
    """Validate all result artifacts against the requested evaluation contract."""

    if contract.cfg_unconditional_source not in {"same_model", "pretrained_base"}:
        raise EvaluationValidationError(
            "cfg_unconditional_source must be same_model or pretrained_base"
        )
    if (
        contract.cfg_unconditional_source == "pretrained_base"
        and contract.guidance_scale <= 1.0
    ):
        raise EvaluationValidationError(
            "pretrained_base unconditional evaluation requires guidance_scale > 1"
        )

    output_dir = contract.output_dir
    results_path = output_dir / "evaluation_results.jsonl"
    if not results_path.is_file() or results_path.stat().st_size == 0:
        raise EvaluationValidationError(f"missing evaluation results: {results_path}")

    record_count = 0
    try:
        with results_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise EvaluationValidationError(
                        f"invalid JSON result at {results_path}:{line_number}: {error}"
                    ) from error
                sample_id = record.get("sample_id") if isinstance(record, dict) else None
                if type(sample_id) is not int or sample_id != record_count:
                    raise EvaluationValidationError(
                        "expected ordered sample IDs "
                        f"0..{contract.expected_samples - 1}; "
                        f"line {line_number} has sample_id={sample_id!r}"
                    )
                record_count += 1
    except (OSError, UnicodeDecodeError) as error:
        raise EvaluationValidationError(
            f"cannot read evaluation results {results_path}: {error}"
        ) from error
    if record_count != contract.expected_samples:
        raise EvaluationValidationError(
            f"expected {contract.expected_samples} evaluation records, got {record_count}"
        )

    scores = _load_json(output_dir / "average_scores.json", "average scores")
    if not isinstance(scores, dict):
        raise EvaluationValidationError("average scores must be a JSON object")
    for reward_name in contract.reward_names:
        reward_value = scores.get(reward_name)
        if (
            isinstance(reward_value, bool)
            or not isinstance(reward_value, (int, float))
            or not math.isfinite(float(reward_value))
        ):
            raise EvaluationValidationError(
                f"missing or non-finite aggregate reward {reward_name!r}: {scores}"
            )

    metadata = _load_json(output_dir / "evaluation_metadata.json", "metadata")
    if not isinstance(metadata, dict):
        raise EvaluationValidationError("evaluation metadata must be a JSON object")
    expected_metadata = {
        "dataset": contract.dataset,
        "reward_names": list(contract.reward_names),
        "batch_size_per_rank": contract.batch_size_per_rank,
        "world_size": contract.world_size,
        "expected_sample_count": contract.expected_samples,
        "num_inference_steps": contract.eval_steps,
        "guidance_scale": contract.guidance_scale,
        "resolution": contract.resolution,
        "mixed_precision": contract.mixed_precision,
        "seed": contract.seed,
    }
    for key, expected_value in expected_metadata.items():
        _require_metadata_value(metadata, key, expected_value)
    if contract.cfg_unconditional_source == "pretrained_base":
        hybrid_metadata = {
            "cfg_routing_schema": 1,
            "cfg_unconditional_source": "pretrained_base",
            "cfg_conditional_source": "checkpoint_lora",
            "cfg_conditional_adapter_name": "default",
            "cfg_transformer_batch_order": ["unconditional", "conditional"],
            "cfg_formula": (
                "v_pretrained_unconditional + guidance_scale * "
                "(v_checkpoint_conditional - v_pretrained_unconditional)"
            ),
        }
        for key, expected_value in hybrid_metadata.items():
            _require_metadata_value(metadata, key, expected_value)
        routing_counters = []
        for counter_name in (
            "cfg_rank0_transformer_forward_calls",
            "cfg_rank0_routed_sample_pairs",
        ):
            counter_value = metadata.get(counter_name)
            if type(counter_value) is not int or counter_value < 0:
                raise EvaluationValidationError(
                    f"metadata {counter_name} must be a non-negative integer"
                )
            routing_counters.append(counter_value)
        if (routing_counters[0] == 0) != (routing_counters[1] == 0):
            raise EvaluationValidationError(
                "rank-0 CFG routing counters must both be zero or both be positive"
            )
    elif "cfg_unconditional_source" in metadata:
        _require_metadata_value(
            metadata,
            "cfg_unconditional_source",
            "same_model",
        )

    generation_metadata = {
        "sampler": contract.sampler,
        "deterministic": contract.deterministic,
        "max_sequence_length": contract.max_sequence_length,
    }
    if contract.noise_level is not None:
        generation_metadata["noise_level"] = contract.noise_level
    if contract.sampler == "diffusers":
        # Outputs predating the explicit sampler contract used the released
        # Diffusers call with these defaults.  Its metadata used ``sampler``
        # for the distributed data-sharding policy rather than the diffusion
        # solver, so ``strided_without_padding`` is the expected legacy value.
        # Any other explicit solver value must still match ``diffusers``.
        metadata_sampler = metadata.get("sampler")
        if metadata_sampler not in (None, "diffusers", "strided_without_padding"):
            _require_metadata_value(metadata, "sampler", contract.sampler)
        for key, expected_value in generation_metadata.items():
            if key == "sampler":
                continue
            if key in metadata:
                _require_metadata_value(metadata, key, expected_value)
    else:
        for key, expected_value in generation_metadata.items():
            _require_metadata_value(metadata, key, expected_value)

    checkpoint_value = metadata.get("checkpoint_path")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise EvaluationValidationError("evaluation metadata has no checkpoint_path")
    if _resolve_existing(Path(checkpoint_value), "metadata checkpoint") != _resolve_existing(
        contract.checkpoint_dir, "requested checkpoint"
    ):
        raise EvaluationValidationError("evaluation metadata names the wrong checkpoint")

    base_model_value = metadata.get("base_model")
    if not isinstance(base_model_value, str) or not base_model_value:
        raise EvaluationValidationError("evaluation metadata has no base_model")
    if _resolve_existing(Path(base_model_value), "metadata base model") != _resolve_existing(
        contract.base_model, "requested base model"
    ):
        raise EvaluationValidationError("evaluation metadata names the wrong base model")

    images_dir = output_dir / "images"
    if contract.save_images:
        if not images_dir.is_dir():
            raise EvaluationValidationError(f"missing image directory: {images_dir}")
        image_count = sum(1 for path in images_dir.glob("*.jpg") if path.is_file())
        if image_count != contract.expected_samples:
            raise EvaluationValidationError(
                f"expected {contract.expected_samples} images, found {image_count}"
            )
    elif images_dir.exists():
        raise EvaluationValidationError(
            "image directory exists although save_images=false"
        )

    return {
        "records": record_count,
        "rewards": list(contract.reward_names),
        "scores": scores,
    }


def validate_request_provenance(
    contract: EvaluationContract,
    core_input_files: Sequence[Path],
    *,
    allow_legacy_fields: bool = False,
) -> dict[str, Any]:
    """Validate the immutable evaluation inputs recorded for an existing output.

    The core input list deliberately contains only files consumed by generation
    or scoring.  Training-only helpers may still be retained in the full audit
    manifest, but changes to those helpers cannot invalidate results from a
    fixed checkpoint.

    ``allow_legacy_fields`` is reserved for complete pre-CFG evaluations.  It
    permits missing fields that did not exist in the old schema, while rejecting
    every optional field that is present with a conflicting value.
    """

    if not core_input_files:
        raise EvaluationValidationError(
            "request provenance validation requires core input files"
        )

    provenance_dir = contract.output_dir / "provenance"
    input_manifest = provenance_dir / "input-sha256.txt"
    recorded_digests = _read_manifest_digests(input_manifest)
    for input_path in core_input_files:
        digest = _sha256(input_path)
        if digest not in recorded_digests:
            raise EvaluationValidationError(
                f"provenance lacks the current digest for input: {input_path}"
            )

    evaluation_env_path = provenance_dir / "evaluation.env"
    evaluation_env = _read_shell_assignments(evaluation_env_path)
    _require_assignments(
        evaluation_env,
        {
            "TARGET_RUN_ID": contract.target_run_id,
            "CHECKPOINT_STEP": str(contract.checkpoint_step),
            "DATASET": contract.dataset,
            "REWARDS": " ".join(contract.reward_names),
            "BATCH_SIZE_PER_RANK": str(contract.batch_size_per_rank),
            "EXPECTED_SAMPLES": str(contract.expected_samples),
            "SAVE_IMAGES": "yes" if contract.save_images else "no",
        },
        path=evaluation_env_path,
    )
    if allow_legacy_fields:
        _require_optional_assignment(
            evaluation_env,
            "TRAINING_METHOD",
            contract.training_method,
            path=evaluation_env_path,
        )
        _require_optional_assignment(
            evaluation_env,
            "CFG_MODE",
            contract.cfg_mode,
            path=evaluation_env_path,
        )
    else:
        _require_assignments(
            evaluation_env,
            {
                "TRAINING_METHOD": contract.training_method,
                "CFG_MODE": contract.cfg_mode,
            },
            path=evaluation_env_path,
        )
    if contract.cfg_unconditional_source == "pretrained_base":
        _require_assignments(
            evaluation_env,
            {"CFG_UNCONDITIONAL_SOURCE": "pretrained_base"},
            path=evaluation_env_path,
        )
    else:
        _require_optional_assignment(
            evaluation_env,
            "CFG_UNCONDITIONAL_SOURCE",
            "same_model",
            path=evaluation_env_path,
        )
    _require_guidance_assignment(
        evaluation_env,
        contract.guidance_scale,
        path=evaluation_env_path,
        optional=allow_legacy_fields,
    )
    optional_current_fields = {
        "BASE_MODEL": str(contract.base_model),
        "EVAL_STEPS": str(contract.eval_steps),
        "RESOLUTION": str(contract.resolution),
        "SEED": str(contract.seed),
        "WORLD_SIZE": str(contract.world_size),
        "MIXED_PRECISION": contract.mixed_precision,
        "MODEL_TYPE": "sd3",
    }
    for key, expected_value in optional_current_fields.items():
        _require_optional_assignment(
            evaluation_env, key, expected_value, path=evaluation_env_path
        )

    generation_fields = {
        "SAMPLER": contract.sampler,
        "DETERMINISTIC": "true" if contract.deterministic else "false",
        "MAX_SEQUENCE_LENGTH": str(contract.max_sequence_length),
    }
    if contract.noise_level is not None:
        generation_fields["NOISE_LEVEL"] = str(contract.noise_level)
    if contract.sampler == "diffusers":
        for key, expected_value in generation_fields.items():
            _require_optional_assignment(
                evaluation_env, key, expected_value, path=evaluation_env_path
            )
    else:
        _require_assignments(
            evaluation_env, generation_fields, path=evaluation_env_path
        )

    return {
        "core_inputs": len(core_input_files),
        "legacy_fields_allowed": allow_legacy_fields,
    }


def validate_partial_output(contract: EvaluationContract) -> dict[str, Any]:
    """Validate rank-local records that the evaluator will skip on resume."""

    output_dir = contract.output_dir
    if (output_dir / ".complete").exists():
        raise EvaluationValidationError(
            "partial-output validation cannot be used for a complete evaluation"
        )

    completed_ids: set[int] = set()
    records_by_rank: dict[int, int] = {}
    partial_paths = sorted(output_dir.glob(".partial-results-rank-*.json"))
    for partial_path in partial_paths:
        match = PARTIAL_RESULTS_RE.fullmatch(partial_path.name)
        if match is None:
            raise EvaluationValidationError(
                f"invalid rank-local partial result filename: {partial_path}"
            )
        rank = int(match.group(1))
        if rank >= contract.world_size:
            raise EvaluationValidationError(
                f"partial result rank {rank} is outside world size {contract.world_size}"
            )
        payload = _load_json(partial_path, "rank-local partial results")
        if not isinstance(payload, list):
            raise EvaluationValidationError(
                f"rank-local partial results must be a JSON list: {partial_path}"
            )

        rank_count = 0
        for record_index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise EvaluationValidationError(
                    f"invalid partial record at {partial_path}:{record_index}"
                )
            sample_id = record.get("sample_id")
            if (
                type(sample_id) is not int
                or sample_id < 0
                or sample_id >= contract.expected_samples
                or sample_id % contract.world_size != rank
                or sample_id in completed_ids
            ):
                raise EvaluationValidationError(
                    f"invalid or duplicate partial sample_id={sample_id!r} "
                    f"at {partial_path}:{record_index}"
                )
            expected_seed = (contract.seed + sample_id) % (2**63 - 1)
            sample_seed = record.get("sample_seed")
            if type(sample_seed) is not int or sample_seed != expected_seed:
                raise EvaluationValidationError(
                    f"partial sample {sample_id} has the wrong sample_seed"
                )
            scores = record.get("scores")
            if not isinstance(scores, dict):
                raise EvaluationValidationError(
                    f"partial sample {sample_id} has no score mapping"
                )
            for reward_name in contract.reward_names:
                reward_value = scores.get(reward_name)
                if (
                    isinstance(reward_value, bool)
                    or not isinstance(reward_value, (int, float))
                    or not math.isfinite(float(reward_value))
                ):
                    raise EvaluationValidationError(
                        f"partial sample {sample_id} has invalid reward "
                        f"{reward_name!r}: {reward_value!r}"
                    )

            if contract.save_images:
                expected_image = output_dir / "images" / f"{sample_id:05d}.jpg"
                image_value = record.get("image_path")
                if not isinstance(image_value, str) or not image_value:
                    raise EvaluationValidationError(
                        f"partial sample {sample_id} has no image_path"
                    )
                if _resolve_existing(
                    Path(image_value), "partial image"
                ) != _resolve_existing(expected_image, "expected partial image"):
                    raise EvaluationValidationError(
                        f"partial sample {sample_id} names the wrong image"
                    )
                if expected_image.stat().st_size == 0:
                    raise EvaluationValidationError(
                        f"partial sample {sample_id} has an empty image"
                    )
            elif "image_path" in record:
                raise EvaluationValidationError(
                    f"partial sample {sample_id} has an image although save_images=false"
                )

            completed_ids.add(sample_id)
            rank_count += 1
        records_by_rank[rank] = rank_count

    if not contract.save_images and (output_dir / "images").exists():
        raise EvaluationValidationError(
            "partial image directory exists although save_images=false"
        )

    return {
        "partial_records": len(completed_ids),
        "remaining_records": contract.expected_samples - len(completed_ids),
        "records_by_rank": records_by_rank,
    }


def validate_legacy_complete_output(
    contract: EvaluationContract, core_input_files: Sequence[Path]
) -> dict[str, Any]:
    """Validate a pre-request-hash complete output using its legacy witnesses.

    Legacy evaluations did not record ``TRAINING_METHOD``, ``CFG_MODE``,
    ``GUIDANCE_SCALE``, or ``cfg_modes.py``.  Missing new
    fields are accepted only for complete outputs whose old core input hashes,
    old request fields, completion paths, and result metadata all match.
    If an optional new field is present, it must still agree exactly.
    """

    output_dir = contract.output_dir
    complete_path = output_dir / ".complete"
    if not complete_path.is_file() or complete_path.stat().st_size == 0:
        raise EvaluationValidationError(
            "legacy compatibility is allowed only for a complete evaluation"
        )
    validate_request_provenance(
        contract, core_input_files, allow_legacy_fields=True
    )

    completion_env = _read_shell_assignments(complete_path)
    _require_assignments(
        completion_env,
        {
            "CHECKPOINT_DIR": str(
                _resolve_existing(contract.checkpoint_dir, "requested checkpoint")
            ),
            "OUTPUT_DIR": str(_resolve_existing(output_dir, "evaluation output")),
        },
        path=complete_path,
    )

    return validate_output(contract)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--target-run-id", required=True)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--training-method", required=True)
    parser.add_argument("--cfg-mode", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reward", action="append", dest="reward_names", required=True)
    parser.add_argument("--batch-size-per-rank", required=True, type=int)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--save-images", required=True, choices=("yes", "no"))
    parser.add_argument("--eval-steps", required=True, type=int)
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--guidance-scale", required=True, type=float)
    parser.add_argument("--world-size", default=2, type=int)
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--sampler", default="diffusers", choices=("diffusers", "dpm2"))
    parser.add_argument("--deterministic", default="yes", choices=("yes", "no"))
    parser.add_argument("--max-sequence-length", default=256, type=int)
    parser.add_argument("--noise-level", type=float)
    parser.add_argument(
        "--cfg-unconditional-source",
        default="same_model",
        choices=("same_model", "pretrained_base"),
    )
    provenance_mode = parser.add_mutually_exclusive_group()
    provenance_mode.add_argument("--request-provenance", action="store_true")
    provenance_mode.add_argument("--partial-output", action="store_true")
    provenance_mode.add_argument("--legacy-provenance", action="store_true")
    parser.add_argument("--core-input", action="append", type=Path, default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    contract = EvaluationContract(
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        base_model=args.base_model,
        target_run_id=args.target_run_id,
        checkpoint_step=args.checkpoint_step,
        training_method=args.training_method,
        cfg_mode=args.cfg_mode,
        dataset=args.dataset,
        reward_names=tuple(args.reward_names),
        batch_size_per_rank=args.batch_size_per_rank,
        expected_samples=args.expected_samples,
        save_images=args.save_images == "yes",
        eval_steps=args.eval_steps,
        resolution=args.resolution,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        world_size=args.world_size,
        mixed_precision=args.mixed_precision,
        sampler=args.sampler,
        deterministic=args.deterministic == "yes",
        max_sequence_length=args.max_sequence_length,
        noise_level=args.noise_level,
        cfg_unconditional_source=args.cfg_unconditional_source,
    )
    try:
        if args.request_provenance:
            summary = validate_request_provenance(contract, args.core_input)
        elif args.partial_output:
            summary = validate_partial_output(contract)
        elif args.legacy_provenance:
            summary = validate_legacy_complete_output(contract, args.core_input)
        else:
            summary = validate_output(contract)
    except (EvaluationValidationError, OSError) as error:
        print(f"Reward evaluation validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
