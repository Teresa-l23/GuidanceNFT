#!/usr/bin/env python3
"""Score a completed DrawBench JPEG set with one local paper reward.

The diffusion model is deliberately absent from this process.  A caller first
generates one canonical DrawBench set (normally with PickScore), then invokes
this tool once per additional scorer.  Source/result contracts, per-image
hashes, rank-local progress, and a completion marker make a Slurm requeue safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist


LOCAL_PAPER_REWARDS = (
    "pickscore",
    "clipscore",
    "hpsv2",
    "aesthetic",
    "imagereward",
)
PICKSCORE_PAPER_SCALE = 26.0


class DrawBenchRescoreError(RuntimeError):
    """Raised when saved inputs, progress, or outputs violate the contract."""


@dataclass(frozen=True)
class ValidatedSource:
    """Canonical source files and hashes used by a rescore request."""

    directory: Path
    image_paths: tuple[Path, ...]
    results_sha256: str
    metadata_sha256: str
    images_sha256: str
    complete_sha256: str


def sha256_file(path: Path) -> str:
    """Return a regular file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DrawBenchRescoreError(f"Cannot read JSON file {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise DrawBenchRescoreError(
                        f"Blank JSONL record at {path}:{line_number}"
                    )
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DrawBenchRescoreError(
                        f"Invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise DrawBenchRescoreError(f"Cannot read JSONL file {path}: {exc}") from exc
    return records


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise DrawBenchRescoreError(f"Missing or empty {label}: {path}")


def load_prompts(
    path: Path, expected_full_samples: int, expected_samples: int
) -> list[str]:
    """Load the canonical prefix used by full or bounded-smoke evaluation."""

    _require_file(path, "DrawBench prompt file")
    try:
        with path.open("r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle.readlines()]
    except OSError as exc:
        raise DrawBenchRescoreError(f"Cannot read DrawBench prompts: {path}") from exc
    if len(prompts) != expected_full_samples:
        raise DrawBenchRescoreError(
            f"Expected {expected_full_samples} DrawBench records; found {len(prompts)}"
        )
    if any(not prompt for prompt in prompts):
        raise DrawBenchRescoreError("DrawBench prompt records must be nonempty")
    if expected_samples > expected_full_samples:
        raise DrawBenchRescoreError(
            "Scored sample count cannot exceed the full DrawBench size"
        )
    return prompts[:expected_samples]


def validate_source(
    source_dir: Path,
    prompts: Sequence[str],
    *,
    expected_base_model: Path,
    expected_checkpoint: Path | None,
    expected_seed: int,
    expected_eval_steps: int,
    expected_guidance_scale: float,
    expected_resolution: int,
    expected_generation_batch_size: int,
    expected_world_size: int,
    expected_full_samples: int,
    expected_checkpoint_lora_subdir: str | None = "lora",
) -> ValidatedSource:
    """Validate a complete PickScore generation and its exact JPEG inventory."""

    source_dir = source_dir.expanduser().resolve()
    images_dir = source_dir / "images"
    results_path = source_dir / "evaluation_results.jsonl"
    metadata_path = source_dir / "evaluation_metadata.json"
    averages_path = source_dir / "average_scores.json"
    request_path = source_dir / ".request.json"
    image_manifest_path = source_dir / "images.sha256"
    complete_path = source_dir / ".complete"
    for path, label in (
        (request_path, "source request"),
        (results_path, "source results"),
        (metadata_path, "source metadata"),
        (averages_path, "source averages"),
        (image_manifest_path, "source image manifest"),
        (complete_path, "source completion marker"),
    ):
        _require_file(path, label)
    if not images_dir.is_dir():
        raise DrawBenchRescoreError(f"Missing source image directory: {images_dir}")

    metadata = _load_json(metadata_path)
    expected_checkpoint_path = (
        os.fspath(expected_checkpoint.expanduser().resolve())
        if expected_checkpoint is not None
        else None
    )
    expected_metadata = {
        "base_model": os.fspath(expected_base_model.expanduser().resolve()),
        "checkpoint_path": expected_checkpoint_path,
        "local_files_only": True,
        "dataset": "drawbench",
        "reward_names": ["pickscore"],
        "seed": expected_seed,
        "batch_size_per_rank": expected_generation_batch_size,
        "num_inference_steps": expected_eval_steps,
        "guidance_scale": float(expected_guidance_scale),
        "resolution": expected_resolution,
        "mixed_precision": "fp16",
        "world_size": expected_world_size,
        "full_dataset_size": expected_full_samples,
        "max_samples": (
            None if len(prompts) == expected_full_samples else len(prompts)
        ),
        "expected_sample_count": len(prompts),
        "sampler": "strided_without_padding",
    }
    if expected_checkpoint is None:
        expected_metadata["lora_hf_path"] = None
    if not isinstance(metadata, dict):
        raise DrawBenchRescoreError(f"Source metadata must be an object: {metadata_path}")
    actual_lora_subdir = (
        metadata.get("checkpoint_lora_subdir")
        if expected_checkpoint is None
        else metadata.get("checkpoint_lora_subdir", "lora")
    )
    if actual_lora_subdir != expected_checkpoint_lora_subdir:
        raise DrawBenchRescoreError(
            "Source metadata mismatch for checkpoint_lora_subdir: "
            f"expected={expected_checkpoint_lora_subdir!r}, "
            f"actual={metadata.get('checkpoint_lora_subdir')!r}"
        )
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise DrawBenchRescoreError(
                f"Source metadata mismatch for {key}: "
                f"expected={expected!r}, actual={metadata.get(key)!r}"
            )

    expected_names = {f"{sample_id:05d}.jpg" for sample_id in range(len(prompts))}
    actual_names = {
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    }
    if actual_names != expected_names:
        raise DrawBenchRescoreError(
            "Source JPEG inventory mismatch; "
            f"missing={sorted(expected_names - actual_names)[:20]}, "
            f"unexpected={sorted(actual_names - expected_names)[:20]}"
        )

    records = _load_jsonl(results_path)
    if len(records) != len(prompts):
        raise DrawBenchRescoreError(
            f"Expected {len(prompts)} source results; found {len(records)}"
        )
    image_paths: list[Path] = []
    for sample_id, (record, prompt) in enumerate(zip(records, prompts, strict=True)):
        if not isinstance(record, dict) or record.get("sample_id") != sample_id:
            raise DrawBenchRescoreError(f"Noncanonical source sample_id at {sample_id}")
        if record.get("sample_seed") != (expected_seed + sample_id) % (2**63 - 1):
            raise DrawBenchRescoreError(f"Invalid source sample seed at {sample_id}")
        if record.get("prompt") != prompt or record.get("metadata") != {}:
            raise DrawBenchRescoreError(f"Source prompt/metadata mismatch at {sample_id}")
        scores = record.get("scores")
        if not isinstance(scores, dict):
            raise DrawBenchRescoreError(f"Missing source scores at {sample_id}")
        for key in ("pickscore", "pickscore_normalized"):
            value = scores.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise DrawBenchRescoreError(
                    f"Invalid source {key} at sample {sample_id}: {value!r}"
                )
        image_path = (images_dir / f"{sample_id:05d}.jpg").resolve()
        _require_file(image_path, f"source image {sample_id}")
        recorded_path = record.get("image_path")
        if not isinstance(recorded_path, str) or Path(recorded_path).resolve() != image_path:
            raise DrawBenchRescoreError(f"Source image path mismatch at {sample_id}")
        image_paths.append(image_path)

    expected_image_manifest = "".join(
        f"{sha256_file(path)}  images/{sample_id:05d}.jpg\n"
        for sample_id, path in enumerate(image_paths)
    )
    try:
        actual_image_manifest = image_manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DrawBenchRescoreError(
            f"Cannot read source image manifest: {image_manifest_path}"
        ) from exc
    if actual_image_manifest != expected_image_manifest:
        raise DrawBenchRescoreError("Source image manifest does not match current JPEGs")
    complete = _load_json(complete_path)
    generation_hashes = {
        "request_sha256": request_path,
        "evaluation_results_sha256": results_path,
        "average_scores_sha256": averages_path,
        "evaluation_metadata_sha256": metadata_path,
        "images_manifest_sha256": image_manifest_path,
    }
    if not isinstance(complete, dict) or complete.get("records") != len(prompts):
        raise DrawBenchRescoreError("Invalid source generation completion marker")
    for field, path in generation_hashes.items():
        if complete.get(field) != sha256_file(path):
            raise DrawBenchRescoreError(
                f"Source generation completion hash mismatch for {field}"
            )

    return ValidatedSource(
        directory=source_dir,
        image_paths=tuple(image_paths),
        results_sha256=sha256_file(results_path),
        metadata_sha256=sha256_file(metadata_path),
        images_sha256=sha256_file(image_manifest_path),
        complete_sha256=sha256_file(complete_path),
    )


def _decode_image(path: Path, expected_resolution: int) -> torch.Tensor:
    try:
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            if rgb.size != (expected_resolution, expected_resolution):
                raise DrawBenchRescoreError(
                    f"Expected {expected_resolution}x{expected_resolution}; "
                    f"found {rgb.size} at {path}"
                )
            pixels = np.array(rgb, dtype=np.uint8, copy=True)
    except (OSError, ValueError) as exc:
        raise DrawBenchRescoreError(f"Cannot decode source image {path}: {exc}") from exc
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)


def _score_values(values: Any, expected_count: int, reward: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected_count,) or not np.isfinite(array).all():
        raise DrawBenchRescoreError(
            f"Scorer {reward} returned invalid shape/values: shape={array.shape}"
        )
    return array


def _record_scores(reward: str, normalized_score: float) -> dict[str, float]:
    if reward == "pickscore":
        return {
            "pickscore": normalized_score * PICKSCORE_PAPER_SCALE,
            "pickscore_normalized": normalized_score,
        }
    return {reward: normalized_score}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _request_payload(args: argparse.Namespace, source: ValidatedSource) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "saved_drawbench_local_reward_rescore",
        "source_directory": os.fspath(source.directory),
        "source_results_sha256": source.results_sha256,
        "source_metadata_sha256": source.metadata_sha256,
        "source_images_sha256": source.images_sha256,
        "source_complete_sha256": source.complete_sha256,
        "dataset_file": os.fspath(args.dataset_file),
        "dataset_sha256": sha256_file(args.dataset_file),
        "reward": args.reward,
        "full_dataset_samples": args.full_dataset_samples,
        "expected_samples": args.expected_samples,
        "expected_seed": args.expected_seed,
        "expected_eval_steps": args.expected_eval_steps,
        "expected_guidance_scale": args.expected_guidance_scale,
        "expected_resolution": args.expected_resolution,
        "expected_generation_batch_size": args.expected_generation_batch_size,
        "expect_no_adapter": getattr(args, "expect_no_adapter", False),
        "expected_checkpoint": (
            os.fspath(args.expected_checkpoint)
            if getattr(args, "expected_checkpoint", None) is not None
            else None
        ),
        "expected_checkpoint_lora_subdir": (
            None
            if getattr(args, "expect_no_adapter", False)
            else getattr(args, "expected_checkpoint_lora_subdir", "lora")
        ),
        "rescore_batch_size_per_rank": args.batch_size,
        "world_size": args.expected_world_size,
        "source_snapshot_sha256": args.source_snapshot_sha256,
    }


def _load_partial(
    path: Path,
    *,
    rank: int,
    world_size: int,
    reward: str,
    prompts: Sequence[str],
    source: ValidatedSource,
    expected_seed: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = _load_json(path)
    if not isinstance(records, list):
        raise DrawBenchRescoreError(f"Partial result must be a list: {path}")
    seen: set[int] = set()
    expected_score_keys = (
        {"pickscore", "pickscore_normalized"} if reward == "pickscore" else {reward}
    )
    for record in records:
        if not isinstance(record, dict):
            raise DrawBenchRescoreError(f"Invalid partial record in {path}")
        sample_id = record.get("sample_id")
        if (
            isinstance(sample_id, bool)
            or not isinstance(sample_id, int)
            or sample_id < 0
            or sample_id >= len(prompts)
            or sample_id % world_size != rank
            or sample_id in seen
        ):
            raise DrawBenchRescoreError(
                f"Invalid rank-{rank} partial sample_id: {sample_id!r}"
            )
        seen.add(sample_id)
        image_path = source.image_paths[sample_id]
        if (
            record.get("sample_seed") != (expected_seed + sample_id) % (2**63 - 1)
            or record.get("prompt") != prompts[sample_id]
            or record.get("metadata") != {}
            or record.get("image_path") != os.fspath(image_path)
            or record.get("image_sha256") != sha256_file(image_path)
        ):
            raise DrawBenchRescoreError(
                f"Rank-{rank} partial source identity mismatch at {sample_id}"
            )
        scores = record.get("scores")
        if not isinstance(scores, dict) or set(scores) != expected_score_keys:
            raise DrawBenchRescoreError(
                f"Rank-{rank} partial score keys mismatch at {sample_id}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in scores.values()
        ):
            raise DrawBenchRescoreError(
                f"Rank-{rank} partial score is nonfinite at {sample_id}"
            )
    return sorted(records, key=lambda item: item["sample_id"])


def _merge_records(
    gathered: Sequence[Sequence[Mapping[str, Any]]], expected_samples: int
) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for rank_records in gathered:
        for raw_record in rank_records:
            record = dict(raw_record)
            sample_id = record.get("sample_id")
            if isinstance(sample_id, bool) or not isinstance(sample_id, int):
                raise DrawBenchRescoreError(f"Invalid scored sample_id: {sample_id!r}")
            if sample_id in by_id:
                raise DrawBenchRescoreError(f"Duplicate scored sample_id: {sample_id}")
            by_id[sample_id] = record
    expected_ids = set(range(expected_samples))
    if set(by_id) != expected_ids:
        raise DrawBenchRescoreError(
            "Distributed rescore coverage mismatch; "
            f"missing={sorted(expected_ids - set(by_id))[:20]}, "
            f"unexpected={sorted(set(by_id) - expected_ids)[:20]}"
        )
    return [by_id[sample_id] for sample_id in range(expected_samples)]


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Mean every finite score field over the exact ordered record set."""

    if not records:
        raise DrawBenchRescoreError("Cannot aggregate an empty result set")
    score_keys = sorted(records[0]["scores"])
    averages: dict[str, float] = {}
    for key in score_keys:
        values = np.asarray([record["scores"][key] for record in records], dtype=np.float64)
        if not np.isfinite(values).all():
            raise DrawBenchRescoreError(f"Nonfinite aggregate input for {key}")
        averages[key] = float(values.mean())
    return averages


def _validate_complete_output(
    output_dir: Path,
    request: Mapping[str, Any],
    prompts: Sequence[str],
    source: ValidatedSource,
) -> dict[str, float] | None:
    complete_path = output_dir / ".complete"
    if not complete_path.exists():
        return None
    paths = {
        "evaluation_results_sha256": output_dir / "evaluation_results.jsonl",
        "average_scores_sha256": output_dir / "average_scores.json",
        "evaluation_metadata_sha256": output_dir / "evaluation_metadata.json",
        "input_images_sha256": output_dir / "input_images.sha256",
    }
    for path in (complete_path, *paths.values()):
        _require_file(path, "complete rescore artifact")
    complete = _load_json(complete_path)
    if not isinstance(complete, dict) or complete.get("records") != len(prompts):
        raise DrawBenchRescoreError(f"Invalid completion marker: {complete_path}")
    for field, path in paths.items():
        if complete.get(field) != sha256_file(path):
            raise DrawBenchRescoreError(f"Completion hash mismatch for {field}")
    metadata = _load_json(paths["evaluation_metadata_sha256"])
    if not isinstance(metadata, dict) or metadata.get("request") != dict(request):
        raise DrawBenchRescoreError("Complete rescore request metadata mismatch")
    records = _load_jsonl(paths["evaluation_results_sha256"])
    if len(records) != len(prompts):
        raise DrawBenchRescoreError("Complete rescore record count mismatch")
    for sample_id, record in enumerate(records):
        image_path = source.image_paths[sample_id]
        if (
            not isinstance(record, dict)
            or record.get("sample_id") != sample_id
            or record.get("prompt") != prompts[sample_id]
            or record.get("image_path") != os.fspath(image_path)
            or record.get("image_sha256") != sha256_file(image_path)
        ):
            raise DrawBenchRescoreError(
                f"Complete rescore source identity mismatch at {sample_id}"
            )
    averages = _load_json(paths["average_scores_sha256"])
    if not isinstance(averages, dict) or not averages:
        raise DrawBenchRescoreError("Complete rescore averages are missing")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in averages.values()
    ):
        raise DrawBenchRescoreError("Complete rescore averages are nonfinite")
    return {str(key): float(value) for key, value in averages.items()}


def _commit_output(
    output_dir: Path,
    request: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    averages = aggregate_records(records)
    results_text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    images_text = "".join(
        f"{record['image_sha256']}  {record['image_path']}\n" for record in records
    )
    metadata = {
        "schema_version": 1,
        "request": dict(request),
        "input_representation": "saved JPEG decoded to RGB float32 tensor in [0, 1]",
        "distributed_sampler": "strided_without_padding",
    }
    _atomic_write(output_dir / "evaluation_results.jsonl", results_text)
    _atomic_write(output_dir / "average_scores.json", _json_text(averages))
    _atomic_write(output_dir / "evaluation_metadata.json", _json_text(metadata))
    _atomic_write(output_dir / "input_images.sha256", images_text)
    complete = {
        "schema_version": 1,
        "records": len(records),
        "evaluation_results_sha256": sha256_file(output_dir / "evaluation_results.jsonl"),
        "average_scores_sha256": sha256_file(output_dir / "average_scores.json"),
        "evaluation_metadata_sha256": sha256_file(output_dir / "evaluation_metadata.json"),
        "input_images_sha256": sha256_file(output_dir / "input_images.sha256"),
    }
    _atomic_write(output_dir / ".complete", _json_text(complete))
    return averages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--reward", choices=LOCAL_PAPER_REWARDS, required=True)
    parser.add_argument("--expected-base-model", type=Path, required=True)
    adapter_source = parser.add_mutually_exclusive_group(required=True)
    adapter_source.add_argument("--expected-checkpoint", type=Path)
    adapter_source.add_argument(
        "--expect-no-adapter",
        action="store_true",
        help="Require source metadata to prove that no checkpoint or Hub LoRA was loaded.",
    )
    parser.add_argument(
        "--expected-checkpoint-lora-subdir",
        default="lora",
        choices=("lora", "lora/v5", "lora/old"),
    )
    parser.add_argument("--full-dataset-samples", type=int, default=1000)
    parser.add_argument("--expected-samples", type=int, default=1000)
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--expected-eval-steps", type=int, default=40)
    parser.add_argument("--expected-guidance-scale", type=float, required=True)
    parser.add_argument("--expected-resolution", type=int, default=512)
    parser.add_argument("--expected-generation-batch-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--expected-world-size", type=int, default=1)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    paths = [
        (args.source_dir, "source directory"),
        (args.output_dir, "output directory"),
        (args.dataset_file, "dataset file"),
        (args.expected_base_model, "base model"),
    ]
    if args.expected_checkpoint is not None:
        paths.append((args.expected_checkpoint, "checkpoint"))
    for path, label in paths:
        if not path.is_absolute():
            raise DrawBenchRescoreError(f"{label} must be absolute: {path}")
    if args.expect_no_adapter != (args.expected_checkpoint is None):
        raise DrawBenchRescoreError(
            "select exactly one checkpoint adapter or explicit no-adapter mode"
        )
    positive = (
        args.full_dataset_samples,
        args.expected_samples,
        args.expected_eval_steps,
        args.expected_resolution,
        args.expected_generation_batch_size,
        args.batch_size,
        args.expected_world_size,
    )
    if any(value <= 0 for value in positive) or args.expected_seed < 0:
        raise DrawBenchRescoreError("Counts must be positive and seed must be nonnegative")
    if args.expected_samples > args.full_dataset_samples:
        raise DrawBenchRescoreError(
            "expected_samples may not exceed full_dataset_samples"
        )
    if args.expected_world_size not in {1, 2}:
        raise DrawBenchRescoreError("Supported rescore world sizes are 1 and 2")
    if args.expected_checkpoint is not None and getattr(
        args, "expected_checkpoint_lora_subdir", "lora"
    ) not in {
        "lora",
        "lora/v5",
        "lora/old",
    }:
        raise DrawBenchRescoreError("Unsupported checkpoint LoRA subdirectory")
    if not math.isfinite(args.expected_guidance_scale) or args.expected_guidance_scale <= 0:
        raise DrawBenchRescoreError("Guidance scale must be finite and positive")
    digest = args.source_snapshot_sha256
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DrawBenchRescoreError("source snapshot SHA-256 must be lowercase hexadecimal")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.dataset_file = args.dataset_file.resolve()
    args.expected_base_model = args.expected_base_model.resolve()
    if args.expected_checkpoint is not None:
        args.expected_checkpoint = args.expected_checkpoint.resolve()
    prompts = load_prompts(
        args.dataset_file, args.full_dataset_samples, args.expected_samples
    )
    source = validate_source(
        args.source_dir,
        prompts,
        expected_base_model=args.expected_base_model,
        expected_checkpoint=args.expected_checkpoint,
        expected_seed=args.expected_seed,
        expected_eval_steps=args.expected_eval_steps,
        expected_guidance_scale=args.expected_guidance_scale,
        expected_resolution=args.expected_resolution,
        expected_generation_batch_size=args.expected_generation_batch_size,
        expected_world_size=args.expected_world_size,
        expected_full_samples=args.full_dataset_samples,
        expected_checkpoint_lora_subdir=(
            None
            if args.expect_no_adapter
            else args.expected_checkpoint_lora_subdir
        ),
    )
    request = _request_payload(args, source)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "source_directory": os.fspath(source.directory),
                    "source_results_sha256": source.results_sha256,
                    "reward": args.reward,
                    "images": len(source.image_paths),
                },
                sort_keys=True,
            )
        )
        return 0

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != args.expected_world_size:
        raise DrawBenchRescoreError(
            f"Expected torchrun world_size={args.expected_world_size}; got {world_size}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise DrawBenchRescoreError(
            f"Expected exactly {world_size} CUDA devices; found {torch.cuda.device_count()}"
        )
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            request_path = args.output_dir / ".request.json"
            if request_path.exists():
                if _load_json(request_path) != request:
                    raise DrawBenchRescoreError(
                        f"Existing output belongs to another request: {args.output_dir}"
                    )
            else:
                unexpected = [
                    path for path in args.output_dir.iterdir() if path.name != ".request.json"
                ]
                if unexpected:
                    raise DrawBenchRescoreError(
                        f"Nonempty output has no request marker: {args.output_dir}"
                    )
                _atomic_write(request_path, _json_text(request))
            previous = _validate_complete_output(
                args.output_dir, request, prompts, source
            )
            reuse = previous is not None
            if reuse:
                print(
                    f"Reusing complete {args.reward} rescore: {args.output_dir} {previous}",
                    flush=True,
                )
        else:
            reuse = False
        reuse_payload = [reuse]
        dist.broadcast_object_list(reuse_payload, src=0)
        if reuse_payload[0]:
            dist.barrier()
            return 0

        partial_path = args.output_dir / f".partial-results-rank-{rank:05d}.json"
        local_records = _load_partial(
            partial_path,
            rank=rank,
            world_size=world_size,
            reward=args.reward,
            prompts=prompts,
            source=source,
            expected_seed=args.expected_seed,
        )
        completed_ids = {record["sample_id"] for record in local_records}

        from flow_grpo.rewards import multi_score

        scoring_fn = multi_score(device, {args.reward: 1.0})
        local_ids = [
            sample_id
            for sample_id in range(rank, args.expected_samples, world_size)
            if sample_id not in completed_ids
        ]
        for offset in range(0, len(local_ids), args.batch_size):
            batch_ids = local_ids[offset : offset + args.batch_size]
            images = torch.stack(
                [_decode_image(source.image_paths[sample_id], args.expected_resolution) for sample_id in batch_ids]
            ).to(device)
            batch_prompts = [prompts[sample_id] for sample_id in batch_ids]
            scores, score_metadata = scoring_fn(
                images, batch_prompts, [{} for _ in batch_ids], only_strict=False
            )
            if score_metadata not in ({}, None):
                raise DrawBenchRescoreError(
                    f"Unexpected scorer metadata: {score_metadata!r}"
                )
            values = _score_values(scores[args.reward], len(batch_ids), args.reward)
            for position, sample_id in enumerate(batch_ids):
                image_path = source.image_paths[sample_id]
                local_records.append(
                    {
                        "sample_id": sample_id,
                        "sample_seed": (args.expected_seed + sample_id) % (2**63 - 1),
                        "prompt": prompts[sample_id],
                        "metadata": {},
                        "scores": _record_scores(args.reward, float(values[position])),
                        "image_path": os.fspath(image_path),
                        "image_sha256": sha256_file(image_path),
                    }
                )
            local_records.sort(key=lambda item: item["sample_id"])
            _atomic_write(partial_path, _json_text(local_records))
            del images, scores
            if rank == 0 and ((offset // args.batch_size) + 1) % 10 == 0:
                print(
                    f"{args.reward}: rank 0 has {len(local_records)}/"
                    f"{len(range(rank, args.expected_samples, world_size))} records",
                    flush=True,
                )

        gathered: list[list[dict[str, Any]]] | None = (
            [list() for _ in range(world_size)] if rank == 0 else None
        )
        dist.gather_object(local_records, gathered, dst=0)
        if rank == 0:
            assert gathered is not None
            ordered = _merge_records(gathered, args.expected_samples)
            averages = _commit_output(args.output_dir, request, ordered)
            for partial_rank in range(world_size):
                try:
                    (args.output_dir / f".partial-results-rank-{partial_rank:05d}.json").unlink()
                except FileNotFoundError:
                    pass
            print(
                f"Committed {args.reward} rescore: {args.output_dir} {averages}",
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DrawBenchRescoreError as exc:
        print(f"drawbench rescore contract error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
