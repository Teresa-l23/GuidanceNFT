#!/usr/bin/env python3
"""Run and validate the resumable 13-policy native OCR/GenEval matrix."""

from __future__ import annotations

import os

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from baselines.diffusion_nft_sd35_v5.tools.run_drawbench_paper_rewards import (
    Coordinate,
    PaperRewardsError,
    _atomic_write,
    _atomic_write_text,
    _canonical_json,
    _load_json,
    _load_jsonl,
    _mapping,
    _run,
    build_direct_coordinates,
    sha256_file,
    validate_direct_checkpoint_matrix,
)


EXPERIMENT = "sd35_native_ocr_geneval_13policy"
SOURCE_TYPES = ("checkpoint", "base_model")
EXPECTED_POLICY_ROWS = (
    ("geneval", "standard", "control-geneval_steps10_u200_53806010", "checkpoint", 200),
    ("geneval", "dual", "control-geneval-cfg1dual_steps10_u200_54427836", "checkpoint", 200),
    ("geneval", "fm100", "geneval_steps10_u200_53654216", "checkpoint", 200),
    ("pickscore", "standard", "control-pickscore_steps10_u100_54871951", "checkpoint", 100),
    ("pickscore", "dual", "control-pickscore-cfg1dual_steps10_u100_54871880", "checkpoint", 100),
    ("pickscore", "fm100", "pickscore_steps10_u100_54872044", "checkpoint", 100),
    ("ocr", "standard", "control-ocr_steps10_u100_54997942", "checkpoint", 50),
    ("ocr", "dual", "control-ocr-cfg1dual_steps10_u100_54997899", "checkpoint", 50),
    ("ocr", "fm100", "ocr_steps10_u100_54997985", "checkpoint", 50),
    ("learned3", "standard", "control-learned3_steps10_u100_54555957", "checkpoint", 100),
    ("learned3", "dual", "control-learned3-cfg1dual_steps10_u100_54555955", "checkpoint", 100),
    ("learned3", "fm100", "learned3_steps10_u100_54555956", "checkpoint", 100),
    ("sd35_base", "base", "sd35-medium-b940f670-base", "base_model", None),
)
EXPECTED_SOURCE_ROWS = (
    (
        "baselines/diffusion_nft_sd35_v5/config/"
        "pickscore100_geneval200_ocr50_drawbench_offline_rewards.json",
        ("geneval", "pickscore", "ocr"),
    ),
    (
        "baselines/diffusion_nft_sd35_v5/config/"
        "learned3_step100_drawbench_offline_rewards.json",
        ("learned3",),
    ),
    (
        "baselines/diffusion_nft_sd35_v5/config/"
        "sd35_base_drawbench_offline_rewards.json",
        ("sd35_base",),
    ),
)
EXPECTED_EXCLUSIONS = (
    ("learned3_v5dual", "pretrained_v5", "learned3-v5dual_steps10_u100_54995803", 0, "lora/v5"),
    ("learned3_v5dual", "cfg4_v5dual", "learned3-v5dual_steps10_u100_54995803", 100, "lora"),
)
EXPECTED_BASE_MODEL = {
    "repo_id": "stabilityai/stable-diffusion-3.5-medium",
    "revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
    "model_index_sha256": "b5e97ca0d3c0cd9a1e3ed91a71bbdd489de7941b23fe038b3aaa9acf36fd1543",
}
EXPECTED_BENCHMARK_ROWS = (
    (
        "ocr",
        "ocr",
        "baselines/diffusion_nft/upstream/dataset/ocr/test.txt",
        1018,
        16,
    ),
    (
        "geneval",
        "geneval",
        "baselines/diffusion_nft/upstream/dataset/geneval/test_metadata.jsonl",
        2212,
        14,
    ),
)
GENEVAL_GROUPS = (
    "single_object",
    "two_object",
    "counting",
    "colors",
    "position",
    "color_attr",
)
GENEVAL_SCORE_NAMES = (
    "accuracy",
    "strict_accuracy",
    *(f"{group}_strict_accuracy" for group in GENEVAL_GROUPS),
    *(f"{group}_accuracy" for group in GENEVAL_GROUPS),
    "geneval",
    "avg",
)
GENEVAL_DENSE_SCORE_NAMES = frozenset(("geneval", "avg"))
# The immutable GenEval scorer keeps at most 16 detections per class. Its dense
# count term is 1 - abs(target - detected) / target, so a target count of one
# establishes the closed lower bound 1 - abs(1 - 16) == -14. Unlike Boolean
# accuracy diagnostics, this shaped training reward is intentionally negative
# after severe over-counting.
GENEVAL_DENSE_SCORE_MIN = -14.0
OCR_SCORE_NAMES = ("ocr", "avg")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PARTIAL_RE = re.compile(r"^\.partial-results-rank-([0-9]{5})\.json$")


class NativeMatrixError(PaperRewardsError):
    """Raised when the native evaluation contract or output is inconsistent."""


@dataclass(frozen=True)
class Benchmark:
    dataset: str
    reward: str
    dataset_path: Path
    dataset_sha256: str
    expected_samples: int
    batch_size_per_rank: int
    prompts: tuple[str, ...]
    metadatas: tuple[Mapping[str, Any], ...]

    @property
    def score_names(self) -> tuple[str, ...]:
        return OCR_SCORE_NAMES if self.dataset == "ocr" else GENEVAL_SCORE_NAMES


@dataclass(frozen=True)
class NativeCoordinate:
    policy: Coordinate
    benchmark: Benchmark

    @property
    def relative_output(self) -> Path:
        return Path(
            "formal",
            self.benchmark.dataset,
            self.policy.family,
            self.policy.method,
            self.policy.guidance_tag,
        )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _require_regular_within(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise NativeMatrixError(f"cannot resolve {label}: {path}: {error}") from error
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise NativeMatrixError(f"{label} is not a regular file inside {root}: {path}")
    return resolved


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeMatrixError(f"{label} must be numeric, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise NativeMatrixError(f"{label} must be finite, got {value!r}")
    return numeric


def validate_native_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the fixed 13-policy, two-benchmark, two-CFG protocol."""

    if manifest.get("schema_version") != 1 or manifest.get("experiment") != EXPERIMENT:
        raise NativeMatrixError("native matrix manifest identity mismatch")
    if manifest.get("base_model") != EXPECTED_BASE_MODEL:
        raise NativeMatrixError("native matrix base-model contract mismatch")
    checkpoint_root = manifest.get("checkpoint_root")
    if isinstance(checkpoint_root, str):
        checkpoint_root = os.path.expandvars(checkpoint_root)
    if not isinstance(checkpoint_root, str) or not Path(checkpoint_root).is_absolute():
        raise NativeMatrixError("native matrix checkpoint_root must be absolute")

    source_values = manifest.get("source_manifests")
    if not isinstance(source_values, list):
        raise NativeMatrixError("source_manifests must be a list")
    actual_sources: list[tuple[str, tuple[str, ...]]] = []
    for index, value in enumerate(source_values):
        source = _mapping(value, f"source_manifests[{index}]")
        if not _is_sha256(source.get("sha256")):
            raise NativeMatrixError(f"source_manifests[{index}].sha256 is invalid")
        families = source.get("families")
        if not isinstance(families, list) or not all(
            isinstance(item, str) for item in families
        ):
            raise NativeMatrixError(f"source_manifests[{index}].families is invalid")
        actual_sources.append((str(source.get("path")), tuple(families)))
    if tuple(actual_sources) != EXPECTED_SOURCE_ROWS:
        raise NativeMatrixError("source manifest order or family coverage mismatch")

    policy_values = manifest.get("policies")
    if not isinstance(policy_values, list):
        raise NativeMatrixError("policies must be a list")
    actual_policy_rows: list[tuple[str, str, str, str, int | None]] = []
    for index, value in enumerate(policy_values):
        policy = _mapping(value, f"policies[{index}]")
        row = (
            str(policy.get("family")),
            str(policy.get("method")),
            str(policy.get("run_id")),
            str(policy.get("source_type")),
            policy.get("checkpoint_step"),
        )
        actual_policy_rows.append(row)
        if row[3] not in SOURCE_TYPES:
            raise NativeMatrixError(f"policies[{index}].source_type is invalid")
        if not _is_sha256(policy.get("identity_sha256")):
            raise NativeMatrixError(f"policies[{index}].identity_sha256 is invalid")
        adapter_sha = policy.get("adapter_model_sha256")
        if row[3] == "base_model":
            if adapter_sha is not None or row[4] is not None:
                raise NativeMatrixError("base policy cannot declare an adapter or checkpoint")
        elif not _is_sha256(adapter_sha):
            raise NativeMatrixError(f"policies[{index}] must pin its adapter hash")
    if tuple(actual_policy_rows) != EXPECTED_POLICY_ROWS:
        raise NativeMatrixError("policy order or identity fields mismatch")

    exclusions = manifest.get("excluded_policies")
    if not isinstance(exclusions, list):
        raise NativeMatrixError("excluded_policies must be a list")
    actual_exclusions = tuple(
        (
            str(_mapping(item, "excluded policy").get("family")),
            str(item.get("method")),
            str(item.get("run_id")),
            item.get("checkpoint_step"),
            str(item.get("checkpoint_lora_subdir")),
        )
        for item in exclusions
    )
    if actual_exclusions != EXPECTED_EXCLUSIONS:
        raise NativeMatrixError("the two Learned3 V5 exclusions are not exact")

    evaluation = _mapping(manifest.get("evaluation"), "evaluation")
    expected_eval = {
        "seed": 42,
        "num_inference_steps": 40,
        "guidance_scales": [1.0, 4.0],
        "resolution": 512,
        "mixed_precision": "fp16",
        "world_size": 2,
        "sampler": "diffusers",
        "deterministic": True,
        "save_images": False,
    }
    for key, expected in expected_eval.items():
        if evaluation.get(key) != expected:
            raise NativeMatrixError(f"evaluation.{key} mismatch")
    benchmark_values = evaluation.get("benchmarks")
    if not isinstance(benchmark_values, list):
        raise NativeMatrixError("evaluation.benchmarks must be a list")
    actual_benchmarks: list[tuple[str, str, str, int, int]] = []
    for index, value in enumerate(benchmark_values):
        benchmark = _mapping(value, f"benchmarks[{index}]")
        if not _is_sha256(benchmark.get("dataset_sha256")):
            raise NativeMatrixError(f"benchmarks[{index}].dataset_sha256 is invalid")
        actual_benchmarks.append(
            (
                str(benchmark.get("dataset")),
                str(benchmark.get("reward")),
                str(benchmark.get("dataset_file")),
                int(benchmark.get("expected_samples", -1)),
                int(benchmark.get("batch_size_per_rank", -1)),
            )
        )
    if tuple(actual_benchmarks) != EXPECTED_BENCHMARK_ROWS:
        raise NativeMatrixError("native benchmark order or settings mismatch")


def _adapter_sha256(inventory: Mapping[str, Any]) -> str | None:
    matches = [
        item.get("sha256")
        for item in inventory.get("files", [])
        if isinstance(item, Mapping)
        and str(item.get("path", "")).endswith("adapter_model.safetensors")
    ]
    if inventory.get("source_type") == "base_model":
        if matches:
            raise NativeMatrixError("base-model inventory unexpectedly contains an adapter")
        return None
    if len(matches) != 1 or not _is_sha256(matches[0]):
        raise NativeMatrixError("checkpoint inventory does not identify one adapter model")
    return str(matches[0])


def load_policies(
    manifest: Mapping[str, Any],
    *,
    project_root: Path,
    checkpoint_root: Path,
    base_model: Path,
) -> tuple[list[Coordinate], dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    """Validate source manifests/checkpoints and return policies in requested order."""

    try:
        resolved_project = project_root.resolve(strict=True)
        resolved_checkpoint_root = checkpoint_root.resolve(strict=True)
        resolved_base_model = base_model.resolve(strict=True)
        declared_checkpoint_root = Path(os.path.expandvars(str(manifest["checkpoint_root"]))).resolve(strict=True)
    except OSError as error:
        raise NativeMatrixError(f"cannot resolve matrix input: {error}") from error
    if resolved_checkpoint_root != declared_checkpoint_root:
        raise NativeMatrixError("checkpoint root differs from the matrix manifest")

    inventories: dict[tuple[str, str], dict[str, Any]] = {}
    coordinates_by_key: dict[tuple[str, str, float], Coordinate] = {}
    source_summaries: list[dict[str, Any]] = []
    for index, source_value in enumerate(manifest["source_manifests"]):
        source = _mapping(source_value, f"source_manifests[{index}]")
        relative = Path(str(source["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise NativeMatrixError(f"unsafe source manifest path: {relative}")
        source_path = _require_regular_within(
            resolved_project / relative, resolved_project, "source manifest"
        )
        actual_sha = sha256_file(source_path)
        if actual_sha != source["sha256"]:
            raise NativeMatrixError(f"source manifest hash mismatch: {relative}")
        source_manifest = _mapping(_load_json(source_path, "source manifest"), "source manifest")
        family_values = source_manifest.get("families")
        if not isinstance(family_values, list):
            raise NativeMatrixError(f"source manifest has no family list: {relative}")
        family_keys = tuple(str(_mapping(item, "source family").get("key")) for item in family_values)
        if family_keys != tuple(source["families"]):
            raise NativeMatrixError(f"source family mismatch: {relative}")
        source_inventory = validate_direct_checkpoint_matrix(
            source_manifest,
            resolved_checkpoint_root,
            base_model=resolved_base_model,
        )
        source_coordinates = build_direct_coordinates(
            source_manifest, resolved_checkpoint_root, source_inventory
        )
        overlap = set(inventories).intersection(source_inventory)
        if overlap:
            raise NativeMatrixError(f"duplicate policies across source manifests: {sorted(overlap)}")
        inventories.update({key: dict(value) for key, value in source_inventory.items()})
        for coordinate in source_coordinates:
            key = (coordinate.family, coordinate.method, coordinate.guidance_scale)
            if key in coordinates_by_key:
                raise NativeMatrixError(f"duplicate source coordinate: {key}")
            coordinates_by_key[key] = coordinate
        source_summaries.append(
            {
                "path": relative.as_posix(),
                "sha256": actual_sha,
                "experiment": source_manifest.get("experiment"),
                "families": list(family_keys),
            }
        )

    expected_keys = {(row[0], row[1]) for row in EXPECTED_POLICY_ROWS}
    if set(inventories) != expected_keys:
        raise NativeMatrixError(
            "source manifests do not resolve exactly the requested 13 policies"
        )

    ordered_coordinates: list[Coordinate] = []
    for policy_value in manifest["policies"]:
        policy = _mapping(policy_value, "policy")
        key = (str(policy["family"]), str(policy["method"]))
        inventory = inventories[key]
        comparisons = {
            "run_id": policy["run_id"],
            "source_type": policy["source_type"],
            "checkpoint_step": policy["checkpoint_step"],
            "identity_sha256": policy["identity_sha256"],
        }
        for field, expected in comparisons.items():
            if inventory.get(field) != expected:
                raise NativeMatrixError(f"policy inventory {key} has mismatched {field}")
        if _adapter_sha256(inventory) != policy["adapter_model_sha256"]:
            raise NativeMatrixError(f"policy inventory {key} has a mismatched adapter hash")
        for guidance in manifest["evaluation"]["guidance_scales"]:
            coordinate_key = (*key, float(guidance))
            try:
                coordinate = coordinates_by_key[coordinate_key]
            except KeyError as error:
                raise NativeMatrixError(f"missing source coordinate: {coordinate_key}") from error
            ordered_coordinates.append(coordinate)

    if len(ordered_coordinates) != 26:
        raise NativeMatrixError("expected 13 policies x 2 CFG values")
    return ordered_coordinates, inventories, source_summaries


def _load_dataset(
    project_root: Path, benchmark: Mapping[str, Any]
) -> Benchmark:
    relative = Path(str(benchmark["dataset_file"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise NativeMatrixError(f"unsafe dataset path: {relative}")
    dataset_path = _require_regular_within(
        project_root / relative, project_root, "evaluation dataset"
    )
    actual_sha = sha256_file(dataset_path)
    if actual_sha != benchmark["dataset_sha256"]:
        raise NativeMatrixError(f"dataset hash mismatch: {relative}")
    dataset = str(benchmark["dataset"])
    if dataset == "ocr":
        prompts = tuple(line.strip() for line in dataset_path.read_text(encoding="utf-8").splitlines())
        metadatas: tuple[Mapping[str, Any], ...] = tuple({} for _ in prompts)
    elif dataset == "geneval":
        values = _load_jsonl(dataset_path, "GenEval dataset")
        metadata_list: list[Mapping[str, Any]] = []
        prompt_list: list[str] = []
        for index, value in enumerate(values):
            metadata = _mapping(value, f"GenEval metadata[{index}]")
            prompt = metadata.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise NativeMatrixError(f"GenEval sample {index} has no prompt")
            metadata_list.append(dict(metadata))
            prompt_list.append(prompt)
        prompts = tuple(prompt_list)
        metadatas = tuple(metadata_list)
    else:
        raise NativeMatrixError(f"unsupported native benchmark: {dataset}")
    if len(prompts) != benchmark["expected_samples"] or any(not prompt for prompt in prompts):
        raise NativeMatrixError(
            f"{dataset} dataset coverage mismatch: expected={benchmark['expected_samples']} actual={len(prompts)}"
        )
    return Benchmark(
        dataset=dataset,
        reward=str(benchmark["reward"]),
        dataset_path=dataset_path,
        dataset_sha256=actual_sha,
        expected_samples=int(benchmark["expected_samples"]),
        batch_size_per_rank=int(benchmark["batch_size_per_rank"]),
        prompts=prompts,
        metadatas=metadatas,
    )


def load_benchmarks(manifest: Mapping[str, Any], project_root: Path) -> list[Benchmark]:
    resolved_project = project_root.resolve(strict=True)
    return [
        _load_dataset(resolved_project, _mapping(value, "benchmark"))
        for value in manifest["evaluation"]["benchmarks"]
    ]


def build_matrix(
    policy_coordinates: Sequence[Coordinate], benchmarks: Sequence[Benchmark]
) -> list[NativeCoordinate]:
    coordinates = [
        NativeCoordinate(policy=policy, benchmark=benchmark)
        for benchmark in benchmarks
        for policy in policy_coordinates
    ]
    identities = {
        (
            item.benchmark.dataset,
            item.policy.family,
            item.policy.method,
            item.policy.guidance_scale,
        )
        for item in coordinates
    }
    if len(coordinates) != 52 or len(identities) != 52:
        raise NativeMatrixError("expected exactly 52 unique native evaluation coordinates")
    return coordinates


def build_evaluation_command(
    coordinate: NativeCoordinate,
    *,
    evaluator: Path,
    base_model: Path,
    output_dir: Path,
    evaluation: Mapping[str, Any],
) -> list[str]:
    """Build one torchrun command; base policies deliberately omit adapter args."""

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={evaluation['world_size']}",
        str(evaluator),
        "--base_model",
        str(base_model),
        "--local_files_only",
        "--seed",
        str(evaluation["seed"]),
        "--model_type",
        "sd3",
        "--dataset",
        coordinate.benchmark.dataset,
        "--output_dir",
        str(output_dir),
        "--num_inference_steps",
        str(evaluation["num_inference_steps"]),
        "--guidance_scale",
        str(coordinate.policy.guidance_scale),
        "--resolution",
        str(evaluation["resolution"]),
        "--mixed_precision",
        str(evaluation["mixed_precision"]),
        "--batch_size",
        str(coordinate.benchmark.batch_size_per_rank),
        "--reward",
        coordinate.benchmark.reward,
    ]
    if coordinate.policy.source_type == "checkpoint":
        if coordinate.policy.checkpoint is None or coordinate.policy.checkpoint_lora_subdir is None:
            raise NativeMatrixError("checkpoint policy has no checkpoint adapter path")
        command.extend(
            [
                "--checkpoint_path",
                str(coordinate.policy.checkpoint),
                "--checkpoint_lora_subdir",
                coordinate.policy.checkpoint_lora_subdir,
            ]
        )
    elif coordinate.policy.source_type != "base_model":
        raise NativeMatrixError(f"unsupported source type: {coordinate.policy.source_type}")
    if evaluation["save_images"]:
        command.append("--save_images")
    return command


def _coordinate_request(
    coordinate: NativeCoordinate,
    *,
    manifest_sha256: str,
    evaluator_sha256: str,
    source_id: str,
    run_name: str,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    policy = coordinate.policy
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "manifest_sha256": manifest_sha256,
        "evaluator_sha256": evaluator_sha256,
        "source_id": source_id,
        "run_name": run_name,
        "family": policy.family,
        "method": policy.method,
        "run_id": policy.run_id,
        "source_type": policy.source_type,
        "checkpoint": str(policy.checkpoint) if policy.checkpoint is not None else None,
        "checkpoint_lora_subdir": policy.checkpoint_lora_subdir,
        "checkpoint_step": policy.checkpoint_step,
        "checkpoint_identity": policy.checkpoint_identity,
        "dataset": coordinate.benchmark.dataset,
        "reward": coordinate.benchmark.reward,
        "dataset_sha256": coordinate.benchmark.dataset_sha256,
        "expected_samples": coordinate.benchmark.expected_samples,
        "batch_size_per_rank": coordinate.benchmark.batch_size_per_rank,
        "seed": evaluation["seed"],
        "num_inference_steps": evaluation["num_inference_steps"],
        "guidance_scale": policy.guidance_scale,
        "resolution": evaluation["resolution"],
        "mixed_precision": evaluation["mixed_precision"],
        "world_size": evaluation["world_size"],
        "sampler": evaluation["sampler"],
        "deterministic": evaluation["deterministic"],
        "save_images": evaluation["save_images"],
    }


def _validate_score_mapping(
    record: Mapping[str, Any], benchmark: Benchmark, label: str
) -> dict[str, float]:
    scores = _mapping(record.get("scores"), f"{label} scores")
    if set(scores) != set(benchmark.score_names):
        raise NativeMatrixError(
            f"{label} score names mismatch: expected={benchmark.score_names} actual={sorted(scores)}"
        )
    validated: dict[str, float] = {}
    for name in benchmark.score_names:
        value = _require_number(scores.get(name), f"{label} {name}")
        if benchmark.dataset == "geneval" and name in GENEVAL_DENSE_SCORE_NAMES:
            if not GENEVAL_DENSE_SCORE_MIN <= value <= 1.0:
                raise NativeMatrixError(
                    f"{label} {name} is outside [{GENEVAL_DENSE_SCORE_MIN:g},1]"
                )
        elif value != -10.0 and not 0.0 <= value <= 1.0:
            raise NativeMatrixError(f"{label} {name} is outside [-10 sentinel] or [0,1]")
        validated[name] = value
    if benchmark.dataset == "geneval" and not math.isclose(
        validated["geneval"], validated["avg"], rel_tol=0.0, abs_tol=1e-12
    ):
        raise NativeMatrixError(f"{label} geneval and avg scores disagree")
    return validated


def _validate_record(
    value: Any,
    *,
    sample_id: int,
    benchmark: Benchmark,
    seed: int,
    label: str,
) -> dict[str, float]:
    record = _mapping(value, label)
    if record.get("sample_id") != sample_id or isinstance(record.get("sample_id"), bool):
        raise NativeMatrixError(f"{label} has a mismatched sample_id")
    expected_seed = (seed + sample_id) % (2**63 - 1)
    if record.get("sample_seed") != expected_seed or isinstance(record.get("sample_seed"), bool):
        raise NativeMatrixError(f"{label} has a mismatched sample_seed")
    if record.get("prompt") != benchmark.prompts[sample_id]:
        raise NativeMatrixError(f"{label} has a mismatched prompt")
    if record.get("metadata") != benchmark.metadatas[sample_id]:
        raise NativeMatrixError(f"{label} has mismatched dataset metadata")
    if "image_path" in record:
        raise NativeMatrixError(f"{label} unexpectedly records an image path")
    return _validate_score_mapping(record, benchmark, label)


def validate_partial_output(
    output_dir: Path,
    *,
    benchmark: Benchmark,
    seed: int,
    world_size: int,
) -> dict[str, Any]:
    """Validate every record the upstream evaluator will skip after requeue."""

    completed_ids: set[int] = set()
    records_by_rank: dict[int, int] = {}
    for partial_path in sorted(output_dir.glob(".partial-results-rank-*.json")):
        match = PARTIAL_RE.fullmatch(partial_path.name)
        if match is None:
            raise NativeMatrixError(f"invalid partial filename: {partial_path}")
        rank = int(match.group(1))
        if rank >= world_size or rank in records_by_rank:
            raise NativeMatrixError(f"partial rank is invalid or duplicated: {rank}")
        values = _load_json(partial_path, "rank-local partial output")
        if not isinstance(values, list):
            raise NativeMatrixError(f"partial output must be a list: {partial_path}")
        previous_id = -1
        for record_index, value in enumerate(values):
            record = _mapping(value, f"partial rank {rank} record {record_index}")
            sample_id = record.get("sample_id")
            if (
                type(sample_id) is not int
                or sample_id < 0
                or sample_id >= benchmark.expected_samples
                or sample_id % world_size != rank
                or sample_id in completed_ids
                or sample_id <= previous_id
            ):
                raise NativeMatrixError(
                    f"invalid, duplicate, or unordered partial sample_id={sample_id!r}"
                )
            _validate_record(
                value,
                sample_id=sample_id,
                benchmark=benchmark,
                seed=seed,
                label=f"partial sample {sample_id}",
            )
            completed_ids.add(sample_id)
            previous_id = sample_id
        records_by_rank[rank] = len(values)
    if (output_dir / "images").exists():
        raise NativeMatrixError("images directory exists although save_images=false")
    return {
        "partial_records": len(completed_ids),
        "remaining_records": benchmark.expected_samples - len(completed_ids),
        "records_by_rank": records_by_rank,
        "completed_ids": completed_ids,
    }


def _validate_metadata(
    metadata: Mapping[str, Any],
    *,
    coordinate: NativeCoordinate,
    base_model: Path,
    evaluation: Mapping[str, Any],
) -> None:
    benchmark = coordinate.benchmark
    expected = {
        "revision": None,
        "local_files_only": True,
        "lora_hf_path": None,
        "seed": evaluation["seed"],
        "dataset": benchmark.dataset,
        "reward_names": [benchmark.reward],
        "batch_size_per_rank": benchmark.batch_size_per_rank,
        "num_inference_steps": evaluation["num_inference_steps"],
        "guidance_scale": coordinate.policy.guidance_scale,
        "resolution": evaluation["resolution"],
        "mixed_precision": evaluation["mixed_precision"],
        "world_size": evaluation["world_size"],
        "full_dataset_size": benchmark.expected_samples,
        "max_samples": None,
        "expected_sample_count": benchmark.expected_samples,
        "sampler": "strided_without_padding",
        "duplicate_records_removed": 0,
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if isinstance(expected_value, float):
            if not math.isclose(
                _require_number(actual, f"metadata.{key}"),
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise NativeMatrixError(f"metadata.{key} mismatch")
        elif actual != expected_value:
            raise NativeMatrixError(
                f"metadata.{key} mismatch: expected={expected_value!r} actual={actual!r}"
            )
    base_value = metadata.get("base_model")
    if not isinstance(base_value, str) or Path(base_value).resolve(strict=True) != base_model.resolve(strict=True):
        raise NativeMatrixError("evaluation metadata names the wrong base model")
    checkpoint_value = metadata.get("checkpoint_path")
    if coordinate.policy.source_type == "base_model":
        if checkpoint_value is not None or metadata.get("checkpoint_lora_subdir") is not None:
            raise NativeMatrixError("base-model evaluation unexpectedly names an adapter")
    else:
        if (
            not isinstance(checkpoint_value, str)
            or coordinate.policy.checkpoint is None
            or Path(checkpoint_value).resolve(strict=True)
            != coordinate.policy.checkpoint.resolve(strict=True)
        ):
            raise NativeMatrixError("evaluation metadata names the wrong checkpoint")
        if metadata.get("checkpoint_lora_subdir") != coordinate.policy.checkpoint_lora_subdir:
            raise NativeMatrixError("evaluation metadata names the wrong LoRA subdirectory")


def validate_complete_output(
    output_dir: Path,
    *,
    coordinate: NativeCoordinate,
    base_model: Path,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate full sample identity, native scores, aggregates, and metadata."""

    benchmark = coordinate.benchmark
    records = _load_jsonl(output_dir / "evaluation_results.jsonl", "evaluation result")
    if len(records) != benchmark.expected_samples:
        raise NativeMatrixError(
            f"result coverage mismatch: expected={benchmark.expected_samples} actual={len(records)}"
        )
    values_by_name = {name: [] for name in benchmark.score_names}
    for sample_id, value in enumerate(records):
        scores = _validate_record(
            value,
            sample_id=sample_id,
            benchmark=benchmark,
            seed=int(evaluation["seed"]),
            label=f"result sample {sample_id}",
        )
        for name, score in scores.items():
            values_by_name[name].append(score)

    averages = _mapping(
        _load_json(output_dir / "average_scores.json", "average scores"),
        "average scores",
    )
    if set(averages) != set(benchmark.score_names):
        raise NativeMatrixError(
            f"average score names mismatch: expected={benchmark.score_names} actual={sorted(averages)}"
        )
    validated_averages: dict[str, float] = {}
    for name, values in values_by_name.items():
        # Mirror the immutable evaluator's output protocol exactly: it filters
        # the overloaded -10 value for every score name before averaging. Dense
        # GenEval can also produce a genuine -10 after severe over-counting, but
        # changing that ambiguity here would make validation disagree with the
        # already-pinned evaluator and its average_scores.json artifacts.
        applicable = [value for value in values if value != -10.0]
        if not applicable:
            raise NativeMatrixError(f"no applicable samples for aggregate {name}")
        expected_average = math.fsum(applicable) / len(applicable)
        actual_average = _require_number(averages.get(name), f"average {name}")
        if not math.isclose(actual_average, expected_average, rel_tol=1e-9, abs_tol=1e-9):
            raise NativeMatrixError(
                f"aggregate {name} mismatch: expected={expected_average} actual={actual_average}"
            )
        validated_averages[name] = actual_average

    metadata = _mapping(
        _load_json(output_dir / "evaluation_metadata.json", "evaluation metadata"),
        "evaluation metadata",
    )
    _validate_metadata(
        metadata,
        coordinate=coordinate,
        base_model=base_model,
        evaluation=evaluation,
    )
    partial = validate_partial_output(
        output_dir,
        benchmark=benchmark,
        seed=int(evaluation["seed"]),
        world_size=int(evaluation["world_size"]),
    )
    if partial["completed_ids"] != set(range(benchmark.expected_samples)):
        raise NativeMatrixError("complete output lacks full rank-local resume coverage")
    return {
        "records": len(records),
        "scores": validated_averages,
        "records_by_rank": partial["records_by_rank"],
    }


def _ensure_request(output_dir: Path, request: Mapping[str, Any]) -> Path:
    request_path = output_dir / ".request.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if request_path.exists():
        actual = _mapping(_load_json(request_path, "coordinate request"), "coordinate request")
        if actual != request:
            raise NativeMatrixError(f"coordinate request mismatch: {output_dir}")
    else:
        existing = list(output_dir.iterdir())
        if existing:
            raise NativeMatrixError(
                f"refusing nonempty coordinate without request identity: {output_dir}"
            )
        _atomic_write(request_path, request)
    allowed = {
        ".request.json",
        ".coordinate-complete.json",
        "evaluation_results.jsonl",
        "average_scores.json",
        "evaluation_metadata.json",
    }
    for child in output_dir.iterdir():
        if child.name in allowed:
            if not child.is_file():
                raise NativeMatrixError(f"coordinate artifact is not a regular file: {child}")
        elif PARTIAL_RE.fullmatch(child.name):
            if not child.is_file():
                raise NativeMatrixError(f"partial artifact is not a regular file: {child}")
        else:
            raise NativeMatrixError(f"unexpected coordinate artifact: {child}")
    return request_path


def _completion_artifacts(output_dir: Path) -> dict[str, str]:
    names = [
        "evaluation_results.jsonl",
        "average_scores.json",
        "evaluation_metadata.json",
    ]
    metadata = _mapping(_load_json(output_dir / "evaluation_metadata.json", "metadata"), "metadata")
    world_size = metadata.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise NativeMatrixError("invalid completion world_size")
    names.extend(f".partial-results-rank-{rank:05d}.json" for rank in range(world_size))
    return {name: sha256_file(output_dir / name) for name in names}


def _result_row(
    coordinate: NativeCoordinate,
    output_dir: Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    policy = coordinate.policy
    return {
        "family": policy.family,
        "method": policy.method,
        "run_id": policy.run_id,
        "source_type": policy.source_type,
        "checkpoint_step": policy.checkpoint_step,
        "checkpoint_identity": policy.checkpoint_identity,
        "dataset": coordinate.benchmark.dataset,
        "reward": coordinate.benchmark.reward,
        "guidance_scale": policy.guidance_scale,
        "samples": validation["records"],
        "scores": validation["scores"],
        "output_dir": str(output_dir),
    }


def run_coordinate(
    coordinate: NativeCoordinate,
    *,
    run_root: Path,
    upstream_root: Path,
    evaluator: Path,
    base_model: Path,
    evaluation: Mapping[str, Any],
    manifest_sha256: str,
    evaluator_sha256: str,
    source_id: str,
    run_name: str,
) -> dict[str, Any]:
    """Reuse, resume, or execute one benchmark/policy/CFG coordinate."""

    output_dir = run_root / coordinate.relative_output
    request = _coordinate_request(
        coordinate,
        manifest_sha256=manifest_sha256,
        evaluator_sha256=evaluator_sha256,
        source_id=source_id,
        run_name=run_name,
        evaluation=evaluation,
    )
    request_path = _ensure_request(output_dir, request)
    request_sha = hashlib.sha256(_canonical_json(request)).hexdigest()
    marker_path = output_dir / ".coordinate-complete.json"
    final_names = (
        "evaluation_results.jsonl",
        "average_scores.json",
        "evaluation_metadata.json",
    )
    if marker_path.exists():
        marker = _mapping(_load_json(marker_path, "coordinate completion marker"), "coordinate marker")
        if marker.get("schema_version") != 1 or marker.get("request_sha256") != request_sha:
            raise NativeMatrixError(f"coordinate completion identity mismatch: {output_dir}")
        if marker.get("request_file_sha256") != sha256_file(request_path):
            raise NativeMatrixError(f"coordinate request-file hash mismatch: {output_dir}")
        artifacts = _mapping(marker.get("artifacts"), "coordinate artifacts")
        if artifacts != _completion_artifacts(output_dir):
            raise NativeMatrixError(f"coordinate artifact hash mismatch: {output_dir}")
        validation = validate_complete_output(
            output_dir,
            coordinate=coordinate,
            base_model=base_model,
            evaluation=evaluation,
        )
        if marker.get("records") != validation["records"] or marker.get("scores") != validation["scores"]:
            raise NativeMatrixError(f"coordinate completion payload mismatch: {output_dir}")
        print(f"REUSE {coordinate.relative_output}", flush=True)
        return _result_row(coordinate, output_dir, validation)

    present_finals = [name for name in final_names if (output_dir / name).exists()]
    validation: dict[str, Any] | None = None
    if len(present_finals) == len(final_names):
        try:
            validation = validate_complete_output(
                output_dir,
                coordinate=coordinate,
                base_model=base_model,
                evaluation=evaluation,
            )
        except (PaperRewardsError, OSError) as error:
            # Slurm can interrupt rank 0 while it is serializing the three final
            # files. A matching request plus valid atomic rank-local state is a
            # sufficient witness to rebuild those derived artifacts safely.
            validate_partial_output(
                output_dir,
                benchmark=coordinate.benchmark,
                seed=int(evaluation["seed"]),
                world_size=int(evaluation["world_size"]),
            )
            print(
                f"REBUILD {coordinate.relative_output}: invalid interrupted final artifacts: {error}",
                flush=True,
            )
    if validation is None:
        validate_partial_output(
            output_dir,
            benchmark=coordinate.benchmark,
            seed=int(evaluation["seed"]),
            world_size=int(evaluation["world_size"]),
        )
        command = build_evaluation_command(
            coordinate,
            evaluator=evaluator,
            base_model=base_model,
            output_dir=output_dir,
            evaluation=evaluation,
        )
        _run(command, cwd=upstream_root)
        validation = validate_complete_output(
            output_dir,
            coordinate=coordinate,
            base_model=base_model,
            evaluation=evaluation,
        )

    marker = {
        "schema_version": 1,
        "request_sha256": request_sha,
        "request_file_sha256": sha256_file(request_path),
        "records": validation["records"],
        "scores": validation["scores"],
        "records_by_rank": validation["records_by_rank"],
        "artifacts": _completion_artifacts(output_dir),
    }
    _atomic_write(marker_path, marker)
    return _result_row(coordinate, output_dir, validation)


def _render_csv(results: Sequence[Mapping[str, Any]]) -> str:
    score_columns = list(dict.fromkeys((*OCR_SCORE_NAMES, *GENEVAL_SCORE_NAMES)))
    fieldnames = [
        "family",
        "method",
        "run_id",
        "source_type",
        "checkpoint_step",
        "dataset",
        "guidance_scale",
        "samples",
        *score_columns,
        "output_dir",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for result in results:
        row = {key: result.get(key) for key in fieldnames}
        row.update(result["scores"])
        writer.writerow(row)
    return buffer.getvalue()


def _render_markdown(results: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# SD3.5 13-policy native OCR / GenEval evaluation",
        "",
        "The matrix uses CFG=1 and CFG=4 for every policy. Images are scored in memory and are not persisted.",
        "",
    ]
    for dataset in ("ocr", "geneval"):
        lines.extend([f"## {dataset.upper() if dataset == 'ocr' else 'GenEval'}", ""])
        if dataset == "ocr":
            lines.extend(
                [
                    "| Training reward | Method | Run | Step | CFG | Samples | OCR |",
                    "|---|---|---|---:|---:|---:|---:|",
                ]
            )
        else:
            lines.extend(
                [
                    "| Training reward | Method | Run | Step | CFG | Samples | GenEval | Accuracy | Strict accuracy |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
        for result in results:
            if result["dataset"] != dataset:
                continue
            step = "base" if result["checkpoint_step"] is None else str(result["checkpoint_step"])
            scores = result["scores"]
            common = (
                f"| {result['family']} | {result['method']} | `{result['run_id']}` | "
                f"{step} | {result['guidance_scale']:g} | {result['samples']} |"
            )
            if dataset == "ocr":
                lines.append(f"{common} {scores['ocr']:.6f} |")
            else:
                lines.append(
                    f"{common} {scores['geneval']:.6f} | {scores['accuracy']:.6f} | "
                    f"{scores['strict_accuracy']:.6f} |"
                )
        lines.append("")
    return "\n".join(lines)


def _write_or_validate_json(path: Path, payload: Mapping[str, Any], label: str) -> None:
    if path.exists():
        actual = _mapping(_load_json(path, label), label)
        if actual != payload:
            raise NativeMatrixError(f"existing {label} differs from the current request: {path}")
    else:
        _atomic_write(path, payload)


def run_matrix(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    project_root: Path,
    checkpoint_root: Path,
    run_root: Path,
    base_model: Path,
    run_name: str,
    source_id: str,
    check_only: bool,
) -> dict[str, Any]:
    validate_native_manifest(manifest)
    if not SAFE_NAME_RE.fullmatch(run_name) or not SHA256_RE.fullmatch(source_id):
        raise NativeMatrixError("run_name or source_id is unsafe")
    resolved_project = project_root.resolve(strict=True)
    resolved_manifest = _require_regular_within(
        manifest_path, resolved_project, "native matrix manifest"
    )
    manifest_sha = sha256_file(resolved_manifest)
    upstream_root = (resolved_project / "baselines/diffusion_nft/upstream").resolve(strict=True)
    evaluator = _require_regular_within(
        upstream_root / "scripts/evaluation.py", resolved_project, "native evaluator"
    )
    evaluator_sha = sha256_file(evaluator)
    policies, inventories, source_summaries = load_policies(
        manifest,
        project_root=resolved_project,
        checkpoint_root=checkpoint_root,
        base_model=base_model,
    )
    benchmarks = load_benchmarks(manifest, resolved_project)
    coordinates = build_matrix(policies, benchmarks)
    total_samples = sum(item.benchmark.expected_samples for item in coordinates)
    check_summary = {
        "status": "ok",
        "experiment": EXPERIMENT,
        "manifest_sha256": manifest_sha,
        "source_id": source_id,
        "policies": 13,
        "benchmarks": [item.dataset for item in benchmarks],
        "guidance_scales": manifest["evaluation"]["guidance_scales"],
        "coordinates": len(coordinates),
        "total_sample_generations": total_samples,
        "save_images": manifest["evaluation"]["save_images"],
        "excluded_policies": manifest["excluded_policies"],
    }
    if check_only:
        return check_summary

    if not run_root.is_absolute():
        raise NativeMatrixError("run_root must be absolute")
    run_root.mkdir(parents=True, exist_ok=True)
    matrix_request = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "manifest_sha256": manifest_sha,
        "evaluator_sha256": evaluator_sha,
        "source_id": source_id,
        "run_name": run_name,
        "checkpoint_root": str(checkpoint_root.resolve(strict=True)),
        "base_model": str(base_model.resolve(strict=True)),
        "evaluation": manifest["evaluation"],
        "policies": manifest["policies"],
        "excluded_policies": manifest["excluded_policies"],
        "source_manifests": source_summaries,
    }
    _write_or_validate_json(run_root / ".matrix-request.json", matrix_request, "matrix request")
    inventory_payload = {
        "schema_version": 1,
        "policies": [
            inventories[(str(policy["family"]), str(policy["method"]))]
            for policy in manifest["policies"]
        ],
    }
    _write_or_validate_json(
        run_root / "policy-inventory.json", inventory_payload, "policy inventory"
    )

    results = [
        run_coordinate(
            coordinate,
            run_root=run_root,
            upstream_root=upstream_root,
            evaluator=evaluator,
            base_model=base_model.resolve(strict=True),
            evaluation=manifest["evaluation"],
            manifest_sha256=manifest_sha,
            evaluator_sha256=evaluator_sha,
            source_id=source_id,
            run_name=run_name,
        )
        for coordinate in coordinates
    ]
    summary_payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "manifest_sha256": manifest_sha,
        "source_id": source_id,
        "run_name": run_name,
        "policy_count": 13,
        "coordinate_count": len(results),
        "total_sample_generations": total_samples,
        "save_images": False,
        "excluded_policies": manifest["excluded_policies"],
        "results": results,
    }
    summary_json = run_root / "summary.json"
    summary_csv = run_root / "summary.csv"
    summary_md = run_root / "summary.md"
    complete_path = run_root / ".complete.json"
    if complete_path.exists():
        marker = _mapping(_load_json(complete_path, "matrix completion marker"), "matrix marker")
        if _load_json(summary_json, "matrix summary") != summary_payload:
            raise NativeMatrixError("completed matrix JSON summary no longer matches results")
        expected_artifacts = {
            "summary.json": sha256_file(summary_json),
            "summary.csv": sha256_file(summary_csv),
            "summary.md": sha256_file(summary_md),
            ".matrix-request.json": sha256_file(run_root / ".matrix-request.json"),
            "policy-inventory.json": sha256_file(run_root / "policy-inventory.json"),
        }
        if (
            marker.get("schema_version") != 1
            or marker.get("coordinate_count") != 52
            or marker.get("artifacts") != expected_artifacts
        ):
            raise NativeMatrixError("matrix completion marker or summary hashes mismatch")
        return {**check_summary, "run_root": str(run_root), "reused_complete": True}

    _atomic_write(summary_json, summary_payload)
    _atomic_write_text(summary_csv, _render_csv(results))
    _atomic_write_text(summary_md, _render_markdown(results))
    artifacts = {
        "summary.json": sha256_file(summary_json),
        "summary.csv": sha256_file(summary_csv),
        "summary.md": sha256_file(summary_md),
        ".matrix-request.json": sha256_file(run_root / ".matrix-request.json"),
        "policy-inventory.json": sha256_file(run_root / "policy-inventory.json"),
    }
    _atomic_write(
        complete_path,
        {"schema_version": 1, "coordinate_count": 52, "artifacts": artifacts},
    )
    return {**check_summary, "run_root": str(run_root), "reused_complete": False}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = _mapping(_load_json(args.manifest, "native matrix manifest"), "manifest")
        summary = run_matrix(
            manifest_path=args.manifest,
            manifest=manifest,
            project_root=args.project_root,
            checkpoint_root=args.checkpoint_root,
            run_root=args.run_root,
            base_model=args.base_model,
            run_name=args.run_name,
            source_id=args.source_id,
            check_only=args.check_only,
        )
    except (PaperRewardsError, OSError, ValueError, KeyError) as error:
        print(f"Native OCR/GenEval matrix failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
