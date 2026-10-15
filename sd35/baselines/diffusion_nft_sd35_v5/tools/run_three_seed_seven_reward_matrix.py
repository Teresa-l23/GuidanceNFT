#!/usr/bin/env python3
"""Run one array coordinate from the formal SD3.5 three-seed/seven-reward matrix."""

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
import statistics
import sys
from typing import Any, Mapping, Sequence

from baselines.diffusion_nft_sd35_v5.tools.run_drawbench_paper_rewards import (
    Coordinate,
    PaperRewardsError,
    _atomic_write,
    _atomic_write_text,
    _canonical_json,
    _load_json,
    _mapping,
    _run,
    run_generation,
    run_rescore,
    sha256_file,
)
from baselines.diffusion_nft_sd35_v5.tools.run_native_ocr_geneval_matrix import (
    Benchmark,
    NativeCoordinate,
    NativeMatrixError,
    _completion_artifacts,
    _ensure_request,
    validate_complete_output,
    validate_partial_output,
)


EXPERIMENT = "sd35_four_family_three_seed_seven_reward_eval"
FAMILY_STEPS = {"learned3": 200, "geneval": 200, "ocr": 50, "pickscore": 100}
SOURCE_KEYS = ("standard", "dual", "matched")
POLICY_PLAN = (
    ("standard", "standard", (1.0, 4.0), "same_model"),
    ("dual", "dual", (4.0,), "same_model"),
    ("pretrained_unconditional", "standard", (4.0,), "pretrained_base"),
    ("matched", "matched", (1.0, 4.0), "same_model"),
)
SEEDS = (42, 123, 2026)
DRAWBENCH_REWARDS = ("pickscore", "clipscore", "hpsv2", "aesthetic", "imagereward")
ALL_REWARDS = (*DRAWBENCH_REWARDS, "geneval", "ocr")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MatrixError(PaperRewardsError):
    """Raised when a matrix input or artifact violates the frozen protocol."""


@dataclass(frozen=True)
class MatrixCoordinate:
    index: int
    family: str
    policy: str
    source_key: str
    source: Mapping[str, Any]
    guidance_scale: float
    seed: int
    cfg_unconditional_source: str

    @property
    def guidance_tag(self) -> str:
        return f"cfg{int(self.guidance_scale)}"

    @property
    def relative_output(self) -> Path:
        return Path(
            "formal",
            self.family,
            self.policy,
            f"seed-{self.seed}",
            self.guidance_tag,
        )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject any change to the policy, seed, CFG, dataset, or checkpoint contract."""

    if manifest.get("schema_version") != 1 or manifest.get("experiment") != EXPERIMENT:
        raise MatrixError("matrix manifest identity mismatch")
    if os.path.expandvars(str(manifest.get("checkpoint_root"))) != os.path.expandvars(
        "${GUIDANCENFT_ASSETS}/runs/diffusion_nft_sd35_v5"
    ):
        raise MatrixError("checkpoint_root mismatch")
    base_model = _mapping(manifest.get("base_model"), "base_model")
    if base_model != {
        "repo_id": "stabilityai/stable-diffusion-3.5-medium",
        "revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
        "model_index_sha256": "b5e97ca0d3c0cd9a1e3ed91a71bbdd489de7941b23fe038b3aaa9acf36fd1543",
    }:
        raise MatrixError("base-model identity mismatch")

    evaluation = _mapping(manifest.get("evaluation"), "evaluation")
    expected_scalar = {
        "seeds": list(SEEDS),
        "num_inference_steps": 40,
        "resolution": 512,
        "mixed_precision": "fp16",
        "world_size": 2,
        "save_drawbench_images": True,
        "save_native_images": False,
    }
    for key, expected in expected_scalar.items():
        if evaluation.get(key) != expected:
            raise MatrixError(f"evaluation.{key} mismatch")
    expected_plan = [
        {
            "key": key,
            "source": source,
            "guidance_scales": list(cfgs),
            "cfg_unconditional_source": unconditional,
        }
        for key, source, cfgs, unconditional in POLICY_PLAN
    ]
    if evaluation.get("policy_plan") != expected_plan:
        raise MatrixError("policy plan mismatch")

    datasets = _mapping(evaluation.get("datasets"), "datasets")
    expected_datasets = {
        "drawbench": (1000, 8, list(DRAWBENCH_REWARDS)),
        "geneval": (2212, 14, ["geneval"]),
        "ocr": (1018, 16, ["ocr"]),
    }
    if list(datasets) != list(expected_datasets):
        raise MatrixError("dataset order mismatch")
    for key, (samples, batch, rewards) in expected_datasets.items():
        spec = _mapping(datasets[key], f"dataset {key}")
        if (
            spec.get("expected_samples") != samples
            or spec.get("batch_size_per_rank") != batch
            or spec.get("rewards") != rewards
            or not _is_sha256(spec.get("dataset_sha256"))
        ):
            raise MatrixError(f"dataset {key} contract mismatch")
    if datasets["drawbench"].get("rescore_batch_size_per_rank") != {
        "clipscore": 16,
        "hpsv2": 16,
        "aesthetic": 16,
        "imagereward": 8,
    }:
        raise MatrixError("DrawBench rescore batches mismatch")

    families = manifest.get("families")
    if not isinstance(families, list) or [item.get("key") for item in families] != list(FAMILY_STEPS):
        raise MatrixError("family order mismatch")
    seen_runs: set[str] = set()
    for family in families:
        family_key = str(family["key"])
        if family.get("training_step") != FAMILY_STEPS[family_key]:
            raise MatrixError(f"family step mismatch: {family_key}")
        sources = _mapping(family.get("sources"), f"{family_key} sources")
        if tuple(sources) != SOURCE_KEYS:
            raise MatrixError(f"source order mismatch: {family_key}")
        for source_key in SOURCE_KEYS:
            source = _mapping(sources[source_key], f"{family_key}/{source_key}")
            run_id = source.get("run_id")
            if (
                not isinstance(run_id, str)
                or not run_id
                or Path(run_id).name != run_id
                or run_id in seen_runs
            ):
                raise MatrixError(f"unsafe or duplicate run ID: {run_id!r}")
            seen_runs.add(run_id)
            if (
                source.get("checkpoint_step") != FAMILY_STEPS[family_key]
                or source.get("checkpoint_storage") != "checkpoints"
                or source.get("training_method")
                != ("v5flow" if source_key == "matched" else "old_center")
                or source.get("cfg_mode")
                != {"standard": "cfg1", "dual": "cfg1_dual", "matched": "cfg4_rollout"}[source_key]
                or not _is_sha256(source.get("config_sha256"))
                or not _is_sha256(source.get("adapter_model_sha256"))
            ):
                raise MatrixError(f"checkpoint contract mismatch: {family_key}/{source_key}")


def build_coordinates(manifest: Mapping[str, Any]) -> list[MatrixCoordinate]:
    validate_manifest(manifest)
    coordinates: list[MatrixCoordinate] = []
    for family in manifest["families"]:
        for policy, source_key, guidance_scales, unconditional in POLICY_PLAN:
            for guidance_scale in guidance_scales:
                for seed in SEEDS:
                    coordinates.append(
                        MatrixCoordinate(
                            index=len(coordinates),
                            family=str(family["key"]),
                            policy=policy,
                            source_key=source_key,
                            source=family["sources"][source_key],
                            guidance_scale=guidance_scale,
                            seed=seed,
                            cfg_unconditional_source=unconditional,
                        )
                    )
    if len(coordinates) != 72:
        raise MatrixError("expected exactly 72 seed/policy/CFG coordinates")
    return coordinates


def _checkpoint_path(root: Path, coordinate: MatrixCoordinate) -> Path:
    return (
        root
        / str(coordinate.source["run_id"])
        / "train"
        / str(coordinate.source["checkpoint_storage"])
        / f"checkpoint-{coordinate.source['checkpoint_step']}"
    )


def validate_base_model(manifest: Mapping[str, Any], base_model: Path) -> Path:
    resolved = base_model.resolve(strict=True)
    revision = str(manifest["base_model"]["revision"])
    if not resolved.is_dir() or resolved.name != revision:
        raise MatrixError("base model does not resolve to the pinned SD3.5 revision")
    marker = resolved / ".diffusion_nft_revision"
    model_index = resolved / "model_index.json"
    if marker.read_text(encoding="utf-8").strip() != revision:
        raise MatrixError("base-model revision marker mismatch")
    if sha256_file(model_index) != manifest["base_model"]["model_index_sha256"]:
        raise MatrixError("base-model model_index hash mismatch")
    return resolved


def validate_datasets(manifest: Mapping[str, Any], project_root: Path) -> None:
    for key, spec in manifest["evaluation"]["datasets"].items():
        path = (project_root / spec["dataset_file"]).resolve(strict=True)
        if not path.is_file() or sha256_file(path) != spec["dataset_sha256"]:
            raise MatrixError(f"dataset identity mismatch: {key}")
        if len(path.read_text(encoding="utf-8").splitlines()) != spec["expected_samples"]:
            raise MatrixError(f"dataset sample count mismatch: {key}")


def validate_checkpoint(
    root: Path, coordinate: MatrixCoordinate, *, deep: bool
) -> Path:
    source = coordinate.source
    checkpoint = _checkpoint_path(root, coordinate).resolve(strict=True)
    run_root = checkpoint.parents[2]
    config_path = run_root / "provenance/config.json"
    adapter_path = checkpoint / "lora/adapter_model.safetensors"
    if sha256_file(config_path) != source["config_sha256"]:
        raise MatrixError(f"config hash mismatch: {coordinate.family}/{coordinate.source_key}")
    if sha256_file(adapter_path) != source["adapter_model_sha256"]:
        raise MatrixError(f"adapter hash mismatch: {coordinate.family}/{coordinate.source_key}")
    config = _mapping(_load_json(config_path, "training config"), "training config")
    training = _mapping(config.get("training"), "training")
    if (
        config.get("task") != source["task"]
        or config.get("cfg_mode") != source["cfg_mode"]
        or training.get("max_train_steps") < source["checkpoint_step"]
    ):
        raise MatrixError(f"training provenance mismatch: {coordinate.family}/{coordinate.source_key}")
    marker = _mapping(
        _load_json(checkpoint / ".checkpoint_complete.json", "checkpoint marker"),
        "checkpoint marker",
    )
    if (
        marker.get("checkpoint_step") != source["checkpoint_step"]
        or marker.get("cfg_mode") != source["cfg_mode"]
        or marker.get("checkpoint_kind") not in {"outer_boundary", "final"}
    ):
        raise MatrixError(f"checkpoint marker mismatch: {coordinate.family}/{coordinate.source_key}")
    if deep:
        from baselines.diffusion_nft_sd35_v5.tools.validate_v5flow_checkpoint import (
            load_and_validate_resumable,
        )

        _, summary = load_and_validate_resumable(
            checkpoint,
            require_complete_marker=True,
            expect_method=str(source["training_method"]),
            expect_cfg_mode=str(source["cfg_mode"]),
        )
        if summary["checkpoint_step"] != source["checkpoint_step"]:
            raise MatrixError("deep checkpoint validation returned the wrong step")
    return checkpoint


def _policy_coordinate(
    matrix_coordinate: MatrixCoordinate, checkpoint: Path
) -> Coordinate:
    return Coordinate(
        family=matrix_coordinate.family,
        method=matrix_coordinate.policy,
        run_id=str(matrix_coordinate.source["run_id"]),
        guidance_scale=matrix_coordinate.guidance_scale,
        checkpoint=checkpoint,
        checkpoint_lora_subdir="lora",
        checkpoint_step=int(matrix_coordinate.source["checkpoint_step"]),
        checkpoint_identity=str(matrix_coordinate.source["adapter_model_sha256"]),
        source_type="checkpoint",
    )


def _evaluator_contract(
    project_root: Path, coordinate: MatrixCoordinate
) -> tuple[Path, tuple[str, ...]]:
    if coordinate.cfg_unconditional_source == "pretrained_base":
        if coordinate.guidance_scale != 4.0 or coordinate.source_key != "standard":
            raise MatrixError("pretrained-unconditional routing is restricted to Standard CFG4")
        return (
            project_root
            / "baselines/diffusion_nft_sd35_v5/scripts/evaluate_pretrained_null_cfg_sd35.py",
            ("--cfg_unconditional_source", "pretrained_base"),
        )
    return (
        project_root / "baselines/diffusion_nft/upstream/scripts/evaluation.py",
        (),
    )


def _validate_routing(metadata_path: Path, coordinate: MatrixCoordinate) -> None:
    metadata = _mapping(_load_json(metadata_path, "evaluation metadata"), "metadata")
    actual = metadata.get("cfg_unconditional_source")
    if coordinate.cfg_unconditional_source == "pretrained_base":
        expected = {
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
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise MatrixError(f"pretrained-unconditional metadata mismatch: {key}")
        for key in ("cfg_rank0_transformer_forward_calls", "cfg_rank0_routed_sample_pairs"):
            value = metadata.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MatrixError(f"invalid pretrained-unconditional routing counter: {key}")
    elif actual not in {None, "same_model"}:
        raise MatrixError(f"unexpected same-model routing metadata: {actual!r}")


def _load_benchmark(
    manifest: Mapping[str, Any], project_root: Path, dataset: str
) -> Benchmark:
    spec = manifest["evaluation"]["datasets"][dataset]
    path = (project_root / spec["dataset_file"]).resolve(strict=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    if dataset == "geneval":
        metadatas = tuple(json.loads(line) for line in lines)
        prompts = tuple(item["prompt"] for item in metadatas)
    else:
        prompts = tuple(line.strip() for line in lines)
        metadatas = tuple({} for _ in prompts)
    return Benchmark(
        dataset=dataset,
        reward=dataset,
        dataset_path=path,
        dataset_sha256=str(spec["dataset_sha256"]),
        expected_samples=int(spec["expected_samples"]),
        batch_size_per_rank=int(spec["batch_size_per_rank"]),
        prompts=prompts,
        metadatas=metadatas,
    )


def _native_command(
    native: NativeCoordinate,
    *,
    evaluator: Path,
    evaluator_extra_args: Sequence[str],
    base_model: Path,
    output_dir: Path,
    evaluation: Mapping[str, Any],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={evaluation['world_size']}",
        str(evaluator),
        *evaluator_extra_args,
        "--base_model",
        str(base_model),
        "--local_files_only",
        "--seed",
        str(evaluation["seed"]),
        "--model_type",
        "sd3",
        "--dataset",
        native.benchmark.dataset,
        "--reward",
        native.benchmark.reward,
        "--output_dir",
        str(output_dir),
        "--num_inference_steps",
        str(evaluation["num_inference_steps"]),
        "--guidance_scale",
        str(native.policy.guidance_scale),
        "--resolution",
        str(evaluation["resolution"]),
        "--mixed_precision",
        str(evaluation["mixed_precision"]),
        "--batch_size",
        str(native.benchmark.batch_size_per_rank),
        "--checkpoint_path",
        str(native.policy.checkpoint),
        "--checkpoint_lora_subdir",
        "lora",
    ]


def run_native_reward(
    matrix_coordinate: MatrixCoordinate,
    policy_coordinate: Coordinate,
    dataset: str,
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    project_root: Path,
    upstream_root: Path,
    base_model: Path,
    evaluator: Path,
    evaluator_extra_args: Sequence[str],
    source_id: str,
) -> float:
    benchmark = _load_benchmark(manifest, project_root, dataset)
    native = NativeCoordinate(policy=policy_coordinate, benchmark=benchmark)
    evaluation = {
        "seed": matrix_coordinate.seed,
        "num_inference_steps": 40,
        "guidance_scales": [matrix_coordinate.guidance_scale],
        "resolution": 512,
        "mixed_precision": "fp16",
        "world_size": int(manifest["evaluation"]["world_size"]),
        "sampler": "diffusers",
        "deterministic": True,
        "save_images": False,
    }
    request = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "manifest_sha256": manifest_sha256,
        "source_id": source_id,
        "task_index": matrix_coordinate.index,
        "family": matrix_coordinate.family,
        "policy": matrix_coordinate.policy,
        "source_key": matrix_coordinate.source_key,
        "run_id": policy_coordinate.run_id,
        "checkpoint": str(policy_coordinate.checkpoint),
        "checkpoint_step": policy_coordinate.checkpoint_step,
        "adapter_model_sha256": matrix_coordinate.source["adapter_model_sha256"],
        "dataset": dataset,
        "dataset_sha256": benchmark.dataset_sha256,
        "expected_samples": benchmark.expected_samples,
        "batch_size_per_rank": benchmark.batch_size_per_rank,
        "seed": matrix_coordinate.seed,
        "guidance_scale": matrix_coordinate.guidance_scale,
        "cfg_unconditional_source": matrix_coordinate.cfg_unconditional_source,
        "evaluator": str(evaluator),
        "evaluator_sha256": sha256_file(evaluator),
        "evaluator_extra_args": list(evaluator_extra_args),
        "base_model": str(base_model),
    }
    request_path = _ensure_request(output_dir, request)
    request_sha = hashlib.sha256(_canonical_json(request)).hexdigest()
    marker_path = output_dir / ".coordinate-complete.json"
    if marker_path.exists():
        marker = _mapping(_load_json(marker_path, "native completion marker"), "marker")
        if (
            marker.get("schema_version") != 1
            or marker.get("request_sha256") != request_sha
            or marker.get("request_file_sha256") != sha256_file(request_path)
            or marker.get("artifacts") != _completion_artifacts(output_dir)
        ):
            raise MatrixError(f"native completion identity mismatch: {output_dir}")
        validation = validate_complete_output(
            output_dir,
            coordinate=native,
            base_model=base_model,
            evaluation=evaluation,
        )
    else:
        validate_partial_output(
            output_dir,
            benchmark=benchmark,
            seed=matrix_coordinate.seed,
            world_size=int(manifest["evaluation"]["world_size"]),
        )
        command = _native_command(
            native,
            evaluator=evaluator,
            evaluator_extra_args=evaluator_extra_args,
            base_model=base_model,
            output_dir=output_dir,
            evaluation=evaluation,
        )
        _run(command, cwd=upstream_root)
        validation = validate_complete_output(
            output_dir,
            coordinate=native,
            base_model=base_model,
            evaluation=evaluation,
        )
        _validate_routing(output_dir / "evaluation_metadata.json", matrix_coordinate)
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
    _validate_routing(output_dir / "evaluation_metadata.json", matrix_coordinate)
    metric = manifest["evaluation"].get("geneval_metric", "geneval") if dataset == "geneval" else dataset
    if dataset == "geneval" and metric not in {"geneval", "accuracy"}:
        raise MatrixError(f"unsupported GenEval metric: {metric}")
    value = validation["scores"].get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MatrixError(f"invalid native {dataset} result")
    return float(value)


def run_matrix_coordinate(
    coordinate: MatrixCoordinate,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    checkpoint_root: Path,
    run_root: Path,
    base_model: Path,
    project_root: Path,
    source_id: str,
) -> dict[str, Any]:
    checkpoint = validate_checkpoint(checkpoint_root, coordinate, deep=False)
    policy = _policy_coordinate(coordinate, checkpoint)
    evaluator, evaluator_extra_args = _evaluator_contract(project_root, coordinate)
    upstream_root = project_root / "baselines/diffusion_nft/upstream"
    rescorer = project_root / "baselines/diffusion_nft_sd35_v5/tools/rescore_saved_drawbench.py"
    coordinate_root = run_root / coordinate.relative_output
    drawbench_root = coordinate_root / "drawbench"
    common = {
        "upstream_root": upstream_root,
        "base_model": base_model,
        "sample_count": 1000,
        "full_dataset_samples": 1000,
        "seed": coordinate.seed,
        "eval_steps": 40,
        "resolution": 512,
        "source_id": source_id,
        "world_size": int(manifest["evaluation"]["world_size"]),
    }
    generation = run_generation(
        policy,
        drawbench_root / "generation",
        evaluator=evaluator,
        evaluator_extra_args=evaluator_extra_args,
        batch_size=8,
        **common,
    )
    _validate_routing(drawbench_root / "generation/evaluation_metadata.json", coordinate)
    scores: dict[str, float] = {"pickscore": float(generation["pickscore"])}
    drawbench_spec = manifest["evaluation"]["datasets"]["drawbench"]
    for reward in DRAWBENCH_REWARDS[1:]:
        scores.update(
            run_rescore(
                policy,
                drawbench_root / "generation",
                drawbench_root / "scores" / reward,
                reward,
                rescorer=rescorer,
                dataset_file=upstream_root / "dataset/drawbench/test.txt",
                generation_batch_size=8,
                batch_size=int(drawbench_spec["rescore_batch_size_per_rank"][reward]),
                **common,
            )
        )
    manifest_sha256 = sha256_file(manifest_path)
    for dataset in ("geneval", "ocr"):
        scores[dataset] = run_native_reward(
            coordinate,
            policy,
            dataset,
            coordinate_root / dataset,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            project_root=project_root,
            upstream_root=upstream_root,
            base_model=base_model,
            evaluator=evaluator,
            evaluator_extra_args=evaluator_extra_args,
            source_id=source_id,
        )
    if set(scores) != set(ALL_REWARDS) or any(not math.isfinite(value) for value in scores.values()):
        raise MatrixError("coordinate did not produce seven finite rewards")
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "task_index": coordinate.index,
        "family": coordinate.family,
        "training_step": int(coordinate.source["checkpoint_step"]),
        "policy": coordinate.policy,
        "source_key": coordinate.source_key,
        "run_id": coordinate.source["run_id"],
        "checkpoint_step": coordinate.source["checkpoint_step"],
        "adapter_model_sha256": coordinate.source["adapter_model_sha256"],
        "seed": coordinate.seed,
        "guidance_scale": coordinate.guidance_scale,
        "cfg_unconditional_source": coordinate.cfg_unconditional_source,
        "scores": scores,
        "samples": {"drawbench": 1000, "geneval": 2212, "ocr": 1018},
        "source_id": source_id,
        "manifest_sha256": manifest_sha256,
    }
    if manifest["evaluation"].get("geneval_metric") == "accuracy":
        result["score_semantics"] = {"geneval": "accuracy (non-strict)"}
    result_path = coordinate_root / "result.json"
    _atomic_write(result_path, result)
    _atomic_write(
        coordinate_root / ".complete.json",
        {
            "schema_version": 1,
            "task_index": coordinate.index,
            "result_sha256": sha256_file(result_path),
            "source_id": source_id,
            "manifest_sha256": manifest_sha256,
        },
    )
    return result


def aggregate_results(
    coordinates: Sequence[MatrixCoordinate],
    *,
    run_root: Path,
    manifest_sha256: str,
    source_id: str,
) -> dict[str, Any]:
    """Validate all array artifacts and publish the three-seed mean/sample std."""

    results: list[Mapping[str, Any]] = []
    for coordinate in coordinates:
        coordinate_root = run_root / coordinate.relative_output
        result_path = coordinate_root / "result.json"
        marker_path = coordinate_root / ".complete.json"
        result = _mapping(_load_json(result_path, "coordinate result"), "coordinate result")
        marker = _mapping(_load_json(marker_path, "coordinate completion"), "completion")
        expected_identity = {
            "task_index": coordinate.index,
            "family": coordinate.family,
            "policy": coordinate.policy,
            "source_key": coordinate.source_key,
            "run_id": coordinate.source["run_id"],
            "checkpoint_step": coordinate.source["checkpoint_step"],
            "seed": coordinate.seed,
            "guidance_scale": coordinate.guidance_scale,
            "cfg_unconditional_source": coordinate.cfg_unconditional_source,
            "source_id": source_id,
            "manifest_sha256": manifest_sha256,
        }
        for key, expected in expected_identity.items():
            if result.get(key) != expected:
                raise MatrixError(f"coordinate result identity mismatch: {coordinate.relative_output}/{key}")
        scores = _mapping(result.get("scores"), "coordinate scores")
        if set(scores) != set(ALL_REWARDS) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in scores.values()
        ):
            raise MatrixError(f"coordinate scores are incomplete: {coordinate.relative_output}")
        if marker != {
            "schema_version": 1,
            "task_index": coordinate.index,
            "result_sha256": sha256_file(result_path),
            "source_id": source_id,
            "manifest_sha256": manifest_sha256,
        }:
            raise MatrixError(f"coordinate completion mismatch: {coordinate.relative_output}")
        results.append(result)

    aggregates: list[dict[str, Any]] = []
    for family in FAMILY_STEPS:
        for policy, _source_key, guidance_scales, unconditional in POLICY_PLAN:
            for guidance_scale in guidance_scales:
                selected = [
                    result
                    for result in results
                    if result["family"] == family
                    and result["policy"] == policy
                    and result["guidance_scale"] == guidance_scale
                ]
                if [result["seed"] for result in selected] != list(SEEDS):
                    raise MatrixError(f"seed coverage mismatch: {family}/{policy}/cfg{guidance_scale:g}")
                reward_summary: dict[str, Any] = {}
                for reward in ALL_REWARDS:
                    values = [float(result["scores"][reward]) for result in selected]
                    reward_summary[reward] = {
                        "values": values,
                        "mean": statistics.mean(values),
                        "sample_std": statistics.stdev(values),
                    }
                aggregates.append(
                    {
                        "family": family,
                        "training_step": FAMILY_STEPS[family],
                        "policy": policy,
                        "guidance_scale": guidance_scale,
                        "cfg_unconditional_source": unconditional,
                        "seeds": list(SEEDS),
                        "rewards": reward_summary,
                    }
                )
    if len(aggregates) != 24:
        raise MatrixError("expected exactly 24 policy/CFG aggregate rows")
    summary = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "manifest_sha256": manifest_sha256,
        "source_id": source_id,
        "coordinate_count": len(results),
        "aggregate_row_count": len(aggregates),
        "seeds": list(SEEDS),
        "reward_names": list(ALL_REWARDS),
        "aggregates": aggregates,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary.json"
    _atomic_write(summary_path, summary)

    columns = [
        "family",
        "training_step",
        "policy",
        "guidance_scale",
        "cfg_unconditional_source",
        *[f"{reward}_mean" for reward in ALL_REWARDS],
        *[f"{reward}_sample_std" for reward in ALL_REWARDS],
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for aggregate in aggregates:
        row = {key: aggregate[key] for key in columns[:5]}
        row.update(
            {
                f"{reward}_{statistic}": aggregate["rewards"][reward][statistic]
                for reward in ALL_REWARDS
                for statistic in ("mean", "sample_std")
            }
        )
        writer.writerow(row)
    csv_path = run_root / "summary.csv"
    _atomic_write_text(csv_path, buffer.getvalue())
    _atomic_write(
        run_root / ".complete.json",
        {
            "schema_version": 1,
            "coordinate_count": 72,
            "aggregate_row_count": 24,
            "summary_sha256": sha256_file(summary_path),
            "summary_csv_sha256": sha256_file(csv_path),
            "manifest_sha256": manifest_sha256,
            "source_id": source_id,
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    if sum((args.check_only, args.aggregate_only, args.task_index is not None)) != 1:
        raise MatrixError("choose exactly one of --check-only, --aggregate-only, or --task-index")
    for path, label in (
        (args.manifest, "manifest"),
        (args.checkpoint_root, "checkpoint root"),
        (args.run_root, "run root"),
        (args.base_model, "base model"),
    ):
        if not path.is_absolute():
            raise MatrixError(f"{label} must be absolute: {path}")
    if not _is_sha256(args.source_id):
        raise MatrixError("source ID must be a lowercase SHA-256")
    manifest = _mapping(_load_json(args.manifest, "matrix manifest"), "manifest")
    coordinates = build_coordinates(manifest)
    project_root = Path(__file__).resolve().parents[3]
    validate_base_model(manifest, args.base_model)
    validate_datasets(manifest, project_root)
    manifest_sha256 = sha256_file(args.manifest)
    if args.check_only:
        checked: set[tuple[str, str]] = set()
        for coordinate in coordinates:
            key = (coordinate.family, coordinate.source_key)
            if key not in checked:
                validate_checkpoint(args.checkpoint_root, coordinate, deep=True)
                checked.add(key)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "checkpoints": len(checked),
                    "array_tasks": len(coordinates),
                    "families": list(FAMILY_STEPS),
                    "seeds": list(SEEDS),
                    "rewards": list(ALL_REWARDS),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.aggregate_only:
        summary = aggregate_results(
            coordinates,
            run_root=args.run_root,
            manifest_sha256=manifest_sha256,
            source_id=args.source_id,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "coordinates": summary["coordinate_count"],
                    "aggregate_rows": summary["aggregate_row_count"],
                    "summary": str(args.run_root / "summary.json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    assert args.task_index is not None
    if not 0 <= args.task_index < len(coordinates):
        raise MatrixError(f"task index is outside [0,{len(coordinates) - 1}]")
    result = run_matrix_coordinate(
        coordinates[args.task_index],
        manifest=manifest,
        manifest_path=args.manifest,
        checkpoint_root=args.checkpoint_root,
        run_root=args.run_root,
        base_model=args.base_model.resolve(strict=True),
        project_root=project_root,
        source_id=args.source_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MatrixError, NativeMatrixError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
