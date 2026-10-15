#!/usr/bin/env python3
"""Run a resumable multi-policy, 2-CFG DrawBench learned-reward matrix."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from typing import Any, Mapping, Sequence


PAPER_REWARDS = ("pickscore", "clipscore", "hpsv2", "aesthetic", "imagereward")
RESCORE_REWARDS = PAPER_REWARDS[1:]
FAMILY_KEYS = ("geneval", "pickscore", "learned3")
ADDITIONAL_FAMILY_KEYS = ("ocr",)
METHOD_KEYS = ("standard", "dual", "fm100")
GENEVAL_METHOD_KEYS = ("standard", "dual", "fm", "fm100")
SPECIAL_METHOD_KEYS = {
    "learned3_v5dual": ("pretrained_v5", "cfg4_v5dual"),
    "sd35_base": ("base",),
}
CHECKPOINT_LORA_SUBDIRS = ("lora", "lora/v5", "lora/old")
SOURCE_TYPES = ("checkpoint", "base_model")


class PaperRewardsError(RuntimeError):
    """Raised when runtime inputs or persistent outputs violate the contract."""


@dataclass(frozen=True)
class Coordinate:
    family: str
    method: str
    run_id: str
    guidance_scale: float
    checkpoint: Path | None
    checkpoint_lora_subdir: str | None = "lora"
    checkpoint_step: int | None = 100
    checkpoint_identity: str = ""
    source_type: str = "checkpoint"

    @property
    def guidance_tag(self) -> str:
        return "cfg1" if self.guidance_scale == 1.0 else "cfg4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_json_text(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PaperRewardsError(f"missing {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise PaperRewardsError(f"invalid {label}: {path}: {error}") from error


def _load_jsonl(path: Path, label: str) -> list[Any]:
    records: list[Any] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise PaperRewardsError(f"blank {label} record at line {line_number}")
                records.append(json.loads(line))
    except FileNotFoundError as error:
        raise PaperRewardsError(f"missing {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise PaperRewardsError(f"invalid {label}: {path}: {error}") from error
    return records


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PaperRewardsError(f"{label} must be a JSON object")
    return value


def _checkpoint_lora_subdir(method: Mapping[str, Any]) -> str:
    value = method.get("checkpoint_lora_subdir", "lora")
    if value not in CHECKPOINT_LORA_SUBDIRS:
        raise PaperRewardsError(
            f"unsupported checkpoint LoRA subdirectory: {value!r}"
        )
    return str(value)


def _source_type(method: Mapping[str, Any]) -> str:
    value = method.get("source_type", "checkpoint")
    if value not in SOURCE_TYPES:
        raise PaperRewardsError(f"unsupported policy source type: {value!r}")
    return str(value)


def validate_manifest(manifest: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Validate a canonical family subset and its explicit method rows."""

    if manifest.get("schema_version") != 2:
        raise PaperRewardsError("manifest must use schema 2")
    evaluation = _mapping(manifest.get("evaluation"), "evaluation")
    expected = {
        "dataset": "drawbench",
        "generation_reward": "pickscore",
        "rescore_rewards": list(RESCORE_REWARDS),
        "reported_rewards": list(PAPER_REWARDS),
        "seed": 42,
        "num_inference_steps": 40,
        "guidance_scales": [1.0, 4.0],
        "resolution": 512,
        "mixed_precision": "fp16",
        "generation_batch_size_per_rank": 8,
        "expected_samples": 1000,
        "smoke_samples": 20,
        "save_all_images": True,
    }
    for key, value in expected.items():
        if evaluation.get(key) != value:
            raise PaperRewardsError(f"manifest evaluation.{key} mismatch")
    if evaluation.get("world_size") not in {1, 2}:
        raise PaperRewardsError("manifest evaluation.world_size must be 1 or 2")
    if evaluation.get("rescore_batch_size_per_rank") != {
        "clipscore": 16,
        "hpsv2": 16,
        "aesthetic": 16,
        "imagereward": 8,
    }:
        raise PaperRewardsError("manifest rescore batch sizes mismatch")
    if set(_mapping(evaluation.get("unavailable_rewards"), "unavailable rewards")) != {
        "unifiedreward"
    }:
        raise PaperRewardsError("UnifiedReward must be the sole unavailable scorer")

    families = manifest.get("families")
    if not isinstance(families, list) or not families:
        raise PaperRewardsError("manifest families must be a nonempty list")
    family_keys = [
        _mapping(item, f"families[{index}]").get("key")
        for index, item in enumerate(families)
    ]
    canonical_family_keys = [
        key
        for key in (*FAMILY_KEYS, *ADDITIONAL_FAMILY_KEYS, *SPECIAL_METHOD_KEYS)
        if key in family_keys
    ]
    if family_keys != canonical_family_keys:
        raise PaperRewardsError("manifest families are not a unique canonical subset")
    rows: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    policy_inputs: set[tuple[str, str, object, object, object]] = set()
    for family_value in families:
        family = _mapping(family_value, "family")
        methods = family.get("methods")
        expected_methods = SPECIAL_METHOD_KEYS.get(
            str(family.get("key")),
            (
                GENEVAL_METHOD_KEYS
                if family.get("key") == "geneval"
                and isinstance(methods, list)
                and any(item.get("key") == "fm" for item in methods)
                else METHOD_KEYS
            ),
        )
        if (
            not isinstance(methods, list)
            or [item.get("key") for item in methods] != list(expected_methods)
        ):
            raise PaperRewardsError(f"method order mismatch for {family.get('key')}")
        for method_value in methods:
            method = _mapping(method_value, "method")
            run_id = method.get("run_id")
            source_type = _source_type(method)
            if source_type == "base_model":
                forbidden_checkpoint_fields = {
                    "checkpoint_step",
                    "checkpoint_storage",
                    "checkpoint_lora_subdir",
                    "adapter_model_sha256",
                    "config_sha256",
                    "max_train_steps",
                    "resume_epoch",
                    "cfg_mode",
                    "checkpoint_kind",
                    "require_zero_delta_lora",
                }
                present_fields = sorted(forbidden_checkpoint_fields.intersection(method))
                if present_fields:
                    raise PaperRewardsError(
                        "base-model policy declares checkpoint fields: "
                        f"{present_fields}"
                    )
                if (
                    method.get("method") != "base_model"
                    or method.get("require_no_adapter") is not True
                ):
                    raise PaperRewardsError(
                        "base-model policy must require an adapter-free base_model"
                    )
                policy_input = (source_type, run_id, None, None, None)
            else:
                policy_input = (
                    source_type,
                    run_id,
                    method.get("checkpoint_step", manifest.get("checkpoint_step")),
                    method.get("checkpoint_storage", "checkpoints"),
                    _checkpoint_lora_subdir(method),
                )
            if (
                not isinstance(run_id, str)
                or not run_id
                or policy_input in policy_inputs
            ):
                raise PaperRewardsError(
                    f"invalid or duplicate policy input: {policy_input!r}"
                )
            policy_inputs.add(policy_input)
            rows.append((family, method))
    return rows


def validate_zero_delta_lora(lora_dir: Path) -> dict[str, Any]:
    """Prove that a PEFT LoRA checkpoint has an exactly zero effective delta."""

    adapter_config = _mapping(
        _load_json(lora_dir / "adapter_config.json", "LoRA adapter config"),
        "LoRA adapter config",
    )
    if adapter_config.get("bias") != "none":
        raise PaperRewardsError("zero-delta LoRA must use bias='none'")
    if adapter_config.get("modules_to_save") not in (None, []):
        raise PaperRewardsError("zero-delta LoRA cannot contain modules_to_save")
    if adapter_config.get("use_dora", False) is not False:
        raise PaperRewardsError("zero-delta LoRA cannot enable DoRA")

    weights_path = lora_dir / "adapter_model.safetensors"
    try:
        from safetensors import safe_open
    except Exception as error:
        raise PaperRewardsError(f"cannot import safetensors: {error}") from error

    try:
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            lora_a_keys = [key for key in keys if key.endswith(".lora_A.weight")]
            lora_b_keys = [key for key in keys if key.endswith(".lora_B.weight")]
            unsupported = [
                key
                for key in keys
                if key not in set(lora_a_keys) and key not in set(lora_b_keys)
            ]
            nonzero_b_keys = [
                key
                for key in lora_b_keys
                if bool(handle.get_tensor(key).count_nonzero().item())
            ]
    except Exception as error:
        raise PaperRewardsError(f"cannot inspect zero-delta LoRA: {weights_path}: {error}") from error

    if not lora_a_keys or len(lora_a_keys) != len(lora_b_keys):
        raise PaperRewardsError("zero-delta LoRA must contain paired A/B tensors")
    if unsupported:
        raise PaperRewardsError(
            f"zero-delta LoRA contains unsupported tensors: {unsupported[:3]}"
        )
    if nonzero_b_keys:
        raise PaperRewardsError(
            f"zero-delta LoRA has nonzero B tensors: {nonzero_b_keys[:3]}"
        )
    return {
        "verified": True,
        "lora_a_tensors": len(lora_a_keys),
        "lora_b_tensors": len(lora_b_keys),
        "nonzero_lora_b_tensors": 0,
        "effective_delta": "exactly_zero",
    }


def build_coordinates(manifest: Mapping[str, Any], checkpoint_root: Path) -> list[Coordinate]:
    """Expand the legacy fixed-checkpoint layout for compatibility tests."""

    rows = validate_manifest(manifest)
    guidance_scales = manifest["evaluation"]["guidance_scales"]
    return [
        Coordinate(
            family=str(family["key"]),
            method=str(method["key"]),
            run_id=str(method["run_id"]),
            guidance_scale=float(guidance_scale),
            checkpoint=(
                checkpoint_root
                / "checkpoints"
                / str(family["key"])
                / str(method["key"])
                / "checkpoint-100"
            ),
            checkpoint_lora_subdir=_checkpoint_lora_subdir(method),
        )
        for family, method in rows
        for guidance_scale in guidance_scales
    ]


def validate_base_model_input(
    manifest: Mapping[str, Any], base_model: Path | None
) -> dict[str, Any]:
    """Validate the pinned local base model used by an adapter-free policy."""

    if base_model is None or not base_model.is_absolute():
        raise PaperRewardsError("base-model policy requires an absolute local model path")
    base_spec = _mapping(manifest.get("base_model"), "base model")
    repo_id = base_spec.get("repo_id")
    revision = base_spec.get("revision")
    model_index_sha256 = base_spec.get("model_index_sha256")
    if not isinstance(repo_id, str) or not repo_id:
        raise PaperRewardsError("base-model policy requires a repository ID")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise PaperRewardsError("base-model policy requires a pinned commit revision")
    if (
        not isinstance(model_index_sha256, str)
        or len(model_index_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in model_index_sha256
        )
    ):
        raise PaperRewardsError("base-model policy requires model_index_sha256")
    try:
        resolved_model = base_model.resolve(strict=True)
    except FileNotFoundError as error:
        raise PaperRewardsError(f"missing local base model: {base_model}") from error
    if not resolved_model.is_dir() or resolved_model.name != revision:
        raise PaperRewardsError(
            f"local base model does not match revision {revision}: {resolved_model}"
        )
    revision_marker = resolved_model / ".diffusion_nft_revision"
    model_index = resolved_model / "model_index.json"
    for path in (revision_marker, model_index):
        if not path.is_file() or path.stat().st_size <= 0:
            raise PaperRewardsError(f"missing local base-model identity file: {path}")
    if revision_marker.read_text(encoding="utf-8").strip() != revision:
        raise PaperRewardsError("local base-model revision marker mismatch")
    if sha256_file(model_index) != model_index_sha256:
        raise PaperRewardsError("local base-model model_index.json hash mismatch")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "local_path": str(resolved_model),
        "model_index_sha256": model_index_sha256,
        "revision_marker_sha256": sha256_file(revision_marker),
    }


def validate_direct_checkpoint_matrix(
    manifest: Mapping[str, Any], checkpoint_root: Path, *, base_model: Path | None = None
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate checkpoint or adapter-free base policies and hash their identities."""

    rows = validate_manifest(manifest)
    has_checkpoint_policy = any(
        _source_type(method) == "checkpoint" for _family, method in rows
    )
    has_base_model_policy = any(
        _source_type(method) == "base_model" for _family, method in rows
    )
    resolved_root: Path | None = None
    if has_checkpoint_policy:
        declared_root_value = manifest.get("checkpoint_root")
        if isinstance(declared_root_value, str):
            declared_root_value = os.path.expandvars(declared_root_value)
        if (
            not isinstance(declared_root_value, str)
            or not Path(declared_root_value).is_absolute()
        ):
            raise PaperRewardsError("manifest checkpoint_root must be absolute")
        try:
            resolved_root = checkpoint_root.resolve(strict=True)
            declared_root = Path(declared_root_value).resolve(strict=True)
        except FileNotFoundError as error:
            raise PaperRewardsError(
                f"missing direct checkpoint root: {error.filename}"
            ) from error
        if resolved_root != declared_root or not resolved_root.is_dir():
            raise PaperRewardsError(
                "direct checkpoint root differs from manifest: "
                f"{resolved_root} != {declared_root}"
            )
    base_model_inventory = (
        validate_base_model_input(manifest, base_model)
        if has_base_model_policy
        else None
    )

    try:
        from baselines.diffusion_nft_sd35_v5.tools.validate_v5flow_checkpoint import (
            load_and_validate_resumable,
        )
    except Exception as error:
        raise PaperRewardsError(f"cannot import checkpoint validator: {error}") from error

    inventories: dict[tuple[str, str], dict[str, Any]] = {}
    for family, method in rows:
        family_key = str(family["key"])
        method_key = str(method["key"])
        run_id = method.get("run_id")
        if (
            not isinstance(run_id, str)
            or not run_id
            or Path(run_id).name != run_id
            or "/" in run_id
        ):
            raise PaperRewardsError(f"unsafe direct run ID: {run_id!r}")
        source_type = _source_type(method)
        if source_type == "base_model":
            if base_model_inventory is None:
                raise PaperRewardsError("base-model inventory was not initialized")
            identity = {
                "family": family_key,
                "method": method_key,
                "run_id": run_id,
                "source_type": source_type,
                "checkpoint_step": None,
                "checkpoint_storage": None,
                "checkpoint_lora_subdir": None,
                "checkpoint_method": "base_model",
                "cfg_mode": None,
                "no_adapter": True,
                "base_model": base_model_inventory,
                "files": [],
            }
            inventories[(family_key, method_key)] = {
                **identity,
                "identity_sha256": hashlib.sha256(
                    _canonical_json(identity)
                ).hexdigest(),
            }
            continue
        if resolved_root is None:
            raise PaperRewardsError("checkpoint root was not initialized")
        checkpoint_step = method.get("checkpoint_step", manifest.get("checkpoint_step"))
        if (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step < 0
        ):
            raise PaperRewardsError(f"invalid checkpoint step for {run_id}")
        checkpoint_storage = method.get("checkpoint_storage", "checkpoints")
        if checkpoint_storage not in {"checkpoints", "rolling"}:
            raise PaperRewardsError(f"invalid checkpoint storage for {run_id}")
        run_dir = resolved_root / run_id
        config_path = run_dir / "provenance/config.json"
        checkpoint = (
            run_dir / "train" / checkpoint_storage / f"checkpoint-{checkpoint_step}"
        )
        checkpoint_lora_subdir = _checkpoint_lora_subdir(method)
        lora_dir = checkpoint / checkpoint_lora_subdir
        files = (
            config_path,
            checkpoint / ".checkpoint_complete.json",
            lora_dir / "adapter_config.json",
            lora_dir / "adapter_model.safetensors",
        )
        for path in files:
            try:
                resolved_path = path.resolve(strict=True)
            except FileNotFoundError as error:
                raise PaperRewardsError(f"missing direct checkpoint input: {path}") from error
            if (
                not resolved_path.is_file()
                or path.is_symlink()
                or not resolved_path.is_relative_to(resolved_root)
                or resolved_path.stat().st_size <= 0
            ):
                raise PaperRewardsError(f"invalid direct checkpoint input: {path}")
        if sha256_file(config_path) != method.get("config_sha256"):
            raise PaperRewardsError(f"training config hash mismatch for {run_id}")
        expected_adapter_sha256 = method.get("adapter_model_sha256")
        if expected_adapter_sha256 is not None:
            if (
                not isinstance(expected_adapter_sha256, str)
                or len(expected_adapter_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_adapter_sha256
                )
            ):
                raise PaperRewardsError(
                    f"invalid adapter model hash declaration for {run_id}"
                )
            if (
                sha256_file(lora_dir / "adapter_model.safetensors")
                != expected_adapter_sha256
            ):
                raise PaperRewardsError(f"adapter model hash mismatch for {run_id}")
        marker = _mapping(
            _load_json(checkpoint / ".checkpoint_complete.json", "checkpoint marker"),
            "checkpoint marker",
        )
        expected_marker = {
            "checkpoint_step": checkpoint_step,
            "resume_epoch": method.get("resume_epoch"),
            "checkpoint_kind": method.get("checkpoint_kind"),
            "cfg_mode": method.get("cfg_mode"),
        }
        if any(marker.get(key) != value for key, value in expected_marker.items()):
            raise PaperRewardsError(f"checkpoint marker mismatch for {run_id}")
        try:
            state, _summary = load_and_validate_resumable(
                checkpoint,
                require_complete_marker=True,
                expect_v5flow=method.get("method") == "v5flow",
                expect_method=str(method.get("method")),
                expect_cfg_mode=str(method.get("cfg_mode")),
            )
            del state
        except Exception as error:
            raise PaperRewardsError(
                f"checkpoint resume-state validation failed for {run_id}: {error}"
            ) from error
        inventory_files = [
            {
                "path": path.relative_to(resolved_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
        zero_delta_lora = (
            validate_zero_delta_lora(lora_dir)
            if method.get("require_zero_delta_lora") is True
            else None
        )
        if method.get("require_zero_delta_lora") not in {None, False, True}:
            raise PaperRewardsError(
                f"require_zero_delta_lora must be boolean for {run_id}"
            )
        identity = {
            "family": family_key,
            "method": method_key,
            "run_id": run_id,
            "source_type": source_type,
            "checkpoint_step": checkpoint_step,
            "checkpoint_storage": checkpoint_storage,
            "checkpoint_lora_subdir": checkpoint_lora_subdir,
            "checkpoint_method": method.get("method"),
            "cfg_mode": method.get("cfg_mode"),
            "zero_delta_lora": zero_delta_lora,
            "files": inventory_files,
        }
        inventories[(family_key, method_key)] = {
            **identity,
            "identity_sha256": hashlib.sha256(_canonical_json(identity)).hexdigest(),
        }
    return inventories


def build_direct_coordinates(
    manifest: Mapping[str, Any],
    checkpoint_root: Path,
    inventories: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Coordinate]:
    """Expand validated checkpoint/base rows into one coordinate per requested CFG."""

    rows = validate_manifest(manifest)
    guidance_scales = manifest["evaluation"]["guidance_scales"]
    coordinates: list[Coordinate] = []
    for family, method in rows:
        family_key = str(family["key"])
        method_key = str(method["key"])
        inventory = inventories[(family_key, method_key)]
        source_type = str(inventory.get("source_type", _source_type(method)))
        if source_type == "base_model":
            checkpoint = None
            checkpoint_lora_subdir = None
            checkpoint_step = None
        else:
            checkpoint_step = int(inventory["checkpoint_step"])
            checkpoint = (
                checkpoint_root
                / str(method["run_id"])
                / "train"
                / str(method.get("checkpoint_storage", "checkpoints"))
                / f"checkpoint-{checkpoint_step}"
            )
            checkpoint_lora_subdir = str(
                inventory.get(
                    "checkpoint_lora_subdir", _checkpoint_lora_subdir(method)
                )
            )
        for guidance_scale in guidance_scales:
            coordinates.append(
                Coordinate(
                    family=family_key,
                    method=method_key,
                    run_id=str(method["run_id"]),
                    guidance_scale=float(guidance_scale),
                    checkpoint=checkpoint,
                    checkpoint_lora_subdir=checkpoint_lora_subdir,
                    checkpoint_step=checkpoint_step,
                    checkpoint_identity=str(inventory["identity_sha256"]),
                    source_type=source_type,
                )
            )
    return coordinates


def _run(command: Sequence[str], *, cwd: Path) -> None:
    print(f"RUN {json.dumps(list(command))}", flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise PaperRewardsError(
            f"subprocess exited with {completed.returncode}: {command[0]}"
        )


def _validate_generation(
    output_dir: Path,
    *,
    base_model: Path,
    coordinate: Coordinate,
    sample_count: int,
    full_dataset_samples: int,
    batch_size: int,
    seed: int,
    eval_steps: int,
    resolution: int,
    world_size: int = 1,
) -> dict[str, float]:
    results_path = output_dir / "evaluation_results.jsonl"
    averages_path = output_dir / "average_scores.json"
    metadata_path = output_dir / "evaluation_metadata.json"
    images_dir = output_dir / "images"
    records = _load_jsonl(results_path, "generation result")
    if len(records) != sample_count or [item.get("sample_id") for item in records] != list(
        range(sample_count)
    ):
        raise PaperRewardsError(f"generation coverage mismatch: {output_dir}")
    for sample_id, record in enumerate(records):
        scores = _mapping(record.get("scores"), "generation scores")
        for key in ("pickscore", "pickscore_normalized"):
            value = scores.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise PaperRewardsError(f"invalid generation {key} at sample {sample_id}")
        image_path = images_dir / f"{sample_id:05d}.jpg"
        if not image_path.is_file() or Path(str(record.get("image_path"))).resolve() != image_path.resolve():
            raise PaperRewardsError(f"generation image mismatch at sample {sample_id}")
    actual_images = {path.name for path in images_dir.glob("*.jpg") if path.is_file()}
    expected_images = {f"{sample_id:05d}.jpg" for sample_id in range(sample_count)}
    if actual_images != expected_images:
        raise PaperRewardsError(f"generation JPEG inventory mismatch: {output_dir}")
    metadata = _mapping(_load_json(metadata_path, "generation metadata"), "metadata")
    checkpoint_path = (
        str(coordinate.checkpoint.resolve())
        if coordinate.checkpoint is not None
        else None
    )
    expected_metadata = {
        "base_model": str(base_model.resolve()),
        "checkpoint_path": checkpoint_path,
        "checkpoint_lora_subdir": coordinate.checkpoint_lora_subdir,
        "local_files_only": True,
        "seed": seed,
        "dataset": "drawbench",
        "reward_names": ["pickscore"],
        "batch_size_per_rank": batch_size,
        "num_inference_steps": eval_steps,
        "guidance_scale": coordinate.guidance_scale,
        "resolution": resolution,
        "mixed_precision": "fp16",
        "world_size": world_size,
        "full_dataset_size": full_dataset_samples,
        "max_samples": None if sample_count == full_dataset_samples else sample_count,
        "expected_sample_count": sample_count,
        "sampler": "strided_without_padding",
    }
    if coordinate.source_type == "base_model":
        expected_metadata["lora_hf_path"] = None
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise PaperRewardsError(f"generation metadata mismatch for {key}")
    averages = _mapping(_load_json(averages_path, "generation averages"), "averages")
    result = {}
    for key in ("pickscore", "pickscore_normalized"):
        value = averages.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PaperRewardsError(f"invalid generation average: {key}")
        result[key] = float(value)
    return result


def _generation_image_manifest(output_dir: Path, sample_count: int) -> str:
    images_dir = output_dir / "images"
    return "".join(
        f"{sha256_file(images_dir / f'{sample_id:05d}.jpg')}  "
        f"images/{sample_id:05d}.jpg\n"
        for sample_id in range(sample_count)
    )


def run_generation(
    coordinate: Coordinate,
    output_dir: Path,
    *,
    evaluator: Path,
    upstream_root: Path,
    base_model: Path,
    sample_count: int,
    full_dataset_samples: int,
    batch_size: int,
    seed: int,
    eval_steps: int,
    resolution: int,
    source_id: str,
    world_size: int = 1,
    evaluator_extra_args: Sequence[str] = (),
) -> dict[str, float]:
    evaluator_extra_args = tuple(evaluator_extra_args)
    if any(not isinstance(value, str) or not value for value in evaluator_extra_args):
        raise PaperRewardsError("evaluator extra arguments must be nonempty strings")
    if coordinate.source_type == "base_model":
        if (
            coordinate.checkpoint is not None
            or coordinate.checkpoint_lora_subdir is not None
            or coordinate.checkpoint_step is not None
        ):
            raise PaperRewardsError(
                "base-model coordinate must not select a checkpoint or adapter"
            )
    elif coordinate.source_type == "checkpoint":
        if coordinate.checkpoint is None or coordinate.checkpoint_lora_subdir is None:
            raise PaperRewardsError("checkpoint coordinate is missing its adapter")
    else:
        raise PaperRewardsError(
            f"unsupported coordinate source type: {coordinate.source_type!r}"
        )
    request = {
        "schema_version": 1,
        "family": coordinate.family,
        "method": coordinate.method,
        "run_id": coordinate.run_id,
        "guidance_scale": coordinate.guidance_scale,
        "source_type": coordinate.source_type,
        "checkpoint": (
            str(coordinate.checkpoint.resolve())
            if coordinate.checkpoint is not None
            else None
        ),
        "checkpoint_lora_subdir": coordinate.checkpoint_lora_subdir,
        "base_model": str(base_model.resolve()),
        "sample_count": sample_count,
        "full_dataset_samples": full_dataset_samples,
        "batch_size_per_rank": batch_size,
        "seed": seed,
        "num_inference_steps": eval_steps,
        "resolution": resolution,
        "source_id": source_id,
        "world_size": world_size,
        "checkpoint_step": coordinate.checkpoint_step,
        "checkpoint_identity": coordinate.checkpoint_identity,
    }
    if evaluator_extra_args:
        request["evaluator_extra_args"] = list(evaluator_extra_args)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / ".request.json"
    if request_path.exists():
        if _load_json(request_path, "generation request") != request:
            raise PaperRewardsError(f"generation output belongs to another request: {output_dir}")
    else:
        if any(output_dir.iterdir()):
            raise PaperRewardsError(f"nonempty generation output has no request: {output_dir}")
        _atomic_write(request_path, request)

    complete_path = output_dir / ".complete"
    if not complete_path.is_file():
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={world_size}",
            str(evaluator),
            *evaluator_extra_args,
            "--base_model",
            str(base_model),
            "--local_files_only",
            "--seed",
            str(seed),
            "--model_type",
            "sd3",
            "--dataset",
            "drawbench",
            "--reward",
            "pickscore",
            "--output_dir",
            str(output_dir),
            "--num_inference_steps",
            str(eval_steps),
            "--guidance_scale",
            str(coordinate.guidance_scale),
            "--resolution",
            str(resolution),
            "--mixed_precision",
            "fp16",
            "--batch_size",
            str(batch_size),
            "--save_images",
        ]
        if coordinate.checkpoint is not None:
            command.extend(
                [
                    "--checkpoint_path",
                    str(coordinate.checkpoint),
                    "--checkpoint_lora_subdir",
                    str(coordinate.checkpoint_lora_subdir),
                ]
            )
        if sample_count != full_dataset_samples:
            command.extend(["--max_samples", str(sample_count)])
        _run(command, cwd=upstream_root)
        averages = _validate_generation(
            output_dir,
            base_model=base_model,
            coordinate=coordinate,
            sample_count=sample_count,
            full_dataset_samples=full_dataset_samples,
            batch_size=batch_size,
            seed=seed,
            eval_steps=eval_steps,
            resolution=resolution,
            world_size=world_size,
        )
        image_manifest_path = output_dir / "images.sha256"
        _atomic_write_text(
            image_manifest_path,
            _generation_image_manifest(output_dir, sample_count),
        )
        _atomic_write(
            complete_path,
            {
                "schema_version": 1,
                "records": sample_count,
                "request_sha256": sha256_file(request_path),
                "evaluation_results_sha256": sha256_file(
                    output_dir / "evaluation_results.jsonl"
                ),
                "average_scores_sha256": sha256_file(output_dir / "average_scores.json"),
                "evaluation_metadata_sha256": sha256_file(
                    output_dir / "evaluation_metadata.json"
                ),
                "images_manifest_sha256": sha256_file(image_manifest_path),
            },
        )
        return averages
    complete = _mapping(_load_json(complete_path, "generation completion marker"), "complete")
    paths = {
        "request_sha256": request_path,
        "evaluation_results_sha256": output_dir / "evaluation_results.jsonl",
        "average_scores_sha256": output_dir / "average_scores.json",
        "evaluation_metadata_sha256": output_dir / "evaluation_metadata.json",
        "images_manifest_sha256": output_dir / "images.sha256",
    }
    if complete.get("records") != sample_count or any(
        complete.get(key) != sha256_file(path) for key, path in paths.items()
    ):
        raise PaperRewardsError(f"generation completion hash mismatch: {output_dir}")
    expected_image_manifest = _generation_image_manifest(output_dir, sample_count)
    if (output_dir / "images.sha256").read_text(encoding="utf-8") != expected_image_manifest:
        raise PaperRewardsError(f"generation image hashes changed: {output_dir}")
    return _validate_generation(
        output_dir,
        base_model=base_model,
        coordinate=coordinate,
        sample_count=sample_count,
        full_dataset_samples=full_dataset_samples,
        batch_size=batch_size,
        seed=seed,
        eval_steps=eval_steps,
        resolution=resolution,
        world_size=world_size,
    )


def run_rescore(
    coordinate: Coordinate,
    generation_dir: Path,
    output_dir: Path,
    reward: str,
    *,
    rescorer: Path,
    upstream_root: Path,
    dataset_file: Path,
    base_model: Path,
    sample_count: int,
    full_dataset_samples: int,
    generation_batch_size: int,
    batch_size: int,
    seed: int,
    eval_steps: int,
    resolution: int,
    source_id: str,
    world_size: int = 1,
) -> dict[str, float]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={world_size}",
        str(rescorer),
        "--source-dir",
        str(generation_dir),
        "--output-dir",
        str(output_dir),
        "--dataset-file",
        str(dataset_file),
        "--reward",
        reward,
        "--expected-base-model",
        str(base_model),
        "--full-dataset-samples",
        str(full_dataset_samples),
        "--expected-samples",
        str(sample_count),
        "--expected-seed",
        str(seed),
        "--expected-eval-steps",
        str(eval_steps),
        "--expected-guidance-scale",
        str(coordinate.guidance_scale),
        "--expected-resolution",
        str(resolution),
        "--expected-generation-batch-size",
        str(generation_batch_size),
        "--batch-size",
        str(batch_size),
        "--expected-world-size",
        str(world_size),
        "--source-snapshot-sha256",
        source_id,
    ]
    if coordinate.source_type == "base_model":
        command.append("--expect-no-adapter")
    elif coordinate.checkpoint is not None:
        command.extend(
            [
                "--expected-checkpoint",
                str(coordinate.checkpoint),
                "--expected-checkpoint-lora-subdir",
                str(coordinate.checkpoint_lora_subdir),
            ]
        )
    else:
        raise PaperRewardsError("checkpoint rescore is missing its checkpoint")
    _run(command, cwd=upstream_root)
    averages = _mapping(_load_json(output_dir / "average_scores.json", "rescore averages"), "averages")
    value = averages.get(reward)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PaperRewardsError(f"invalid {reward} average: {output_dir}")
    return {reward: float(value)}


def run_coordinate(
    coordinate: Coordinate,
    coordinate_root: Path,
    *,
    evaluator: Path,
    rescorer: Path,
    upstream_root: Path,
    dataset_file: Path,
    base_model: Path,
    evaluation: Mapping[str, Any],
    sample_count: int,
    source_id: str,
) -> dict[str, Any]:
    generation_dir = coordinate_root / "generation"
    scores = run_generation(
        coordinate,
        generation_dir,
        evaluator=evaluator,
        upstream_root=upstream_root,
        base_model=base_model,
        sample_count=sample_count,
        full_dataset_samples=int(evaluation["expected_samples"]),
        batch_size=int(evaluation["generation_batch_size_per_rank"]),
        seed=int(evaluation["seed"]),
        eval_steps=int(evaluation["num_inference_steps"]),
        resolution=int(evaluation["resolution"]),
        source_id=source_id,
        world_size=int(evaluation["world_size"]),
    )
    for reward in RESCORE_REWARDS:
        scores.update(
            run_rescore(
                coordinate,
                generation_dir,
                coordinate_root / "scores" / reward,
                reward,
                rescorer=rescorer,
                upstream_root=upstream_root,
                dataset_file=dataset_file,
                base_model=base_model,
                sample_count=sample_count,
                full_dataset_samples=int(evaluation["expected_samples"]),
                generation_batch_size=int(evaluation["generation_batch_size_per_rank"]),
                batch_size=int(evaluation["rescore_batch_size_per_rank"][reward]),
                seed=int(evaluation["seed"]),
                eval_steps=int(evaluation["num_inference_steps"]),
                resolution=int(evaluation["resolution"]),
                source_id=source_id,
                world_size=int(evaluation["world_size"]),
            )
        )
    result = {
        "family": coordinate.family,
        "method": coordinate.method,
        "run_id": coordinate.run_id,
        "source_type": coordinate.source_type,
        "checkpoint_step": coordinate.checkpoint_step,
        "checkpoint_lora_subdir": coordinate.checkpoint_lora_subdir,
        "guidance_scale": coordinate.guidance_scale,
        "samples": sample_count,
        "scores": scores,
        "generation_dir": str(generation_dir),
    }
    _atomic_write(coordinate_root / ".coordinate-complete.json", result)
    return result


def render_score_tables(results: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Render one stable row per policy/CFG coordinate as CSV and Markdown."""

    columns = (
        "family",
        "method",
        "run_id",
        "checkpoint_step",
        "guidance_scale",
        "samples",
        *PAPER_REWARDS,
    )
    rows: list[dict[str, Any]] = []
    for result in results:
        scores = _mapping(result.get("scores"), "coordinate scores")
        row = {name: result.get(name) for name in columns[:6]}
        for reward in PAPER_REWARDS:
            value = scores.get(reward)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise PaperRewardsError(
                    f"cannot render nonfinite {reward} for {result.get('run_id')}"
                )
            row[reward] = float(value)
        rows.append(row)

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    markdown_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            cells.append(f"{value:.8f}" if column in PAPER_REWARDS else str(value))
        markdown_lines.append("| " + " | ".join(cells) + " |")
    return csv_buffer.getvalue(), "\n".join(markdown_lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direct-checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(argv)

    if args.check_only and args.smoke_only:
        raise PaperRewardsError("--check-only and --smoke-only are mutually exclusive")

    for path, label in (
        (args.manifest, "manifest"),
        (args.run_root, "run root"),
        (args.base_model, "base model"),
    ):
        if not path.is_absolute():
            raise PaperRewardsError(f"{label} must be absolute: {path}")
    if len(args.source_id) != 64 or any(char not in "0123456789abcdef" for char in args.source_id):
        raise PaperRewardsError("source ID must be lowercase SHA-256")
    manifest = _mapping(_load_json(args.manifest, "evaluation manifest"), "manifest")
    rows = validate_manifest(manifest)
    manifest_sha256 = sha256_file(args.manifest)
    if not args.direct_checkpoint_root.is_absolute():
        raise PaperRewardsError(
            f"direct checkpoint root must be absolute: {args.direct_checkpoint_root}"
        )
    direct_inventories = validate_direct_checkpoint_matrix(
        manifest, args.direct_checkpoint_root, base_model=args.base_model
    )
    coordinates = build_direct_coordinates(
        manifest, args.direct_checkpoint_root, direct_inventories
    )
    source_types = {
        str(inventory.get("source_type", "checkpoint"))
        for inventory in direct_inventories.values()
    }
    checkpoint_source_payload: dict[str, Any] = {
        "kind": (
            "base_model_only"
            if source_types == {"base_model"}
            else (
                "direct_shared_filesystem"
                if source_types == {"checkpoint"}
                else "mixed_base_model_and_direct_checkpoints"
            )
        ),
        "policies": [
            direct_inventories[(str(family["key"]), str(method["key"]))]
            for family, method in rows
        ],
    }
    if "checkpoint" in source_types:
        checkpoint_source_payload["root"] = str(
            args.direct_checkpoint_root.resolve()
        )
    evaluation = _mapping(manifest["evaluation"], "evaluation")

    repo_root = Path(__file__).resolve().parents[3]
    upstream_root = repo_root / "baselines/diffusion_nft/upstream"
    evaluator = upstream_root / "scripts/evaluation.py"
    rescorer = Path(__file__).resolve().parent / "rescore_saved_drawbench.py"
    dataset_file = upstream_root / "dataset/drawbench/test.txt"
    for path in (evaluator, rescorer, dataset_file, args.base_model):
        if not path.exists():
            raise PaperRewardsError(f"missing runtime input: {path}")
    if args.check_only:
        print(
            _json_text(
                {
                    "status": "ready",
                    "policies": len(rows),
                    "coordinates": len(coordinates),
                    "world_size": evaluation["world_size"],
                    "checkpoint_source": checkpoint_source_payload["kind"],
                }
            ),
            end="",
        )
        return 0

    args.run_root.mkdir(parents=True, exist_ok=True)
    complete_path = args.run_root / ".complete"
    summary_path = args.run_root / "summary.json"
    summary_csv_path = args.run_root / "summary.csv"
    summary_markdown_path = args.run_root / "summary.md"
    if complete_path.is_file():
        complete = _mapping(_load_json(complete_path, "run completion marker"), "complete")
        if (
            complete.get("run_name") != args.run_name
            or complete.get("source_id") != args.source_id
            or not all(
                path.is_file()
                for path in (summary_path, summary_csv_path, summary_markdown_path)
            )
            or complete.get("summary_sha256") != sha256_file(summary_path)
            or complete.get("summary_csv_sha256") != sha256_file(summary_csv_path)
            or complete.get("summary_markdown_sha256")
            != sha256_file(summary_markdown_path)
        ):
            raise PaperRewardsError("existing run completion marker is inconsistent")
        print(f"Reusing completed paper-reward matrix: {summary_path}", flush=True)
        return 0

    copied_manifest = args.run_root / "evaluation-manifest.json"
    if copied_manifest.exists():
        if sha256_file(copied_manifest) != manifest_sha256:
            raise PaperRewardsError("run root contains a different evaluation manifest")
    else:
        shutil.copy2(args.manifest, copied_manifest)
    checkpoint_source_path = args.run_root / "checkpoint-source.json"
    if checkpoint_source_path.exists():
        if _load_json(checkpoint_source_path, "checkpoint source") != checkpoint_source_payload:
            raise PaperRewardsError("run root contains a different checkpoint source")
    else:
        _atomic_write(checkpoint_source_path, checkpoint_source_payload)

    smoke_summary_path = args.run_root / "smoke-summary.json"
    smoke_complete_path = args.run_root / ".smoke-complete"
    if args.smoke_only and smoke_complete_path.is_file():
        smoke_complete = _mapping(
            _load_json(smoke_complete_path, "smoke completion marker"),
            "smoke complete",
        )
        if (
            smoke_complete.get("run_name") != args.run_name
            or smoke_complete.get("source_id") != args.source_id
            or smoke_complete.get("evaluation_manifest_sha256") != manifest_sha256
            or not smoke_summary_path.is_file()
            or smoke_complete.get("summary_sha256") != sha256_file(smoke_summary_path)
        ):
            raise PaperRewardsError("existing smoke completion marker is inconsistent")
        print(f"Reusing completed nonformal smoke: {smoke_summary_path}", flush=True)
        return 0

    smoke_coordinate = next(
        (
            item
            for item in coordinates
            if item.family == "learned3"
            and item.method == "fm100"
            and item.guidance_scale == 4.0
        ),
        None,
    )
    if smoke_coordinate is None:
        smoke_coordinate = next(
            item for item in reversed(coordinates) if item.guidance_scale == 4.0
        )
    smoke_result = run_coordinate(
        smoke_coordinate,
        args.run_root
        / "smoke"
        / smoke_coordinate.family
        / smoke_coordinate.method
        / smoke_coordinate.guidance_tag,
        evaluator=evaluator,
        rescorer=rescorer,
        upstream_root=upstream_root,
        dataset_file=dataset_file,
        base_model=args.base_model,
        evaluation=evaluation,
        sample_count=int(evaluation["smoke_samples"]),
        source_id=args.source_id,
    )

    if args.smoke_only:
        smoke_summary = {
            "schema_version": 1,
            "run_name": args.run_name,
            "source_id": args.source_id,
            "evaluation_manifest_sha256": manifest_sha256,
            "checkpoint_source": checkpoint_source_payload,
            "purpose": "nonformal_capacity_smoke",
            "reported_rewards": list(PAPER_REWARDS),
            "result": smoke_result,
        }
        _atomic_write(smoke_summary_path, smoke_summary)
        _atomic_write(
            smoke_complete_path,
            {
                "schema_version": 1,
                "run_name": args.run_name,
                "source_id": args.source_id,
                "evaluation_manifest_sha256": manifest_sha256,
                "summary_sha256": sha256_file(smoke_summary_path),
            },
        )
        print(f"Nonformal smoke completed: {smoke_summary_path}", flush=True)
        return 0

    results = []
    for coordinate in coordinates:
        results.append(
            run_coordinate(
                coordinate,
                args.run_root
                / "formal"
                / coordinate.family
                / coordinate.method
                / coordinate.guidance_tag,
                evaluator=evaluator,
                rescorer=rescorer,
                upstream_root=upstream_root,
                dataset_file=dataset_file,
                base_model=args.base_model,
                evaluation=evaluation,
                sample_count=int(evaluation["expected_samples"]),
                source_id=args.source_id,
            )
        )
    summary = {
        "schema_version": 1,
        "run_name": args.run_name,
        "source_id": args.source_id,
        "evaluation_manifest_sha256": manifest_sha256,
        "checkpoint_source": checkpoint_source_payload,
        "policy_checkpoints": [
            {
                "family": str(family["key"]),
                "method": str(method["key"]),
                "run_id": str(method["run_id"]),
                "source_type": direct_inventories[
                    (str(family["key"]), str(method["key"]))
                ].get("source_type", _source_type(method)),
                "checkpoint_step": direct_inventories[
                    (str(family["key"]), str(method["key"]))
                ]["checkpoint_step"],
                "checkpoint_storage": direct_inventories[
                    (str(family["key"]), str(method["key"]))
                ]["checkpoint_storage"],
                "checkpoint_lora_subdir": direct_inventories[
                    (str(family["key"]), str(method["key"]))
                ].get(
                    "checkpoint_lora_subdir",
                    (
                        None
                        if _source_type(method) == "base_model"
                        else _checkpoint_lora_subdir(method)
                    ),
                ),
            }
            for family, method in rows
        ],
        "dataset": "drawbench",
        "seed": 42,
        "num_inference_steps": 40,
        "samples_per_coordinate": 1000,
        "coordinate_count": len(results),
        "reported_rewards": list(PAPER_REWARDS),
        "unavailable_rewards": dict(evaluation["unavailable_rewards"]),
        "smoke_gate": smoke_result,
        "results": results,
    }
    _atomic_write(summary_path, summary)
    summary_csv, summary_markdown = render_score_tables(results)
    _atomic_write_text(summary_csv_path, summary_csv)
    _atomic_write_text(summary_markdown_path, summary_markdown)
    _atomic_write(
        complete_path,
        {
            "schema_version": 1,
            "run_name": args.run_name,
            "source_id": args.source_id,
            "coordinates": len(results),
            "summary_sha256": sha256_file(summary_path),
            "summary_csv_sha256": sha256_file(summary_csv_path),
            "summary_markdown_sha256": sha256_file(summary_markdown_path),
        },
    )
    print(f"Paper-reward matrix completed: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PaperRewardsError as error:
        print(f"paper-reward runtime contract error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
