#!/usr/bin/env python
"""Validate and optionally compare SD3.5 v5flow extension checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


STATE_FILENAME = "v5flow_state.pt"
INITIALIZER_STATE_FILENAME = "v5flow_initializer.pt"
INITIALIZER_MARKER_FILENAME = ".v5flow_initializer_complete.json"
POLICY_STATE_FILENAME = "policy_state.pt"
COMPLETE_MARKER_FILENAME = ".checkpoint_complete.json"
REQUIRED_KEYS = {
    "schema_version",
    "adapter_default_raw",
    "adapter_old",
    "adapter_v5_raw",
    "optimizer_v5",
    "v5_scaler",
    "v5_ema",
    "theta_ema",
    "v5_optimization_step",
}
INITIALIZER_REQUIRED_KEYS = {
    "schema_version",
    "source_adapter_default",
    "adapter_v5_raw",
    "optimizer_v5",
    "v5_scaler",
    "v5_ema",
    "v5_optimization_step",
    "metadata",
}
POLICY_REQUIRED_KEYS = {
    "schema_version",
    "adapter_default_raw",
    "adapter_old",
    "theta_ema",
}
EXACT_RANK_KEYS = {
    "schema_version",
    "rank",
    "world_size",
    "checkpoint_step",
    "resume_epoch",
    "resume_phase",
    "next_sampling_batch",
    "rng_state",
    "payload",
}
EXACT_RESUME_PHASES = {"sampling", "stage1", "stage2", "outer_boundary"}
CFG_MODES = {
    "cfg1": 1.0,
    # Paired conditional/empty objective, shared by historical and current runs.
    "cfg1_dual": 1.0,
    "cfg1_dropout": 1.0,
    "cfg4_rollout": 4.0,
    "cfg4_shifted_center": 4.0,
    "cfg4_dual_shifted_center": 4.0,
    "cfg4_full": 4.0,
    "cfg4_alternating": 4.0,
}
CFG_MARKER_KEYS = {"cfg_mode", "rollout_guidance_scale"}


def _validate_marker_cfg_contract(marker: dict[str, Any] | None) -> tuple[str, float]:
    """Validate optional CFG metadata, treating legacy markers as CFG1."""

    if marker is None or not (set(marker) & CFG_MARKER_KEYS):
        return "cfg1", 1.0
    if not CFG_MARKER_KEYS <= set(marker):
        raise ValueError("Completion marker has a partial CFG contract")
    cfg_mode = marker["cfg_mode"]
    if not isinstance(cfg_mode, str) or cfg_mode not in CFG_MODES:
        raise ValueError(f"Completion marker has invalid cfg_mode: {cfg_mode!r}")
    guidance_scale = marker["rollout_guidance_scale"]
    if (
        isinstance(guidance_scale, bool)
        or not isinstance(guidance_scale, (int, float))
        or float(guidance_scale) != CFG_MODES[cfg_mode]
    ):
        raise ValueError(
            "Completion marker rollout guidance disagrees with cfg_mode: "
            f"mode={cfg_mode!r}, scale={guidance_scale!r}"
        )
    return cfg_mode, float(guidance_scale)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_tensors(value: Any, prefix: str = ""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from _iter_tensors(child, child_prefix)


def load_and_validate(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state_path = checkpoint_dir / STATE_FILENAME
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing extension state: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if set(state) != REQUIRED_KEYS:
        raise ValueError(
            f"Unexpected state keys: missing={sorted(REQUIRED_KEYS - set(state))}, "
            f"extra={sorted(set(state) - REQUIRED_KEYS)}"
        )
    if state["schema_version"] != 1:
        raise ValueError(f"Unsupported schema: {state['schema_version']!r}")
    if int(state["v5_optimization_step"]) < 0:
        raise ValueError("v5_optimization_step must be non-negative")

    adapter_keys = None
    adapter_tensor_counts = {}
    for adapter_name in ("adapter_default_raw", "adapter_old", "adapter_v5_raw"):
        adapter_state = state[adapter_name]
        if not adapter_state:
            raise ValueError(f"{adapter_name} is empty")
        if adapter_keys is None:
            adapter_keys = set(adapter_state)
        elif set(adapter_state) != adapter_keys:
            raise ValueError(f"{adapter_name} structure differs from the other adapters")
        adapter_tensor_counts[adapter_name] = len(adapter_state)

    tensor_count = 0
    total_elements = 0
    for name, tensor in _iter_tensors(state):
        tensor_count += 1
        total_elements += tensor.numel()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Non-finite tensor in {name}")

    optimizer_state = state["optimizer_v5"].get("state", {})
    if not optimizer_state and int(state["v5_optimization_step"]) > 0:
        raise ValueError("v5 optimizer has no parameter state")
    summary = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "state_path": str(state_path.resolve()),
        "schema_version": state["schema_version"],
        "v5_optimization_step": int(state["v5_optimization_step"]),
        "adapter_tensor_counts": adapter_tensor_counts,
        "optimizer_parameter_states": len(optimizer_state),
        "has_v5_scaler": state["v5_scaler"] is not None,
        "recursive_tensor_count": tensor_count,
        "recursive_tensor_elements": total_elements,
        "has_v5_ema": state["v5_ema"] is not None,
        "has_theta_ema": state["theta_ema"] is not None,
    }
    return state, summary


def load_and_validate_initializer(
    initializer_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a complete reusable v5-only initializer artifact."""

    initializer_dir = initializer_dir.resolve()
    state_path = initializer_dir / INITIALIZER_STATE_FILENAME
    marker_path = initializer_dir / INITIALIZER_MARKER_FILENAME
    if not state_path.is_file() or state_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing v5flow initializer state: {state_path}")
    if not marker_path.is_file() or marker_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing v5flow initializer marker: {marker_path}")
    with marker_path.open("r", encoding="utf-8") as marker_file:
        marker = json.load(marker_file)
    legacy_marker_keys = {
        "schema_version",
        "state_filename",
        "v5_optimization_step",
        "metadata",
    }
    hashed_marker_keys = legacy_marker_keys | {"state_sha256"}
    if (
        frozenset(marker)
        not in {frozenset(legacy_marker_keys), frozenset(hashed_marker_keys)}
        or marker.get("schema_version") != 1
    ):
        raise ValueError("Unexpected v5flow initializer marker schema")
    if marker["state_filename"] != INITIALIZER_STATE_FILENAME:
        raise ValueError("v5flow initializer marker names the wrong state file")
    if "state_sha256" in marker:
        digest = marker["state_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or _sha256(state_path) != digest
        ):
            raise ValueError("v5flow initializer state SHA256 mismatch")

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or set(state) != INITIALIZER_REQUIRED_KEYS:
        actual_keys = set(state) if isinstance(state, dict) else set()
        raise ValueError(
            "Unexpected v5flow initializer state keys: "
            f"missing={sorted(INITIALIZER_REQUIRED_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - INITIALIZER_REQUIRED_KEYS)}"
        )
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported v5flow initializer state schema")
    optimization_step = state["v5_optimization_step"]
    if (
        not isinstance(optimization_step, int)
        or isinstance(optimization_step, bool)
        or optimization_step <= 0
    ):
        raise ValueError("v5flow initializer optimization step must be positive")
    if (
        marker["v5_optimization_step"] != optimization_step
        or marker["metadata"] != state["metadata"]
        or not isinstance(state["metadata"], dict)
    ):
        raise ValueError("v5flow initializer marker and state disagree")

    source_state = state["source_adapter_default"]
    v5_state = state["adapter_v5_raw"]
    if (
        not isinstance(source_state, dict)
        or not source_state
        or not isinstance(v5_state, dict)
        or source_state.keys() != v5_state.keys()
    ):
        raise ValueError("v5flow initializer adapter structures differ or are empty")
    for key, source_tensor in source_state.items():
        v5_tensor = v5_state[key]
        if (
            not isinstance(source_tensor, torch.Tensor)
            or not isinstance(v5_tensor, torch.Tensor)
            or source_tensor.shape != v5_tensor.shape
        ):
            raise ValueError(f"Invalid v5flow initializer adapter tensor: {key}")
    optimizer_state = state["optimizer_v5"]
    if (
        not isinstance(optimizer_state, dict)
        or not isinstance(optimizer_state.get("param_groups"), list)
        or not optimizer_state.get("state")
    ):
        raise ValueError("v5flow initializer is missing trained optimizer state")
    if state["v5_ema"] is None:
        raise ValueError("v5flow initializer is missing the configured v5 EMA")
    if state["metadata"].get("adapter_storage_dtype") == "float32":
        for adapter_name, adapter_state in (
            ("source", source_state),
            ("v5", v5_state),
        ):
            if any(tensor.dtype != torch.float32 for tensor in adapter_state.values()):
                raise ValueError(f"{adapter_name} adapter violates float32 storage")
        expected_shapes = [tensor.shape for tensor in v5_state.values()]
        parameter_ids = [
            parameter_id
            for group in optimizer_state["param_groups"]
            for parameter_id in group["params"]
        ]
        if len(parameter_ids) != len(expected_shapes) or set(parameter_ids) != set(
            optimizer_state["state"]
        ):
            raise ValueError("Initializer AdamW structure differs from v5")
        for shape, parameter_id in zip(expected_shapes, parameter_ids, strict=True):
            slots = optimizer_state["state"][parameter_id]
            for slot_name in ("exp_avg", "exp_avg_sq"):
                slot = slots.get(slot_name)
                if (
                    not isinstance(slot, torch.Tensor)
                    or slot.dtype != torch.float32
                    or slot.shape != shape
                ):
                    raise ValueError(
                        f"Initializer AdamW {slot_name} violates float32 storage"
                    )
        ema_parameters = state["v5_ema"].get("ema_parameters")
        if not isinstance(ema_parameters, list) or len(ema_parameters) != len(
            expected_shapes
        ):
            raise ValueError("Initializer EMA structure differs from v5")
        for ema_parameter, shape in zip(
            ema_parameters, expected_shapes, strict=True
        ):
            if ema_parameter.dtype != torch.float32 or ema_parameter.shape != shape:
                raise ValueError("Initializer EMA violates float32 storage")

    tensor_count = 0
    total_elements = 0
    for name, tensor in _iter_tensors(state):
        tensor_count += 1
        total_elements += tensor.numel()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Non-finite tensor in {name}")
    return state, {
        "initializer_dir": str(initializer_dir),
        "state_path": str(state_path),
        "state_sha256": marker.get("state_sha256"),
        "marker_path": str(marker_path),
        "schema_version": 1,
        "v5_optimization_step": optimization_step,
        "adapter_tensor_count": len(v5_state),
        "optimizer_parameter_states": len(optimizer_state["state"]),
        "has_v5_scaler": state["v5_scaler"] is not None,
        "has_v5_ema": True,
        "metadata": state["metadata"],
        "recursive_tensor_count": tensor_count,
        "recursive_tensor_elements": total_elements,
    }


def load_and_validate_policy(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate raw theta/old/theta-EMA state for an old-center control."""

    state_path = checkpoint_dir / POLICY_STATE_FILENAME
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing old-center policy state: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if set(state) != POLICY_REQUIRED_KEYS or state.get("schema_version") != 1:
        raise ValueError("Unexpected old-center policy state schema")
    default_state = state["adapter_default_raw"]
    old_state = state["adapter_old"]
    if not default_state or default_state.keys() != old_state.keys():
        raise ValueError("Raw theta and old adapter structures differ or are empty")
    tensor_count = 0
    total_elements = 0
    for name, tensor in _iter_tensors(state):
        tensor_count += 1
        total_elements += tensor.numel()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Non-finite tensor in {name}")
    if state["theta_ema"] is None:
        raise ValueError("Old-center policy state is missing theta EMA")
    return state, {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "state_path": str(state_path.resolve()),
        "schema_version": state["schema_version"],
        "v5_optimization_step": 0,
        "adapter_tensor_counts": {
            "adapter_default_raw": len(default_state),
            "adapter_old": len(old_state),
        },
        "recursive_tensor_count": tensor_count,
        "recursive_tensor_elements": total_elements,
        "has_theta_ema": True,
        "v5flow_enabled": False,
    }


def load_and_validate_resumable(
    checkpoint_dir: Path,
    *,
    require_complete_marker: bool,
    expect_v5flow: bool = True,
    expect_method: str | None = None,
    expect_cfg_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every state component used by a production resume.

    The exported LoRA is the theta EMA used by evaluation. Resume deliberately
    restores raw theta from ``v5flow_state.pt`` and its matching AdamW from
    ``optimizer.pt``; checking both prevents a partially written multi-file
    checkpoint from being selected after preemption.
    """

    checkpoint_dir = checkpoint_dir.resolve()
    marker_path = checkpoint_dir / COMPLETE_MARKER_FILENAME
    marker = None
    if marker_path.is_file():
        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
    elif require_complete_marker:
        raise FileNotFoundError(f"Missing completion marker: {marker_path}")

    checkpoint_name = checkpoint_dir.name
    if marker is not None:
        checkpoint_step = marker.get("checkpoint_step")
        if not isinstance(checkpoint_step, int) or isinstance(checkpoint_step, bool) or checkpoint_step < 0:
            raise ValueError("Completion marker checkpoint_step must be non-negative")
    else:
        if not checkpoint_name.startswith("checkpoint-"):
            raise ValueError(f"Checkpoint directory has an invalid name: {checkpoint_name}")
        try:
            checkpoint_step = int(checkpoint_name.removeprefix("checkpoint-"))
        except ValueError as error:
            raise ValueError(f"Checkpoint directory has a nonnumeric step: {checkpoint_name}") from error
        if checkpoint_step < 0:
            raise ValueError("Checkpoint step must be non-negative")

    marker_schema = marker.get("schema_version") if marker is not None else None
    marker_cfg_mode, marker_guidance_scale = _validate_marker_cfg_contract(marker)
    if expect_cfg_mode is not None:
        if expect_cfg_mode not in CFG_MODES:
            raise ValueError(f"Unsupported expected CFG mode: {expect_cfg_mode!r}")
        if marker_cfg_mode != expect_cfg_mode:
            raise ValueError(
                "Checkpoint CFG mode does not match the validator expectation: "
                f"{marker_cfg_mode!r} != {expect_cfg_mode!r}"
            )
    if expect_method is None:
        expect_method = "v5flow" if expect_v5flow else "old_center"
    if expect_method not in {"v5flow", "old_center"}:
        raise ValueError(f"Unsupported expected training method: {expect_method!r}")
    if marker_schema == 2:
        checkpoint_method = "v5flow" if bool(marker.get("v5flow_enabled")) else "old_center"
    else:
        checkpoint_method = expect_method
    if checkpoint_method != expect_method:
        raise ValueError(
            "Checkpoint training method does not match the validator expectation: "
            f"{checkpoint_method!r} != {expect_method!r}"
        )
    allowed_cfg_modes = {
        "v5flow": {
            "cfg1",
            "cfg4_rollout",
            "cfg4_dual_shifted_center",
            "cfg4_full",
            "cfg4_alternating",
        },
        "old_center": set(CFG_MODES),
    }[checkpoint_method]
    if marker_cfg_mode not in allowed_cfg_modes:
        raise ValueError(
            "Checkpoint CFG mode is incompatible with its training method: "
            f"method={checkpoint_method!r}, cfg_mode={marker_cfg_mode!r}, "
            f"allowed={sorted(allowed_cfg_modes)}"
        )

    if checkpoint_method == "v5flow":
        state, summary = load_and_validate(checkpoint_dir)
        summary["v5flow_enabled"] = True
        summary["training_method"] = "v5flow"
    else:
        state, summary = load_and_validate_policy(checkpoint_dir)
        summary["training_method"] = "old_center"

    required_nonempty_files = {
        "theta_optimizer": checkpoint_dir / "optimizer.pt",
        "theta_scaler": checkpoint_dir / "scaler.pt",
        "theta_ema_lora": checkpoint_dir / "lora" / "adapter_model.safetensors",
        "theta_ema_lora_config": checkpoint_dir / "lora" / "adapter_config.json",
    }
    for name, path in required_nonempty_files.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty {name}: {path}")

    theta_optimizer = torch.load(
        required_nonempty_files["theta_optimizer"],
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(theta_optimizer, dict) or not isinstance(
        theta_optimizer.get("param_groups"), list
    ):
        raise ValueError("Theta optimizer checkpoint is not an optimizer state dictionary")
    theta_scaler = torch.load(
        required_nonempty_files["theta_scaler"],
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(theta_scaler, dict):
        raise ValueError("Theta scaler checkpoint is not a state dictionary")

    if marker is not None:
        common_marker_keys = {
            "schema_version",
            "checkpoint_step",
            "v5_optimization_step",
            "resume_epoch",
            "checkpoint_kind",
        }
        if marker_schema == 1:
            expected_marker_keys = common_marker_keys
            if set(marker) not in (
                expected_marker_keys,
                expected_marker_keys | CFG_MARKER_KEYS,
            ):
                raise ValueError(
                    "Unexpected completion marker keys: "
                    f"missing={sorted(expected_marker_keys - set(marker))}, "
                    f"extra={sorted(set(marker) - expected_marker_keys)}"
                )
            if marker["checkpoint_kind"] not in {"outer_boundary", "final"}:
                raise ValueError("Unsupported completion marker checkpoint_kind")
            if checkpoint_name != f"checkpoint-{checkpoint_step}":
                raise ValueError("Schema-1 marker step does not match its checkpoint directory")
        elif marker_schema == 2:
            expected_marker_keys = common_marker_keys | {
                "v5flow_enabled",
                "resume_phase",
                "next_sampling_batch",
                "world_size",
                "rank_state_files",
            }
            if set(marker) not in (
                expected_marker_keys,
                expected_marker_keys | CFG_MARKER_KEYS,
            ):
                raise ValueError(
                    "Unexpected exact-resume completion marker keys: "
                    f"missing={sorted(expected_marker_keys - set(marker))}, "
                    f"extra={sorted(set(marker) - expected_marker_keys)}"
                )
            checkpoint_kind = marker["checkpoint_kind"]
            if checkpoint_kind not in {"requeue", "rolling"}:
                raise ValueError(
                    "Schema-2 marker checkpoint_kind must be 'requeue' or 'rolling'"
                )
            if not isinstance(marker["v5flow_enabled"], bool):
                raise ValueError("Exact marker v5flow_enabled must be boolean")
            if marker["v5flow_enabled"] != (expect_method == "v5flow"):
                raise ValueError("Exact marker v5flow_enabled does not match the validator expectation")
            if marker["resume_phase"] not in EXACT_RESUME_PHASES:
                raise ValueError("Unsupported exact resume_phase")
            if (
                not isinstance(marker["next_sampling_batch"], int)
                or isinstance(marker["next_sampling_batch"], bool)
                or marker["next_sampling_batch"] < 0
            ):
                raise ValueError("Exact next_sampling_batch must be non-negative")
            if checkpoint_kind == "rolling":
                if checkpoint_name != f"checkpoint-{checkpoint_step}":
                    raise ValueError("Rolling marker step does not match its checkpoint directory")
                if marker["resume_phase"] != "outer_boundary" or marker["next_sampling_batch"] != 0:
                    raise ValueError("Rolling checkpoints must contain an outer-boundary cursor")
            world_size = marker["world_size"]
            if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
                raise ValueError("Exact marker world_size must be positive")
            expected_rank_files = [f"rank-{rank:05d}.pt" for rank in range(world_size)]
            if marker["rank_state_files"] != expected_rank_files:
                raise ValueError("Exact rank_state_files do not exactly match world_size")
            for rank, filename in enumerate(expected_rank_files):
                rank_path = checkpoint_dir / filename
                if not rank_path.is_file() or rank_path.stat().st_size <= 0:
                    raise FileNotFoundError(f"Missing or empty exact rank state: {rank_path}")
                rank_state = torch.load(rank_path, map_location="cpu", weights_only=False)
                if set(rank_state) != EXACT_RANK_KEYS:
                    raise ValueError(f"Unexpected exact rank state keys: {rank_path}")
                expected_rank_values = {
                    "schema_version": 1,
                    "rank": rank,
                    "world_size": world_size,
                    "checkpoint_step": checkpoint_step,
                    "resume_epoch": marker["resume_epoch"],
                    "resume_phase": marker["resume_phase"],
                    "next_sampling_batch": marker["next_sampling_batch"],
                }
                for key, expected in expected_rank_values.items():
                    if rank_state[key] != expected:
                        raise ValueError(f"Exact rank state {key} mismatch in {rank_path}")
                rng_keys = {"python", "numpy", "torch_cpu", "torch_cuda"}
                if not isinstance(rank_state["rng_state"], dict) or set(rank_state["rng_state"]) != rng_keys:
                    raise ValueError(f"Incomplete RNG state in {rank_path}")
                if checkpoint_kind == "rolling" and rank_state["payload"] is not None:
                    raise ValueError(f"Rolling rank payload must be None in {rank_path}")
                for name, tensor in _iter_tensors(rank_state):
                    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
                        raise ValueError(f"Non-finite exact-resume tensor in {rank_path}:{name}")
        else:
            raise ValueError("Unsupported completion marker schema")

        if marker["checkpoint_step"] != checkpoint_step:
            raise ValueError("Completion marker checkpoint_step mismatch")
        if marker["v5_optimization_step"] != summary["v5_optimization_step"]:
            raise ValueError("Completion marker v5_optimization_step mismatch")
        if (
            not isinstance(marker["resume_epoch"], int)
            or isinstance(marker["resume_epoch"], bool)
            or marker["resume_epoch"] < 0
        ):
            raise ValueError("Completion marker resume_epoch must be non-negative")

    summary.update(
        {
            "checkpoint_step": checkpoint_step,
            "completion_marker": str(marker_path) if marker is not None else None,
            "checkpoint_kind": marker["checkpoint_kind"] if marker is not None else "legacy",
            "resume_epoch": marker["resume_epoch"] if marker is not None else None,
            "resume_phase": marker.get("resume_phase") if marker is not None else None,
            "cfg_mode": marker_cfg_mode,
            "rollout_guidance_scale": marker_guidance_scale,
            "theta_optimizer_parameter_states": len(theta_optimizer.get("state", {})),
            "theta_scaler_keys": sorted(theta_scaler),
        }
    )
    return state, summary


def assert_identical(left: Any, right: Any, prefix: str = "state") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise ValueError(f"Type mismatch at {prefix}")
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0, equal_nan=False)
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError(f"Dictionary mismatch at {prefix}")
        for key in left:
            assert_identical(left[key], right[key], f"{prefix}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise ValueError(f"Sequence mismatch at {prefix}")
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            assert_identical(left_value, right_value, f"{prefix}[{index}]")
        return
    if left != right:
        raise ValueError(f"Value mismatch at {prefix}: {left!r} != {right!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--resumable", action="store_true")
    parser.add_argument("--initializer", action="store_true")
    parser.add_argument("--require-complete-marker", action="store_true")
    parser.add_argument("--expect-step", type=int, default=None)
    parser.add_argument("--expect-source-sha256", default=None)
    parser.add_argument("--expect-source-config-sha256", default=None)
    parser.add_argument("--expect-config-sha256", default=None)
    parser.add_argument("--expect-prompt-sha256", default=None)
    parser.add_argument(
        "--expect-adapter-storage-dtype",
        choices=("model", "float32"),
        default=None,
    )
    parser.add_argument("--expect-base-revision", default=None)
    parser.add_argument(
        "--expect-v5flow",
        choices=("true", "false"),
        default=None,
        help="Backward-compatible alias selecting v5flow (true) or old_center (false).",
    )
    parser.add_argument(
        "--expect-method",
        choices=("v5flow", "old_center"),
        default=None,
        help="Select the complete training state expected in the checkpoint.",
    )
    parser.add_argument(
        "--expect-cfg-mode",
        choices=tuple(CFG_MODES),
        default=None,
        help="Require the checkpoint marker's CFG training contract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.initializer and args.resumable:
        raise ValueError("--initializer and --resumable are mutually exclusive")
    if args.require_complete_marker and not args.resumable:
        raise ValueError("--require-complete-marker requires --resumable")
    if args.expect_cfg_mode is not None and not args.resumable:
        raise ValueError("--expect-cfg-mode requires --resumable")
    if args.expect_method is not None and args.expect_v5flow is not None:
        alias_method = "v5flow" if args.expect_v5flow == "true" else "old_center"
        if alias_method != args.expect_method:
            raise ValueError("--expect-method conflicts with --expect-v5flow")
    expected_method = (
        args.expect_method
        if args.expect_method is not None
        else "v5flow"
        if args.expect_v5flow is None or args.expect_v5flow == "true"
        else "old_center"
    )
    if args.initializer:
        if args.expect_method is not None or args.expect_v5flow is not None:
            raise ValueError("initializer validation does not accept a training method")
        state, summary = load_and_validate_initializer(args.checkpoint_dir)
    elif args.resumable:
        state, summary = load_and_validate_resumable(
            args.checkpoint_dir,
            require_complete_marker=args.require_complete_marker,
            expect_v5flow=expected_method == "v5flow",
            expect_method=expected_method,
            expect_cfg_mode=args.expect_cfg_mode,
        )
    else:
        state, summary = load_and_validate(args.checkpoint_dir)
    initializer_expectations = {
        "source_lora_adapter_model_sha256": args.expect_source_sha256,
        "source_lora_adapter_config_sha256": args.expect_source_config_sha256,
        "submission_config_sha256": args.expect_config_sha256,
        "warmup_prompt_sha256": args.expect_prompt_sha256,
        "adapter_storage_dtype": args.expect_adapter_storage_dtype,
        "base_model_revision": args.expect_base_revision,
    }
    if args.expect_step is not None:
        if not args.initializer:
            raise ValueError("--expect-step is supported only with --initializer")
        if summary["v5_optimization_step"] != args.expect_step:
            raise ValueError(
                "Initializer optimization step mismatch: "
                f"{summary['v5_optimization_step']} != {args.expect_step}"
            )
    for metadata_key, expected in initializer_expectations.items():
        if expected is None:
            continue
        if not args.initializer:
            raise ValueError(
                f"--expect-{metadata_key.replace('_', '-')} requires --initializer"
            )
        actual = summary["metadata"].get(metadata_key)
        if actual != expected:
            raise ValueError(
                f"Initializer metadata mismatch for {metadata_key}: "
                f"{actual!r} != {expected!r}"
            )
    if args.compare is not None:
        if args.initializer:
            other_state, other_summary = load_and_validate_initializer(args.compare)
        else:
            other_state, other_summary = load_and_validate(args.compare)
        assert_identical(state, other_state)
        summary["bitwise_identical_to"] = other_summary.get(
            "checkpoint_dir", other_summary.get("initializer_dir")
        )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
