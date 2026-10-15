#!/usr/bin/env python
"""Validate an atomic multi-refresh SD3.5 endpoint-FM pretrain checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


STATE_FILENAME = "v5flow_pretrain_state.pt"
MARKER_FILENAME = ".v5flow_pretrain_complete.json"
STATE_KEYS = {
    "schema_version",
    "source_adapter_default",
    "adapter_v5_raw",
    "optimizer_v5",
    "v5_scaler",
    "v5_ema",
    "v5_optimization_step",
    "completed_refreshes",
    "metadata",
}
MARKER_KEYS = {
    "schema_version",
    "state_filename",
    "v5_optimization_step",
    "completed_refreshes",
    "world_size",
    "rank_rng_files",
    "state_sha256",
    "rank_rng_sha256",
    "metadata",
}
RANK_KEYS = {
    "schema_version",
    "rank",
    "world_size",
    "v5_optimization_step",
    "completed_refreshes",
    "rng_state",
}
RNG_KEYS = {"python", "numpy", "torch_cpu", "torch_cuda"}
CHECKPOINT_NAME = re.compile(r"checkpoint-(0|[1-9][0-9]*)\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


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


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_and_validate(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_dir = checkpoint_dir.resolve()
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint must be a real directory: {checkpoint_dir}")
    match = CHECKPOINT_NAME.fullmatch(checkpoint_dir.name)
    if match is None:
        raise ValueError(f"Invalid checkpoint directory name: {checkpoint_dir.name}")

    marker_path = checkpoint_dir / MARKER_FILENAME
    state_path = checkpoint_dir / STATE_FILENAME
    if not marker_path.is_file() or marker_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing pretrain completion marker: {marker_path}")
    if not state_path.is_file() or state_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing pretrain common state: {state_path}")
    with marker_path.open("r", encoding="utf-8") as marker_file:
        marker = json.load(marker_file)
    if not isinstance(marker, dict) or set(marker) != MARKER_KEYS:
        raise ValueError("Unexpected v5flow pretrain marker keys")
    if marker.get("schema_version") != 1 or marker.get("state_filename") != STATE_FILENAME:
        raise ValueError("Unsupported v5flow pretrain marker schema")
    step = _positive_int(marker["v5_optimization_step"], "v5_optimization_step")
    refreshes = _positive_int(marker["completed_refreshes"], "completed_refreshes")
    world_size = _positive_int(marker["world_size"], "world_size")
    if int(match.group(1)) != step:
        raise ValueError("Checkpoint directory and marker optimizer steps disagree")
    if not isinstance(marker["metadata"], dict):
        raise ValueError("Marker metadata must be a mapping")
    expected_rank_files = [f"rank-{rank:05d}-rng.pt" for rank in range(world_size)]
    if marker["rank_rng_files"] != expected_rank_files:
        raise ValueError("Marker rank_rng_files do not match world_size")
    if (
        not isinstance(marker["state_sha256"], str)
        or not DIGEST.fullmatch(marker["state_sha256"])
        or _sha256(state_path) != marker["state_sha256"]
    ):
        raise ValueError("Common-state SHA256 does not match the completion marker")
    rank_digests = marker["rank_rng_sha256"]
    if (
        not isinstance(rank_digests, dict)
        or set(rank_digests) != set(expected_rank_files)
        or any(
            not isinstance(value, str) or not DIGEST.fullmatch(value)
            for value in rank_digests.values()
        )
    ):
        raise ValueError("Marker rank_rng_sha256 does not match world_size")

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise ValueError("Unexpected v5flow pretrain common-state keys")
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported v5flow pretrain common-state schema")
    if (
        state["v5_optimization_step"] != step
        or state["completed_refreshes"] != refreshes
        or state["metadata"] != marker["metadata"]
    ):
        raise ValueError("v5flow pretrain common state and marker disagree")
    source = state["source_adapter_default"]
    v5 = state["adapter_v5_raw"]
    if (
        not isinstance(source, dict)
        or not source
        or not isinstance(v5, dict)
        or source.keys() != v5.keys()
    ):
        raise ValueError("Source and v5 adapter structures differ or are empty")
    for key, source_tensor in source.items():
        v5_tensor = v5[key]
        if (
            not isinstance(source_tensor, torch.Tensor)
            or not isinstance(v5_tensor, torch.Tensor)
            or source_tensor.shape != v5_tensor.shape
        ):
            raise ValueError(f"Invalid adapter tensor pair: {key}")
    optimizer = state["optimizer_v5"]
    if (
        not isinstance(optimizer, dict)
        or not isinstance(optimizer.get("param_groups"), list)
        or not optimizer.get("state")
    ):
        raise ValueError("Pretrain checkpoint is missing AdamW parameter state")
    if state["v5_ema"] is None:
        raise ValueError("Pretrain checkpoint is missing configured v5 EMA state")
    if state["metadata"].get("adapter_storage_dtype") == "float32":
        if any(
            tensor.dtype != torch.float32
            for adapter_state in (source, v5)
            for tensor in adapter_state.values()
        ):
            raise ValueError("Pretrain raw adapters violate float32 storage")
        expected_shapes = [tensor.shape for tensor in v5.values()]
        parameter_ids = [
            parameter_id
            for group in optimizer["param_groups"]
            for parameter_id in group["params"]
        ]
        if len(parameter_ids) != len(expected_shapes) or set(parameter_ids) != set(
            optimizer["state"]
        ):
            raise ValueError("Pretrain AdamW structure differs from v5")
        for shape, parameter_id in zip(expected_shapes, parameter_ids, strict=True):
            slots = optimizer["state"][parameter_id]
            for slot_name in ("exp_avg", "exp_avg_sq"):
                slot = slots.get(slot_name)
                if (
                    not isinstance(slot, torch.Tensor)
                    or slot.dtype != torch.float32
                    or slot.shape != shape
                ):
                    raise ValueError(
                        f"Pretrain AdamW {slot_name} violates float32 storage"
                    )
        ema_parameters = state["v5_ema"].get("ema_parameters")
        if not isinstance(ema_parameters, list) or len(ema_parameters) != len(
            expected_shapes
        ):
            raise ValueError("Pretrain EMA structure differs from v5")
        for ema_parameter, shape in zip(
            ema_parameters, expected_shapes, strict=True
        ):
            if ema_parameter.dtype != torch.float32 or ema_parameter.shape != shape:
                raise ValueError("Pretrain EMA violates float32 storage")

    rank_rng_summaries = []
    for rank, filename in enumerate(expected_rank_files):
        rank_path = checkpoint_dir / filename
        if not rank_path.is_file() or rank_path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing rank-local pretrain RNG state: {rank_path}")
        if _sha256(rank_path) != rank_digests[filename]:
            raise ValueError(f"Rank-local SHA256 mismatch: {rank_path}")
        rank_state = torch.load(rank_path, map_location="cpu", weights_only=False)
        if not isinstance(rank_state, dict) or set(rank_state) != RANK_KEYS:
            raise ValueError(f"Unexpected rank-local pretrain state keys: {rank_path}")
        expected_values = {
            "schema_version": 1,
            "rank": rank,
            "world_size": world_size,
            "v5_optimization_step": step,
            "completed_refreshes": refreshes,
        }
        for key, expected in expected_values.items():
            if rank_state[key] != expected:
                raise ValueError(f"Rank-local {key} mismatch in {rank_path}")
        rng_state = rank_state["rng_state"]
        if not isinstance(rng_state, dict) or set(rng_state) != RNG_KEYS:
            raise ValueError(f"Incomplete rank-local RNG state: {rank_path}")
        rank_rng_summaries.append(
            {
                "rank": rank,
                "path": str(rank_path),
                "sha256": rank_digests[filename],
                "has_cuda_rng": rng_state["torch_cuda"] is not None,
            }
        )

    tensor_count = 0
    tensor_elements = 0
    for name, tensor in _iter_tensors(state):
        tensor_count += 1
        tensor_elements += tensor.numel()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Non-finite common-state tensor: {name}")
    return state, {
        "checkpoint_dir": str(checkpoint_dir),
        "marker_path": str(marker_path),
        "state_path": str(state_path),
        "state_sha256": marker["state_sha256"],
        "schema_version": 1,
        "v5_optimization_step": step,
        "completed_refreshes": refreshes,
        "world_size": world_size,
        "adapter_tensor_count": len(v5),
        "optimizer_parameter_states": len(optimizer["state"]),
        "has_v5_scaler": state["v5_scaler"] is not None,
        "has_v5_ema": True,
        "rank_rng_states": rank_rng_summaries,
        "metadata": state["metadata"],
        "recursive_tensor_count": tensor_count,
        "recursive_tensor_elements": tensor_elements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expect-step", type=int)
    parser.add_argument("--expect-refreshes", type=int)
    parser.add_argument("--expect-world-size", type=int)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    _, summary = load_and_validate(args.checkpoint)
    expectations = {
        "v5_optimization_step": args.expect_step,
        "completed_refreshes": args.expect_refreshes,
        "world_size": args.expect_world_size,
    }
    for field, expected in expectations.items():
        if expected is not None and summary[field] != expected:
            raise ValueError(
                f"Expected {field}={expected}, got {summary[field]}"
            )
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.summary.with_name(f".{args.summary.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.summary)
    print(payload, end="")


if __name__ == "__main__":
    main()
