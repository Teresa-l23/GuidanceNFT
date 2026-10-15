"""Full-shared-model online checkpoints for ELF NFT."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flax import serialization


# Version 7 adds the evaluation-only theta EMA to ``current_state``.  A v6
# checkpoint cannot reconstruct that history from its final raw parameters, so
# it must not resume a theta-EMA experiment silently.
SCHEMA_VERSION = 7
MARKER_FILENAME = ".elf_nft_checkpoint.json"
CHECKPOINT_PREFIX = "checkpoint_"
CHECKPOINT_SUFFIX = ".msgpack"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_payload(
    *, current_state, center_state, old_params, reward_state, rng, outer_step: int,
    parameterization: str, phase: str = "online", center_pretrain_step: int = 0,
) -> dict[str, Any]:
    if parameterization != "full_shared":
        raise ValueError("unsupported checkpoint parameterization")
    if phase not in {"online", "center_pretrain"}:
        raise ValueError("checkpoint phase must be online or center_pretrain")
    outer_step = int(outer_step)
    center_pretrain_step = int(center_pretrain_step)
    if phase == "online" and outer_step < 0:
        raise ValueError("online checkpoint outer_step must be non-negative")
    if phase == "center_pretrain" and (
        outer_step != 0 or center_pretrain_step <= 0
    ):
        raise ValueError(
            "center-pretrain checkpoint requires outer_step=0 and positive step"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "parameterization": parameterization,
        "phase": phase,
        "outer_step": outer_step,
        "center_pretrain_step": center_pretrain_step,
        "current_state": current_state,
        "center_state": center_state,
        "old_params": old_params,
        "reward_state": reward_state,
        "rng": rng,
    }


def save_online_checkpoint(
    *,
    output_dir: str,
    payload: dict[str, Any],
    keep: int,
    method: str,
    base_checkpoint: str,
    base_checkpoint_revision: str | None = None,
    config_path: str,
    keep_every_n_steps: int = 25,
    self_condition_mode: str = "bernoulli",
    source_cfg_mode: str = "legacy",
    theta_ema_decay: float = 0.0,
) -> str:
    """Atomically publish a checkpoint and prune only non-milestone history."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint payload schema mismatch")
    if payload.get("parameterization") != "full_shared":
        raise ValueError("checkpoint payload parameterization mismatch")
    phase = payload.get("phase", "online")
    outer_step = int(payload["outer_step"])
    center_pretrain_step = int(payload.get("center_pretrain_step", 0))
    if phase == "online":
        if outer_step <= 0:
            raise ValueError("completed online checkpoint outer_step must be positive")
        checkpoint_name = f"{CHECKPOINT_PREFIX}{outer_step:08d}{CHECKPOINT_SUFFIX}"
    elif phase == "center_pretrain":
        if outer_step != 0 or center_pretrain_step <= 0:
            raise ValueError(
                "center-pretrain checkpoint requires outer_step=0 and positive step"
            )
        checkpoint_name = (
            f"{CHECKPOINT_PREFIX}center_pretrain_"
            f"{center_pretrain_step:08d}{CHECKPOINT_SUFFIX}"
        )
    else:
        raise ValueError("checkpoint phase must be online or center_pretrain")
    if isinstance(keep, bool) or not isinstance(keep, int) or keep <= 0:
        raise ValueError("keep must be a positive integer")
    if (
        isinstance(keep_every_n_steps, bool)
        or not isinstance(keep_every_n_steps, int)
        or keep_every_n_steps <= 0
    ):
        raise ValueError("keep_every_n_steps must be a positive integer")
    if (
        isinstance(theta_ema_decay, bool)
        or not isinstance(theta_ema_decay, (int, float))
        or not math.isfinite(float(theta_ema_decay))
        or not 0.0 <= float(theta_ema_decay) < 1.0
    ):
        raise ValueError("theta_ema_decay must lie in [0, 1)")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / checkpoint_name
    checkpoint_temporary = root / f".{checkpoint_path.name}.tmp.{os.getpid()}"
    checkpoint_bytes = serialization.to_bytes(payload)
    with checkpoint_temporary.open("wb") as handle:
        handle.write(checkpoint_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(checkpoint_temporary, checkpoint_path)

    marker = {
        "schema_version": SCHEMA_VERSION,
        "parameterization": payload["parameterization"],
        "phase": phase,
        "outer_step": outer_step,
        "center_pretrain_step": center_pretrain_step,
        "method": method,
        "self_condition_mode": self_condition_mode,
        "source_cfg_mode": source_cfg_mode,
        "base_checkpoint": base_checkpoint,
        "base_checkpoint_revision": base_checkpoint_revision,
        "theta_ema_decay": float(theta_ema_decay),
        "theta_ema_step": outer_step if theta_ema_decay > 0.0 else 0,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    marker_path = root / MARKER_FILENAME
    temporary = root / f"{MARKER_FILENAME}.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(marker, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, marker_path)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    # Commit the new pointer before pruning: a killed save must leave the
    # previous published checkpoint resumable. Center pretraining publishes at
    # milestones but retains only the newest complete state. Once an online
    # checkpoint is durable, its no-longer-needed pretrain predecessor is
    # removed. Online numeric milestones keep the historical retention policy.
    pretrain_checkpoints = list(
        root.glob(f"{CHECKPOINT_PREFIX}center_pretrain_*{CHECKPOINT_SUFFIX}")
    )
    if phase == "center_pretrain":
        for stale_path in pretrain_checkpoints:
            if stale_path != checkpoint_path:
                stale_path.unlink()
        return str(checkpoint_path)
    for stale_path in pretrain_checkpoints:
        stale_path.unlink()
    all_checkpoints = sorted(
        (
            path
            for path in root.glob(f"{CHECKPOINT_PREFIX}*{CHECKPOINT_SUFFIX}")
            if "center_pretrain_" not in path.name
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    latest_checkpoints = set(all_checkpoints[-keep:])
    for stale_path in all_checkpoints:
        step_text = stale_path.name[len(CHECKPOINT_PREFIX):-len(CHECKPOINT_SUFFIX)]
        is_milestone = (
            step_text.isdigit()
            and int(step_text) % keep_every_n_steps == 0
        )
        if stale_path not in latest_checkpoints and not is_milestone:
            stale_path.unlink()
    return str(checkpoint_path)


def resolve_resume_checkpoint(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return str(candidate)
    if not candidate.is_dir():
        raise FileNotFoundError(f"resume path does not exist: {candidate}")
    marker_path = candidate / MARKER_FILENAME
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        published = Path(marker["checkpoint_path"]).resolve()
        if marker.get("phase", "online") == "center_pretrain":
            expected_name = (
                f"{CHECKPOINT_PREFIX}center_pretrain_"
                f"{int(marker['center_pretrain_step']):08d}{CHECKPOINT_SUFFIX}"
            )
        else:
            expected_name = (
                f"{CHECKPOINT_PREFIX}{int(marker['outer_step']):08d}"
                f"{CHECKPOINT_SUFFIX}"
            )
        if published.parent != candidate or published.name != expected_name:
            raise ValueError("checkpoint marker must reference its own completed outer-step file")
        if not published.is_file():
            raise FileNotFoundError(f"published checkpoint is missing: {published}")
        return str(published)
    matches = sorted(
        candidate.glob(f"{CHECKPOINT_PREFIX}*{CHECKPOINT_SUFFIX}"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not matches:
        raise FileNotFoundError(f"no ELF NFT checkpoints under {candidate}")
    return str(matches[-1])


def validate_resume_metadata(
    path: str,
    *,
    expected_method: str,
    expected_base_checkpoint: str,
    expected_base_checkpoint_revision: str | None = None,
    expected_config_path: str,
    expected_parameterization: str,
    expected_self_condition_mode: str = "bernoulli",
    expected_source_cfg_mode: str = "legacy",
    expected_theta_ema_decay: float = 0.0,
) -> None:
    """Reject a resume whose published scientific configuration has changed."""

    checkpoint_path = Path(resolve_resume_checkpoint(path))
    marker_path = checkpoint_path.parent / MARKER_FILENAME
    if not marker_path.is_file():
        raise FileNotFoundError(f"resume marker is missing: {marker_path}")
    with marker_path.open("r", encoding="utf-8") as handle:
        marker = json.load(handle)
    if marker.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported resume marker schema")
    if marker.get("parameterization") != expected_parameterization:
        raise ValueError("resume parameterization does not match active configuration")
    if marker.get("method") != expected_method:
        raise ValueError(
            f"resume method mismatch: {marker.get('method')!r} != {expected_method!r}"
        )
    if marker.get("self_condition_mode", "bernoulli") != expected_self_condition_mode:
        raise ValueError("resume self_condition_mode does not match active configuration")
    if marker.get("source_cfg_mode", "legacy") != expected_source_cfg_mode:
        raise ValueError("resume source_cfg_mode does not match active configuration")
    if marker.get("base_checkpoint") != expected_base_checkpoint:
        raise ValueError("resume base checkpoint does not match active configuration")
    if (
        marker.get("base_checkpoint_revision")
        != expected_base_checkpoint_revision
    ):
        raise ValueError(
            "resume base checkpoint revision does not match active configuration"
        )
    if marker.get("theta_ema_decay", 0.0) != float(expected_theta_ema_decay):
        raise ValueError("resume theta EMA decay does not match active configuration")
    expected_theta_ema_step = (
        int(marker["outer_step"]) if expected_theta_ema_decay > 0.0 else 0
    )
    if marker.get("theta_ema_step", 0) != expected_theta_ema_step:
        raise ValueError("resume theta EMA step does not match outer step")
    expected_config_sha256 = sha256_file(expected_config_path)
    if marker.get("config_sha256") != expected_config_sha256:
        raise ValueError("resume config content does not match active configuration")


def load_online_checkpoint(path: str, template: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = resolve_resume_checkpoint(path)
    with Path(checkpoint_path).open("rb") as handle:
        restored = serialization.from_bytes(template, handle.read())
    if restored.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported ELF NFT checkpoint schema")
    if restored.get("parameterization") != "full_shared":
        raise ValueError("unsupported ELF NFT checkpoint parameterization")
    phase = restored.get("phase", "online")
    outer_step = int(restored["outer_step"])
    center_pretrain_step = int(restored.get("center_pretrain_step", 0))
    if phase == "online" and outer_step <= 0:
        raise ValueError("restored online outer_step must be positive")
    if phase == "center_pretrain" and (
        outer_step != 0 or center_pretrain_step <= 0
    ):
        raise ValueError("restored center-pretrain progress is invalid")
    return restored


def load_policy_params_from_checkpoint(
    path: str, *, snapshot: str = "raw"
) -> tuple[Mapping, int]:
    """Restore raw or theta-EMA policy parameters from an online checkpoint.

    Evaluation does not need optimizer moments, the old policy, reward-controller
    state, or a learned center.  Reading the raw Flax state dictionary avoids
    constructing equally large optimizer templates solely to discard them.
    """

    checkpoint_path = resolve_resume_checkpoint(path)
    with Path(checkpoint_path).open("rb") as handle:
        restored = serialization.msgpack_restore(handle.read())
    if not isinstance(restored, Mapping):
        raise ValueError("ELF NFT checkpoint root must be a mapping")
    if restored.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported ELF NFT checkpoint schema")
    if restored.get("parameterization") != "full_shared":
        raise ValueError("unsupported ELF NFT checkpoint parameterization")
    outer_step = int(restored.get("outer_step", 0))
    if outer_step <= 0:
        raise ValueError("restored outer_step must be positive")
    current_state = restored.get("current_state")
    if not isinstance(current_state, Mapping):
        raise ValueError("checkpoint current_state must be a mapping")
    if snapshot not in {"raw", "theta_ema"}:
        raise ValueError("policy snapshot must be raw or theta_ema")
    params_key = "params" if snapshot == "raw" else "ema_params"
    params = current_state.get(params_key)
    if not isinstance(params, Mapping):
        raise ValueError(
            f"checkpoint current_state is missing {snapshot} parameter mapping"
        )
    return params, outer_step
