#!/usr/bin/env python3
"""Validate deterministic GenEval outputs and compare them with paper targets."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_SAMPLE_COUNT = 2_212
PAPER_BASE_CFG_FREE_ACCURACY = 0.24
PAPER_SINGLE_REWARD_ACCURACY = 0.98


def _finite_number(value: Any, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: field {field!r} is not numeric: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path}: field {field!r} is not finite: {value!r}")
    return number


def load_geneval_results(path: Path, expected_count: int) -> dict[str, Any]:
    """Load one JSONL evaluation, reject padding/duplicates, and aggregate scores.

    Input records must contain an integer ``sample_id`` and a ``scores`` mapping.
    The IDs must be exactly ``range(expected_count)``.  The returned mapping contains
    the sample count plus means for every numeric score whose sentinel value is not
    ``-10``.  The paper-facing metric is returned as ``accuracy``.
    """
    records_by_id: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            sample_id = record.get("sample_id")
            if isinstance(sample_id, bool) or not isinstance(sample_id, int):
                raise ValueError(f"{path}:{line_number}: invalid sample_id {sample_id!r}")
            if sample_id in records_by_id:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id}")
            if not isinstance(record.get("scores"), dict):
                raise ValueError(f"{path}:{line_number}: missing scores mapping")
            records_by_id[sample_id] = record

    expected_ids = set(range(expected_count))
    actual_ids = set(records_by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        unexpected = sorted(actual_ids - expected_ids)[:10]
        raise ValueError(
            f"{path}: expected sample IDs 0..{expected_count - 1}; "
            f"missing(first 10)={missing}, unexpected(first 10)={unexpected}"
        )

    values: dict[str, list[float]] = {}
    for sample_id in range(expected_count):
        for name, value in records_by_id[sample_id]["scores"].items():
            number = _finite_number(value, field=f"scores.{name}", path=path)
            if number != -10.0:
                values.setdefault(name, []).append(number)

    if "accuracy" not in values or len(values["accuracy"]) != expected_count:
        raise ValueError(
            f"{path}: paper metric 'accuracy' must be present for all {expected_count} samples"
        )

    return {
        "results_path": str(path.resolve()),
        "sample_count": expected_count,
        "metrics": {name: sum(scores) / len(scores) for name, scores in sorted(values.items())},
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace ``path`` with a UTF-8 JSON serialization of ``payload``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--trained-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_SAMPLE_COUNT)
    parser.add_argument(
        "--minimum-trained-accuracy",
        type=float,
        default=None,
        help="Optional user-chosen threshold; when set, exit 2 if trained accuracy is lower.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if args.minimum_trained_accuracy is not None and not 0.0 <= args.minimum_trained_accuracy <= 1.0:
        raise ValueError("--minimum-trained-accuracy must be in [0, 1]")

    base = load_geneval_results(args.base_results, args.expected_count)
    trained = load_geneval_results(args.trained_results, args.expected_count)
    base_accuracy = base["metrics"]["accuracy"]
    trained_accuracy = trained["metrics"]["accuracy"]
    passed = (
        None
        if args.minimum_trained_accuracy is None
        else trained_accuracy >= args.minimum_trained_accuracy
    )

    summary = {
        "protocol": {
            "dataset": "GenEval test",
            "sample_count": args.expected_count,
            "paper_metric": "accuracy",
            "note": "geneval is the dense training reward; strict_accuracy is a separate diagnostic.",
        },
        "paper_targets": {
            "base_cfg_free_accuracy": PAPER_BASE_CFG_FREE_ACCURACY,
            "single_reward_1000_step_accuracy": PAPER_SINGLE_REWARD_ACCURACY,
        },
        "acceptance": {
            "minimum_trained_accuracy": args.minimum_trained_accuracy,
            "passed": passed,
            "meets_paper_reported_point_estimate": (
                trained_accuracy >= PAPER_SINGLE_REWARD_ACCURACY
            ),
        },
        "base": base,
        "trained": trained,
        "accuracy_delta": trained_accuracy - base_accuracy,
    }
    write_json_atomic(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
