#!/usr/bin/env python
"""Plot completed native SC-CFG evaluations with one color per policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_ASSET_ROOT = Path(os.environ.get("GUIDANCENFT_ASSETS", "assets"))
ASSET_ROOT = Path(os.environ.get("ELF_NFT_ASSET_ROOT", str(DEFAULT_ASSET_ROOT)))
MODEL_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#B279A2",
    "#72B7B2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
)
PLOT_LABELS_BY_POLICY_ID = {
    "standard_trained_step100": "standard / dual / CFG1 (step 100)",
    "shiftcenter_untrained_step100": "shiftcenter / conditional / CFG3 (step 100)",
    "recoveredcenter_trained_step100": "recovered center / dual / CFG1 (step 100)",
    "standard_recent_bernoulli_trained_step100": (
        "standard / bernoulli / CFG1 (step 100)"
    ),
}
CENTER_LABELS = {
    "standard": "standard",
    "shifted_center": "shiftcenter",
    "recovered_center": "recovered center",
}
BRANCH_LABELS = {
    "paired": "dual",
    "conditional_only": "conditional",
    "bernoulli": "bernoulli",
}
ANNOTATION_OFFSETS = (
    (-6, 8),
    (-6, -9),
    (0, 14),
    (0, -15),
    (-9, 18),
    (-9, -19),
)


@dataclass(frozen=True)
class SweepPoint:
    """One completed policy evaluation at one native SC-CFG weight."""

    name: str
    weight: float
    entropy: float
    ppl: float
    num_samples: int
    seed: int


@dataclass(frozen=True)
class PolicyCurve:
    """All comparable sweep points for one released or adapted policy."""

    policy_id: str
    display_name: str
    policy_source: str
    outer_step: int
    results_path: Path
    checkpoint_path: Path | None
    source_dir: Path
    points: tuple[SweepPoint, ...]


@dataclass(frozen=True)
class CompletedSweep:
    """A suite whose parent and every policy have published completion markers."""

    suite_dir: Path
    cfg_scales: tuple[float, ...]
    evaluation: dict[str, Any]
    curves: tuple[PolicyCurve, ...]
    additional_run_dirs: tuple[Path, ...] = ()
    additional_evaluation_dirs: tuple[Path, ...] = ()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON at {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                rows.append(row)
    except FileNotFoundError as error:
        raise ValueError(f"missing required file: {path}") from error
    return rows


def _finite_float(record: dict[str, Any], key: str, context: str) -> float:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context}: missing or invalid {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"{context}: {key} must be finite")
    return value


def _positive_int(record: dict[str, Any], key: str, context: str) -> int:
    try:
        value = int(record[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context}: missing or invalid {key}") from error
    if value <= 0:
        raise ValueError(f"{context}: {key} must be positive")
    return value


def _nonnegative_int(record: dict[str, Any], key: str, context: str) -> int:
    try:
        value = int(record[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context}: missing or invalid {key}") from error
    if value < 0:
        raise ValueError(f"{context}: {key} must be non-negative")
    return value


def _ordered_unique_scales(values: Any, context: str) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{context}: cfg_scales must be a non-empty list")
    scales = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in scales):
        raise ValueError(f"{context}: cfg_scales must be finite and non-negative")
    if len(set(scales)) != len(scales):
        raise ValueError(f"{context}: cfg_scales contains duplicates")
    if scales != tuple(sorted(scales)):
        raise ValueError(f"{context}: cfg_scales must be strictly increasing")
    return scales


def _load_policy_curve(
    suite_dir: Path,
    model_record: dict[str, Any],
    expected_scales: tuple[float, ...],
    expected_evaluation: dict[str, Any] | None,
    *,
    policy_dir: Path | None = None,
) -> tuple[PolicyCurve, dict[str, Any]]:
    context = f"{suite_dir}/.complete model entry"
    try:
        policy_id = str(model_record["policy_id"])
        display_name = str(model_record["display_name"])
        policy_source = str(model_record["policy_source"])
        outer_step = int(model_record["outer_step"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context}: invalid model identity") from error
    if not policy_id or Path(policy_id).name != policy_id:
        raise ValueError(f"{context}: unsafe policy_id {policy_id!r}")
    if not display_name or not policy_source or outer_step < 0:
        raise ValueError(f"{context}: empty label/source or negative outer_step")

    policy_dir = (suite_dir / policy_id) if policy_dir is None else policy_dir.resolve()
    policy_complete_path = policy_dir / ".complete"
    if not policy_complete_path.is_file():
        raise ValueError(f"policy is not complete: {policy_complete_path}")
    policy_complete = _load_json(policy_complete_path)
    manifest = _load_json(policy_dir / "evaluation_manifest.json")
    for key, expected in (
        ("policy_id", policy_id),
        ("display_name", display_name),
        ("policy_source", policy_source),
        ("outer_step", outer_step),
    ):
        if policy_complete.get(key) != expected:
            raise ValueError(f"{policy_complete_path}: {key} does not match suite")
        if manifest.get(key) != expected:
            raise ValueError(f"{policy_dir}/evaluation_manifest.json: {key} mismatch")

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"{policy_dir}/evaluation_manifest.json: missing evaluation")
    manifest_scales = _ordered_unique_scales(
        evaluation.get("model_cfg_scales"),
        f"{policy_dir}/evaluation_manifest.json",
    )
    if manifest_scales != expected_scales:
        raise ValueError(f"{policy_id}: manifest CFG scales do not match suite")
    if expected_evaluation is not None and evaluation != expected_evaluation:
        raise ValueError(f"{policy_id}: evaluation settings differ across policies")
    expected_external_cfg = _finite_float(
        evaluation, "external_cfg_scale", f"{policy_id} evaluation manifest"
    )
    expected_num_samples = _positive_int(
        evaluation, "num_samples", f"{policy_id} evaluation manifest"
    )
    expected_seed = _nonnegative_int(
        evaluation, "seed", f"{policy_id} evaluation manifest"
    )

    results_path = policy_dir / "results.jsonl"
    rows = _load_jsonl(results_path)
    if len(rows) != len(expected_scales):
        raise ValueError(
            f"{results_path}: expected {len(expected_scales)} rows, got {len(rows)}"
        )
    points_by_weight: dict[float, SweepPoint] = {}
    for row_number, row in enumerate(rows, start=1):
        row_context = f"{results_path}:{row_number}"
        weight = _finite_float(row, "model_cfg_scale", row_context)
        entropy = _finite_float(row, "entropy", row_context)
        ppl = _finite_float(row, "ppl", row_context)
        num_samples = _positive_int(row, "num_samples", row_context)
        seed = _nonnegative_int(row, "seed", row_context)
        if weight not in expected_scales:
            raise ValueError(f"{row_context}: unexpected model_cfg_scale={weight:g}")
        if weight in points_by_weight:
            raise ValueError(f"{row_context}: duplicate model_cfg_scale={weight:g}")
        if entropy < 0 or ppl <= 0:
            raise ValueError(f"{row_context}: entropy/PPL is outside its valid range")
        if row.get("evaluation_type") != "sccfg_sweep":
            raise ValueError(f"{row_context}: expected evaluation_type=sccfg_sweep")
        if row.get("policy_source") != policy_source or row.get("outer_step") != outer_step:
            raise ValueError(f"{row_context}: policy provenance does not match manifest")
        if not math.isclose(
            _finite_float(row, "external_cfg_scale", row_context),
            expected_external_cfg,
        ):
            raise ValueError(f"{row_context}: external CFG does not match manifest")
        if num_samples != expected_num_samples or seed != expected_seed:
            raise ValueError(f"{row_context}: sample count or seed does not match manifest")
        points_by_weight[weight] = SweepPoint(
            name=str(row.get("name", f"w{weight:g}")),
            weight=weight,
            entropy=entropy,
            ppl=ppl,
            num_samples=num_samples,
            seed=seed,
        )

    points = tuple(points_by_weight[weight] for weight in expected_scales)
    if policy_complete.get("num_results") != len(points):
        raise ValueError(f"{policy_complete_path}: num_results does not match results")
    compact_results = [
        {
            "name": point.name,
            "model_cfg_scale": point.weight,
            "entropy": point.entropy,
            "ppl": point.ppl,
        }
        for point in points
    ]
    if policy_complete.get("results") != compact_results:
        raise ValueError(f"{policy_complete_path}: result summary does not match JSONL")
    checkpoint_value = manifest.get("checkpoint_path")
    checkpoint_path = (
        Path(str(checkpoint_value)).resolve() if checkpoint_value is not None else None
    )
    return (
        PolicyCurve(
            policy_id=policy_id,
            display_name=display_name,
            policy_source=policy_source,
            outer_step=outer_step,
            results_path=results_path.resolve(),
            checkpoint_path=checkpoint_path,
            source_dir=policy_dir.resolve(),
            points=points,
        ),
        evaluation,
    )


def load_completed_sweep(suite_dir: Path) -> CompletedSweep:
    """Load a complete suite and reject partial or incomparable policy results."""

    suite_dir = suite_dir.resolve()
    suite_complete_path = suite_dir / ".complete"
    if not suite_complete_path.is_file():
        raise ValueError(f"suite is not complete: {suite_complete_path}")
    suite_complete = _load_json(suite_complete_path)
    if suite_complete.get("formal_evaluation") is not True:
        raise ValueError(f"{suite_complete_path}: expected formal_evaluation=true")
    if suite_complete.get("native_sc_cfg_only") is not True:
        raise ValueError(f"{suite_complete_path}: expected native_sc_cfg_only=true")
    cfg_scales = _ordered_unique_scales(
        suite_complete.get("cfg_scales"), str(suite_complete_path)
    )
    model_records = suite_complete.get("models")
    if not isinstance(model_records, list) or not model_records:
        raise ValueError(f"{suite_complete_path}: models must be a non-empty list")
    if len(model_records) > len(MODEL_COLORS):
        raise ValueError(
            f"the plot supports at most {len(MODEL_COLORS)} uniquely colored models"
        )
    if suite_complete.get("num_models") != len(model_records):
        raise ValueError(f"{suite_complete_path}: num_models does not match models")
    if suite_complete.get("results_per_model") != len(cfg_scales):
        raise ValueError(f"{suite_complete_path}: results_per_model mismatch")
    if suite_complete.get("total_results") != len(model_records) * len(cfg_scales):
        raise ValueError(f"{suite_complete_path}: total_results mismatch")

    curves: list[PolicyCurve] = []
    common_evaluation: dict[str, Any] | None = None
    seen_policy_ids: set[str] = set()
    for model_record in model_records:
        if not isinstance(model_record, dict):
            raise ValueError(f"{suite_complete_path}: each model must be an object")
        policy_id = str(model_record.get("policy_id", ""))
        if policy_id in seen_policy_ids:
            raise ValueError(f"{suite_complete_path}: duplicate policy_id={policy_id!r}")
        curve, evaluation = _load_policy_curve(
            suite_dir,
            model_record,
            cfg_scales,
            common_evaluation,
        )
        if common_evaluation is None:
            common_evaluation = evaluation
        curves.append(curve)
        seen_policy_ids.add(policy_id)
    assert common_evaluation is not None
    return CompletedSweep(
        suite_dir=suite_dir,
        cfg_scales=cfg_scales,
        evaluation=common_evaluation,
        curves=tuple(curves),
    )


def _training_run_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    metadata_path = run_dir / ".elf_nft_checkpoint.json"
    metadata = _load_json(metadata_path)
    try:
        checkpoint_path = Path(str(metadata["checkpoint_path"])).resolve()
        outer_step = int(metadata["outer_step"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{metadata_path}: invalid checkpoint identity") from error
    if outer_step < 0 or checkpoint_path.parent != run_dir:
        raise ValueError(f"{metadata_path}: checkpoint does not belong to the run")
    if not checkpoint_path.is_file():
        raise ValueError(f"missing training checkpoint: {checkpoint_path}")
    return checkpoint_path, metadata


def _training_run_label(nft_config: dict[str, Any], outer_step: int) -> str:
    """Build the center/branch/training-rollout label used by the plot legend."""

    try:
        center = CENTER_LABELS[str(nft_config["method"])]
        branch = BRANCH_LABELS[str(nft_config["self_condition_mode"])]
        rollout_scale = float(nft_config["rollout_self_cond_cfg_scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("effective config has an unsupported center/branch/rollout") from error
    if not math.isfinite(rollout_scale) or rollout_scale < 0:
        raise ValueError("effective config rollout_self_cond_cfg_scale is invalid")
    rollout = f"CFG{rollout_scale:g}"
    if nft_config.get("rollout_field_mode") == "recovered_base":
        rollout = f"CFG{float(nft_config['rollout_cfg_scale']):g} construct"
    return f"{center} / {branch} / {rollout} (step {outer_step})"


def _validate_training_run_settings(
    run_dir: Path,
    nft_config: dict[str, Any],
    expected_evaluation: dict[str, Any],
) -> None:
    settings = (
        ("rollout_sampling_method", "sampling_method", str),
        ("rollout_steps", "num_sampling_steps", int),
        ("rollout_sde_gamma", "sde_gamma", float),
        ("rollout_time_schedule", "time_schedule", str),
        ("rollout_cfg_scale", "input_cfg_scale", float),
        ("rollout_generation_batch_size", "generation_batch_size", int),
        ("evaluation_num_samples", "num_samples", int),
    )
    for config_key, evaluation_key, converter in settings:
        try:
            actual = converter(nft_config[config_key])
            expected = converter(expected_evaluation[evaluation_key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{run_dir}/effective_config.json: missing or invalid {config_key}"
            ) from error
        if actual != expected:
            raise ValueError(
                f"{run_dir}: {config_key}={actual!r} is not comparable to "
                f"the base sweep value {expected!r}"
            )


def _load_training_run_curve(
    run_dir: Path,
    expected_scales: tuple[float, ...],
    expected_evaluation: dict[str, Any],
) -> PolicyCurve:
    """Load final native-CFG points from one completed formal training run."""

    run_dir = run_dir.resolve()
    completion_path = run_dir / ".complete"
    completion = _load_json(completion_path)
    if completion.get("formal_experiment") is not True:
        raise ValueError(f"{completion_path}: expected formal_experiment=true")
    checkpoint_path, checkpoint_metadata = _training_run_checkpoint(run_dir)
    effective_path = run_dir / "effective_config.json"
    effective = _load_json(effective_path)
    nft_config = effective.get("nft")
    if not isinstance(nft_config, dict):
        raise ValueError(f"{effective_path}: missing nft config")
    if nft_config.get("formal_experiment") is not True:
        raise ValueError(f"{effective_path}: expected formal_experiment=true")
    if Path(str(nft_config.get("output_dir", ""))).resolve() != run_dir:
        raise ValueError(f"{effective_path}: output_dir does not match the run")
    _validate_training_run_settings(run_dir, nft_config, expected_evaluation)

    try:
        outer_step = int(checkpoint_metadata["outer_step"])
        method = str(checkpoint_metadata["method"])
        self_condition_mode = str(checkpoint_metadata["self_condition_mode"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{run_dir}: invalid checkpoint training metadata") from error
    for key, expected in (
        ("method", method),
        ("self_condition_mode", self_condition_mode),
        ("max_outer_steps", outer_step),
    ):
        if nft_config.get(key) != expected:
            raise ValueError(f"{effective_path}: {key} does not match checkpoint metadata")
    if completion.get("outer_steps") != outer_step:
        raise ValueError(f"{completion_path}: outer_steps does not match checkpoint")

    cases = nft_config.get("evaluation_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{effective_path}: evaluation_cases must be non-empty")
    native_cases: dict[float, str] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"{effective_path}: each evaluation case must be an object")
        external_cfg = _finite_float(case, "external_cfg_scale", str(effective_path))
        if not math.isclose(external_cfg, expected_evaluation["external_cfg_scale"]):
            continue
        weight = _finite_float(case, "model_cfg_scale", str(effective_path))
        if weight not in expected_scales:
            raise ValueError(f"{effective_path}: unexpected native weight {weight:g}")
        if weight in native_cases:
            raise ValueError(f"{effective_path}: duplicate native weight {weight:g}")
        native_cases[weight] = str(case.get("name", f"w{weight:g}"))
    if not native_cases:
        raise ValueError(f"{effective_path}: no native external-CFG=1 evaluation cases")

    results_path = run_dir / "sccfg_evaluations.jsonl"
    rows = _load_jsonl(results_path)
    final_rows: dict[float, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        row_context = f"{results_path}:{row_number}"
        if (
            row.get("trigger") != "after_training"
            or row.get("policy_source") != "current_policy_checkpoint"
            or row.get("outer_step") != outer_step
        ):
            continue
        external_cfg = _finite_float(row, "external_cfg_scale", row_context)
        if not math.isclose(external_cfg, expected_evaluation["external_cfg_scale"]):
            continue
        weight = _finite_float(row, "model_cfg_scale", row_context)
        if weight not in native_cases:
            raise ValueError(f"{row_context}: unexpected completed native weight {weight:g}")
        if weight in final_rows:
            raise ValueError(f"{row_context}: duplicate completed native weight {weight:g}")
        if row.get("evaluation_type") != "sccfg_sweep":
            raise ValueError(f"{row_context}: expected evaluation_type=sccfg_sweep")
        if str(row.get("name")) != native_cases[weight]:
            raise ValueError(f"{row_context}: case name does not match effective config")
        if _positive_int(row, "num_samples", row_context) != int(
            expected_evaluation["num_samples"]
        ):
            raise ValueError(f"{row_context}: sample count does not match base sweep")
        if _nonnegative_int(row, "seed", row_context) != int(expected_evaluation["seed"]):
            raise ValueError(f"{row_context}: seed does not match base sweep")
        if _positive_int(row, "generation_batch_size", row_context) != int(
            expected_evaluation["generation_batch_size"]
        ):
            raise ValueError(f"{row_context}: generation batch size does not match")
        final_rows[weight] = row
    if set(final_rows) != set(native_cases):
        missing = sorted(set(native_cases) - set(final_rows))
        raise ValueError(f"{results_path}: missing completed native weights {missing}")

    points = tuple(
        SweepPoint(
            name=native_cases[weight],
            weight=weight,
            entropy=_finite_float(final_rows[weight], "entropy", str(results_path)),
            ppl=_finite_float(final_rows[weight], "ppl", str(results_path)),
            num_samples=int(expected_evaluation["num_samples"]),
            seed=int(expected_evaluation["seed"]),
        )
        for weight in sorted(native_cases)
    )
    if any(point.entropy < 0 or point.ppl <= 0 for point in points):
        raise ValueError(f"{results_path}: entropy/PPL is outside its valid range")
    display_name = _training_run_label(nft_config, outer_step)
    return PolicyCurve(
        policy_id=run_dir.name,
        display_name=display_name,
        policy_source="current_policy_checkpoint",
        outer_step=outer_step,
        results_path=results_path.resolve(),
        checkpoint_path=checkpoint_path,
        source_dir=run_dir,
        points=points,
    )


def add_completed_training_runs(
    sweep: CompletedSweep, run_dirs: list[Path]
) -> CompletedSweep:
    """Add unique completed training runs, skipping checkpoints already swept."""

    curves = list(sweep.curves)
    checkpoint_paths = {
        curve.checkpoint_path for curve in curves if curve.checkpoint_path is not None
    }
    added_run_dirs: list[Path] = []
    seen_run_dirs: set[Path] = set()
    for raw_run_dir in run_dirs:
        run_dir = raw_run_dir.resolve()
        if run_dir in seen_run_dirs:
            continue
        seen_run_dirs.add(run_dir)
        checkpoint_path, _ = _training_run_checkpoint(run_dir)
        if checkpoint_path in checkpoint_paths:
            continue
        curve = _load_training_run_curve(
            run_dir, sweep.cfg_scales, sweep.evaluation
        )
        curves.append(curve)
        checkpoint_paths.add(checkpoint_path)
        added_run_dirs.append(run_dir)
    if len(curves) > len(MODEL_COLORS):
        raise ValueError(
            f"the plot supports at most {len(MODEL_COLORS)} uniquely colored models"
        )
    return replace(
        sweep,
        curves=tuple(curves),
        additional_run_dirs=tuple(added_run_dirs),
    )


def _load_completed_evaluation_curve(
    evaluation_dir: Path,
    expected_scales: tuple[float, ...],
    expected_evaluation: dict[str, Any],
) -> PolicyCurve:
    """Load one standalone completed sweep with full checkpoint provenance."""

    evaluation_dir = evaluation_dir.resolve()
    manifest = _load_json(evaluation_dir / "evaluation_manifest.json")
    model_record = {
        key: manifest.get(key)
        for key in ("policy_id", "display_name", "policy_source", "outer_step")
    }
    curve, evaluation = _load_policy_curve(
        evaluation_dir.parent,
        model_record,
        expected_scales,
        expected_evaluation,
        policy_dir=evaluation_dir,
    )
    if evaluation != expected_evaluation:
        raise ValueError(f"{evaluation_dir}: evaluation settings differ from base sweep")
    if curve.checkpoint_path is None or not curve.checkpoint_path.is_file():
        raise ValueError(f"{evaluation_dir}: missing evaluated checkpoint")
    return curve


def add_completed_evaluations(
    sweep: CompletedSweep, evaluation_dirs: list[Path]
) -> CompletedSweep:
    """Add full standalone sweeps, replacing partial curves for the same checkpoint."""

    curves = list(sweep.curves)
    added_evaluation_dirs: list[Path] = []
    seen_evaluation_dirs: set[Path] = set()
    for raw_evaluation_dir in evaluation_dirs:
        evaluation_dir = raw_evaluation_dir.resolve()
        if evaluation_dir in seen_evaluation_dirs:
            continue
        seen_evaluation_dirs.add(evaluation_dir)
        curve = _load_completed_evaluation_curve(
            evaluation_dir, sweep.cfg_scales, sweep.evaluation
        )
        matching_indices = [
            index
            for index, existing in enumerate(curves)
            if existing.checkpoint_path == curve.checkpoint_path
        ]
        if len(matching_indices) > 1:
            raise ValueError(
                f"multiple plotted curves use checkpoint {curve.checkpoint_path}"
            )
        if matching_indices:
            curves[matching_indices[0]] = curve
        else:
            curves.append(curve)
        added_evaluation_dirs.append(evaluation_dir)
    if len(curves) > len(MODEL_COLORS):
        raise ValueError(
            f"the plot supports at most {len(MODEL_COLORS)} uniquely colored models"
        )
    return replace(
        sweep,
        curves=tuple(curves),
        additional_evaluation_dirs=tuple(added_evaluation_dirs),
    )


def _format_weight(weight: float) -> str:
    return f"{weight:g}"


def _plot_label(curve: PolicyCurve) -> str:
    """Return an explicit center/branch/training-rollout legend label."""

    return PLOT_LABELS_BY_POLICY_ID.get(curve.policy_id, curve.display_name)


def _source_caption(sweep: CompletedSweep) -> str:
    if not sweep.additional_run_dirs and not sweep.additional_evaluation_dirs:
        return f"Source: {sweep.suite_dir.name}."
    extras: list[str] = []
    if sweep.additional_run_dirs:
        extras.append(f"{len(sweep.additional_run_dirs)} completed training runs")
    if sweep.additional_evaluation_dirs:
        extras.append(f"{len(sweep.additional_evaluation_dirs)} standalone sweeps")
    return f"Sources: {sweep.suite_dir.name} + {' + '.join(extras)}."


def _padded_limits(values: list[float], fraction: float) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = high - low
    padding = max(span * fraction, max(abs(low), abs(high), 1.0) * 0.005)
    return low - padding, high + padding


def plot_sweep(sweep: CompletedSweep, output_path: Path, dpi: int) -> None:
    """Render PPL against entropy, labeling every policy node by its CFG weight."""

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelcolor": "#334155",
            "text.color": "#0f172a",
        }
    )
    fig, ax = plt.subplots(figsize=(13.6, 8.4))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.14, top=0.76)
    all_entropy = [point.entropy for curve in sweep.curves for point in curve.points]
    all_ppl = [point.ppl for curve in sweep.curves for point in curve.points]

    for color, curve in zip(MODEL_COLORS, sweep.curves):
        entropy = [point.entropy for point in curve.points]
        ppl = [point.ppl for point in curve.points]
        ax.plot(entropy, ppl, color=color, linewidth=2.0, alpha=0.88, zorder=2)
        ax.scatter(
            entropy,
            ppl,
            color=color,
            s=58,
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )
        for index, point in enumerate(curve.points):
            offset_x, offset_y = ANNOTATION_OFFSETS[index % len(ANNOTATION_OFFSETS)]
            ax.annotate(
                _format_weight(point.weight),
                (point.entropy, point.ppl),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha="right" if offset_x < 0 else "left",
                va="bottom" if offset_y > 0 else "top",
                fontsize=8.0,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.7},
                zorder=4,
            )

    ax.set_xlim(*_padded_limits(all_entropy, 0.055))
    ax.set_ylim(*_padded_limits(all_ppl, 0.055))
    ax.set_xlabel("Mean unigram entropy H (nats)  → higher diversity", fontsize=11)
    ax.set_ylabel("GPT-2 Large generative perplexity  → lower is better", fontsize=11)
    ax.grid(color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#94a3b8")
    ax.tick_params(colors="#475569")

    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker="o",
            markeredgecolor="white",
            linewidth=2.0,
            markersize=7,
            label=_plot_label(curve),
        )
        for color, curve in zip(MODEL_COLORS, sweep.curves)
    ]
    fig.suptitle(
        "All completed native SC-CFG runs: PPL–entropy trade-off",
        x=0.5,
        y=0.965,
        fontsize=17,
        weight="bold",
    )
    evaluation = sweep.evaluation
    fig.text(
        0.5,
        0.91,
        (
            f"ELF-B / OWT   |   {evaluation['num_samples']:,} samples per point   |   "
            f"{evaluation['sampling_method'].upper()}-{evaluation['num_sampling_steps']}, "
            f"γ={evaluation['sde_gamma']:g}   |   common seed {evaluation['seed']}"
        ),
        ha="center",
        fontsize=10.2,
        color="#475569",
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.4,
        handlelength=2.1,
    )
    fig.text(
        0.5,
        0.035,
        (
            "Legend: center / branch / training rollout; node labels are native "
            "evaluation weights w; "
            f"external CFG={evaluation['external_cfg_scale']:g}. "
            f"{_source_caption(sweep)}"
        ),
        ha="center",
        fontsize=9,
        color="#64748b",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_snapshot(sweep: CompletedSweep, output_dir: Path) -> tuple[Path, Path]:
    """Write exactly the plotted scalars and a compact provenance summary."""

    csv_path = output_dir / "completed_cfg_sweep_points.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "policy_id",
                "display_name",
                "plot_label",
                "policy_source",
                "outer_step",
                "checkpoint_path",
                "source_dir",
                "w",
                "entropy",
                "ppl",
                "num_samples",
                "seed",
                "source_results",
            )
        )
        for curve in sweep.curves:
            for point in curve.points:
                writer.writerow(
                    (
                        curve.policy_id,
                        curve.display_name,
                        _plot_label(curve),
                        curve.policy_source,
                        curve.outer_step,
                        curve.checkpoint_path or "",
                        curve.source_dir,
                        point.weight,
                        point.entropy,
                        point.ppl,
                        point.num_samples,
                        point.seed,
                        curve.results_path,
                    )
                )

    summary_path = output_dir / "completed_cfg_sweep_summary.json"
    summary = {
        "source_suite": str(sweep.suite_dir),
        "completion_marker": str(sweep.suite_dir / ".complete"),
        "additional_run_dirs": [str(path) for path in sweep.additional_run_dirs],
        "additional_evaluation_dirs": [
            str(path) for path in sweep.additional_evaluation_dirs
        ],
        "num_models": len(sweep.curves),
        "num_points": sum(len(curve.points) for curve in sweep.curves),
        "cfg_scales": list(sweep.cfg_scales),
        "common_evaluation": sweep.evaluation,
        "models": [
            {
                "policy_id": curve.policy_id,
                "display_name": curve.display_name,
                "plot_label": _plot_label(curve),
                "policy_source": curve.policy_source,
                "outer_step": curve.outer_step,
                "checkpoint_path": (
                    str(curve.checkpoint_path)
                    if curve.checkpoint_path is not None
                    else None
                ),
                "source_dir": str(curve.source_dir),
                "results_path": str(curve.results_path),
            }
            for curve in sweep.curves
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return csv_path, summary_path


def _validate_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    asset_root = ASSET_ROOT.resolve()
    if output_dir == asset_root or not output_dir.is_relative_to(asset_root):
        raise ValueError(f"output-dir must be a subdirectory of {asset_root}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path, help="completed CFG sweep suite")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        type=Path,
        help=(
            "completed formal training run to add; may be repeated; checkpoints "
            "already represented by suite_dir are skipped"
        ),
    )
    parser.add_argument(
        "--evaluation-dir",
        action="append",
        default=[],
        type=Path,
        help=(
            "completed standalone CFG sweep to add; may be repeated; a full "
            "sweep replaces a partial training-run curve for the same checkpoint"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if not 100 <= args.dpi <= 600:
        parser.error("dpi must lie in [100, 600]")
    return args


def main() -> None:
    args = parse_args()
    try:
        output_dir = _validate_output_dir(args.output_dir)
        sweep = load_completed_sweep(args.suite_dir)
        sweep = add_completed_training_runs(sweep, args.run_dir)
        sweep = add_completed_evaluations(sweep, args.evaluation_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "completed_cfg_sweep_ppl_entropy.png"
    pdf_path = output_dir / "completed_cfg_sweep_ppl_entropy.pdf"
    plot_sweep(sweep, png_path, args.dpi)
    plot_sweep(sweep, pdf_path, args.dpi)
    csv_path, summary_path = write_snapshot(sweep, output_dir)
    print(
        json.dumps(
            {
                "source_suite": str(sweep.suite_dir),
                "additional_run_dirs": [
                    str(path) for path in sweep.additional_run_dirs
                ],
                "additional_evaluation_dirs": [
                    str(path) for path in sweep.additional_evaluation_dirs
                ],
                "models": len(sweep.curves),
                "points": sum(len(curve.points) for curve in sweep.curves),
                "png": str(png_path),
                "pdf": str(pdf_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
