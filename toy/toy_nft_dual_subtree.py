"""Evaluate independent A64/AB32 NFT on an equal-value dual-subtree reward."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_example import do_sample, gt, tree_segments
from toy_nft_guidance import (
    ToyNFTConfig,
    branch_distance_reward_formula,
    distance_to_segments,
    load_model_bundle,
    reward_from_branch_distance,
    reward_branch_selection_specs,
    select_branch_segments,
)


TRAINED_METHODS = ("a64_nft", "ab32_nft")
METHODS = (
    "a64_nft",
    "a64_self_guided",
    "ab32_nft",
    "ab32_self_guided",
)
METHOD_LABELS = {
    "a64_nft": "A64-NFT",
    "a64_self_guided": "A64 self-guided",
    "ab32_nft": "AB32-NFT",
    "ab32_self_guided": "AB32 self-guided",
}
METHOD_COLORS = {
    "a64_nft": "#D55E00",
    "a64_self_guided": "#A2142F",
    "ab32_nft": "#CC79A7",
    "ab32_self_guided": "#6F4C9B",
}
TARGET_COLORS = {"A": "#205B73", "B": "#665191"}
TREE_BACKGROUND_COLORS = {"A": "#C8A77F", "B": "#D8CFE3"}
TREE_BACKGROUND_LINE_WIDTH = 0.8
TREE_BACKGROUND_ALPHA = 0.76


@dataclass(frozen=True)
class DualSubtreeConfig:
    schema_version: int = 1
    name: str = "toy-nft-dual-subtree-a64-ab32-10step"
    seed: int = 20260916
    outer_steps: int = 10
    a_branch_class: str = "A"
    a_branch_path: str = "LRL"
    a_branch_levels: int | None = None
    b_branch_class: str = "B"
    b_branch_path: str = "RRL"
    b_branch_levels: int | None = None
    initial_classes: str = "AB"
    evaluation_samples: int = 2048
    evaluation_steps: int = 32
    sigma_min: float = 0.002
    sigma_max: float = 5.0
    rho: float = 7.0
    near_branch_threshold: float = 0.04
    reward: str = "one_minus_branch_distance"
    reward_distance_scale: float = 1.0
    overview_samples: int = 768
    self_guidance_delta_scale: float = 3.0

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "DualSubtreeConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("dual-subtree config must be a JSON object")
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - fields)
        if unknown:
            raise ValueError(f"unknown dual-subtree config keys: {unknown}")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not self.name:
            raise ValueError("name must be nonempty")
        if (self.a_branch_class, self.b_branch_class) != ("A", "B"):
            raise ValueError("dual targets must use an A primary and B secondary")
        a_segments = select_branch_segments(
            self.a_branch_class, self.a_branch_path, self.a_branch_levels
        )
        b_segments = select_branch_segments(
            self.b_branch_class, self.b_branch_path, self.b_branch_levels
        )
        if len(a_segments) != len(b_segments):
            raise ValueError("A and B reward subtrees must have equal topology size")
        if len(self.a_branch_path) != len(self.b_branch_path):
            raise ValueError("A and B reward subtrees must start at equal depth")
        if self.initial_classes not in {"A", "B", "AB"}:
            raise ValueError("initial_classes must be A, B, or AB")
        if self.reward not in {
            "one_minus_branch_distance",
            "exponential_branch_distance",
        }:
            raise ValueError(
                "dual-subtree reward must be one_minus_branch_distance or "
                "exponential_branch_distance"
            )
        for name, value in {
            "outer_steps": self.outer_steps,
            "evaluation_samples": self.evaluation_samples,
            "evaluation_steps": self.evaluation_steps,
            "overview_samples": self.overview_samples,
        }.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.evaluation_steps < 2:
            raise ValueError("evaluation_steps must be at least two")
        if self.overview_samples > self.evaluation_samples:
            raise ValueError("overview_samples cannot exceed evaluation_samples")
        for name, value in {
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "rho": self.rho,
            "near_branch_threshold": self.near_branch_threshold,
            "reward_distance_scale": self.reward_distance_scale,
            "self_guidance_delta_scale": self.self_guidance_delta_scale,
        }.items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.sigma_min >= self.sigma_max:
            raise ValueError("sigma_min must be smaller than sigma_max")


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _method_label(method: str, config: DualSubtreeConfig) -> str:
    label = METHOD_LABELS[method]
    if method.endswith("_self_guided"):
        label += f" (delta x{config.self_guidance_delta_scale:g})"
    return label


def _frozen_clone(
    base: torch.nn.Module, state: dict[str, torch.Tensor] | None = None
) -> torch.nn.Module:
    model = copy.deepcopy(base).eval().requires_grad_(False)
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def _checkpoint_payload(
    run_dir: Path, config: ToyNFTConfig, step: int, device: torch.device
) -> dict[str, Any] | None:
    if step == 0:
        return None
    path = run_dir / "checkpoints" / f"step-{step:06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing step checkpoint: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported checkpoint schema: {path}")
    if int(payload.get("completed_outer_step", -1)) != step:
        raise ValueError(f"checkpoint step mismatch: {path}")
    checkpoint_config = ToyNFTConfig(**payload.get("config", {}))
    checkpoint_config.validate()
    if dataclasses.asdict(checkpoint_config) != dataclasses.asdict(config):
        raise ValueError(f"checkpoint/config mismatch: {path}")
    return payload


def _validate_training_config(
    config: ToyNFTConfig,
    matrix: DualSubtreeConfig,
    *,
    sampling_classes: str,
) -> None:
    if config.objective_mode != "good_only":
        raise ValueError("dual-subtree runs must independently train one model")
    if config.sampling_classes != sampling_classes:
        raise ValueError(
            f"expected sampling_classes={sampling_classes}, got {config.sampling_classes}"
        )
    if config.reward != matrix.reward:
        raise ValueError(
            f"training reward {config.reward!r} does not match "
            f"evaluation reward {matrix.reward!r}"
        )
    if not math.isclose(
        float(config.reward_distance_scale),
        float(matrix.reward_distance_scale),
    ):
        raise ValueError("training and evaluation reward_distance_scale disagree")
    if reward_branch_selection_specs(config) != (
        (matrix.a_branch_class, matrix.a_branch_path, matrix.a_branch_levels),
        (matrix.b_branch_class, matrix.b_branch_path, matrix.b_branch_levels),
    ):
        raise ValueError("training and evaluation reward subtrees disagree")
    if config.outer_steps != matrix.outer_steps:
        raise ValueError("training and evaluation outer_steps disagree")
    if config.evaluation_steps != matrix.evaluation_steps:
        raise ValueError("training and evaluation sampler steps disagree")
    for name in ("sigma_min", "sigma_max", "rho", "near_branch_threshold"):
        if not math.isclose(float(getattr(config, name)), float(getattr(matrix, name))):
            raise ValueError(f"training and evaluation {name} disagree")


def _sample(
    model: torch.nn.Module,
    initial: torch.Tensor,
    config: DualSubtreeConfig,
    *,
    reference: torch.nn.Module | None = None,
) -> torch.Tensor:
    # Official ToyModel.score() needs input autograd during inference.
    # do_sample evaluates reference + w * (model - reference).  Setting
    # w=1+delta therefore gives model + delta * (model - reference).
    sampler_guidance = (
        1.0 + config.self_guidance_delta_scale if reference is not None else 1.0
    )
    return do_sample(
        net=model,
        gnet=reference,
        guidance=sampler_guidance,
        x_init=initial,
        num_steps=config.evaluation_steps,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
    )[-1].detach()


def _segment_tensors(
    branch_class: str,
    branch_path: str,
    device: torch.device,
    dtype: torch.dtype,
    branch_levels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    segments = select_branch_segments(branch_class, branch_path, branch_levels)
    starts = torch.tensor(
        [segment.start for segment in segments], device=device, dtype=dtype
    )
    ends = torch.tensor(
        [segment.end for segment in segments], device=device, dtype=dtype
    )
    return starts, ends


def _full_tree_tensors(
    classes: str, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    segments = tree_segments(classes)
    starts = torch.tensor(
        [segment.start for segment in segments], device=device, dtype=dtype
    )
    ends = torch.tensor(
        [segment.end for segment in segments], device=device, dtype=dtype
    )
    return starts, ends


def collect_dual_subtree_data(
    matrix_root: Path,
    config: DualSubtreeConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Sample A64/AB32 checkpoints with common noise and return paired arrays."""

    training_root = matrix_root / "training"
    run_dirs = {
        "a64_nft": training_root / "a64_nft",
        "ab32_nft": training_root / "ab32_nft",
    }
    for run_dir in run_dirs.values():
        if not (run_dir / "checkpoint.pt").is_file():
            raise FileNotFoundError(f"incomplete training run: {run_dir}")
    training_configs = {
        method: ToyNFTConfig.from_json(run_dir / "config.json")
        for method, run_dir in run_dirs.items()
    }
    _validate_training_config(
        training_configs["a64_nft"], config, sampling_classes="A"
    )
    _validate_training_config(
        training_configs["ab32_nft"], config, sampling_classes="AB"
    )
    bundles = {
        method: load_model_bundle(training_configs[method], device)
        for method in TRAINED_METHODS
    }
    initial_models = {
        method: _frozen_clone(bundles[method].good_ema)
        for method in TRAINED_METHODS
    }

    generator = torch.Generator(device=device).manual_seed(config.seed + 70_000_003)
    initial = gt(config.initial_classes, device).sample(
        config.evaluation_samples,
        config.sigma_max,
        generator=generator,
    )
    a_starts, a_ends = _segment_tensors(
        config.a_branch_class,
        config.a_branch_path,
        device,
        initial.dtype,
        config.a_branch_levels,
    )
    b_starts, b_ends = _segment_tensors(
        config.b_branch_class,
        config.b_branch_path,
        device,
        initial.dtype,
        config.b_branch_levels,
    )
    full_a_starts, full_a_ends = _full_tree_tensors("A", device, initial.dtype)
    full_b_starts, full_b_ends = _full_tree_tensors("B", device, initial.dtype)
    arrays: dict[str, list[np.ndarray]] = {
        "endpoints": [],
        "union_distance": [],
        "a_target_distance": [],
        "b_target_distance": [],
        "target_a_assignment": [],
        "full_tree_a_assignment": [],
    }

    for step in range(config.outer_steps + 1):
        step_endpoints = []
        for method in TRAINED_METHODS:
            payload = _checkpoint_payload(
                run_dirs[method], training_configs[method], step, device
            )
            model = _frozen_clone(
                bundles[method].good_ema,
                None if payload is None else payload["models"]["good_ema"],
            )
            base_endpoints = _sample(model, initial, config)
            guided_endpoints = _sample(
                model,
                initial,
                config,
                reference=initial_models[method],
            )
            if step == 0 and not torch.equal(base_endpoints, guided_endpoints):
                max_error = float((base_endpoints - guided_endpoints).abs().max())
                raise RuntimeError(
                    f"{method} step-0 self-guidance must equal its base output; "
                    f"max_error={max_error}"
                )
            step_endpoints.extend((base_endpoints, guided_endpoints))
            del model
        endpoints = torch.stack(step_endpoints)
        flat = endpoints.reshape(-1, 2)
        shape = (len(METHODS), config.evaluation_samples)
        a_distance = distance_to_segments(flat, a_starts, a_ends).reshape(shape)
        b_distance = distance_to_segments(flat, b_starts, b_ends).reshape(shape)
        union_distance = torch.minimum(a_distance, b_distance)
        full_a_distance = distance_to_segments(
            flat, full_a_starts, full_a_ends
        ).reshape(shape)
        full_b_distance = distance_to_segments(
            flat, full_b_starts, full_b_ends
        ).reshape(shape)
        arrays["endpoints"].append(endpoints.cpu().numpy())
        arrays["union_distance"].append(union_distance.cpu().numpy())
        arrays["a_target_distance"].append(a_distance.cpu().numpy())
        arrays["b_target_distance"].append(b_distance.cpu().numpy())
        arrays["target_a_assignment"].append(
            (a_distance <= b_distance).cpu().numpy()
        )
        arrays["full_tree_a_assignment"].append(
            (full_a_distance <= full_b_distance).cpu().numpy()
        )
        print(f"DUAL_SUBTREE_SAMPLE step={step}/{config.outer_steps}", flush=True)

    return {name: np.stack(values) for name, values in arrays.items()}


def _metric_rows(
    data: dict[str, np.ndarray], config: DualSubtreeConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in range(config.outer_steps + 1):
        for method_index, method in enumerate(METHODS):
            union = data["union_distance"][step, method_index]
            a_distance = data["a_target_distance"][step, method_index]
            b_distance = data["b_target_distance"][step, method_index]
            movement = (
                0.0
                if step == 0
                else float(
                    np.linalg.norm(
                        data["endpoints"][step, method_index]
                        - data["endpoints"][step - 1, method_index],
                        axis=1,
                    ).mean()
                )
            )
            rows.append(
                {
                    "outer_step": step,
                    "method": method,
                    "reward_mean": float(
                        reward_from_branch_distance(
                            union,
                            config.reward,
                            config.reward_distance_scale,
                        ).mean()
                    ),
                    "union_distance_mean": float(union.mean()),
                    "union_distance_median": float(np.median(union)),
                    "near_union_rate": float(
                        (union <= config.near_branch_threshold).mean()
                    ),
                    "near_a_target_rate": float(
                        (a_distance <= config.near_branch_threshold).mean()
                    ),
                    "near_b_target_rate": float(
                        (b_distance <= config.near_branch_threshold).mean()
                    ),
                    "target_a_assignment_rate": float(
                        data["target_a_assignment"][step, method_index].mean()
                    ),
                    "full_tree_a_assignment_rate": float(
                        data["full_tree_a_assignment"][step, method_index].mean()
                    ),
                    "mean_endpoint_change_from_previous_step": movement,
                }
            )
    return rows


def _tree_plot_limits() -> tuple[tuple[float, float], tuple[float, float]]:
    segments = tree_segments("AB")
    xs = [v for segment in segments for v in (segment.start[0], segment.end[0])]
    ys = [v for segment in segments for v in (segment.start[1], segment.end[1])]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    return (
        (min(xs) - 0.06 * x_span, max(xs) + 0.06 * x_span),
        (min(ys) - 0.06 * y_span, max(ys) + 0.06 * y_span),
    )


def _draw_tree(axis: plt.Axes, config: DualSubtreeConfig) -> None:
    target_specs = {
        (segment.cls, segment.path): branch_class
        for branch_class, branch_path, branch_levels in (
            (
                config.a_branch_class,
                config.a_branch_path,
                config.a_branch_levels,
            ),
            (
                config.b_branch_class,
                config.b_branch_path,
                config.b_branch_levels,
            ),
        )
        for segment in select_branch_segments(
            branch_class, branch_path, branch_levels
        )
    }
    for segment in tree_segments("AB"):
        target_class = target_specs.get((segment.cls, segment.path))
        is_target = target_class is not None
        axis.plot(
            [segment.start[0], segment.end[0]],
            [segment.start[1], segment.end[1]],
            color=(
                TARGET_COLORS[target_class]
                if is_target
                else TREE_BACKGROUND_COLORS[segment.cls]
            ),
            linewidth=(
                2.45 if is_target else TREE_BACKGROUND_LINE_WIDTH
            ),
            alpha=0.96 if is_target else TREE_BACKGROUND_ALPHA,
            zorder=2 if is_target else 1,
        )


def _style_axis(axis: plt.Axes) -> None:
    x_limits, y_limits = _tree_plot_limits()
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal")
    axis.set_facecolor("#FCFCFB")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(
        0.97,
        0.97,
        "A target = 1",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color=TARGET_COLORS["A"],
        fontsize=7,
        fontweight="semibold",
    )
    axis.text(
        0.03,
        0.03,
        "B target = 1",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=TARGET_COLORS["B"],
        fontsize=7,
        fontweight="semibold",
    )
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_step(
    step: int,
    data: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    config: DualSubtreeConfig,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(METHODS), figsize=(17.2, 4.7))
    step_rows = {row["method"]: row for row in rows if row["outer_step"] == step}
    for method_index, method in enumerate(METHODS):
        axis = axes[method_index]
        _draw_tree(axis, config)
        values = data["endpoints"][step, method_index]
        axis.scatter(
            values[:, 0],
            values[:, 1],
            s=8,
            color=METHOD_COLORS[method],
            alpha=0.62,
            edgecolors="white",
            linewidths=0.18,
            rasterized=True,
            zorder=3,
        )
        _style_axis(axis)
        row = step_rows[method]
        axis.set_title(
            f"{_method_label(method, config)}\n"
            f"d={row['union_distance_mean']:.3f}, "
            f"near A/B={row['near_a_target_rate']:.2f}/"
            f"{row['near_b_target_rate']:.2f}",
            fontsize=10,
        )
    figure.suptitle(
        f"Equal-value A/B subtree reward with step-0 self-guidance — outer step {step}",
        y=0.99,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_overview(
    data: dict[str, np.ndarray],
    config: DualSubtreeConfig,
    output_dir: Path,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(
        config.outer_steps + 1,
        len(METHODS),
        figsize=(17.5, 4.0 * (config.outer_steps + 1)),
        squeeze=False,
    )
    indices = np.linspace(
        0,
        config.evaluation_samples - 1,
        num=config.overview_samples,
        dtype=np.int64,
    )
    for step in range(config.outer_steps + 1):
        for method_index, method in enumerate(METHODS):
            axis = axes[step, method_index]
            _draw_tree(axis, config)
            values = data["endpoints"][step, method_index, indices]
            axis.scatter(
                values[:, 0],
                values[:, 1],
                s=8,
                color=METHOD_COLORS[method],
                alpha=0.68,
                edgecolors="white",
                linewidths=0.18,
                rasterized=True,
                zorder=3,
            )
            _style_axis(axis)
            if step == 0:
                axis.set_title(
                    _method_label(method, config),
                    fontsize=10,
                    fontweight="semibold",
                )
            if method_index == 0:
                axis.text(
                    -0.08,
                    0.5,
                    f"step {step}",
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="semibold",
                )
    figure.suptitle(
        "A64 and AB32 NFT with self-guidance from each model's step-0 reference",
        y=0.998,
        fontsize=14,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0.035, 0, 1, 0.983))
    png_path = output_dir / "dual_subtree_all_steps.png"
    pdf_path = output_dir / "dual_subtree_all_steps.pdf"
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def _plot_metrics(
    rows: list[dict[str, Any]], config: DualSubtreeConfig, output_dir: Path
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        steps = [row["outer_step"] for row in method_rows]
        color = METHOD_COLORS[method]
        label = _method_label(method, config)
        axes[0, 0].plot(
            steps,
            [row["union_distance_mean"] for row in method_rows],
            marker="o",
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            steps,
            [row["near_union_rate"] for row in method_rows],
            marker="o",
            color=color,
            label=f"{label}: either",
        )
        axes[0, 1].plot(
            steps,
            [row["near_a_target_rate"] for row in method_rows],
            color=color,
            linestyle="--",
            alpha=0.85,
            label=f"{label}: A",
        )
        axes[0, 1].plot(
            steps,
            [row["near_b_target_rate"] for row in method_rows],
            color=color,
            linestyle=":",
            alpha=0.85,
            label=f"{label}: B",
        )
        axes[1, 0].plot(
            steps,
            [row["target_a_assignment_rate"] for row in method_rows],
            marker="o",
            color=color,
            label=f"{label}: nearest target",
        )
        axes[1, 0].plot(
            steps,
            [row["full_tree_a_assignment_rate"] for row in method_rows],
            color=color,
            linestyle="--",
            label=f"{label}: nearest full tree",
        )
        axes[1, 1].plot(
            steps,
            [
                row["mean_endpoint_change_from_previous_step"]
                for row in method_rows
            ],
            marker="o",
            color=color,
            label=label,
        )
    specs = (
        (axes[0, 0], "mean distance to either target ↓"),
        (axes[0, 1], "near-target rate ↑"),
        (axes[1, 0], "A-side assignment rate"),
        (axes[1, 1], "same-noise endpoint change"),
    )
    for axis, ylabel in specs:
        axis.set_xlabel("outer NFT step")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(len(method_rows)))
        axis.grid(alpha=0.22, linewidth=0.5)
        axis.legend(frameon=False, fontsize=7)
    png_path = output_dir / "dual_subtree_metrics.png"
    pdf_path = output_dir / "dual_subtree_metrics.pdf"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def render_dual_subtree(
    matrix_root: Path,
    config: DualSubtreeConfig,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "config.json", dataclasses.asdict(config))
    data = collect_dual_subtree_data(matrix_root, config, device)
    np.savez_compressed(
        output_dir / "dual_subtree_data.npz",
        **data,
        methods=np.asarray(METHODS),
        outer_steps=np.arange(config.outer_steps + 1),
    )
    rows = _metric_rows(data, config)
    metrics_path = output_dir / "dual_subtree_metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    step_dir = output_dir / "steps"
    step_dir.mkdir()
    for step in range(config.outer_steps + 1):
        _plot_step(
            step,
            data,
            rows,
            config,
            step_dir / f"step-{step:02d}.png",
        )
    overview_png, overview_pdf = _plot_overview(data, config, output_dir)
    metrics_png, metrics_pdf = _plot_metrics(rows, config, output_dir)
    final_rows = {
        row["method"]: row
        for row in rows
        if row["outer_step"] == config.outer_steps
    }
    initial_rows = {row["method"]: row for row in rows if row["outer_step"] == 0}
    summary = {
        "schema_version": 1,
        "formal_experiment": False,
        "name": config.name,
        "matrix_root": str(matrix_root.resolve()),
        "outer_steps": config.outer_steps,
        "methods": list(METHODS),
        "common_initial_noise": True,
        "self_guidance": {
            "formula": (
                f"D_current + {config.self_guidance_delta_scale:g} * "
                "(D_current - D_step0)"
            ),
            "delta_scale": config.self_guidance_delta_scale,
            "sampler_guidance": 1.0 + config.self_guidance_delta_scale,
            "reference": "each trained model's own frozen step-0 good_ema",
            "step0_base_guided_exactly_equal": True,
        },
        "reward": {
            "name": config.reward,
            "distance_scale": config.reward_distance_scale,
            "formula": branch_distance_reward_formula(
                config.reward,
                config.reward_distance_scale,
            ),
        },
        "targets": {
            "A": {
                "branch": f"{config.a_branch_class}:{config.a_branch_path}",
                "levels": config.a_branch_levels,
                "edge_ids": [
                    f"{segment.cls}:{segment.path}"
                    for segment in select_branch_segments(
                        config.a_branch_class,
                        config.a_branch_path,
                        config.a_branch_levels,
                    )
                ],
                "segments": len(
                    select_branch_segments(
                        config.a_branch_class,
                        config.a_branch_path,
                        config.a_branch_levels,
                    )
                ),
                "value_on_centerline": 1.0,
            },
            "B": {
                "branch": f"{config.b_branch_class}:{config.b_branch_path}",
                "levels": config.b_branch_levels,
                "edge_ids": [
                    f"{segment.cls}:{segment.path}"
                    for segment in select_branch_segments(
                        config.b_branch_class,
                        config.b_branch_path,
                        config.b_branch_levels,
                    )
                ],
                "segments": len(
                    select_branch_segments(
                        config.b_branch_class,
                        config.b_branch_path,
                        config.b_branch_levels,
                    )
                ),
                "value_on_centerline": 1.0,
            },
        },
        "initial_metrics": initial_rows,
        "final_metrics": final_rows,
        "artifacts": {
            "data": str(output_dir / "dual_subtree_data.npz"),
            "metrics_jsonl": str(metrics_path),
            "step_figures": str(step_dir / "step-00.png"),
            "overview_png": str(overview_png),
            "overview_pdf": str(overview_pdf),
            "metrics_png": str(metrics_png),
            "metrics_pdf": str(metrics_pdf),
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(
        f"DUAL_SUBTREE_CONFIRMED outer_steps={config.outer_steps} "
        f"methods={len(METHODS)} output_dir={output_dir}",
        flush=True,
    )
    return summary


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA was requested but is not available")
    return device


@click.command()
@click.option(
    "--matrix-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
)
@click.option("--device", type=str, default="cuda", show_default=True)
def main(
    matrix_root: Path,
    config_path: Path,
    output_dir: Path | None,
    device: str,
) -> None:
    """Render the A64/AB32 equal-value dual-subtree experiment."""

    config = DualSubtreeConfig.from_json(config_path)
    resolved_output = matrix_root / "evaluation" if output_dir is None else output_dir
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32))
    render_dual_subtree(
        matrix_root,
        config,
        resolved_output,
        _resolve_device(device),
    )


if __name__ == "__main__":
    main()
