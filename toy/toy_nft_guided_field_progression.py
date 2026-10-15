"""Render every checkpoint from joint w=3 A64/AB32 guided-field NFT."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import click
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_example import do_sample, gt, tree_segments
from toy_nft_dual_subtree import (
    DualSubtreeConfig,
    TREE_BACKGROUND_ALPHA,
    TREE_BACKGROUND_COLORS,
    TREE_BACKGROUND_LINE_WIDTH,
    _atomic_json,
    _checkpoint_payload,
    _frozen_clone,
    _full_tree_tensors,
    _segment_tensors,
)
from toy_nft_guidance import (
    ToyNFTConfig,
    branch_distance_reward_formula,
    distance_to_segments,
    load_model_bundle,
    reward_from_branch_distance,
    reward_branch_selection_specs,
    select_branch_segments,
)


METHODS = (
    "a64",
    "ab32",
    "a64_init_ab32_w3",
    "a64_trained_ab32_w3",
)
METHOD_LABELS = {
    "a64": "A64",
    "ab32": "AB32",
    "a64_init_ab32_w3": "A64 + initial AB32 (w=3)",
    "a64_trained_ab32_w3": "A64 + trained AB32 (w=3)",
}
METHOD_COLORS = {
    "a64": "#D55E00",
    "ab32": "#6F4C9B",
    "a64_init_ab32_w3": "#009E73",
    "a64_trained_ab32_w3": "#0072B2",
}
POINT_SIZE = 3.0
POINT_ALPHA = 0.78
OVERVIEW_SAMPLES = 768
TARGET_LINE_COLOR = "#5B9DB5"
TARGET_LINE_WIDTH = 1.25
TARGET_LINE_ALPHA = 0.9


def _validate_training_config(config: ToyNFTConfig) -> None:
    expected = {
        "objective_mode": "guided_field",
        "sampling_classes": "AB",
        "ema_decay": 0.0,
        "old_policy_decay": 0.0,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(
                f"guided-field progression requires {name}={value!r}, "
                f"got {getattr(config, name)!r}"
            )
    if config.guidance_scale != 3.0:
        raise ValueError("guided-field progression requires guidance_scale=3")
    if config.reward not in {
        "one_minus_branch_distance",
        "exponential_branch_distance",
    }:
        raise ValueError(
            "guided-field progression requires a unit-maximum dual-subtree reward"
        )
    target_specs = reward_branch_selection_specs(config)
    if len(target_specs) != 2 or tuple(spec[0] for spec in target_specs) != (
        "A",
        "B",
    ):
        raise ValueError(
            "guided-field progression requires one A target and one B target"
        )
    segment_counts = [
        len(select_branch_segments(*target_spec)) for target_spec in target_specs
    ]
    if segment_counts[0] != segment_counts[1]:
        raise ValueError("guided-field A/B targets must have equal topology size")


def _plot_config(config: ToyNFTConfig) -> DualSubtreeConfig:
    assert config.secondary_branch_class is not None
    assert config.secondary_branch_path is not None
    plot_config = DualSubtreeConfig(
        name=f"{config.name}-progression",
        seed=config.seed,
        outer_steps=config.outer_steps,
        a_branch_class=config.branch_class,
        a_branch_path=config.branch_path,
        a_branch_levels=config.branch_levels,
        b_branch_class=config.secondary_branch_class,
        b_branch_path=config.secondary_branch_path,
        b_branch_levels=config.secondary_branch_levels,
        initial_classes=config.sampling_classes or config.branch_class,
        evaluation_samples=config.evaluation_samples,
        evaluation_steps=config.evaluation_steps,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
        near_branch_threshold=config.near_branch_threshold,
        reward=config.reward,
        reward_distance_scale=config.reward_distance_scale,
        overview_samples=min(OVERVIEW_SAMPLES, config.evaluation_samples),
    )
    plot_config.validate()
    return plot_config


def _state_dicts_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _verify_snapshot_refresh(payload: dict[str, Any], step: int) -> None:
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"checkpoint step {step} has no model state dictionary")
    for current_name in ("good", "bad"):
        current = models[current_name]
        for suffix in ("old", "ema"):
            snapshot_name = f"{current_name}_{suffix}"
            if not _state_dicts_equal(current, models[snapshot_name]):
                raise ValueError(
                    f"checkpoint step {step} does not exactly refresh "
                    f"{snapshot_name} from {current_name}"
                )


def _sample(
    model: torch.nn.Module,
    initial: torch.Tensor,
    config: ToyNFTConfig,
    *,
    guide: torch.nn.Module | None = None,
) -> torch.Tensor:
    return do_sample(
        net=model,
        gnet=guide,
        guidance=config.guidance_scale if guide is not None else 1.0,
        x_init=initial,
        num_steps=config.evaluation_steps,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
    )[-1].detach()


def collect_guided_field_progression_data(
    run_dir: Path,
    config: ToyNFTConfig,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Sample component and composed fields at steps 0--10 with common noise."""

    _validate_training_config(config)
    if not (run_dir / "checkpoint.pt").is_file():
        raise FileNotFoundError(f"incomplete guided-field run: {run_dir}")
    bundle = load_model_bundle(config, device)
    for current_name in ("good", "bad"):
        current = getattr(bundle, current_name).state_dict()
        for suffix in ("old", "ema"):
            snapshot = getattr(bundle, f"{current_name}_{suffix}").state_dict()
            if not _state_dicts_equal(current, snapshot):
                raise RuntimeError(f"initial {current_name}_{suffix} is not synchronized")
    initial_bad = _frozen_clone(bundle.bad_ema)

    generator = torch.Generator(device=device).manual_seed(config.seed + 70_000_003)
    initial = gt("AB", device).sample(
        config.evaluation_samples,
        config.sigma_max,
        generator=generator,
    )
    a_starts, a_ends = _segment_tensors(
        config.branch_class,
        config.branch_path,
        device,
        initial.dtype,
        config.branch_levels,
    )
    assert config.secondary_branch_class is not None
    assert config.secondary_branch_path is not None
    b_starts, b_ends = _segment_tensors(
        config.secondary_branch_class,
        config.secondary_branch_path,
        device,
        initial.dtype,
        config.secondary_branch_levels,
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
    verified_steps: list[int] = []

    for step in range(config.outer_steps + 1):
        payload = _checkpoint_payload(run_dir, config, step, device)
        if payload is not None:
            _verify_snapshot_refresh(payload, step)
            verified_steps.append(step)
        good = _frozen_clone(
            bundle.good_ema,
            None if payload is None else payload["models"]["good_ema"],
        )
        bad = _frozen_clone(
            bundle.bad_ema,
            None if payload is None else payload["models"]["bad_ema"],
        )
        a64_endpoints = _sample(good, initial, config)
        ab32_endpoints = _sample(bad, initial, config)
        fixed_reference_endpoints = _sample(
            good, initial, config, guide=initial_bad
        )
        jointly_trained_endpoints = _sample(good, initial, config, guide=bad)
        if step == 0 and not torch.equal(
            fixed_reference_endpoints, jointly_trained_endpoints
        ):
            max_error = float(
                (fixed_reference_endpoints - jointly_trained_endpoints).abs().max()
            )
            raise RuntimeError(
                "step-0 initial-AB32 and trained-AB32 guidance must match; "
                f"max_error={max_error}"
            )
        endpoints = torch.stack(
            (
                a64_endpoints,
                ab32_endpoints,
                fixed_reference_endpoints,
                jointly_trained_endpoints,
            )
        )
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
        print(
            f"GUIDED_FIELD_PROGRESSION_SAMPLE step={step}/{config.outer_steps}",
            flush=True,
        )
        del good, bad, endpoints

    return (
        {name: np.stack(values) for name, values in arrays.items()},
        verified_steps,
    )


def _metric_rows(
    data: dict[str, np.ndarray], config: ToyNFTConfig
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


def _scatter(axis: plt.Axes, values: np.ndarray, method: str) -> None:
    axis.scatter(
        values[:, 0],
        values[:, 1],
        s=POINT_SIZE,
        marker="o",
        color=METHOD_COLORS[method],
        alpha=POINT_ALPHA,
        edgecolors="none",
        linewidths=0,
        rasterized=True,
        zorder=3,
    )


def _draw_tree(axis: plt.Axes, config: DualSubtreeConfig) -> None:
    target_segments = {
        (segment.cls, segment.path)
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
        is_target = (segment.cls, segment.path) in target_segments
        axis.plot(
            [segment.start[0], segment.end[0]],
            [segment.start[1], segment.end[1]],
            color=(
                TARGET_LINE_COLOR
                if is_target
                else TREE_BACKGROUND_COLORS[segment.cls]
            ),
            linewidth=(
                TARGET_LINE_WIDTH if is_target else TREE_BACKGROUND_LINE_WIDTH
            ),
            alpha=TARGET_LINE_ALPHA if is_target else TREE_BACKGROUND_ALPHA,
            zorder=2 if is_target else 1,
        )


def _style_axis(axis: plt.Axes) -> None:
    segments = tree_segments("AB")
    xs = [v for segment in segments for v in (segment.start[0], segment.end[0])]
    ys = [v for segment in segments for v in (segment.start[1], segment.end[1])]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    axis.set_xlim(min(xs) - 0.06 * x_span, max(xs) + 0.06 * x_span)
    axis.set_ylim(min(ys) - 0.06 * y_span, max(ys) + 0.06 * y_span)
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
        color="#4C8CA4",
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
        color="#4C8CA4",
        fontsize=7,
        fontweight="semibold",
    )
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_step(
    step: int,
    data: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    plot_config: DualSubtreeConfig,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(METHODS), figsize=(17.2, 4.7), squeeze=False)
    step_rows = {row["method"]: row for row in rows if row["outer_step"] == step}
    for method_index, method in enumerate(METHODS):
        axis = axes[0, method_index]
        _draw_tree(axis, plot_config)
        _scatter(axis, data["endpoints"][step, method_index], method)
        _style_axis(axis)
        row = step_rows[method]
        axis.set_title(
            f"{METHOD_LABELS[method]}\n"
            f"d={row['union_distance_mean']:.3f}, "
            f"near A/B={row['near_a_target_rate']:.2f}/"
            f"{row['near_b_target_rate']:.2f}",
            fontsize=10,
        )
    figure.suptitle(
        f"Joint w=3 guided-field NFT — outer step {step}",
        y=0.99,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_overview(
    data: dict[str, np.ndarray],
    config: ToyNFTConfig,
    plot_config: DualSubtreeConfig,
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
        num=plot_config.overview_samples,
        dtype=np.int64,
    )
    for step in range(config.outer_steps + 1):
        for method_index, method in enumerate(METHODS):
            axis = axes[step, method_index]
            _draw_tree(axis, plot_config)
            _scatter(axis, data["endpoints"][step, method_index, indices], method)
            _style_axis(axis)
            if step == 0:
                axis.set_title(METHOD_LABELS[method], fontsize=10, fontweight="semibold")
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
        "Joint w=3 NFT: components, fixed initial AB32, and trained AB32",
        y=0.998,
        fontsize=14,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0.035, 0, 1, 0.983))
    png_path = output_dir / "guided_field_progression_all_steps.png"
    pdf_path = output_dir / "guided_field_progression_all_steps.pdf"
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def _plot_metrics(
    rows: list[dict[str, Any]], config: ToyNFTConfig, output_dir: Path
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        steps = [row["outer_step"] for row in method_rows]
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
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
            label=f"{label}: target",
        )
        axes[1, 0].plot(
            steps,
            [row["full_tree_a_assignment_rate"] for row in method_rows],
            color=color,
            linestyle="--",
            label=f"{label}: full tree",
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
    for axis, ylabel in (
        (axes[0, 0], "mean distance to either target ↓"),
        (axes[0, 1], "near-target rate ↑"),
        (axes[1, 0], "A-side assignment rate"),
        (axes[1, 1], "same-noise endpoint change"),
    ):
        axis.set_xlabel("outer NFT step")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(config.outer_steps + 1))
        axis.grid(alpha=0.22, linewidth=0.5)
        axis.legend(frameon=False, fontsize=7)
    png_path = output_dir / "guided_field_progression_metrics.png"
    pdf_path = output_dir / "guided_field_progression_metrics.pdf"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def render_guided_field_progression(
    run_dir: Path, output_dir: Path, device: torch.device
) -> dict[str, Any]:
    config = ToyNFTConfig.from_json(run_dir / "config.json")
    _validate_training_config(config)
    plot_config = _plot_config(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "training_config.json", dataclasses.asdict(config))
    data, verified_steps = collect_guided_field_progression_data(
        run_dir, config, device
    )
    np.savez_compressed(
        output_dir / "guided_field_progression_data.npz",
        **data,
        methods=np.asarray(METHODS),
        outer_steps=np.arange(config.outer_steps + 1),
    )
    rows = _metric_rows(data, config)
    metrics_path = output_dir / "guided_field_progression_metrics.jsonl"
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
            plot_config,
            step_dir / f"step-{step:02d}.png",
        )
    final_comparison_png = output_dir / "guided_field_step10_comparison.png"
    final_comparison_pdf = output_dir / "guided_field_step10_comparison.pdf"
    _plot_step(
        config.outer_steps,
        data,
        rows,
        plot_config,
        final_comparison_png,
    )
    _plot_step(
        config.outer_steps,
        data,
        rows,
        plot_config,
        final_comparison_pdf,
    )
    overview_png, overview_pdf = _plot_overview(
        data, config, plot_config, output_dir
    )
    metrics_png, metrics_pdf = _plot_metrics(rows, config, output_dir)
    initial_rows = {row["method"]: row for row in rows if row["outer_step"] == 0}
    final_rows = {
        row["method"]: row
        for row in rows
        if row["outer_step"] == config.outer_steps
    }
    summary = {
        "schema_version": 1,
        "formal_experiment": False,
        "name": config.name,
        "run_dir": str(run_dir.resolve()),
        "outer_steps": config.outer_steps,
        "methods": list(METHODS),
        "common_initial_noise": True,
        "trained_jointly": True,
        "guided_rollout": True,
        "combination": {
            "joint_formula": "D_AB32_current + 3 * (D_A64_current - D_AB32_current)",
            "fixed_reference_formula": "D_AB32_initial + 3 * (D_A64_current - D_AB32_initial)",
            "good": "A64",
            "bad": "AB32",
            "guidance_scale": config.guidance_scale,
            "optimized_components": ["A64", "AB32"],
            "step0_guided_variants_exactly_equal": True,
        },
        "snapshot_refresh": {
            "ema_decay": config.ema_decay,
            "old_policy_decay": config.old_policy_decay,
            "semantics": "exact current-parameter copy after every outer step",
            "verified_exact_checkpoint_steps": verified_steps,
            "old_and_ema_exact_refresh_verified": (
                verified_steps == list(range(1, config.outer_steps + 1))
            ),
        },
        "point_style": {
            "marker": "o",
            "size": POINT_SIZE,
            "alpha": POINT_ALPHA,
            "filled": True,
            "edge": "none",
        },
        "target_line_style": {
            "shared_color": TARGET_LINE_COLOR,
            "width": TARGET_LINE_WIDTH,
            "alpha": TARGET_LINE_ALPHA,
        },
        "tree_background_style": {
            "colors": TREE_BACKGROUND_COLORS,
            "width": TREE_BACKGROUND_LINE_WIDTH,
            "alpha": TREE_BACKGROUND_ALPHA,
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
                "branch": f"{config.branch_class}:{config.branch_path}",
                "levels": config.branch_levels,
                "edge_ids": [
                    f"{segment.cls}:{segment.path}"
                    for segment in select_branch_segments(
                        config.branch_class,
                        config.branch_path,
                        config.branch_levels,
                    )
                ],
                "segments": len(
                    select_branch_segments(
                        config.branch_class,
                        config.branch_path,
                        config.branch_levels,
                    )
                ),
                "value_on_centerline": 1.0,
            },
            "B": {
                "branch": (
                    f"{config.secondary_branch_class}:"
                    f"{config.secondary_branch_path}"
                ),
                "levels": config.secondary_branch_levels,
                "edge_ids": [
                    f"{segment.cls}:{segment.path}"
                    for segment in select_branch_segments(
                        config.secondary_branch_class,
                        config.secondary_branch_path,
                        config.secondary_branch_levels,
                    )
                ],
                "segments": len(
                    select_branch_segments(
                        config.secondary_branch_class,
                        config.secondary_branch_path,
                        config.secondary_branch_levels,
                    )
                ),
                "value_on_centerline": 1.0,
            },
        },
        "initial_metrics": initial_rows,
        "final_metrics": final_rows,
        "artifacts": {
            "data": str(output_dir / "guided_field_progression_data.npz"),
            "metrics_jsonl": str(metrics_path),
            "step_figures": str(step_dir / "step-00.png"),
            "step10_comparison_png": str(final_comparison_png),
            "step10_comparison_pdf": str(final_comparison_pdf),
            "overview_png": str(overview_png),
            "overview_pdf": str(overview_pdf),
            "metrics_png": str(metrics_png),
            "metrics_pdf": str(metrics_pdf),
        },
    }
    if not summary["snapshot_refresh"]["old_and_ema_exact_refresh_verified"]:
        raise RuntimeError("not every checkpoint passed old/EMA refresh validation")
    _atomic_json(output_dir / "summary.json", summary)
    print(
        f"GUIDED_FIELD_PROGRESSION_CONFIRMED outer_steps={config.outer_steps} "
        f"output_dir={output_dir}",
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
    "--run-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
)
@click.option("--device", type=str, default="cuda", show_default=True)
def main(run_dir: Path, output_dir: Path | None, device: str) -> None:
    """Render A64, AB32, and their jointly trained w=3 field at every step."""

    config = ToyNFTConfig.from_json(run_dir / "config.json")
    resolved_output = run_dir.parent / "evaluation" if output_dir is None else output_dir
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32))
    render_guided_field_progression(
        run_dir, resolved_output, _resolve_device(device)
    )


if __name__ == "__main__":
    main()
