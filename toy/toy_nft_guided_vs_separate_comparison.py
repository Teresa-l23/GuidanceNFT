"""Compare joint guided-field NFT against separately trained A64/AB32 NFT."""

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

from toy_example import gt, tree_segments
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
    _validate_training_config as _validate_separate_config,
)
from toy_nft_guidance import (
    ToyNFTConfig,
    branch_distance_reward_formula,
    distance_and_assignment_to_segments,
    distance_to_segments,
    load_model_bundle,
    reward_from_branch_distance,
    reward_branch_selection_specs,
    select_branch_segments,
)
from toy_nft_guided_field_progression import (
    POINT_ALPHA,
    POINT_SIZE,
    TARGET_LINE_ALPHA,
    TARGET_LINE_COLOR,
    TARGET_LINE_WIDTH,
    _draw_tree,
    _plot_config,
    _sample,
    _state_dicts_equal,
    _style_axis,
    _validate_training_config as _validate_joint_config,
    _verify_snapshot_refresh,
)


GROUPS = ("joint_guided_field", "separate_nft")
THREE_METHOD_GROUPS = (
    "separate_nft",
    "joint_guided_field",
    "joint_paired_branches",
)
GROUP_LABELS = {
    "joint_guided_field": "Joint guided-field NFT",
    "separate_nft": "Separately trained NFT",
    "joint_paired_branches": "Joint paired-branches NFT",
}
GROUP_LINESTYLES = {
    "joint_guided_field": "-",
    "separate_nft": "--",
    "joint_paired_branches": ":",
}
OVERVIEW_SAMPLES = 512
ON_TREE_THRESHOLD = 0.025
ON_TREE_SENSITIVITY_THRESHOLDS = (0.01, 0.02, 0.04)
METHODS = (
    "a64",
    "ab32",
    "a64_self_guided_w3",
    "a64_init_ab32_w3",
    "a64_trained_ab32_w3",
)
METHOD_LABELS = {
    "a64": "A64",
    "ab32": "AB32",
    "a64_self_guided_w3": "Self-guided A64 (w=3)",
    "a64_init_ab32_w3": "A64 + initial AB32 (w=3)",
    "a64_trained_ab32_w3": "A64 + trained AB32 (w=3)",
}
METHOD_COLORS = {
    "a64": "#D55E00",
    "ab32": "#6F4C9B",
    "a64_self_guided_w3": "#CC79A7",
    "a64_init_ab32_w3": "#009E73",
    "a64_trained_ab32_w3": "#0072B2",
}


def _comparison_groups(paired_run_dir: Path | None) -> tuple[str, ...]:
    return THREE_METHOD_GROUPS if paired_run_dir is not None else GROUPS


def _load_configs(
    joint_run_dir: Path,
    separate_root: Path,
    paired_run_dir: Path | None = None,
) -> tuple[
    ToyNFTConfig,
    dict[str, ToyNFTConfig],
    dict[str, Path],
    ToyNFTConfig | None,
]:
    joint_config = ToyNFTConfig.from_json(joint_run_dir / "config.json")
    _validate_joint_config(joint_config)
    separate_dirs = {
        "a64": separate_root / "training" / "a64_nft",
        "ab32": separate_root / "training" / "ab32_nft",
    }
    for name, run_dir in separate_dirs.items():
        if not (run_dir / "checkpoint.pt").is_file():
            raise FileNotFoundError(f"incomplete separate {name} run: {run_dir}")
    separate_configs = {
        name: ToyNFTConfig.from_json(run_dir / "config.json")
        for name, run_dir in separate_dirs.items()
    }
    matrix = _plot_config(joint_config)
    a64_sampling_classes = separate_configs["a64"].sampling_classes
    if a64_sampling_classes not in {"A", "AB"}:
        raise ValueError("separate A64 sampling_classes must be A or AB")
    _validate_separate_config(
        separate_configs["a64"],
        matrix,
        sampling_classes=a64_sampling_classes,
    )
    _validate_separate_config(
        separate_configs["ab32"], matrix, sampling_classes="AB"
    )
    for name, config in separate_configs.items():
        for field in (
            "seed",
            "outer_steps",
            "evaluation_samples",
            "evaluation_steps",
            "sigma_min",
            "sigma_max",
            "rho",
            "near_branch_threshold",
            "reward",
            "reward_distance_scale",
            "branch_class",
            "branch_path",
            "branch_levels",
            "secondary_branch_class",
            "secondary_branch_path",
            "secondary_branch_levels",
        ):
            if getattr(config, field) != getattr(joint_config, field):
                raise ValueError(
                    f"joint and separate {name} configs disagree on {field}"
                )
    paired_config = None
    if paired_run_dir is not None:
        if not (paired_run_dir / "checkpoint.pt").is_file():
            raise FileNotFoundError(
                f"incomplete paired-branches run: {paired_run_dir}"
            )
        paired_config = ToyNFTConfig.from_json(paired_run_dir / "config.json")
        if paired_config.objective_mode != "paired_branches":
            raise ValueError(
                "paired run must use objective_mode='paired_branches'"
            )
        if a64_sampling_classes != "AB":
            raise ValueError(
                "three-method comparison requires AB sampling for separate A64"
            )
        comparable_fields = (
            "seed",
            "good_model",
            "bad_model",
            "guidance_scale",
            "branch_class",
            "branch_path",
            "branch_levels",
            "secondary_branch_class",
            "secondary_branch_path",
            "secondary_branch_levels",
            "sampling_classes",
            "reward",
            "reward_distance_scale",
            "outer_steps",
            "rollout_samples",
            "rollout_steps",
            "sigma_min",
            "sigma_max",
            "rho",
            "timestep_fraction",
            "train_batch_size",
            "inner_epochs",
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "max_grad_norm",
            "mix_beta",
            "advantage_clip",
            "reward_std_epsilon",
            "ema_decay",
            "old_policy_decay",
            "evaluation_samples",
            "evaluation_steps",
            "near_branch_threshold",
        )
        for field in comparable_fields:
            if getattr(paired_config, field) != getattr(joint_config, field):
                raise ValueError(
                    f"guided-field and paired configs disagree on {field}"
                )
    return joint_config, separate_configs, separate_dirs, paired_config


def _checkpoint_model(
    base: torch.nn.Module,
    run_dir: Path,
    config: ToyNFTConfig,
    step: int,
    model_name: str,
    device: torch.device,
) -> tuple[torch.nn.Module, bool]:
    payload = _checkpoint_payload(run_dir, config, step, device)
    if payload is None:
        return _frozen_clone(base), False
    _verify_snapshot_refresh(payload, step)
    return _frozen_clone(base, payload["models"][model_name]), True


def _sample_conditions(
    a64: torch.nn.Module,
    ab32: torch.nn.Module,
    initial_a64: torch.nn.Module,
    initial_ab32: torch.nn.Module,
    initial: torch.Tensor,
    sampler_config: ToyNFTConfig,
) -> torch.Tensor:
    return torch.stack(
        (
            _sample(a64, initial, sampler_config),
            _sample(ab32, initial, sampler_config),
            _sample(a64, initial, sampler_config, guide=initial_a64),
            _sample(a64, initial, sampler_config, guide=initial_ab32),
            _sample(a64, initial, sampler_config, guide=ab32),
        )
    )


def collect_comparison_data(
    joint_run_dir: Path,
    separate_root: Path,
    device: torch.device,
    paired_run_dir: Path | None = None,
) -> tuple[
    ToyNFTConfig,
    DualSubtreeConfig,
    dict[str, np.ndarray],
    dict[str, list[int]],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Sample all training strategies at every checkpoint with common noise."""

    joint_config, separate_configs, separate_dirs, paired_config = _load_configs(
        joint_run_dir, separate_root, paired_run_dir
    )
    groups = _comparison_groups(paired_run_dir)
    plot_config = _plot_config(joint_config)
    joint_bundle = load_model_bundle(joint_config, device)
    paired_bundle = (
        None
        if paired_config is None
        else load_model_bundle(paired_config, device)
    )
    separate_bundles = {
        name: load_model_bundle(config, device)
        for name, config in separate_configs.items()
    }
    if not _state_dicts_equal(
        joint_bundle.good_ema.state_dict(),
        separate_bundles["a64"].good_ema.state_dict(),
    ):
        raise ValueError("joint and separate runs do not start from the same A64")
    if not _state_dicts_equal(
        joint_bundle.bad_ema.state_dict(),
        separate_bundles["ab32"].good_ema.state_dict(),
    ):
        raise ValueError("joint and separate runs do not start from the same AB32")
    if paired_bundle is not None:
        if not _state_dicts_equal(
            joint_bundle.good_ema.state_dict(),
            paired_bundle.good_ema.state_dict(),
        ):
            raise ValueError(
                "guided-field and paired runs do not start from the same A64"
            )
        if not _state_dicts_equal(
            joint_bundle.bad_ema.state_dict(),
            paired_bundle.bad_ema.state_dict(),
        ):
            raise ValueError(
                "guided-field and paired runs do not start from the same AB32"
            )
    initial_a64 = _frozen_clone(joint_bundle.good_ema)
    initial_ab32 = _frozen_clone(joint_bundle.bad_ema)

    generator = torch.Generator(device=device).manual_seed(
        joint_config.seed + 70_000_003
    )
    initial = gt("AB", device).sample(
        joint_config.evaluation_samples,
        joint_config.sigma_max,
        generator=generator,
    )
    target_specs = reward_branch_selection_specs(joint_config)
    if len(target_specs) != 2:
        raise ValueError("comparison requires exactly two reward targets")
    a_spec, b_spec = target_specs
    if (a_spec[0], b_spec[0]) != ("A", "B"):
        raise ValueError("comparison requires A then B reward targets")
    a_starts, a_ends = _segment_tensors(
        a_spec[0], a_spec[1], device, initial.dtype, a_spec[2]
    )
    b_starts, b_ends = _segment_tensors(
        b_spec[0], b_spec[1], device, initial.dtype, b_spec[2]
    )
    target_starts = torch.cat((a_starts, b_starts))
    target_ends = torch.cat((a_ends, b_ends))
    target_segments = tuple(
        segment
        for target_spec in target_specs
        for segment in select_branch_segments(*target_spec)
    )
    target_edge_ids = tuple(
        f"{segment.cls}:{segment.path}" for segment in target_segments
    )
    full_a_starts, full_a_ends = _full_tree_tensors("A", device, initial.dtype)
    full_b_starts, full_b_ends = _full_tree_tensors("B", device, initial.dtype)
    arrays: dict[str, list[np.ndarray]] = {
        "endpoints": [],
        "union_distance": [],
        "a_target_distance": [],
        "b_target_distance": [],
        "target_edge_assignment": [],
        "full_tree_distance": [],
        "target_a_assignment": [],
        "full_tree_a_assignment": [],
    }
    verified_steps = {"joint": [], "separate_a64": [], "separate_ab32": []}
    if paired_run_dir is not None:
        verified_steps["paired"] = []

    for step in range(joint_config.outer_steps + 1):
        joint_payload = _checkpoint_payload(
            joint_run_dir, joint_config, step, device
        )
        if joint_payload is not None:
            _verify_snapshot_refresh(joint_payload, step)
            verified_steps["joint"].append(step)
        joint_a64 = _frozen_clone(
            joint_bundle.good_ema,
            None if joint_payload is None else joint_payload["models"]["good_ema"],
        )
        joint_ab32 = _frozen_clone(
            joint_bundle.bad_ema,
            None if joint_payload is None else joint_payload["models"]["bad_ema"],
        )
        separate_a64, a64_verified = _checkpoint_model(
            separate_bundles["a64"].good_ema,
            separate_dirs["a64"],
            separate_configs["a64"],
            step,
            "good_ema",
            device,
        )
        separate_ab32, ab32_verified = _checkpoint_model(
            separate_bundles["ab32"].good_ema,
            separate_dirs["ab32"],
            separate_configs["ab32"],
            step,
            "good_ema",
            device,
        )
        if a64_verified:
            verified_steps["separate_a64"].append(step)
        if ab32_verified:
            verified_steps["separate_ab32"].append(step)
        group_endpoints = {
            "joint_guided_field": _sample_conditions(
                joint_a64,
                joint_ab32,
                initial_a64,
                initial_ab32,
                initial,
                joint_config,
            ),
            "separate_nft": _sample_conditions(
                separate_a64,
                separate_ab32,
                initial_a64,
                initial_ab32,
                initial,
                joint_config,
            ),
        }
        paired_a64 = None
        paired_ab32 = None
        if paired_run_dir is not None:
            assert paired_bundle is not None
            assert paired_config is not None
            paired_payload = _checkpoint_payload(
                paired_run_dir, paired_config, step, device
            )
            if paired_payload is not None:
                _verify_snapshot_refresh(paired_payload, step)
                verified_steps["paired"].append(step)
            paired_a64 = _frozen_clone(
                paired_bundle.good_ema,
                None
                if paired_payload is None
                else paired_payload["models"]["good_ema"],
            )
            paired_ab32 = _frozen_clone(
                paired_bundle.bad_ema,
                None
                if paired_payload is None
                else paired_payload["models"]["bad_ema"],
            )
            group_endpoints["joint_paired_branches"] = _sample_conditions(
                paired_a64,
                paired_ab32,
                initial_a64,
                initial_ab32,
                initial,
                joint_config,
            )
        endpoints = torch.stack(tuple(group_endpoints[group] for group in groups))
        if step == 0:
            reference = endpoints[0]
            for group_index, group in enumerate(groups[1:], start=1):
                if not torch.equal(reference, endpoints[group_index]):
                    max_error = float(
                        (reference - endpoints[group_index]).abs().max()
                    )
                    raise RuntimeError(
                        "all comparison groups must match exactly at step 0; "
                        f"group={group} max_error={max_error}"
                    )

        flat = endpoints.reshape(-1, 2)
        shape = (
            len(groups),
            len(METHODS),
            joint_config.evaluation_samples,
        )
        a_distance = distance_to_segments(flat, a_starts, a_ends).reshape(shape)
        b_distance = distance_to_segments(flat, b_starts, b_ends).reshape(shape)
        union_distance, target_edge_assignment = (
            distance_and_assignment_to_segments(flat, target_starts, target_ends)
        )
        union_distance = union_distance.reshape(shape)
        target_edge_assignment = target_edge_assignment.reshape(shape)
        full_a_distance = distance_to_segments(
            flat, full_a_starts, full_a_ends
        ).reshape(shape)
        full_b_distance = distance_to_segments(
            flat, full_b_starts, full_b_ends
        ).reshape(shape)
        full_tree_distance = torch.minimum(full_a_distance, full_b_distance)
        arrays["endpoints"].append(endpoints.cpu().numpy())
        arrays["union_distance"].append(union_distance.cpu().numpy())
        arrays["a_target_distance"].append(a_distance.cpu().numpy())
        arrays["b_target_distance"].append(b_distance.cpu().numpy())
        arrays["target_edge_assignment"].append(
            target_edge_assignment.cpu().numpy()
        )
        arrays["full_tree_distance"].append(full_tree_distance.cpu().numpy())
        arrays["target_a_assignment"].append(
            (a_distance <= b_distance).cpu().numpy()
        )
        arrays["full_tree_a_assignment"].append(
            (full_a_distance <= full_b_distance).cpu().numpy()
        )
        print(
            f"GUIDED_VS_SEPARATE_SAMPLE step={step}/{joint_config.outer_steps}",
            flush=True,
        )
        del joint_a64, joint_ab32, separate_a64, separate_ab32, endpoints
        del paired_a64, paired_ab32

    return (
        joint_config,
        plot_config,
        {name: np.stack(values) for name, values in arrays.items()},
        verified_steps,
        groups,
        target_edge_ids,
    )


def _metric_rows(
    data: dict[str, np.ndarray],
    config: ToyNFTConfig,
    groups: tuple[str, ...],
    target_edge_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in range(config.outer_steps + 1):
        for group_index, group in enumerate(groups):
            for method_index, method in enumerate(METHODS):
                union = data["union_distance"][step, group_index, method_index]
                full_tree = data["full_tree_distance"][
                    step, group_index, method_index
                ]
                a_distance = data["a_target_distance"][
                    step, group_index, method_index
                ]
                b_distance = data["b_target_distance"][
                    step, group_index, method_index
                ]
                movement = (
                    0.0
                    if step == 0
                    else float(
                        np.linalg.norm(
                            data["endpoints"][step, group_index, method_index]
                            - data["endpoints"][
                                step - 1, group_index, method_index
                            ],
                            axis=1,
                        ).mean()
                    )
                )
                target_hits = union <= config.near_branch_threshold
                edge_assignments = data["target_edge_assignment"][
                    step, group_index, method_index
                ][target_hits]
                edge_hit_counts = np.bincount(
                    edge_assignments.astype(np.int64, copy=False),
                    minlength=len(target_edge_ids),
                )
                hit_count = int(target_hits.sum())
                rows.append(
                    {
                        "outer_step": step,
                        "group": group,
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
                        "target_distance_mean": float(union.mean()),
                        "target_distance_median": float(np.median(union)),
                        "near_union_rate": float(
                            target_hits.mean()
                        ),
                        "target_hit_union": float(target_hits.mean()),
                        "near_a_target_rate": float(
                            (a_distance <= config.near_branch_threshold).mean()
                        ),
                        "target_hit_a": float(
                            (a_distance <= config.near_branch_threshold).mean()
                        ),
                        "near_b_target_rate": float(
                            (b_distance <= config.near_branch_threshold).mean()
                        ),
                        "target_hit_b": float(
                            (b_distance <= config.near_branch_threshold).mean()
                        ),
                        "tree_distance_mean": float(full_tree.mean()),
                        "tree_distance_median": float(np.median(full_tree)),
                        "tree_distance_p95": float(
                            np.quantile(full_tree, 0.95)
                        ),
                        "on_tree_rate": float(
                            (full_tree <= ON_TREE_THRESHOLD).mean()
                        ),
                        "on_tree_threshold": ON_TREE_THRESHOLD,
                        "on_tree_sensitivity": {
                            f"{threshold:g}": float(
                                (full_tree <= threshold).mean()
                            )
                            for threshold in ON_TREE_SENSITIVITY_THRESHOLDS
                        },
                        "target_edge_hit_counts": {
                            edge_id: int(count)
                            for edge_id, count in zip(
                                target_edge_ids,
                                edge_hit_counts,
                                strict=True,
                            )
                        },
                        "target_edge_hit_rates": {
                            edge_id: (
                                float(count / hit_count) if hit_count else 0.0
                            )
                            for edge_id, count in zip(
                                target_edge_ids,
                                edge_hit_counts,
                                strict=True,
                            )
                        },
                        "target_a_assignment_rate": float(
                            data["target_a_assignment"][
                                step, group_index, method_index
                            ].mean()
                        ),
                        "full_tree_a_assignment_rate": float(
                            data["full_tree_a_assignment"][
                                step, group_index, method_index
                            ].mean()
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


def _plot_target_preview(
    config: ToyNFTConfig,
    plot_config: DualSubtreeConfig,
    output_dir: Path,
) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(8.4, 7.2))
    _draw_tree(axis, plot_config)
    _style_axis(axis)
    target_keys = {
        (segment.cls, segment.path)
        for branch_class, branch_path, branch_levels in (
            reward_branch_selection_specs(config)
        )
        for segment in select_branch_segments(
            branch_class, branch_path, branch_levels
        )
    }
    for segment in tree_segments("AB"):
        if (segment.cls, segment.path) not in target_keys:
            continue
        midpoint = (
            (segment.start[0] + segment.end[0]) / 2,
            (segment.start[1] + segment.end[1]) / 2,
        )
        axis.text(
            midpoint[0],
            midpoint[1],
            f"{segment.cls}:{segment.path}",
            fontsize=7.2,
            ha="center",
            va="center",
            color="#173B4A",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=4,
        )
    axis.set_title(
        "Reward target: three finite trunk levels per tree (14 edges)",
        fontsize=12,
        fontweight="semibold",
    )
    figure.tight_layout()
    png_path = output_dir / "target_edges_preview.png"
    pdf_path = output_dir / "target_edges_preview.pdf"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def _plot_step(
    step: int,
    data: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    plot_config: DualSubtreeConfig,
    output_path: Path,
    groups: tuple[str, ...] = GROUPS,
) -> None:
    figure, axes = plt.subplots(
        len(groups),
        len(METHODS),
        figsize=(4.3 * len(METHODS), 4.55 * len(groups)),
        squeeze=False,
    )
    step_rows = {
        (row["group"], row["method"]): row
        for row in rows
        if row["outer_step"] == step
    }
    for group_index, group in enumerate(groups):
        for method_index, method in enumerate(METHODS):
            axis = axes[group_index, method_index]
            _draw_tree(axis, plot_config)
            _scatter(axis, data["endpoints"][step, group_index, method_index], method)
            _style_axis(axis)
            row = step_rows[(group, method)]
            heading = f"{METHOD_LABELS[method]}\n" if group_index == 0 else ""
            axis.set_title(
                heading
                + f"d_target={row['target_distance_mean']:.3f}; "
                + f"hit A/B={row['target_hit_a']:.2f}/"
                + f"{row['target_hit_b']:.2f}; "
                + f"Q@{ON_TREE_THRESHOLD:g}={row['on_tree_rate']:.2f}",
                fontsize=8.7,
            )
            if method_index == 0:
                axis.text(
                    -0.09,
                    0.5,
                    GROUP_LABELS[group],
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=10,
                    fontweight="semibold",
                )
    comparison_title = (
        "Three NFT update strategies"
        if len(groups) == 3
        else "Joint guided-field NFT vs separate NFT"
    )
    figure.suptitle(
        f"{comparison_title} — outer step {step}, w=3",
        y=0.995,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0.025, 0, 1, 0.94))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_overview(
    data: dict[str, np.ndarray],
    config: ToyNFTConfig,
    plot_config: DualSubtreeConfig,
    output_dir: Path,
    groups: tuple[str, ...] = GROUPS,
) -> tuple[Path, Path]:
    column_count = len(groups) * len(METHODS)
    figure, axes = plt.subplots(
        config.outer_steps + 1,
        column_count,
        figsize=(4.125 * column_count, 4.0 * (config.outer_steps + 1)),
        squeeze=False,
    )
    indices = np.linspace(
        0,
        config.evaluation_samples - 1,
        num=min(OVERVIEW_SAMPLES, config.evaluation_samples),
        dtype=np.int64,
    )
    for step in range(config.outer_steps + 1):
        for group_index, group in enumerate(groups):
            for method_index, method in enumerate(METHODS):
                column = group_index * len(METHODS) + method_index
                axis = axes[step, column]
                _draw_tree(axis, plot_config)
                _scatter(
                    axis,
                    data["endpoints"][step, group_index, method_index, indices],
                    method,
                )
                _style_axis(axis)
                if step == 0:
                    axis.set_title(
                        f"{GROUP_LABELS[group]}\n{METHOD_LABELS[method]}",
                        fontsize=8.5,
                        fontweight="semibold",
                    )
                if column == 0:
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
        "NFT update strategies under identical w=3 evaluation",
        y=0.998,
        fontsize=14,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0.018, 0, 1, 0.983))
    png_path = output_dir / "guided_vs_separate_all_steps.png"
    pdf_path = output_dir / "guided_vs_separate_all_steps.pdf"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def _plot_metrics(
    rows: list[dict[str, Any]],
    config: ToyNFTConfig,
    output_dir: Path,
    groups: tuple[str, ...] = GROUPS,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    group_markers = {
        "joint_guided_field": "o",
        "separate_nft": None,
        "joint_paired_branches": "s",
    }
    for group in groups:
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["group"] == group and row["method"] == method
            ]
            steps = [row["outer_step"] for row in selected]
            label = f"{GROUP_LABELS[group]} | {METHOD_LABELS[method]}"
            style = GROUP_LINESTYLES[group]
            color = METHOD_COLORS[method]
            axes[0, 0].plot(
                steps,
                [row["reward_mean"] for row in selected],
                color=color,
                linestyle=style,
                marker=group_markers[group],
                label=label,
            )
            axes[0, 1].plot(
                steps,
                [row["target_hit_union"] for row in selected],
                color=color,
                linestyle=style,
                marker=group_markers[group],
                label=label,
            )
            axes[1, 0].plot(
                steps,
                [row["on_tree_rate"] for row in selected],
                color=color,
                linestyle=style,
                marker=group_markers[group],
                label=label,
            )
            axes[1, 1].plot(
                steps,
                [row["tree_distance_mean"] for row in selected],
                color=color,
                linestyle=style,
                marker=group_markers[group],
                label=label,
            )
    for axis, ylabel in (
        (axes[0, 0], "mean reward ↑"),
        (axes[0, 1], "target-hit rate ↑"),
        (axes[1, 0], f"on-tree rate Q@{ON_TREE_THRESHOLD:g} ↑"),
        (axes[1, 1], "mean distance to full tree ↓"),
    ):
        axis.set_xlabel("outer NFT step")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(config.outer_steps + 1))
        axis.grid(alpha=0.22, linewidth=0.5)
        axis.legend(frameon=False, fontsize=6.2, ncol=2)
    png_path = output_dir / "guided_vs_separate_metrics.png"
    pdf_path = output_dir / "guided_vs_separate_metrics.pdf"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def render_comparison(
    joint_run_dir: Path,
    separate_root: Path,
    output_dir: Path,
    device: torch.device,
    paired_run_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    (
        config,
        plot_config,
        data,
        verified_steps,
        groups,
        target_edge_ids,
    ) = collect_comparison_data(
        joint_run_dir,
        separate_root,
        device,
        paired_run_dir,
    )
    _atomic_json(
        output_dir / "comparison_sources.json",
        {
            "joint_run_dir": str(joint_run_dir.resolve()),
            "separate_root": str(separate_root.resolve()),
            "paired_run_dir": (
                None
                if paired_run_dir is None
                else str(paired_run_dir.resolve())
            ),
            "joint_config": dataclasses.asdict(config),
        },
    )
    np.savez_compressed(
        output_dir / "guided_vs_separate_data.npz",
        **data,
        groups=np.asarray(groups),
        methods=np.asarray(METHODS),
        target_edge_ids=np.asarray(target_edge_ids),
        outer_steps=np.arange(config.outer_steps + 1),
    )
    rows = _metric_rows(data, config, groups, target_edge_ids)
    metrics_path = output_dir / "guided_vs_separate_metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    step_dir = output_dir / "steps"
    step_dir.mkdir()
    target_preview_png, target_preview_pdf = _plot_target_preview(
        config, plot_config, output_dir
    )
    for step in range(config.outer_steps + 1):
        _plot_step(
            step,
            data,
            rows,
            plot_config,
            step_dir / f"step-{step:02d}.png",
            groups,
        )
    step10_png = output_dir / "guided_vs_separate_step10.png"
    step10_pdf = output_dir / "guided_vs_separate_step10.pdf"
    _plot_step(
        config.outer_steps, data, rows, plot_config, step10_png, groups
    )
    _plot_step(
        config.outer_steps, data, rows, plot_config, step10_pdf, groups
    )
    overview_png, overview_pdf = _plot_overview(
        data, config, plot_config, output_dir, groups
    )
    metrics_png, metrics_pdf = _plot_metrics(rows, config, output_dir, groups)
    final_metrics = {
        group: {
            method: next(
                row
                for row in rows
                if row["outer_step"] == config.outer_steps
                and row["group"] == group
                and row["method"] == method
            )
            for method in METHODS
        }
        for group in groups
    }
    expected_steps = list(range(1, config.outer_steps + 1))
    training_semantics: dict[str, Any] = {
        "joint_guided_field": {
            "rollout": "guided field",
            "nft_loss": "one classic NFT loss on the guided field",
            "old_center": "guided field composed from both old snapshots",
            "updated_parameters": ["A64", "AB32"],
        },
        "separate_nft": {
            "a64": "independent good_only NFT",
            "ab32": "independent good_only NFT",
            "combination": "inference-only same-step w=3 composition",
        },
    }
    if paired_run_dir is not None:
        training_semantics["joint_paired_branches"] = {
            "rollout": "shared guided field",
            "nft_loss": (
                "mean of separate A64 and AB32 shifted NFT branch losses"
            ),
            "old_center": (
                "shared stop-gradient guided field composed from both old snapshots"
            ),
            "gradient_isolation": (
                "A64 loss updates only A64; AB32 loss updates only AB32"
            ),
            "snapshot_refresh": "after both branch gradients are applied",
            "updated_parameters": ["A64", "AB32"],
        }
    summary = {
        "schema_version": 1,
        "formal_experiment": False,
        "outer_steps": config.outer_steps,
        "groups": list(groups),
        "methods": list(METHODS),
        "common_initial_noise": True,
        "step0_groups_exactly_equal": True,
        "guidance_scale": config.guidance_scale,
        "training_semantics": training_semantics,
        "combination": {
            "self_guidance": "D_A64_initial + 3 * (D_A64_current - D_A64_initial)",
            "fixed_reference": "D_AB32_initial + 3 * (D_A64_current - D_AB32_initial)",
            "trained_reference": "D_AB32_current + 3 * (D_A64_current - D_AB32_current)",
        },
        "snapshot_refresh_verified_steps": verified_steps,
        "all_snapshot_refreshes_verified": all(
            steps == expected_steps for steps in verified_steps.values()
        ),
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
            "edge_ids": list(target_edge_ids),
            "edge_count": len(target_edge_ids),
            "branch_specs": [
                {
                    "class": branch_class,
                    "path": branch_path,
                    "levels": branch_levels,
                    "edge_ids": [
                        f"{segment.cls}:{segment.path}"
                        for segment in select_branch_segments(
                            branch_class, branch_path, branch_levels
                        )
                    ],
                }
                for branch_class, branch_path, branch_levels in (
                    reward_branch_selection_specs(config)
                )
            ],
            "hit_threshold": config.near_branch_threshold,
            "coverage_assignment": (
                "nearest finite target edge among target hits; ties use the "
                "first edge in target edge_ids order"
            ),
        },
        "tree_quality": {
            "reference": "all 254 finite edges in A+B",
            "primary_threshold": ON_TREE_THRESHOLD,
            "sensitivity_thresholds": list(
                ON_TREE_SENSITIVITY_THRESHOLDS
            ),
            "reward_independent": True,
        },
        "final_metrics": final_metrics,
        "artifacts": {
            "data": str(output_dir / "guided_vs_separate_data.npz"),
            "metrics_jsonl": str(metrics_path),
            "step_figures": str(step_dir / "step-00.png"),
            "step10_png": str(step10_png),
            "step10_pdf": str(step10_pdf),
            "overview_png": str(overview_png),
            "overview_pdf": str(overview_pdf),
            "metrics_png": str(metrics_png),
            "metrics_pdf": str(metrics_pdf),
            "target_preview_png": str(target_preview_png),
            "target_preview_pdf": str(target_preview_pdf),
        },
    }
    if not summary["all_snapshot_refreshes_verified"]:
        raise RuntimeError("not every source checkpoint passed snapshot validation")
    _atomic_json(output_dir / "summary.json", summary)
    print(
        f"GUIDED_VS_SEPARATE_CONFIRMED outer_steps={config.outer_steps} "
        f"groups={len(groups)} output_dir={output_dir}",
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
    "--joint-run-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--separate-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--paired-run-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Optional joint paired-branches run; enables the three-row comparison.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
@click.option("--device", type=str, default="cuda", show_default=True)
def main(
    joint_run_dir: Path,
    separate_root: Path,
    paired_run_dir: Path | None,
    output_dir: Path,
    device: str,
) -> None:
    """Render joint guided-field and independently trained checkpoint groups."""

    config = ToyNFTConfig.from_json(joint_run_dir / "config.json")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32))
    render_comparison(
        joint_run_dir,
        separate_root,
        output_dir,
        _resolve_device(device),
        paired_run_dir,
    )


if __name__ == "__main__":
    main()
