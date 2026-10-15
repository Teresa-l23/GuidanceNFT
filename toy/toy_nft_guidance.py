"""Reward-tune toy score fields on the recursive 2D distribution with NFT.

The selected branch is a subtree identified by its L/R path.  Configurations
can map Euclidean distance to a linear reward or to an exponential squared-
distance reward.  The default objective applies the NFT displacement directly
to the composed Autoguidance denoiser field.  ``good_only`` instead trains the
main model from its own rollouts while keeping the reference model frozen for
a post-NFT guidance comparison.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import pickle
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import dnnlib
from nft.math import (
    advantage_to_reward_weight,
    compose_autoguidance,
    compose_shifted_nft_branches,
    compose_standard_nft_branches,
    nft_policy_loss_per_example,
    update_ema_,
)
from toy_example import do_sample, gt, tree_segments


DEFAULT_GOOD_MODEL = (
    "https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/"
    "clsA-layers04-dim64/iter4096.pkl"
)
DEFAULT_BAD_MODEL = (
    "https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/"
    "clsA-layers04-dim32/iter0512.pkl"
)
DEFAULT_CFG_REFERENCE_MODEL = (
    "https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/"
    "clsAB-layers04-dim32/iter0512.pkl"
)


@dataclass(frozen=True)
class ToyNFTConfig:
    schema_version: int = 1
    name: str = "toy-nft-branch-distance"
    seed: int = 20260916
    good_model: str = DEFAULT_GOOD_MODEL
    bad_model: str = DEFAULT_BAD_MODEL
    guidance_scale: float = 3.0
    branch_class: str = "A"
    branch_path: str = "LRL"
    branch_levels: int | None = None
    secondary_branch_class: str | None = None
    secondary_branch_path: str | None = None
    secondary_branch_levels: int | None = None
    sampling_classes: str | None = None
    reward: str = "negative_branch_distance"
    reward_distance_scale: float = 1.0
    objective_mode: str = "guided_field"
    outer_steps: int = 2
    rollout_samples: int = 512
    rollout_steps: int = 8
    sigma_min: float = 0.002
    sigma_max: float = 5.0
    rho: float = 7.0
    timestep_fraction: float = 0.99
    train_batch_size: int = 512
    inner_epochs: int = 1
    learning_rate: float = 2e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    max_grad_norm: float = 5.0
    mix_beta: float = 1.0
    advantage_clip: float = 5.0
    reward_std_epsilon: float = 1e-4
    ema_decay: float = 0.0
    old_policy_decay: float = 0.0
    evaluation_every: int = 1
    evaluation_samples: int = 2048
    evaluation_steps: int = 32
    near_branch_threshold: float = 0.04
    figure_samples: int = 2048
    figure_trajectories: int = 48
    figure_grid_resolution: int = 180

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "ToyNFTConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("toy NFT config must be a JSON object")
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - fields)
        if unknown:
            raise ValueError(f"unknown toy NFT config keys: {unknown}")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not self.name:
            raise ValueError("name must be nonempty")
        if not self.good_model or not self.bad_model:
            raise ValueError("good_model and bad_model must be nonempty")
        if self.branch_class not in {"A", "B"}:
            raise ValueError("branch_class must be A or B")
        if not self.branch_path or set(self.branch_path) - {"L", "R"}:
            raise ValueError("branch_path must be a nonempty L/R string")
        if len(self.branch_path) > 6:
            raise ValueError("branch_path cannot be deeper than six turns")
        if self.branch_levels is not None and (
            not isinstance(self.branch_levels, int) or self.branch_levels <= 0
        ):
            raise ValueError("branch_levels must be a positive integer or null")
        if (self.secondary_branch_class is None) != (
            self.secondary_branch_path is None
        ):
            raise ValueError(
                "secondary_branch_class and secondary_branch_path must be set together"
            )
        if self.secondary_branch_class is not None:
            if self.secondary_branch_class not in {"A", "B"}:
                raise ValueError("secondary_branch_class must be A, B, or null")
            assert self.secondary_branch_path is not None
            if (
                not self.secondary_branch_path
                or set(self.secondary_branch_path) - {"L", "R"}
            ):
                raise ValueError(
                    "secondary_branch_path must be a nonempty L/R string or null"
                )
            if len(self.secondary_branch_path) > 6:
                raise ValueError(
                    "secondary_branch_path cannot be deeper than six turns"
                )
            if (self.secondary_branch_class, self.secondary_branch_path) == (
                self.branch_class,
                self.branch_path,
            ):
                raise ValueError("primary and secondary reward branches must differ")
        elif self.secondary_branch_levels is not None:
            raise ValueError(
                "secondary_branch_levels requires a secondary reward branch"
            )
        if self.secondary_branch_levels is not None and (
            not isinstance(self.secondary_branch_levels, int)
            or self.secondary_branch_levels <= 0
        ):
            raise ValueError(
                "secondary_branch_levels must be a positive integer or null"
            )
        if self.sampling_classes not in {None, "A", "B", "AB"}:
            raise ValueError("sampling_classes must be A, B, AB, or null")
        if self.reward not in {
            "negative_branch_distance",
            "one_minus_branch_distance",
            "exponential_branch_distance",
        }:
            raise ValueError(
                "reward must be negative_branch_distance, "
                "one_minus_branch_distance, or exponential_branch_distance"
            )
        if self.objective_mode not in {
            "guided_field",
            "paired_branches",
            "good_only",
        }:
            raise ValueError(
                "objective_mode must be guided_field, paired_branches, or good_only"
            )
        positive_ints = {
            "outer_steps": self.outer_steps,
            "rollout_samples": self.rollout_samples,
            "rollout_steps": self.rollout_steps,
            "train_batch_size": self.train_batch_size,
            "inner_epochs": self.inner_epochs,
            "evaluation_every": self.evaluation_every,
            "evaluation_samples": self.evaluation_samples,
            "evaluation_steps": self.evaluation_steps,
            "figure_samples": self.figure_samples,
            "figure_trajectories": self.figure_trajectories,
            "figure_grid_resolution": self.figure_grid_resolution,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.rollout_steps < 2 or self.evaluation_steps < 2:
            raise ValueError("rollout_steps and evaluation_steps must be at least two")
        if self.figure_trajectories > self.figure_samples:
            raise ValueError("figure_trajectories cannot exceed figure_samples")
        positive_floats = {
            "guidance_scale": self.guidance_scale,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "rho": self.rho,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "mix_beta": self.mix_beta,
            "advantage_clip": self.advantage_clip,
            "reward_std_epsilon": self.reward_std_epsilon,
            "reward_distance_scale": self.reward_distance_scale,
            "near_branch_threshold": self.near_branch_threshold,
        }
        for name, value in positive_floats.items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.sigma_min >= self.sigma_max:
            raise ValueError("sigma_min must be smaller than sigma_max")
        if not 0 < self.timestep_fraction <= 1:
            raise ValueError("timestep_fraction must lie in (0, 1]")
        for name, value in {
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "ema_decay": self.ema_decay,
            "old_policy_decay": self.old_policy_decay,
        }.items():
            if not math.isfinite(float(value)) or not 0 <= value < 1:
                raise ValueError(f"{name} must lie in [0, 1)")
        # Resolve the selector now so malformed but syntactically valid paths
        # fail before any model is loaded or output directory is mutated.
        for branch_class, branch_path, branch_levels in reward_branch_selection_specs(
            self
        ):
            select_branch_segments(branch_class, branch_path, branch_levels)


@dataclass
class ToyModelBundle:
    good: torch.nn.Module
    bad: torch.nn.Module
    good_old: torch.nn.Module
    bad_old: torch.nn.Module
    good_ema: torch.nn.Module
    bad_ema: torch.nn.Module


def reward_branch_specs(config: ToyNFTConfig) -> tuple[tuple[str, str], ...]:
    """Return the primary and optional secondary reward subtrees."""

    specs = [(config.branch_class, config.branch_path)]
    if config.secondary_branch_class is not None:
        assert config.secondary_branch_path is not None
        specs.append(
            (config.secondary_branch_class, config.secondary_branch_path)
        )
    return tuple(specs)


def reward_branch_selection_specs(
    config: ToyNFTConfig,
) -> tuple[tuple[str, str, int | None], ...]:
    """Return reward branch roots together with optional level limits."""

    specs = [(config.branch_class, config.branch_path, config.branch_levels)]
    if config.secondary_branch_class is not None:
        assert config.secondary_branch_path is not None
        specs.append(
            (
                config.secondary_branch_class,
                config.secondary_branch_path,
                config.secondary_branch_levels,
            )
        )
    return tuple(specs)


def reward_branch_label(config: ToyNFTConfig) -> str:
    return "+".join(
        f"{branch_class}:{branch_path}"
        + ("" if branch_levels is None else f"@{branch_levels}")
        for branch_class, branch_path, branch_levels in reward_branch_selection_specs(
            config
        )
    )


def select_branch_segments(
    branch_class: str,
    branch_path: str,
    branch_levels: int | None = None,
):
    """Select a branch root and optionally a bounded number of edge levels.

    ``branch_levels=None`` preserves the original all-descendants behavior.
    A value of three includes path lengths ``len(branch_path)`` through
    ``len(branch_path) + 2`` and therefore selects 1 + 2 + 4 edges.
    """

    if branch_class not in {"A", "B"}:
        raise ValueError("branch_class must be A or B")
    if not branch_path or set(branch_path) - {"L", "R"}:
        raise ValueError("branch_path must be a nonempty L/R string")
    if len(branch_path) > 6:
        raise ValueError("branch_path cannot be deeper than six turns")
    if branch_levels is not None and (
        not isinstance(branch_levels, int) or branch_levels <= 0
    ):
        raise ValueError("branch_levels must be a positive integer or null")
    maximum_path_length = max(
        len(segment.path) for segment in tree_segments(branch_class)
    )
    if (
        branch_levels is not None
        and len(branch_path) + branch_levels - 1 > maximum_path_length
    ):
        raise ValueError(
            f"branch_levels={branch_levels} extends beyond tree branch "
            f"{branch_class}:{branch_path}"
        )
    maximum_selected_length = (
        None
        if branch_levels is None
        else len(branch_path) + branch_levels - 1
    )
    selected = tuple(
        segment
        for segment in tree_segments(branch_class)
        if segment.path.startswith(branch_path)
        and segment.depth >= len(branch_path)
        and (
            maximum_selected_length is None
            or len(segment.path) <= maximum_selected_length
        )
    )
    if not selected or not any(segment.path == branch_path for segment in selected):
        raise ValueError(
            f"tree branch {branch_class}:{branch_path} does not exist"
        )
    return selected


def _segment_tensors(
    branch_class: str,
    branch_path: str,
    *,
    branch_levels: int | None = None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = select_branch_segments(branch_class, branch_path, branch_levels)
    starts = torch.tensor(
        [segment.start for segment in selected], device=device, dtype=dtype
    )
    ends = torch.tensor(
        [segment.end for segment in selected], device=device, dtype=dtype
    )
    return starts, ends


def distance_to_segments(
    points: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    """Return each point's Euclidean distance to a union of line segments."""

    distance, _assignment = distance_and_assignment_to_segments(
        points, starts, ends
    )
    return distance


def distance_and_assignment_to_segments(
    points: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return minimum finite-segment distance and stable nearest-edge index."""

    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [..., 2]")
    if starts.ndim != 2 or starts.shape[-1] != 2 or starts.shape != ends.shape:
        raise ValueError("starts and ends must have the same [segments, 2] shape")
    if starts.shape[0] == 0:
        raise ValueError("at least one segment is required")
    flat = points.reshape(-1, 2)
    vectors = ends - starts
    squared_lengths = vectors.square().sum(dim=-1).clamp_min(1e-20)
    offsets = flat[:, None, :] - starts[None, :, :]
    projections = (
        (offsets * vectors[None, :, :]).sum(dim=-1) / squared_lengths[None, :]
    ).clamp(0, 1)
    closest = starts[None, :, :] + projections[..., None] * vectors[None, :, :]
    squared = (flat[:, None, :] - closest).square().sum(dim=-1)
    minimum, assignment = squared.min(dim=1)
    output_shape = points.shape[:-1]
    return (
        minimum.clamp_min(0).sqrt().reshape(output_shape),
        assignment.reshape(output_shape),
    )


def branch_distance_reward(
    points: torch.Tensor,
    branch_class: str,
    branch_path: str,
    *,
    secondary_branch_class: str | None = None,
    secondary_branch_path: str | None = None,
    branch_levels: int | None = None,
    secondary_branch_levels: int | None = None,
    reward_maximum: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reward and distance to one subtree or a two-subtree union."""

    if (secondary_branch_class is None) != (secondary_branch_path is None):
        raise ValueError("secondary branch class and path must be set together")
    if secondary_branch_class is None and secondary_branch_levels is not None:
        raise ValueError(
            "secondary_branch_levels requires a secondary reward branch"
        )
    specs = [(branch_class, branch_path, branch_levels)]
    if secondary_branch_class is not None:
        assert secondary_branch_path is not None
        specs.append(
            (
                secondary_branch_class,
                secondary_branch_path,
                secondary_branch_levels,
            )
        )
    tensors = [
        _segment_tensors(
            selected_class,
            selected_path,
            branch_levels=selected_levels,
            device=points.device,
            dtype=points.dtype,
        )
        for selected_class, selected_path, selected_levels in specs
    ]
    starts = torch.cat([value[0] for value in tensors])
    ends = torch.cat([value[1] for value in tensors])
    distance = distance_to_segments(points, starts, ends)
    return float(reward_maximum) - distance, distance


def reward_from_branch_distance(
    distance: torch.Tensor | np.ndarray,
    reward: str,
    reward_distance_scale: float = 1.0,
) -> torch.Tensor | np.ndarray:
    """Map distance to the configured reward without changing its shape."""

    if (
        not math.isfinite(float(reward_distance_scale))
        or reward_distance_scale <= 0
    ):
        raise ValueError("reward_distance_scale must be positive and finite")
    if reward == "negative_branch_distance":
        return -distance
    if reward == "one_minus_branch_distance":
        return 1.0 - distance
    if reward == "exponential_branch_distance":
        scaled = distance / float(reward_distance_scale)
        if isinstance(distance, torch.Tensor):
            return torch.exp(-scaled.square())
        return np.exp(-np.square(scaled))
    raise ValueError(f"unsupported branch-distance reward: {reward}")


def branch_distance_reward_formula(
    reward: str, reward_distance_scale: float = 1.0
) -> str:
    """Return a stable, human-readable formula for run metadata."""

    if reward == "negative_branch_distance":
        return "-distance_to_target_union"
    if reward == "one_minus_branch_distance":
        return "1 - distance_to_target_union"
    if reward == "exponential_branch_distance":
        return (
            "exp(-(distance_to_target_union / "
            f"{reward_distance_scale:g})^2)"
        )
    raise ValueError(f"unsupported branch-distance reward: {reward}")


def configured_reward_formula(config: ToyNFTConfig) -> str:
    return branch_distance_reward_formula(
        config.reward,
        config.reward_distance_scale,
    )


def configured_branch_distance_reward(
    points: torch.Tensor, config: ToyNFTConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the configured one- or two-subtree distance reward."""

    _, distance = branch_distance_reward(
        points,
        config.branch_class,
        config.branch_path,
        secondary_branch_class=config.secondary_branch_class,
        secondary_branch_path=config.secondary_branch_path,
        branch_levels=config.branch_levels,
        secondary_branch_levels=config.secondary_branch_levels,
    )
    rewards = reward_from_branch_distance(
        distance,
        config.reward,
        config.reward_distance_scale,
    )
    assert isinstance(rewards, torch.Tensor)
    return rewards, distance


def _sigma_grid(config: ToyNFTConfig, device: torch.device) -> torch.Tensor:
    indices = torch.arange(config.rollout_steps, device=device, dtype=torch.float32)
    return (
        config.sigma_max ** (1 / config.rho)
        + indices
        / (config.rollout_steps - 1)
        * (
            config.sigma_min ** (1 / config.rho)
            - config.sigma_max ** (1 / config.rho)
        )
    ) ** config.rho


def _training_timestep_count(config: ToyNFTConfig) -> int:
    return max(1, int(config.rollout_steps * config.timestep_fraction))


def _load_pickle_model(source: str, device: torch.device) -> torch.nn.Module:
    with dnnlib.util.open_url(source, verbose=True) as handle:
        model = pickle.load(handle)
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "score"):
        raise TypeError(f"{source!r} is not a compatible toy score model")
    return model.to(device).eval()


def _frozen_copy(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model).eval().requires_grad_(False)


def load_model_bundle(config: ToyNFTConfig, device: torch.device) -> ToyModelBundle:
    """Load main/reference models and initialize NFT snapshots."""

    good_initial = _load_pickle_model(config.good_model, device)
    bad_initial = _load_pickle_model(config.bad_model, device)
    good = copy.deepcopy(good_initial).eval().requires_grad_(True)
    bad = copy.deepcopy(bad_initial).eval().requires_grad_(
        config.objective_mode != "good_only"
    )
    bundle = ToyModelBundle(
        good=good,
        bad=bad,
        good_old=_frozen_copy(good_initial),
        bad_old=_frozen_copy(bad_initial),
        good_ema=_frozen_copy(good_initial),
        bad_ema=_frozen_copy(bad_initial),
    )
    del good_initial, bad_initial
    probe = torch.zeros(2, 2, device=device)
    sigma = torch.ones(2, device=device)
    for name in ("good", "bad", "good_old", "bad_old"):
        output = getattr(bundle, name).score(probe, sigma)
        if output.shape != probe.shape or not torch.isfinite(output).all():
            raise ValueError(f"{name} model failed the toy score interface check")
    return bundle


def _guided_sample(
    good: torch.nn.Module,
    bad: torch.nn.Module,
    initial: torch.Tensor,
    config: ToyNFTConfig,
    *,
    num_steps: int,
) -> torch.Tensor:
    return do_sample(
        net=good,
        gnet=bad,
        guidance=config.guidance_scale,
        x_init=initial,
        num_steps=num_steps,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
    )


def _single_model_sample(
    model: torch.nn.Module,
    initial: torch.Tensor,
    config: ToyNFTConfig,
    *,
    num_steps: int,
) -> torch.Tensor:
    return do_sample(
        net=model,
        x_init=initial,
        num_steps=num_steps,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
    )


def _sample_initial(
    count: int, config: ToyNFTConfig, device: torch.device, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    sampling_classes = config.sampling_classes or config.branch_class
    return gt(sampling_classes, device).sample(
        count, config.sigma_max, generator=generator
    )


def collect_rollout(
    bundle: ToyModelBundle,
    config: ToyNFTConfig,
    *,
    outer_step: int,
    device: torch.device,
) -> dict[str, torch.Tensor | float | str]:
    initial = _sample_initial(
        config.rollout_samples,
        config,
        device,
        config.seed + outer_step * 1_000_003,
    )
    if config.objective_mode == "good_only":
        trajectory = _single_model_sample(
            bundle.good_ema,
            initial,
            config,
            num_steps=config.rollout_steps,
        )
        rollout_policy = "good_only"
    else:
        trajectory = _guided_sample(
            bundle.good_ema,
            bundle.bad_ema,
            initial,
            config,
            num_steps=config.rollout_steps,
        )
        rollout_policy = "guided"
    endpoints = trajectory[-1].detach()
    rewards, distances = configured_branch_distance_reward(endpoints, config)
    reward_std = rewards.float().std(unbiased=False)
    advantages = (rewards.float() - rewards.float().mean()) / (
        reward_std + config.reward_std_epsilon
    )
    clipped, reward_weight = advantage_to_reward_weight(
        advantages, config.advantage_clip
    )
    return {
        "endpoints": endpoints,
        "rewards": rewards.detach(),
        "distances": distances.detach(),
        "advantages": advantages.detach(),
        "clipped_advantages": clipped.detach(),
        "reward_weight": reward_weight.detach(),
        "rollout_policy": rollout_policy,
        "reward_mean": float(rewards.mean().item()),
        "reward_std": float(reward_std.item()),
        "distance_mean": float(distances.mean().item()),
        "distance_median": float(distances.median().item()),
        "near_branch_rate": float(
            (distances <= config.near_branch_threshold).float().mean().item()
        ),
    }


def _denoised_output(
    model: torch.nn.Module,
    state: torch.Tensor,
    sigma: torch.Tensor,
    *,
    graph: bool,
) -> torch.Tensor:
    score = model.score(state, sigma, graph=graph)
    return state + score * sigma[:, None].square()


def _toy_branch_loss(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-5
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("toy branch prediction and target must match [batch, dim]")
    residual = prediction.float() - target.float()
    denominator = residual.detach().abs().mean(dim=1).clamp_min(epsilon)
    return residual.square().mean(dim=1) / denominator


def _parameter_delta_rms(
    current: Iterable[torch.nn.Parameter], old: Iterable[torch.nn.Parameter]
) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    count = 0
    for current_value, old_value in zip(current, old, strict=True):
        delta = current_value.detach().double().cpu() - old_value.detach().double().cpu()
        squared += delta.square().sum()
        count += delta.numel()
    return float((squared / max(count, 1)).sqrt().item())


def _objective_for_batch(
    bundle: ToyModelBundle,
    config: ToyNFTConfig,
    *,
    state: torch.Tensor,
    sigma: torch.Tensor,
    endpoint: torch.Tensor,
    reward_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    good_old = _denoised_output(
        bundle.good_old, state, sigma, graph=False
    ).detach()
    good_current = _denoised_output(bundle.good, state, sigma, graph=True)

    if config.objective_mode == "good_only":
        positive, negative = compose_standard_nft_branches(
            current=good_current,
            old=good_old,
            beta=config.mix_beta,
        )
        positive_loss = _toy_branch_loss(positive, endpoint)
        negative_loss = _toy_branch_loss(negative, endpoint)
        policy = nft_policy_loss_per_example(
            positive_loss,
            negative_loss,
            reward_weight,
            beta=config.mix_beta,
            advantage_clip=config.advantage_clip,
        )
        return policy, {
            "positive": positive_loss,
            "negative": negative_loss,
        }

    bad_old = _denoised_output(
        bundle.bad_old, state, sigma, graph=False
    ).detach()
    center = compose_autoguidance(
        good_old, bad_old, config.guidance_scale
    ).detach()
    bad_current = _denoised_output(bundle.bad, state, sigma, graph=True)

    if config.objective_mode == "guided_field":
        current = compose_autoguidance(
            good_current, bad_current, config.guidance_scale
        )
        positive, negative = compose_standard_nft_branches(
            current=current,
            old=center,
            beta=config.mix_beta,
        )
        positive_loss = _toy_branch_loss(positive, endpoint)
        negative_loss = _toy_branch_loss(negative, endpoint)
        policy = nft_policy_loss_per_example(
            positive_loss,
            negative_loss,
            reward_weight,
            beta=config.mix_beta,
            advantage_clip=config.advantage_clip,
        )
        return policy, {
            "positive": positive_loss,
            "negative": negative_loss,
        }

    policies = []
    positive_losses = []
    negative_losses = []
    for current, old in (
        (good_current, good_old),
        (bad_current, bad_old),
    ):
        positive, negative = compose_shifted_nft_branches(
            current=current,
            old=old,
            center=center,
            beta=config.mix_beta,
        )
        positive_loss = _toy_branch_loss(positive, endpoint)
        negative_loss = _toy_branch_loss(negative, endpoint)
        policies.append(
            nft_policy_loss_per_example(
                positive_loss,
                negative_loss,
                reward_weight,
                beta=config.mix_beta,
                advantage_clip=config.advantage_clip,
            )
        )
        positive_losses.append(positive_loss)
        negative_losses.append(negative_loss)
    return torch.stack(policies).mean(dim=0), {
        "positive": torch.stack(positive_losses).mean(dim=0),
        "negative": torch.stack(negative_losses).mean(dim=0),
        "good_policy": policies[0],
        "bad_policy": policies[1],
    }


def _optimized_parameters(
    bundle: ToyModelBundle, config: ToyNFTConfig
) -> list[torch.nn.Parameter]:
    parameters = list(bundle.good.parameters())
    if config.objective_mode != "good_only":
        parameters.extend(bundle.bad.parameters())
    return parameters


def _gradient_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().double().square().sum().cpu()
    return float(squared.sqrt().item())


def train_outer_step(
    bundle: ToyModelBundle,
    config: ToyNFTConfig,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor | float | str],
    *,
    outer_step: int,
    device: torch.device,
) -> dict[str, float | int | bool]:
    """Apply one optimizer update over a sealed rollout/reward buffer."""

    endpoints = rollout["endpoints"]
    reward_weight = rollout["reward_weight"]
    assert isinstance(endpoints, torch.Tensor)
    assert isinstance(reward_weight, torch.Tensor)
    endpoint_count = endpoints.shape[0]
    timestep_count = _training_timestep_count(config)
    sigma_grid = _sigma_grid(config, device)
    sigma_generator = torch.Generator(device=device).manual_seed(
        config.seed + outer_step * 10_000_019 + 17
    )
    selected = torch.stack(
        [
            torch.randperm(
                sigma_grid.numel(), device=device, generator=sigma_generator
            )[:timestep_count]
            for _ in range(endpoint_count)
        ]
    )
    sigmas = sigma_grid[selected].reshape(-1)
    record_count = endpoint_count * timestep_count
    total_count = record_count * config.inner_epochs
    optimizer.zero_grad(set_to_none=True)
    sums = {
        "policy": torch.zeros((), device=device, dtype=torch.float64),
        "positive": torch.zeros((), device=device, dtype=torch.float64),
        "negative": torch.zeros((), device=device, dtype=torch.float64),
    }
    noising_generator = torch.Generator(device=device).manual_seed(
        config.seed + outer_step * 10_000_019 + 29
    )
    permutation_generator = torch.Generator(device=device).manual_seed(
        config.seed + outer_step * 10_000_019 + 43
    )
    for _epoch in range(config.inner_epochs):
        permutation = torch.randperm(
            record_count, device=device, generator=permutation_generator
        )
        for start in range(0, record_count, config.train_batch_size):
            record_indices = permutation[start : start + config.train_batch_size]
            endpoint_indices = torch.div(
                record_indices, timestep_count, rounding_mode="floor"
            )
            target = endpoints[endpoint_indices]
            sigma = sigmas[record_indices]
            noise = torch.randn(
                target.shape,
                device=device,
                dtype=target.dtype,
                generator=noising_generator,
            )
            state = target + sigma[:, None] * noise
            policy, branches = _objective_for_batch(
                bundle,
                config,
                state=state,
                sigma=sigma,
                endpoint=target,
                reward_weight=reward_weight[endpoint_indices],
            )
            scaled_loss = policy.mean() * (target.shape[0] / total_count)
            scaled_loss.backward()
            sums["policy"] += policy.detach().double().sum()
            sums["positive"] += branches["positive"].detach().double().sum()
            sums["negative"] += branches["negative"].detach().double().sum()

    parameters = _optimized_parameters(bundle, config)
    good_grad_norm = _gradient_l2_norm(bundle.good.parameters())
    bad_grad_norm = _gradient_l2_norm(bundle.bad.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError("toy NFT gradient norm is non-finite")
    optimizer.step()
    good_delta_rms = _parameter_delta_rms(
        bundle.good.parameters(), bundle.good_old.parameters()
    )
    bad_delta_rms = _parameter_delta_rms(
        bundle.bad.parameters(), bundle.bad_old.parameters()
    )
    good_parameters_changed = good_delta_rms > 0
    bad_parameters_changed = bad_delta_rms > 0
    frozen_reference_unchanged = bad_delta_rms == 0
    if config.objective_mode == "good_only":
        parameters_changed = good_parameters_changed and frozen_reference_unchanged
    else:
        parameters_changed = good_parameters_changed and bad_parameters_changed
    update_ema_(bundle.good_ema, bundle.good, config.ema_decay)
    update_ema_(bundle.good_old, bundle.good, config.old_policy_decay)
    if config.objective_mode != "good_only":
        update_ema_(bundle.bad_ema, bundle.bad, config.ema_decay)
        update_ema_(bundle.bad_old, bundle.bad, config.old_policy_decay)
    return {
        "optimizer_step": outer_step,
        "policy_loss": float((sums["policy"] / total_count).item()),
        "positive_loss": float((sums["positive"] / total_count).item()),
        "negative_loss": float((sums["negative"] / total_count).item()),
        "grad_norm": float(grad_norm),
        "good_grad_norm": good_grad_norm,
        "bad_grad_norm": bad_grad_norm,
        "grad_clipped": float(grad_norm) > config.max_grad_norm,
        "good_parameter_delta_rms": good_delta_rms,
        "bad_parameter_delta_rms": bad_delta_rms,
        "good_parameters_changed": good_parameters_changed,
        "bad_parameters_changed": bad_parameters_changed,
        "frozen_reference_unchanged": frozen_reference_unchanged,
        "parameters_changed": parameters_changed,
        "training_records": total_count,
        "training_timesteps_per_endpoint": timestep_count,
        "training_sigma_min": float(sigmas.min().item()),
        "training_sigma_max": float(sigmas.max().item()),
    }


def evaluate_models(
    good: torch.nn.Module,
    bad: torch.nn.Module,
    config: ToyNFTConfig,
    *,
    outer_step: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    initial = _sample_initial(
        config.evaluation_samples,
        config,
        device,
        config.seed + 90_000_001,
    )
    good_endpoints = _single_model_sample(
        good,
        initial,
        config,
        num_steps=config.evaluation_steps,
    )[-1].detach()
    guided_endpoints = _guided_sample(
        good,
        bad,
        initial,
        config,
        num_steps=config.evaluation_steps,
    )[-1].detach()

    def statistics(endpoints: torch.Tensor) -> dict[str, float]:
        rewards, distances = configured_branch_distance_reward(endpoints, config)
        return {
            "reward_mean": float(rewards.mean().item()),
            "distance_mean": float(distances.mean().item()),
            "distance_median": float(distances.median().item()),
            "near_branch_rate": float(
                (distances <= config.near_branch_threshold).float().mean().item()
            ),
        }

    good_statistics = statistics(good_endpoints)
    guided_statistics = statistics(guided_endpoints)
    active_statistics = (
        good_statistics
        if config.objective_mode == "good_only"
        else guided_statistics
    )
    result: dict[str, float | int | str] = {
        "outer_step": outer_step,
        "evaluation_policy": (
            "good_only" if config.objective_mode == "good_only" else "guided"
        ),
    }
    result.update(
        {f"evaluation_{name}": value for name, value in active_statistics.items()}
    )
    result.update(
        {
            f"evaluation_good_{name}": value
            for name, value in good_statistics.items()
        }
    )
    result.update(
        {
            f"evaluation_guided_{name}": value
            for name, value in guided_statistics.items()
        }
    )
    return result


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_payload(
    bundle: ToyModelBundle,
    optimizer: torch.optim.Optimizer,
    config: ToyNFTConfig,
    completed_outer_step: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "completed_outer_step": completed_outer_step,
        "config": dataclasses.asdict(config),
        "models": {
            name: getattr(bundle, name).state_dict()
            for name in (
                "good",
                "bad",
                "good_old",
                "bad_old",
                "good_ema",
                "bad_ema",
            )
        },
        "optimizer": optimizer.state_dict(),
    }


def _save_checkpoint(
    output_dir: Path,
    bundle: ToyModelBundle,
    optimizer: torch.optim.Optimizer,
    config: ToyNFTConfig,
    completed_outer_step: int,
) -> Path:
    payload = _checkpoint_payload(bundle, optimizer, config, completed_outer_step)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step_path = checkpoint_dir / f"step-{completed_outer_step:06d}.pt"
    step_temporary = step_path.with_suffix(step_path.suffix + ".tmp")
    torch.save(payload, step_temporary)
    os.replace(step_temporary, step_path)

    latest_path = output_dir / "checkpoint.pt"
    latest_temporary = output_dir / "checkpoint.pt.tmp"
    torch.save(payload, latest_temporary)
    os.replace(latest_temporary, latest_path)
    _atomic_json(
        checkpoint_dir / "latest.json",
        {
            "completed_outer_step": completed_outer_step,
            "checkpoint": step_path.name,
        },
    )
    return latest_path


def _load_checkpoint_bundle(
    run_dir: Path, device: torch.device
) -> tuple[ToyNFTConfig, ToyModelBundle, int]:
    config = ToyNFTConfig.from_json(run_dir / "config.json")
    bundle = load_model_bundle(config, device)
    payload = torch.load(
        run_dir / "checkpoint.pt", map_location=device, weights_only=True
    )
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported toy NFT checkpoint schema")
    checkpoint_config = ToyNFTConfig(**payload.get("config", {}))
    checkpoint_config.validate()
    if dataclasses.asdict(checkpoint_config) != dataclasses.asdict(config):
        raise ValueError("checkpoint and run config disagree")
    for name, state in payload["models"].items():
        getattr(bundle, name).load_state_dict(state, strict=True)
    return config, bundle, int(payload["completed_outer_step"])


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_training(
    config: ToyNFTConfig,
    output_dir: Path,
    device: torch.device,
) -> tuple[ToyModelBundle, list[dict[str, Any]]]:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "config.json", dataclasses.asdict(config))
    bundle = load_model_bundle(config, device)
    optimizer = torch.optim.Adam(
        _optimized_parameters(bundle, config),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    initial_eval = evaluate_models(
        bundle.good_ema,
        bundle.bad_ema,
        config,
        outer_step=0,
        device=device,
    )
    _append_jsonl(
        output_dir / "metrics.jsonl", {"stage": "evaluation", **initial_eval}
    )
    records: list[dict[str, Any]] = []
    for outer_step in range(1, config.outer_steps + 1):
        started = time.time()
        rollout = collect_rollout(
            bundle, config, outer_step=outer_step, device=device
        )
        training = train_outer_step(
            bundle,
            config,
            optimizer,
            rollout,
            outer_step=outer_step,
            device=device,
        )
        record: dict[str, Any] = {
            "stage": "train",
            "outer_step": outer_step,
            "objective_mode": config.objective_mode,
            "rollout_policy": rollout["rollout_policy"],
            "branch": reward_branch_label(config),
            "reward": config.reward,
            "reward_distance_scale": config.reward_distance_scale,
            "reward_formula": configured_reward_formula(config),
            "rollout_reward_mean": rollout["reward_mean"],
            "rollout_reward_std": rollout["reward_std"],
            "rollout_distance_mean": rollout["distance_mean"],
            "rollout_distance_median": rollout["distance_median"],
            "rollout_near_branch_rate": rollout["near_branch_rate"],
            **training,
            "seconds": time.time() - started,
        }
        _append_jsonl(output_dir / "metrics.jsonl", record)
        records.append(record)
        print(
            "TRAIN "
            f"outer_step={outer_step}/{config.outer_steps} "
            f"loss={record['policy_loss']:.6f} "
            f"grad_norm={record['grad_norm']:.6f} "
            f"good_delta={record['good_parameter_delta_rms']:.8f} "
            f"bad_delta={record['bad_parameter_delta_rms']:.8f} "
            f"reference_frozen={record['frozen_reference_unchanged']} "
            f"distance={record['rollout_distance_mean']:.6f}",
            flush=True,
        )
        if outer_step % config.evaluation_every == 0 or outer_step == config.outer_steps:
            evaluation = evaluate_models(
                bundle.good_ema,
                bundle.bad_ema,
                config,
                outer_step=outer_step,
                device=device,
            )
            _append_jsonl(
                output_dir / "metrics.jsonl",
                {"stage": "evaluation", **evaluation},
            )
            print(
                "EVAL "
                f"outer_step={outer_step} "
                f"mean_distance={evaluation['evaluation_distance_mean']:.6f} "
                f"near_branch_rate={evaluation['evaluation_near_branch_rate']:.6f}",
                flush=True,
            )
        _save_checkpoint(output_dir, bundle, optimizer, config, outer_step)
    if not all(record["parameters_changed"] for record in records):
        expected = (
            "the main model while preserving the frozen reference"
            if config.objective_mode == "good_only"
            else "both Autoguidance models"
        )
        raise RuntimeError(f"an optimizer step did not change {expected}")
    print(
        f"TRAINING_CONFIRMED completed_outer_steps={config.outer_steps} "
        f"optimizer_steps={len(records)}",
        flush=True,
    )
    return bundle, records


def _draw_tree(
    axis: plt.Axes,
    config: ToyNFTConfig,
    *,
    selected_color: str = "#74B3D2",
) -> None:
    selected = {
        (segment.cls, segment.path)
        for branch_class, branch_path, branch_levels in reward_branch_selection_specs(
            config
        )
        for segment in select_branch_segments(
            branch_class, branch_path, branch_levels
        )
    }
    for segment in tree_segments("AB"):
        is_selected = (segment.cls, segment.path) in selected
        color = selected_color if is_selected else (
            "#D9D9D9" if segment.cls == "B" else "#E8C7A6"
        )
        linewidth = 2.6 if is_selected else (1.25 if segment.cls == "A" else 0.9)
        alpha = 0.9 if is_selected else 0.75
        axis.plot(
            [segment.start[0], segment.end[0]],
            [segment.start[1], segment.end[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2 if is_selected else 1,
        )


def _spatial_limits(
    config: ToyNFTConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if config.secondary_branch_class is None:
        return (-0.62, 1.42), (-0.75, 1.58)
    segments = tree_segments("AB")
    x_values = [
        value
        for segment in segments
        for value in (segment.start[0], segment.end[0])
    ]
    y_values = [
        value
        for segment in segments
        for value in (segment.start[1], segment.end[1])
    ]
    x_span = max(x_values) - min(x_values)
    y_span = max(y_values) - min(y_values)
    return (
        (min(x_values) - 0.06 * x_span, max(x_values) + 0.06 * x_span),
        (min(y_values) - 0.06 * y_span, max(y_values) + 0.06 * y_span),
    )


def _style_spatial_axis(axis: plt.Axes, config: ToyNFTConfig) -> None:
    x_limits, y_limits = _spatial_limits(config)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _panel_title(axis: plt.Axes, letter: str, title: str) -> None:
    axis.set_title(f"({letter}) {title}", loc="left", fontweight="semibold", pad=7)


def _trajectory_data(
    config: ToyNFTConfig,
    initial_bundle: ToyModelBundle,
    final_bundle: ToyModelBundle,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    initial = _sample_initial(
        config.figure_samples,
        config,
        device,
        config.seed + 90_000_001,
    )
    before = _guided_sample(
        initial_bundle.good_ema,
        initial_bundle.bad_ema,
        initial,
        config,
        num_steps=config.evaluation_steps,
    ).detach()
    after = _guided_sample(
        final_bundle.good_ema,
        final_bundle.bad_ema,
        initial,
        config,
        num_steps=config.evaluation_steps,
    ).detach()
    return initial, before, after


def _good_only_trajectory_data(
    config: ToyNFTConfig,
    initial_bundle: ToyModelBundle,
    final_bundle: ToyModelBundle,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample the two-by-two NFT/guidance comparison from shared noise."""

    initial = _sample_initial(
        config.figure_samples,
        config,
        device,
        config.seed + 90_000_001,
    )
    pretrained_a64 = _single_model_sample(
        initial_bundle.good_ema,
        initial,
        config,
        num_steps=config.evaluation_steps,
    ).detach()
    nft_a64 = _single_model_sample(
        final_bundle.good_ema,
        initial,
        config,
        num_steps=config.evaluation_steps,
    ).detach()
    pretrained_cfg = _guided_sample(
        initial_bundle.good_ema,
        initial_bundle.bad_ema,
        initial,
        config,
        num_steps=config.evaluation_steps,
    ).detach()
    nft_cfg = _guided_sample(
        final_bundle.good_ema,
        final_bundle.bad_ema,
        initial,
        config,
        num_steps=config.evaluation_steps,
    ).detach()
    return {
        "pretrained_a64": pretrained_a64,
        "nft_a64": nft_a64,
        "pretrained_cfg": pretrained_cfg,
        "nft_cfg": nft_cfg,
    }


def _make_good_only_comparison_figure(
    config: ToyNFTConfig,
    initial_bundle: ToyModelBundle | None,
    final_bundle: ToyModelBundle | None,
    metrics: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    *,
    source_figure_data: Path | None,
) -> dict[str, Any]:
    """Render A64 NFT before/after, with and without a fixed reference."""

    condition_names = (
        "pretrained_a64",
        "nft_a64",
        "pretrained_cfg",
        "nft_cfg",
    )
    distance_names = tuple(f"{name}_distance" for name in condition_names)
    if source_figure_data is None:
        if initial_bundle is None or final_bundle is None:
            raise ValueError("model bundles are required when figure data is resampled")
        trajectories = _good_only_trajectory_data(
            config, initial_bundle, final_bundle, device
        )
        endpoints = {name: trajectories[name][-1] for name in condition_names}
        distances = {
            f"{name}_distance": configured_branch_distance_reward(
                endpoints[name], config
            )[1]
            for name in condition_names
        }
        trajectory_indices = torch.argsort(distances["nft_a64_distance"])[
            : config.figure_trajectories
        ]
        selected_trajectories = {
            "nft_a64_trajectory": trajectories["nft_a64"][:, trajectory_indices],
            "nft_cfg_trajectory": trajectories["nft_cfg"][:, trajectory_indices],
        }
    else:
        required = {
            *condition_names,
            *distance_names,
            "trajectory_indices",
            "nft_a64_trajectory",
            "nft_cfg_trajectory",
        }
        with np.load(source_figure_data) as source:
            missing = sorted(required - set(source.files))
            if missing:
                raise ValueError(f"source figure data is missing arrays: {missing}")
            endpoints = {
                name: torch.from_numpy(source[name].copy()) for name in condition_names
            }
            distances = {
                name: torch.from_numpy(source[name].copy()) for name in distance_names
            }
            trajectory_indices = torch.from_numpy(
                source["trajectory_indices"].copy()
            ).to(torch.long)
            selected_trajectories = {
                name: torch.from_numpy(source[name].copy())
                for name in ("nft_a64_trajectory", "nft_cfg_trajectory")
            }
        endpoint_shape = (config.figure_samples, 2)
        distance_shape = (config.figure_samples,)
        trajectory_shape = (
            config.evaluation_steps + 1,
            config.figure_trajectories,
            2,
        )
        if any(tensor.shape != endpoint_shape for tensor in endpoints.values()):
            raise ValueError("source comparison endpoints disagree with the config")
        if any(tensor.shape != distance_shape for tensor in distances.values()):
            raise ValueError("source comparison distances disagree with the config")
        if any(
            tensor.shape != trajectory_shape
            for tensor in selected_trajectories.values()
        ) or trajectory_indices.shape != (config.figure_trajectories,):
            raise ValueError("source comparison trajectories disagree with the config")

    data_path = output_dir / "figure_data.npz"
    np.savez_compressed(
        data_path,
        **{
            **{name: tensor.cpu().numpy() for name, tensor in endpoints.items()},
            **{name: tensor.cpu().numpy() for name, tensor in distances.items()},
            "trajectory_indices": trajectory_indices.cpu().numpy(),
            **{
                name: tensor.cpu().numpy()
                for name, tensor in selected_trajectories.items()
            },
        },
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2, 4, figsize=(14.2, 7.0), constrained_layout=True
    )
    axes = axes.reshape(2, 4)
    selected_color = "#74B3D2"
    colors = {
        "pretrained_a64": "#666666",
        "nft_a64": "#D55E00",
        "pretrained_cfg": "#4C78A8",
        "nft_cfg": "#7A5195",
    }
    reference_label = (
        "AB32"
        if "clsAB" in config.bad_model and "dim32" in config.bad_model
        else "reference"
    )

    axis = axes[0, 0]
    resolution = config.figure_grid_resolution
    x_limits, y_limits = _spatial_limits(config)
    grid_x = torch.linspace(*x_limits, resolution, device=device)
    grid_y = torch.linspace(*y_limits, resolution, device=device)
    grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing="xy"), dim=-1)
    grid_distance = configured_branch_distance_reward(grid, config)[1]
    levels = np.linspace(0, min(0.7, float(grid_distance.max().item())), 18)
    reward_cmap = LinearSegmentedColormap.from_list(
        "branch_distance_light",
        ["#A9D2E5", "#C6E0EC", "#DEEDF4", "#F8FBFD"],
    )
    axis.contourf(
        grid_x.cpu().numpy(),
        grid_y.cpu().numpy(),
        grid_distance.cpu().numpy(),
        levels=levels,
        cmap=reward_cmap,
        alpha=0.68,
        extend="max",
    )
    _draw_tree(axis, config, selected_color=selected_color)
    _style_spatial_axis(axis, config)
    _panel_title(axis, "a", f"distance reward: {reward_branch_label(config)}")

    scatter_panels = (
        (axes[0, 1], "pretrained_a64", "b", "pretrained A64"),
        (
            axes[0, 2],
            "nft_a64",
            "c",
            f"A64 after {config.outer_steps}-step NFT",
        ),
        (
            axes[0, 3],
            "nft_cfg",
            "d",
            f"NFT A64 + fixed {reference_label} CFG",
        ),
        (
            axes[1, 0],
            "pretrained_cfg",
            "e",
            f"pretrained A64 + fixed {reference_label} CFG",
        ),
    )
    for axis, name, letter, title in scatter_panels:
        _draw_tree(axis, config, selected_color=selected_color)
        values = endpoints[name]
        axis.scatter(
            values[:, 0].cpu(),
            values[:, 1].cpu(),
            s=5,
            c=colors[name],
            alpha=0.32,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        _style_spatial_axis(axis, config)
        _panel_title(axis, letter, title)

    labels = {
        "pretrained_a64": "A64",
        "nft_a64": "A64 NFT",
        "pretrained_cfg": f"A64 + {reference_label}",
        "nft_cfg": f"A64 NFT + {reference_label}",
    }
    axis = axes[1, 1]
    maximum = max(float(tensor.max().item()) for tensor in distances.values())
    bins = np.linspace(0, max(maximum, config.near_branch_threshold * 2), 36)
    for name, distance_name in zip(condition_names, distance_names, strict=True):
        axis.hist(
            distances[distance_name].cpu().numpy(),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=colors[name],
            label=labels[name],
        )
    axis.axvline(
        config.near_branch_threshold,
        color=selected_color,
        linestyle="--",
        linewidth=1.1,
    )
    axis.set_xlabel("distance to selected branch")
    axis.set_ylabel("density")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(frameon=False)
    _panel_title(axis, "f", "paired distance distributions")

    evaluations = [row for row in metrics if row.get("stage") == "evaluation"]
    axis = axes[1, 2]
    axis.plot(
        [row["outer_step"] for row in evaluations],
        [row["evaluation_good_distance_mean"] for row in evaluations],
        marker="o",
        color=colors["nft_a64"],
        linewidth=1.8,
        label="A64 alone",
    )
    axis.plot(
        [row["outer_step"] for row in evaluations],
        [row["evaluation_guided_distance_mean"] for row in evaluations],
        marker="s",
        color=colors["nft_cfg"],
        linewidth=1.5,
        label=f"A64 + fixed {reference_label}",
    )
    axis.set_xlabel("outer NFT step")
    axis.set_ylabel("mean distance ↓")
    axis.set_xticks([int(row["outer_step"]) for row in evaluations])
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False)
    _panel_title(
        axis, "g", f"fixed-seed effect over {config.outer_steps} NFT steps"
    )

    means = [float(distances[name].mean().item()) for name in distance_names]
    rates = [
        float(
            (distances[name] <= config.near_branch_threshold).float().mean().item()
        )
        for name in distance_names
    ]
    axis = axes[1, 3]
    x = np.arange(len(condition_names))
    width = 0.38
    axis.bar(
        x - width / 2,
        means,
        width,
        color=[colors[name] for name in condition_names],
        alpha=0.85,
        label="mean distance",
    )
    axis.set_ylabel("mean distance ↓")
    axis.set_xticks(x, [labels[name] for name in condition_names], rotation=18)
    twin = axis.twinx()
    twin.bar(
        x + width / 2,
        rates,
        width,
        color=[colors[name] for name in condition_names],
        alpha=0.28,
        hatch="//",
        label="near-branch rate",
    )
    twin.set_ylabel(f"fraction within {config.near_branch_threshold:g} ↑")
    twin.set_ylim(0, max(0.05, *rates) * 1.35)
    axis.grid(axis="y", alpha=0.22, linewidth=0.5)
    axis.legend(loc="upper left", frameon=False)
    twin.legend(loc="upper right", frameon=False)
    _panel_title(axis, "h", "four-way fixed-seed summary")

    figure.suptitle(
        f"A64-only NFT followed by fixed-{reference_label} classifier-free guidance",
        fontsize=14,
        fontweight="semibold",
    )
    png_path = output_dir / "toy_a64_nft_fixed_ab32.png"
    pdf_path = output_dir / "toy_a64_nft_fixed_ab32.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    training = [row for row in metrics if row.get("stage") == "train"]
    completed_outer_steps = max(int(row["outer_step"]) for row in training)
    reference_unchanged = bool(
        training and all(row.get("frozen_reference_unchanged") for row in training)
    )
    training_confirmed = bool(
        training
        and all(row.get("parameters_changed") for row in training)
        and reference_unchanged
    )
    summary = {
        "formal_experiment": False,
        "branch": reward_branch_label(config),
        "reward": config.reward,
        "reward_distance_scale": config.reward_distance_scale,
        "reward_formula": configured_reward_formula(config),
        "objective_mode": config.objective_mode,
        "training_policy": "A64 only",
        "fixed_reference_model": config.bad_model,
        "guidance_scale": config.guidance_scale,
        "completed_outer_steps": completed_outer_steps,
        "training_confirmed": training_confirmed,
        "fixed_reference_unchanged": reference_unchanged,
        "pretrained_a64_distance_mean": means[0],
        "nft_a64_distance_mean": means[1],
        "pretrained_cfg_distance_mean": means[2],
        "nft_cfg_distance_mean": means[3],
        "a64_nft_distance_mean_change": means[1] - means[0],
        "pretrained_cfg_distance_mean_change": means[2] - means[0],
        "post_nft_cfg_distance_mean_change": means[3] - means[1],
        "combined_distance_mean_change": means[3] - means[0],
        "pretrained_a64_near_branch_rate": rates[0],
        "nft_a64_near_branch_rate": rates[1],
        "pretrained_cfg_near_branch_rate": rates[2],
        "nft_cfg_near_branch_rate": rates[3],
        "a64_nft_near_branch_rate_change": rates[1] - rates[0],
        "post_nft_cfg_near_branch_rate_change": rates[3] - rates[1],
        "figure_png": str(png_path),
        "figure_pdf": str(pdf_path),
        "figure_data": str(data_path),
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(
        "FIGURE "
        f"png={png_path} pdf={pdf_path} "
        f"a64_before={means[0]:.6f} a64_nft={means[1]:.6f} "
        f"a64_nft_cfg={means[3]:.6f}",
        flush=True,
    )
    return summary


def make_paper_figure(
    config: ToyNFTConfig,
    initial_bundle: ToyModelBundle | None,
    final_bundle: ToyModelBundle | None,
    metrics: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    *,
    source_figure_data: Path | None = None,
) -> dict[str, Any]:
    """Create the static eight-panel PNG/PDF and paired plotting data."""

    if config.objective_mode == "good_only":
        return _make_good_only_comparison_figure(
            config,
            initial_bundle,
            final_bundle,
            metrics,
            output_dir,
            device,
            source_figure_data=source_figure_data,
        )

    if source_figure_data is None:
        if initial_bundle is None or final_bundle is None:
            raise ValueError("model bundles are required when figure data is resampled")
        _initial, before_trajectory, after_trajectory = _trajectory_data(
            config, initial_bundle, final_bundle, device
        )
        before = before_trajectory[-1]
        after = after_trajectory[-1]
        _, before_distance = configured_branch_distance_reward(before, config)
        _, after_distance = configured_branch_distance_reward(after, config)
        trajectory_indices = torch.argsort(after_distance)[
            : config.figure_trajectories
        ]
        selected_before_trajectory = before_trajectory[:, trajectory_indices]
        selected_after_trajectory = after_trajectory[:, trajectory_indices]
    else:
        with np.load(source_figure_data) as source:
            required = {
                "before",
                "after",
                "trajectory_indices",
                "before_trajectory",
                "after_trajectory",
                "before_distance",
                "after_distance",
            }
            missing = sorted(required - set(source.files))
            if missing:
                raise ValueError(f"source figure data is missing arrays: {missing}")
            before = torch.from_numpy(source["before"].copy())
            after = torch.from_numpy(source["after"].copy())
            trajectory_indices = torch.from_numpy(
                source["trajectory_indices"].copy()
            ).to(torch.long)
            selected_before_trajectory = torch.from_numpy(
                source["before_trajectory"].copy()
            )
            selected_after_trajectory = torch.from_numpy(
                source["after_trajectory"].copy()
            )
            before_distance = torch.from_numpy(source["before_distance"].copy())
            after_distance = torch.from_numpy(source["after_distance"].copy())
        expected_trajectory_shape = (
            config.evaluation_steps + 1,
            config.figure_trajectories,
            2,
        )
        if before.shape != (config.figure_samples, 2) or after.shape != before.shape:
            raise ValueError("source before/after endpoints disagree with the config")
        if (
            tuple(selected_before_trajectory.shape) != expected_trajectory_shape
            or selected_after_trajectory.shape != selected_before_trajectory.shape
            or trajectory_indices.shape != (config.figure_trajectories,)
        ):
            raise ValueError("source trajectories disagree with the config")
        if (
            before_distance.shape != (config.figure_samples,)
            or after_distance.shape != before_distance.shape
        ):
            raise ValueError("source distances disagree with the config")
    data_path = output_dir / "figure_data.npz"
    np.savez_compressed(
        data_path,
        before=before.cpu().numpy(),
        after=after.cpu().numpy(),
        trajectory_indices=trajectory_indices.cpu().numpy(),
        before_trajectory=selected_before_trajectory.cpu().numpy(),
        after_trajectory=selected_after_trajectory.cpu().numpy(),
        before_distance=before_distance.cpu().numpy(),
        after_distance=after_distance.cpu().numpy(),
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2, 4, figsize=(14.2, 7.0), constrained_layout=True
    )
    axes = axes.reshape(2, 4)
    selected_color = "#74B3D2"
    before_color = "#666666"
    after_color = "#D55E00"

    axis = axes[0, 0]
    resolution = config.figure_grid_resolution
    x_limits, y_limits = _spatial_limits(config)
    grid_x = torch.linspace(*x_limits, resolution, device=device)
    grid_y = torch.linspace(*y_limits, resolution, device=device)
    grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing="xy"), dim=-1)
    _grid_reward, grid_distance = configured_branch_distance_reward(grid, config)
    levels = np.linspace(0, min(0.7, float(grid_distance.max().item())), 18)
    reward_cmap = LinearSegmentedColormap.from_list(
        "branch_distance_light",
        ["#A9D2E5", "#C6E0EC", "#DEEDF4", "#F8FBFD"],
    )
    axis.contourf(
        grid_x.cpu().numpy(),
        grid_y.cpu().numpy(),
        grid_distance.cpu().numpy(),
        levels=levels,
        cmap=reward_cmap,
        alpha=0.68,
        extend="max",
    )
    _draw_tree(axis, config, selected_color=selected_color)
    _style_spatial_axis(axis, config)
    _panel_title(axis, "a", f"distance reward: {reward_branch_label(config)}")

    for axis, endpoints, letter, title, color in (
        (axes[0, 1], before, "b", "pretrained Autoguidance", before_color),
        (axes[0, 2], after, "c", "after guided-field NFT", after_color),
    ):
        _draw_tree(axis, config, selected_color=selected_color)
        axis.scatter(
            endpoints[:, 0].cpu(),
            endpoints[:, 1].cpu(),
            s=5,
            c=color,
            alpha=0.32,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        _style_spatial_axis(axis, config)
        _panel_title(axis, letter, title)

    axis = axes[0, 3]
    _draw_tree(axis, config, selected_color=selected_color)
    trajectories = selected_after_trajectory.cpu()
    visible_trajectories = trajectories[
        max(0, trajectories.shape[0] // 2 - 1) :
    ]
    for trajectory in visible_trajectories.transpose(0, 1):
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=after_color,
            linewidth=0.65,
            alpha=0.42,
            zorder=3,
        )
    axis.scatter(
        trajectories[-1, :, 0],
        trajectories[-1, :, 1],
        s=8,
        c="#111111",
        linewidths=0,
        zorder=4,
    )
    _style_spatial_axis(axis, config)
    selected_segments = tuple(
        segment
        for branch_class, branch_path, branch_levels in reward_branch_selection_specs(
            config
        )
        for segment in select_branch_segments(
            branch_class, branch_path, branch_levels
        )
    )
    branch_x = [
        coordinate
        for segment in selected_segments
        for coordinate in (segment.start[0], segment.end[0])
    ]
    branch_y = [
        coordinate
        for segment in selected_segments
        for coordinate in (segment.start[1], segment.end[1])
    ]
    padding = 0.16
    axis.set_xlim(min(branch_x) - padding, max(branch_x) + padding)
    axis.set_ylim(min(branch_y) - padding, max(branch_y) + padding)
    _panel_title(
        axis,
        "d",
        f"{config.figure_trajectories} closest NFT trajectories (late ODE)",
    )

    axis = axes[1, 0]
    _draw_tree(axis, config, selected_color=selected_color)
    stride = max(1, config.figure_samples // 180)
    paired_before = before[::stride].cpu()
    paired_delta = (after[::stride] - before[::stride]).cpu()
    axis.quiver(
        paired_before[:, 0],
        paired_before[:, 1],
        paired_delta[:, 0],
        paired_delta[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
        color=after_color,
        alpha=0.65,
        zorder=3,
    )
    _style_spatial_axis(axis, config)
    _panel_title(axis, "e", "same-noise endpoint pairs")

    axis = axes[1, 1]
    maximum = float(torch.quantile(before_distance, 0.98).item())
    bins = np.linspace(0, max(maximum, config.near_branch_threshold * 2), 36)
    axis.hist(
        before_distance.cpu().numpy(),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color=before_color,
        label="pretrained",
    )
    axis.hist(
        after_distance.cpu().numpy(),
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.25,
        linewidth=1.5,
        color=after_color,
        label="NFT",
    )
    axis.axvline(
        config.near_branch_threshold,
        color=selected_color,
        linestyle="--",
        linewidth=1.2,
        label="near-branch threshold",
    )
    axis.set_xlabel("distance to selected branch")
    axis.set_ylabel("density")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2, linewidth=0.5)
    _panel_title(axis, "f", "branch-distance distribution")

    evaluations = [row for row in metrics if row.get("stage") == "evaluation"]
    axis = axes[1, 2]
    axis.plot(
        [row["outer_step"] for row in evaluations],
        [row["evaluation_distance_mean"] for row in evaluations],
        marker="o",
        color=after_color,
        linewidth=1.8,
        label="fixed-seed evaluation",
    )
    training = [row for row in metrics if row.get("stage") == "train"]
    axis.plot(
        [row["outer_step"] for row in training],
        [row["rollout_distance_mean"] for row in training],
        marker="s",
        color=selected_color,
        linewidth=1.2,
        alpha=0.8,
        label="online rollout",
    )
    axis.set_xlabel("outer NFT step")
    axis.set_ylabel("mean distance ↓")
    axis.set_xticks(sorted({int(row["outer_step"]) for row in evaluations}))
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False)
    _panel_title(axis, "g", "reward optimization")

    axis = axes[1, 3]
    before_mean = float(before_distance.mean().item())
    after_mean = float(after_distance.mean().item())
    before_rate = float(
        (before_distance <= config.near_branch_threshold).float().mean().item()
    )
    after_rate = float(
        (after_distance <= config.near_branch_threshold).float().mean().item()
    )
    x = np.arange(2)
    width = 0.34
    axis.bar(
        x - width / 2,
        [before_mean, after_mean],
        width,
        color=[before_color, after_color],
        alpha=0.85,
    )
    axis.set_xticks(x - width / 2, ["pretrained", "NFT"], rotation=12)
    axis.set_ylabel("mean distance ↓")
    twin = axis.twinx()
    twin.bar(
        x + width / 2,
        [before_rate, after_rate],
        width,
        color=[before_color, after_color],
        alpha=0.35,
        hatch="//",
    )
    twin.set_ylabel(f"fraction within {config.near_branch_threshold:g} ↑")
    twin.set_ylim(0, max(0.05, before_rate, after_rate) * 1.35)
    axis.grid(axis="y", alpha=0.22, linewidth=0.5)
    axis.legend(
        handles=[
            Line2D([0], [0], color=before_color, linewidth=6, label="mean distance"),
            Line2D(
                [0],
                [0],
                color=after_color,
                linewidth=6,
                alpha=0.35,
                label="near-branch fraction",
            ),
        ],
        frameon=False,
        loc="upper center",
    )
    _panel_title(axis, "h", "fixed-seed summary")

    figure.suptitle(
        "NFT-guided Autoguidance on the recursive 2D toy distribution",
        fontsize=14,
        fontweight="semibold",
    )
    png_path = output_dir / "toy_nft_branch_distance.png"
    pdf_path = output_dir / "toy_nft_branch_distance.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    summary = {
        "formal_experiment": False,
        "branch": reward_branch_label(config),
        "reward": config.reward,
        "reward_distance_scale": config.reward_distance_scale,
        "reward_formula": configured_reward_formula(config),
        "objective_mode": config.objective_mode,
        "completed_outer_steps": max(
            int(row["outer_step"]) for row in training
        ),
        "training_confirmed": bool(
            training and all(row.get("parameters_changed") for row in training)
        ),
        "before_distance_mean": before_mean,
        "after_distance_mean": after_mean,
        "distance_mean_change": after_mean - before_mean,
        "before_near_branch_rate": before_rate,
        "after_near_branch_rate": after_rate,
        "near_branch_rate_change": after_rate - before_rate,
        "figure_png": str(png_path),
        "figure_pdf": str(pdf_path),
        "figure_data": str(data_path),
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(
        "FIGURE "
        f"png={png_path} pdf={pdf_path} "
        f"distance_before={before_mean:.6f} distance_after={after_mean:.6f}",
        flush=True,
    )
    return summary


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA was requested but is not available")
    return device


@click.group()
def main() -> None:
    """Train and visualize distance-reward NFT on the Autoguidance toy tree."""


@main.command("run")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
@click.option("--good-model", type=str, default=None)
@click.option("--bad-model", type=str, default=None)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--figure/--no-figure", default=True, show_default=True)
def run_command(
    config_path: Path,
    output_dir: Path,
    good_model: str | None,
    bad_model: str | None,
    device: str,
    figure: bool,
) -> None:
    """Run a fresh NFT experiment and optionally render its static figure."""

    config = ToyNFTConfig.from_json(config_path)
    overrides = {}
    if good_model is not None:
        overrides["good_model"] = good_model
    if bad_model is not None:
        overrides["bad_model"] = bad_model
    if overrides:
        config = dataclasses.replace(config, **overrides)
        config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32))
    resolved_device = _resolve_device(device)
    initial_bundle = load_model_bundle(config, resolved_device) if figure else None
    final_bundle, _training = run_training(config, output_dir, resolved_device)
    if figure:
        assert initial_bundle is not None
        metrics = _read_metrics(output_dir / "metrics.jsonl")
        make_paper_figure(
            config,
            initial_bundle,
            final_bundle,
            metrics,
            output_dir,
            resolved_device,
        )


@main.command("plot")
@click.option(
    "--run-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Write rebuilt figure artifacts here instead of overwriting the run.",
)
@click.option(
    "--reuse-figure-data/--resample",
    default=True,
    show_default=True,
    help=(
        "Reuse the run's saved paired endpoints and trajectories so a redraw "
        "only changes presentation."
    ),
)
@click.option("--device", type=str, default="cuda", show_default=True)
def plot_command(
    run_dir: Path,
    output_dir: Path | None,
    reuse_figure_data: bool,
    device: str,
) -> None:
    """Rebuild the static figure from a completed checkpoint."""

    resolved_device = _resolve_device(device)
    config, final_bundle, completed_outer_step = _load_checkpoint_bundle(
        run_dir, resolved_device
    )
    if completed_outer_step <= 0:
        raise click.ClickException("checkpoint does not contain a trained outer step")
    source_figure_data = run_dir / "figure_data.npz"
    if reuse_figure_data and not source_figure_data.is_file():
        raise click.ClickException(
            f"saved figure data does not exist: {source_figure_data}; "
            "use --resample to regenerate it"
        )
    if not reuse_figure_data:
        source_figure_data = None
    initial_bundle = (
        None if reuse_figure_data else load_model_bundle(config, resolved_device)
    )
    metrics = _read_metrics(run_dir / "metrics.jsonl")
    figure_dir = run_dir if output_dir is None else output_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        figure_dir / "visualization_source.json",
        {
            "source_run": str(run_dir.resolve()),
            "completed_outer_step": completed_outer_step,
            "figure_data_source": (
                str(source_figure_data.resolve())
                if source_figure_data is not None
                else None
            ),
            "resampled": not reuse_figure_data,
            "config": dataclasses.asdict(config),
        },
    )
    make_paper_figure(
        config,
        initial_bundle,
        final_bundle,
        metrics,
        figure_dir,
        resolved_device,
        source_figure_data=source_figure_data,
    )


if __name__ == "__main__":
    main()
