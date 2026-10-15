from __future__ import annotations

from typing import NamedTuple

import torch


class FlowMatchingBatch(NamedTuple):
    state: torch.Tensor
    target_velocity: torch.Tensor
    noise: torch.Tensor


def group_normalize_rewards(rewards: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    """Normalize rewards independently for each ``[class, samples]`` group."""

    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [num_groups, samples_per_group>=2]")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    return centered / (rewards.std(dim=1, unbiased=False, keepdim=True) + epsilon)


def advantage_to_weight(advantages: torch.Tensor, clip_max: float) -> torch.Tensor:
    if clip_max <= 0:
        raise ValueError("clip_max must be positive")
    clipped = advantages.clamp(-clip_max, clip_max)
    return ((clipped / clip_max) / 2.0 + 0.5).clamp(0.0, 1.0)


def make_flow_matching_batch(
    endpoints: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> FlowMatchingBatch:
    """Build the repository's noise-to-data linear path.

    This codebase parameterizes time as ``x_t=(1-t)x_0+t*eps`` and predicts
    ``eps-x_0``.  It is the time reversal of the OWT/SiT NFT notation.
    """

    if endpoints.ndim < 2 or endpoints.shape[0] == 0:
        raise ValueError("endpoints must be a non-empty batch")
    if t.shape != (endpoints.shape[0],):
        raise ValueError(f"t must have shape [{endpoints.shape[0]}]")
    if noise is None:
        noise = torch.randn_like(endpoints)
    if noise.shape != endpoints.shape:
        raise ValueError("noise must match endpoints")
    t_view = t.view(-1, *([1] * (endpoints.ndim - 1)))
    state = (1.0 - t_view) * endpoints + t_view * noise
    target_velocity = noise - endpoints
    return FlowMatchingBatch(state=state, target_velocity=target_velocity, noise=noise)


def sampler_timesteps(sampler: str, sampling_steps: int) -> torch.Tensor:
    """Return the exact model-input time grid used by this repository's sampler."""

    if sampling_steps < 2:
        raise ValueError("sampling_steps must be at least 2")
    if sampler == "ode":
        return torch.linspace(1.0, 0.0, sampling_steps + 1, dtype=torch.float32)[:-1]
    if sampler == "sde":
        return torch.linspace(1.0, 0.04, sampling_steps, dtype=torch.float32)
    raise ValueError("sampler must be 'ode' or 'sde'")


def nft_training_timestep_count(sampling_steps: int, timestep_fraction: float) -> int:
    """Match DiffusionNFT's ``int(num_steps * timestep_fraction)`` rule."""

    if sampling_steps < 2:
        raise ValueError("sampling_steps must be at least 2")
    if not 0.0 < timestep_fraction <= 1.0:
        raise ValueError("timestep_fraction must be in (0, 1]")
    count = int(sampling_steps * timestep_fraction)
    if count < 1:
        raise ValueError("timestep_fraction selects no sampling timesteps")
    return count


def sample_nft_timesteps(
    time_grid: torch.Tensor,
    *,
    num_examples: int,
    inner_epochs: int,
    timestep_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Choose DiffusionNFT training times from shuffled sampler grids.

    The published implementation independently permutes all sampler time indices
    for every endpoint and keeps ``int(num_steps * timestep_fraction)`` entries.
    A random-key sort is the batched equivalent of those ``randperm`` calls.
    """

    if time_grid.ndim != 1 or time_grid.numel() < 2:
        raise ValueError("time_grid must be one-dimensional with at least two entries")
    if num_examples < 1 or inner_epochs < 1:
        raise ValueError("num_examples and inner_epochs must be positive")
    count = nft_training_timestep_count(time_grid.numel(), timestep_fraction)
    random_keys = torch.rand(
        inner_epochs,
        num_examples,
        time_grid.numel(),
        generator=generator,
        device=time_grid.device,
        dtype=torch.float32,
    )
    indices = random_keys.argsort(dim=-1)[..., :count]
    expanded_grid = time_grid.reshape(1, 1, -1).expand(inner_epochs, num_examples, -1)
    return torch.gather(expanded_grid, dim=-1, index=indices)


def compose_nft_predictions(
    current: torch.Tensor,
    old: torch.Tensor,
    *,
    beta: float,
    center: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``selected_center +/- beta * (current - old)``."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    if current.shape != old.shape or (center is not None and center.shape != current.shape):
        raise ValueError("current, old, and center must have identical shapes")
    selected_center = old if center is None else center
    displacement = current - old
    return selected_center + beta * displacement, selected_center - beta * displacement


def mse_per_example(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    return (prediction.float() - target.float()).square().flatten(1).mean(dim=1)


def diffusion_nft_loss(
    *,
    current_prediction: torch.Tensor,
    old_prediction: torch.Tensor,
    reference_prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    advantages: torch.Tensor,
    beta: float,
    advantage_clip: float,
    kl_beta: float,
    center_prediction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute original or ShiftCenter DiffusionNFT in velocity space."""

    batch_size = current_prediction.shape[0]
    if advantages.shape != (batch_size,):
        raise ValueError(f"advantages must have shape [{batch_size}]")
    if kl_beta < 0:
        raise ValueError("kl_beta must be non-negative")

    positive, negative = compose_nft_predictions(
        current_prediction,
        old_prediction,
        beta=beta,
        center=center_prediction,
    )
    positive_loss = mse_per_example(positive, target_velocity)
    negative_loss = mse_per_example(negative, target_velocity)
    reward_weight = advantage_to_weight(advantages, advantage_clip)
    unscaled_policy = (
        reward_weight * positive_loss + (1.0 - reward_weight) * negative_loss
    ) / beta
    policy_loss = advantage_clip * unscaled_policy.mean()
    kl_per_example = mse_per_example(current_prediction, reference_prediction)
    kl_loss = kl_per_example.mean()
    total_loss = policy_loss + kl_beta * kl_loss

    old_deviation = mse_per_example(current_prediction, old_prediction)
    if center_prediction is None:
        center_deviation = torch.zeros_like(old_deviation)
    else:
        center_deviation = mse_per_example(center_prediction, old_prediction)
    metrics = {
        "total_loss": total_loss.detach(),
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "positive_loss": positive_loss.mean().detach(),
        "negative_loss": negative_loss.mean().detach(),
        "reward_weight_mean": reward_weight.mean().detach(),
        "old_deviation": old_deviation.mean().detach(),
        "center_deviation": center_deviation.mean().detach(),
    }
    return total_loss, metrics


def old_policy_decay(step: int, decay_type: int) -> float:
    """Match the published DiffusionNFT old-policy schedule."""

    if step < 0:
        raise ValueError("step must be non-negative")
    if decay_type == 0:
        flat, rate, maximum = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, rate, maximum = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, rate, maximum = 75, 0.0075, 0.999
    else:
        raise ValueError("decay_type must be 0, 1, or 2")
    if step < flat:
        return 0.0
    return min((step - flat) * rate, maximum)
