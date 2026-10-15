from __future__ import annotations

import math

import torch


DIFFUSION_NFT_MAE_EPSILON = 1e-5


def _same_shape(*tensors: torch.Tensor) -> None:
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError(f"tensor shapes must match, got {sorted(shapes)}")


def compose_autoguidance(
    good: torch.Tensor, bad: torch.Tensor, guidance_scale: float
) -> torch.Tensor:
    """Return ``bad + w * (good - bad)`` in EDM denoiser space."""

    _same_shape(good, bad)
    if not math.isfinite(float(guidance_scale)) or guidance_scale <= 0:
        raise ValueError("guidance_scale must be positive and finite")
    return bad + float(guidance_scale) * (good - bad)


def compose_shifted_nft_branches(
    *,
    current: torch.Tensor,
    old: torch.Tensor,
    center: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``center +/- beta * (current - old)`` with detached anchors."""

    _same_shape(current, old, center)
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    displacement = current - old.detach()
    fixed_center = center.detach()
    return (
        fixed_center + float(beta) * displacement,
        fixed_center - float(beta) * displacement,
    )


def compose_standard_nft_branches(
    *,
    current: torch.Tensor,
    old: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the classic NFT pair ``old +/- beta * (current - old)``."""

    _same_shape(current, old)
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    fixed_old = old.detach()
    displacement = current - fixed_old
    return (
        fixed_old + float(beta) * displacement,
        fixed_old - float(beta) * displacement,
    )


def advantage_to_reward_weight(
    advantages: torch.Tensor, clip_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip advantages and map ``[-clip_max, clip_max]`` to ``[0, 1]``."""

    if advantages.ndim != 1:
        raise ValueError("advantages must be rank one")
    if not math.isfinite(float(clip_max)) or clip_max <= 0:
        raise ValueError("clip_max must be positive and finite")
    clipped = advantages.clamp(-float(clip_max), float(clip_max))
    return clipped, (clipped / float(clip_max) / 2.0 + 0.5).clamp(0, 1)


def standardize_group_rewards(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    global_std: bool,
    epsilon: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Standardize each condition group, optionally sharing one global std."""

    if rewards.ndim != 1 or group_ids.ndim != 1 or rewards.shape != group_ids.shape:
        raise ValueError("rewards and group_ids must be equally sized rank-one tensors")
    if rewards.numel() == 0 or not torch.isfinite(rewards).all():
        raise ValueError("rewards must be nonempty and finite")
    if epsilon <= 0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be positive and finite")
    unique = torch.unique(group_ids, sorted=True)
    global_scale = rewards.std(unbiased=False) if global_std else None
    advantages = torch.empty_like(rewards, dtype=torch.float32)
    stds = []
    zero_std = 0
    for group_id in unique:
        mask = group_ids == group_id
        values = rewards[mask].float()
        std = values.std(unbiased=False)
        stds.append(float(std.item()))
        zero_std += int(float(std.item()) == 0.0)
        denominator = global_scale if global_scale is not None else std
        advantages[mask] = (values - values.mean()) / (denominator + float(epsilon))
    return advantages, {
        "group_count": float(unique.numel()),
        "zero_std_ratio": float(zero_std / unique.numel()),
        "group_std_mean": float(sum(stds) / len(stds)),
    }


def make_edm_noising_batch(
    endpoints: torch.Tensor,
    *,
    P_mean: float,
    P_std: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw log-normal EDM noise for shared-center pretraining only."""

    if endpoints.ndim != 4 or endpoints.shape[0] == 0:
        raise ValueError("endpoints must be a nonempty NCHW tensor")
    if not math.isfinite(float(P_mean)):
        raise ValueError("P_mean must be finite")
    if P_std <= 0 or not math.isfinite(float(P_std)):
        raise ValueError("P_std must be positive and finite")
    batch = endpoints.shape[0]
    normal = torch.randn(
        (batch, 1, 1, 1),
        device=endpoints.device,
        dtype=torch.float32,
        generator=generator,
    )
    sigma = (normal * float(P_std) + float(P_mean)).exp()
    noise = torch.randn(
        endpoints.shape,
        device=endpoints.device,
        dtype=torch.float32,
        generator=generator,
    )
    state = endpoints.float() + sigma * noise
    return state, sigma.reshape(batch)


def make_diffusion_nft_noising_batch(
    endpoints: torch.Tensor,
    sigma: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct one independently noised NFT record for every endpoint row."""

    if endpoints.ndim != 4 or endpoints.shape[0] == 0:
        raise ValueError("endpoints must be a nonempty NCHW tensor")
    if sigma.shape != (endpoints.shape[0],):
        raise ValueError("sigma must contain one value per endpoint")
    sigma = sigma.to(device=endpoints.device, dtype=torch.float32)
    if not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("sigma values must be positive and finite")
    noise = torch.randn(
        endpoints.shape,
        device=endpoints.device,
        dtype=torch.float32,
        generator=generator,
    )
    state = endpoints.float() + sigma.reshape(-1, 1, 1, 1) * noise
    return state, sigma


def diffusion_nft_branch_loss_per_example(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mae_epsilon: float = DIFFUSION_NFT_MAE_EPSILON,
) -> torch.Tensor:
    """Return DiffusionNFT's detached-MAE-normalized clean-x0 branch loss."""

    _same_shape(prediction, target)
    if prediction.ndim != 4 or prediction.shape[0] == 0:
        raise ValueError("prediction and target must be nonempty NCHW tensors")
    if mae_epsilon <= 0 or not math.isfinite(float(mae_epsilon)):
        raise ValueError("mae_epsilon must be positive and finite")
    residual = prediction.float() - target.float()
    reduction_dims = tuple(range(1, residual.ndim))
    denominator = (
        residual.detach()
        .double()
        .abs()
        .mean(dim=reduction_dims)
        .clamp(min=float(mae_epsilon))
    )
    # The denominator is constant within each example, so reducing the
    # numerator first is algebraically identical to the released implementation
    # and avoids materializing a full-resolution float64 quotient.
    return residual.square().mean(dim=reduction_dims).double() / denominator


def x0_mse_loss_per_example(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return unweighted clean-x0 MSE per example."""

    _same_shape(prediction, target)
    if prediction.ndim != 4 or prediction.shape[0] == 0:
        raise ValueError("prediction and target must be nonempty NCHW tensors")
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def edm_weighted_reconstruction_loss_per_example(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    *,
    sigma_data: float,
) -> torch.Tensor:
    """Return the legacy EDM-weighted loss used only to fit a shared center."""

    _same_shape(prediction, target)
    if prediction.ndim != 4 or sigma.shape != (prediction.shape[0],):
        raise ValueError("expected NCHW predictions and one sigma per example")
    if sigma_data <= 0 or not math.isfinite(float(sigma_data)):
        raise ValueError("sigma_data must be positive and finite")
    if not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("sigma values must be positive and finite")
    loss = x0_mse_loss_per_example(prediction, target)
    sigma64 = sigma.double()
    weight = (sigma64.square() + float(sigma_data) ** 2) / (
        sigma64 * float(sigma_data)
    ).square()
    return loss * weight.to(loss.dtype)


def nft_policy_loss_per_example(
    positive_loss: torch.Tensor,
    negative_loss: torch.Tensor,
    reward_weight: torch.Tensor,
    *,
    beta: float,
    advantage_clip: float,
) -> torch.Tensor:
    _same_shape(positive_loss, negative_loss, reward_weight)
    if positive_loss.ndim != 1:
        raise ValueError("NFT losses and reward weights must be rank one")
    if beta <= 0 or advantage_clip <= 0:
        raise ValueError("beta and advantage_clip must be positive")
    if ((reward_weight < 0) | (reward_weight > 1)).any():
        raise ValueError("reward_weight must lie in [0, 1]")
    return (
        reward_weight * positive_loss + (1.0 - reward_weight) * negative_loss
    ) * (float(advantage_clip) / float(beta))


def old_policy_decay(step: int, decay_type: int) -> float:
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
    return 0.0 if step < flat else min((step - flat) * rate, maximum)


@torch.no_grad()
def update_ema_(target: torch.nn.Module, source: torch.nn.Module, decay: float) -> None:
    if not 0 <= decay < 1:
        raise ValueError("decay must lie in [0, 1)")
    target_state = target.state_dict()
    source_state = source.state_dict()
    if target_state.keys() != source_state.keys():
        raise ValueError("EMA source and target structures differ")
    for name, target_value in target_state.items():
        source_value = source_state[name].detach().to(
            device=target_value.device, dtype=target_value.dtype
        )
        if target_value.is_floating_point():
            target_value.mul_(float(decay)).add_(source_value, alpha=1.0 - float(decay))
        else:
            target_value.copy_(source_value)


def flattened_cosine(
    left: torch.Tensor, right: torch.Tensor, epsilon: float = 1e-12
) -> float:
    _same_shape(left, right)
    left64 = left.detach().double().reshape(left.shape[0], -1)
    right64 = right.detach().double().reshape(right.shape[0], -1)
    numerator = (left64 * right64).sum(dim=1)
    denominator = left64.norm(dim=1) * right64.norm(dim=1)
    valid = denominator > epsilon
    if not valid.any():
        return 0.0
    return float((numerator[valid] / denominator[valid]).mean().item())
