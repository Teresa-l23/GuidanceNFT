"""Pure classic-OT DiffusionNFT and flow-center mathematics for ELF."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from utils.sampling_utils import restore_cond


class EndpointFlowBatch(NamedTuple):
    state: jax.Array
    target_velocity: jax.Array
    noise: jax.Array


class SCCFGReconstruction(NamedTuple):
    """A CFG1-equivalent field and its detached intrinsic guidance term."""

    base_field: jax.Array
    guidance: jax.Array


def group_normalize_rewards(rewards, epsilon: float = 1e-4) -> np.ndarray:
    """Normalize rewards independently inside each prompt group.

    Input must be ``[num_prompts, samples_per_prompt]``. Population standard
    deviation is computed per prompt group; a constant-reward group maps to
    all-zero advantages.  This deliberately differs from the released
    DiffusionNFT training config's ``sample.global_std=True`` setting.
    """

    values = np.asarray(rewards, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("rewards must have shape [num_prompts, samples_per_prompt>=2]")
    if not np.isfinite(values).all():
        raise ValueError("rewards must be finite")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / (values.std(axis=1, keepdims=True) + epsilon)


def combine_group_normalized_rewards(
    primary_rewards, auxiliary_rewards, *, auxiliary_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize two reward streams independently, then combine advantages.

    ``auxiliary_weight`` is the convex weight assigned to the auxiliary stream;
    the primary stream receives ``1 - auxiliary_weight``. Independent
    normalization prevents the raw units or variances of the two rewards from
    silently determining their optimization priority.
    """

    primary = np.asarray(primary_rewards, dtype=np.float32)
    auxiliary = np.asarray(auxiliary_rewards, dtype=np.float32)
    if primary.shape != auxiliary.shape:
        raise ValueError("primary and auxiliary rewards must have identical shapes")
    if not np.isfinite(auxiliary_weight) or not 0.0 <= auxiliary_weight < 1.0:
        raise ValueError("auxiliary_weight must lie in [0, 1)")
    primary_advantages = group_normalize_rewards(primary)
    auxiliary_advantages = group_normalize_rewards(auxiliary)
    combined = (
        (1.0 - auxiliary_weight) * primary_advantages
        + auxiliary_weight * auxiliary_advantages
    )
    return (
        combined.astype(np.float32),
        primary_advantages.astype(np.float32),
        auxiliary_advantages.astype(np.float32),
    )


def advantage_to_weight(advantages, clip_max: float):
    """Map clipped advantages from ``[-clip_max, clip_max]`` into ``[0, 1]``."""

    if clip_max <= 0:
        raise ValueError("clip_max must be positive")
    clipped = jnp.clip(advantages, -clip_max, clip_max)
    return jnp.clip((clipped / clip_max) / 2.0 + 0.5, 0.0, 1.0)


def compose_nft_predictions(current, old, *, beta: float, center=None):
    """Return ``center +/- beta * (current - old)``.

    ``center=None`` gives classic DiffusionNFT by using ``old`` as the center.
    A supplied center gives learned flow-matching ShiftCenter. Callers evaluate
    old and center outside the differentiated current-policy closure, so this
    pure composition needs no synthetic stop-gradient construction.
    """

    if beta <= 0:
        raise ValueError("beta must be positive")
    if current.shape != old.shape or (center is not None and center.shape != current.shape):
        raise ValueError("current, old, and center must have identical shapes")
    selected_center = old if center is None else center
    displacement = current - old
    return (
        selected_center + beta * displacement,
        selected_center - beta * displacement,
    )


def recover_sccfg_base_field(conditional, empty, *, cfg_scale: float):
    """Recover the base/plain field represented by an intrinsic SC-CFG pair.

    ELF trains its intrinsic scale-``w`` conditional field with the stopped
    guidance correction ``(1 - 1 / w) * (conditional - empty)``.  Removing
    that correction gives

    ``conditional / w + (1 - 1 / w) * empty``.

    The operation is purely numerical.  Callers decide which inputs are
    frozen; the NFT trainer supplies old-policy predictions, so its returned
    center is outside the current-policy differentiation closure.
    """

    if conditional.shape != empty.shape:
        raise ValueError("conditional and empty fields must have identical shapes")
    if not np.isfinite(cfg_scale) or cfg_scale <= 0:
        raise ValueError("cfg_scale must be finite and positive")
    correction_fraction = 1.0 - 1.0 / float(cfg_scale)
    return conditional - correction_fraction * (conditional - empty)


def reconstruct_sccfg1_field(
    conditional,
    empty,
    *,
    use_self_condition,
    cfg_scale,
) -> SCCFGReconstruction:
    """Reconstruct a CFG1 field while differentiating only the main pass.

    ``conditional`` and ``empty`` are the same model's main and zero-pass
    fields.  ``cfg_scale`` and ``use_self_condition`` contain one value per
    record.  The numerical reconstruction is

    ``conditional - u * (1 - 1 / w) * (conditional - empty)``.

    The complete correction is stop-gradient, matching ELF's training target
    construction and the mathematical SCCFG experiment: gradients therefore
    pass through ``conditional`` with an identity Jacobian and never through
    the zero pass or guidance difference.
    """

    if conditional.shape != empty.shape or conditional.ndim != 3:
        raise ValueError(
            "conditional and empty fields must share shape [B, S, D]"
        )
    batch_size = conditional.shape[0]
    use = jnp.asarray(use_self_condition)
    scale = jnp.asarray(cfg_scale)
    if use.shape != (batch_size,):
        raise ValueError(f"use_self_condition must have shape [{batch_size}]")
    if scale.shape != (batch_size,):
        raise ValueError(f"cfg_scale must have shape [{batch_size}]")
    use = use.astype(jnp.float32).reshape(-1, 1, 1)
    scale = scale.astype(jnp.float32).reshape(-1, 1, 1)
    guidance = use * (1.0 - 1.0 / scale) * (
        conditional.astype(jnp.float32) - empty.astype(jnp.float32)
    )
    guidance = jax.lax.stop_gradient(guidance).astype(conditional.dtype)
    return SCCFGReconstruction(
        base_field=conditional - guidance,
        guidance=guidance,
    )


def masked_mse_per_example(prediction, target, loss_mask):
    """Return per-example embedding MSE using ELF's target-token mask."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [B, S, D]")
    if loss_mask.shape != prediction.shape[:2]:
        raise ValueError("loss_mask must have shape [B, S]")
    mask = loss_mask.astype(jnp.float32)
    residual_sq = (prediction.astype(jnp.float32) - target.astype(jnp.float32)) ** 2
    numerator = (residual_sq * mask[..., None]).sum(axis=(1, 2))
    denominator = jnp.maximum(mask.sum(axis=1) * prediction.shape[-1], 1.0)
    return numerator / denominator


def masked_token_cross_entropy_per_example(logits, targets, loss_mask):
    """Return per-example decoder CE on ELF target-token positions."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, S, V]")
    if targets.shape != logits.shape[:2] or loss_mask.shape != targets.shape:
        raise ValueError("targets/loss_mask must match logits [B, S]")
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    token_loss = -jnp.take_along_axis(
        log_probs, targets.astype(jnp.int32)[..., None], axis=-1
    ).squeeze(-1)
    mask = loss_mask.astype(jnp.float32)
    numerator = (token_loss * mask).sum(axis=1)
    denominator = jnp.maximum(mask.sum(axis=1), 1.0)
    return numerator / denominator


def make_endpoint_flow_batch(
    endpoints,
    t,
    rng,
    *,
    noise_scale: float,
    t_eps: float,
    cond_seq,
    cond_seq_mask,
    noise=None,
) -> EndpointFlowBatch:
    """Fresh classic-OT forward process from rewarded raw endpoints.

    Constructs ``z_t=t*x+(1-t)*sigma*eps`` and the ordinary sample velocity
    ``x-sigma*eps``. Condition-prefix positions are restored clean and their
    target velocity is zero because those tokens do not move. ``t_eps`` remains
    in the signature for the base ELF configuration contract, but it does not
    clip or rescale the classic OT target. Endpoints are never passed directly
    to the model.
    """

    if endpoints.ndim != 3 or endpoints.shape[0] == 0:
        raise ValueError("endpoints must be nonempty [B, S, D]")
    if t.shape != (endpoints.shape[0],):
        raise ValueError(f"t must have shape [{endpoints.shape[0]}]")
    if cond_seq.shape != endpoints.shape:
        raise ValueError("cond_seq must match endpoints")
    if cond_seq_mask.shape != endpoints.shape[:2]:
        raise ValueError("cond_seq_mask must have shape [B, S]")
    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive")
    if t_eps <= 0:
        raise ValueError("t_eps must be positive")
    if noise is None:
        noise = jax.random.normal(rng, endpoints.shape, dtype=endpoints.dtype)
    elif noise.shape != endpoints.shape:
        raise ValueError("noise must match endpoints")
    scaled_noise = noise * noise_scale
    t_view = t.reshape(-1, 1, 1)
    state = t_view * endpoints + (1.0 - t_view) * scaled_noise
    state = restore_cond(state, cond_seq, cond_seq_mask)
    target = endpoints - scaled_noise
    target = restore_cond(target, jnp.zeros_like(cond_seq), cond_seq_mask)
    return EndpointFlowBatch(state=state, target_velocity=target, noise=noise)


def old_policy_decay(step: int, decay_type: int) -> float:
    """Port ``return_decay`` from the referenced SD3 DiffusionNFT trainer."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError("decay_type must be 0, 1, or 2")
    if step < flat:
        return 0.0
    return min((step - flat) * uprate, uphold)
