"""SC-CFG field wrappers used by ELF NFT optimization."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from utils.sampling_utils import restore_cond, restore_vx

from .nft_math import reconstruct_sccfg1_field


OT_MIN_REMAINING_TIME = 1e-5


class OldPolicySelfCondition(NamedTuple):
    """One shared old-policy first-pass result for an NFT field comparison."""

    value: jax.Array
    empty_velocity: jax.Array
    empty_clean_prediction: jax.Array


class ReconstructedSCCFGVelocity(NamedTuple):
    """Two-pass velocity together with its detached CFG1 reconstruction."""

    effective_velocity: jax.Array
    empty_velocity: jax.Array
    main_velocity: jax.Array
    guidance: jax.Array
    self_condition: jax.Array


def empty_self_condition(z, cond_seq, cond_seq_mask):
    """Return ELF's empty SC input while retaining a clean source prefix."""

    if z.shape != cond_seq.shape or cond_seq_mask.shape != z.shape[:2]:
        raise ValueError("z/cond_seq/cond_seq_mask shapes are inconsistent")
    return restore_cond(jnp.zeros_like(z), cond_seq, cond_seq_mask)


def clean_prediction_to_ot_velocity(
    clean_prediction,
    z,
    t,
    *,
    min_remaining_time: float = OT_MIN_REMAINING_TIME,
):
    """Convert ELF's clean prediction to classic noise-to-data OT velocity.

    For ``z_t=(1-t)*x_0+t*x_1``, the velocity represented by a predicted
    endpoint ``x_1_hat`` is ``(x_1_hat-z_t)/(1-t)``. Training timesteps are
    strictly below one; ``min_remaining_time`` is only a finite-precision
    guard and does not replace ELF's classic OT target with a clipped target.
    """

    if clean_prediction.shape != z.shape or clean_prediction.ndim != 3:
        raise ValueError("clean_prediction and z must share shape [B, S, D]")
    if t.shape != (z.shape[0],):
        raise ValueError(f"t must have shape [{z.shape[0]}]")
    if min_remaining_time <= 0:
        raise ValueError("min_remaining_time must be positive")
    remaining = jnp.maximum(
        1.0 - t.astype(jnp.float32),
        jnp.asarray(min_remaining_time, dtype=jnp.float32),
    ).reshape(-1, 1, 1)
    velocity = (
        clean_prediction.astype(jnp.float32) - z.astype(jnp.float32)
    ) / remaining
    return velocity.astype(z.dtype)


def velocity_with_self_condition(
    *,
    model_apply_fn,
    model_params,
    z,
    t,
    cond_seq,
    cond_seq_mask,
    self_condition,
    input_cfg_scale: float,
    self_cond_cfg_scale: float,
    config,
):
    """Evaluate one ELF velocity at an explicitly supplied SC input.

    The caller owns ``self_condition``. In NFT training it is either the
    frozen old policy's empty-SC clean prediction or the empty SC tensor, and
    the exact same tensor is supplied to current, old, reference, and center.
    This function performs no first pass. For conditional tasks it optionally
    composes the same source-input CFG field used by rollout; OWT uses scale
    one and therefore evaluates only the null-conditioned field.
    """

    if config.self_cond_prob <= 0 or config.num_self_cond_cfg_tokens <= 0:
        raise ValueError("SC-CFG NFT requires ELF's self-conditioning interface")
    if z.shape != cond_seq.shape or self_condition.shape != z.shape:
        raise ValueError("z, cond_seq, and self_condition must share shape")
    if cond_seq_mask.shape != z.shape[:2] or t.shape != (z.shape[0],):
        raise ValueError("cond_seq_mask or t has an incompatible shape")
    if isinstance(self_cond_cfg_scale, (int, float)) and self_cond_cfg_scale <= 0:
        raise ValueError("self_cond_cfg_scale must be positive")
    if input_cfg_scale <= 0:
        raise ValueError("input_cfg_scale must be positive")

    conditioned = restore_cond(self_condition, cond_seq, cond_seq_mask)
    model_input = jnp.concatenate([z, conditioned], axis=-1)
    scale = jnp.asarray(self_cond_cfg_scale, dtype=t.dtype)
    if scale.ndim == 0:
        scale = jnp.full((z.shape[0],), scale, dtype=t.dtype)
    elif scale.shape != (z.shape[0],):
        raise ValueError(
            f"self_cond_cfg_scale must be scalar or have shape [{z.shape[0]}]"
        )
    model_output, _ = model_apply_fn(
        {"params": model_params},
        model_input,
        t,
        deterministic=True,
        self_cond_cfg_scale=scale,
        decoder_step_active=jnp.array(False),
    )
    conditional_clean = restore_cond(model_output, cond_seq, cond_seq_mask)
    conditional_velocity = clean_prediction_to_ot_velocity(conditional_clean, z, t)
    conditional_velocity, conditional_clean = restore_vx(
        conditional_velocity,
        conditional_clean,
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
    )
    if input_cfg_scale == 1.0:
        return conditional_velocity, conditional_clean

    zero_condition = jnp.zeros_like(cond_seq)
    unconditional_z = restore_cond(z, zero_condition, cond_seq_mask)
    unconditional_sc = restore_cond(
        self_condition, zero_condition, cond_seq_mask
    )
    unconditional_input = jnp.concatenate(
        [unconditional_z, unconditional_sc], axis=-1
    )
    unconditional_output, _ = model_apply_fn(
        {"params": model_params},
        unconditional_input,
        t,
        deterministic=True,
        self_cond_cfg_scale=scale,
        decoder_step_active=jnp.array(False),
    )
    unconditional_clean = restore_cond(
        unconditional_output, zero_condition, cond_seq_mask
    )
    unconditional_velocity = clean_prediction_to_ot_velocity(
        unconditional_clean, unconditional_z, t
    )
    unconditional_velocity, unconditional_clean = restore_vx(
        unconditional_velocity,
        unconditional_clean,
        cond_seq=zero_condition,
        cond_seq_mask=cond_seq_mask,
    )
    velocity = unconditional_velocity + input_cfg_scale * (
        conditional_velocity - unconditional_velocity
    )
    clean_prediction = unconditional_clean + input_cfg_scale * (
        conditional_clean - unconditional_clean
    )
    return restore_vx(
        velocity,
        clean_prediction,
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
    )


def reconstructed_sccfg1_velocity(
    *,
    model_apply_fn,
    model_params,
    z,
    t,
    cond_seq,
    cond_seq_mask,
    use_self_condition,
    input_cfg_scale: float,
    self_cond_cfg_scale,
    config,
) -> ReconstructedSCCFGVelocity:
    """Run one model's zero/main passes and reconstruct its CFG1 velocity.

    Each model owns its zero pass.  That clean prediction is detached before it
    becomes the main-pass self-condition, so current, old, and reference fields
    use the same operator without sharing a prediction across model snapshots.
    """

    if use_self_condition.shape != (z.shape[0],):
        raise ValueError(f"use_self_condition must have shape [{z.shape[0]}]")
    empty = empty_self_condition(z, cond_seq, cond_seq_mask)
    empty_velocity, empty_clean_prediction = velocity_with_self_condition(
        model_apply_fn=model_apply_fn,
        model_params=model_params,
        z=z,
        t=t,
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
        self_condition=empty,
        input_cfg_scale=input_cfg_scale,
        self_cond_cfg_scale=self_cond_cfg_scale,
        config=config,
    )
    empty_velocity = jax.lax.stop_gradient(empty_velocity)
    empty_clean_prediction = jax.lax.stop_gradient(empty_clean_prediction)
    selected = jnp.where(
        use_self_condition[:, None, None],
        empty_clean_prediction,
        empty,
    )
    selected = restore_cond(selected, cond_seq, cond_seq_mask)
    main_velocity, _ = velocity_with_self_condition(
        model_apply_fn=model_apply_fn,
        model_params=model_params,
        z=z,
        t=t,
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
        self_condition=selected,
        input_cfg_scale=input_cfg_scale,
        self_cond_cfg_scale=self_cond_cfg_scale,
        config=config,
    )
    reconstruction = reconstruct_sccfg1_field(
        main_velocity,
        empty_velocity,
        use_self_condition=use_self_condition,
        cfg_scale=self_cond_cfg_scale,
    )
    return ReconstructedSCCFGVelocity(
        effective_velocity=reconstruction.base_field,
        empty_velocity=empty_velocity,
        main_velocity=main_velocity,
        guidance=reconstruction.guidance,
        self_condition=jax.lax.stop_gradient(selected),
    )


def build_old_policy_self_condition(
    *,
    model_apply_fn,
    old_params,
    z,
    t,
    cond_seq,
    cond_seq_mask,
    use_self_condition,
    input_cfg_scale: float,
    self_cond_cfg_scale: float,
    config,
) -> OldPolicySelfCondition:
    """Build the Bernoulli-routed SC input once with the frozen old policy.

    ``use_self_condition`` is per-example. True rows receive the old policy's
    empty-SC clean prediction; false rows remain empty. The result is then
    reused without modification by every field in the center or policy loss.
    """

    if use_self_condition.shape != (z.shape[0],):
        raise ValueError(f"use_self_condition must have shape [{z.shape[0]}]")
    empty = empty_self_condition(z, cond_seq, cond_seq_mask)
    empty_velocity, empty_clean_prediction = velocity_with_self_condition(
        model_apply_fn=model_apply_fn,
        model_params=old_params,
        z=z,
        t=t,
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
        self_condition=empty,
        input_cfg_scale=input_cfg_scale,
        self_cond_cfg_scale=self_cond_cfg_scale,
        config=config,
    )
    selected = jnp.where(
        use_self_condition[:, None, None],
        empty_clean_prediction,
        empty,
    )
    selected = restore_cond(selected, cond_seq, cond_seq_mask)
    return OldPolicySelfCondition(
        value=selected,
        empty_velocity=empty_velocity,
        empty_clean_prediction=empty_clean_prediction,
    )
