"""Pmappable full-shared-model Stage-1 center and Stage-2 NFT updates."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from utils.generation_utils import _dlm_decoder_logits
from utils.sampling_utils import sample_cfg_scale, sample_timesteps

from .full_model import (
    frozen_decoder_delta_norm,
    full_parameter_labels,
    restore_frozen_decoder_params,
    zero_frozen_decoder_grads,
)
from .nft_math import (
    advantage_to_weight,
    compose_nft_predictions,
    make_endpoint_flow_batch,
    masked_mse_per_example,
    masked_token_cross_entropy_per_example,
    recover_sccfg_base_field,
)
from .policy import (
    OT_MIN_REMAINING_TIME,
    build_old_policy_self_condition,
    empty_self_condition,
    reconstructed_sccfg1_velocity,
    velocity_with_self_condition,
)
from .source_cfg import (
    source_null_field_inputs, training_input_cfg_scale,
)


CENTER_ACCUMULATED_METRIC_NAMES = (
    "center_loss",
    "center_loss_max",
    "center_t_mean",
    "center_sc_branch_fraction",
    "center_old_empty_velocity_rms",
)

POLICY_ACCUMULATED_METRIC_NAMES = (
    "total_loss",
    "policy_loss",
    "policy_sc_loss",
    "policy_empty_loss",
    "policy_source_cond_loss",
    "policy_source_uncond_loss",
    "kl_loss",
    "decoder_ce_loss",
    "decoder_token_accuracy",
    "positive_loss",
    "negative_loss",
    "old_deviation",
    "center_deviation",
    "reward_weight_mean",
    "theta_t_mean",
    "theta_sc_branch_fraction",
    "theta_old_empty_velocity_rms",
    "theta_sccfg_scale_mean",
    "theta_reconstructed_guidance_rms",
)

TRAINING_SCCFG_RNG_FOLD_IN = 0x53434346


class CenterTrainState(train_state.TrainState):
    """Full-model ShiftCenter optimizer state plus its local EMA."""

    ema_params: Any = None


class PolicyTrainState(train_state.TrainState):
    """Raw policy optimizer state plus an evaluation-only theta EMA.

    ``params`` remains the only policy used by the optimizer.  Callers also
    keep rollout/old-policy/anchor construction on their existing raw/old
    parameter paths; ``ema_params`` is an inert snapshot for evaluation and
    checkpoint resume only.
    """

    ema_params: Any = None


def _full_model_optimizer(
    params, *, learning_rate: float, max_grad_norm: float, weight_decay: float
):
    return optax.multi_transform(
        {
            "train": optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adamw(
                    learning_rate=learning_rate,
                    b1=0.9,
                    b2=0.999,
                    eps=1e-8,
                    weight_decay=weight_decay,
                ),
            ),
            "frozen_decoder": optax.set_to_zero(),
        },
        full_parameter_labels(params),
    )


def create_model_state(
    *, model_params, apply_fn, learning_rate: float, max_grad_norm: float,
    weight_decay: float, theta_ema_decay: float = 0.0,
) -> PolicyTrainState:
    return PolicyTrainState.create(
        apply_fn=apply_fn,
        params=model_params,
        tx=_full_model_optimizer(
            model_params,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
        ),
        ema_params=model_params if theta_ema_decay > 0.0 else None,
    )


def create_center_state(
    *, model_params, apply_fn, learning_rate: float, max_grad_norm: float,
    weight_decay: float,
) -> CenterTrainState:
    return CenterTrainState.create(
        apply_fn=apply_fn,
        params=model_params,
        tx=_full_model_optimizer(
            model_params,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
        ),
        ema_params=model_params,
    )


def _ema_update(ema_params, params, decay: float):
    return jax.tree_util.tree_map(
        lambda ema, value: ema * decay + value * (1.0 - decay),
        ema_params,
        params,
    )


def theta_ema_effective_decay(optimization_step, decay_cap: float):
    """Return the SD3 NFT early-ramp EMA decay for a completed theta step.

    This matches ``EMAModuleWrapper.get_current_decay`` when it is called
    after incrementing the optimizer step: ``min((1+s)/(10+s), cap)``.  Thus a
    cap of 0.9 starts at 2/11 after the first update and reaches 0.9 at step 80.
    """

    step = jnp.asarray(optimization_step, dtype=jnp.float32)
    return jnp.minimum((1.0 + step) / (10.0 + step), decay_cap)


def _sample_training_sccfg_scale(rng, batch_size, nft_config):
    """Draw the update-only SCCFG scale without perturbing legacy RNG draws."""

    distribution = getattr(
        nft_config, "training_self_cond_cfg_distribution", "fixed"
    )
    if distribution == "fixed":
        return jnp.full(
            (batch_size,),
            nft_config.self_cond_cfg_scale,
            dtype=jnp.float32,
        )
    if distribution == "log_uniform_one_plus_scale":
        return sample_cfg_scale(
            jax.random.fold_in(rng, TRAINING_SCCFG_RNG_FOLD_IN),
            batch_size,
            cfg_min=nft_config.training_self_cond_cfg_min,
            cfg_max=nft_config.training_self_cond_cfg_max,
        )
    raise ValueError(
        f"unsupported training_self_cond_cfg_distribution: {distribution!r}"
    )


def _sample_forward_process(rng, batch, elf_config, nft_config):
    """Draw a fresh classic-OT bridge independently of the rollout solver.

    In particular, an OWT endpoint may have been produced by an SDE rollout,
    but neither ``rollout_sampling_method`` nor ``rollout_sde_gamma`` enters
    this construction. NFT and center training use only the sealed endpoint,
    a newly sampled Gaussian source, and a newly sampled time.
    """

    timestep_rng, noise_rng, sc_branch_rng = jax.random.split(rng, 3)
    batch_size = batch["endpoints"].shape[0]
    t = sample_timesteps(
        timestep_rng,
        batch_size,
        P_mean=elf_config.denoiser_p_mean,
        P_std=elf_config.denoiser_p_std,
        time_schedule=nft_config.timestep_schedule,
    )
    # Classic OT velocity is defined for t < 1. Random samplers already obey
    # this almost surely; this guard prevents float32 sigmoid saturation.
    t = jnp.minimum(t, 1.0 - OT_MIN_REMAINING_TIME)
    fm_batch = make_endpoint_flow_batch(
        batch["endpoints"],
        t,
        noise_rng,
        noise_scale=elf_config.denoiser_noise_scale,
        t_eps=elf_config.t_eps,
        cond_seq=batch["cond_seq"],
        cond_seq_mask=batch["cond_seq_mask"],
    )
    mode = getattr(nft_config, "self_condition_mode", "bernoulli")
    if mode == "bernoulli":
        use_self_condition = jax.random.bernoulli(
            sc_branch_rng,
            p=nft_config.self_cond_update_probability,
            shape=(batch_size,),
        )
    elif mode in {"conditional_only", "paired"}:
        use_self_condition = jnp.ones((batch_size,), dtype=jnp.bool_)
    else:
        raise ValueError(f"unsupported self_condition_mode: {mode!r}")
    self_cond_cfg_scale = _sample_training_sccfg_scale(
        rng, batch_size, nft_config
    )
    return t, fm_batch, use_self_condition, self_cond_cfg_scale


def _training_field_batch(batch, t, fm_batch, old_condition, use_sc, elf_config, nft_config):
    """Route one OT batch over the SC axis or the independent source CFG axis.

    Paired mode concatenates the SAME state/time/target/mask twice, first with
    the frozen old SC prediction, then with empty SC. Multiplying the expanded
    mean by two yields the sum of branch means, not their average. The old
    first pass is outside the differentiated closure in both training stages.
    Classic source dual instead pairs source-present/source-null inputs and
    retains target-side SC in both; its caller reduces by an equal-weight mean.
    """

    common = {
        "z": fm_batch.state,
        "t": t,
        "cond_seq": batch["cond_seq"],
        "cond_seq_mask": batch["cond_seq_mask"],
        "self_condition": old_condition.value,
    }
    target, mask = fm_batch.target_velocity, batch["loss_mask"]
    branch_count = 1
    if getattr(nft_config, "source_cfg_mode", "legacy") == "cfg1_dual":
        # Both halves are SC-conditioned; only the source-text CFG condition differs.
        null = source_null_field_inputs(common)
        common = {key: jnp.concatenate([value, null[key]], axis=0)
                  for key, value in common.items()}
        target = jnp.concatenate([target, target], axis=0)
        mask = jnp.concatenate([mask, mask], axis=0)
        use_sc = jnp.concatenate([use_sc, use_sc], axis=0)
        branch_count = 2
    elif getattr(nft_config, "self_condition_mode", "bernoulli") == "paired":
        empty = empty_self_condition(fm_batch.state, batch["cond_seq"], batch["cond_seq_mask"])
        common = {key: jnp.concatenate([value, value], axis=0) for key, value in common.items()}
        common["self_condition"] = jnp.concatenate([old_condition.value, empty], axis=0)
        target = jnp.concatenate([target, target], axis=0)
        mask = jnp.concatenate([mask, mask], axis=0)
        use_sc = jnp.concatenate([use_sc, jnp.zeros_like(use_sc)], axis=0)
        branch_count = 2
    common.update(
        input_cfg_scale=training_input_cfg_scale(nft_config),
        self_cond_cfg_scale=nft_config.self_cond_cfg_scale,
        config=elf_config,
    )
    return common, target, mask, use_sc, branch_count


def build_recovered_paired_center(old_prediction, *, cfg_scale: float):
    """Build ``[recovered conditional, old empty]`` from a paired old field.

    ``_training_field_batch`` orders paired predictions with every conditional
    record first and the matching empty-SC records second.  Only the conditional
    center has ELF's intrinsic SC-CFG correction removed.  The empty half stays
    exactly equal to its old-policy prediction, preserving standard dual NFT
    semantics for that branch.
    """

    if (
        old_prediction.ndim != 3
        or old_prediction.shape[0] == 0
        or old_prediction.shape[0] % 2
    ):
        raise ValueError(
            "paired old_prediction must have even nonempty shape [2B, S, D]"
        )
    conditional, empty = jnp.split(old_prediction, 2, axis=0)
    recovered = recover_sccfg_base_field(
        conditional, empty, cfg_scale=cfg_scale
    )
    return jnp.concatenate([recovered, empty], axis=0)


def initialize_gradient_accumulator(params, *, metric_names):
    """Create device-local sums for one effective-batch optimizer update."""

    return (
        jax.tree_util.tree_map(jnp.zeros_like, params),
        {
            name: jnp.zeros((), dtype=jnp.float32)
            for name in metric_names
        },
        jnp.zeros((), dtype=jnp.int32),
    )


def accumulate_gradient_statistics(
    accumulator, gradients, metrics, num_records
):
    """Add a globally averaged microbatch, weighted by its record count."""

    gradient_sums, metric_sums, accumulated_records = accumulator
    gradient_sums = jax.tree_util.tree_map(
        lambda total, gradient: total
        + gradient * jnp.asarray(num_records, dtype=gradient.dtype),
        gradient_sums,
        gradients,
    )
    metric_sums = {
        name: metric_sums[name]
        + jnp.asarray(metrics[name], dtype=metric_sums[name].dtype)
        * jnp.asarray(num_records, dtype=metric_sums[name].dtype)
        for name in metric_sums
    }
    return (
        gradient_sums,
        metric_sums,
        accumulated_records + jnp.asarray(num_records, dtype=jnp.int32),
    )


def finalize_gradient_accumulator(accumulator):
    """Return record-weighted mean gradients/metrics without updating params."""

    gradient_sums, metric_sums, num_records = accumulator
    denominator = jnp.maximum(num_records, 1)
    mean_gradients = jax.tree_util.tree_map(
        lambda total: total / jnp.asarray(denominator, dtype=total.dtype),
        gradient_sums,
    )
    mean_metrics = {
        name: total / jnp.asarray(denominator, dtype=total.dtype)
        for name, total in metric_sums.items()
    }
    return (
        mean_gradients,
        mean_metrics,
        optax.global_norm(mean_gradients),
        num_records,
    )


def _global_microbatch_records(batch):
    local_records = jnp.asarray(batch["endpoints"].shape[0], dtype=jnp.int32)
    return jax.lax.psum(local_records, axis_name="batch")


def updates_empty_branch(nft_config, outer_step: int) -> bool:
    """Whether 1-based ``outer_step`` updates only the empty-SC policy field."""

    every = getattr(nft_config, "unconditional_update_every", 0)
    return bool(every) and outer_step % every == 0


def center_sccfg_scale(nft_config) -> float:
    """Model SC-CFG input of the learned center (defaults to the training field)."""

    return getattr(
        nft_config, "center_self_cond_cfg_scale", nft_config.self_cond_cfg_scale
    )


def _center_field_batch(
    old_params,
    batch,
    t,
    fm_batch,
    use_self_condition,
    *,
    model_apply_fn,
    elf_config,
    nft_config,
):
    """Route one OT batch for the center at its own native SC-CFG scale.

    The center is always a native field: the frozen old policy's first pass and
    the center's main pass both receive ``center_sccfg_scale``.  Center training
    and the policy's center evaluation share this construction, also when the
    updated policy field uses another scale or the CFG1 reconstruction.
    """

    scale = center_sccfg_scale(nft_config)
    old_condition = build_old_policy_self_condition(
        model_apply_fn=model_apply_fn,
        old_params=old_params,
        z=fm_batch.state,
        t=t,
        cond_seq=batch["cond_seq"],
        cond_seq_mask=batch["cond_seq_mask"],
        use_self_condition=use_self_condition,
        input_cfg_scale=training_input_cfg_scale(nft_config),
        self_cond_cfg_scale=scale,
        config=elf_config,
    )
    common, target, loss_mask, branch_mask, branch_count = _training_field_batch(
        batch, t, fm_batch, old_condition, use_self_condition, elf_config, nft_config
    )
    common["self_cond_cfg_scale"] = scale
    return old_condition, common, target, loss_mask, branch_mask, branch_count


def center_gradient_step(
    model_params,
    old_params,
    batch,
    rng,
    *,
    model_apply_fn,
    elf_config,
    nft_config,
):
    """Compute one globally averaged center microbatch gradient."""

    t, fm_batch, use_self_condition, _ = _sample_forward_process(
        rng, batch, elf_config, nft_config
    )
    old_condition, common, target, loss_mask, branch_mask, branch_count = (
        _center_field_batch(
            old_params,
            batch,
            t,
            fm_batch,
            use_self_condition,
            model_apply_fn=model_apply_fn,
            elf_config=elf_config,
            nft_config=nft_config,
        )
    )

    def loss_fn(model_params):
        prediction, _ = velocity_with_self_condition(
            model_apply_fn=model_apply_fn,
            model_params=model_params,
            **common,
        )
        per_example = masked_mse_per_example(
            prediction, target, loss_mask
        )
        return branch_count * per_example.mean(), {
            "center_loss_max": per_example.max(),
            "t_mean": t.mean(),
            "sc_branch_fraction": branch_mask.astype(jnp.float32).mean(),
            "old_empty_velocity_rms": jnp.sqrt(
                jnp.mean(old_condition.empty_velocity.astype(jnp.float32) ** 2)
            ),
        }

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_params)
    grads = zero_frozen_decoder_grads(grads)
    grads = jax.lax.pmean(grads, axis_name="batch")
    metrics = {
        "center_loss": loss,
        "center_loss_max": aux["center_loss_max"],
        "center_t_mean": aux["t_mean"],
        "center_sc_branch_fraction": aux["sc_branch_fraction"],
        "center_old_empty_velocity_rms": aux["old_empty_velocity_rms"],
    }
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return grads, metrics


def accumulate_center_gradients(
    accumulator,
    model_params,
    old_params,
    batch,
    rng,
    *,
    model_apply_fn,
    elf_config,
    nft_config,
):
    """Accumulate one center microbatch without changing optimizer state."""

    gradients, metrics = center_gradient_step(
        model_params,
        old_params,
        batch,
        rng,
        model_apply_fn=model_apply_fn,
        elf_config=elf_config,
        nft_config=nft_config,
    )
    return accumulate_gradient_statistics(
        accumulator,
        gradients,
        metrics,
        _global_microbatch_records(batch),
    )


def apply_center_gradients(
    state: CenterTrainState,
    gradients,
    *,
    nft_config,
):
    """Clip/apply one effective-batch center gradient and update its EMA."""

    grad_norm = optax.global_norm(gradients)
    new_state = state.apply_gradients(grads=gradients)
    updated_ema = _ema_update(
        state.ema_params, new_state.params, nft_config.center_ema_decay
    )
    updated_ema = restore_frozen_decoder_params(updated_ema, new_state.params)
    new_state = new_state.replace(ema_params=updated_ema)
    metrics = {
        "center_grad_norm": grad_norm,
        "center_param_norm": optax.global_norm(
            zero_frozen_decoder_grads(new_state.params)
        ),
        "center_frozen_decoder_delta_norm": frozen_decoder_delta_norm(
            state.params, new_state.params
        ),
        "center_frozen_decoder_ema_delta_norm": frozen_decoder_delta_norm(
            new_state.params, new_state.ema_params
        ),
    }
    return new_state, metrics


def center_train_step(
    state: CenterTrainState,
    old_params,
    batch,
    rng,
    *,
    model_apply_fn,
    elf_config,
    nft_config,
):
    """Apply one immediate center update (used by focused step tests)."""

    gradients, loss_metrics = center_gradient_step(
        state.params,
        old_params,
        batch,
        rng,
        model_apply_fn=model_apply_fn,
        elf_config=elf_config,
        nft_config=nft_config,
    )
    new_state, update_metrics = apply_center_gradients(
        state, gradients, nft_config=nft_config
    )
    return new_state, {**loss_metrics, **update_metrics}


def _native_center_velocity(
    center_params,
    old_params,
    batch,
    t,
    fm_batch,
    use_self_condition,
    *,
    model_apply_fn,
    elf_config,
    nft_config,
):
    """Evaluate the frozen center exactly as ``center_gradient_step`` fits it."""

    _, common, _, _, _, _ = _center_field_batch(
        old_params,
        batch,
        t,
        fm_batch,
        use_self_condition,
        model_apply_fn=model_apply_fn,
        elf_config=elf_config,
        nft_config=nft_config,
    )
    prediction, _ = velocity_with_self_condition(
        model_apply_fn=model_apply_fn,
        model_params=center_params,
        **common,
    )
    return prediction


def policy_gradient_step(
    model_params,
    base_params,
    old_params,
    center_params,
    batch,
    rng,
    *,
    model_apply_fn,
    elf_config,
    nft_config,
    update_branch: str = "configured",
):
    """Compute one globally averaged DiffusionNFT policy microbatch gradient.

    ``update_branch="empty"`` moves current/old/reference to the empty-SC
    input for this step while the learned center stays the single conditional
    center fitted by ``center_gradient_step``.
    """

    if update_branch not in {"configured", "empty"}:
        raise ValueError("update_branch must be configured or empty")
    t, fm_batch, use_self_condition, training_sccfg_scale = _sample_forward_process(
        rng, batch, elf_config, nft_config
    )
    center_use_self_condition = use_self_condition
    empty_update = update_branch == "empty"
    if empty_update:
        if nft_config.method != "shifted_center" or getattr(
            nft_config, "self_condition_mode", "bernoulli"
        ) != "conditional_only":
            raise ValueError(
                "an empty-branch update requires conditional_only shifted_center"
            )
        use_self_condition = jnp.zeros_like(use_self_condition)
    velocity_reconstruction = getattr(
        nft_config, "training_velocity_reconstruction", False
    )
    source_mode = getattr(nft_config, "source_cfg_mode", "legacy")
    if velocity_reconstruction:
        common = {
            "model_apply_fn": model_apply_fn,
            "z": fm_batch.state,
            "t": t,
            "cond_seq": batch["cond_seq"],
            "cond_seq_mask": batch["cond_seq_mask"],
            "use_self_condition": use_self_condition,
            "input_cfg_scale": training_input_cfg_scale(nft_config),
            "self_cond_cfg_scale": training_sccfg_scale,
            "config": elf_config,
        }
        target = fm_batch.target_velocity
        loss_mask = batch["loss_mask"]
        branch_mask = use_self_condition
        branch_count = 1
        loss_scale = 1
        old_operator = reconstructed_sccfg1_velocity(
            model_params=old_params,
            **common,
        )
        old_prediction = old_operator.effective_velocity
        old_empty_velocity = old_operator.empty_velocity
        reference_prediction = reconstructed_sccfg1_velocity(
            model_params=base_params,
            **common,
        ).effective_velocity
        center_prediction = (
            _native_center_velocity(
                center_params,
                old_params,
                batch,
                t,
                fm_batch,
                use_self_condition,
                model_apply_fn=model_apply_fn,
                elf_config=elf_config,
                nft_config=nft_config,
            )
            if nft_config.method == "shifted_center"
            else None
        )
    else:
        old_condition = build_old_policy_self_condition(
            model_apply_fn=model_apply_fn,
            old_params=old_params,
            z=fm_batch.state,
            t=t,
            cond_seq=batch["cond_seq"],
            cond_seq_mask=batch["cond_seq_mask"],
            use_self_condition=use_self_condition,
            input_cfg_scale=training_input_cfg_scale(nft_config),
            self_cond_cfg_scale=nft_config.self_cond_cfg_scale,
            config=elf_config,
        )
        common, target, loss_mask, branch_mask, branch_count = _training_field_batch(
            batch, t, fm_batch, old_condition, use_self_condition, elf_config, nft_config
        )
        common["model_apply_fn"] = model_apply_fn
        # Source dual follows SiT's equal-weight mean; legacy SC paired keeps its sum.
        loss_scale = 1 if source_mode == "cfg1_dual" else branch_count

        # Legacy fields share this exact frozen old-policy condition.
        old_prediction, _ = velocity_with_self_condition(
            model_params=old_params,
            **common,
        )
        reference_prediction, _ = velocity_with_self_condition(
            model_params=base_params,
            **common,
        )
        old_empty_velocity = old_condition.empty_velocity

        if nft_config.method == "shifted_center":
            if (
                center_sccfg_scale(nft_config) == nft_config.self_cond_cfg_scale
                and not empty_update
            ):
                center_prediction, _ = velocity_with_self_condition(
                    model_params=center_params,
                    **common,
                )
            else:
                center_prediction = _native_center_velocity(
                    center_params,
                    old_params,
                    batch,
                    t,
                    fm_batch,
                    center_use_self_condition,
                    model_apply_fn=model_apply_fn,
                    elf_config=elf_config,
                    nft_config=nft_config,
                )
        elif nft_config.method == "recovered_center":
            if branch_count != 2:
                raise ValueError("recovered_center requires a paired field batch")
            center_prediction = build_recovered_paired_center(
                old_prediction,
                cfg_scale=nft_config.self_cond_cfg_scale,
            )
        else:
            center_prediction = None

    reward_weight = advantage_to_weight(
        jnp.tile(batch["advantages"], branch_count), nft_config.adv_clip_max
    )

    def loss_fn(model_params):
        if velocity_reconstruction:
            current_operator = reconstructed_sccfg1_velocity(
                model_params=model_params,
                **common,
            )
            current_prediction = current_operator.effective_velocity
            reconstructed_guidance_rms = jnp.sqrt(
                jnp.mean(
                    current_operator.guidance.astype(jnp.float32) ** 2
                )
            )
        else:
            current_prediction, _ = velocity_with_self_condition(
                model_params=model_params,
                **common,
            )
            reconstructed_guidance_rms = jnp.zeros((), dtype=jnp.float32)
        positive, negative = compose_nft_predictions(
            current_prediction,
            old_prediction,
            beta=nft_config.beta,
            center=center_prediction,
        )
        positive_loss = masked_mse_per_example(
            positive, target, loss_mask
        )
        negative_loss = masked_mse_per_example(
            negative, target, loss_mask
        )
        unscaled_policy = (
            reward_weight * positive_loss
            + (1.0 - reward_weight) * negative_loss
        ) / nft_config.beta
        per_field_policy = nft_config.adv_clip_max * unscaled_policy
        policy_loss = loss_scale * per_field_policy.mean()
        policy_sc_loss = loss_scale * jnp.where(branch_mask, per_field_policy, 0.0).mean()
        policy_empty_loss = loss_scale * jnp.where(branch_mask, 0.0, per_field_policy).mean()
        source_cond_loss = source_uncond_loss = jnp.zeros_like(policy_loss)
        if source_mode == "cfg1_dual":
            conditional_losses, unconditional_losses = jnp.split(per_field_policy, 2)
            source_cond_loss = 0.5 * conditional_losses.mean()
            source_uncond_loss = 0.5 * unconditional_losses.mean()
        elif source_mode != "legacy":
            source_cond_loss = policy_loss
        kl_per_example = masked_mse_per_example(
            current_prediction, reference_prediction, loss_mask
        )
        kl_loss = loss_scale * kl_per_example.mean()
        if nft_config.decoder_ce_weight > 0:
            decoder_logits = _dlm_decoder_logits(
                z=batch["endpoints"],
                model_params=model_params,
                model_apply_fn=model_apply_fn,
                t_final_val=jnp.array(1.0, dtype=batch["endpoints"].dtype),
                config=elf_config,
                self_cond_cfg_scale=nft_config.self_cond_cfg_scale,
            )
            decoder_per_example = masked_token_cross_entropy_per_example(
                decoder_logits, batch["decoder_targets"], batch["loss_mask"]
            )
            decoder_ce_loss = decoder_per_example.mean()
            decoder_predictions = jnp.argmax(decoder_logits, axis=-1)
            decoder_mask = batch["loss_mask"].astype(jnp.float32)
            decoder_correct = (
                decoder_predictions == batch["decoder_targets"]
            ).astype(jnp.float32)
            decoder_accuracy = (
                (decoder_correct * decoder_mask).sum(axis=1)
                / jnp.maximum(decoder_mask.sum(axis=1), 1.0)
            ).mean()
        else:
            decoder_ce_loss = jnp.zeros((), dtype=policy_loss.dtype)
            decoder_accuracy = jnp.zeros((), dtype=policy_loss.dtype)
        total_loss = (
            policy_loss
            + nft_config.kl_beta * kl_loss
            + nft_config.decoder_ce_weight * decoder_ce_loss
        )
        old_deviation = masked_mse_per_example(
            current_prediction, old_prediction, loss_mask
        )
        if center_prediction is None:
            center_deviation = jnp.zeros_like(old_deviation)
        else:
            center_deviation = masked_mse_per_example(
                center_prediction, old_prediction, loss_mask
            )
        return total_loss, {
            "policy_loss": policy_loss,
            "policy_sc_loss": policy_sc_loss,
            "policy_empty_loss": policy_empty_loss,
            "policy_source_cond_loss": source_cond_loss,
            "policy_source_uncond_loss": source_uncond_loss,
            "kl_loss": kl_loss,
            "decoder_ce_loss": decoder_ce_loss,
            "decoder_token_accuracy": decoder_accuracy,
            "positive_loss": loss_scale * positive_loss.mean(),
            "negative_loss": loss_scale * negative_loss.mean(),
            "old_deviation": old_deviation.mean(),
            "center_deviation": center_deviation.mean(),
            "reward_weight_mean": reward_weight.mean(),
            "t_mean": t.mean(),
            "sc_branch_fraction": branch_mask.astype(jnp.float32).mean(),
            "old_empty_velocity_rms": jnp.sqrt(
                jnp.mean(old_empty_velocity.astype(jnp.float32) ** 2)
            ),
            "sccfg_scale_mean": training_sccfg_scale.astype(jnp.float32).mean(),
            "reconstructed_guidance_rms": reconstructed_guidance_rms,
        }

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_params)
    grads = zero_frozen_decoder_grads(grads)
    grads = jax.lax.pmean(grads, axis_name="batch")
    metrics = {
        "total_loss": loss,
        "policy_loss": aux["policy_loss"],
        "policy_sc_loss": aux["policy_sc_loss"],
        "policy_empty_loss": aux["policy_empty_loss"],
        "policy_source_cond_loss": aux["policy_source_cond_loss"],
        "policy_source_uncond_loss": aux["policy_source_uncond_loss"],
        "kl_loss": aux["kl_loss"],
        "decoder_ce_loss": aux["decoder_ce_loss"],
        "decoder_token_accuracy": aux["decoder_token_accuracy"],
        "positive_loss": aux["positive_loss"],
        "negative_loss": aux["negative_loss"],
        "old_deviation": aux["old_deviation"],
        "center_deviation": aux["center_deviation"],
        "reward_weight_mean": aux["reward_weight_mean"],
        "theta_t_mean": aux["t_mean"],
        "theta_sc_branch_fraction": aux["sc_branch_fraction"],
        "theta_old_empty_velocity_rms": aux["old_empty_velocity_rms"],
        "theta_sccfg_scale_mean": aux["sccfg_scale_mean"],
        "theta_reconstructed_guidance_rms": aux[
            "reconstructed_guidance_rms"
        ],
    }
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return grads, metrics


def accumulate_policy_gradients(
    accumulator,
    model_params,
    base_params,
    old_params,
    center_params,
    batch,
    rng,
    *,
    model_apply_fn,
    elf_config,
    nft_config,
    update_branch: str = "configured",
):
    """Accumulate one policy microbatch without changing optimizer state."""

    gradients, metrics = policy_gradient_step(
        model_params,
        base_params,
        old_params,
        center_params,
        batch,
        rng,
        model_apply_fn=model_apply_fn,
        elf_config=elf_config,
        nft_config=nft_config,
        update_branch=update_branch,
    )
    return accumulate_gradient_statistics(
        accumulator,
        gradients,
        metrics,
        _global_microbatch_records(batch),
    )


def apply_policy_gradients(
    state: train_state.TrainState,
    gradients,
    *,
    theta_ema_decay: float = 0.0,
):
    """Apply one policy update, then advance its evaluation-only theta EMA."""

    grad_norm = optax.global_norm(gradients)
    new_state = state.apply_gradients(grads=gradients)
    ema_decay = jnp.asarray(0.0, dtype=jnp.float32)
    ema_deviation = jnp.asarray(0.0, dtype=jnp.float32)
    if getattr(state, "ema_params", None) is not None:
        ema_decay = theta_ema_effective_decay(
            new_state.step, theta_ema_decay
        )
        updated_ema = _ema_update(
            state.ema_params, new_state.params, ema_decay
        )
        updated_ema = restore_frozen_decoder_params(
            updated_ema, new_state.params
        )
        new_state = new_state.replace(ema_params=updated_ema)
        ema_deviation = optax.global_norm(
            jax.tree_util.tree_map(
                lambda raw, ema: raw - ema,
                new_state.params,
                new_state.ema_params,
            )
        )
    metrics = {
        "theta_grad_norm": grad_norm,
        "theta_param_norm": optax.global_norm(
            zero_frozen_decoder_grads(new_state.params)
        ),
        "theta_frozen_decoder_delta_norm": frozen_decoder_delta_norm(
            state.params, new_state.params
        ),
        "theta_ema_effective_decay": ema_decay,
        "theta_ema_deviation": ema_deviation,
    }
    return new_state, metrics


def policy_train_step(
    state: train_state.TrainState,
    base_params,
    old_params,
    center_params,
    batch,
    rng,
    *,
    model_apply_fn,
    elf_config,
    nft_config,
):
    """Apply one immediate policy update (used by focused step tests)."""

    gradients, loss_metrics = policy_gradient_step(
        state.params,
        base_params,
        old_params,
        center_params,
        batch,
        rng,
        model_apply_fn=model_apply_fn,
        elf_config=elf_config,
        nft_config=nft_config,
    )
    new_state, update_metrics = apply_policy_gradients(
        state,
        gradients,
        theta_ema_decay=getattr(nft_config, "theta_ema_decay", 0.0),
    )
    return new_state, {**loss_metrics, **update_metrics}
