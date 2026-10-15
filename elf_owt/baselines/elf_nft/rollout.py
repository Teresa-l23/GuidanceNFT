"""Old-policy rollout collection for WMT/BLEU and null-conditioned OWT."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from configs.config import SamplingConfig
from utils.data_utils import get_pad_token_id
from utils.encoder_utils import encode_text
from utils.generation_utils import (
    _dlm_decode_batch,
    _generate_samples_single_batch,
)
from utils.sampling_utils import get_sampling_steps

from .nft_math import combine_group_normalized_rewards, group_normalize_rewards
from .reward import (
    PPL_ONLY,
    batch_reference_texts,
    capped_unigram_entropy_rewards,
    conditional_decoder_target_mask,
    corpus_bleu,
    decode_target_texts,
    decode_unconditional_texts,
    decoder_target_mask,
    sentence_bleu_rewards,
)
from .xsum_scoring import (
    decode_xsum_hypotheses,
    xsum_decoder_target_mask,
    xsum_rouge_scores,
)


@dataclass(frozen=True)
class RolloutFunctions:
    encode: object | None
    generate: object
    decode: object


@dataclass(frozen=True)
class RolloutResult:
    buffer: dict[str, np.ndarray]
    metrics: dict[str, float]
    hypotheses: list[str]
    references: list[str] | None


@dataclass(frozen=True)
class PolicyEvaluationResult:
    metrics: dict[str, float]
    hypotheses: list[str]


def resolve_rollout_sccfg_settings(nft_config) -> tuple[float, float]:
    """Return ``(model_w, external_scale)`` for the configured rollout field.

    Legacy configurations use their single ``self_cond_cfg_scale`` as the
    rollout model input and return the native field.  Schema-v8 experiments can
    instead choose a separate model input and request ``recovered_base``.  The
    latter derives the only valid extrapolation coefficient, ``1 / model_w``,
    so ``empty + coefficient * (conditional - empty)`` exactly removes ELF's
    intrinsic SC-CFG correction.
    """

    model_scale = float(
        getattr(
            nft_config,
            "rollout_self_cond_cfg_scale",
            nft_config.self_cond_cfg_scale,
        )
    )
    field_mode = getattr(nft_config, "rollout_field_mode", "native")
    if not np.isfinite(model_scale) or model_scale <= 0:
        raise ValueError("rollout model SC-CFG scale must be finite and positive")
    if field_mode == "native":
        return model_scale, 1.0
    if field_mode == "recovered_base":
        if model_scale <= 1.0:
            raise ValueError("recovered_base rollout requires model SC-CFG > 1")
        return model_scale, 1.0 / model_scale
    raise ValueError(f"unsupported rollout_field_mode: {field_mode!r}")


def external_sc_guidance_apply(model_apply_fn, scale):
    """Wrap an OWT denoiser with external CFG at a fixed network CFG input.

    Each call receives [z, SC] and evaluates [z, SC] and [z, 0] with the
    same parameters, t, and model w. Only clean predictions are extrapolated;
    the caller must use the original apply function for token decoding.
    """

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("external SC guidance scale must be finite and positive")
    if scale == 1.0:
        return model_apply_fn

    def guided_apply(variables, x, t, **kwargs):
        state, self_condition = jnp.split(x, 2, axis=-1)
        conditional, logits = model_apply_fn(variables, x, t, **kwargs)
        empty, _ = model_apply_fn(
            variables, jnp.concatenate([state, jnp.zeros_like(self_condition)], axis=-1),
            t, **kwargs,
        )
        return empty + scale * (conditional - empty), logits

    return guided_apply


def make_rollout_functions(
    *, model_apply_fn, elf_config, nft_config, encoder_apply_fn=None,
    external_cfg_scale=1.0,
) -> RolloutFunctions:
    """Compile task-aware pmap entry points once for the full online run."""

    rollout_model_scale, configured_external_scale = (
        resolve_rollout_sccfg_settings(nft_config)
    )
    effective_external_scale = configured_external_scale * external_cfg_scale
    if effective_external_scale != 1.0 and nft_config.task != "owt":
        raise ValueError("external SC guidance currently supports OWT only")
    sampling_apply_fn = external_sc_guidance_apply(
        model_apply_fn, effective_external_scale
    )
    sampling_config = SamplingConfig(
        sampling_method=nft_config.rollout_sampling_method,
        num_sampling_steps=[nft_config.rollout_steps],
        cfgs=[nft_config.rollout_cfg_scale],
        self_cond_cfg_scales=[rollout_model_scale],
        time_schedule=nft_config.rollout_time_schedule,
        sde_gamma=nft_config.rollout_sde_gamma,
    )
    p_encode = None
    if nft_config.task in {"wmt", "xsum"}:
        if encoder_apply_fn is None:
            raise ValueError("conditional rollout requires encoder_apply_fn")
        p_encode = jax.pmap(
            partial(
                encode_text,
                encoder_apply_fn=encoder_apply_fn,
                latent_mean=elf_config.latent_mean,
                latent_std=elf_config.latent_std,
            ),
            axis_name="batch",
        )
    p_generate = jax.pmap(
        partial(
            _generate_samples_single_batch,
            model_apply_fn=sampling_apply_fn,
            config=elf_config,
            sampling_config=sampling_config,
            cfg_scale=nft_config.rollout_cfg_scale,
            self_cond_cfg_scale=rollout_model_scale,
        ),
        axis_name="batch",
    )
    p_decode = jax.pmap(
        partial(
            _dlm_decode_batch,
            model_apply_fn=model_apply_fn,
            config=elf_config,
            self_cond_cfg_scale=rollout_model_scale,
        )
    )
    return RolloutFunctions(encode=p_encode, generate=p_generate, decode=p_decode)


def _reshape_for_devices(array, num_devices: int):
    if array.shape[0] % num_devices:
        raise ValueError(
            f"batch {array.shape[0]} is not divisible by {num_devices} devices"
        )
    return array.reshape(num_devices, array.shape[0] // num_devices, *array.shape[1:])


def _repeat_rows(array, repeats: int):
    return jnp.repeat(jnp.asarray(array), repeats=repeats, axis=0)


def _generate_and_decode(
    *,
    functions,
    old_params_replicated,
    cond_seq,
    cond_seq_mask,
    embedding_dim,
    batch_size,
    rng,
    elf_config,
    nft_config,
):
    """Generate endpoint latents; SDE/ODE is confined to this rollout helper."""

    num_devices = jax.local_device_count()
    if batch_size % num_devices:
        raise ValueError("rollout generation batch must divide across local devices")
    noise_rng, schedule_rng, sampler_rng = jax.random.split(rng, 3)
    per_device = batch_size // num_devices
    device_noise_rngs = jax.random.split(noise_rng, num_devices)
    noise_dtype = jnp.float32 if cond_seq is None else jnp.asarray(cond_seq).dtype
    z_sharded = jnp.stack(
        [
            jax.random.normal(
                device_noise_rngs[index],
                (per_device, elf_config.max_length, embedding_dim),
                dtype=noise_dtype,
            )
            * elf_config.denoiser_noise_scale
            for index in range(num_devices)
        ]
    )
    device_schedule_rngs = jax.random.split(schedule_rng, num_devices)
    t_steps_sharded = jnp.stack(
        [
            get_sampling_steps(
                device_schedule_rngs[index],
                n_steps=nft_config.rollout_steps,
                time_schedule=nft_config.rollout_time_schedule,
                P_mean=elf_config.denoiser_p_mean,
                P_std=elf_config.denoiser_p_std,
            )
            for index in range(num_devices)
        ]
    )
    device_sampler_rngs = jax.random.split(sampler_rng, num_devices)
    sharded_cond_seq = (
        None if cond_seq is None else _reshape_for_devices(cond_seq, num_devices)
    )
    sharded_cond_mask = (
        None
        if cond_seq_mask is None
        else _reshape_for_devices(cond_seq_mask, num_devices)
    )
    endpoints_sharded = functions.generate(
        model_params=old_params_replicated,
        rng=device_sampler_rngs,
        z=z_sharded,
        t_steps=t_steps_sharded,
        cond_seq=sharded_cond_seq,
        cond_seq_mask=sharded_cond_mask,
    )
    endpoints_sharded.block_until_ready()
    predicted_ids_sharded = functions.decode(
        z=endpoints_sharded,
        model_params=old_params_replicated,
        t_final_val=t_steps_sharded[:, -1],
    )
    predicted_ids_sharded.block_until_ready()

    endpoints = np.asarray(
        endpoints_sharded.reshape(
            batch_size,
            endpoints_sharded.shape[-2],
            endpoints_sharded.shape[-1],
        )
    )
    if not np.isfinite(endpoints).all():
        raise FloatingPointError("old-policy rollout produced non-finite endpoints")
    predicted_ids = np.asarray(
        predicted_ids_sharded.reshape(
            batch_size, predicted_ids_sharded.shape[-1]
        )
    )
    return endpoints, predicted_ids


def _generate_and_decode_in_batches(
    *,
    functions,
    old_params_replicated,
    cond_seq,
    cond_seq_mask,
    embedding_dim,
    rng,
    elf_config,
    nft_config,
):
    """Generate a sealed rollout in bounded batches without changing its order."""

    total = int(nft_config.rollout_batch_size)
    generation_batch_size = int(
        getattr(nft_config, "rollout_generation_batch_size", total)
    )
    if total % generation_batch_size:
        raise ValueError(
            "rollout batch must divide evenly into rollout generation batches"
        )
    num_batches = total // generation_batch_size
    batch_rngs = (
        (rng,) if num_batches == 1 else tuple(jax.random.split(rng, num_batches))
    )
    endpoint_batches = []
    predicted_id_batches = []
    for batch_index, batch_rng in enumerate(batch_rngs):
        start = batch_index * generation_batch_size
        stop = start + generation_batch_size
        endpoint_batch, predicted_id_batch = _generate_and_decode(
            functions=functions,
            old_params_replicated=old_params_replicated,
            cond_seq=None if cond_seq is None else cond_seq[start:stop],
            cond_seq_mask=(
                None if cond_seq_mask is None else cond_seq_mask[start:stop]
            ),
            embedding_dim=embedding_dim,
            batch_size=generation_batch_size,
            rng=batch_rng,
            elf_config=elf_config,
            nft_config=nft_config,
        )
        endpoint_batches.append(endpoint_batch)
        predicted_id_batches.append(predicted_id_batch)
    return (
        np.concatenate(endpoint_batches, axis=0),
        np.concatenate(predicted_id_batches, axis=0),
    )


def evaluate_owt_policy(
    *,
    functions,
    policy_params_replicated,
    tokenizer,
    reward_scorer,
    embedding_dim,
    rng,
    elf_config,
    nft_config,
    num_samples: int,
    generation_batch_size: int,
) -> PolicyEvaluationResult:
    """Evaluate an OWT policy on independent fixed-seed generations.

    Generation is streamed in bounded batches and endpoint latents are discarded
    after decoding, so a full evaluation does not materialize a
    ``[num_samples, sequence, embedding]`` host buffer.  The supplied RNG is
    independent of the training RNG, and this function never updates the reward
    controller or any model parameters.
    """

    if reward_scorer is None:
        raise ValueError("OWT full evaluation requires a causal-LM reward scorer")
    if embedding_dim is None or embedding_dim <= 0:
        raise ValueError("OWT full evaluation requires a positive embedding_dim")
    if num_samples <= 0 or generation_batch_size <= 0:
        raise ValueError("evaluation sample and generation batch counts must be positive")
    if num_samples % generation_batch_size:
        raise ValueError("evaluation samples must divide into generation batches")
    num_devices = jax.local_device_count()
    if generation_batch_size % num_devices:
        raise ValueError("evaluation generation batch must divide across local devices")

    num_batches = num_samples // generation_batch_size
    batch_rngs = (
        (rng,) if num_batches == 1 else tuple(jax.random.split(rng, num_batches))
    )
    hypotheses = []
    for batch_rng in batch_rngs:
        endpoints, predicted_ids = _generate_and_decode(
            functions=functions,
            old_params_replicated=policy_params_replicated,
            cond_seq=None,
            cond_seq_mask=None,
            embedding_dim=embedding_dim,
            batch_size=generation_batch_size,
            rng=batch_rng,
            elf_config=elf_config,
            nft_config=nft_config,
        )
        hypotheses.extend(decode_unconditional_texts(tokenizer, predicted_ids))
        del endpoints, predicted_ids
    if len(hypotheses) != num_samples:
        raise RuntimeError(
            f"full evaluation decoded {len(hypotheses)} samples, expected {num_samples}"
        )

    rewards, sample_entropies, reward_metrics = reward_scorer.score(hypotheses)
    rewards = np.asarray(rewards, dtype=np.float64)
    sample_entropies = np.asarray(sample_entropies, dtype=np.float64)
    if rewards.shape != (num_samples,) or sample_entropies.shape != (num_samples,):
        raise ValueError("full evaluation scorer returned misaligned sample metrics")
    metrics = {
        "ppl": float(reward_metrics["reward_model_ppl"]),
        "entropy": float(reward_metrics["reward_model_unigram_entropy"]),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "sample_entropy_mean": float(sample_entropies.mean()),
        "sample_entropy_std": float(sample_entropies.std()),
        **{name: float(value) for name, value in reward_metrics.items()},
    }
    return PolicyEvaluationResult(metrics=metrics, hypotheses=hypotheses)


def _collect_wmt_rollout(
    *,
    functions,
    old_params_replicated,
    encoder_params_replicated,
    prompt_batch,
    tokenizer,
    rng,
    elf_config,
    nft_config,
):
    if functions.encode is None or encoder_params_replicated is None:
        raise ValueError("WMT rollout requires encoder function and parameters")
    if prompt_batch is None:
        raise ValueError("WMT rollout requires a prompt batch")
    num_devices = jax.local_device_count()
    num_prompts = int(prompt_batch["input_ids"].shape[0])
    if num_prompts != nft_config.prompt_groups_per_outer:
        raise ValueError(
            f"prompt batch has {num_prompts}, expected {nft_config.prompt_groups_per_outer}"
        )
    if num_prompts % num_devices:
        raise ValueError("WMT prompt batch must divide across local devices")

    input_ids = jnp.asarray(prompt_batch["input_ids"])
    encoder_attention_mask = jnp.asarray(prompt_batch["encoder_attention_mask"])
    cond_seq_mask_unique = jnp.asarray(prompt_batch["cond_seq_mask"])
    encoded_unique = functions.encode(
        input_ids=_reshape_for_devices(input_ids, num_devices),
        attention_mask=_reshape_for_devices(encoder_attention_mask, num_devices),
        encoder_params=encoder_params_replicated,
    )
    encoded_unique = encoded_unique.reshape(
        num_prompts, encoded_unique.shape[-2], encoded_unique.shape[-1]
    )

    repeats = nft_config.samples_per_prompt
    cond_seq = _repeat_rows(encoded_unique, repeats)
    cond_seq_mask = _repeat_rows(cond_seq_mask_unique, repeats)
    references = batch_reference_texts(
        prompt_batch,
        expected_count=num_prompts,
        repeats=repeats,
    )

    endpoints, predicted_ids = _generate_and_decode_in_batches(
        functions=functions,
        old_params_replicated=old_params_replicated,
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
        embedding_dim=encoded_unique.shape[-1],
        rng=rng,
        elf_config=elf_config,
        nft_config=nft_config,
    )
    cond_mask_np = np.asarray(cond_seq_mask)
    hypothesis_max_tokens = (
        getattr(nft_config, "hypothesis_max_tokens", None) or None
    )
    loss_mask = conditional_decoder_target_mask(
        predicted_ids,
        cond_mask_np,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=get_pad_token_id(tokenizer, elf_config.pad_token),
        max_target_tokens=hypothesis_max_tokens,
    )
    hypotheses = decode_target_texts(
        tokenizer,
        predicted_ids,
        cond_mask_np,
        max_target_tokens=hypothesis_max_tokens,
    )
    rewards = sentence_bleu_rewards(hypotheses, references)
    grouped_rewards = rewards.reshape(num_prompts, repeats)
    advantages = group_normalize_rewards(grouped_rewards).reshape(-1)
    reward_std = grouped_rewards.std(axis=1)
    buffer = {
        "endpoints": endpoints,
        "cond_seq": np.asarray(cond_seq),
        "cond_seq_mask": cond_mask_np,
        "loss_mask": loss_mask,
        # As in OWT, CE preserves the sealed endpoint's own old-policy decode.
        # The paired WMT target is used only for sentence BLEU and evaluation.
        "decoder_targets": predicted_ids.astype(np.int32),
        "advantages": advantages.astype(np.float32),
        "rewards": rewards.astype(np.float32),
    }
    metrics = {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "group_reward_std_mean": float(reward_std.mean()),
        "zero_reward_std_fraction": float((reward_std == 0).mean()),
        "rollout_corpus_bleu": corpus_bleu(hypotheses, references),
    }
    return RolloutResult(buffer, metrics, hypotheses, references)


def _collect_xsum_rollout(
    *,
    functions,
    old_params_replicated,
    encoder_params_replicated,
    prompt_batch,
    tokenizer,
    rng,
    elf_config,
    nft_config,
):
    """Collect conditional XSum endpoints scored by per-example ROUGE-L."""

    if functions.encode is None or encoder_params_replicated is None:
        raise ValueError("XSum rollout requires encoder function and parameters")
    if prompt_batch is None:
        raise ValueError("XSum rollout requires a prompt batch")
    num_devices = jax.local_device_count()
    num_prompts = int(prompt_batch["input_ids"].shape[0])
    if num_prompts != nft_config.prompt_groups_per_outer:
        raise ValueError(
            f"prompt batch has {num_prompts}, expected "
            f"{nft_config.prompt_groups_per_outer}"
        )
    if num_prompts % num_devices:
        raise ValueError("XSum prompt batch must divide across local devices")
    references = prompt_batch.get("target")
    if references is None or isinstance(references, (str, bytes)):
        raise ValueError("XSum rollout requires one original target per article")
    references = [str(reference) for reference in references]
    if len(references) != num_prompts:
        raise ValueError("XSum rollout target references are misaligned")

    input_ids = jnp.asarray(prompt_batch["input_ids"])
    encoder_attention_mask = jnp.asarray(prompt_batch["encoder_attention_mask"])
    cond_seq_mask_unique = jnp.asarray(prompt_batch["cond_seq_mask"])
    encoded_unique = functions.encode(
        input_ids=_reshape_for_devices(input_ids, num_devices),
        attention_mask=_reshape_for_devices(encoder_attention_mask, num_devices),
        encoder_params=encoder_params_replicated,
    )
    encoded_unique = encoded_unique.reshape(
        num_prompts, encoded_unique.shape[-2], encoded_unique.shape[-1]
    )

    repeats = nft_config.samples_per_prompt
    cond_seq = _repeat_rows(encoded_unique, repeats)
    cond_seq_mask = _repeat_rows(cond_seq_mask_unique, repeats)
    references = [reference for reference in references for _ in range(repeats)]
    endpoints, predicted_ids = _generate_and_decode_in_batches(
        functions=functions,
        old_params_replicated=old_params_replicated,
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
        embedding_dim=encoded_unique.shape[-1],
        rng=rng,
        elf_config=elf_config,
        nft_config=nft_config,
    )
    cond_mask_np = np.asarray(cond_seq_mask)
    max_tokens = int(nft_config.hypothesis_max_tokens)
    loss_mask = xsum_decoder_target_mask(
        predicted_ids,
        cond_mask_np,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=get_pad_token_id(tokenizer, elf_config.pad_token),
        max_tokens=max_tokens,
    )
    hypotheses = decode_xsum_hypotheses(
        tokenizer,
        predicted_ids,
        cond_mask_np,
        max_tokens=max_tokens,
    )
    scores = xsum_rouge_scores(hypotheses, references)
    # The optimization reward is the evaluator's exact per-example ROUGE-L
    # array, not a second scorer or a different aggregation.
    rewards = scores.rougeL.copy()
    grouped_rewards = rewards.reshape(num_prompts, repeats)
    advantages = group_normalize_rewards(grouped_rewards).reshape(-1)
    reward_std = grouped_rewards.std(axis=1)
    buffer = {
        "endpoints": endpoints,
        "cond_seq": np.asarray(cond_seq),
        "cond_seq_mask": cond_mask_np,
        "loss_mask": loss_mask,
        "decoder_targets": predicted_ids.astype(np.int32),
        "advantages": advantages.astype(np.float32),
        "rewards": rewards.astype(np.float32),
    }
    metrics = {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "group_reward_std_mean": float(reward_std.mean()),
        "zero_reward_std_fraction": float((reward_std == 0).mean()),
        **{f"rollout_{name}": value for name, value in scores.means().items()},
        "hypothesis_max_tokens": float(max_tokens),
        "empty_hypothesis_fraction": float(
            np.asarray([not text.strip() for text in hypotheses]).mean()
        ),
    }
    return RolloutResult(buffer, metrics, hypotheses, references)


def _collect_owt_rollout(
    *,
    functions,
    old_params_replicated,
    tokenizer,
    reward_scorer,
    embedding_dim,
    rng,
    elf_config,
    nft_config,
    entropy_weight,
):
    if reward_scorer is None:
        raise ValueError("OWT rollout requires a causal-LM reward scorer")
    if embedding_dim is None or embedding_dim <= 0:
        raise ValueError("OWT rollout requires a positive embedding_dim")
    endpoints, predicted_ids = _generate_and_decode_in_batches(
        functions=functions,
        old_params_replicated=old_params_replicated,
        cond_seq=None,
        cond_seq_mask=None,
        embedding_dim=embedding_dim,
        rng=rng,
        elf_config=elf_config,
        nft_config=nft_config,
    )
    hypotheses = decode_unconditional_texts(tokenizer, predicted_ids)
    ppl_rewards, sample_entropies, reward_metrics = reward_scorer.score(hypotheses)
    grouped_ppl_rewards = ppl_rewards.reshape(1, nft_config.rollout_batch_size)
    reward_mode = getattr(nft_config, "reward_combination_mode", PPL_ONLY)
    entropy_target = float(getattr(nft_config, "entropy_target", 0.0))
    if reward_mode == PPL_ONLY:
        entropy_rewards = sample_entropies.copy()
        entropy_clip_fraction = 0.0
    else:
        entropy_rewards = capped_unigram_entropy_rewards(
            sample_entropies, entropy_target
        )
        entropy_clip_fraction = float((sample_entropies >= entropy_target).mean())
    grouped_entropy_rewards = entropy_rewards.reshape(
        1, nft_config.rollout_batch_size
    )
    combined, ppl_advantages, entropy_advantages = (
        combine_group_normalized_rewards(
            grouped_ppl_rewards,
            grouped_entropy_rewards,
            auxiliary_weight=entropy_weight,
        )
    )
    advantages = combined.reshape(-1)
    ppl_advantages = ppl_advantages.reshape(-1)
    entropy_advantages = entropy_advantages.reshape(-1)
    reward_std = float(grouped_ppl_rewards.std())
    ppl_entropy_correlation = 0.0
    if ppl_rewards.std() > 0 and sample_entropies.std() > 0:
        ppl_entropy_correlation = float(
            np.corrcoef(ppl_rewards, sample_entropies)[0, 1]
        )

    cond_seq = np.zeros_like(endpoints)
    cond_seq_mask = np.zeros(endpoints.shape[:2], dtype=np.float32)
    loss_mask = decoder_target_mask(
        predicted_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    buffer = {
        "endpoints": endpoints,
        "cond_seq": cond_seq,
        "cond_seq_mask": cond_seq_mask,
        "loss_mask": loss_mask,
        # The generated endpoint's own old-policy decode is the only aligned CE
        # pseudo-target; an unrelated OWT corpus row would be scientifically wrong.
        "decoder_targets": predicted_ids.astype(np.int32),
        "advantages": advantages.astype(np.float32),
        # Keep ``rewards`` as the historical PPL reward for preview/checkpoint
        # consumers; the independently normalized streams are explicit below.
        "rewards": ppl_rewards.astype(np.float32),
        "ppl_rewards": ppl_rewards.astype(np.float32),
        "sample_entropies": sample_entropies.astype(np.float32),
        "entropy_rewards": entropy_rewards.astype(np.float32),
        "ppl_advantages": ppl_advantages.astype(np.float32),
        "entropy_advantages": entropy_advantages.astype(np.float32),
    }
    metrics = {
        "reward_mean": float(ppl_rewards.mean()),
        "reward_std": float(ppl_rewards.std()),
        "group_reward_std_mean": reward_std,
        "zero_reward_std_fraction": float(reward_std == 0.0),
        "reward_ppl_weight": float(1.0 - entropy_weight),
        "reward_entropy_weight": float(entropy_weight),
        "reward_entropy_target": entropy_target,
        "reward_entropy_mean_all": float(sample_entropies.mean()),
        "reward_entropy_capped_mean": float(entropy_rewards.mean()),
        "reward_entropy_clip_fraction": entropy_clip_fraction,
        "reward_ppl_entropy_correlation": ppl_entropy_correlation,
        "reward_ppl_advantage_std": float(ppl_advantages.std()),
        "reward_entropy_advantage_std": float(entropy_advantages.std()),
        "reward_combined_advantage_std": float(advantages.std()),
        "reward_combined_advantage_clip_fraction": float(
            (np.abs(advantages) >= nft_config.adv_clip_max).mean()
        ),
        **reward_metrics,
    }
    return RolloutResult(buffer, metrics, hypotheses, None)


def collect_old_policy_rollout(
    *,
    functions: RolloutFunctions,
    old_params_replicated,
    tokenizer,
    rng,
    elf_config,
    nft_config,
    encoder_params_replicated=None,
    prompt_batch=None,
    reward_scorer=None,
    embedding_dim=None,
    entropy_weight=0.0,
) -> RolloutResult:
    """Seal a rewarded endpoint buffer for the explicitly selected task.

    WMT uses sentence BLEU, XSum uses per-example ROUGE-L F1, and OWT uses
    frozen causal-LM log-likelihood.  The selected ODE or SDE solver ends here:
    center/policy updates independently construct fresh classic-OT bridges from
    the returned endpoint buffer.
    """

    if nft_config.task == "wmt":
        return _collect_wmt_rollout(
            functions=functions,
            old_params_replicated=old_params_replicated,
            encoder_params_replicated=encoder_params_replicated,
            prompt_batch=prompt_batch,
            tokenizer=tokenizer,
            rng=rng,
            elf_config=elf_config,
            nft_config=nft_config,
        )
    if nft_config.task == "xsum":
        return _collect_xsum_rollout(
            functions=functions,
            old_params_replicated=old_params_replicated,
            encoder_params_replicated=encoder_params_replicated,
            prompt_batch=prompt_batch,
            tokenizer=tokenizer,
            rng=rng,
            elf_config=elf_config,
            nft_config=nft_config,
        )
    if nft_config.task == "owt":
        return _collect_owt_rollout(
            functions=functions,
            old_params_replicated=old_params_replicated,
            tokenizer=tokenizer,
            reward_scorer=reward_scorer,
            embedding_dim=embedding_dim,
            rng=rng,
            elf_config=elf_config,
            nft_config=nft_config,
            entropy_weight=entropy_weight,
        )
    raise ValueError(f"unsupported task: {nft_config.task!r}")
