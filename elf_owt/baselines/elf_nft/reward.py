"""Task-specific rewards for conditional WMT and unconditional OWT."""

from __future__ import annotations

import numpy as np
import sacrebleu


PPL_ONLY = "ppl_only"
FIXED_JOINT = "fixed_joint"
ADAPTIVE_ENTROPY_FLOOR = "adaptive_entropy_floor"
OWT_REWARD_MODES = frozenset({PPL_ONLY, FIXED_JOINT, ADAPTIVE_ENTROPY_FLOOR})


def batch_reference_texts(batch, *, expected_count: int, repeats: int = 1) -> list[str]:
    """Return original WMT target strings in row-repeat rollout order.

    Training and held-out evaluation intentionally share this path.  Tokenized
    ``input_ids`` may have been truncated to the model sequence length, so they
    are never decoded back into references here.
    """

    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError("expected_count must be a positive integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if "target" not in batch:
        raise ValueError("WMT batch requires original target reference text")
    targets = batch["target"]
    if isinstance(targets, (str, bytes)):
        raise ValueError("WMT batch target must contain one string per example")
    references = [str(target) for target in targets]
    if len(references) != expected_count:
        raise ValueError(
            f"WMT batch has {len(references)} references, expected {expected_count}"
        )
    return [reference for reference in references for _ in range(repeats)]


def conditional_target_token_rows(
    token_ids,
    cond_seq_mask,
    eos_token_id: int,
    *,
    max_target_tokens: int | None = None,
):
    """Extract generated target rows after the actual contiguous source prefix.

    When ``max_target_tokens`` is provided, truncation happens on token ids
    before first-EOS handling and decoding.  This is the common conditional
    hypothesis contract used by both WMT and XSum reward/evaluation.
    """

    ids = np.asarray(token_ids)
    mask = np.asarray(cond_seq_mask)
    if ids.ndim != 2 or ids.shape[0] == 0 or mask.shape != ids.shape:
        raise ValueError(
            "token_ids and cond_seq_mask must share a nonempty [B, S] shape"
        )
    if max_target_tokens is not None and (
        isinstance(max_target_tokens, bool)
        or not isinstance(max_target_tokens, int)
        or max_target_tokens <= 0
    ):
        raise ValueError("max_target_tokens must be a positive integer or None")
    if not np.isfinite(mask).all() or np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("cond_seq_mask must be finite and binary")
    rows = []
    for sample_ids, sample_mask in zip(ids, mask, strict=True):
        start = int(sample_mask.astype(np.int32).sum())
        if np.any(sample_mask[:start] != 1) or np.any(sample_mask[start:] != 0):
            raise ValueError("each cond_seq_mask row must be a contiguous prefix")
        target = sample_ids[start:].tolist()
        if max_target_tokens is not None:
            target = target[:max_target_tokens]
        if eos_token_id in target:
            target = target[: target.index(eos_token_id)]
        rows.append(target)
    return rows


def decode_target_texts(
    tokenizer,
    token_ids,
    cond_seq_mask,
    *,
    max_target_tokens: int | None = None,
) -> list[str]:
    """Strip each source prefix, cap target ids, and decode one text per row."""

    rows = conditional_target_token_rows(
        token_ids,
        cond_seq_mask,
        eos_token_id=tokenizer.eos_token_id,
        max_target_tokens=max_target_tokens,
    )
    return tokenizer.batch_decode(rows, skip_special_tokens=True)


def decode_unconditional_texts(tokenizer, token_ids) -> list[str]:
    """Decode OWT rows up to (but excluding) the first EOS token."""

    ids = np.asarray(token_ids)
    if ids.ndim != 2 or ids.shape[0] == 0:
        raise ValueError("token_ids must be nonempty [B, S]")
    rows = []
    for sample_ids in ids:
        row = sample_ids.tolist()
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id)]
        rows.append(row)
    return tokenizer.batch_decode(rows, skip_special_tokens=True)


def decoder_target_mask(token_ids, *, eos_token_id: int, pad_token_id: int) -> np.ndarray:
    """Mask visible OWT pseudo-target tokens through the first EOS, inclusive.

    Tokens after the first EOS and explicit padding tokens receive zero weight.
    If a row has no EOS, every non-padding position remains active.
    """

    ids = np.asarray(token_ids)
    if ids.ndim != 2 or ids.shape[0] == 0:
        raise ValueError("token_ids must be nonempty [B, S]")
    is_eos = ids == eos_token_id
    eos_seen_before = np.cumsum(is_eos, axis=1) - is_eos
    through_first_eos = eos_seen_before == 0
    return (through_first_eos & (ids != pad_token_id)).astype(np.float32)


def conditional_decoder_target_mask(
    token_ids,
    cond_seq_mask,
    *,
    eos_token_id: int,
    pad_token_id: int,
    max_target_tokens: int | None = None,
) -> np.ndarray:
    """Mask a generated conditional target through its first target-side EOS.

    Source-prefix positions are always excluded.  The first target-side EOS is
    retained even when ELF uses EOS itself as the padding token; later EOS/pad
    positions are excluded.  If ``max_target_tokens`` is set, positions beyond
    that target-relative cap are also excluded.
    """

    ids = np.asarray(token_ids)
    condition = np.asarray(cond_seq_mask)
    if ids.ndim != 2 or ids.shape[0] == 0 or condition.shape != ids.shape:
        raise ValueError(
            "token_ids and cond_seq_mask must share a nonempty [B, S] shape"
        )
    if max_target_tokens is not None and (
        isinstance(max_target_tokens, bool)
        or not isinstance(max_target_tokens, int)
        or max_target_tokens <= 0
    ):
        raise ValueError("max_target_tokens must be a positive integer or None")
    if not np.isfinite(condition).all() or np.any(
        (condition != 0.0) & (condition != 1.0)
    ):
        raise ValueError("cond_seq_mask must be finite and binary")
    for row in condition:
        source_length = int(row.astype(np.int32).sum())
        if np.any(row[:source_length] != 1) or np.any(row[source_length:] != 0):
            raise ValueError("each cond_seq_mask row must be a contiguous prefix")
    target_position = condition < 0.5
    if max_target_tokens is not None:
        target_ordinal = np.cumsum(target_position, axis=1)
        target_position &= target_ordinal <= max_target_tokens
    target_eos = (ids == eos_token_id) & target_position
    eos_seen_before = np.cumsum(target_eos, axis=1) - target_eos
    through_first_eos = eos_seen_before == 0
    non_padding_or_first_eos = (ids != pad_token_id) | target_eos
    return (
        target_position & through_first_eos & non_padding_or_first_eos
    ).astype(np.float32)


def sentence_bleu_rewards(hypotheses, references) -> np.ndarray:
    """Return lowercased effective-order sentence BLEU in ``[0, 100]``."""

    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("hypotheses and references must be nonempty and aligned")
    rewards = [
        sacrebleu.sentence_bleu(
            hypothesis,
            [reference],
            lowercase=True,
            use_effective_order=True,
        ).score
        for hypothesis, reference in zip(hypotheses, references, strict=True)
    ]
    result = np.asarray(rewards, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("sentence BLEU produced a non-finite reward")
    return result


def corpus_bleu(hypotheses, references) -> float:
    """Return the ELF evaluation metric for rollout diagnostics."""

    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("hypotheses and references must be nonempty and aligned")
    return float(
        sacrebleu.corpus_bleu(
            hypotheses,
            [references],
            lowercase=True,
            use_effective_order=True,
        ).score
    )


def causal_lm_loglik_rewards(per_sample_ppl) -> np.ndarray:
    """Convert positive per-sample perplexities to mean token log-likelihoods."""

    perplexities = np.asarray(per_sample_ppl, dtype=np.float64)
    if perplexities.ndim != 1 or perplexities.size == 0:
        raise ValueError("per_sample_ppl must be a nonempty vector")
    if not np.isfinite(perplexities).all() or np.any(perplexities <= 0):
        raise ValueError("per_sample_ppl must contain finite positive values")
    rewards = -np.log(perplexities)
    if not np.isfinite(rewards).all():
        raise ValueError("causal LM reward produced a non-finite value")
    return rewards.astype(np.float32)


def capped_unigram_entropy_rewards(per_sample_entropy, target: float) -> np.ndarray:
    """Cap sample unigram entropy once the configured diversity floor is met.

    Values below ``target`` retain their measured entropy. Values at or above
    ``target`` all receive exactly ``target``; equivalently, their entropy
    deficit reward is zero. This avoids rewarding arbitrarily high lexical
    entropy after the diversity constraint has been satisfied.
    """

    entropies = np.asarray(per_sample_entropy, dtype=np.float64)
    if entropies.ndim != 1 or entropies.size == 0:
        raise ValueError("per_sample_entropy must be a nonempty vector")
    if not np.isfinite(entropies).all() or np.any(entropies < 0):
        raise ValueError("per_sample_entropy must contain finite non-negative values")
    if not np.isfinite(target) or target <= 0:
        raise ValueError("entropy target must be finite and positive")
    return np.minimum(entropies, target).astype(np.float32)


def entropy_weight_to_lagrange(weight: float) -> float:
    """Convert an entropy mixture weight in ``[0, 1)`` to its multiplier."""

    if not np.isfinite(weight) or not 0.0 <= weight < 1.0:
        raise ValueError("entropy weight must lie in [0, 1)")
    return float(weight / (1.0 - weight))


def lagrange_to_entropy_weight(multiplier: float) -> float:
    """Convert a non-negative Lagrange multiplier to a convex mixture weight."""

    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("entropy Lagrange multiplier must be finite and non-negative")
    return float(multiplier / (1.0 + multiplier))


def make_entropy_constraint_state(initial_entropy_weight: float) -> dict[str, np.ndarray]:
    """Create host-side, checkpointable state for the entropy controller."""

    multiplier = entropy_weight_to_lagrange(initial_entropy_weight)
    return {
        "lagrange_multiplier": np.asarray(multiplier, dtype=np.float32),
        "entropy_ema": np.asarray(0.0, dtype=np.float32),
        "initialized": np.asarray(False, dtype=np.bool_),
    }


def entropy_weight_from_state(
    state: dict[str, np.ndarray], *, mode: str, fixed_weight: float
) -> float:
    """Return the entropy weight to apply to the current rollout."""

    if mode not in OWT_REWARD_MODES:
        raise ValueError(f"unsupported OWT reward combination mode: {mode!r}")
    if mode == PPL_ONLY:
        return 0.0
    if mode == FIXED_JOINT:
        entropy_weight_to_lagrange(fixed_weight)
        return float(fixed_weight)
    return lagrange_to_entropy_weight(float(state["lagrange_multiplier"]))


def update_entropy_constraint_state(
    state: dict[str, np.ndarray],
    observed_entropy: float,
    *,
    mode: str,
    entropy_target: float,
    ema_decay: float,
    dual_learning_rate: float,
    max_entropy_weight: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Update the entropy-floor dual controller after one rewarded rollout.

    The current state's multiplier is used for the rollout that produced
    ``observed_entropy``. This function updates the EMA and multiplier for the
    *next* rollout. Fixed-joint and PPL-only modes still update the EMA for
    diagnostics, but keep their configured mixture weights unchanged.
    """

    if mode not in OWT_REWARD_MODES:
        raise ValueError(f"unsupported OWT reward combination mode: {mode!r}")
    if not np.isfinite(observed_entropy) or observed_entropy < 0:
        raise ValueError("observed_entropy must be finite and non-negative")
    if not np.isfinite(entropy_target) or entropy_target < 0:
        raise ValueError("entropy_target must be finite and non-negative")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("entropy EMA decay must lie in [0, 1)")
    if not np.isfinite(dual_learning_rate) or dual_learning_rate < 0:
        raise ValueError("dual learning rate must be finite and non-negative")
    if not 0.0 <= max_entropy_weight < 1.0:
        raise ValueError("max entropy weight must lie in [0, 1)")

    initialized = bool(state["initialized"])
    previous_ema = float(state["entropy_ema"])
    entropy_ema = (
        float(observed_entropy)
        if not initialized
        else ema_decay * previous_ema + (1.0 - ema_decay) * float(observed_entropy)
    )
    multiplier = float(state["lagrange_multiplier"])
    if mode == ADAPTIVE_ENTROPY_FLOOR:
        max_multiplier = entropy_weight_to_lagrange(max_entropy_weight)
        multiplier = float(
            np.clip(
                multiplier + dual_learning_rate * (entropy_target - entropy_ema),
                0.0,
                max_multiplier,
            )
        )
    elif mode == PPL_ONLY:
        multiplier = 0.0

    updated = {
        "lagrange_multiplier": np.asarray(multiplier, dtype=np.float32),
        "entropy_ema": np.asarray(entropy_ema, dtype=np.float32),
        "initialized": np.asarray(True, dtype=np.bool_),
    }
    next_weight = (
        0.0
        if mode == PPL_ONLY
        else lagrange_to_entropy_weight(multiplier)
    )
    return updated, {
        "reward_entropy_ema": entropy_ema,
        "reward_entropy_constraint_violation": max(entropy_target - entropy_ema, 0.0),
        "reward_entropy_lagrange_multiplier_next": multiplier,
        "reward_entropy_weight_next": next_weight,
    }


def text_diversity_metrics(texts) -> dict[str, float]:
    """Return lightweight collapse diagnostics without changing the reward."""

    values = [str(text) for text in texts]
    if not values:
        raise ValueError("texts must be nonempty")
    nonempty = [text for text in values if text.strip()]
    empty_fraction = 1.0 - len(nonempty) / len(values)
    exact_duplicate_fraction = (
        1.0 - len(set(nonempty)) / len(nonempty) if nonempty else 1.0
    )
    repeated_fractions = []
    for text in nonempty:
        words = text.split()
        ngrams = [tuple(words[index : index + 4]) for index in range(len(words) - 3)]
        repeated_fractions.append(
            1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
        )
    return {
        "empty_text_fraction": float(empty_fraction),
        "exact_duplicate_fraction": float(exact_duplicate_fraction),
        "mean_repeated_4gram_fraction": float(
            np.mean(repeated_fractions) if repeated_fractions else 0.0
        ),
    }


class CausalLMLogLikelihoodReward:
    """Frozen causal-LM fluency reward used by the null-conditioned OWT branch.

    The evaluator returns a perplexity for every nonempty generated sample.
    This scorer uses ``-log(perplexity)``, i.e. mean token log-likelihood, so
    reward magnitude does not grow with sequence length. Empty decoded samples
    receive the configured finite penalty and never enter the LM evaluator.
    """

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        max_length: int,
        empty_text_reward: float,
        evaluator=None,
    ):
        if not model_name.strip():
            raise ValueError("model_name must be nonempty")
        if batch_size <= 0 or max_length < 2:
            raise ValueError("batch_size must be positive and max_length >= 2")
        if not np.isfinite(empty_text_reward) or empty_text_reward >= 0:
            raise ValueError("empty_text_reward must be finite and negative")
        if evaluator is None:
            from utils.metrics_utils import Metrics

            evaluator = Metrics(
                gen_ppl_eval_model_name_or_path=model_name,
                eval_ppl_batch_size=batch_size,
                eval_context_size=max_length,
            )
        self.evaluator = evaluator
        self.max_length = max_length
        self.empty_text_reward = float(empty_text_reward)

    def score(self, texts) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        values = [str(text) for text in texts]
        if not values:
            raise ValueError("texts must be nonempty")
        valid_indices = [index for index, text in enumerate(values) if text.strip()]
        rewards = np.full(len(values), self.empty_text_reward, dtype=np.float32)
        entropies = np.zeros(len(values), dtype=np.float32)
        scored_count = 0
        if valid_indices:
            valid_texts = [values[index] for index in valid_indices]
            self.evaluator.reset()
            result = self.evaluator.record_generative_perplexity(
                valid_texts,
                max_length=self.max_length,
                retokenize=True,
            )
            perplexities = np.asarray(result["per_sample_ppl"], dtype=np.float64)
            if perplexities.shape != (len(valid_indices),):
                raise ValueError("causal LM returned the wrong number of sample scores")
            valid_entropies = np.asarray(
                result["per_sample_entropy"], dtype=np.float64
            )
            if valid_entropies.shape != (len(valid_indices),):
                raise ValueError("causal LM returned the wrong number of entropy scores")
            if not np.isfinite(valid_entropies).all() or np.any(valid_entropies < 0):
                raise ValueError("causal LM returned an invalid sample entropy")
            scorable = np.isfinite(perplexities) & (perplexities > 0)
            valid_rewards = np.full(
                len(valid_indices), self.empty_text_reward, dtype=np.float32
            )
            if scorable.any():
                valid_rewards[scorable] = causal_lm_loglik_rewards(
                    perplexities[scorable]
                )
            scored_count = int(scorable.sum())
            valid_index_array = np.asarray(valid_indices)
            rewards[valid_index_array] = valid_rewards
            entropies[valid_index_array] = valid_entropies.astype(np.float32)
            raw_model_ppl = float(result["ppl"])
            raw_unigram_entropy = float(valid_entropies.mean())
            model_ppl = raw_model_ppl if np.isfinite(raw_model_ppl) else 0.0
            unigram_entropy = (
                raw_unigram_entropy if np.isfinite(raw_unigram_entropy) else 0.0
            )
            valid_reward_mean = (
                float(valid_rewards[scorable].mean()) if scored_count else 0.0
            )
        else:
            # Finite sentinels keep the online metrics record serializable; the
            # explicit valid fraction distinguishes this from an observed PPL.
            model_ppl = 0.0
            unigram_entropy = 0.0
            valid_reward_mean = 0.0
        if not np.isfinite(rewards).all():
            raise ValueError("OWT reward produced a non-finite value")
        if not np.isfinite(entropies).all():
            raise ValueError("OWT entropy produced a non-finite value")
        metrics = {
            "reward_model_ppl": model_ppl,
            "reward_model_unigram_entropy": unigram_entropy,
            "valid_text_reward_mean": valid_reward_mean,
            "valid_text_fraction": float(len(valid_indices) / len(values)),
            "scored_text_fraction": float(scored_count / len(values)),
            "reward_model_invalid_fraction": float(
                (len(valid_indices) - scored_count) / len(values)
            ),
            **text_diversity_metrics(values),
        }
        return rewards, entropies, metrics
