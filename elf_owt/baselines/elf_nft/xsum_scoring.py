"""Shared XSum hypothesis truncation and ROUGE scoring contracts.

Reward collection and held-out evaluation both call this module.  Keeping the
token boundary and scorer in one place prevents an optimization metric from
silently differing from the reported evaluation metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reward import (
    conditional_decoder_target_mask,
    conditional_target_token_rows,
    decode_target_texts,
)


XSUM_HYPOTHESIS_MAX_TOKENS = 64
ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")


@dataclass(frozen=True)
class XSumScores:
    """Per-example ROUGE F1 scores, expressed on the ``[0, 100]`` scale."""

    rouge1: np.ndarray
    rouge2: np.ndarray
    rougeL: np.ndarray

    def means(self) -> dict[str, float]:
        return {
            name: float(np.asarray(getattr(self, name), dtype=np.float64).mean())
            for name in ROUGE_KEYS
        }


def xsum_hypothesis_token_rows(
    token_ids,
    cond_seq_mask,
    *,
    eos_token_id: int,
    max_tokens: int = XSUM_HYPOTHESIS_MAX_TOKENS,
) -> list[list[int]]:
    """Return at most 64 target tokens per row, excluding the first EOS.

    The target starts immediately after each row's *actual* article prefix,
    rather than at the global 1024-token article cap.  Truncation is applied
    before decoding and therefore has identical semantics for reward and eval.
    """

    return conditional_target_token_rows(
        token_ids,
        cond_seq_mask,
        eos_token_id,
        max_target_tokens=max_tokens,
    )


def decode_xsum_hypotheses(
    tokenizer,
    token_ids,
    cond_seq_mask,
    *,
    max_tokens: int = XSUM_HYPOTHESIS_MAX_TOKENS,
) -> list[str]:
    """Decode hypotheses under the shared 64-token/first-EOS contract."""

    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("XSum tokenizer must define eos_token_id")
    return decode_target_texts(
        tokenizer,
        token_ids,
        cond_seq_mask,
        max_target_tokens=max_tokens,
    )


def xsum_decoder_target_mask(
    token_ids,
    cond_seq_mask,
    *,
    eos_token_id: int,
    pad_token_id: int,
    max_tokens: int = XSUM_HYPOTHESIS_MAX_TOKENS,
) -> np.ndarray:
    """Mask only the first 64 generated target positions through first EOS.

    The first target-side EOS is retained in CE/flow losses, even when EOS is
    also the configured padding token.  Source positions, positions after the
    64-token cap, later EOS tokens, and explicit non-EOS padding receive zero.
    """

    return conditional_decoder_target_mask(
        token_ids,
        cond_seq_mask,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_target_tokens=max_tokens,
    )


def xsum_rouge_scores(hypotheses, references) -> XSumScores:
    """Score aligned text pairs with the canonical ELF ROUGE implementation.

    This deliberately uses ``rougeL`` rather than ``rougeLsum`` and invokes
    ``score(reference, hypothesis)`` with stemming, matching upstream ELF.
    """

    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("hypotheses and references must be nonempty and aligned")
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=True)
    values = {name: [] for name in ROUGE_KEYS}
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        scores = scorer.score(str(reference), str(hypothesis))
        for name in ROUGE_KEYS:
            values[name].append(float(scores[name].fmeasure) * 100.0)
    arrays = {
        name: np.asarray(value, dtype=np.float32) for name, value in values.items()
    }
    if any(
        array.shape != (len(hypotheses),) or not np.isfinite(array).all()
        for array in arrays.values()
    ):
        raise ValueError("ROUGE produced a non-finite or misaligned score")
    return XSumScores(**arrays)


def xsum_rouge_l_rewards(hypotheses, references) -> np.ndarray:
    """Return the exact per-example ROUGE-L metric optimized by XSum NFT."""

    return xsum_rouge_scores(hypotheses, references).rougeL.copy()
