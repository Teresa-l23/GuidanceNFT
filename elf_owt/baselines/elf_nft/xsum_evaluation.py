"""Bounded, deterministic XSum evaluation under the shared reward contract."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import jax
import numpy as np
from torch.utils.data import Subset

from utils.data_utils import get_dataloader, get_pad_token_id

from .rollout import _generate_and_decode, _reshape_for_devices
from .xsum_data import CappedXSumDataset, XSUM_TOTAL_MAX_TOKENS
from .xsum_scoring import (
    XSUM_HYPOTHESIS_MAX_TOKENS,
    decode_xsum_hypotheses,
    xsum_hypothesis_token_rows,
    xsum_rouge_scores,
)


@dataclass(frozen=True)
class XSumEvaluationResult:
    metrics: dict[str, float]
    hypotheses: list[str]
    references: list[str]
    examples: list[dict]


def validate_xsum_evaluation_examples(
    examples,
    *,
    expected_count: int,
    aggregate_metrics: Mapping,
    max_tokens: int = XSUM_HYPOTHESIS_MAX_TOKENS,
) -> list[int]:
    """Validate the persisted per-example proof and return its index sequence."""

    if (
        isinstance(examples, (str, bytes))
        or not isinstance(examples, Sequence)
        or len(examples) != expected_count
    ):
        raise ValueError(
            f"XSum evaluation examples must contain exactly {expected_count} rows"
        )
    required = {
        "index",
        "hypothesis",
        "reference",
        "rouge1",
        "rouge2",
        "rougeL",
        "reward",
        "hypothesis_token_count",
    }
    indices = []
    scores = {name: [] for name in ("rouge1", "rouge2", "rougeL")}
    for position, example in enumerate(examples):
        if not isinstance(example, Mapping) or not required.issubset(example):
            raise ValueError(
                f"XSum evaluation example {position} lacks required fields"
            )
        index = example["index"]
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError(f"XSum evaluation example {position} index is invalid")
        indices.append(int(index))
        if not isinstance(example["hypothesis"], str) or not isinstance(
            example["reference"], str
        ):
            raise ValueError("XSum hypotheses/references must be strings")
        numeric = np.asarray(
            [
                example["rouge1"],
                example["rouge2"],
                example["rougeL"],
                example["reward"],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("XSum per-example ROUGE/reward must be finite")
        if float(example["reward"]) != float(example["rougeL"]):
            raise ValueError("XSum per-example reward must equal rougeL exactly")
        token_count = example["hypothesis_token_count"]
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, (int, np.integer))
            or not 0 <= int(token_count) <= max_tokens
        ):
            raise ValueError("XSum hypothesis token count violates its cap")
        for name in scores:
            scores[name].append(float(example[name]))
    if len(set(indices)) != expected_count:
        raise ValueError("XSum evaluation indices must be unique")
    for name, values in scores.items():
        observed = float(np.asarray(values, dtype=np.float64).mean())
        expected = float(aggregate_metrics[name])
        if not np.isclose(observed, expected, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"XSum aggregate {name} does not match per-example macro mean"
            )
    if not np.isclose(
        float(aggregate_metrics["reward_mean"]),
        float(aggregate_metrics["rougeL"]),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("XSum aggregate reward_mean must equal rougeL")
    return indices


def _batch_strings(batch, name: str, expected_count: int) -> list[str]:
    values = batch.get(name)
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f"XSum evaluation batch requires per-example {name}")
    result = [str(value) for value in values]
    if len(result) != expected_count:
        raise ValueError(
            f"XSum evaluation batch has {len(result)} {name} values, "
            f"expected {expected_count}"
        )
    return result


def evaluate_xsum_policy(
    *,
    dataset,
    functions,
    policy_params_replicated,
    encoder_params_replicated,
    tokenizer,
    rng,
    elf_config,
    nft_config,
) -> XSumEvaluationResult:
    """Generate one summary for each of the first fixed validation examples.

    ``policy_params_replicated`` is intentionally snapshot-agnostic.  The
    training driver supplies theta EMA for formal step-0/50/100 evaluations.
    This function never mutates policy, reward-controller, or training RNG.
    """

    count = int(nft_config.evaluation_num_samples)
    batch_size = int(nft_config.rollout_generation_batch_size)
    max_tokens = int(
        getattr(
            nft_config,
            "hypothesis_max_tokens",
            XSUM_HYPOTHESIS_MAX_TOKENS,
        )
    )
    devices = jax.local_device_count()
    if elf_config.max_length != XSUM_TOTAL_MAX_TOKENS:
        raise ValueError("XSum evaluation requires max_length=1088")
    if max_tokens != XSUM_HYPOTHESIS_MAX_TOKENS:
        raise ValueError("formal XSum evaluation requires a 64-token hypothesis cap")
    if count <= 0 or count > len(dataset):
        raise ValueError(
            "XSum evaluation count must be positive and <= validation dataset size"
        )
    if batch_size <= 0 or count % batch_size or batch_size % devices:
        raise ValueError(
            "XSum evaluation count/batch must divide over batches/devices"
        )
    if functions.encode is None or encoder_params_replicated is None:
        raise ValueError("XSum evaluation requires the frozen source encoder")

    capped = dataset if isinstance(dataset, CappedXSumDataset) else CappedXSumDataset(dataset)
    loader = get_dataloader(
        Subset(capped, range(count)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        max_seq_length=elf_config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, elf_config.pad_token),
        max_input_seq_length=elf_config.max_input_length,
        distributed=False,
    )
    hypotheses: list[str] = []
    references: list[str] = []
    example_indices: list[int] = []
    hypothesis_token_counts: list[int] = []
    for batch_index, batch in enumerate(loader):
        encoded = functions.encode(
            input_ids=_reshape_for_devices(
                np.asarray(batch["input_ids"]), devices
            ),
            attention_mask=_reshape_for_devices(
                np.asarray(batch["encoder_attention_mask"]), devices
            ),
            encoder_params=encoder_params_replicated,
        )
        encoded = encoded.reshape(
            batch_size, elf_config.max_length, encoded.shape[-1]
        )
        condition_mask = np.asarray(batch["cond_seq_mask"])
        endpoints, token_ids = _generate_and_decode(
            functions=functions,
            old_params_replicated=policy_params_replicated,
            cond_seq=encoded,
            cond_seq_mask=condition_mask,
            embedding_dim=encoded.shape[-1],
            batch_size=batch_size,
            rng=jax.random.fold_in(rng, batch_index),
            elf_config=elf_config,
            nft_config=nft_config,
        )
        decoded = decode_xsum_hypotheses(
            tokenizer,
            token_ids,
            condition_mask,
            max_tokens=max_tokens,
        )
        token_rows = xsum_hypothesis_token_rows(
            token_ids,
            condition_mask,
            eos_token_id=tokenizer.eos_token_id,
            max_tokens=max_tokens,
        )
        hypotheses.extend(decoded)
        references.extend(_batch_strings(batch, "target", batch_size))
        if "index" not in batch:
            raise ValueError("XSum evaluation batch requires stable row indices")
        example_indices.extend(int(index) for index in batch["index"])
        hypothesis_token_counts.extend(len(row) for row in token_rows)
        del endpoints, encoded, token_ids

    if not (
        len(hypotheses)
        == len(references)
        == len(example_indices)
        == len(hypothesis_token_counts)
        == count
    ):
        raise RuntimeError(
            "XSum evaluation did not produce the requested aligned sample count"
        )
    scores = xsum_rouge_scores(hypotheses, references)
    means = scores.means()
    examples = [
        {
            "index": example_indices[index],
            "hypothesis": hypotheses[index],
            "reference": references[index],
            "rouge1": float(scores.rouge1[index]),
            "rouge2": float(scores.rouge2[index]),
            "rougeL": float(scores.rougeL[index]),
            "reward": float(scores.rougeL[index]),
            "hypothesis_token_count": hypothesis_token_counts[index],
        }
        for index in range(count)
    ]
    metrics = {
        **means,
        # The reported primary evaluation number and online reward are exactly
        # the same per-example ROUGE-L F1 quantity and scale.
        "reward_mean": means["rougeL"],
        "hypothesis_token_count_mean": float(
            np.asarray(hypothesis_token_counts, dtype=np.float64).mean()
        ),
        "empty_hypothesis_fraction": float(
            np.asarray([not text.strip() for text in hypotheses]).mean()
        ),
    }
    if any(not np.isfinite(value) for value in metrics.values()):
        raise ValueError("XSum evaluation produced a non-finite metric")
    validate_xsum_evaluation_examples(
        examples,
        expected_count=count,
        aggregate_metrics=metrics,
        max_tokens=max_tokens,
    )
    return XSumEvaluationResult(
        metrics=metrics,
        hypotheses=hypotheses,
        references=references,
        examples=examples,
    )
