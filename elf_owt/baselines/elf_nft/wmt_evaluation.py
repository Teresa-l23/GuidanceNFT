"""Bounded, reference-aligned WMT evaluation with one translation per source."""

from dataclasses import dataclass

import jax
import numpy as np
from torch.utils.data import Subset

from utils.data_utils import get_dataloader, get_pad_token_id
from utils.metrics_utils import compute_bleu, compute_rouge

from .reward import (
    batch_reference_texts,
    decode_target_texts,
    sentence_bleu_rewards,
)
from .rollout import _generate_and_decode, _reshape_for_devices


@dataclass(frozen=True)
class WMTEvaluationResult:
    metrics: dict[str, float]
    hypotheses: list[str]
    references: list[str]


def evaluate_wmt_policy(
    *, dataset, functions, policy_params_replicated, encoder_params_replicated,
    tokenizer, rng, elf_config, nft_config,
) -> WMTEvaluationResult:
    """Evaluate the first N validation sources with fixed batch-indexed RNGs.

    Generates exactly one hypothesis per source, retaining text only after each
    batch. No reference embedding is installed into a target sampling position.
    The function neither updates a reward controller nor consumes training RNG.
    """

    count = nft_config.evaluation_num_samples
    batch_size = nft_config.rollout_generation_batch_size
    devices = jax.local_device_count()
    if count <= 0 or count > len(dataset):
        raise ValueError("WMT evaluation count must be positive and <= validation dataset size")
    if batch_size <= 0 or count % batch_size or batch_size % devices:
        raise ValueError("WMT evaluation count/batch must divide over batches/devices")
    if functions.encode is None or encoder_params_replicated is None:
        raise ValueError("WMT evaluation requires the frozen source encoder")
    loader = get_dataloader(
        Subset(dataset, range(count)), batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, max_seq_length=elf_config.max_length,
        pad_token_id=get_pad_token_id(tokenizer, elf_config.pad_token),
        max_input_seq_length=elf_config.max_input_length, distributed=False,
    )
    hypotheses, references = [], []
    for index, batch in enumerate(loader):
        encoded = functions.encode(
            input_ids=_reshape_for_devices(np.asarray(batch["input_ids"]), devices),
            attention_mask=_reshape_for_devices(np.asarray(batch["encoder_attention_mask"]), devices),
            encoder_params=encoder_params_replicated,
        )
        encoded = encoded.reshape(batch_size, elf_config.max_length, encoded.shape[-1])
        mask = np.asarray(batch["cond_seq_mask"])
        endpoints, token_ids = _generate_and_decode(
            functions=functions, old_params_replicated=policy_params_replicated,
            cond_seq=encoded, cond_seq_mask=mask, embedding_dim=encoded.shape[-1],
            batch_size=batch_size, rng=jax.random.fold_in(rng, index),
            elf_config=elf_config, nft_config=nft_config,
        )
        hypotheses.extend(
            decode_target_texts(
                tokenizer,
                token_ids,
                mask,
                max_target_tokens=(
                    getattr(nft_config, "hypothesis_max_tokens", None)
                    or None
                ),
            )
        )
        references.extend(
            batch_reference_texts(batch, expected_count=batch_size)
        )
        del endpoints, encoded, token_ids
    if len(hypotheses) != count or len(references) != count:
        raise RuntimeError("WMT evaluation did not produce the requested aligned sample count")
    bleu = float(compute_bleu(hypotheses, references))
    rouge = {
        name: float(value)
        for name, value in compute_rouge(hypotheses, references).items()
    }
    mean_sentence_bleu = float(
        sentence_bleu_rewards(hypotheses, references).mean()
    )
    return WMTEvaluationResult(
        metrics={
            "reward_mean": mean_sentence_bleu,
            "mean_sentence_bleu": mean_sentence_bleu,
            "corpus_bleu": bleu,
            "bleu": bleu,
            **rouge,
        },
        hypotheses=hypotheses,
        references=references,
    )
