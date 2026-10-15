"""Pinned XSum data loading, token alignment, and capped dataset views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from utils.data_utils import get_dataloader, load_dataset_split

from .xsum_scoring import XSUM_HYPOTHESIS_MAX_TOKENS


XSUM_TRAIN_REPO = "embedded-language-flows/xsum_train_t5"
XSUM_TRAIN_REVISION = "1d4f2647d92d9acc62e05df2b5fcab6d226b744d"
XSUM_TRAIN_ROWS = 204_045
XSUM_VALIDATION_REPO = "embedded-language-flows/xsum_validation_t5"
XSUM_VALIDATION_REVISION = "b0b4f5de92434d308eceeb9e38fd3f8d7d2b6a9e"
XSUM_VALIDATION_ROWS = 11_332
XSUM_RAW_REVISION = "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9"
XSUM_CHECKPOINT_REVISION = "8150551968f826c4421cf6bdeff9d6e7b766e23d"
XSUM_ENCODER_REVISION = "c77f0a84a7549dd1ca4b3976840150d46e9da7dd"
XSUM_TOKENIZER_NAME = "google-t5/t5-small"
XSUM_TOKENIZER_REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
XSUM_SOURCE_MAX_TOKENS = 1024
XSUM_TOTAL_MAX_TOKENS = XSUM_SOURCE_MAX_TOKENS + XSUM_HYPOTHESIS_MAX_TOKENS
XSUM_REQUIRED_COLUMNS = frozenset(
    {"index", "input", "target", "condition_input_ids", "input_ids"}
)
XSUM_TRAIN_SELECTION_SEED_TAG = 0x5853554D
XSUM_CENTER_PRETRAIN_SELECTION_SEED_TAG = 0x58535052


@dataclass(frozen=True)
class XSumDataSettings:
    train_repo: str
    train_revision: str
    train_rows: int
    validation_repo: str
    validation_revision: str
    validation_rows: int
    raw_revision: str
    checkpoint_revision: str
    encoder_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    source_max_tokens: int = XSUM_SOURCE_MAX_TOKENS
    target_max_tokens: int = XSUM_HYPOTHESIS_MAX_TOKENS


PINNED_XSUM_SETTINGS = XSumDataSettings(
    train_repo=XSUM_TRAIN_REPO,
    train_revision=XSUM_TRAIN_REVISION,
    train_rows=XSUM_TRAIN_ROWS,
    validation_repo=XSUM_VALIDATION_REPO,
    validation_revision=XSUM_VALIDATION_REVISION,
    validation_rows=XSUM_VALIDATION_ROWS,
    raw_revision=XSUM_RAW_REVISION,
    checkpoint_revision=XSUM_CHECKPOINT_REVISION,
    encoder_revision=XSUM_ENCODER_REVISION,
    tokenizer_name=XSUM_TOKENIZER_NAME,
    tokenizer_revision=XSUM_TOKENIZER_REVISION,
)


def _required_config_value(config, name):
    value = getattr(config, name, None)
    if value is None or isinstance(value, str) and not value.strip():
        raise ValueError(f"XSum base config requires nonempty {name}")
    return value


def xsum_settings_from_config(config, nft_config=None) -> XSumDataSettings:
    """Read the base YAML and reject any drift from the sealed XSum assets."""

    settings = XSumDataSettings(
        train_repo=str(_required_config_value(config, "data_path")),
        train_revision=str(
            _required_config_value(config, "xsum_train_data_revision")
        ),
        train_rows=int(_required_config_value(config, "xsum_train_num_rows")),
        validation_repo=str(_required_config_value(config, "eval_data_path")),
        validation_revision=str(
            _required_config_value(config, "xsum_validation_data_revision")
        ),
        validation_rows=int(
            _required_config_value(config, "xsum_validation_num_rows")
        ),
        raw_revision=str(_required_config_value(config, "xsum_raw_data_revision")),
        checkpoint_revision=str(
            _required_config_value(
                config if nft_config is None else nft_config,
                "pretrained_checkpoint_revision",
            )
        ),
        encoder_revision=str(
            _required_config_value(config, "encoder_checkpoint_revision")
        ),
        tokenizer_name=str(_required_config_value(config, "tokenizer_name")),
        tokenizer_revision=str(
            _required_config_value(config, "tokenizer_revision")
        ),
        source_max_tokens=int(_required_config_value(config, "max_input_length")),
        target_max_tokens=int(
            _required_config_value(config, "xsum_target_max_tokens")
        ),
    )
    if settings != PINNED_XSUM_SETTINGS:
        changed = {
            name: {"configured": getattr(settings, name), "required": value}
            for name, value in vars(PINNED_XSUM_SETTINGS).items()
            if getattr(settings, name) != value
        }
        raise ValueError(f"XSum pinned data/model contract changed: {changed}")
    if int(getattr(config, "max_length", -1)) != XSUM_TOTAL_MAX_TOKENS:
        raise ValueError(
            f"XSum max_length must be {XSUM_TOTAL_MAX_TOKENS} "
            "(1024 source + 64 target)"
        )
    return settings


class CappedXSumDataset:
    """Read-only view that caps article and gold token rows independently.

    Independent caps are essential: simply truncating the concatenated sequence
    to 1088 would allow a short article to expose more than 64 gold-summary
    tokens.  Original article/reference strings are preserved for scoring.
    """

    def __init__(
        self,
        dataset,
        *,
        source_max_tokens: int = XSUM_SOURCE_MAX_TOKENS,
        target_max_tokens: int = XSUM_HYPOTHESIS_MAX_TOKENS,
    ):
        if source_max_tokens <= 0 or target_max_tokens <= 0:
            raise ValueError("XSum source and target caps must be positive")
        self.dataset = dataset
        self.source_max_tokens = int(source_max_tokens)
        self.target_max_tokens = int(target_max_tokens)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        if not isinstance(row, Mapping):
            raise ValueError(f"XSum row {index} is not a mapping")
        _validate_xsum_row_shape(row, position=int(index))
        result = dict(row)
        result["condition_input_ids"] = np.asarray(
            row["condition_input_ids"]
        )[: self.source_max_tokens]
        result["input_ids"] = np.asarray(row["input_ids"])[
            : self.target_max_tokens
        ]
        result["index"] = int(row["index"])
        result["input"] = str(row["input"])
        result["target"] = str(row["target"])
        return result


def xsum_prompt_indices(
    dataset_size: int,
    *,
    outer_step: int,
    seed: int,
    prompt_count: int,
    selection_seed_tag: int = XSUM_TRAIN_SELECTION_SEED_TAG,
) -> list[int]:
    """Select one resume-stable set of row positions for an XSum round."""

    if prompt_count <= 0 or prompt_count > dataset_size:
        raise ValueError("XSum prompt_count must be positive and <= dataset size")
    if outer_step < 0:
        raise ValueError("XSum outer_step must be non-negative")
    selection_rng = np.random.default_rng(
        np.random.SeedSequence([seed, outer_step, selection_seed_tag])
    )
    return selection_rng.choice(
        dataset_size, size=prompt_count, replace=False
    ).tolist()


def xsum_prompt_batch(
    dataset,
    *,
    outer_step: int,
    nft_config,
    elf_config,
    pad_token_id: int,
    selection_seed_tag: int = XSUM_TRAIN_SELECTION_SEED_TAG,
):
    """Collate exactly one independently capped, deterministic XSum prompt set."""

    indices = xsum_prompt_indices(
        len(dataset),
        outer_step=outer_step,
        seed=nft_config.seed,
        prompt_count=nft_config.prompt_groups_per_outer,
        selection_seed_tag=selection_seed_tag,
    )
    capped = dataset if isinstance(dataset, CappedXSumDataset) else CappedXSumDataset(dataset)
    loader = get_dataloader(
        Subset(capped, indices),
        batch_size=nft_config.prompt_groups_per_outer,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        max_seq_length=elf_config.max_length,
        max_input_seq_length=elf_config.max_input_length,
        pad_token_id=pad_token_id,
        distributed=False,
    )
    batch = next(iter(loader))
    if int(np.asarray(batch["input_ids"]).shape[0]) != len(indices):
        raise RuntimeError("XSum prompt collator changed the selected row count")
    return batch, indices


def _schedule_phase_proof(
    *,
    dataset_size: int,
    rounds: int,
    seed: int,
    prompt_count: int,
    selection_seed_tag: int,
) -> dict:
    digest = hashlib.sha256()
    selected = []
    for outer_step in range(rounds):
        indices = xsum_prompt_indices(
            dataset_size,
            outer_step=outer_step,
            seed=seed,
            prompt_count=prompt_count,
            selection_seed_tag=selection_seed_tag,
        )
        selected.extend(indices)
        digest.update(
            json.dumps(indices, separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\n")
    return {
        "rounds": int(rounds),
        "prompt_groups_per_round": int(prompt_count),
        "scheduled_row_count": len(selected),
        "unique_scheduled_row_count": len(set(selected)),
        "selection_seed_tag": int(selection_seed_tag),
        "ordered_indices_sha256": digest.hexdigest(),
    }


def xsum_schedule_proof(dataset_size: int, nft_config) -> dict:
    """Seal online and optional center-pretrain row-position schedules."""

    proof = {
        "online": _schedule_phase_proof(
            dataset_size=dataset_size,
            rounds=nft_config.max_outer_steps,
            seed=nft_config.seed,
            prompt_count=nft_config.prompt_groups_per_outer,
            selection_seed_tag=XSUM_TRAIN_SELECTION_SEED_TAG,
        )
    }
    if nft_config.center_pretrain_steps:
        proof["center_pretrain"] = _schedule_phase_proof(
            dataset_size=dataset_size,
            rounds=nft_config.center_pretrain_steps,
            seed=nft_config.seed,
            prompt_count=nft_config.prompt_groups_per_outer,
            selection_seed_tag=XSUM_CENTER_PRETRAIN_SELECTION_SEED_TAG,
        )
    return proof


def load_pinned_xsum_datasets(config, nft_config=None, *, dataset_cache_dir=None):
    """Load both tokenized splits by immutable Hugging Face commit SHA."""

    settings = xsum_settings_from_config(config, nft_config)
    cache_dir = dataset_cache_dir or os.environ.get("HF_DATASETS_CACHE")
    train = load_dataset_split(
        settings.train_repo,
        dataset_cache_dir=cache_dir,
        revision=settings.train_revision,
    )
    validation = load_dataset_split(
        settings.validation_repo,
        dataset_cache_dir=cache_dir,
        revision=settings.validation_revision,
    )
    return train, validation, settings


def _validate_xsum_row_shape(row, *, position: int):
    if not isinstance(row, Mapping):
        raise ValueError(f"XSum row {position} is not a mapping")
    missing = XSUM_REQUIRED_COLUMNS - set(row)
    if missing:
        raise ValueError(f"XSum row {position} is missing columns {sorted(missing)}")
    if isinstance(row["index"], bool) or not isinstance(
        row["index"], (int, np.integer)
    ):
        raise ValueError(f"XSum row {position} has a non-integer index")
    if not isinstance(row["input"], str) or not isinstance(row["target"], str):
        raise ValueError(f"XSum row {position} requires string input/target")
    for name in ("condition_input_ids", "input_ids"):
        values = np.asarray(row[name])
        if values.ndim != 1 or values.size == 0 or not np.issubdtype(
            values.dtype, np.integer
        ):
            raise ValueError(
                f"XSum row {position} {name} must be a nonempty integer vector"
            )


def _rows_from_slice(dataset, start: int, stop: int) -> list[Mapping]:
    sliced = dataset[start:stop]
    if isinstance(sliced, Mapping):
        lengths = {len(value) for value in sliced.values()}
        if lengths != {stop - start}:
            raise ValueError(f"XSum dataset slice {start}:{stop} is misaligned")
        return [
            {name: values[offset] for name, values in sliced.items()}
            for offset in range(stop - start)
        ]
    if isinstance(sliced, Sequence):
        return list(sliced)
    return [dataset[index] for index in range(start, stop)]


def _token_rows(tokenizer, texts):
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    rows = encoded.get("input_ids")
    if rows is None or len(rows) != len(texts):
        raise ValueError("XSum tokenizer returned a misaligned batch")
    return [[int(token) for token in row] for row in rows]


def validate_xsum_split(
    dataset,
    *,
    split: str,
    tokenizer,
    expected_rows: int,
    repo_id: str,
    revision: str,
    batch_size: int = 256,
) -> dict:
    """Validate every row's schema, unique index, and text/token alignment."""

    if split not in {"train", "validation"}:
        raise ValueError("XSum split must be train or validation")
    if len(dataset) != expected_rows:
        raise ValueError(
            f"XSum {split} row count differs: {len(dataset)} != {expected_rows}"
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("preflight batch_size must be a positive integer")
    columns = set(getattr(dataset, "column_names", ()))
    if columns and not XSUM_REQUIRED_COLUMNS.issubset(columns):
        raise ValueError(
            f"XSum {split} schema lacks {sorted(XSUM_REQUIRED_COLUMNS - columns)}"
        )

    seen_indices = set()
    digest = hashlib.sha256()
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        rows = _rows_from_slice(dataset, start, stop)
        if len(rows) != stop - start:
            raise ValueError(f"XSum {split} slice {start}:{stop} changed length")
        for offset, row in enumerate(rows):
            _validate_xsum_row_shape(row, position=start + offset)
        encoded_inputs = _token_rows(tokenizer, [row["input"] for row in rows])
        encoded_targets = _token_rows(tokenizer, [row["target"] for row in rows])
        for offset, (row, source_ids, target_ids) in enumerate(
            zip(rows, encoded_inputs, encoded_targets, strict=True)
        ):
            position = start + offset
            index = int(row["index"])
            if index in seen_indices:
                raise ValueError(f"XSum {split} duplicate index {index}")
            seen_indices.add(index)
            stored_source = [int(token) for token in row["condition_input_ids"]]
            stored_target = [int(token) for token in row["input_ids"]]
            if stored_source != source_ids:
                raise ValueError(
                    f"XSum {split} source token alignment failed at row {position}"
                )
            if stored_target != target_ids:
                raise ValueError(
                    f"XSum {split} target token alignment failed at row {position}"
                )
            digest.update(
                json.dumps(
                    [index, stored_source, stored_target],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "row_count": len(dataset),
        "required_columns": sorted(XSUM_REQUIRED_COLUMNS),
        "unique_index_count": len(seen_indices),
        "aligned_row_count": len(dataset),
        "token_digest_sha256": digest.hexdigest(),
    }


def validate_xsum_collator(
    dataset,
    *,
    elf_config,
    pad_token_id: int,
    batch_size: int = 2,
) -> dict:
    """Run a real capped collator batch and verify the 1024/64/1088 geometry."""

    if elf_config.max_input_length != XSUM_SOURCE_MAX_TOKENS:
        raise ValueError("XSum collator requires max_input_length=1024")
    if elf_config.max_length != XSUM_TOTAL_MAX_TOKENS:
        raise ValueError("XSum collator requires max_length=1088")
    batch_size = min(int(batch_size), len(dataset))
    if batch_size <= 0:
        raise ValueError("XSum collator requires a nonempty dataset")
    capped = CappedXSumDataset(dataset)
    loader = get_dataloader(
        capped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        max_seq_length=elf_config.max_length,
        max_input_seq_length=elf_config.max_input_length,
        pad_token_id=pad_token_id,
        distributed=False,
    )
    batch = next(iter(loader))
    ids = np.asarray(batch["input_ids"])
    source_mask = np.asarray(batch["cond_seq_mask"])
    valid_mask = np.asarray(batch["attention_mask"])
    expected_shape = (batch_size, XSUM_TOTAL_MAX_TOKENS)
    if ids.shape != expected_shape or source_mask.shape != expected_shape:
        raise ValueError(
            f"XSum collator shape differs: ids={ids.shape}, mask={source_mask.shape}"
        )
    source_lengths = source_mask.sum(axis=1).astype(np.int64)
    target_lengths = (valid_mask - source_mask).sum(axis=1).astype(np.int64)
    if np.any(source_lengths > XSUM_SOURCE_MAX_TOKENS):
        raise ValueError("XSum collator exceeded the 1024-token source cap")
    if np.any(target_lengths > XSUM_HYPOTHESIS_MAX_TOKENS):
        raise ValueError("XSum collator exceeded the 64-token target cap")
    return {
        "batch_size": batch_size,
        "shape": list(expected_shape),
        "source_max_tokens": XSUM_SOURCE_MAX_TOKENS,
        "target_max_tokens": XSUM_HYPOTHESIS_MAX_TOKENS,
        "observed_source_lengths": source_lengths.tolist(),
        "observed_target_lengths": target_lengths.tolist(),
    }


def preflight_xsum_datasets(
    train_dataset,
    validation_dataset,
    *,
    tokenizer,
    elf_config,
    nft_config,
    pad_token_id: int,
    settings: XSumDataSettings = PINNED_XSUM_SETTINGS,
    expected_rows: Mapping[str, int] | None = None,
) -> dict:
    """Return a serializable proof for both immutable splits and one collator."""

    counts = {
        "train": settings.train_rows,
        "validation": settings.validation_rows,
    }
    if expected_rows is not None:
        counts.update(expected_rows)
    train = validate_xsum_split(
        train_dataset,
        split="train",
        tokenizer=tokenizer,
        expected_rows=counts["train"],
        repo_id=settings.train_repo,
        revision=settings.train_revision,
    )
    validation = validate_xsum_split(
        validation_dataset,
        split="validation",
        tokenizer=tokenizer,
        expected_rows=counts["validation"],
        repo_id=settings.validation_repo,
        revision=settings.validation_revision,
    )
    collator = validate_xsum_collator(
        train_dataset,
        elf_config=elf_config,
        pad_token_id=pad_token_id,
    )
    schedule = xsum_schedule_proof(len(train_dataset), nft_config)
    return {
        "schema_version": 1,
        "task": "xsum",
        "raw_dataset_revision": settings.raw_revision,
        "checkpoint_revision": settings.checkpoint_revision,
        "encoder_revision": settings.encoder_revision,
        "tokenizer": {
            "name": settings.tokenizer_name,
            "revision": settings.tokenizer_revision,
        },
        "train": train,
        "validation": validation,
        "collator": collator,
        "schedule": schedule,
    }


def save_xsum_preflight(output_dir: str | Path, report: Mapping) -> Path:
    """Atomically publish one immutable XSum data-contract report."""

    path = Path(output_dir) / "xsum_data_preflight.json"
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(
                "XSum data preflight report changed inside the output directory"
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return path
