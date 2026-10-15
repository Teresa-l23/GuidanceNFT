"""Persist deterministic inline-evaluation previews without trainer dependencies.

The distributed trainer is responsible for collecting the desired records across
all evaluation batches/ranks and converting tensors to :class:`PIL.Image.Image`.
This module deliberately performs no sampling, truncation, or distributed I/O.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import numbers
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


__all__ = ["InlinePreviewRecord", "write_inline_eval_preview"]


_SCHEMA_VERSION = 1
_CONTACT_SHEET_NAME = "contact_sheet.jpg"
_MANIFEST_NAME = "manifest.json"
_COMPLETE_NAME = ".complete"
_LOCK_NAME = ".write.lock"
_HEADER_HEIGHT = 76
_PROMPT_HEIGHT = 48
_SCORE_HEIGHT = 34
_BACKGROUND = (244, 244, 244)
_HEADER_BACKGROUND = (29, 33, 41)
_PROMPT_BACKGROUND = (55, 61, 73)


@dataclass(frozen=True)
class InlinePreviewRecord:
    """One already-collected inline-evaluation preview.

    ``image`` must be a PIL image. ``rewards`` and ``metadata`` are copied and
    normalized before any filesystem mutation, so later caller changes cannot
    alter the persisted result.
    """

    sample_id: int
    prompt: str
    sample_seed: int
    image: Image.Image
    rewards: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _NormalizedRecord:
    sample_id: int
    prompt: str
    sample_seed: int
    image: Image.Image
    rewards: dict[str, float]
    metadata: dict[str, Any]


def _require_integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _normalize_json(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{path} must contain only finite numbers")
        return result
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        invalid_keys = [key for key in value if not isinstance(key, str)]
        if invalid_keys:
            raise TypeError(f"{path} keys must be strings, got {invalid_keys[0]!r}")
        for key in sorted(value):
            normalized[key] = _normalize_json(value[key], f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} is not JSON-compatible: {type(value).__name__}")


def _normalize_rewards(rewards: Mapping[str, float], path: str) -> dict[str, float]:
    if not isinstance(rewards, Mapping) or not rewards:
        raise ValueError(f"{path} must be a nonempty mapping")
    normalized: dict[str, float] = {}
    if any(not isinstance(name, str) or not name for name in rewards):
        raise TypeError(f"{path} keys must be nonempty strings")
    for name in sorted(rewards):
        value = rewards[name]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{path}.{name} must be numeric, got {value!r}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{path}.{name} must be finite")
        normalized[name] = numeric
    return normalized


def _normalize_record(record: InlinePreviewRecord, path: str) -> _NormalizedRecord:
    if not isinstance(record, InlinePreviewRecord):
        raise TypeError(f"{path} must be InlinePreviewRecord, got {type(record).__name__}")
    sample_id = _require_integer(record.sample_id, f"{path}.sample_id", minimum=0)
    sample_seed = _require_integer(
        record.sample_seed, f"{path}.sample_seed", minimum=0
    )
    if not isinstance(record.prompt, str) or not record.prompt:
        raise ValueError(f"{path}.prompt must be a nonempty string")
    if not isinstance(record.image, Image.Image):
        raise TypeError(f"{path}.image must be a PIL Image")
    metadata = _normalize_json(record.metadata, f"{path}.metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"{path}.metadata must be a mapping")
    return _NormalizedRecord(
        sample_id=sample_id,
        prompt=record.prompt,
        sample_seed=sample_seed,
        image=record.image.copy().convert("RGB"),
        rewards=_normalize_rewards(record.rewards, f"{path}.rewards"),
        metadata=metadata,
    )


def _normalize_guidance_scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"guidance scale must be numeric, got {value!r}")
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"guidance scale must be finite and positive, got {value!r}")
    return scale


def _normalize_previews(
    previews_by_guidance: Mapping[float, Sequence[InlinePreviewRecord]],
) -> tuple[list[float], dict[float, list[_NormalizedRecord]]]:
    if not isinstance(previews_by_guidance, Mapping) or not previews_by_guidance:
        raise ValueError("previews_by_guidance must be a nonempty mapping")

    normalized: dict[float, list[_NormalizedRecord]] = {}
    for raw_scale, records in previews_by_guidance.items():
        scale = _normalize_guidance_scale(raw_scale)
        if scale in normalized:
            raise ValueError(f"duplicate normalized guidance scale: {scale}")
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError(f"previews_by_guidance[{scale}] must be a sequence")
        if not records:
            raise ValueError(f"previews_by_guidance[{scale}] must not be empty")
        by_sample_id: dict[int, _NormalizedRecord] = {}
        for index, record in enumerate(records):
            normalized_record = _normalize_record(
                record, f"previews_by_guidance[{scale}][{index}]"
            )
            if normalized_record.sample_id in by_sample_id:
                raise ValueError(
                    f"duplicate sample_id={normalized_record.sample_id} at CFG={scale}"
                )
            by_sample_id[normalized_record.sample_id] = normalized_record
        normalized[scale] = [by_sample_id[key] for key in sorted(by_sample_id)]

    scales = sorted(normalized)
    reference = normalized[scales[0]]
    reference_ids = [record.sample_id for record in reference]
    reference_by_id = {record.sample_id: record for record in reference}
    for scale in scales[1:]:
        records = normalized[scale]
        sample_ids = [record.sample_id for record in records]
        if sample_ids != reference_ids:
            raise ValueError(
                "all guidance scales must contain exactly the same sample IDs; "
                f"CFG={scale} has {sample_ids}, expected {reference_ids}"
            )
        for record in records:
            paired = reference_by_id[record.sample_id]
            for name in ("prompt", "sample_seed", "metadata"):
                if getattr(record, name) != getattr(paired, name):
                    raise ValueError(
                        "paired CFG previews disagree for "
                        f"sample_id={record.sample_id}, field={name}"
                    )
    return scales, normalized


def _guidance_tag(scale: float) -> str:
    if scale.is_integer():
        return f"cfg{int(scale)}"
    return f"cfg{str(scale).replace('.', 'p')}"


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for root in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
    ):
        candidate = root / filename
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _ellipsize(
    draw: ImageDraw.ImageDraw,
    value: str,
    font: ImageFont.ImageFont,
    width: int,
) -> str:
    if draw.textbbox((0, 0), value, font=font)[2] <= width:
        return value
    suffix = "..."
    candidate = value
    while candidate and draw.textbbox(
        (0, 0), candidate + suffix, font=font
    )[2] > width:
        candidate = candidate[:-1]
    return candidate + suffix


def _reward_label(rewards: Mapping[str, float]) -> str:
    return " | ".join(f"{name}={value:.5g}" for name, value in rewards.items())


def _render_contact_sheet(
    *,
    step: int,
    eval_steps: int,
    scales: Sequence[float],
    previews: Mapping[float, Sequence[_NormalizedRecord]],
    cell_size: int,
) -> bytes:
    sample_count = len(previews[scales[0]])
    row_height = _PROMPT_HEIGHT + cell_size + _SCORE_HEIGHT
    width = cell_size * len(scales)
    height = _HEADER_HEIGHT + row_height * sample_count
    sheet = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    header_font = _load_font(21, bold=True)
    detail_font = _load_font(15)
    prompt_font = _load_font(16, bold=True)
    reward_font = _load_font(14)

    draw.rectangle((0, 0, width, _HEADER_HEIGHT), fill=_HEADER_BACKGROUND)
    for column, scale in enumerate(scales):
        left = column * cell_size
        title = f"{_guidance_tag(scale).upper()} | step {step} | eval steps {eval_steps}"
        title = _ellipsize(draw, title, header_font, cell_size - 16)
        box = draw.textbbox((0, 0), title, font=header_font)
        draw.text(
            (left + (cell_size - (box[2] - box[0])) / 2, 13),
            title,
            fill="white",
            font=header_font,
        )
        detail = f"{sample_count} persisted previews"
        box = draw.textbbox((0, 0), detail, font=detail_font)
        draw.text(
            (left + (cell_size - (box[2] - box[0])) / 2, 46),
            detail,
            fill=(218, 225, 238),
            font=detail_font,
        )

    reference = previews[scales[0]]
    for row, reference_record in enumerate(reference):
        row_top = _HEADER_HEIGHT + row * row_height
        prompt_label = (
            f"sample {reference_record.sample_id} | seed {reference_record.sample_seed} | "
            f"{reference_record.prompt}"
        )
        draw.rectangle(
            (0, row_top, width, row_top + _PROMPT_HEIGHT), fill=_PROMPT_BACKGROUND
        )
        draw.text(
            (10, row_top + 14),
            _ellipsize(draw, prompt_label, prompt_font, width - 20),
            fill="white",
            font=prompt_font,
        )
        image_top = row_top + _PROMPT_HEIGHT
        score_top = image_top + cell_size
        for column, scale in enumerate(scales):
            record = previews[scale][row]
            left = column * cell_size
            tile = ImageOps.fit(record.image, (cell_size, cell_size))
            sheet.paste(tile, (left, image_top))
            label = _ellipsize(
                draw, _reward_label(record.rewards), reward_font, cell_size - 16
            )
            box = draw.textbbox((0, 0), label, font=reward_font)
            draw.text(
                (left + (cell_size - (box[2] - box[0])) / 2, score_top + 8),
                label,
                fill=(20, 20, 20),
                font=reward_font,
            )

    payload = io.BytesIO()
    sheet.save(payload, format="JPEG", quality=92, subsampling=0)
    return payload.getvalue()


def _build_manifest(
    *,
    step: int,
    eval_steps: int,
    scales: Sequence[float],
    previews: Mapping[float, Sequence[_NormalizedRecord]],
    cell_size: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row, sample in enumerate(previews[scales[0]]):
        for column, scale in enumerate(scales):
            record = previews[scale][row]
            records.append(
                {
                    "sample_id": record.sample_id,
                    "prompt": record.prompt,
                    "sample_seed": record.sample_seed,
                    "cfg": _guidance_tag(scale),
                    "guidance_scale": scale,
                    "eval_steps": eval_steps,
                    "rewards": record.rewards,
                    "metadata": record.metadata,
                    "contact_sheet_row": row,
                    "contact_sheet_column": column,
                }
            )
    return {
        "schema_version": _SCHEMA_VERSION,
        "step": step,
        "eval_steps": eval_steps,
        "guidance_scales": list(scales),
        "sample_count_per_guidance": len(previews[scales[0]]),
        "record_count": len(records),
        "contact_sheet": _CONTACT_SHEET_NAME,
        "contact_sheet_cell_size": cell_size,
        "records": records,
    }


def _write_staged(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_inline_eval_preview(
    output_dir: str | os.PathLike[str],
    *,
    step: int,
    eval_steps: int,
    previews_by_guidance: Mapping[float, Sequence[InlinePreviewRecord]],
    cell_size: int = 256,
    additional_json_artifacts: Mapping[str, Any] | None = None,
) -> Path:
    """Write one complete, reproducibly ordered inline-evaluation preview.

    The returned directory is ``output_dir/step-NNNNNNNNN``. The function never
    truncates ``previews_by_guidance``. Callers must pass the complete preview
    selection accumulated across evaluation batches/ranks.

    Files are fully staged before commit. An existing complete result remains
    valid if rendering/staging fails. Once commit starts, ``.complete`` is
    removed first and recreated only after both final artifacts are replaced;
    therefore an interrupted mixed generation is never advertised as complete.
    Calls targeting the same step are serialized with ``flock``.
    """

    step = _require_integer(step, "step", minimum=0)
    eval_steps = _require_integer(eval_steps, "eval_steps", minimum=1)
    cell_size = _require_integer(cell_size, "cell_size", minimum=64)
    scales, previews = _normalize_previews(previews_by_guidance)

    contact_sheet = _render_contact_sheet(
        step=step,
        eval_steps=eval_steps,
        scales=scales,
        previews=previews,
        cell_size=cell_size,
    )
    manifest = _build_manifest(
        step=step,
        eval_steps=eval_steps,
        scales=scales,
        previews=previews,
        cell_size=cell_size,
    )
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    normalized_additional_artifacts: dict[str, bytes] = {}
    if additional_json_artifacts is not None:
        if not isinstance(additional_json_artifacts, Mapping):
            raise TypeError("additional_json_artifacts must be a mapping")
        reserved_names = {
            _CONTACT_SHEET_NAME,
            _MANIFEST_NAME,
            _COMPLETE_NAME,
            _LOCK_NAME,
        }
        for artifact_name in sorted(additional_json_artifacts):
            if (
                not isinstance(artifact_name, str)
                or not artifact_name
                or artifact_name != Path(artifact_name).name
                or artifact_name.startswith(".")
                or not artifact_name.endswith(".json")
                or artifact_name in reserved_names
            ):
                raise ValueError(
                    f"invalid additional JSON artifact name: {artifact_name!r}"
                )
            normalized_payload = _normalize_json(
                additional_json_artifacts[artifact_name],
                f"additional_json_artifacts.{artifact_name}",
            )
            normalized_additional_artifacts[artifact_name] = (
                json.dumps(
                    normalized_payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
    artifact_payloads = {
        _CONTACT_SHEET_NAME: contact_sheet,
        _MANIFEST_NAME: manifest_payload,
        **normalized_additional_artifacts,
    }
    artifact_sha256 = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(artifact_payloads.items())
    }
    complete_payload = (
        json.dumps(
            {
                "schema_version": _SCHEMA_VERSION,
                "step": step,
                "contact_sheet_sha256": artifact_sha256[_CONTACT_SHEET_NAME],
                "manifest_sha256": artifact_sha256[_MANIFEST_NAME],
                "artifact_sha256": artifact_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    root = Path(output_dir).expanduser()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"inline preview output root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    step_dir = root / f"step-{step:09d}"
    if step_dir.exists() and not step_dir.is_dir():
        raise NotADirectoryError(f"inline preview step path is not a directory: {step_dir}")
    step_dir.mkdir(parents=False, exist_ok=True)

    lock_path = step_dir / _LOCK_NAME
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        nonce = f"{os.getpid()}-{uuid.uuid4().hex}"
        staged_artifacts = {
            name: step_dir / f".{name}.{nonce}.tmp"
            for name in artifact_payloads
        }
        staged_complete = step_dir / f".{_COMPLETE_NAME}.{nonce}.tmp"
        complete_path = step_dir / _COMPLETE_NAME
        try:
            for name, payload in artifact_payloads.items():
                _write_staged(staged_artifacts[name], payload)
            _write_staged(staged_complete, complete_payload)

            _remove_if_present(complete_path)
            try:
                for name in artifact_payloads:
                    os.replace(staged_artifacts[name], step_dir / name)
                os.replace(staged_complete, complete_path)
                directory_fd = os.open(step_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except BaseException:
                _remove_if_present(complete_path)
                raise
        finally:
            for staged in (*staged_artifacts.values(), staged_complete):
                _remove_if_present(staged)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return step_dir
