#!/usr/bin/env python3

"""Rewrite ImageReward 1.5's obsolete exact timm requirement in a wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import tempfile
import zipfile


METADATA_MEMBER = "image_reward-1.5.dist-info/METADATA"
RECORD_MEMBER = "image_reward-1.5.dist-info/RECORD"
OLD_REQUIREMENT = b"Requires-Dist: timm (==0.6.13)\n"
NEW_REQUIREMENT = b"Requires-Dist: timm (>=1.0.17)\n"


def _record_digest(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def rewrite_imagereward_wheel(source: Path, destination: Path) -> None:
    """Write a wheel differing only in METADATA and its matching RECORD row."""

    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if source == destination:
        raise ValueError("source and destination wheels must differ")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if destination.name != "image_reward-1.5-py3-none-any.whl":
        raise ValueError(
            "destination must retain the canonical ImageReward wheel filename"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source, "r") as source_wheel:
        infos = source_wheel.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate member names")
        if names.count(METADATA_MEMBER) != 1 or names.count(RECORD_MEMBER) != 1:
            raise ValueError("wheel does not contain the expected unique metadata files")
        payloads = {info.filename: source_wheel.read(info.filename) for info in infos}

    metadata = payloads[METADATA_MEMBER]
    if metadata.count(OLD_REQUIREMENT) != 1:
        raise ValueError("wheel does not contain the expected exact timm requirement")
    if NEW_REQUIREMENT in metadata:
        raise ValueError("wheel already contains the replacement timm requirement")
    payloads[METADATA_MEMBER] = metadata.replace(OLD_REQUIREMENT, NEW_REQUIREMENT)

    record_text = payloads[RECORD_MEMBER].decode("utf-8")
    rows = list(csv.reader(io.StringIO(record_text)))
    metadata_rows = [row for row in rows if row and row[0] == METADATA_MEMBER]
    record_rows = [row for row in rows if row and row[0] == RECORD_MEMBER]
    if len(metadata_rows) != 1 or len(record_rows) != 1:
        raise ValueError("wheel RECORD does not contain the expected unique rows")
    metadata_rows[0][1:] = [
        _record_digest(payloads[METADATA_MEMBER]),
        str(len(payloads[METADATA_MEMBER])),
    ]
    record_rows[0][1:] = ["", ""]
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    writer.writerows(rows)
    payloads[RECORD_MEMBER] = record_buffer.getvalue().encode("utf-8")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".partial",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        with zipfile.ZipFile(temporary_path, "w") as destination_wheel:
            for info in infos:
                destination_wheel.writestr(info, payloads[info.filename])
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    rewrite_imagereward_wheel(arguments.source, arguments.destination)
    print(arguments.destination)


if __name__ == "__main__":
    main()
