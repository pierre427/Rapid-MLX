# SPDX-License-Identifier: Apache-2.0
"""Canonical checkpoint weight-manifest identity for Qwen4 attestation.

The helper is pure Python and deliberately does not hash tensor payloads.  It
binds every index-referenced shard by filename, byte size, and exact
safetensors header bytes; then binds the explicit MTP sidecar the same way and
the PLE rows by filename, byte size, and manifest-file digest.  Any ambiguous
or incomplete local layout is refused rather than silently omitted.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

INDEX_FILENAME = "model.safetensors.index.json"
MTP_SIDECAR_FILENAME = "model-mtp-q4.safetensors"
PLE_ROWS_FILENAME = "ple_rows.bin"
PLE_MANIFEST_FILENAME = "ple_rows.bin.manifest.json"
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024

_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class CheckpointManifestError(ValueError):
    """Checkpoint layout cannot be used as trusted runtime identity."""


def compute_checkpoint_weight_manifest_sha256(
    checkpoint_directory: str | Path,
) -> str:
    """Return SHA-256 of canonical, sorted checkpoint manifest records."""

    records = checkpoint_weight_manifest_records(checkpoint_directory)
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_weight_manifest_records(
    checkpoint_directory: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Build canonical records without reading safetensors payload bytes."""

    root = Path(checkpoint_directory)
    if not root.is_dir():
        raise CheckpointManifestError("checkpoint directory does not exist")

    index = _read_json_mapping(root / INDEX_FILENAME, label="safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise CheckpointManifestError("safetensors index has no weight_map")

    indexed_names: set[str] = set()
    for tensor_name, raw_filename in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise CheckpointManifestError("weight_map tensor names must be strings")
        indexed_names.add(_validate_safetensors_filename(raw_filename))

    if MTP_SIDECAR_FILENAME in indexed_names:
        raise CheckpointManifestError(
            "MTP sidecar is ambiguously both indexed and explicit"
        )
    expected_safetensors = indexed_names | {MTP_SIDECAR_FILENAME}
    if len({name.casefold() for name in expected_safetensors}) != len(
        expected_safetensors
    ):
        raise CheckpointManifestError("safetensors filenames are ambiguous")

    actual_safetensors = {
        entry.name
        for entry in root.iterdir()
        if entry.name.endswith(".safetensors")
    }
    missing = expected_safetensors - actual_safetensors
    extra = actual_safetensors - expected_safetensors
    if missing:
        raise CheckpointManifestError(
            f"missing safetensors files: {', '.join(sorted(missing))}"
        )
    if extra:
        raise CheckpointManifestError(
            f"unindexed safetensors files: {', '.join(sorted(extra))}"
        )

    records = [
        _safetensors_record(root, filename)
        for filename in sorted(expected_safetensors)
    ]

    ple_path = root / PLE_ROWS_FILENAME
    if not ple_path.is_file():
        raise CheckpointManifestError(f"missing {PLE_ROWS_FILENAME}")
    ple_manifest_path = root / PLE_MANIFEST_FILENAME
    # Parsing prevents arbitrary malformed bytes from being blessed merely
    # because they happen to have a stable digest.
    _read_json_mapping(ple_manifest_path, label="PLE rows manifest")
    records.append(
        {
            "filename": PLE_ROWS_FILENAME,
            "byte_size": _file_size(ple_path, PLE_ROWS_FILENAME),
            "manifest_sha256": _sha256_file(ple_manifest_path),
        }
    )
    return tuple(sorted(records, key=lambda record: record["filename"]))


def _safetensors_record(root: Path, filename: str) -> dict[str, Any]:
    path = root / filename
    size = _file_size(path, filename)
    if size < 9:
        raise CheckpointManifestError(f"malformed safetensors file: {filename}")
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise CheckpointManifestError(
                    f"truncated safetensors header length: {filename}"
                )
            header_length = int.from_bytes(raw_length, "little", signed=False)
            if (
                header_length < 2
                or header_length > MAX_SAFETENSORS_HEADER_BYTES
                or header_length > size - 8
            ):
                raise CheckpointManifestError(
                    f"invalid safetensors header length: {filename}"
                )
            header = handle.read(header_length)
    except OSError as exc:
        raise CheckpointManifestError(f"cannot read {filename}") from exc
    if len(header) != header_length:
        raise CheckpointManifestError(f"truncated safetensors header: {filename}")
    try:
        decoded = json.loads(header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointManifestError(
            f"malformed safetensors header JSON: {filename}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise CheckpointManifestError(
            f"safetensors header must be a JSON object: {filename}"
        )
    return {
        "filename": filename,
        "byte_size": size,
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }


def _validate_safetensors_filename(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.endswith(".safetensors")
        or _SAFE_FILENAME.fullmatch(value) is None
        or Path(value).name != value
    ):
        raise CheckpointManifestError("weight_map contains an unsafe filename")
    return value


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        with path.open("rb") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointManifestError(f"cannot read valid {label}") from exc
    if not isinstance(value, Mapping):
        raise CheckpointManifestError(f"{label} must be a JSON object")
    return value


def _file_size(path: Path, filename: str) -> int:
    try:
        if not path.is_file():
            raise CheckpointManifestError(f"missing file: {filename}")
        size = path.stat().st_size
    except OSError as exc:
        raise CheckpointManifestError(f"cannot stat {filename}") from exc
    if size < 1:
        raise CheckpointManifestError(f"empty file: {filename}")
    return size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CheckpointManifestError(f"cannot read {path.name}") from exc
    return digest.hexdigest()


__all__ = [
    "CheckpointManifestError",
    "INDEX_FILENAME",
    "MAX_SAFETENSORS_HEADER_BYTES",
    "MTP_SIDECAR_FILENAME",
    "PLE_MANIFEST_FILENAME",
    "PLE_ROWS_FILENAME",
    "checkpoint_weight_manifest_records",
    "compute_checkpoint_weight_manifest_sha256",
]
