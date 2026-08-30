from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vllm_mlx.spec_decode.mtp.prompt_lookup_checkpoint_manifest import (
    CheckpointManifestError,
    checkpoint_weight_manifest_records,
    compute_checkpoint_weight_manifest_sha256,
)


def _write_safetensors(
    path: Path,
    *,
    header: bytes | None = None,
    payload: bytes = b"payload",
) -> bytes:
    header = header or json.dumps(
        {"tensor": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
    return header


def _checkpoint(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    headers = {
        "model-00002-of-00002.safetensors": _write_safetensors(
            tmp_path / "model-00002-of-00002.safetensors",
            header=b'{"z":{}}',
        ),
        "model-00001-of-00002.safetensors": _write_safetensors(
            tmp_path / "model-00001-of-00002.safetensors",
            header=b'{"a":{}}',
        ),
        "model-mtp-q4.safetensors": _write_safetensors(
            tmp_path / "model-mtp-q4.safetensors",
            header=b'{"mtp":{}}',
        ),
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "z.weight": "model-00002-of-00002.safetensors",
                    "a.weight": "model-00001-of-00002.safetensors",
                    # Repeated shard references are normal and are deduplicated.
                    "a.bias": "model-00001-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ple_rows.bin").write_bytes(b"ple-row-data")
    (tmp_path / "ple_rows.bin.manifest.json").write_text(
        '{"rows":3,"version":1}', encoding="utf-8"
    )
    return tmp_path, headers


def test_manifest_records_bind_unique_shards_mtp_and_ple(tmp_path: Path) -> None:
    root, headers = _checkpoint(tmp_path)

    records = checkpoint_weight_manifest_records(root)

    assert [record["filename"] for record in records] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model-mtp-q4.safetensors",
        "ple_rows.bin",
    ]
    for record in records[:3]:
        filename = record["filename"]
        assert record == {
            "filename": filename,
            "byte_size": (root / filename).stat().st_size,
            "header_sha256": hashlib.sha256(headers[filename]).hexdigest(),
        }
    assert records[3] == {
        "filename": "ple_rows.bin",
        "byte_size": len(b"ple-row-data"),
        "manifest_sha256": hashlib.sha256(
            b'{"rows":3,"version":1}'
        ).hexdigest(),
    }


def test_manifest_digest_is_canonical_across_weight_map_order(
    tmp_path: Path,
) -> None:
    root, _ = _checkpoint(tmp_path)
    first = compute_checkpoint_weight_manifest_sha256(root)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a.bias": "model-00001-of-00002.safetensors",
                    "a.weight": "model-00001-of-00002.safetensors",
                    "z.weight": "model-00002-of-00002.safetensors",
                }
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    assert compute_checkpoint_weight_manifest_sha256(root) == first


def test_payload_bytes_are_not_read_into_manifest_identity(tmp_path: Path) -> None:
    root, _ = _checkpoint(tmp_path)
    first = compute_checkpoint_weight_manifest_sha256(root)
    shard = root / "model-00001-of-00002.safetensors"
    raw = shard.read_bytes()
    shard.write_bytes(raw[:-7] + b"PAYLOAD")
    ple = root / "ple_rows.bin"
    ple.write_bytes(b"altered-data")

    assert compute_checkpoint_weight_manifest_sha256(root) == first


@pytest.mark.parametrize(
    "filename",
    [
        "../outside.safetensors",
        "/tmp/outside.safetensors",
        "subdir/shard.safetensors",
        "subdir\\shard.safetensors",
        ".hidden.safetensors",
        "not-a-shard.bin",
    ],
)
def test_unsafe_index_filenames_fail_closed(
    tmp_path: Path, filename: str
) -> None:
    root, _ = _checkpoint(tmp_path)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor": filename}}), encoding="utf-8"
    )

    with pytest.raises(CheckpointManifestError, match="unsafe filename"):
        compute_checkpoint_weight_manifest_sha256(root)


@pytest.mark.parametrize(
    ("remove_name", "message"),
    [
        ("model.safetensors.index.json", "valid safetensors index"),
        ("model-00001-of-00002.safetensors", "missing safetensors"),
        ("model-mtp-q4.safetensors", "missing safetensors"),
        ("ple_rows.bin", "missing ple_rows.bin"),
        ("ple_rows.bin.manifest.json", "valid PLE rows manifest"),
    ],
)
def test_required_manifest_inputs_must_exist(
    tmp_path: Path, remove_name: str, message: str
) -> None:
    root, _ = _checkpoint(tmp_path)
    (root / remove_name).unlink()

    with pytest.raises(CheckpointManifestError, match=message):
        compute_checkpoint_weight_manifest_sha256(root)


def test_unindexed_safetensors_file_is_ambiguous(tmp_path: Path) -> None:
    root, _ = _checkpoint(tmp_path)
    _write_safetensors(root / "extra.safetensors")

    with pytest.raises(CheckpointManifestError, match="unindexed safetensors"):
        compute_checkpoint_weight_manifest_sha256(root)


def test_mtp_sidecar_cannot_also_be_an_index_shard(tmp_path: Path) -> None:
    root, _ = _checkpoint(tmp_path)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"mtp.weight": "model-mtp-q4.safetensors"}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointManifestError, match="ambiguously"):
        compute_checkpoint_weight_manifest_sha256(root)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\x08\x00\x00\x00\x00\x00\x00\x00{}",
        (2**63).to_bytes(8, "little") + b"{}",
        (4).to_bytes(8, "little") + b"nope",
        (2).to_bytes(8, "little") + b"[]",
    ],
)
def test_malformed_safetensors_headers_fail_closed(
    tmp_path: Path, raw: bytes
) -> None:
    root, _ = _checkpoint(tmp_path)
    (root / "model-00001-of-00002.safetensors").write_bytes(raw)

    with pytest.raises(CheckpointManifestError, match="safetensors"):
        compute_checkpoint_weight_manifest_sha256(root)


def test_malformed_index_and_ple_manifest_fail_closed(tmp_path: Path) -> None:
    root, _ = _checkpoint(tmp_path)
    (root / "model.safetensors.index.json").write_text("[]", encoding="utf-8")
    with pytest.raises(CheckpointManifestError, match="JSON object"):
        compute_checkpoint_weight_manifest_sha256(root)

    _checkpoint(tmp_path)
    (root / "ple_rows.bin.manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(CheckpointManifestError, match="JSON object"):
        compute_checkpoint_weight_manifest_sha256(root)


def test_header_or_ple_manifest_change_changes_digest(tmp_path: Path) -> None:
    root, _ = _checkpoint(tmp_path)
    first = compute_checkpoint_weight_manifest_sha256(root)
    _write_safetensors(
        root / "model-00001-of-00002.safetensors", header=b'{"b":{}}'
    )
    second = compute_checkpoint_weight_manifest_sha256(root)
    assert second != first

    (root / "ple_rows.bin.manifest.json").write_text(
        '{"rows":4,"version":1}', encoding="utf-8"
    )
    assert compute_checkpoint_weight_manifest_sha256(root) != second
