# SPDX-License-Identifier: Apache-2.0

"""Model-free Qwen4 artifact and transaction conformance contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

CONTRACT_VERSION = 1


class ParityTier(str, Enum):
    STRUCTURAL = "structural"
    TRANSACTIONAL = "transactional"
    TOKEN = "token"
    DISTRIBUTIONAL = "distributional"


@dataclass(frozen=True)
class Qwen4ArtifactReceipt:
    contract_version: int
    config_sha256: str
    index_sha256: str
    model_type: str
    text_model_type: str
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    hc_count: int
    ple_layer_ids: tuple[int, ...]
    indexer_n_heads: int
    indexer_budget: int
    indexer_compress_ratio: int
    mtp_num_hidden_layers: int
    tensor_count: int
    shard_names: tuple[str, ...]
    ple_tensor_count: int
    mtp_tensor_count: int


@dataclass(frozen=True)
class Qwen4TransactionRecord:
    lane_uids: tuple[int, ...]
    draft_depths: tuple[int, ...]
    accepted_lengths: tuple[int, ...]
    emitted_counts: tuple[int, ...]
    terminal: tuple[bool, ...]
    physical_target_drops: tuple[int, ...]
    physical_trim_count: int


def _positive_int(config: Mapping, name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Qwen4 config needs a positive {name}")
    return value


def validate_qwen4_config(config: Mapping) -> Mapping:
    if config.get("model_type") != "qwen4_exp":
        raise ValueError("Qwen4 artifact model_type must be 'qwen4_exp'")
    text = config.get("text_config", config)
    if not isinstance(text, Mapping):
        raise ValueError("Qwen4 text_config must be an object")
    if text.get("model_type") not in (None, "qwen4_exp_text"):
        raise ValueError("Qwen4 text model_type must be 'qwen4_exp_text'")

    layers = _positive_int(text, "num_hidden_layers")
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, Sequence) or isinstance(layer_types, str):
        raise ValueError("Qwen4 layer_types must be a sequence")
    if len(layer_types) != layers:
        raise ValueError("Qwen4 layer_types must cover every hidden layer")
    layer_types = tuple(
        "full_attention" if value == "qwen_sparse_attention" else value
        for value in layer_types
    )
    allowed = {"linear_attention", "full_attention"}
    if set(layer_types) - allowed or set(layer_types) != allowed:
        raise ValueError("Qwen4 needs both GDN and QSA layer types")

    hc_count = _positive_int(text, "hc_count")
    ple_layer_ids = text.get("ple_layer_ids")
    if not isinstance(ple_layer_ids, Sequence) or isinstance(ple_layer_ids, str):
        raise ValueError("Qwen4 ple_layer_ids must be a sequence")
    if not ple_layer_ids or any(
        not isinstance(value, int) or not 0 <= value < layers for value in ple_layer_ids
    ):
        raise ValueError("Qwen4 ple_layer_ids must name valid hidden layers")

    indexer_n_heads = _positive_int(text, "indexer_n_heads")
    indexer_budget = _positive_int(text, "indexer_budget")
    indexer_compress_ratio = _positive_int(text, "indexer_compress_ratio")
    if indexer_budget % indexer_compress_ratio:
        raise ValueError("Qwen4 indexer budget must align to its compress ratio")
    _positive_int(text, "ngram_size")
    _positive_int(text, "heads_per_ngram")
    _positive_int(text, "split_ngram_parts")
    mtp_layers = _positive_int(text, "mtp_num_hidden_layers")
    if mtp_layers != 1:
        raise ValueError("Qwen4-Exp requires one native MTP layer")

    return {
        "text": text,
        "num_hidden_layers": layers,
        "layer_types": layer_types,
        "hc_count": hc_count,
        "ple_layer_ids": tuple(int(value) for value in ple_layer_ids),
        "indexer_n_heads": indexer_n_heads,
        "indexer_budget": indexer_budget,
        "indexer_compress_ratio": indexer_compress_ratio,
        "mtp_num_hidden_layers": mtp_layers,
    }


def inspect_qwen4_artifact(model_path: Path) -> Qwen4ArtifactReceipt:
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    config_bytes = config_path.read_bytes()
    index_bytes = index_path.read_bytes()
    config = json.loads(config_bytes)
    index = json.loads(index_bytes)
    validated = validate_qwen4_config(config)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Qwen4 artifact needs a non-empty safetensors weight_map")

    root = model_path.resolve()
    shard_names = tuple(sorted(set(weight_map.values())))
    for shard_name in shard_names:
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("Qwen4 weight_map contains an invalid shard name")
        shard_path = (root / shard_name).resolve()
        if shard_path == root or root not in shard_path.parents:
            raise ValueError(f"Qwen4 shard {shard_name!r} escapes the artifact")
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing Qwen4 shard: {shard_path}")

    tensor_names = tuple(weight_map)
    ple_count = sum(
        ".ple" in name or ".ngram" in name or "shared_embedding" in name
        for name in tensor_names
    )
    mtp_count = sum(name.startswith("mtp.") or ".mtp." in name for name in tensor_names)
    if ple_count == 0:
        raise ValueError("Qwen4 artifact index contains no PLE tensors")

    text = validated["text"]
    return Qwen4ArtifactReceipt(
        contract_version=CONTRACT_VERSION,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        model_type=config["model_type"],
        text_model_type=text.get("model_type", "qwen4_exp_text"),
        num_hidden_layers=validated["num_hidden_layers"],
        layer_types=validated["layer_types"],
        hc_count=validated["hc_count"],
        ple_layer_ids=validated["ple_layer_ids"],
        indexer_n_heads=validated["indexer_n_heads"],
        indexer_budget=validated["indexer_budget"],
        indexer_compress_ratio=validated["indexer_compress_ratio"],
        mtp_num_hidden_layers=validated["mtp_num_hidden_layers"],
        tensor_count=len(tensor_names),
        shard_names=shard_names,
        ple_tensor_count=ple_count,
        mtp_tensor_count=mtp_count,
    )


def compare_artifact_receipts(
    expected: Qwen4ArtifactReceipt,
    actual: Qwen4ArtifactReceipt,
) -> None:
    if expected != actual:
        differences = [
            name
            for name in expected.__dataclass_fields__
            if getattr(expected, name) != getattr(actual, name)
        ]
        raise ValueError("Qwen4 artifact receipt mismatch: " + ", ".join(differences))


def validate_transaction(record: Qwen4TransactionRecord) -> tuple[int, ...]:
    widths = {
        len(record.lane_uids),
        len(record.draft_depths),
        len(record.accepted_lengths),
        len(record.emitted_counts),
        len(record.terminal),
        len(record.physical_target_drops),
    }
    if len(widths) != 1 or not record.lane_uids:
        raise ValueError("Qwen4 transaction vectors must have one non-empty width")
    if len(set(record.lane_uids)) != len(record.lane_uids):
        raise ValueError("Qwen4 transaction lane ids must be unique")

    expected_drops = []
    for draft, accepted, emitted, terminal in zip(
        record.draft_depths,
        record.accepted_lengths,
        record.emitted_counts,
        record.terminal,
    ):
        if not 0 <= accepted <= draft:
            raise ValueError("Qwen4 accepted length is outside the draft")
        available = accepted + 1
        if not 0 <= emitted <= available:
            raise ValueError("Qwen4 emitted count is outside the proposal")
        if not terminal and emitted != available:
            raise ValueError("A live Qwen4 lane must consume its full proposal")
        if terminal and emitted < available and emitted > accepted:
            raise ValueError("A terminal Qwen4 prefix cannot split the bonus token")
        delivery_drop = (
            accepted - emitted + 1 if terminal and emitted <= accepted else 0
        )
        expected_drops.append(draft - accepted + delivery_drop)

    expected_drops = tuple(expected_drops)
    if record.physical_target_drops != expected_drops:
        raise ValueError("Qwen4 physical target rollback is not the combined drop")
    expected_trim_count = 1 if any(expected_drops) else 0
    if record.physical_trim_count != expected_trim_count:
        raise ValueError("Qwen4 target cache needs exactly one physical trim")
    return expected_drops


def required_runtime_parity_tier(
    *,
    stateful: bool,
    same_backend_dtype: bool,
    topology_changed: bool,
    quantization_changed: bool,
) -> ParityTier:
    if topology_changed or quantization_changed or not same_backend_dtype:
        return ParityTier.DISTRIBUTIONAL
    if stateful:
        return ParityTier.TRANSACTIONAL
    return ParityTier.TOKEN
