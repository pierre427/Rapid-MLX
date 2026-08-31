# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path

import pytest

from vllm_mlx.models.qwen4_conformance import (
    ParityTier,
    Qwen4TransactionRecord,
    compare_artifact_receipts,
    inspect_qwen4_artifact,
    required_runtime_parity_tier,
    validate_qwen4_config,
    validate_transaction,
)


def _config():
    return {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "qwen_sparse_attention",
            ],
            "hc_count": 4,
            "ple_layer_ids": [2],
            "indexer_n_heads": 24,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "ngram_size": 3,
            "heads_per_ngram": 16,
            "split_ngram_parts": 128,
            "mtp_num_hidden_layers": 1,
        },
    }


def _artifact():
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    (root / "config.json").write_text(json.dumps(_config(), sort_keys=True))
    shard = "model-00001-of-00001.safetensors"
    (root / shard).touch()
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.model.layers.2.ple.shard_0.weight": shard,
                    "language_model.model.layers.3.self_attn.q_proj.weight": shard,
                    "mtp.layers.0.mlp.up_proj.weight": shard,
                }
            },
            sort_keys=True,
        )
    )
    return temporary, root


def test_artifact_receipt_is_exact_and_comparable():
    temporary, root = _artifact()
    try:
        receipt = inspect_qwen4_artifact(root)
        assert receipt.model_type == "qwen4_exp"
        assert receipt.tensor_count == 3
        assert receipt.ple_tensor_count == 1
        assert receipt.mtp_tensor_count == 1
        assert receipt.layer_types[-1] == "full_attention"
        compare_artifact_receipts(receipt, receipt)
    finally:
        temporary.cleanup()


def test_artifact_receipt_detects_config_drift():
    first_temporary, first = _artifact()
    second_temporary, second = _artifact()
    try:
        changed = _config()
        changed["text_config"]["indexer_budget"] = 1024
        (second / "config.json").write_text(json.dumps(changed, sort_keys=True))
        with pytest.raises(ValueError, match="config_sha256.*indexer_budget"):
            compare_artifact_receipts(
                inspect_qwen4_artifact(first), inspect_qwen4_artifact(second)
            )
    finally:
        first_temporary.cleanup()
        second_temporary.cleanup()


def test_config_rejects_an_incomplete_architecture():
    config = _config()
    config["text_config"]["layer_types"] = ["linear_attention"] * 4
    with pytest.raises(ValueError, match="both GDN and QSA"):
        validate_qwen4_config(config)


def test_transaction_requires_one_combined_physical_trim():
    record = Qwen4TransactionRecord(
        lane_uids=(10, 20),
        draft_depths=(3, 2),
        accepted_lengths=(1, 2),
        emitted_counts=(0, 3),
        terminal=(True, False),
        physical_target_drops=(4, 0),
        physical_trim_count=1,
    )
    assert validate_transaction(record) == (4, 0)

    wrong = Qwen4TransactionRecord(
        **{
            **record.__dict__,
            "physical_target_drops": (2, 0),
            "physical_trim_count": 2,
        }
    )
    with pytest.raises(ValueError, match="combined drop"):
        validate_transaction(wrong)


def test_runtime_parity_tiers_are_explicit():
    assert (
        required_runtime_parity_tier(
            stateful=True,
            same_backend_dtype=True,
            topology_changed=False,
            quantization_changed=False,
        )
        == ParityTier.TRANSACTIONAL
    )
    assert (
        required_runtime_parity_tier(
            stateful=False,
            same_backend_dtype=True,
            topology_changed=False,
            quantization_changed=False,
        )
        == ParityTier.TOKEN
    )
    assert (
        required_runtime_parity_tier(
            stateful=True,
            same_backend_dtype=False,
            topology_changed=False,
            quantization_changed=True,
        )
        == ParityTier.DISTRIBUTIONAL
    )
