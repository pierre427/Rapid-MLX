# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_mlx.spec_decode.mtp.prepared_state import (
    ABSENT_STATE_LAYOUT,
    PreparedMTPState,
    PreparedStateIdentity,
    RestoreReason,
    evaluate_restore,
    fingerprint_config,
    prepare_mtp_state,
)


def _identity(**changes) -> PreparedStateIdentity:
    values = {
        "model_id": "Qwen/Qwen3.8-Flash-Next",
        "model_revision": "f5d08274",
        "speculative_config_fingerprint": fingerprint_config(
            {"method": "mtp", "num_speculative_tokens": 3}
        ),
        "target_cache_layout": "qwen4:12qsa+36arrays:bf16",
        "mtp_cache_layout": "qwen4-mtp:1qsa:bf16",
        "seed_hidden_layout": "bf16[1,1,2048]",
        "gdn_state_layout": "qwen4:gdn:conv+matrix:bf16+fp32",
        "ple_state_layout": "qwen4:ple:conv+history+atomic-rollback:v1",
        "qsa_state_layout": "qwen4:qsa:kv+raw-index+pooled+shared-topk:v1",
        "adapter_id": None,
        "tokenizer_fingerprint": "tokenizer-sha256",
    }
    values.update(changes)
    return PreparedStateIdentity(**values)


def _state(
    *,
    covered: int = 128,
    identity: PreparedStateIdentity | None = None,
    captured_at: float = 100.0,
) -> tuple[PreparedMTPState, tuple[int, ...]]:
    prefix = tuple(range(covered))
    return (
        prepare_mtp_state(
            identity=identity or _identity(),
            prefix_tokens=prefix,
            target_cache=object(),
            target_cache_tokens=covered,
            mtp_cache=object(),
            mtp_cache_pairs=covered - 1,
            seed_hidden=object(),
            gdn_state=object(),
            ple_state=object(),
            qsa_state=object(),
            captured_at=captured_at,
        ),
        prefix,
    )


def _evaluate(
    state: PreparedMTPState,
    prefix: tuple[int, ...],
    **changes,
):
    values = {
        "expected_identity": _identity(),
        "request_tokens": prefix + (999,),
        "target_cache_tokens": len(prefix),
        "mtp_cache_pairs": len(prefix) - 1,
        "now": 110.0,
        "max_age_seconds": 60.0,
        "min_useful_prefix_tokens": 64,
    }
    values.update(changes)
    return evaluate_restore(state, **values)


def test_exact_joint_boundary_is_restore_eligible() -> None:
    state, prefix = _state()

    decision = _evaluate(state, prefix)

    assert decision.eligible is True
    assert decision.reason is RestoreReason.ELIGIBLE
    assert decision.covered_tokens == 128
    assert decision.resume_at == 128
    assert decision.bypass_hit is False
    assert state.metadata.logical_context_tokens == 128


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"target_cache_tokens": 127}, "exact covered prefix length"),
        ({"mtp_cache_pairs": 126}, "target_cache_tokens - 1"),
        ({"seed_hidden": None}, "seed_hidden"),
        ({"target_cache": None}, "target_cache"),
        ({"mtp_cache": None}, "mtp_cache"),
    ],
)
def test_capture_rejects_incomplete_or_unaligned_state(changes, message) -> None:
    kwargs = {
        "identity": _identity(),
        "prefix_tokens": tuple(range(128)),
        "target_cache": object(),
        "target_cache_tokens": 128,
        "mtp_cache": object(),
        "mtp_cache_pairs": 127,
        "seed_hidden": object(),
        "gdn_state": object(),
        "ple_state": object(),
        "qsa_state": object(),
        "captured_at": 100.0,
    }
    kwargs.update(changes)

    with pytest.raises(ValueError, match=message):
        prepare_mtp_state(**kwargs)


def test_valid_trivial_hit_fails_open_to_normal_mtp() -> None:
    state, prefix = _state(covered=8)

    decision = _evaluate(state, prefix)

    assert decision.eligible is False
    assert decision.reason is RestoreReason.TRIVIAL_HIT
    assert decision.bypass_hit is True
    assert decision.resume_at is None


@pytest.mark.parametrize(
    "expected_identity",
    [
        _identity(model_id="other/model"),
        _identity(model_revision="different-revision"),
        _identity(adapter_id="adapter-a"),
        _identity(tokenizer_fingerprint="different-tokenizer"),
    ],
)
def test_model_identity_mismatch_refuses_restore(expected_identity) -> None:
    state, prefix = _state()

    decision = _evaluate(
        state,
        prefix,
        expected_identity=expected_identity,
    )

    assert decision.eligible is False
    assert decision.reason is RestoreReason.MODEL_MISMATCH
    assert decision.bypass_hit is False


@pytest.mark.parametrize(
    "expected_identity",
    [
        _identity(
            speculative_config_fingerprint=fingerprint_config(
                {"method": "mtp", "num_speculative_tokens": 2}
            )
        ),
        _identity(target_cache_layout="different-target-layout"),
        _identity(mtp_cache_layout="different-mtp-layout"),
        _identity(seed_hidden_layout="bf16[1,1,4096]"),
        _identity(gdn_state_layout="different-gdn-layout"),
        _identity(ple_state_layout="different-ple-layout"),
        _identity(qsa_state_layout="different-qsa-layout"),
    ],
)
def test_config_or_layout_mismatch_refuses_restore(expected_identity) -> None:
    state, prefix = _state()

    decision = _evaluate(
        state,
        prefix,
        expected_identity=expected_identity,
    )

    assert decision.eligible is False
    assert decision.reason is RestoreReason.CONFIG_MISMATCH


def test_stale_state_refuses_restore() -> None:
    state, prefix = _state(captured_at=100.0)

    decision = _evaluate(
        state,
        prefix,
        now=161.0,
        max_age_seconds=60.0,
    )

    assert decision.eligible is False
    assert decision.reason is RestoreReason.STALE


@pytest.mark.parametrize(
    "changes",
    [
        {"target_cache_tokens": 127},
        {"mtp_cache_pairs": 126},
        {"request_tokens": tuple(range(127)) + (777, 999)},
        {"request_tokens": tuple(range(128))},
    ],
)
def test_live_or_token_boundary_mismatch_refuses_restore(changes) -> None:
    state, prefix = _state()

    decision = _evaluate(state, prefix, **changes)

    assert decision.eligible is False
    assert decision.reason is RestoreReason.BOUNDARY_MISMATCH


@pytest.mark.parametrize(
    "metadata_change",
    [
        {"mtp_covered_pairs": 3},
        {"schema_version": 999},
        {"captured_at": "not-a-timestamp"},
        {"boundary_fingerprint": "z" * 64},
    ],
)
def test_corrupt_persisted_metadata_refuses_without_raising(
    metadata_change,
) -> None:
    state, prefix = _state()
    corrupt = PreparedMTPState(
        metadata=replace(state.metadata, **metadata_change),
        target_cache=state.target_cache,
        mtp_cache=state.mtp_cache,
        seed_hidden=state.seed_hidden,
        gdn_state=state.gdn_state,
        ple_state=state.ple_state,
        qsa_state=state.qsa_state,
    )

    decision = _evaluate(corrupt, prefix)

    assert decision.eligible is False
    assert decision.reason is RestoreReason.MALFORMED


def test_config_fingerprint_is_order_independent_and_value_sensitive() -> None:
    left = fingerprint_config({"method": "mtp", "k": 3})
    reordered = fingerprint_config({"k": 3, "method": "mtp"})
    changed = fingerprint_config({"method": "mtp", "k": 2})

    assert left == reordered
    assert left != changed


@pytest.mark.parametrize("surface", ["gdn_state", "ple_state", "qsa_state"])
def test_capture_requires_every_declared_mutable_surface(surface) -> None:
    kwargs = {
        "identity": _identity(),
        "prefix_tokens": tuple(range(128)),
        "target_cache": object(),
        "target_cache_tokens": 128,
        "mtp_cache": object(),
        "mtp_cache_pairs": 127,
        "seed_hidden": object(),
        "gdn_state": object(),
        "ple_state": object(),
        "qsa_state": object(),
        "captured_at": 100.0,
    }
    kwargs[surface] = None

    with pytest.raises(ValueError, match=f"{surface} must be present"):
        prepare_mtp_state(**kwargs)


@pytest.mark.parametrize(
    ("surface", "layout"),
    [
        ("gdn_state", "gdn_state_layout"),
        ("ple_state", "ple_state_layout"),
        ("qsa_state", "qsa_state_layout"),
    ],
)
def test_restore_refuses_payload_missing_from_declared_surface(
    surface, layout
) -> None:
    state, prefix = _state()
    values = {
        "metadata": state.metadata,
        "target_cache": state.target_cache,
        "mtp_cache": state.mtp_cache,
        "seed_hidden": state.seed_hidden,
        "gdn_state": state.gdn_state,
        "ple_state": state.ple_state,
        "qsa_state": state.qsa_state,
    }
    values[surface] = None

    decision = _evaluate(PreparedMTPState(**values), prefix)

    assert getattr(state.metadata.identity, layout) != "absent"
    assert decision.eligible is False
    assert decision.reason is RestoreReason.BOUNDARY_MISMATCH


def test_architecture_absence_is_explicit_and_accepts_no_payload() -> None:
    identity = _identity(
        ple_state_layout=ABSENT_STATE_LAYOUT,
        qsa_state_layout=ABSENT_STATE_LAYOUT,
    )
    prefix = tuple(range(128))

    state = prepare_mtp_state(
        identity=identity,
        prefix_tokens=prefix,
        target_cache=object(),
        target_cache_tokens=128,
        mtp_cache=object(),
        mtp_cache_pairs=127,
        seed_hidden=object(),
        gdn_state=object(),
        ple_state=None,
        qsa_state=None,
        captured_at=100.0,
    )

    decision = _evaluate(state, prefix, expected_identity=identity)

    assert decision.eligible is True


def test_declared_absent_surface_refuses_non_none_payload() -> None:
    identity = _identity(ple_state_layout=ABSENT_STATE_LAYOUT)
    kwargs = {
        "identity": identity,
        "prefix_tokens": tuple(range(128)),
        "target_cache": object(),
        "target_cache_tokens": 128,
        "mtp_cache": object(),
        "mtp_cache_pairs": 127,
        "seed_hidden": object(),
        "gdn_state": object(),
        "ple_state": object(),
        "qsa_state": object(),
        "captured_at": 100.0,
    }

    with pytest.raises(ValueError, match="ple_state must be None"):
        prepare_mtp_state(**kwargs)
