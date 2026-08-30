# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_mlx.spec_decode.mtp.prompt_lookup_capability import (
    GDN_RECURRENT,
    MTP_KV,
    PLE_HISTORY,
    QSA_INDEX,
    TARGET_KV,
    PromptLookupRefusal,
    PromptLookupVerificationIdentity,
    evaluate_prompt_lookup_capability,
    make_prompt_lookup_capability,
)

_VERIFICATION = PromptLookupVerificationIdentity(
    model_id="test/qwen-hybrid",
    model_revision="weights-sha256",
    runtime_commit="rapid-commit-sha",
    cache_topology="target-kv+mtp-kv+gdn",
    state_dtype="bf16+fp32",
    verify_geometry="batch=1,width=1..4",
    oracle_version="raw-bit-state-v1",
    test_digest="a" * 64,
)


def _model(capability=None, *, supported=True, runtime_identity=_VERIFICATION):
    values = {"mtp_prompt_lookup_supported": supported}
    if capability is not None:
        values["mtp_prompt_lookup_capability"] = capability
    if runtime_identity is not None:
        values["mtp_prompt_lookup_runtime_identity"] = runtime_identity
    return SimpleNamespace(**values)


def _capability(*, surfaces=(TARGET_KV, MTP_KV, GDN_RECURRENT), **changes):
    values = {
        "mutable_state_surfaces": surfaces,
        "target_rollback_to_accepted": True,
        "mtp_advance_by_accepted": True,
        "recurrent_advance_by_accepted": True,
        "auxiliary_rollback_to_accepted": True,
        "verification_identity": _VERIFICATION,
    }
    values.update(changes)
    return make_prompt_lookup_capability(**values)


def test_historical_boolean_alone_cannot_enable_adaptive_pld() -> None:
    decision = evaluate_prompt_lookup_capability(_model())

    assert decision.eligible is False
    assert decision.reason is PromptLookupRefusal.MISSING_DESCRIPTOR


def test_mechanism_descriptor_without_oracle_attestation_fails_closed() -> None:
    capability = _capability(verification_identity=None)

    decision = evaluate_prompt_lookup_capability(_model(capability))

    assert decision.eligible is False
    assert decision.reason is PromptLookupRefusal.MISSING_VERIFICATION


def test_family_injector_cannot_self_attest_from_model_config() -> None:
    from vllm_mlx.spec_decode.mtp.qwen3_5_inject import PROMPT_LOOKUP_CAPABILITY

    decision = evaluate_prompt_lookup_capability(
        _model(PROMPT_LOOKUP_CAPABILITY, runtime_identity=None)
    )

    assert decision.eligible is False
    assert decision.reason is PromptLookupRefusal.MISSING_VERIFICATION


def test_environment_and_config_cannot_create_positive_attestation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP", "1")
    capability = _capability(verification_identity=None)
    model = _model(capability)
    model.config = {
        "mtp_prompt_lookup_verified": True,
        "verification_identity": _VERIFICATION.__dict__,
    }
    model.mtp_prompt_lookup_verification_config = _VERIFICATION.__dict__

    decision = evaluate_prompt_lookup_capability(model)

    assert decision.eligible is False
    assert decision.reason is PromptLookupRefusal.MISSING_VERIFICATION


def test_missing_or_mismatched_runtime_identity_fails_closed() -> None:
    capability = _capability()
    missing = evaluate_prompt_lookup_capability(
        _model(capability, runtime_identity=None)
    )
    mismatched_identity = PromptLookupVerificationIdentity(
        **{
            **_VERIFICATION.__dict__,
            "model_revision": "different-weights",
        }
    )
    mismatched = evaluate_prompt_lookup_capability(
        _model(capability, runtime_identity=mismatched_identity)
    )

    assert missing.reason is PromptLookupRefusal.MISSING_VERIFICATION
    assert mismatched.reason is PromptLookupRefusal.VERIFICATION_IDENTITY_MISMATCH


@pytest.mark.parametrize("surface", [GDN_RECURRENT, PLE_HISTORY])
def test_hybrid_recurrent_surface_requires_accepted_token_advance(surface) -> None:
    capability = _capability(
        surfaces=(TARGET_KV, MTP_KV, surface),
        recurrent_advance_by_accepted=False,
    )

    decision = evaluate_prompt_lookup_capability(_model(capability))

    assert decision.eligible is False
    assert decision.reason is PromptLookupRefusal.MISSING_ACCEPTED_ADVANCE


def test_qwen4_shaped_state_fails_closed_without_auxiliary_rollback() -> None:
    capability = _capability(
        surfaces=(
            TARGET_KV,
            MTP_KV,
            GDN_RECURRENT,
            PLE_HISTORY,
            QSA_INDEX,
        ),
        auxiliary_rollback_to_accepted=False,
    )

    decision = evaluate_prompt_lookup_capability(_model(capability))

    assert decision.eligible is False
    assert decision.reason is PromptLookupRefusal.MISSING_ACCEPTED_ADVANCE


def test_audited_hybrid_contract_is_eligible() -> None:
    decision = evaluate_prompt_lookup_capability(_model(_capability()))

    assert decision.eligible is True
    assert decision.reason is PromptLookupRefusal.ELIGIBLE
    assert decision.mutable_state_surfaces == frozenset(
        {TARGET_KV, MTP_KV, GDN_RECURRENT}
    )


@pytest.mark.parametrize(
    "surfaces",
    [
        (TARGET_KV,),
        (TARGET_KV, MTP_KV, "unknown"),
        (TARGET_KV, MTP_KV, MTP_KV),
    ],
)
def test_malformed_surface_inventory_fails_closed(surfaces) -> None:
    decision = evaluate_prompt_lookup_capability(
        _model(_capability(surfaces=surfaces))
    )

    assert decision.eligible is False
    assert decision.reason is PromptLookupRefusal.MALFORMED_DESCRIPTOR
