# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_mlx.spec_decode.mtp.prompt_lookup_attestation import (
    DIRECT_SPARSE_M3_QSA_ROUTE,
    FIXED_M4_COMPILED_VERIFY_ROUTE,
    HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    REQUIRED_STATE_SURFACES,
    AttestationSubject,
    OracleCaseEvidence,
    OracleExecutionKind,
    OracleGeometry,
    PromptLookupOracleEvidence,
    RawBitValue,
    RouteEngagementEvidence,
    SurfaceEvidence,
    issue_prompt_lookup_attestation,
)
from vllm_mlx.spec_decode.mtp.verify_candidate_attestation import (
    VerifyCandidate,
    VerifyCandidateRefusal,
    evaluate_verify_candidate_authority,
)


def _subject(geometry: OracleGeometry) -> AttestationSubject:
    return AttestationSubject(
        model_id="qwen4_exp",
        model_revision="weights",
        runtime_commit="rapid+mlx-lm",
        cache_topology="target+mtp+gdn+ple+qsa",
        state_dtype="bf16+fp32",
        verify_geometry=geometry.fingerprint,
        oracle_version="raw-bit-v1",
    )


def _authority(candidate: VerifyCandidate):
    if candidate is VerifyCandidate.FIXED_M4_COMPILED:
        route = FIXED_M4_COMPILED_VERIFY_ROUTE
        kind = OracleExecutionKind.COMPILED_CANDIDATE
        width = 4
        compile_count = 1
    else:
        route = DIRECT_SPARSE_M3_QSA_ROUTE
        kind = OracleExecutionKind.SPARSE_CANDIDATE
        width = 3
        compile_count = 0
    geometry = OracleGeometry(batch_size=1, verify_width=width)
    subject = _subject(geometry)
    cases = []
    for key in geometry.required_case_keys:
        surfaces = []
        for name in sorted(REQUIRED_STATE_SURFACES):
            value = RawBitValue.from_canonical_value(
                dtype="test", value=[key.accepted_prefixes, key.phase.value, name]
            )
            surfaces.append(SurfaceEvidence(name, value, value))
        cases.append(OracleCaseEvidence(key, tuple(surfaces)))
    evidence = PromptLookupOracleEvidence(
        subject=subject,
        geometry=geometry,
        production_metal=True,
        route=RouteEngagementEvidence(
            route_name=route,
            execution_kind=kind,
            compiled_candidate=(kind is OracleExecutionKind.COMPILED_CANDIDATE),
            candidate_invocations=len(cases),
            fallback_count=0,
            compile_count=compile_count,
            warmup_compile_count=compile_count,
            post_warmup_recompile_count=0,
        ),
        cases=tuple(cases),
    )
    return (
        issue_prompt_lookup_attestation(
            evidence,
            expected_subject=subject,
            expected_geometry=geometry,
            expected_route=route,
        ),
        subject,
    )


def _eager_authority():
    geometry = OracleGeometry(batch_size=1, verify_width=3)
    subject = _subject(geometry)
    cases = []
    for key in geometry.required_case_keys:
        surfaces = []
        for name in sorted(REQUIRED_STATE_SURFACES):
            value = RawBitValue.from_canonical_value(
                dtype="test", value=[key.accepted_prefixes, key.phase.value, name]
            )
            surfaces.append(SurfaceEvidence(name, value, value))
        cases.append(OracleCaseEvidence(key, tuple(surfaces)))
    evidence = PromptLookupOracleEvidence(
        subject=subject,
        geometry=geometry,
        production_metal=True,
        route=RouteEngagementEvidence(
            route_name=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
            execution_kind=OracleExecutionKind.EAGER_TRANSACTION,
            compiled_candidate=False,
            candidate_invocations=len(cases),
            fallback_count=0,
            compile_count=0,
            warmup_compile_count=0,
            post_warmup_recompile_count=0,
        ),
        cases=tuple(cases),
    )
    return (
        issue_prompt_lookup_attestation(
            evidence,
            expected_subject=subject,
            expected_geometry=geometry,
            expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
        ),
        subject,
    )


@pytest.mark.parametrize("candidate", list(VerifyCandidate))
def test_exact_candidate_receipt_is_consumed(candidate):
    authority, subject = _authority(candidate)

    result = evaluate_verify_candidate_authority(
        authority, expected_subject=subject, candidate=candidate
    )

    assert result.eligible is True
    assert result.reason is VerifyCandidateRefusal.ELIGIBLE
    assert result.evidence_digest == authority.evidence_digest


def test_existing_eager_or_other_candidate_receipt_cannot_cross_authorize():
    compiled, compiled_subject = _authority(VerifyCandidate.FIXED_M4_COMPILED)
    sparse, sparse_subject = _authority(VerifyCandidate.DIRECT_SPARSE_M3_QSA)
    eager, eager_subject = _eager_authority()

    wrong_route = evaluate_verify_candidate_authority(
        sparse,
        expected_subject=sparse_subject,
        candidate=VerifyCandidate.FIXED_M4_COMPILED,
    )
    wrong_identity = evaluate_verify_candidate_authority(
        compiled,
        expected_subject=replace(compiled_subject, model_revision="other"),
        candidate=VerifyCandidate.FIXED_M4_COMPILED,
    )
    eager_route = evaluate_verify_candidate_authority(
        eager,
        expected_subject=eager_subject,
        candidate=VerifyCandidate.DIRECT_SPARSE_M3_QSA,
    )

    assert wrong_route.eligible is False
    assert wrong_route.reason is VerifyCandidateRefusal.ROUTE_MISMATCH
    assert wrong_identity.eligible is False
    assert wrong_identity.reason is VerifyCandidateRefusal.IDENTITY_MISMATCH
    assert eager_route.eligible is False
    assert eager_route.reason is VerifyCandidateRefusal.ROUTE_MISMATCH


def test_missing_authority_fails_closed():
    geometry = OracleGeometry(batch_size=1, verify_width=4)
    result = evaluate_verify_candidate_authority(
        None,
        expected_subject=_subject(geometry),
        candidate=VerifyCandidate.FIXED_M4_COMPILED,
    )
    assert result == (
        result.__class__(False, VerifyCandidateRefusal.MISSING_AUTHORITY)
    )
