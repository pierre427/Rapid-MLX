# SPDX-License-Identifier: Apache-2.0
"""Fail-closed receipt consumer for Qwen4 verify candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .prompt_lookup_attestation import (
    DIRECT_SPARSE_M3_QSA_ROUTE,
    FIXED_M4_COMPILED_VERIFY_ROUTE,
    AttestationSubject,
    AttestationValidationError,
    OracleExecutionKind,
    TrustedPromptLookupReceipt,
    verify_trusted_prompt_lookup_receipt,
)


class VerifyCandidate(str, Enum):
    FIXED_M4_COMPILED = "fixed_m4_compiled"
    DIRECT_SPARSE_M3_QSA = "direct_sparse_m3_qsa"


class VerifyCandidateRefusal(str, Enum):
    ELIGIBLE = "eligible"
    MISSING_AUTHORITY = "missing_authority"
    IDENTITY_MISMATCH = "identity_mismatch"
    ROUTE_MISMATCH = "route_mismatch"
    EXECUTION_KIND_MISMATCH = "execution_kind_mismatch"
    GEOMETRY_MISMATCH = "geometry_mismatch"


@dataclass(frozen=True)
class VerifyCandidateEligibility:
    eligible: bool
    reason: VerifyCandidateRefusal
    evidence_digest: str | None = None


_CONTRACTS = {
    VerifyCandidate.FIXED_M4_COMPILED: (
        FIXED_M4_COMPILED_VERIFY_ROUTE,
        OracleExecutionKind.COMPILED_CANDIDATE,
        4,
    ),
    VerifyCandidate.DIRECT_SPARSE_M3_QSA: (
        DIRECT_SPARSE_M3_QSA_ROUTE,
        OracleExecutionKind.SPARSE_CANDIDATE,
        3,
    ),
}


def evaluate_verify_candidate_authority(
    authority: TrustedPromptLookupReceipt | None,
    *,
    expected_subject: AttestationSubject,
    candidate: VerifyCandidate,
) -> VerifyCandidateEligibility:
    """Require one sealed, exact-identity receipt for the selected route."""

    if not isinstance(candidate, VerifyCandidate):
        raise TypeError("candidate must be a VerifyCandidate")
    if not isinstance(expected_subject, AttestationSubject):
        raise TypeError("expected_subject must be an AttestationSubject")
    if authority is None:
        return VerifyCandidateEligibility(
            False, VerifyCandidateRefusal.MISSING_AUTHORITY
        )
    route, execution_kind, verify_width = _CONTRACTS[candidate]
    try:
        receipt = verify_trusted_prompt_lookup_receipt(
            authority,
            expected_subject=expected_subject,
            expected_route=route,
        )
    except AttestationValidationError as exc:
        reason = (
            VerifyCandidateRefusal.ROUTE_MISMATCH
            if "route" in str(exc)
            else VerifyCandidateRefusal.IDENTITY_MISMATCH
        )
        return VerifyCandidateEligibility(False, reason)
    if receipt.execution_kind is not execution_kind:
        return VerifyCandidateEligibility(
            False, VerifyCandidateRefusal.EXECUTION_KIND_MISMATCH
        )
    if receipt.batch_size != 1 or receipt.verify_width != verify_width:
        return VerifyCandidateEligibility(
            False, VerifyCandidateRefusal.GEOMETRY_MISMATCH
        )
    return VerifyCandidateEligibility(
        True,
        VerifyCandidateRefusal.ELIGIBLE,
        evidence_digest=receipt.evidence_digest,
    )


__all__ = [
    "VerifyCandidate",
    "VerifyCandidateEligibility",
    "VerifyCandidateRefusal",
    "evaluate_verify_candidate_authority",
]
