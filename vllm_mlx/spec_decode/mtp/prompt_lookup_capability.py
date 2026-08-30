# SPDX-License-Identifier: Apache-2.0
"""Fail-closed capability contract for adaptive prompt-lookup decoding.

Prompt lookup verifies several proposed tokens in one target forward.  On a
hybrid target that forward mutates more than ordinary attention KV: recurrent
GDN state, and on Qwen4 also PLE and QSA auxiliary state.  A historical boolean
``mtp_prompt_lookup_supported`` cannot prove that a partial acceptance advances
every mutable surface to the accepted boundary, so the generator requires this
versioned descriptor as well.

The module is intentionally MLX-free.  Family adapters publish a descriptor
only after their target rollback and MTP-history synchronization paths have
been audited.  Missing, malformed, or incomplete descriptors refuse the route.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

PROMPT_LOOKUP_CAPABILITY_VERSION = 1

TARGET_KV = "target_kv"
MTP_KV = "mtp_kv"
GDN_RECURRENT = "gdn_recurrent"
PLE_HISTORY = "ple_history"
QSA_INDEX = "qsa_index"

KNOWN_MUTABLE_SURFACES = frozenset(
    {TARGET_KV, MTP_KV, GDN_RECURRENT, PLE_HISTORY, QSA_INDEX}
)
RECURRENT_SURFACES = frozenset({GDN_RECURRENT, PLE_HISTORY})


class PromptLookupRefusal(str, Enum):
    ELIGIBLE = "eligible"
    MODEL_DISABLED = "model_disabled"
    MISSING_DESCRIPTOR = "missing_descriptor"
    MALFORMED_DESCRIPTOR = "malformed_descriptor"
    MISSING_VERIFICATION = "missing_verification"
    VERIFICATION_IDENTITY_MISMATCH = "verification_identity_mismatch"
    MISSING_ACCEPTED_ADVANCE = "missing_accepted_advance"


@dataclass(frozen=True)
class PromptLookupEligibility:
    eligible: bool
    reason: PromptLookupRefusal
    mutable_state_surfaces: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PromptLookupVerificationIdentity:
    """Immutable identity of the exact state oracle that passed.

    This value is produced by audited code, never inferred from CLI flags,
    environment variables, or user configuration.  The runtime must publish an
    independently constructed identity and match every field before admission.
    """

    model_id: str
    model_revision: str
    runtime_commit: str
    cache_topology: str
    state_dtype: str
    verify_geometry: str
    oracle_version: str
    test_digest: str

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "runtime_commit",
            "cache_topology",
            "state_dtype",
            "verify_geometry",
            "oracle_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (
            not isinstance(self.test_digest, str)
            or len(self.test_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.test_digest
            )
        ):
            raise ValueError("test_digest must be a lowercase SHA-256 hex digest")


def make_prompt_lookup_capability(
    *,
    mutable_state_surfaces: tuple[str, ...],
    target_rollback_to_accepted: bool,
    mtp_advance_by_accepted: bool,
    recurrent_advance_by_accepted: bool,
    auxiliary_rollback_to_accepted: bool,
    verification_identity: PromptLookupVerificationIdentity | None = None,
) -> Mapping[str, Any]:
    """Build an immutable family descriptor with explicit booleans."""

    return MappingProxyType(
        {
            "protocol_version": PROMPT_LOOKUP_CAPABILITY_VERSION,
            "mutable_state_surfaces": tuple(mutable_state_surfaces),
            "target_rollback_to_accepted": target_rollback_to_accepted,
            "mtp_advance_by_accepted": mtp_advance_by_accepted,
            "recurrent_advance_by_accepted": recurrent_advance_by_accepted,
            "auxiliary_rollback_to_accepted": auxiliary_rollback_to_accepted,
            "verification_identity": verification_identity,
        }
    )


def evaluate_prompt_lookup_capability(model: Any) -> PromptLookupEligibility:
    """Return whether adaptive PLD may mutate this model's state safely."""

    if getattr(model, "mtp_prompt_lookup_supported", False) is not True:
        return PromptLookupEligibility(False, PromptLookupRefusal.MODEL_DISABLED)
    descriptor = getattr(model, "mtp_prompt_lookup_capability", None)
    if not isinstance(descriptor, Mapping):
        return PromptLookupEligibility(
            False, PromptLookupRefusal.MISSING_DESCRIPTOR
        )
    if descriptor.get("protocol_version") != PROMPT_LOOKUP_CAPABILITY_VERSION:
        return PromptLookupEligibility(
            False, PromptLookupRefusal.MALFORMED_DESCRIPTOR
        )
    verified_identity = descriptor.get("verification_identity")
    if not isinstance(verified_identity, PromptLookupVerificationIdentity):
        return PromptLookupEligibility(
            False, PromptLookupRefusal.MISSING_VERIFICATION
        )
    runtime_identity = getattr(model, "mtp_prompt_lookup_runtime_identity", None)
    if not isinstance(runtime_identity, PromptLookupVerificationIdentity):
        return PromptLookupEligibility(
            False, PromptLookupRefusal.MISSING_VERIFICATION
        )
    if runtime_identity != verified_identity:
        return PromptLookupEligibility(
            False, PromptLookupRefusal.VERIFICATION_IDENTITY_MISMATCH
        )
    raw_surfaces = descriptor.get("mutable_state_surfaces")
    if not isinstance(raw_surfaces, (tuple, list)) or not raw_surfaces:
        return PromptLookupEligibility(
            False, PromptLookupRefusal.MALFORMED_DESCRIPTOR
        )
    if any(not isinstance(surface, str) for surface in raw_surfaces):
        return PromptLookupEligibility(
            False, PromptLookupRefusal.MALFORMED_DESCRIPTOR
        )
    surfaces = frozenset(raw_surfaces)
    if (
        len(surfaces) != len(raw_surfaces)
        or not surfaces <= KNOWN_MUTABLE_SURFACES
        or not {TARGET_KV, MTP_KV} <= surfaces
    ):
        return PromptLookupEligibility(
            False, PromptLookupRefusal.MALFORMED_DESCRIPTOR
        )

    required_flags = (
        "target_rollback_to_accepted",
        "mtp_advance_by_accepted",
        "auxiliary_rollback_to_accepted",
    )
    if any(descriptor.get(name) is not True for name in required_flags):
        return PromptLookupEligibility(
            False,
            PromptLookupRefusal.MISSING_ACCEPTED_ADVANCE,
            surfaces,
        )
    if surfaces & RECURRENT_SURFACES and (
        descriptor.get("recurrent_advance_by_accepted") is not True
    ):
        return PromptLookupEligibility(
            False,
            PromptLookupRefusal.MISSING_ACCEPTED_ADVANCE,
            surfaces,
        )
    recurrent_flag = descriptor.get("recurrent_advance_by_accepted")
    if recurrent_flag is not True and recurrent_flag is not False:
        return PromptLookupEligibility(
            False, PromptLookupRefusal.MALFORMED_DESCRIPTOR, surfaces
        )
    return PromptLookupEligibility(True, PromptLookupRefusal.ELIGIBLE, surfaces)


__all__ = [
    "GDN_RECURRENT",
    "KNOWN_MUTABLE_SURFACES",
    "MTP_KV",
    "PLE_HISTORY",
    "PROMPT_LOOKUP_CAPABILITY_VERSION",
    "PromptLookupEligibility",
    "PromptLookupRefusal",
    "PromptLookupVerificationIdentity",
    "QSA_INDEX",
    "TARGET_KV",
    "evaluate_prompt_lookup_capability",
    "make_prompt_lookup_capability",
]
