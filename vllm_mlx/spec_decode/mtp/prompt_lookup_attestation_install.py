# SPDX-License-Identifier: Apache-2.0
"""Install oracle-attested hybrid prompt-lookup state capability.

This module is deliberately narrower than the oracle collector.  It accepts
only immutable runtime facts produced by audited runtime code, derives the
same canonical subject identity used by the oracle, and looks that identity up
in an in-package registry of sealed receipts.  Files, JSON payloads, CLI
values, and environment variables are not authority and are never read here.

Installation attests the hybrid transactional-accept state contract only.  It
does not attest or enable a request-scoped adaptive-PLD executor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .prompt_lookup_attestation import (
    HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    AttestationSubject,
    AttestationValidationError,
    TrustedPromptLookupReceipt,
    verify_trusted_prompt_lookup_receipt,
)
from .prompt_lookup_capability import (
    GDN_RECURRENT,
    MTP_KV,
    PLE_HISTORY,
    QSA_INDEX,
    TARGET_KV,
    make_prompt_lookup_capability,
)

TRUSTED_TRANSACTIONAL_ROUTE = HYBRID_TRANSACTIONAL_ACCEPT_ROUTE
QWEN4_EXP_MODEL_ID = "qwen4_exp"

# Authorities are source-controlled package data.  A production entry may be
# added only as a sealed object returned by issue_prompt_lookup_attestation()
# from complete production-Metal oracle evidence.  There is intentionally no
# path, payload, environment, config, or caller-supplied registry parameter.
_TRUSTED_PROMPT_LOOKUP_RECEIPTS: tuple[TrustedPromptLookupReceipt, ...] = ()


class PromptLookupInstallReason(str, Enum):
    INSTALLED = "installed"
    OPERATOR_DISABLED = "operator_disabled"
    UNSUPPORTED_MODEL = "unsupported_model"
    NO_TRUSTED_RECEIPT = "no_trusted_receipt"
    REGISTRY_INVALID = "trusted_registry_invalid"
    RUNTIME_IDENTITY_MISMATCH = "runtime_identity_mismatch"
    AMBIGUOUS_RECEIPT = "ambiguous_trusted_receipt"
    INSTALL_FAILED = "install_failed"


@dataclass(frozen=True)
class RuntimeAttestationIdentity:
    """Canonical identity independently observed by audited runtime code."""

    model_id: str
    model_revision: str
    runtime_commit: str
    cache_topology: str
    state_dtype: str
    verify_geometry: str
    oracle_version: str

    def __post_init__(self) -> None:
        # Reuse the evidence contract's validation and spelling.
        self.to_subject()

    def to_subject(self) -> AttestationSubject:
        return AttestationSubject(
            model_id=self.model_id,
            model_revision=self.model_revision,
            runtime_commit=self.runtime_commit,
            cache_topology=self.cache_topology,
            state_dtype=self.state_dtype,
            verify_geometry=self.verify_geometry,
            oracle_version=self.oracle_version,
        )


@dataclass(frozen=True)
class PromptLookupInstallResult:
    installed: bool
    reason: PromptLookupInstallReason
    evidence_digest: str | None = None


def build_runtime_attestation_identity(
    *,
    model_id: str,
    checkpoint_config_sha256: str,
    checkpoint_index_sha256: str,
    checkpoint_weight_manifest_sha256: str,
    rapid_runtime_commit: str,
    mlx_lm_runtime_commit: str,
    layer_types: tuple[str, ...],
    ple_layer_ids: tuple[int, ...],
    mtp_layer_types: tuple[str, ...],
    cache_classes: tuple[str, ...],
    state_dtypes: tuple[str, ...],
    verify_geometry: str,
    oracle_version: str,
) -> RuntimeAttestationIdentity:
    """Build identity from facts observed by the audited runtime harness.

    The three checkpoint digests bind config, index, and the referenced weight
    manifest without hashing multi-gigabyte shards at boot.  Both source-tree
    commits are bound explicitly.  Topology and dtype are canonical hashes of
    live, ordered observations, not configuration promises.
    """

    if model_id != QWEN4_EXP_MODEL_ID:
        raise ValueError("model_id must be qwen4_exp")
    for name, digest in (
        ("checkpoint_config_sha256", checkpoint_config_sha256),
        ("checkpoint_index_sha256", checkpoint_index_sha256),
        ("checkpoint_weight_manifest_sha256", checkpoint_weight_manifest_sha256),
    ):
        _require_sha256(name, digest)
    for name, commit in (
        ("rapid_runtime_commit", rapid_runtime_commit),
        ("mlx_lm_runtime_commit", mlx_lm_runtime_commit),
    ):
        _require_git_commit(name, commit)
    _require_string_tuple("layer_types", layer_types)
    _require_string_tuple("mtp_layer_types", mtp_layer_types)
    _require_string_tuple("cache_classes", cache_classes)
    _require_string_tuple("state_dtypes", state_dtypes)
    if not isinstance(ple_layer_ids, tuple) or any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        for layer in ple_layer_ids
    ):
        raise ValueError("ple_layer_ids must contain non-negative integers")

    model_revision = _canonical_digest(
        {
            "config_sha256": checkpoint_config_sha256,
            "index_sha256": checkpoint_index_sha256,
            "weight_manifest_sha256": checkpoint_weight_manifest_sha256,
        }
    )
    runtime_commit = (
        f"rapid={rapid_runtime_commit};mlx-lm={mlx_lm_runtime_commit}"
    )
    cache_topology = _canonical_digest(
        {
            "layer_types": layer_types,
            "ple_layer_ids": ple_layer_ids,
            "mtp_layer_types": mtp_layer_types,
            "cache_classes": cache_classes,
        }
    )
    state_dtype = _canonical_digest({"state_dtypes": state_dtypes})
    return RuntimeAttestationIdentity(
        model_id=model_id,
        model_revision=f"sha256:{model_revision}",
        runtime_commit=runtime_commit,
        cache_topology=f"sha256:{cache_topology}",
        state_dtype=f"sha256:{state_dtype}",
        verify_geometry=verify_geometry,
        oracle_version=oracle_version,
    )


def install_trusted_prompt_lookup_capability(
    model: Any,
    runtime_identity: RuntimeAttestationIdentity,
    *,
    disabled: bool = False,
) -> PromptLookupInstallResult:
    """Install state capability only for one exact sealed receipt match.

    ``disabled`` is the sole operator control and can only remove eligibility.
    There is intentionally no enable flag and no receipt/registry argument.
    """

    if not isinstance(disabled, bool):
        raise TypeError("disabled must be a bool")
    if not isinstance(runtime_identity, RuntimeAttestationIdentity):
        raise TypeError("runtime_identity must be RuntimeAttestationIdentity")

    target = _resolve_qwen4_target(model)
    if target is None:
        _disable_prompt_lookup(model)
        return PromptLookupInstallResult(
            False, PromptLookupInstallReason.UNSUPPORTED_MODEL
        )
    if disabled:
        _disable_prompt_lookup(target)
        return PromptLookupInstallResult(
            False, PromptLookupInstallReason.OPERATOR_DISABLED
        )
    if runtime_identity.model_id != QWEN4_EXP_MODEL_ID:
        _disable_prompt_lookup(target)
        return PromptLookupInstallResult(
            False, PromptLookupInstallReason.RUNTIME_IDENTITY_MISMATCH
        )

    authorities = _TRUSTED_PROMPT_LOOKUP_RECEIPTS
    if not isinstance(authorities, tuple) or any(
        not isinstance(authority, TrustedPromptLookupReceipt)
        for authority in authorities
    ):
        _disable_prompt_lookup(target)
        return PromptLookupInstallResult(
            False, PromptLookupInstallReason.REGISTRY_INVALID
        )
    if not authorities:
        _disable_prompt_lookup(target)
        return PromptLookupInstallResult(
            False, PromptLookupInstallReason.NO_TRUSTED_RECEIPT
        )

    subject = runtime_identity.to_subject()
    matches = []
    for authority in authorities:
        try:
            receipt = verify_trusted_prompt_lookup_receipt(
                authority,
                expected_subject=subject,
                expected_route=TRUSTED_TRANSACTIONAL_ROUTE,
            )
        except AttestationValidationError:
            continue
        matches.append(receipt)
    if not matches:
        _disable_prompt_lookup(target)
        return PromptLookupInstallResult(
            False, PromptLookupInstallReason.RUNTIME_IDENTITY_MISMATCH
        )
    if len(matches) != 1:
        _disable_prompt_lookup(target)
        return PromptLookupInstallResult(
            False, PromptLookupInstallReason.AMBIGUOUS_RECEIPT
        )

    receipt = matches[0]
    capability = make_prompt_lookup_capability(
        mutable_state_surfaces=(
            TARGET_KV,
            MTP_KV,
            GDN_RECURRENT,
            PLE_HISTORY,
            QSA_INDEX,
        ),
        target_rollback_to_accepted=True,
        mtp_advance_by_accepted=True,
        recurrent_advance_by_accepted=True,
        auxiliary_rollback_to_accepted=True,
        verification_identity=receipt.identity,
    )
    previous = {
        name: getattr(target, name, _MISSING)
        for name in (
            "mtp_prompt_lookup_runtime_identity",
            "mtp_prompt_lookup_capability",
            "mtp_prompt_lookup_supported",
        )
    }
    try:
        # Publish the boolean last so readers cannot observe a partially
        # installed positive capability.
        target.mtp_prompt_lookup_runtime_identity = receipt.identity
        target.mtp_prompt_lookup_capability = capability
        target.mtp_prompt_lookup_supported = True
    except Exception:  # noqa: BLE001 - optional capability must fail closed
        _restore_attributes(target, previous)
        _disable_prompt_lookup(target)
        return PromptLookupInstallResult(
            False, PromptLookupInstallReason.INSTALL_FAILED
        )
    return PromptLookupInstallResult(
        True,
        PromptLookupInstallReason.INSTALLED,
        evidence_digest=receipt.evidence_digest,
    )


_MISSING = object()


def _resolve_qwen4_target(model: Any) -> Any | None:
    if getattr(model, "model_type", None) == QWEN4_EXP_MODEL_ID:
        return model
    inner = getattr(model, "language_model", None)
    if getattr(inner, "model_type", None) in {
        QWEN4_EXP_MODEL_ID,
        "qwen4_exp_text",
    }:
        return inner
    return None


def _disable_prompt_lookup(model: Any) -> None:
    try:
        model.mtp_prompt_lookup_supported = False
    except Exception:  # noqa: BLE001 - best-effort denial on immutable object
        pass


def _restore_attributes(model: Any, previous: dict[str, Any]) -> None:
    for name, value in previous.items():
        try:
            if value is _MISSING:
                delattr(model, name)
            else:
                setattr(model, name, value)
        except (AttributeError, TypeError):
            pass


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _require_git_commit(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase Git commit")


def _require_string_tuple(name: str, value: tuple[str, ...]) -> None:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty tuple of strings")


__all__ = [
    "PromptLookupInstallReason",
    "PromptLookupInstallResult",
    "QWEN4_EXP_MODEL_ID",
    "RuntimeAttestationIdentity",
    "TRUSTED_TRANSACTIONAL_ROUTE",
    "build_runtime_attestation_identity",
    "install_trusted_prompt_lookup_capability",
]
