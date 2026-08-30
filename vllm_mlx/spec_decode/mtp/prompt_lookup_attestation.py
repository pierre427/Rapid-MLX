# SPDX-License-Identifier: Apache-2.0
"""Pure-Python evidence contract for hybrid prompt-lookup attestation.

The collector that runs the production Metal oracle lives outside this module.
It converts every compared value to :class:`RawBitValue`, records the complete
accepted-prefix/ragged/continuation matrix, and submits that immutable evidence
here.  Receipt issuance is fail-closed and deterministic; it cannot run an
oracle, infer a capability from config, or import MLX.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .prompt_lookup_capability import PromptLookupVerificationIdentity

ATTESTATION_SCHEMA_VERSION = 1
ATTESTATION_ISSUER = "rapid-mlx-production-metal-oracle-v1"
HYBRID_TRANSACTIONAL_ACCEPT_ROUTE = "hybrid_transactional_accept_v1"
_TRUSTED_RECEIPT_SEAL = object()

REQUIRED_STATE_SURFACES = frozenset(
    {
        "verify_logits",
        "target_cache_kv",
        "mtp_cache_kv",
        "seed_hidden",
        "gdn_conv_tail",
        "gdn_matrix",
        "gdn_host_metadata",
        "gdn_rollback_stack",
        "ple_conv_state",
        "ple_token_history",
        "ple_atomic_rollback",
        "qsa_keys",
        "qsa_values",
        "qsa_offset",
        "qsa_index_keys",
        "qsa_pooled_keys",
        "qsa_pooled_ratio",
        "qsa_share_topk_flag",
        "qsa_shared_topk",
        "batch_membership_epoch",
        "batch_proposal_state",
        "proposal_transaction_metadata",
        "proposal_outputs",
        "lane_identity_metadata",
        "lane_decode_state",
        "lane_pending_state",
        "lane_statistics",
        "lane_rng_aliasing",
        "lane_rng_state",
    }
)


class OraclePhase(str, Enum):
    POST_TRIM = "post_trim"
    CONTINUATION = "continuation"


class OracleExecutionKind(str, Enum):
    EAGER_TRANSACTION = "eager_transaction"
    COMPILED_CANDIDATE = "compiled_candidate"


@dataclass(frozen=True)
class RawBitValue:
    """One value represented by storage dtype, shape, and exact bytes."""

    is_none: bool
    dtype: str
    shape: tuple[int, ...]
    raw_bits: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.is_none, bool):
            raise TypeError("is_none must be a bool")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError("dtype must be a non-empty string")
        if not isinstance(self.shape, tuple) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in self.shape
        ):
            raise ValueError("shape must contain non-negative integer dimensions")
        if not isinstance(self.raw_bits, bytes):
            raise TypeError("raw_bits must be bytes")
        if self.is_none and (
            self.dtype != "none" or self.shape != () or self.raw_bits != b""
        ):
            raise ValueError("None must use dtype='none', shape=(), and empty bits")
        if not self.is_none and self.dtype == "none":
            raise ValueError("non-None values cannot use dtype='none'")

    @classmethod
    def none(cls) -> RawBitValue:
        return cls(True, "none", (), b"")

    @classmethod
    def from_buffer(
        cls,
        *,
        dtype: str,
        shape: tuple[int, ...],
        raw_bits: bytes | bytearray | memoryview,
    ) -> RawBitValue:
        try:
            bits = bytes(raw_bits)
        except (TypeError, ValueError) as exc:
            raise TypeError("raw_bits must support the buffer protocol") from exc
        return cls(False, dtype, shape, bits)

    @classmethod
    def from_canonical_value(cls, *, dtype: str, value: Any) -> RawBitValue:
        """Encode JSON-compatible host metadata deterministically."""

        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        except (TypeError, ValueError) as exc:
            raise ValueError("value must contain canonical JSON data") from exc
        return cls(False, dtype, (), encoded)


@dataclass(frozen=True)
class RawBitComparison:
    equal: bool
    reason: str


def compare_raw_bits(
    reference: RawBitValue,
    candidate: RawBitValue,
) -> RawBitComparison:
    """Compare None identity, dtype, shape, then payload storage bits."""

    if not isinstance(reference, RawBitValue) or not isinstance(
        candidate, RawBitValue
    ):
        raise TypeError("reference and candidate must be RawBitValue instances")
    if reference.is_none != candidate.is_none:
        return RawBitComparison(False, "none_identity")
    if reference.dtype != candidate.dtype:
        return RawBitComparison(False, "dtype")
    if reference.shape != candidate.shape:
        return RawBitComparison(False, "shape")
    if reference.raw_bits != candidate.raw_bits:
        return RawBitComparison(False, "raw_bits")
    return RawBitComparison(True, "equal")


@dataclass(frozen=True)
class SurfaceEvidence:
    surface: str
    reference: RawBitValue
    candidate: RawBitValue

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface.strip():
            raise ValueError("surface must be a non-empty string")

    @property
    def comparison(self) -> RawBitComparison:
        return compare_raw_bits(self.reference, self.candidate)


@dataclass(frozen=True, order=True)
class OracleCaseKey:
    accepted_prefix: int
    ragged_drops: tuple[int, ...]
    phase: OraclePhase

    def __post_init__(self) -> None:
        if (
            isinstance(self.accepted_prefix, bool)
            or not isinstance(self.accepted_prefix, int)
            or self.accepted_prefix < 0
        ):
            raise ValueError("accepted_prefix must be a non-negative integer")
        _validate_drop_vector(self.ragged_drops, "ragged_drops")
        if not isinstance(self.phase, OraclePhase):
            raise TypeError("phase must be an OraclePhase")


@dataclass(frozen=True)
class OracleCaseEvidence:
    key: OracleCaseKey
    surfaces: tuple[SurfaceEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, OracleCaseKey):
            raise TypeError("key must be an OracleCaseKey")
        if not isinstance(self.surfaces, tuple) or not self.surfaces:
            raise ValueError("surfaces must be a non-empty tuple")
        if any(not isinstance(surface, SurfaceEvidence) for surface in self.surfaces):
            raise TypeError("surfaces must contain SurfaceEvidence instances")


@dataclass(frozen=True)
class OracleGeometry:
    batch_size: int
    verify_width: int
    ragged_drop_vectors: tuple[tuple[int, ...], ...]
    ragged_domain_complete: bool
    ragged_domain_definition: str

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("verify_width", self.verify_width),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.ragged_drop_vectors, tuple) or not (
            self.ragged_drop_vectors
        ):
            raise ValueError("ragged_drop_vectors must be a non-empty tuple")
        if len(set(self.ragged_drop_vectors)) != len(self.ragged_drop_vectors):
            raise ValueError("ragged_drop_vectors must be unique")
        for drops in self.ragged_drop_vectors:
            _validate_drop_vector(drops, "ragged_drop_vectors")
            if len(drops) != self.batch_size:
                raise ValueError("each drop vector must match batch_size")
            if any(drop > self.verify_width for drop in drops):
                raise ValueError("ragged drops cannot exceed verify_width")
        if (0,) * self.batch_size not in self.ragged_drop_vectors:
            raise ValueError("ragged_drop_vectors must include the all-zero case")
        if not isinstance(self.ragged_domain_complete, bool):
            raise TypeError("ragged_domain_complete must be a bool")
        if (
            not isinstance(self.ragged_domain_definition, str)
            or not self.ragged_domain_definition.strip()
        ):
            raise ValueError("ragged_domain_definition must be non-empty")

    @property
    def fingerprint(self) -> str:
        drops = ";".join(
            ",".join(str(drop) for drop in vector)
            for vector in self.ragged_drop_vectors
        )
        return (
            f"batch={self.batch_size},width={self.verify_width},"
            f"ragged={drops},domain={self.ragged_domain_definition}"
        )

    @property
    def required_case_keys(self) -> frozenset[OracleCaseKey]:
        return frozenset(
            OracleCaseKey(m, drops, phase)
            for m in range(self.verify_width + 1)
            for drops in self.ragged_drop_vectors
            for phase in OraclePhase
        )


@dataclass(frozen=True)
class AttestationSubject:
    model_id: str
    model_revision: str
    runtime_commit: str
    cache_topology: str
    state_dtype: str
    verify_geometry: str
    oracle_version: str

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

    def verification_identity(
        self, test_digest: str
    ) -> PromptLookupVerificationIdentity:
        return PromptLookupVerificationIdentity(
            model_id=self.model_id,
            model_revision=self.model_revision,
            runtime_commit=self.runtime_commit,
            cache_topology=self.cache_topology,
            state_dtype=self.state_dtype,
            verify_geometry=self.verify_geometry,
            oracle_version=self.oracle_version,
            test_digest=test_digest,
        )


@dataclass(frozen=True)
class RouteEngagementEvidence:
    route_name: str
    execution_kind: OracleExecutionKind
    compiled_candidate: bool
    candidate_invocations: int
    fallback_count: int
    compile_count: int
    warmup_compile_count: int
    post_warmup_recompile_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.route_name, str) or not self.route_name.strip():
            raise ValueError("route_name must be a non-empty string")
        if not isinstance(self.execution_kind, OracleExecutionKind):
            raise TypeError("execution_kind must be an OracleExecutionKind")
        if not isinstance(self.compiled_candidate, bool):
            raise TypeError("compiled_candidate must be a bool")
        for name in (
            "candidate_invocations",
            "fallback_count",
            "compile_count",
            "warmup_compile_count",
            "post_warmup_recompile_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class PromptLookupOracleEvidence:
    subject: AttestationSubject
    geometry: OracleGeometry
    production_metal: bool
    route: RouteEngagementEvidence
    cases: tuple[OracleCaseEvidence, ...]
    schema_version: int = ATTESTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.subject, AttestationSubject):
            raise TypeError("subject must be an AttestationSubject")
        if not isinstance(self.geometry, OracleGeometry):
            raise TypeError("geometry must be an OracleGeometry")
        if not isinstance(self.production_metal, bool):
            raise TypeError("production_metal must be a bool")
        if not isinstance(self.route, RouteEngagementEvidence):
            raise TypeError("route must be RouteEngagementEvidence")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError("cases must be a non-empty tuple")
        if any(not isinstance(case, OracleCaseEvidence) for case in self.cases):
            raise TypeError("cases must contain OracleCaseEvidence instances")


@dataclass(frozen=True)
class AttestationValidation:
    eligible: bool
    reasons: tuple[str, ...]
    evidence_digest: str


@dataclass(frozen=True)
class PromptLookupAttestationReceipt:
    identity: PromptLookupVerificationIdentity
    evidence_digest: str
    schema_version: int
    case_count: int
    state_surfaces: tuple[str, ...]
    route_name: str
    execution_kind: OracleExecutionKind
    compiled_candidate: bool
    production_metal: bool
    accepted_prefixes: tuple[int, ...]
    ragged_drop_vectors: tuple[tuple[int, ...], ...]
    continuation_covered: bool
    fallback_count: int
    post_warmup_recompile_count: int
    compile_count: int
    warmup_compile_count: int


@dataclass(frozen=True)
class TrustedPromptLookupReceipt:
    """Authority wrapper obtainable only from a successful oracle issuance.

    Parsed JSON intentionally produces only :class:`PromptLookupAttestationReceipt`.
    Runtime loaders must additionally require this sealed type from their
    in-package registry; an arbitrary path or caller-supplied payload therefore
    cannot become positive authority.
    """

    receipt: PromptLookupAttestationReceipt
    issuer: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _TRUSTED_RECEIPT_SEAL:
            raise ValueError("trusted receipt seal is invalid")
        if self.issuer != ATTESTATION_ISSUER:
            raise ValueError("trusted receipt issuer is invalid")
        if not isinstance(self.receipt, PromptLookupAttestationReceipt):
            raise TypeError("receipt must be PromptLookupAttestationReceipt")
        _validate_receipt(self.receipt)

    @property
    def identity(self) -> PromptLookupVerificationIdentity:
        return self.receipt.identity

    @property
    def evidence_digest(self) -> str:
        return self.receipt.evidence_digest


class AttestationValidationError(ValueError):
    pass


def validate_prompt_lookup_oracle(
    evidence: PromptLookupOracleEvidence,
    *,
    expected_subject: AttestationSubject,
    expected_geometry: OracleGeometry,
    expected_route: str,
) -> AttestationValidation:
    """Validate complete oracle evidence without issuing authority."""

    if not isinstance(evidence, PromptLookupOracleEvidence):
        raise TypeError("evidence must be PromptLookupOracleEvidence")
    reasons: list[str] = []
    if evidence.schema_version != ATTESTATION_SCHEMA_VERSION:
        reasons.append("schema_version")
    if evidence.subject != expected_subject:
        reasons.append("identity_mismatch")
    if evidence.geometry != expected_geometry:
        reasons.append("geometry_mismatch")
    if evidence.subject.verify_geometry != evidence.geometry.fingerprint:
        reasons.append("geometry_identity_mismatch")
    if evidence.production_metal is not True:
        reasons.append("not_production_metal")
    if evidence.geometry.ragged_domain_complete is not True:
        reasons.append("ragged_domain_incomplete")
    if expected_route != HYBRID_TRANSACTIONAL_ACCEPT_ROUTE:
        reasons.append("unsupported_attestation_route")
    if evidence.route.route_name != expected_route:
        reasons.append("route_mismatch")
    if evidence.route.candidate_invocations < len(
        expected_geometry.required_case_keys
    ):
        reasons.append("candidate_route_not_observed")
    if evidence.route.fallback_count != 0:
        reasons.append("fallback_observed")
    if evidence.route.execution_kind is OracleExecutionKind.EAGER_TRANSACTION:
        if evidence.route.compiled_candidate is not False:
            reasons.append("execution_kind_mismatch")
        if any(
            count != 0
            for count in (
                evidence.route.compile_count,
                evidence.route.warmup_compile_count,
                evidence.route.post_warmup_recompile_count,
            )
        ):
            reasons.append("eager_route_has_compile_activity")
    elif evidence.route.execution_kind is OracleExecutionKind.COMPILED_CANDIDATE:
        if evidence.route.compiled_candidate is not True:
            reasons.append("execution_kind_mismatch")
        if evidence.route.compile_count < 1:
            reasons.append("candidate_never_compiled")
        if evidence.route.warmup_compile_count < 1:
            reasons.append("candidate_not_compiled_during_warmup")
        if evidence.route.warmup_compile_count != evidence.route.compile_count:
            reasons.append("compile_outside_warmup")
        if evidence.route.post_warmup_recompile_count != 0:
            reasons.append("post_warmup_recompile_observed")

    case_by_key: dict[OracleCaseKey, OracleCaseEvidence] = {}
    duplicate_keys: set[OracleCaseKey] = set()
    for case in evidence.cases:
        if case.key in case_by_key:
            duplicate_keys.add(case.key)
        case_by_key[case.key] = case
    if duplicate_keys:
        reasons.append("duplicate_cases")
    required_cases = expected_geometry.required_case_keys
    actual_cases = frozenset(case_by_key)
    if actual_cases != required_cases:
        reasons.append("incomplete_case_matrix")

    for case in evidence.cases:
        surface_by_name: dict[str, SurfaceEvidence] = {}
        duplicate_surfaces: set[str] = set()
        for surface in case.surfaces:
            if surface.surface in surface_by_name:
                duplicate_surfaces.add(surface.surface)
            surface_by_name[surface.surface] = surface
        if duplicate_surfaces:
            reasons.append("duplicate_state_surfaces")
        if frozenset(surface_by_name) != REQUIRED_STATE_SURFACES:
            reasons.append("incomplete_state_surfaces")
        if any(not surface.comparison.equal for surface in case.surfaces):
            reasons.append("raw_bit_mismatch")

    digest = _evidence_digest(evidence)
    return AttestationValidation(not reasons, tuple(dict.fromkeys(reasons)), digest)


def issue_prompt_lookup_attestation(
    evidence: PromptLookupOracleEvidence,
    *,
    expected_subject: AttestationSubject,
    expected_geometry: OracleGeometry,
    expected_route: str,
) -> TrustedPromptLookupReceipt:
    """Issue an immutable receipt only for complete exact Metal evidence."""

    validation = validate_prompt_lookup_oracle(
        evidence,
        expected_subject=expected_subject,
        expected_geometry=expected_geometry,
        expected_route=expected_route,
    )
    if not validation.eligible:
        raise AttestationValidationError(", ".join(validation.reasons))
    receipt = PromptLookupAttestationReceipt(
        identity=evidence.subject.verification_identity(
            validation.evidence_digest
        ),
        evidence_digest=validation.evidence_digest,
        schema_version=ATTESTATION_SCHEMA_VERSION,
        case_count=len(evidence.cases),
        state_surfaces=tuple(sorted(REQUIRED_STATE_SURFACES)),
        route_name=evidence.route.route_name,
        execution_kind=evidence.route.execution_kind,
        compiled_candidate=evidence.route.compiled_candidate,
        production_metal=True,
        accepted_prefixes=tuple(range(evidence.geometry.verify_width + 1)),
        ragged_drop_vectors=evidence.geometry.ragged_drop_vectors,
        continuation_covered=True,
        fallback_count=0,
        post_warmup_recompile_count=0,
        compile_count=evidence.route.compile_count,
        warmup_compile_count=evidence.route.warmup_compile_count,
    )
    _validate_receipt(receipt)
    return TrustedPromptLookupReceipt(
        receipt=receipt,
        issuer=ATTESTATION_ISSUER,
        _seal=_TRUSTED_RECEIPT_SEAL,
    )


def verify_trusted_prompt_lookup_receipt(
    authority: TrustedPromptLookupReceipt,
    *,
    expected_subject: AttestationSubject,
    expected_route: str,
) -> PromptLookupAttestationReceipt:
    """Verify sealed authority and exact runtime subject before installation."""

    if not isinstance(authority, TrustedPromptLookupReceipt):
        raise AttestationValidationError("receipt is not trusted authority")
    if authority._seal is not _TRUSTED_RECEIPT_SEAL:
        raise AttestationValidationError("trusted receipt seal is invalid")
    receipt = authority.receipt
    try:
        _validate_receipt(receipt)
    except (TypeError, ValueError) as exc:
        raise AttestationValidationError(str(exc)) from exc
    identity = receipt.identity
    actual_subject = AttestationSubject(
        model_id=identity.model_id,
        model_revision=identity.model_revision,
        runtime_commit=identity.runtime_commit,
        cache_topology=identity.cache_topology,
        state_dtype=identity.state_dtype,
        verify_geometry=identity.verify_geometry,
        oracle_version=identity.oracle_version,
    )
    if actual_subject != expected_subject:
        raise AttestationValidationError("trusted receipt identity mismatch")
    if receipt.route_name != expected_route:
        raise AttestationValidationError("trusted receipt route mismatch")
    if expected_route != HYBRID_TRANSACTIONAL_ACCEPT_ROUTE:
        raise AttestationValidationError("trusted receipt route is unsupported")
    return receipt


def prompt_lookup_attestation_receipt_to_payload(
    receipt: PromptLookupAttestationReceipt,
) -> dict[str, Any]:
    """Return the canonical JSON-compatible receipt payload."""

    if not isinstance(receipt, PromptLookupAttestationReceipt):
        raise TypeError("receipt must be PromptLookupAttestationReceipt")
    return {
        "identity": receipt.identity.__dict__,
        "evidence_digest": receipt.evidence_digest,
        "schema_version": receipt.schema_version,
        "case_count": receipt.case_count,
        "state_surfaces": list(receipt.state_surfaces),
        "route_name": receipt.route_name,
        "execution_kind": receipt.execution_kind.value,
        "compiled_candidate": receipt.compiled_candidate,
        "production_metal": receipt.production_metal,
        "accepted_prefixes": list(receipt.accepted_prefixes),
        "ragged_drop_vectors": [list(v) for v in receipt.ragged_drop_vectors],
        "continuation_covered": receipt.continuation_covered,
        "fallback_count": receipt.fallback_count,
        "post_warmup_recompile_count": receipt.post_warmup_recompile_count,
        "compile_count": receipt.compile_count,
        "warmup_compile_count": receipt.warmup_compile_count,
    }


def parse_prompt_lookup_attestation_receipt(
    payload: Mapping[str, Any],
) -> PromptLookupAttestationReceipt:
    """Strictly parse a receipt; provenance validation belongs to the loader."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    expected_keys = {
        "identity",
        "evidence_digest",
        "schema_version",
        "case_count",
        "state_surfaces",
        "route_name",
        "execution_kind",
        "compiled_candidate",
        "production_metal",
        "accepted_prefixes",
        "ragged_drop_vectors",
        "continuation_covered",
        "fallback_count",
        "post_warmup_recompile_count",
        "compile_count",
        "warmup_compile_count",
    }
    if set(payload) != expected_keys:
        raise ValueError("receipt keys do not match the attestation schema")
    identity_payload = payload["identity"]
    if not isinstance(identity_payload, Mapping):
        raise ValueError("receipt identity must be a mapping")
    try:
        identity = PromptLookupVerificationIdentity(**dict(identity_payload))
        surfaces = tuple(payload["state_surfaces"])
        prefixes = tuple(payload["accepted_prefixes"])
        drops = tuple(tuple(vector) for vector in payload["ragged_drop_vectors"])
        receipt = PromptLookupAttestationReceipt(
            identity=identity,
            evidence_digest=payload["evidence_digest"],
            schema_version=payload["schema_version"],
            case_count=payload["case_count"],
            state_surfaces=surfaces,
            route_name=payload["route_name"],
            execution_kind=OracleExecutionKind(payload["execution_kind"]),
            compiled_candidate=payload["compiled_candidate"],
            production_metal=payload["production_metal"],
            accepted_prefixes=prefixes,
            ragged_drop_vectors=drops,
            continuation_covered=payload["continuation_covered"],
            fallback_count=payload["fallback_count"],
            post_warmup_recompile_count=payload[
                "post_warmup_recompile_count"
            ],
            compile_count=payload["compile_count"],
            warmup_compile_count=payload["warmup_compile_count"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed prompt-lookup attestation receipt") from exc
    _validate_receipt(receipt)
    return receipt


def _validate_receipt(receipt: PromptLookupAttestationReceipt) -> None:
    if receipt.schema_version != ATTESTATION_SCHEMA_VERSION:
        raise ValueError("unsupported attestation receipt schema")
    if receipt.evidence_digest != receipt.identity.test_digest:
        raise ValueError("receipt digest does not match verification identity")
    if (
        not isinstance(receipt.case_count, int)
        or isinstance(receipt.case_count, bool)
        or receipt.case_count < 1
    ):
        raise ValueError("receipt case_count must be positive")
    if frozenset(receipt.state_surfaces) != REQUIRED_STATE_SURFACES or len(
        receipt.state_surfaces
    ) != len(REQUIRED_STATE_SURFACES):
        raise ValueError("receipt state surfaces are incomplete")
    if not isinstance(receipt.route_name, str) or not receipt.route_name.strip():
        raise ValueError("receipt route_name must be non-empty")
    if receipt.production_metal is not True:
        raise ValueError("receipt does not attest production Metal")
    if receipt.continuation_covered is not True:
        raise ValueError("receipt does not cover continuation")
    for name in (
        "fallback_count",
        "post_warmup_recompile_count",
        "compile_count",
        "warmup_compile_count",
    ):
        value = getattr(receipt, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"receipt {name} must be a non-negative integer")
    if receipt.fallback_count != 0:
        raise ValueError("receipt records a fallback")
    if receipt.post_warmup_recompile_count != 0:
        raise ValueError("receipt records a post-warmup recompile")
    if receipt.execution_kind is OracleExecutionKind.EAGER_TRANSACTION:
        if receipt.compiled_candidate is not False or any(
            count != 0
            for count in (
                receipt.compile_count,
                receipt.warmup_compile_count,
            )
        ):
            raise ValueError("eager receipt records compile activity")
    elif receipt.execution_kind is OracleExecutionKind.COMPILED_CANDIDATE:
        if receipt.compiled_candidate is not True:
            raise ValueError("compiled receipt lacks compiled-candidate identity")
        if receipt.compile_count < 1 or receipt.warmup_compile_count < 1:
            raise ValueError("compiled receipt lacks warmup compilation")
        if receipt.compile_count != receipt.warmup_compile_count:
            raise ValueError("compiled receipt records compile outside warmup")
    prefixes = receipt.accepted_prefixes
    if (
        not prefixes
        or any(
            isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 0
            for prefix in prefixes
        )
        or prefixes != tuple(range(prefixes[-1] + 1))
    ):
        raise ValueError("receipt accepted prefixes must cover 0..M")
    drops = receipt.ragged_drop_vectors
    if not drops or len(set(drops)) != len(drops):
        raise ValueError("receipt ragged drop vectors are empty or duplicated")
    batch_size = len(drops[0])
    for vector in drops:
        _validate_drop_vector(vector, "receipt ragged_drop_vectors")
        if len(vector) != batch_size:
            raise ValueError("receipt ragged vectors have inconsistent widths")
    if (0,) * batch_size not in drops:
        raise ValueError("receipt ragged vectors omit the all-zero case")
    if receipt.case_count != len(prefixes) * len(drops) * len(OraclePhase):
        raise ValueError("receipt case count does not cover the complete matrix")


def _evidence_digest(evidence: PromptLookupOracleEvidence) -> str:
    payload = {
        "schema_version": evidence.schema_version,
        "subject": evidence.subject.__dict__,
        "geometry": {
            **evidence.geometry.__dict__,
            "ragged_drop_vectors": evidence.geometry.ragged_drop_vectors,
        },
        "production_metal": evidence.production_metal,
        "route": {
            **evidence.route.__dict__,
            "execution_kind": evidence.route.execution_kind.value,
        },
        "cases": [
            {
                "accepted_prefix": case.key.accepted_prefix,
                "ragged_drops": case.key.ragged_drops,
                "phase": case.key.phase.value,
                "surfaces": [
                    {
                        "surface": surface.surface,
                        "reference": _raw_value_payload(surface.reference),
                        "candidate": _raw_value_payload(surface.candidate),
                    }
                    for surface in sorted(case.surfaces, key=lambda item: item.surface)
                ],
            }
            for case in sorted(evidence.cases, key=lambda item: item.key)
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _raw_value_payload(value: RawBitValue) -> dict[str, Any]:
    return {
        "is_none": value.is_none,
        "dtype": value.dtype,
        "shape": value.shape,
        "raw_bits_base64": base64.b64encode(value.raw_bits).decode("ascii"),
    }


def _validate_drop_vector(value: tuple[int, ...], name: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty tuple")
    if any(
        isinstance(drop, bool) or not isinstance(drop, int) or drop < 0
        for drop in value
    ):
        raise ValueError(f"{name} must contain non-negative integers")


__all__ = [
    "ATTESTATION_ISSUER",
    "ATTESTATION_SCHEMA_VERSION",
    "HYBRID_TRANSACTIONAL_ACCEPT_ROUTE",
    "REQUIRED_STATE_SURFACES",
    "AttestationSubject",
    "AttestationValidation",
    "AttestationValidationError",
    "OracleCaseEvidence",
    "OracleCaseKey",
    "OracleExecutionKind",
    "OracleGeometry",
    "OraclePhase",
    "PromptLookupAttestationReceipt",
    "PromptLookupOracleEvidence",
    "RawBitComparison",
    "RawBitValue",
    "RouteEngagementEvidence",
    "SurfaceEvidence",
    "TrustedPromptLookupReceipt",
    "compare_raw_bits",
    "issue_prompt_lookup_attestation",
    "parse_prompt_lookup_attestation_receipt",
    "prompt_lookup_attestation_receipt_to_payload",
    "validate_prompt_lookup_oracle",
    "verify_trusted_prompt_lookup_receipt",
]
