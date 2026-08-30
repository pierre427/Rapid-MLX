# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import math
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from vllm_mlx.spec_decode.mtp.prompt_lookup_attestation import (
    HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    REQUIRED_STATE_SURFACES,
    AttestationSubject,
    AttestationValidationError,
    OracleCaseEvidence,
    OracleCaseKey,
    OracleExecutionKind,
    OracleGeometry,
    OraclePhase,
    PromptLookupOracleEvidence,
    PromptLookupAttestationReceipt,
    RawBitValue,
    RouteEngagementEvidence,
    SurfaceEvidence,
    TrustedPromptLookupReceipt,
    compare_raw_bits,
    issue_prompt_lookup_attestation,
    parse_prompt_lookup_attestation_receipt,
    prompt_lookup_attestation_receipt_to_payload,
    validate_prompt_lookup_oracle,
    verify_trusted_prompt_lookup_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def _geometry(**changes) -> OracleGeometry:
    values = {
        "batch_size": 2,
        "verify_width": 2,
        "ragged_drop_vectors": ((0, 0), (1, 0), (0, 1)),
        "ragged_domain_complete": True,
        "ragged_domain_definition": "all commit-path drops for B2/M2",
    }
    values.update(changes)
    return OracleGeometry(**values)


def _subject(geometry: OracleGeometry, **changes) -> AttestationSubject:
    values = {
        "model_id": "Qwen/Qwen3.8-Flash-Next",
        "model_revision": "weights-sha256",
        "runtime_commit": "rapid-commit-sha",
        "cache_topology": "target+mtp+gdn+ple+qsa:v2",
        "state_dtype": "bf16+fp32+int64",
        "verify_geometry": geometry.fingerprint,
        "oracle_version": "raw-bit-state-v1",
    }
    values.update(changes)
    return AttestationSubject(**values)


def _raw(label: str) -> RawBitValue:
    return RawBitValue.from_buffer(
        dtype="uint8",
        shape=(len(label),),
        raw_bits=label.encode("ascii"),
    )


def _case(key: OracleCaseKey) -> OracleCaseEvidence:
    surfaces = []
    for surface in sorted(REQUIRED_STATE_SURFACES):
        label = (
            f"{key.accepted_prefix}:{key.ragged_drops}:{key.phase.value}:{surface}"
        )
        value = _raw(label)
        surfaces.append(SurfaceEvidence(surface, value, value))
    return OracleCaseEvidence(key, tuple(surfaces))


def _evidence(
    *,
    geometry: OracleGeometry | None = None,
    subject: AttestationSubject | None = None,
    route: RouteEngagementEvidence | None = None,
    cases: tuple[OracleCaseEvidence, ...] | None = None,
    production_metal: bool = True,
) -> tuple[PromptLookupOracleEvidence, AttestationSubject, OracleGeometry]:
    geometry = geometry or _geometry()
    subject = subject or _subject(geometry)
    cases = cases or tuple(_case(key) for key in geometry.required_case_keys)
    route = route or RouteEngagementEvidence(
        route_name=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
        execution_kind=OracleExecutionKind.EAGER_TRANSACTION,
        compiled_candidate=False,
        candidate_invocations=len(geometry.required_case_keys),
        fallback_count=0,
        compile_count=0,
        warmup_compile_count=0,
        post_warmup_recompile_count=0,
    )
    return (
        PromptLookupOracleEvidence(
            subject=subject,
            geometry=geometry,
            production_metal=production_metal,
            route=route,
            cases=cases,
        ),
        subject,
        geometry,
    )


def _validate(
    evidence,
    subject,
    geometry,
    *,
    route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
):
    return validate_prompt_lookup_oracle(
        evidence,
        expected_subject=subject,
        expected_geometry=geometry,
        expected_route=route,
    )


def test_attestation_module_has_no_mlx_imports() -> None:
    source = (
        ROOT / "vllm_mlx/spec_decode/mtp/prompt_lookup_attestation.py"
    ).read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert not any(name == "mlx" or name.startswith("mlx.") for name in imported)


def test_raw_bit_comparison_checks_none_dtype_shape_and_bits() -> None:
    value = RawBitValue.from_buffer(dtype="bf16", shape=(1,), raw_bits=b"\x00\x00")

    assert compare_raw_bits(value, value).equal is True
    assert compare_raw_bits(value, RawBitValue.none()).reason == "none_identity"
    assert (
        compare_raw_bits(
            value,
            RawBitValue.from_buffer(dtype="fp16", shape=(1,), raw_bits=b"\x00\x00"),
        ).reason
        == "dtype"
    )
    assert (
        compare_raw_bits(
            value,
            RawBitValue.from_buffer(dtype="bf16", shape=(2,), raw_bits=b"\x00\x00"),
        ).reason
        == "shape"
    )
    assert (
        compare_raw_bits(
            value,
            RawBitValue.from_buffer(dtype="bf16", shape=(1,), raw_bits=b"\x00\x01"),
        ).reason
        == "raw_bits"
    )


def test_signed_zero_and_nan_payloads_are_compared_by_storage_bits() -> None:
    positive_zero = RawBitValue.from_buffer(
        dtype="fp32", shape=(), raw_bits=struct.pack(">f", 0.0)
    )
    negative_zero = RawBitValue.from_buffer(
        dtype="fp32", shape=(), raw_bits=struct.pack(">f", -0.0)
    )
    nan_a = RawBitValue.from_buffer(
        dtype="fp32", shape=(), raw_bits=bytes.fromhex("7fc00001")
    )
    nan_b = RawBitValue.from_buffer(
        dtype="fp32", shape=(), raw_bits=bytes.fromhex("7fc00002")
    )

    assert compare_raw_bits(positive_zero, negative_zero).reason == "raw_bits"
    assert compare_raw_bits(nan_a, nan_b).reason == "raw_bits"
    assert compare_raw_bits(nan_a, nan_a).equal is True


def test_canonical_host_metadata_rejects_nan() -> None:
    metadata = RawBitValue.from_canonical_value(
        dtype="host-json", value={"lengths": [2, 1], "speculating": True}
    )

    assert metadata.raw_bits == b'{"lengths":[2,1],"speculating":true}'
    with pytest.raises(ValueError, match="canonical JSON"):
        RawBitValue.from_canonical_value(dtype="host-json", value=math.nan)


def test_complete_production_metal_evidence_issues_deterministic_receipt() -> None:
    evidence, subject, geometry = _evidence()

    authority = issue_prompt_lookup_attestation(
        evidence,
        expected_subject=subject,
        expected_geometry=geometry,
        expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    )

    receipt = authority.receipt
    assert len(authority.evidence_digest) == 64
    assert authority.identity.test_digest == authority.evidence_digest
    assert receipt.identity.model_revision == subject.model_revision
    assert receipt.identity.verify_geometry == geometry.fingerprint
    assert receipt.case_count == len(geometry.required_case_keys)
    assert set(receipt.state_surfaces) == REQUIRED_STATE_SURFACES

    reversed_evidence = replace(
        evidence,
        cases=tuple(
            replace(case, surfaces=tuple(reversed(case.surfaces)))
            for case in reversed(evidence.cases)
        ),
    )
    reversed_authority = issue_prompt_lookup_attestation(
        reversed_evidence,
        expected_subject=subject,
        expected_geometry=geometry,
        expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    )
    assert reversed_authority.evidence_digest == receipt.evidence_digest


def test_parsed_payload_is_untrusted_and_cannot_be_installed() -> None:
    evidence, subject, geometry = _evidence()
    authority = issue_prompt_lookup_attestation(
        evidence,
        expected_subject=subject,
        expected_geometry=geometry,
        expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    )
    payload = prompt_lookup_attestation_receipt_to_payload(authority.receipt)

    parsed = parse_prompt_lookup_attestation_receipt(payload)

    assert isinstance(parsed, PromptLookupAttestationReceipt)
    assert not isinstance(parsed, TrustedPromptLookupReceipt)
    with pytest.raises(AttestationValidationError, match="not trusted"):
        verify_trusted_prompt_lookup_receipt(
            parsed,  # type: ignore[arg-type]
            expected_subject=subject,
            expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
        )


def test_sealed_authority_verifies_exact_subject_and_route() -> None:
    evidence, subject, geometry = _evidence()
    authority = issue_prompt_lookup_attestation(
        evidence,
        expected_subject=subject,
        expected_geometry=geometry,
        expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    )

    receipt = verify_trusted_prompt_lookup_receipt(
        authority,
        expected_subject=subject,
        expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    )

    assert receipt is authority.receipt
    with pytest.raises(AttestationValidationError, match="identity mismatch"):
        verify_trusted_prompt_lookup_receipt(
            authority,
            expected_subject=replace(subject, model_revision="foreign"),
            expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
        )
    with pytest.raises(AttestationValidationError, match="route mismatch"):
        verify_trusted_prompt_lookup_receipt(
            authority,
            expected_subject=subject,
            expected_route="plain_decode",
        )


def test_trusted_authority_cannot_be_constructed_without_private_seal() -> None:
    evidence, subject, geometry = _evidence()
    authority = issue_prompt_lookup_attestation(
        evidence,
        expected_subject=subject,
        expected_geometry=geometry,
        expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    )

    with pytest.raises(ValueError, match="seal"):
        TrustedPromptLookupReceipt(
            receipt=authority.receipt,
            issuer=authority.issuer,
            _seal=object(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("production_metal", False, "production Metal"),
        ("continuation_covered", False, "continuation"),
        ("fallback_count", 1, "fallback"),
        ("post_warmup_recompile_count", 1, "recompile"),
        ("case_count", 1, "complete matrix"),
    ],
)
def test_parser_refuses_receipt_claim_regressions(field, value, message) -> None:
    evidence, subject, geometry = _evidence()
    authority = issue_prompt_lookup_attestation(
        evidence,
        expected_subject=subject,
        expected_geometry=geometry,
        expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    )
    payload = prompt_lookup_attestation_receipt_to_payload(authority.receipt)
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        parse_prompt_lookup_attestation_receipt(payload)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda e: replace(e, production_metal=False), "not_production_metal"),
        (
            lambda e: replace(e, schema_version=999),
            "schema_version",
        ),
        (
            lambda e: replace(
                e,
                route=replace(e.route, candidate_invocations=0),
            ),
            "candidate_route_not_observed",
        ),
        (
            lambda e: replace(e, route=replace(e.route, fallback_count=1)),
            "fallback_observed",
        ),
        (
            lambda e: replace(
                e,
                route=replace(e.route, post_warmup_recompile_count=1),
            ),
            "eager_route_has_compile_activity",
        ),
        (
            lambda e: replace(e, route=replace(e.route, compile_count=1)),
            "eager_route_has_compile_activity",
        ),
        (
            lambda e: replace(e, route=replace(e.route, warmup_compile_count=1)),
            "eager_route_has_compile_activity",
        ),
        (
            lambda e: replace(e, route=replace(e.route, compiled_candidate=True)),
            "execution_kind_mismatch",
        ),
    ],
)
def test_route_and_hardware_failures_refuse_receipt(mutation, reason) -> None:
    evidence, subject, geometry = _evidence()
    changed = mutation(evidence)

    validation = _validate(changed, subject, geometry)

    assert validation.eligible is False
    assert reason in validation.reasons
    with pytest.raises(AttestationValidationError, match=reason):
        issue_prompt_lookup_attestation(
            changed,
            expected_subject=subject,
            expected_geometry=geometry,
            expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
        )


def test_compiled_candidate_requires_warmup_compile_and_zero_recompile() -> None:
    evidence, subject, geometry = _evidence()
    compiled = replace(
        evidence,
        route=replace(
            evidence.route,
            execution_kind=OracleExecutionKind.COMPILED_CANDIDATE,
            compiled_candidate=True,
            compile_count=1,
            warmup_compile_count=1,
            post_warmup_recompile_count=0,
        ),
    )

    authority = issue_prompt_lookup_attestation(
        compiled,
        expected_subject=subject,
        expected_geometry=geometry,
        expected_route=HYBRID_TRANSACTIONAL_ACCEPT_ROUTE,
    )

    assert authority.receipt.compiled_candidate is True
    assert authority.receipt.compile_count == 1

    for changes, reason in (
        ({"compiled_candidate": False}, "execution_kind_mismatch"),
        ({"compile_count": 0}, "candidate_never_compiled"),
        ({"warmup_compile_count": 0}, "candidate_not_compiled_during_warmup"),
        ({"compile_count": 2}, "compile_outside_warmup"),
        ({"post_warmup_recompile_count": 1}, "post_warmup_recompile_observed"),
    ):
        changed = replace(compiled, route=replace(compiled.route, **changes))
        assert reason in _validate(changed, subject, geometry).reasons


def test_identity_geometry_and_route_must_match_trusted_expectations() -> None:
    evidence, subject, geometry = _evidence()
    foreign_subject = replace(subject, model_revision="different-weights")
    foreign_geometry = _geometry(ragged_domain_definition="different domain")

    assert "identity_mismatch" in _validate(
        evidence, foreign_subject, geometry
    ).reasons
    assert "geometry_mismatch" in _validate(
        evidence, subject, foreign_geometry
    ).reasons
    assert "route_mismatch" in _validate(
        evidence, subject, geometry, route="plain_decode"
    ).reasons


def test_subject_geometry_string_must_be_canonical() -> None:
    geometry = _geometry()
    subject = _subject(geometry, verify_geometry="batch=2,width=2")
    evidence, _, _ = _evidence(geometry=geometry, subject=subject)

    validation = _validate(evidence, subject, geometry)

    assert "geometry_identity_mismatch" in validation.reasons


def test_ragged_domain_must_be_declared_complete() -> None:
    geometry = _geometry(ragged_domain_complete=False)
    subject = _subject(geometry)
    evidence, _, _ = _evidence(geometry=geometry, subject=subject)

    validation = _validate(evidence, subject, geometry)

    assert "ragged_domain_incomplete" in validation.reasons


@pytest.mark.parametrize(
    "remove_key",
    [
        OracleCaseKey(0, (0, 0), OraclePhase.POST_TRIM),
        OracleCaseKey(2, (0, 0), OraclePhase.POST_TRIM),
        OracleCaseKey(1, (1, 0), OraclePhase.POST_TRIM),
        OracleCaseKey(1, (0, 0), OraclePhase.CONTINUATION),
    ],
)
def test_every_m_ragged_vector_and_continuation_case_is_required(remove_key) -> None:
    evidence, subject, geometry = _evidence()
    cases = tuple(case for case in evidence.cases if case.key != remove_key)

    validation = _validate(replace(evidence, cases=cases), subject, geometry)

    assert "incomplete_case_matrix" in validation.reasons


def test_duplicate_case_is_refused() -> None:
    evidence, subject, geometry = _evidence()
    duplicate = replace(evidence, cases=evidence.cases + (evidence.cases[0],))

    validation = _validate(duplicate, subject, geometry)

    assert "duplicate_cases" in validation.reasons


def test_missing_duplicate_or_extra_state_surface_is_refused() -> None:
    evidence, subject, geometry = _evidence()
    first = evidence.cases[0]
    missing = replace(first, surfaces=first.surfaces[1:])
    duplicate = replace(first, surfaces=first.surfaces + (first.surfaces[0],))
    extra_surface = SurfaceEvidence("unknown_surface", _raw("x"), _raw("x"))
    extra = replace(first, surfaces=first.surfaces + (extra_surface,))

    for changed, expected in (
        (missing, "incomplete_state_surfaces"),
        (duplicate, "duplicate_state_surfaces"),
        (extra, "incomplete_state_surfaces"),
    ):
        cases = (changed,) + evidence.cases[1:]
        validation = _validate(replace(evidence, cases=cases), subject, geometry)
        assert expected in validation.reasons


def test_one_bit_state_difference_refuses_receipt_and_changes_digest() -> None:
    evidence, subject, geometry = _evidence()
    first = evidence.cases[0]
    original = first.surfaces[0]
    mismatched = replace(original, candidate=_raw("different-bits"))
    changed_case = replace(first, surfaces=(mismatched,) + first.surfaces[1:])
    changed = replace(evidence, cases=(changed_case,) + evidence.cases[1:])

    baseline = _validate(evidence, subject, geometry)
    validation = _validate(changed, subject, geometry)

    assert "raw_bit_mismatch" in validation.reasons
    assert validation.evidence_digest != baseline.evidence_digest


def test_candidate_invocations_must_cover_every_declared_case() -> None:
    evidence, subject, geometry = _evidence()
    changed = replace(
        evidence,
        route=replace(
            evidence.route,
            candidate_invocations=len(geometry.required_case_keys) - 1,
        ),
    )

    validation = _validate(changed, subject, geometry)

    assert "candidate_route_not_observed" in validation.reasons
