from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_mlx.spec_decode.mtp import prompt_lookup_attestation_install as install
from vllm_mlx.spec_decode.mtp.prompt_lookup_attestation import (
    REQUIRED_STATE_SURFACES,
    OracleCaseEvidence,
    OracleExecutionKind,
    OracleGeometry,
    PromptLookupOracleEvidence,
    RawBitValue,
    RouteEngagementEvidence,
    SurfaceEvidence,
    issue_prompt_lookup_attestation,
    parse_prompt_lookup_attestation_receipt,
    prompt_lookup_attestation_receipt_to_payload,
)
from vllm_mlx.spec_decode.mtp.prompt_lookup_capability import (
    PromptLookupRefusal,
    evaluate_prompt_lookup_capability,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_RAPID_COMMIT = "1" * 40
_MLX_LM_COMMIT = "2" * 40


def _geometry() -> OracleGeometry:
    return OracleGeometry(batch_size=1, verify_width=1)


def _runtime_identity() -> install.RuntimeAttestationIdentity:
    return install.build_runtime_attestation_identity(
        model_id="qwen4_exp",
        checkpoint_config_sha256=_SHA_A,
        checkpoint_index_sha256=_SHA_B,
        checkpoint_weight_manifest_sha256=_SHA_C,
        rapid_runtime_commit=_RAPID_COMMIT,
        mlx_lm_runtime_commit=_MLX_LM_COMMIT,
        layer_types=("linear_attention", "full_attention"),
        ple_layer_ids=(1,),
        mtp_layer_types=("full_attention",),
        cache_classes=("Qwen4ExpStateCache", "KVCache", "QSAIndexCache"),
        state_dtypes=("bfloat16", "uint32"),
        verify_geometry=_geometry().fingerprint,
        oracle_version="raw-bit-oracle-v1",
    )


def _trusted_receipt(
    *,
    runtime: install.RuntimeAttestationIdentity | None = None,
    route_name: str = install.TRUSTED_TRANSACTIONAL_ROUTE,
):
    runtime = runtime or _runtime_identity()
    geometry = _geometry()
    value = RawBitValue.from_buffer(dtype="uint8", shape=(1,), raw_bits=b"\x01")
    surfaces = tuple(
        SurfaceEvidence(surface=surface, reference=value, candidate=value)
        for surface in sorted(REQUIRED_STATE_SURFACES)
    )
    cases = tuple(
        OracleCaseEvidence(key=key, surfaces=surfaces)
        for key in sorted(geometry.required_case_keys)
    )
    evidence = PromptLookupOracleEvidence(
        subject=runtime.to_subject(),
        geometry=geometry,
        production_metal=True,
        route=RouteEngagementEvidence(
            route_name=route_name,
            execution_kind=OracleExecutionKind.EAGER_TRANSACTION,
            compiled_candidate=False,
            candidate_invocations=len(cases),
            fallback_count=0,
            compile_count=0,
            warmup_compile_count=0,
            post_warmup_recompile_count=0,
        ),
        cases=cases,
    )
    return issue_prompt_lookup_attestation(
        evidence,
        expected_subject=runtime.to_subject(),
        expected_geometry=geometry,
        expected_route=route_name,
    )


def _model():
    return SimpleNamespace(model_type="qwen4_exp")


def test_runtime_identity_binds_checkpoint_runtime_topology_and_dtype() -> None:
    identity = _runtime_identity()

    assert identity.model_revision.startswith("sha256:")
    assert identity.runtime_commit == (
        f"rapid={_RAPID_COMMIT};mlx-lm={_MLX_LM_COMMIT}"
    )
    assert identity.cache_topology.startswith("sha256:")
    assert identity.state_dtype.startswith("sha256:")
    assert identity.model_id == "qwen4_exp"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("checkpoint_config_sha256", "A" * 64),
        ("checkpoint_index_sha256", "a" * 63),
        ("checkpoint_weight_manifest_sha256", "not-a-digest"),
        ("rapid_runtime_commit", "1" * 39),
        ("mlx_lm_runtime_commit", "G" * 40),
    ],
)
def test_runtime_identity_rejects_noncanonical_source_identity(
    name: str, value: str
) -> None:
    values = {
        "model_id": "qwen4_exp",
        "checkpoint_config_sha256": _SHA_A,
        "checkpoint_index_sha256": _SHA_B,
        "checkpoint_weight_manifest_sha256": _SHA_C,
        "rapid_runtime_commit": _RAPID_COMMIT,
        "mlx_lm_runtime_commit": _MLX_LM_COMMIT,
        "layer_types": ("full_attention",),
        "ple_layer_ids": (),
        "mtp_layer_types": ("full_attention",),
        "cache_classes": ("KVCache",),
        "state_dtypes": ("bfloat16",),
        "verify_geometry": _geometry().fingerprint,
        "oracle_version": "raw-bit-oracle-v1",
    }
    values[name] = value

    with pytest.raises(ValueError):
        install.build_runtime_attestation_identity(**values)


def test_empty_package_registry_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(install, "_TRUSTED_PROMPT_LOOKUP_RECEIPTS", ())
    model = _model()

    result = install.install_trusted_prompt_lookup_capability(
        model, _runtime_identity()
    )

    assert result.reason is install.PromptLookupInstallReason.NO_TRUSTED_RECEIPT
    assert model.mtp_prompt_lookup_supported is False
    assert not evaluate_prompt_lookup_capability(model).eligible


def test_parsed_json_receipt_cannot_become_authority(monkeypatch) -> None:
    authority = _trusted_receipt()
    parsed = parse_prompt_lookup_attestation_receipt(
        prompt_lookup_attestation_receipt_to_payload(authority.receipt)
    )
    monkeypatch.setattr(install, "_TRUSTED_PROMPT_LOOKUP_RECEIPTS", (parsed,))
    model = _model()

    result = install.install_trusted_prompt_lookup_capability(
        model, _runtime_identity()
    )

    assert result.reason is install.PromptLookupInstallReason.REGISTRY_INVALID
    assert model.mtp_prompt_lookup_supported is False


@pytest.mark.parametrize(
    "field",
    [
        "model_id",
        "model_revision",
        "runtime_commit",
        "cache_topology",
        "state_dtype",
        "verify_geometry",
        "oracle_version",
    ],
)
def test_every_runtime_identity_field_must_match(monkeypatch, field: str) -> None:
    runtime = _runtime_identity()
    monkeypatch.setattr(
        install, "_TRUSTED_PROMPT_LOOKUP_RECEIPTS", (_trusted_receipt(),)
    )
    changed = replace(runtime, **{field: f"mismatch-{getattr(runtime, field)}"})
    model = _model()

    result = install.install_trusted_prompt_lookup_capability(model, changed)

    assert (
        result.reason
        is install.PromptLookupInstallReason.RUNTIME_IDENTITY_MISMATCH
    )
    assert model.mtp_prompt_lookup_supported is False


def test_operator_control_can_only_disable_exact_authority(monkeypatch) -> None:
    monkeypatch.setattr(
        install, "_TRUSTED_PROMPT_LOOKUP_RECEIPTS", (_trusted_receipt(),)
    )
    model = _model()

    result = install.install_trusted_prompt_lookup_capability(
        model, _runtime_identity(), disabled=True
    )

    assert result.reason is install.PromptLookupInstallReason.OPERATOR_DISABLED
    assert model.mtp_prompt_lookup_supported is False
    with pytest.raises(TypeError, match="disabled must be a bool"):
        install.install_trusted_prompt_lookup_capability(
            model, _runtime_identity(), disabled=1
        )


def test_config_and_environment_cannot_supply_positive_authority(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RAPID_MLX_PROMPT_LOOKUP_ATTESTED", "1")
    monkeypatch.setattr(install, "_TRUSTED_PROMPT_LOOKUP_RECEIPTS", ())
    model = _model()
    model.config = {"prompt_lookup_attested": True}
    model.mtp_prompt_lookup_supported = True

    result = install.install_trusted_prompt_lookup_capability(
        model, _runtime_identity()
    )

    assert result.reason is install.PromptLookupInstallReason.NO_TRUSTED_RECEIPT
    assert model.mtp_prompt_lookup_supported is False


def test_exact_sealed_receipt_installs_evaluator_eligible_capability(
    monkeypatch,
) -> None:
    authority = _trusted_receipt()
    monkeypatch.setattr(
        install, "_TRUSTED_PROMPT_LOOKUP_RECEIPTS", (authority,)
    )
    model = _model()

    result = install.install_trusted_prompt_lookup_capability(
        model, _runtime_identity()
    )
    decision = evaluate_prompt_lookup_capability(model)

    assert result.installed is True
    assert result.reason is install.PromptLookupInstallReason.INSTALLED
    assert result.evidence_digest == authority.evidence_digest
    assert decision.eligible is True
    assert decision.reason is PromptLookupRefusal.ELIGIBLE
    assert model.mtp_prompt_lookup_runtime_identity == authority.identity
    assert not hasattr(model, "request_scoped_pld_executor")


def test_product_default_installs_b2_verified_qwen4_capability() -> None:
    model = _model()

    result = install.install_default_qwen4_prompt_lookup_capability(model)
    decision = evaluate_prompt_lookup_capability(model)

    assert result.installed is True
    assert result.reason is install.PromptLookupInstallReason.INSTALLED
    assert result.evidence_digest == (
        "25670f0a5ece08ba0f717e9fbf66a67143846e929de416d2264019c4fac1a4fb"
    )
    assert decision.eligible is True
    assert decision.reason is PromptLookupRefusal.ELIGIBLE
    assert (
        model.mtp_prompt_lookup_runtime_identity
        is install.QWEN4_FLASH_NEXT_DEFAULT_VERIFICATION
    )


def test_product_default_cannot_enable_non_qwen4_or_override_disable() -> None:
    unsupported = SimpleNamespace(model_type="qwen3_5")
    unsupported_result = install.install_default_qwen4_prompt_lookup_capability(
        unsupported
    )
    assert unsupported_result.reason is install.PromptLookupInstallReason.UNSUPPORTED_MODEL
    assert unsupported.mtp_prompt_lookup_supported is False

    model = _model()
    disabled_result = install.install_default_qwen4_prompt_lookup_capability(
        model, disabled=True
    )
    assert disabled_result.reason is install.PromptLookupInstallReason.OPERATOR_DISABLED
    assert model.mtp_prompt_lookup_supported is False


def test_duplicate_exact_authority_fails_closed(monkeypatch) -> None:
    authority = _trusted_receipt()
    monkeypatch.setattr(
        install,
        "_TRUSTED_PROMPT_LOOKUP_RECEIPTS",
        (authority, authority),
    )
    model = _model()

    result = install.install_trusted_prompt_lookup_capability(
        model, _runtime_identity()
    )

    assert result.reason is install.PromptLookupInstallReason.AMBIGUOUS_RECEIPT
    assert model.mtp_prompt_lookup_supported is False


def test_partial_install_rolls_back_and_denies(monkeypatch) -> None:
    authority = _trusted_receipt()
    monkeypatch.setattr(
        install, "_TRUSTED_PROMPT_LOOKUP_RECEIPTS", (authority,)
    )

    class RejectCapability:
        model_type = "qwen4_exp"

        def __setattr__(self, name, value):
            if name == "mtp_prompt_lookup_capability":
                raise TypeError("immutable capability slot")
            super().__setattr__(name, value)

    model = RejectCapability()
    result = install.install_trusted_prompt_lookup_capability(
        model, _runtime_identity()
    )

    assert result.reason is install.PromptLookupInstallReason.INSTALL_FAILED
    assert model.mtp_prompt_lookup_supported is False
    assert not hasattr(model, "mtp_prompt_lookup_runtime_identity")


def test_non_qwen4_model_is_never_installed(monkeypatch) -> None:
    monkeypatch.setattr(
        install, "_TRUSTED_PROMPT_LOOKUP_RECEIPTS", (_trusted_receipt(),)
    )
    model = SimpleNamespace(model_type="qwen3_5")

    result = install.install_trusted_prompt_lookup_capability(
        model, _runtime_identity()
    )

    assert result.reason is install.PromptLookupInstallReason.UNSUPPORTED_MODEL
    assert model.mtp_prompt_lookup_supported is False
