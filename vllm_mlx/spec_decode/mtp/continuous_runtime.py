# SPDX-License-Identifier: Apache-2.0
"""Production assembly for Rapid's continuous self-MTP runtime.

The engine and MLX backend deliberately accept injected protocols.  This
module is the narrow production bridge from an MTP-injected loaded model to
those protocols.  It validates the injector's capability descriptor before it
constructs any runtime object and imports ``mlx_lm`` lazily when a target cache
is actually requested.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from .continuous_engine import (
    ContinuousSelfMTPCapabilities,
    ContinuousSelfMTPConfig,
    ContinuousSelfMTPRuntime,
    ContinuousSelfMTPUnsupportedError,
    RapidForwardSeams,
)
from .mlx_backend import RapidMLXSelfMTPBackend, RapidRaggedCacheAdapter
from .ragged_cache import preflight_ragged_cache, trim_ragged_cache

_SUPPORTED_FAMILIES = frozenset({"qwen3_5", "qwen4_exp"})


def _unsupported(message: str) -> ContinuousSelfMTPUnsupportedError:
    return ContinuousSelfMTPUnsupportedError(
        f"cannot assemble continuous self-MTP runtime: {message}"
    )


def _make_prompt_cache(model: Any) -> Any:
    """Construct a target-trunk cache without making module import eager."""

    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


def _descriptor_for(model: Any) -> Mapping[str, Any]:
    candidate = getattr(model, "language_model", None)
    found: list[Mapping[str, Any]] = []
    for owner in (candidate, model):
        descriptor = getattr(owner, "batched_mtp_capability", None)
        if isinstance(descriptor, Mapping):
            found.append(descriptor)
    if not found:
        raise _unsupported("model has no batched_mtp_capability descriptor")
    if any(dict(descriptor) != dict(found[0]) for descriptor in found[1:]):
        raise _unsupported("outer and inner capability descriptors disagree")
    return found[0]


def _resolve_inner(model: Any, family: str) -> Any:
    if family == "qwen3_5":
        # Use the injector's resolver rather than growing a subtly different
        # list of supported wrapper shapes here.
        from .qwen3_5_inject import _resolve_inner_text_model

        inner = _resolve_inner_text_model(model)
    elif family == "qwen4_exp":
        from .qwen4_exp_inject import _resolve_inner

        inner = _resolve_inner(model)
    else:  # Guarded by descriptor validation; retained for direct testability.
        inner = None
    if inner is None:
        raise _unsupported(f"cannot resolve injected {family} text model")
    return inner


def _require_descriptor(
    descriptor: Mapping[str, Any],
) -> tuple[str, str]:
    family = descriptor.get("model_family")
    if not isinstance(family, str) or family not in _SUPPORTED_FAMILIES:
        raise _unsupported(f"unsupported model family: {family!r}")

    required = {
        "protocol_version": 1,
        "recursive_draft_depth": 2,
        "fixed_membership": True,
        "target_return_hidden": True,
        "mtp_return_hidden": True,
        "confirmed_target_forward": True,
        "ragged_rollback": True,
        "atomic_cache_commit": True,
        "quantized_cache": False,
        "windowed_cache": False,
        "xtc": False,
    }
    for name, expected in required.items():
        if descriptor.get(name) != expected:
            raise _unsupported(
                f"capability descriptor mismatch: {name} must be {expected!r}"
            )

    if family == "qwen4_exp" and descriptor.get("target_verify_mode") != (
        "tokenwise_exact"
    ):
        raise _unsupported(
            "Qwen4 capability descriptor requires target_verify_mode="
            "'tokenwise_exact'"
        )

    batch_forward_name = descriptor.get("batch_forward")
    if not isinstance(batch_forward_name, str) or not batch_forward_name:
        raise _unsupported("capability descriptor has no batch_forward method")
    return family, batch_forward_name


def _qwen4_state_caches(cache: Any) -> list[Any]:
    from vllm_mlx.models.qwen4_exp_cache import Qwen4ExpStateCache

    found = []

    def visit(value: Any) -> None:
        if isinstance(value, Qwen4ExpStateCache):
            found.append(value)
            return
        children = getattr(value, "caches", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(cache)
    return found


def _qwen4_exact_target_forward(
    inner: Any,
    inputs: Any,
    *,
    cache: Any,
    return_hidden: bool,
    n_confirmed: int,
):
    """Verify Qwen4 drafts with the exact tokenwise greedy target path.

    Qwen4's block-shaped GDN, PLE, QSA, and quantized matmuls are close but not
    bit-exact to ordinary one-token decode and can change a real checkpoint's
    argmax.  Sequential target calls preserve distributional identity while
    still batching independent lanes.  State-cache snapshots retain the exact
    per-token rollback boundaries expected by the ragged transaction adapter.
    """

    if n_confirmed <= 0 or int(inputs.shape[1]) <= 1:
        return inner(
            inputs,
            cache=cache,
            return_hidden=return_hidden,
            n_confirmed=n_confirmed,
        )
    if return_hidden is not True:
        raise _unsupported("Qwen4 exact target verification must return hidden state")

    import mlx.core as mx

    state_caches = _qwen4_state_caches(cache)
    snapshots: dict[int, list[list[Any]]] = {id(item): [] for item in state_caches}
    logits = []
    hidden = []
    for position in range(int(inputs.shape[1])):
        step_logits, step_hidden = inner(
            inputs[:, position : position + 1],
            cache=cache,
            return_hidden=True,
            n_confirmed=0,
        )
        logits.append(step_logits)
        hidden.append(step_hidden)
        for item in state_caches:
            if any(value is None for value in item.cache):
                raise _unsupported("Qwen4 target state cache is incomplete")
            snapshots[id(item)].append([mx.array(value) for value in item.cache])
    for item in state_caches:
        item.rollback_state = snapshots[id(item)]
        item._rollback_slots = None
    return mx.concatenate(logits, axis=1), mx.concatenate(hidden, axis=1)


def _require_target_abi(inner: Any) -> None:
    if not callable(inner):
        raise _unsupported("resolved text model is not callable")
    try:
        signature = inspect.signature(inner.__call__)
    except (TypeError, ValueError) as exc:
        raise _unsupported("cannot inspect target forward ABI") from exc
    missing = tuple(
        name
        for name in ("return_hidden", "n_confirmed")
        if name not in signature.parameters
    )
    if missing:
        raise _unsupported("target forward ABI is missing " + ", ".join(missing))


def assemble_continuous_self_mtp_runtime(
    model: Any,
    *,
    allow_dynamic_membership: bool = False,
    array_ops: Any = None,
    logits_processor: Any = None,
    prefill_step_size: int = 512,
) -> ContinuousSelfMTPRuntime:
    """Build a ready runtime from an MTP-injected Rapid model.

    Fixed-core capabilities are admitted only after the versioned descriptor,
    target ABI, injected batch-forward seam, and cache factories are all
    present.  Dynamic membership is an additional conjunction of caller policy
    and descriptor attestation; requesting it cannot manufacture a capability.
    """

    descriptor = _descriptor_for(model)
    family, batch_forward_name = _require_descriptor(descriptor)
    inner = _resolve_inner(model, family)
    _require_target_abi(inner)

    inner_descriptor = getattr(inner, "batched_mtp_capability", None)
    if not isinstance(inner_descriptor, Mapping) or dict(inner_descriptor) != dict(
        descriptor
    ):
        raise _unsupported("resolved text model does not carry the same descriptor")

    batch_forward = getattr(inner, batch_forward_name, None)
    if not callable(batch_forward):
        raise _unsupported(f"injected method {batch_forward_name!r} is not callable")
    make_mtp_cache = getattr(inner, "make_mtp_cache", None)
    if not callable(make_mtp_cache):
        raise _unsupported("injected make_mtp_cache is not callable")

    from .ragged_cache import install_ragged_cache_rollback

    if family == "qwen4_exp":
        # Qwen4 owns both recurrent state and QSA index caches.  Install their
        # exact ragged rollback hooks before any cache factory can run.
        install_ragged_cache_rollback()
    else:
        install_ragged_cache_rollback(qwen4_state_cls=None, qsa_cls=None)

    def mtp_forward(hidden: Any, token_ids: Any, cache: Any, *, return_hidden: bool):
        # RapidForwardSeams always asks for hidden state.  The injected batched
        # method bakes that request into its contract and accepts no flag.
        if return_hidden is not True:
            raise _unsupported("batched MTP forward must return hidden state")
        return batch_forward(hidden, token_ids, cache)

    dynamic_membership = (
        allow_dynamic_membership and descriptor.get("dynamic_join") is True
    )
    capabilities = ContinuousSelfMTPCapabilities(
        target_return_hidden=descriptor.get("target_return_hidden") is True,
        mtp_return_hidden=descriptor.get("mtp_return_hidden") is True,
        confirmed_target_forward=descriptor.get("confirmed_target_forward") is True,
        ragged_rollback=descriptor.get("ragged_rollback") is True,
        atomic_cache_commit=descriptor.get("atomic_cache_commit") is True,
        dynamic_membership=dynamic_membership,
        flash_dynamic_membership_attested=False,
    )
    missing = capabilities.missing_fixed_core()
    if missing:  # Defensive: future capability additions remain fail-closed.
        raise _unsupported("missing fixed-core capability: " + ", ".join(missing))

    target_forward = inner
    if family == "qwen4_exp":

        def target_forward(
            inputs: Any,
            *,
            cache: Any,
            return_hidden: bool,
            n_confirmed: int,
        ):
            return _qwen4_exact_target_forward(
                inner,
                inputs,
                cache=cache,
                return_hidden=return_hidden,
                n_confirmed=n_confirmed,
            )

    return ContinuousSelfMTPRuntime(
        config=ContinuousSelfMTPConfig(
            enabled=True,
            allow_dynamic_membership=allow_dynamic_membership,
            architecture=family,
        ),
        capabilities=capabilities,
        forwards=RapidForwardSeams(target_forward, mtp_forward),
        compute=RapidMLXSelfMTPBackend(
            target_cache_factory=lambda: _make_prompt_cache(inner),
            draft_cache_factory=make_mtp_cache,
            array_ops=array_ops,
            logits_processor=logits_processor,
            prefill_step_size=prefill_step_size,
        ),
        caches=RapidRaggedCacheAdapter(
            preflight=preflight_ragged_cache,
            trim=trim_ragged_cache,
        ),
    )


__all__ = ["assemble_continuous_self_mtp_runtime"]
