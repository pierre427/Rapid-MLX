"""NumPy-only gates for transformed continuous self-MTP verification."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from vllm_mlx.spec_decode.mtp.continuous_engine import (
    ContinuousSelfMTPUnsupportedError,
    SelfMTPLane,
    SelfMTPSampling,
)
from vllm_mlx.spec_decode.mtp.mlx_backend import RapidMLXSelfMTPBackend
from vllm_mlx.spec_decode.mtp.residual_sampling import (
    TransformedResidualSamplingHooks,
    TransformedSamplingProfile,
)


class _NumpyResidualOps:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.transform_calls = 0

    @staticmethod
    def _logsumexp_native(values):
        values = np.asarray(values)
        maximum = np.max(values, axis=-1, keepdims=True)
        exponentials = np.exp(values - maximum).astype(values.dtype)
        total = np.sum(exponentials, axis=-1, keepdims=True, dtype=values.dtype)
        return (maximum + np.log(total).astype(values.dtype)).astype(values.dtype)

    def native_logprobs(self, logits):
        self.transform_calls += 1
        logits = np.asarray(logits)
        return (logits - self._logsumexp_native(logits)).astype(logits.dtype)

    @staticmethod
    def top_p(logprobs, value):
        values = np.asarray(logprobs)
        indices = np.argsort(values, axis=-1)
        probabilities = np.exp(values)
        sorted_probabilities = np.take_along_axis(probabilities, indices, axis=-1)
        cumulative = np.cumsum(sorted_probabilities, axis=-1)
        inverse = np.argsort(indices, axis=-1)
        cumulative = np.take_along_axis(cumulative, inverse, axis=-1)
        return np.where(cumulative > 1 - value, values, -np.inf)

    @staticmethod
    def min_p(logprobs, value, min_tokens):
        values = np.asarray(logprobs)
        remove = values < np.max(values, axis=-1, keepdims=True) + np.log(value)
        if min_tokens > 1:
            indices = np.argpartition(values, -min_tokens, axis=-1)[..., -min_tokens:]
            np.put_along_axis(remove, indices, False, axis=-1)
        return np.where(remove, -np.inf, values)

    @staticmethod
    def top_k(logprobs, value):
        values = np.asarray(logprobs)
        vocab = values.shape[-1]
        if not 0 < value < vocab:
            raise ValueError(f"top_k must be in (0, {vocab}); got {value}")
        indices = np.argpartition(-values, value - 1, axis=-1)[..., value:]
        result = values.copy()
        np.put_along_axis(result, indices, -np.inf, axis=-1)
        return result

    @staticmethod
    def scaled_float32_logprobs(logprobs, temperature):
        values = (np.asarray(logprobs) * (1 / temperature)).astype(np.float32)
        maximum = np.max(values, axis=-1, keepdims=True)
        total = np.sum(np.exp(values - maximum), axis=-1, keepdims=True)
        return values - (maximum + np.log(total))

    def categorical(self, logprobs):
        probabilities = np.exp(np.asarray(logprobs, dtype=np.float64))
        probabilities /= probabilities.sum()
        return int(self.rng.choice(len(probabilities), p=probabilities))

    @staticmethod
    def gather_rows(logprobs, tokens):
        values = np.asarray(logprobs)
        return values[np.arange(len(tokens)), np.asarray(tokens)]

    @staticmethod
    def stack(values):
        return np.stack(values)

    @staticmethod
    def exp(value):
        return np.exp(value)

    @staticmethod
    def minimum_scalar(value, scalar):
        return np.minimum(value, scalar)

    @staticmethod
    def maximum_scalar(value, scalar):
        return np.maximum(value, scalar)

    @staticmethod
    def subtract(left, right):
        return left - right

    @staticmethod
    def sum_float(value):
        return float(np.sum(value))

    @staticmethod
    def normalize_probabilities_to_logprobs(probabilities):
        probabilities = np.asarray(probabilities)
        with np.errstate(divide="ignore"):
            return np.log(probabilities / probabilities.sum())

    def uniforms(self, count):
        return self.rng.uniform(size=count).tolist()

    @staticmethod
    def tolist(value):
        return np.asarray(value).tolist()


class _CombinedOps(_NumpyResidualOps):
    @staticmethod
    def uint32(value):
        return np.asarray(value, dtype=np.uint32)

    @staticmethod
    def concatenate(values, *, axis):
        return np.concatenate(list(values), axis=axis)

    @staticmethod
    def pad(value, widths):
        return np.pad(value, widths)

    @staticmethod
    def expand_dims(value, axis):
        return np.expand_dims(value, axis)

    @staticmethod
    def logprobs(logits):
        values = np.asarray(logits, dtype=np.float64)
        maximum = np.max(values, axis=-1, keepdims=True)
        return (
            values
            - maximum
            - np.log(np.exp(values - maximum).sum(axis=-1, keepdims=True))
        )

    @staticmethod
    def argmax_int(logprobs):
        return int(np.argmax(logprobs))


def _hook(profile, *, seed=0, ops=None):
    return TransformedResidualSamplingHooks(
        TransformedSamplingProfile(**profile),
        array_ops=ops or _NumpyResidualOps(seed),
    )


PROFILES = [
    dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0),
    dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0),
    dict(temperature=0.9, top_p=0.6, top_k=0, min_p=0.0),
    dict(temperature=1.3, top_p=1.0, top_k=5, min_p=0.0),
    dict(temperature=0.8, top_p=1.0, top_k=0, min_p=0.05),
    dict(temperature=0.7, top_p=0.85, top_k=12, min_p=0.02),
]


@pytest.mark.parametrize("profile", PROFILES)
def test_transformed_profiles_are_normalized_and_respect_filter_support(profile):
    logits = np.random.default_rng(17).normal(size=32).astype(np.float32) * 2
    hooks = _hook(profile)
    transformed = hooks.logprobs(logits, profile["temperature"])
    probabilities = np.exp(transformed)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-6)

    # Independent filter-chain reference in Rapid order.
    reference_ops = _NumpyResidualOps()
    reference = reference_ops.native_logprobs(logits)
    if 0 < profile["top_p"] < 1:
        reference = reference_ops.top_p(reference, profile["top_p"])
    if profile["min_p"]:
        reference = reference_ops.min_p(reference, profile["min_p"], 1)
    if profile["top_k"]:
        reference = reference_ops.top_k(reference, profile["top_k"])
    assert set(np.flatnonzero(np.isfinite(transformed))) == set(
        np.flatnonzero(np.isfinite(reference))
    )


def test_temperature_only_and_top_p_zero_match_direct_softmax():
    logits = np.array([3.0, 2.0, 0.0, -1.0], dtype=np.float32)
    profile = dict(temperature=0.7, top_p=0.0)
    transformed = _hook(profile).logprobs(logits, 0.7)
    scaled = logits.astype(np.float32) / 0.7
    expected = scaled - np.log(np.exp(scaled).sum())
    assert transformed == pytest.approx(expected, abs=1e-6)


def test_top_k_and_min_p_boundary_semantics():
    logits = np.array([1.0, 0.5, 0.5, 0.5, 0.0, -1.0])
    top_k = _hook(dict(temperature=0.8, top_k=2)).logprobs(logits, 0.8)
    assert np.isfinite(top_k).sum() == 2
    assert np.isfinite(top_k[0])

    min_p_logits = np.array([0.0, np.log(0.1), np.log(0.05), -6.0])
    min_p = _hook(dict(temperature=1.0, min_p=0.1)).logprobs(min_p_logits, 1.0)
    # Strict '<' keeps a token exactly at max probability * min_p.
    assert np.isfinite(min_p[1])
    assert not np.isfinite(min_p[3])


def test_disjoint_support_forces_rejection_and_target_residual():
    target_logits = np.array([3.0, 2.5, 2.0, -1.0, -1.5, -2.0])
    draft_logits = np.array([-1.0, -1.5, -2.0, 3.0, 2.5, 2.0])
    hooks = _hook(dict(temperature=0.8, top_k=3), seed=9)
    target = hooks.logprobs(target_logits, 0.8)
    draft = hooks.logprobs(draft_logits, 0.8)
    target_rows = np.stack([target, target, target])
    for _ in range(100):
        drafts = [hooks.sample(draft, 0.8), hooks.sample(draft, 0.8)]
        accepted, bonus = hooks.verify(target_rows, [draft, draft], drafts, 0.8)
        assert accepted == 0
        assert bonus in {0, 1, 2}


def test_residual_verifier_preserves_transformed_target_marginal():
    target_logits = np.array([2.8, 2.2, 1.7, 0.2, -0.5, -1.0])
    draft_logits = np.array([-0.5, -1.0, 1.7, 2.8, 2.2, 0.2])
    hooks = _hook(dict(temperature=0.8, top_k=3), seed=1234)
    target = hooks.logprobs(target_logits, 0.8)
    draft = hooks.logprobs(draft_logits, 0.8)
    rows = np.stack([target, target])
    counts = Counter()
    trials = 12000
    for _ in range(trials):
        proposed = hooks.sample(draft, 0.8)
        accepted, correction = hooks.verify(rows, [draft], [proposed], 0.8)
        counts[proposed if accepted else correction] += 1
    empirical = np.array([counts[i] / trials for i in range(len(target))])
    tv = 0.5 * np.abs(empirical - np.exp(target)).sum()
    assert tv < 0.025


def test_zero_residual_falls_back_to_target_distribution():
    hooks = _hook(dict(temperature=0.8, top_k=3), seed=44)
    target = hooks.logprobs(np.array([3.0, 2.0, 1.0, -4.0]), 0.8)
    rows = np.stack([target, target])
    for _ in range(50):
        proposed = hooks.sample(target, 0.8)
        accepted, bonus = hooks.verify(rows, [target], [proposed], 0.8)
        assert accepted == 1
        assert bonus in {0, 1, 2}


def test_xtc_and_profile_mismatch_fail_closed():
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="XTC"):
        TransformedSamplingProfile(temperature=0.8, xtc_probability=0.1)
    hooks = _hook(dict(temperature=0.8))
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="temperature"):
        hooks.validate_sampling(SelfMTPSampling(temperature=0.7))
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="XTC"):
        hooks.validate_sampling(SelfMTPSampling(temperature=0.8, uses_xtc=True))


def test_backend_consumes_hooks_without_changing_greedy_path():
    ops = _CombinedOps(seed=5)
    hooks = _hook(dict(temperature=0.8, top_k=3), ops=ops)
    backend = RapidMLXSelfMTPBackend(
        array_ops=ops,
        residual_sampling=hooks,
    )
    logits = np.array([3.0, 2.0, 1.0, -2.0])
    transformed_lane = SelfMTPLane(
        uid=1,
        cur=4,
        seed_hidden=None,
        token_prefix=np.array([1, 2]),
        ntoks=1,
        max_tokens=8,
        num_draft=2,
        sampling=SelfMTPSampling(temperature=0.8),
    )
    token, transformed = backend._distribution(
        transformed_lane, transformed_lane.token_prefix, logits
    )
    assert token in {0, 1, 2}
    assert np.isfinite(transformed).sum() == 3
    transformed_calls = ops.transform_calls

    greedy_lane = SelfMTPLane(
        uid=2,
        cur=4,
        seed_hidden=None,
        token_prefix=np.array([1, 2]),
        ntoks=1,
        max_tokens=8,
        num_draft=2,
        sampling=SelfMTPSampling(temperature=0.0),
    )
    greedy_token, greedy = backend._distribution(
        greedy_lane, greedy_lane.token_prefix, logits
    )
    assert greedy_token == 0
    assert ops.transform_calls == transformed_calls  # residual hook was bypassed
    assert np.isfinite(greedy).all()


def test_backend_rejects_processors_without_exact_processor_hook():
    ops = _CombinedOps(seed=1)
    hooks = _hook(dict(temperature=0.8), ops=ops)
    backend = RapidMLXSelfMTPBackend(array_ops=ops, residual_sampling=hooks)
    lane = SelfMTPLane(
        uid=1,
        cur=4,
        seed_hidden=None,
        token_prefix=np.array([1, 2]),
        ntoks=1,
        max_tokens=8,
        num_draft=2,
        sampling=SelfMTPSampling(
            temperature=0.8,
            has_logits_processors=True,
        ),
    )
    with pytest.raises(ContinuousSelfMTPUnsupportedError, match="processors"):
        backend._distribution(lane, lane.token_prefix, np.array([1.0, 0.0]))


def test_sampling_module_has_no_eager_mlx_import():
    source = (
        Path(__file__).parents[1]
        / "vllm_mlx"
        / "spec_decode"
        / "mtp"
        / "residual_sampling.py"
    ).read_text()
    tree = ast.parse(source)
    eager = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and getattr(node, "module", "") == "mlx.core"
    ]
    assert eager == []
