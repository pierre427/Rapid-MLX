"""CPU/mock contracts for the Rapid continuous self-MTP data plane."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from vllm_mlx.spec_decode.mtp.continuous_engine import (
    ContinuousSelfMTPCapabilities,
    ContinuousSelfMTPConfig,
    ContinuousSelfMTPRuntime,
    RapidForwardSeams,
    SelfMTPCachePair,
    SelfMTPLaneSpec,
    SelfMTPSampling,
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)
from vllm_mlx.spec_decode.mtp.continuous_engine import (
    ContinuousSelfMTPUnsupportedError as ContinuousSelfMTPUnsupported,
)
from vllm_mlx.spec_decode.mtp.mlx_backend import (
    RapidMLXSelfMTPBackend,
    RapidRaggedCacheAdapter,
)

ROOT = Path(__file__).resolve().parents[1]


class _NumpyOps:
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
        logits = np.asarray(logits, dtype=np.float64)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    @staticmethod
    def argmax_int(logprobs):
        return int(np.argmax(logprobs, axis=-1))


class _LayerCache:
    def __init__(self, label, rows=None):
        self.label = label
        self.rows = list([label] if rows is None else rows)
        self.events = []

    @classmethod
    def merge(cls, caches):
        merged = cls("merged", [])
        for cache in caches:
            merged.rows.extend(cache.rows)
        return merged

    def extend(self, other):
        self.events.append(("extend", tuple(other.rows)))
        self.rows.extend(other.rows)

    def prepare(self, *, lengths, right_padding):
        self.events.append(("prepare", tuple(lengths), tuple(right_padding)))

    def finalize(self):
        self.events.append(("finalize",))

    def extract(self, index):
        self.events.append(("extract", index))
        return type(self)(f"{self.label}:{index}", [self.rows[index]])

    def filter(self, indices):
        self.events.append(("filter", tuple(indices)))
        self.rows = [self.rows[index] for index in indices]


class _SharedQSAFake(_LayerCache):
    def __init__(self, label, rows=None):
        super().__init__(label, rows)
        self._mtp_share_topk = False
        self._mtp_shared_topk = None

    def begin_self_mtp_cycle(self):
        assert self._mtp_share_topk is False
        assert self._mtp_shared_topk is None
        self._mtp_share_topk = True
        self.events.append(("share_begin",))

    def end_self_mtp_cycle(self):
        self.events.append(("share_end", self._mtp_shared_topk))
        self._mtp_share_topk = False
        self._mtp_shared_topk = None

    def prepare_self_mtp_step(self, *, lengths, right_padding):
        assert self._mtp_share_topk is True
        mode = "capture" if self._mtp_shared_topk is None else "reuse"
        self.events.append(
            ("share_prepare", mode, tuple(lengths), tuple(right_padding))
        )
        if self._mtp_shared_topk is None:
            self._mtp_shared_topk = tuple(
                f"row-{index}" for index in range(len(lengths))
            )

    def finalize_self_mtp_step(self):
        assert self._mtp_share_topk is True
        assert self._mtp_shared_topk is not None
        self.events.append(("share_finalize", self._mtp_shared_topk))


class _CacheListFake:
    """Production-shaped composite: one ordinary KV plus one QSA sidecar."""

    def __init__(self, *caches):
        self.caches = list(caches)

    @classmethod
    def merge(cls, cache_lists):
        return cls(
            *(
                type(rows[0]).merge(list(rows))
                for rows in zip(*(cache_list.caches for cache_list in cache_lists))
            )
        )


class _FakeForwards:
    def __init__(self):
        self.calls = []

    @staticmethod
    def _logits(tokens, chosen):
        batch, width = tokens.shape
        logits = np.full((batch, width, 32), -20.0)
        for row in range(batch):
            for position in range(width):
                logits[row, position, chosen(row, position)] = 20.0
        return logits

    def target(self, inputs, *, cache, return_hidden, n_confirmed):
        tokens = np.asarray(inputs)
        self.calls.append(("target", tokens.copy(), cache, return_hidden, n_confirmed))
        hidden = np.repeat(tokens[..., None].astype(float), 4, axis=-1)
        if tokens.shape[1] == 3:
            # Row 0 accepts d1 then rejects d2; row 1 accepts both.
            selected = (
                [int(tokens[0, 1]), 19, 20]
                if tokens.shape[0] == 1
                else [
                    [int(tokens[0, 1]), 19, 20],
                    [int(tokens[1, 1]), int(tokens[1, 2]), 10],
                ]
            )
            if tokens.shape[0] == 1:
                logits = self._logits(tokens, lambda _r, p: selected[p])
            else:
                logits = self._logits(tokens, lambda r, p: selected[r][p])
        else:
            logits = self._logits(tokens, lambda r, p: (int(tokens[r, p]) + 1) % 32)
        return logits, hidden

    def draft(self, hidden, token_ids, cache, *, return_hidden):
        tokens = np.asarray(token_ids)
        self.calls.append(
            ("draft", np.asarray(hidden).copy(), tokens.copy(), cache, return_hidden)
        )
        logits = self._logits(tokens, lambda r, p: (int(tokens[r, p]) + 1) % 32)
        post = np.repeat(tokens[..., None].astype(float) + 0.5, 4, axis=-1)
        return logits, post


def _runtime(*, share_qsa_indices=False, draft_cache_factory=None):
    forward = _FakeForwards()
    backend = RapidMLXSelfMTPBackend(
        target_cache_factory=lambda: [_LayerCache("target")],
        draft_cache_factory=draft_cache_factory or (lambda: [_LayerCache("draft")]),
        array_ops=_NumpyOps(),
        share_qsa_indices=share_qsa_indices,
        prefill_step_size=8,
    )
    cache_events = []

    def preflight(group, drops, **kwargs):
        cache_events.append(("preflight", tuple(drops), kwargs))

    def trim(group, drops, **kwargs):
        cache_events.append(("trim", tuple(drops), kwargs))

    cache_adapter = RapidRaggedCacheAdapter(preflight=preflight, trim=trim)
    runtime = ContinuousSelfMTPRuntime(
        config=ContinuousSelfMTPConfig(enabled=True),
        capabilities=ContinuousSelfMTPCapabilities(
            target_return_hidden=True,
            mtp_return_hidden=True,
            confirmed_target_forward=True,
            ragged_rollback=True,
            atomic_cache_commit=True,
        ),
        forwards=RapidForwardSeams(forward.target, forward.draft),
        compute=backend,
        caches=cache_adapter,
    )
    return runtime, forward, cache_events


def _prepare(runtime, uid, prompt):
    return prepare_self_mtp_lane(
        SelfMTPLaneSpec(
            uid=uid,
            prompt=prompt,
            max_tokens=12,
            num_draft=2,
        ),
        runtime,
    )


def test_k2_recursive_draft_target_verify_and_delivery_commit():
    runtime, forward, cache_events = _runtime()
    detached0, first0 = _prepare(runtime, 0, [1, 2])
    detached1, first1 = _prepare(runtime, 1, [5, 6])
    assert (first0.token, first1.token) == (3, 7)
    batch = attach_self_mtp_lanes(None, [detached0, detached1])

    proposal = propose_batched_self_mtp(batch)
    assert proposal.draft_depths == (2, 2)
    assert proposal.accepted_lengths == (1, 2)
    assert [[token.token for token in row] for row in proposal.outputs] == [
        [4, 19],
        [8, 9, 10],
    ]
    verify = [call for call in forward.calls if call[0] == "target"][-1]
    assert verify[1].tolist() == [[3, 4, 5], [7, 8, 9]]
    assert verify[-1] == 2  # Rapid n_confirmed ABI for a K=2 verify.
    assert ("trim", (1, 0), {"verify_size": 3, "validate": False}) in cache_events
    assert ("trim", (2, 2), {"verify_size": 3, "validate": False}) in cache_events

    commit_batched_self_mtp(
        batch,
        proposal,
        emitted_counts=[2, 3],
        terminal=[False, False],
    )
    assert [lane.cur for lane in batch.lanes] == [19, 10]
    assert [lane.pending_tokens for lane in batch.lanes] == [[3, 4], [7, 8, 9]]
    assert [lane.ntoks for lane in batch.lanes] == [3, 4]


def test_next_cycle_flushes_persistent_pending_pairs_before_new_drafts():
    runtime, forward, _events = _runtime()
    lane0, _ = _prepare(runtime, 0, [1, 2])
    lane1, _ = _prepare(runtime, 1, [5, 6])
    batch = attach_self_mtp_lanes(None, [lane0, lane1])
    first = propose_batched_self_mtp(batch)
    commit_batched_self_mtp(
        batch, first, emitted_counts=[2, 3], terminal=[False, False]
    )

    draft_calls_before = len([call for call in forward.calls if call[0] == "draft"])
    propose_batched_self_mtp(batch)
    draft_calls = [call for call in forward.calls if call[0] == "draft"]
    first_cycle_call = draft_calls[draft_calls_before]
    # Pending [old cur + accepted drafts] plus current bonus are flushed in one
    # recursive first-head job; rows are padded to the longest valid length.
    assert first_cycle_call[2].shape == (2, 4)
    assert first_cycle_call[2][0].tolist() == [3, 4, 19, 0]
    assert first_cycle_call[2][1].tolist() == [7, 8, 9, 10]


def test_recursive_drafts_preserve_qsa_selection_then_disarm_cycle():
    runtime, _forward, _events = _runtime(
        share_qsa_indices=True,
        draft_cache_factory=lambda: [
            _CacheListFake(_LayerCache("kv"), _SharedQSAFake("qsa"))
        ],
    )
    lane0, _ = _prepare(runtime, 0, [1, 2])
    lane1, _ = _prepare(runtime, 1, [5, 6])
    batch = attach_self_mtp_lanes(None, [lane0, lane1])
    merged_cache_list = batch.caches.draft[0]
    merged_kv, merged_qsa = merged_cache_list.caches

    propose_batched_self_mtp(batch)

    share_events = [
        event for event in merged_qsa.events if event[0].startswith("share_")
    ]
    assert [event[0] for event in share_events] == [
        "share_begin",
        "share_prepare",
        "share_finalize",
        "share_prepare",
        "share_finalize",
        "share_end",
    ]
    assert share_events[1][1] == "capture"
    assert share_events[3][1] == "reuse"
    assert share_events[-1][1] == ("row-0", "row-1")
    assert merged_qsa._mtp_share_topk is False
    assert merged_qsa._mtp_shared_topk is None
    assert [event[0] for event in merged_kv.events] == [
        "prepare",
        "finalize",
        "prepare",
        "finalize",
    ]


def test_terminal_delivery_prefix_updates_cur_seed_and_detach_flushes_debt():
    runtime, forward, _events = _runtime()
    lane0, _ = _prepare(runtime, 0, [1, 2])
    lane1, _ = _prepare(runtime, 1, [5, 6])
    batch = attach_self_mtp_lanes(None, [lane0, lane1])
    proposal = propose_batched_self_mtp(batch)
    commit_batched_self_mtp(
        batch,
        proposal,
        emitted_counts=[1, 2],
        terminal=[True, True],
    )
    assert [lane.cur for lane in batch.lanes] == [4, 9]
    assert [lane.pending_tokens for lane in batch.lanes] == [[3], [7, 8]]

    draft_count = len([call for call in forward.calls if call[0] == "draft"])
    batch, detached = detach_self_mtp_lanes(batch, [0, 1])
    assert batch.lanes == []
    assert [item.lane.uid for item in detached] == [0, 1]
    flushes = [call for call in forward.calls if call[0] == "draft"][draft_count:]
    assert [call[2].tolist() for call in flushes] == [[[3]], [[7, 8]]]
    assert all(item.lane.pending_tokens == [] for item in detached)


def test_transformed_sampling_fails_closed_without_residual_hooks():
    runtime, _forward, _events = _runtime()
    runtime = ContinuousSelfMTPRuntime(
        config=runtime.config,
        capabilities=ContinuousSelfMTPCapabilities(
            target_return_hidden=True,
            mtp_return_hidden=True,
            confirmed_target_forward=True,
            ragged_rollback=True,
            atomic_cache_commit=True,
            transformed_sampling=True,
        ),
        forwards=runtime.forwards,
        compute=runtime.compute,
        caches=runtime.caches,
    )
    with pytest.raises(ContinuousSelfMTPUnsupported, match="residual hooks"):
        prepare_self_mtp_lane(
            SelfMTPLaneSpec(
                uid=0,
                prompt=[1, 2],
                max_tokens=8,
                num_draft=2,
                sampling=SelfMTPSampling(temperature=0.8),
            ),
            runtime,
        )


def test_cache_adapter_merge_extend_extract_filter_and_atomic_preflight():
    events = []
    adapter = RapidRaggedCacheAdapter(
        preflight=lambda group, drops, **kwargs: events.append(
            ("preflight", tuple(drops), kwargs)
        ),
        trim=lambda group, drops, **kwargs: events.append(
            ("trim", tuple(drops), kwargs)
        ),
    )
    one = SelfMTPCachePair([_LayerCache("t1")], [_LayerCache("d1")])
    two = SelfMTPCachePair([_LayerCache("t2")], [_LayerCache("d2")])
    merged = adapter.attach(None, [one, two])
    assert merged.target[0].rows == ["t1", "t2"]
    assert merged.draft[0].rows == ["d1", "d2"]

    adapter.rollback(
        merged,
        target_drops=[1, 0],
        draft_drops=[2, 2],
        verify_width=3,
    )
    assert [event[0] for event in events] == [
        "preflight",
        "preflight",
        "trim",
        "trim",
    ]
    remaining, detached = adapter.detach(merged, [1], [0])
    assert remaining.target[0].rows == ["t1"]
    assert detached[0].target[0].rows == ["t2"]

    third = SelfMTPCachePair([_LayerCache("t3")], [_LayerCache("d3")])
    adapter.attach(remaining, [third])
    assert remaining.target[0].rows == ["t1", "t3"]


@pytest.mark.parametrize("name", ["QuantizedKVCache", "SinkWindowKVCache"])
def test_cache_adapter_rejects_quantized_and_windowed_classes(name):
    unsupported = type(name, (_LayerCache,), {})
    pair = SelfMTPCachePair([unsupported("target")], [_LayerCache("draft")])
    adapter = RapidRaggedCacheAdapter(
        preflight=lambda *a, **k: None, trim=lambda *a, **k: None
    )
    with pytest.raises(ContinuousSelfMTPUnsupported, match="unsupported"):
        adapter.attach(None, [pair])


def test_backend_has_no_eager_mlx_import_and_uses_only_rapid_forward_seams():
    source = (
        Path(__file__).parents[1]
        / "vllm_mlx"
        / "spec_decode"
        / "mtp"
        / "mlx_backend.py"
    ).read_text()
    tree = ast.parse(source)
    eager_mlx = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name.startswith("mlx") for alias in getattr(node, "names", []))
            or getattr(node, "module", "") == "mlx.core"
        )
    ]
    assert eager_mlx == []
    assert "forwards.target(" in source
    assert "forwards.draft(" in source


def test_qwen4_qsa_source_has_cycle_local_capture_reuse_and_cleanup():
    cache_source = (ROOT / "vllm_mlx/models/qwen4_exp_cache.py").read_text()
    model_source = (ROOT / "vllm_mlx/models/qwen4_exp.py").read_text()

    assert "def begin_self_mtp_cycle" in cache_source
    assert "def prepare_self_mtp_step" in cache_source
    assert "def finalize_self_mtp_step" in cache_source
    assert "def end_self_mtp_cycle" in cache_source
    assert "capture_shared = cache._mtp_share_topk" in model_source
    assert "cache._mtp_shared_topk = mx.stack(shared_rows)" in model_source
    assert "shared_topk[batch_index : batch_index + 1]" in model_source
