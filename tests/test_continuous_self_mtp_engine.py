"""Mock-only tests for the fixed-membership continuous self-MTP engine."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "vllm_mlx"
    / "spec_decode"
    / "mtp"
    / "continuous_engine.py"
)
SPEC = importlib.util.spec_from_file_location("continuous_engine_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
engine = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = engine
SPEC.loader.exec_module(engine)


def _capabilities(**overrides):
    values = {
        "target_return_hidden": True,
        "mtp_return_hidden": True,
        "confirmed_target_forward": True,
        "ragged_rollback": True,
        "atomic_cache_commit": True,
    }
    values.update(overrides)
    return engine.ContinuousSelfMTPCapabilities(**values)


class _Compute:
    def __init__(self):
        self.calls = []

    def prepare(self, spec, forwards):
        target = forwards.target(spec.prompt, f"target-{spec.uid}", n_confirmed=0)
        draft = forwards.draft(target[1], spec.prompt, f"draft-{spec.uid}")
        self.calls.append(("prepare", spec.uid, target, draft))
        token = 100 + spec.uid
        return engine.PreparedLaneData(
            cur=token,
            seed_hidden=f"hidden-{spec.uid}",
            token_prefix=(spec.uid,),
            caches=engine.SelfMTPCachePair(
                target=[f"target-{spec.uid}"], draft=[f"draft-{spec.uid}"]
            ),
            first_token=engine.MTPToken(token, f"lp-{token}", False),
            backend_state={"uid": spec.uid},
        )

    def propose(self, lanes, caches, forwards):
        del caches, forwards
        self.calls.append(("propose", tuple(lane.uid for lane in lanes)))
        outputs = []
        accepted = []
        for row, lane in enumerate(lanes):
            n_accept = 1 if row == 0 else 0
            accepted.append(n_accept)
            row_outputs = [engine.MTPToken(lane.cur + 1, f"lp-draft-{lane.uid}", True)][
                :n_accept
            ]
            row_outputs.append(
                engine.MTPToken(lane.cur + 10, f"lp-target-{lane.uid}", False)
            )
            outputs.append(tuple(row_outputs))
        depths = tuple(2 for _ in lanes)
        accepted_tuple = tuple(accepted)
        return engine.CycleComputation(
            lane_uids=tuple(lane.uid for lane in lanes),
            draft_depths=depths,
            accepted_lengths=accepted_tuple,
            target_drops=tuple(2 - count for count in accepted_tuple),
            draft_drops=depths,
            outputs=tuple(outputs),
            payload="opaque-cycle",
        )

    def commit(self, lanes, computation, *, emitted_counts, terminal):
        self.calls.append(("commit", computation.payload, emitted_counts, terminal))
        for lane, outputs in zip(lanes, computation.outputs):
            if outputs:
                lane.cur = outputs[-1].token

    def detach_lane(self, lane, caches):
        self.calls.append(("detach", lane.uid, caches.target, caches.draft))


class _Caches:
    def __init__(self):
        self.calls = []

    def attach(self, current, joining):
        self.calls.append(("attach", current, tuple(joining)))
        target = [] if current is None else list(current.target)
        draft = [] if current is None else list(current.draft)
        for item in joining:
            target.extend(item.target)
            draft.extend(item.draft)
        return engine.SelfMTPCachePair(target, draft)

    def rollback(self, caches, *, target_drops, draft_drops, verify_width):
        self.calls.append(
            (
                "rollback",
                tuple(target_drops),
                tuple(draft_drops),
                verify_width,
            )
        )

    def detach(self, caches, indices, keep_indices):
        self.calls.append(("detach", tuple(indices), tuple(keep_indices)))
        detached = [
            engine.SelfMTPCachePair([caches.target[index]], [caches.draft[index]])
            for index in indices
        ]
        remaining = engine.SelfMTPCachePair(
            [caches.target[index] for index in keep_indices],
            [caches.draft[index] for index in keep_indices],
        )
        return remaining, detached


def _runtime(*, config=None, capabilities=None):
    calls = []

    def target(inputs, **kwargs):
        calls.append(("target", inputs, kwargs))
        return "target-logits", f"target-hidden-{inputs}"

    def draft(hidden, token_ids, cache, **kwargs):
        calls.append(("draft", hidden, token_ids, cache, kwargs))
        return "draft-logits", "draft-hidden"

    compute = _Compute()
    caches = _Caches()
    runtime = engine.ContinuousSelfMTPRuntime(
        config=config or engine.ContinuousSelfMTPConfig(enabled=True),
        capabilities=capabilities or _capabilities(),
        forwards=engine.RapidForwardSeams(target, draft),
        compute=compute,
        caches=caches,
    )
    return runtime, compute, caches, calls


def _prepare(runtime, uid, **spec_kwargs):
    spec = engine.SelfMTPLaneSpec(
        uid=uid,
        prompt=(uid, uid + 1),
        max_tokens=20,
        num_draft=2,
        **spec_kwargs,
    )
    return engine.prepare_self_mtp_lane(spec, runtime)


def test_rapid_forward_seams_use_return_hidden_and_n_confirmed():
    runtime, _compute, _caches, calls = _runtime()
    detached, first = _prepare(runtime, 1)

    assert first.token == detached.lane.cur == 101
    assert calls[0] == (
        "target",
        (1, 2),
        {
            "cache": "target-1",
            "return_hidden": True,
            "n_confirmed": 0,
        },
    )
    assert calls[1][-1] == {"return_hidden": True}


def test_fixed_membership_prepare_attach_propose_commit_detach_lifecycle():
    runtime, compute, caches, _calls = _runtime()
    lane1, first1 = _prepare(runtime, 1)
    lane2, first2 = _prepare(runtime, 2)
    assert (first1.token, first2.token) == (101, 102)

    batch = engine.attach_self_mtp_lanes(None, [lane1, lane2])
    assert [lane.uid for lane in batch.lanes] == [1, 2]
    assert batch.membership_epoch == 1

    proposal = engine.propose_batched_self_mtp(batch)
    assert proposal.lane_uids == (1, 2)
    assert proposal.accepted_lengths == (1, 0)
    assert caches.calls[-1] == ("rollback", (1, 2), (2, 2), 3)
    with pytest.raises(engine.ContinuousSelfMTPError, match="proposal is open"):
        engine.detach_self_mtp_lanes(batch, [0, 1])

    engine.commit_batched_self_mtp(
        batch,
        proposal,
        emitted_counts=[2, 1],
        terminal=[False, False],
    )
    assert [lane.ntoks for lane in batch.lanes] == [3, 2]
    assert compute.calls[-1] == (
        "commit",
        "opaque-cycle",
        (2, 1),
        (False, False),
    )

    previous_epoch = batch.membership_epoch
    batch, detached = engine.detach_self_mtp_lanes(batch, [0, 1])
    assert batch.lanes == []
    assert batch.membership_epoch == previous_epoch + 1
    assert [item.lane.uid for item in detached] == [1, 2]


def test_default_off_refuses_before_compute():
    runtime, compute, _caches, _calls = _runtime(
        config=engine.ContinuousSelfMTPConfig()
    )
    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="disabled"):
        _prepare(runtime, 1)
    assert compute.calls == []


def test_xtc_is_unconditionally_fail_closed():
    runtime, compute, _caches, _calls = _runtime(
        capabilities=_capabilities(
            transformed_sampling=True,
            logits_processors_exact=True,
            dynamic_membership=True,
            flash_dynamic_membership_attested=True,
        )
    )
    sampling = engine.SelfMTPSampling(temperature=0.8, uses_xtc=True)
    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="XTC"):
        _prepare(runtime, 1, sampling=sampling)
    assert compute.calls == []


def test_fixed_membership_refuses_incremental_attach_and_partial_detach():
    runtime, _compute, _caches, _calls = _runtime()
    lane1, _ = _prepare(runtime, 1)
    lane2, _ = _prepare(runtime, 2)
    batch = engine.attach_self_mtp_lanes(None, [lane1, lane2])
    lane3, _ = _prepare(runtime, 3)

    with pytest.raises(
        engine.ContinuousSelfMTPUnsupportedError, match="fixed-membership"
    ):
        engine.attach_self_mtp_lanes(batch, [lane3])
    with pytest.raises(
        engine.ContinuousSelfMTPUnsupportedError, match="fixed-membership"
    ):
        engine.detach_self_mtp_lanes(batch, [1])


def test_flash_dynamic_membership_requires_specific_attestation():
    config = engine.ContinuousSelfMTPConfig(
        enabled=True,
        allow_dynamic_membership=True,
        architecture="qwen4-flash-next",
    )
    runtime, _compute, _caches, _calls = _runtime(
        config=config,
        capabilities=_capabilities(dynamic_membership=True),
    )
    lane1, _ = _prepare(runtime, 1)
    batch = engine.attach_self_mtp_lanes(None, [lane1])
    lane2, _ = _prepare(runtime, 2)
    with pytest.raises(engine.ContinuousSelfMTPUnsupportedError, match="Flash dynamic"):
        engine.attach_self_mtp_lanes(batch, [lane2])


def test_attested_flash_runtime_can_use_explicit_dynamic_seam():
    config = engine.ContinuousSelfMTPConfig(
        enabled=True,
        allow_dynamic_membership=True,
        architecture="qwen4-flash-next",
    )
    runtime, _compute, _caches, _calls = _runtime(
        config=config,
        capabilities=_capabilities(
            dynamic_membership=True,
            flash_dynamic_membership_attested=True,
        ),
    )
    lane1, _ = _prepare(runtime, 1)
    lane2, _ = _prepare(runtime, 2)
    batch = engine.attach_self_mtp_lanes(None, [lane1])
    batch = engine.attach_self_mtp_lanes(batch, [lane2])
    assert [lane.uid for lane in batch.lanes] == [1, 2]
    batch, detached = engine.detach_self_mtp_lanes(batch, [1])
    assert [lane.uid for lane in batch.lanes] == [1]
    assert detached[0].lane.uid == 2


def test_commit_rejects_partial_nonterminal_delivery_without_closing_cycle():
    runtime, _compute, _caches, _calls = _runtime()
    lane1, _ = _prepare(runtime, 1)
    lane2, _ = _prepare(runtime, 2)
    batch = engine.attach_self_mtp_lanes(None, [lane1, lane2])
    proposal = engine.propose_batched_self_mtp(batch)

    with pytest.raises(ValueError, match="nonterminal lane"):
        engine.commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[1, 1],
            terminal=[False, False],
        )
    assert batch.proposal_open is True
    assert [lane.ntoks for lane in batch.lanes] == [1, 1]


@pytest.mark.parametrize(
    ("emitted", "terminal", "message"),
    [
        ([True, 1], [False, False], "emitted_counts values"),
        ([1.0, 1], [False, False], "emitted_counts values"),
        ([2, 1], [0, False], "terminal values"),
        ([2, 1], ["yes", False], "terminal values"),
    ],
)
def test_commit_rejects_coercible_vector_values(emitted, terminal, message):
    runtime, _compute, _caches, _calls = _runtime()
    lane1, _ = _prepare(runtime, 1)
    lane2, _ = _prepare(runtime, 2)
    batch = engine.attach_self_mtp_lanes(None, [lane1, lane2])
    proposal = engine.propose_batched_self_mtp(batch)

    with pytest.raises(ValueError, match=message):
        engine.commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=emitted,
            terminal=terminal,
        )
    assert batch.proposal_open is True
    assert [lane.ntoks for lane in batch.lanes] == [1, 1]
