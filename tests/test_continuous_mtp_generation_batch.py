"""Pure-Python contract tests for the continuous MTP generation wrapper."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

MTP_DIR = Path(__file__).parents[1] / "vllm_mlx" / "spec_decode" / "mtp"
PACKAGE = "_continuous_mtp_generation_batch_probe"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(MTP_DIR)]
sys.modules[PACKAGE] = package


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{name}", MTP_DIR / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = _load("continuous_engine")
generation = _load("continuous_batch")


class _Compute:
    def __init__(self):
        self.calls = []
        self.queued_outputs = []

    def prepare(self, spec, forwards):
        del forwards
        self.calls.append(("prepare", spec.uid))
        token = 100 + spec.uid
        return engine.PreparedLaneData(
            cur=token,
            seed_hidden=f"hidden-{spec.uid}",
            token_prefix=(spec.uid,),
            caches=engine.SelfMTPCachePair(
                target=f"target-cache-{spec.uid}",
                draft=f"draft-cache-{spec.uid}",
            ),
            first_token=engine.MTPToken(token, f"lp-{token}", False),
        )

    def propose(self, lanes, caches, forwards):
        del caches, forwards
        self.calls.append(("propose", tuple(lane.uid for lane in lanes)))
        rows = self.queued_outputs.pop(0)
        accepted = tuple(
            sum(1 for token in row[:-1] if token.from_draft) for row in rows
        )
        return engine.CycleComputation(
            lane_uids=tuple(lane.uid for lane in lanes),
            draft_depths=accepted,
            accepted_lengths=accepted,
            target_drops=tuple(0 for _ in lanes),
            draft_drops=accepted,
            outputs=tuple(tuple(row) for row in rows),
            payload=f"cycle-{len(self.calls)}",
        )

    def commit(self, lanes, computation, *, emitted_counts, terminal):
        self.calls.append(("commit", emitted_counts, terminal))
        for lane, outputs, count in zip(lanes, computation.outputs, emitted_counts):
            if count:
                lane.cur = outputs[count - 1].token

    def detach_lane(self, lane, caches):
        self.calls.append(("detach", lane.uid, caches.target, caches.draft))


class _Caches:
    def __init__(self):
        self.calls = []

    def attach(self, current, joining):
        assert current is None
        self.calls.append(("attach", tuple(joining)))
        return engine.SelfMTPCachePair(
            target=[pair.target for pair in joining],
            draft=[pair.draft for pair in joining],
        )

    def rollback(self, caches, *, target_drops, draft_drops, verify_width):
        del caches
        self.calls.append(
            ("rollback", tuple(target_drops), tuple(draft_drops), verify_width)
        )

    def detach(self, caches, indices, keep_indices):
        self.calls.append(("detach", tuple(indices), tuple(keep_indices)))
        detached = [
            engine.SelfMTPCachePair(
                target=caches.target[index], draft=caches.draft[index]
            )
            for index in indices
        ]
        remaining = engine.SelfMTPCachePair(
            target=[caches.target[index] for index in keep_indices],
            draft=[caches.draft[index] for index in keep_indices],
        )
        return remaining, detached


def _runtime(*, flash_dynamic=False):
    compute = _Compute()
    caches = _Caches()
    capabilities = engine.ContinuousSelfMTPCapabilities(
        target_return_hidden=True,
        mtp_return_hidden=True,
        confirmed_target_forward=True,
        ragged_rollback=True,
        atomic_cache_commit=True,
        dynamic_membership=flash_dynamic,
        flash_dynamic_membership_attested=flash_dynamic,
    )
    runtime = engine.ContinuousSelfMTPRuntime(
        config=engine.ContinuousSelfMTPConfig(
            enabled=True,
            allow_dynamic_membership=flash_dynamic,
            architecture="qwen4_flash_next" if flash_dynamic else "qwen3_5",
        ),
        capabilities=capabilities,
        forwards=engine.RapidForwardSeams(
            lambda *args, **kwargs: None,
            lambda *args, **kwargs: None,
        ),
        compute=compute,
        caches=caches,
    )
    return runtime, compute, caches


def _spec(uid, *, max_tokens=8):
    return engine.SelfMTPLaneSpec(
        uid=uid,
        prompt=(uid,),
        max_tokens=max_tokens,
        num_draft=2,
    )


def _draft(token):
    return engine.MTPToken(token, f"lp-{token}", True)


def _target(token):
    return engine.MTPToken(token, f"lp-{token}", False)


def test_initial_tokens_are_emitted_once_and_terminal_detaches_whole_cohort():
    runtime, compute, caches = _runtime()
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1, max_tokens=1), _spec(2)], runtime
    )

    burst = batch.next_burst()

    assert burst.initial is True
    assert burst.emitted_counts == (1, 1)
    assert [emission.token_ids for emission in burst.emissions] == [(101,), (102,)]
    assert burst.emissions[0].finish_reason == "length"
    assert burst.emissions[1].finish_reason is None
    assert [package.uid for package in burst.terminal_detaches] == [1]
    assert [package.uid for package in burst.resumable_detaches] == [2]
    assert burst.resumable_detaches[0].token_ids == (102,)
    assert batch.closed is True
    assert not any(call[0] == "propose" for call in compute.calls)
    assert caches.calls[-1] == ("detach", (0, 1), ())


def test_one_proposal_burst_commits_exact_stop_prefix_then_extracts_caches():
    runtime, compute, _caches = _runtime()
    compute.queued_outputs.append([(_draft(111), _target(112)), (_target(212),)])
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1), _spec(2)], runtime, stop_tokens={1: {111}}
    )
    batch.next_burst()  # prepared first-token cohort

    burst = batch.next_burst()

    assert burst.initial is False
    assert burst.emitted_counts == (1, 1)
    assert [emission.token_ids for emission in burst.emissions] == [(111,), (212,)]
    assert ("commit", (1, 1), (True, False)) in compute.calls
    terminal = burst.terminal_detaches[0]
    assert terminal.uid == 1
    assert terminal.finish_reason == "stop"
    assert terminal.token_ids == (101, 111)
    assert terminal.target_cache == "target-cache-1"
    assert terminal.draft_cache == "draft-cache-1"
    companion = burst.resumable_detaches[0]
    assert companion.uid == 2
    assert companion.token_ids == (102, 212)
    assert companion.terminal is False


def test_max_token_boundary_marks_the_full_final_proposal_as_length():
    runtime, compute, _caches = _runtime()
    compute.queued_outputs.append([(_draft(111), _target(112)), (_target(212),)])
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1, max_tokens=3), _spec(2)], runtime
    )
    batch.next_burst()

    burst = batch.next_burst()

    assert burst.emitted_counts == (2, 1)
    assert burst.emissions[0].token_ids == (111, 112)
    assert burst.emissions[0].finish_reason == "length"
    assert ("commit", (2, 1), (True, False)) in compute.calls
    assert burst.terminal_detaches[0].token_ids == (101, 111, 112)


def test_one_proposal_per_call_and_manual_detach_is_idempotent():
    runtime, compute, _caches = _runtime()
    compute.queued_outputs.extend(
        [
            [(_target(111),), (_target(211),)],
            [(_target(112),), (_target(212),)],
        ]
    )
    batch = generation.ContinuousMTPGenerationBatch.create(
        [_spec(1), _spec(2)], runtime
    )

    initial = batch.next_burst()
    first = batch.next_burst()
    second = batch.next_burst()

    assert initial.initial is True
    assert first.initial is second.initial is False
    assert sum(call[0] == "propose" for call in compute.calls) == 2
    assert [state.emitted_tokens for state in batch.lane_states] == [3, 3]
    detached = batch.detach_all()
    assert [package.token_ids for package in detached] == [
        (101, 111, 112),
        (102, 211, 212),
    ]
    assert all(not package.terminal for package in detached)
    assert batch.detach_all() is detached
    with pytest.raises(
        generation.ContinuousMTPGenerationBatchError, match="already detached"
    ):
        batch.next_burst()


def test_failed_commit_does_not_publish_delivery_ledger():
    runtime, compute, _caches = _runtime()
    compute.queued_outputs.append([(_draft(111), _target(112))])
    batch = generation.ContinuousMTPGenerationBatch.create([_spec(1)], runtime)
    batch.next_burst()

    def fail_commit(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("commit failed")

    compute.commit = fail_commit
    with pytest.raises(RuntimeError, match="commit failed"):
        batch.next_burst()

    assert batch.lane_states[0].emitted_tokens == 1
    assert batch.closed is False


def test_flash_incremental_join_stays_impossible_despite_core_attestation():
    runtime, _compute, _caches = _runtime(flash_dynamic=True)
    batch = generation.ContinuousMTPGenerationBatch.create([_spec(1)], runtime)

    with pytest.raises(
        engine.ContinuousSelfMTPUnsupportedError,
        match="fixed-cohort.*including Flash",
    ):
        batch.attach_lanes([])


def test_stop_token_configuration_fails_closed_on_unknown_lane_or_bad_token():
    runtime, _compute, _caches = _runtime()
    with pytest.raises(ValueError, match="unknown lane uid"):
        generation.ContinuousMTPGenerationBatch.create(
            [_spec(1)], runtime, stop_tokens={2: {200}}
        )
    with pytest.raises(ValueError, match="must be integers"):
        generation.ContinuousMTPGenerationBatch.create(
            [_spec(1)], runtime, stop_tokens={1: {True}}
        )
