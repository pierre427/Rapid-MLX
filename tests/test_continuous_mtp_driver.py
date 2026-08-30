"""Pure-Python coordination tests for continuous self-MTP delivery."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

MTP_DIR = Path(__file__).parents[1] / "vllm_mlx" / "spec_decode" / "mtp"
PACKAGE = "_continuous_mtp_driver_probe"
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
driver = _load("continuous_driver")


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
        self.calls.append(
            (
                "attach",
                None if current is None else tuple(current.target),
                tuple(pair.target for pair in joining),
            )
        )
        target = [] if current is None else list(current.target)
        draft = [] if current is None else list(current.draft)
        target.extend(pair.target for pair in joining)
        draft.extend(pair.draft for pair in joining)
        return engine.SelfMTPCachePair(target=target, draft=draft)

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


def _runtime(*, dynamic=True):
    compute = _Compute()
    caches = _Caches()
    runtime = engine.ContinuousSelfMTPRuntime(
        config=engine.ContinuousSelfMTPConfig(
            enabled=True,
            allow_dynamic_membership=dynamic,
            architecture="qwen3_5",
        ),
        capabilities=engine.ContinuousSelfMTPCapabilities(
            target_return_hidden=True,
            mtp_return_hidden=True,
            confirmed_target_forward=True,
            ragged_rollback=True,
            atomic_cache_commit=True,
            dynamic_membership=dynamic,
            flash_dynamic_membership_attested=dynamic,
        ),
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


def _triples(responses):
    return [
        (response.uid, response.token, response.finish_reason) for response in responses
    ]


def _drive_length_boundary(*, lane_count: int, max_tokens: int = 256):
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create(
        [_spec(uid, max_tokens=max_tokens) for uid in range(1, lane_count + 1)],
        runtime,
    )
    delivered = {uid: [] for uid in range(1, lane_count + 1)}
    for response in batch_driver.next():
        delivered[response.uid].append(response)
    compute.queued_outputs.extend(
        [
            [(_target(uid * 1000 + step),) for uid in delivered]
            for step in range(1, max_tokens)
        ]
    )
    for _ in range(1, max_tokens):
        for response in batch_driver.next():
            delivered[response.uid].append(response)
    return batch_driver, compute, delivered


def test_variable_burst_queues_and_drains_one_response_per_uid_before_next_step():
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1), _spec(2)], runtime)

    assert _triples(batch_driver.next()) == [(1, 101, None), (2, 102, None)]
    compute.queued_outputs.extend(
        [
            [
                (_draft(111), _draft(112), _target(113)),
                (_draft(211), _target(212)),
            ],
            [(_target(114),), (_target(213),)],
        ]
    )

    first = batch_driver.next()
    second = batch_driver.next()
    third = batch_driver.next()

    assert _triples(first) == [(1, 111, None), (2, 211, None)]
    assert _triples(second) == [(1, 112, None), (2, 212, None)]
    assert _triples(third) == [(1, 113, None)]
    assert sum(call[0] == "propose" for call in compute.calls) == 1
    assert batch_driver.has_pending_responses is False

    assert _triples(batch_driver.next()) == [(1, 114, None), (2, 213, None)]
    assert sum(call[0] == "propose" for call in compute.calls) == 2


def test_b1_target_only_terminal_response_drains_before_driver_closes() -> None:
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1, max_tokens=2)], runtime)
    assert _triples(batch_driver.next()) == [(1, 101, None)]
    compute.queued_outputs.append([(_target(111),)])

    final = batch_driver.next()

    assert _triples(final) == [(1, 111, "length")]
    assert final[0].prompt_cache == "target-cache-1"
    assert final[0].all_tokens == [101, 111]
    assert final[0].mtp_state == ("draft-cache-1", "hidden-1")
    assert batch_driver.closed is True
    assert batch_driver.has_pending_responses is False
    assert batch_driver.next() == []
    assert [package.uid for package in batch_driver.take_terminal_detaches()] == [1]


def test_256_token_boundary_delivers_final_token_and_trailer_once_for_b1_and_b2():
    """Pin the live 255-token/missing-trailer failure at its observed size."""
    for lane_count in (1, 2):
        batch_driver, compute, delivered = _drive_length_boundary(lane_count=lane_count)

        assert batch_driver.closed is True
        assert batch_driver.has_pending_responses is False
        assert batch_driver.next() == []
        assert sum(call[0] == "propose" for call in compute.calls) == 255
        for uid, responses in delivered.items():
            assert len(responses) == 256
            assert all(response.finish_reason is None for response in responses[:-1])
            final = responses[-1]
            assert final.finish_reason == "length"
            assert final.all_tokens == [response.token for response in responses]
            assert len(final.all_tokens) == 256
            assert final.prompt_cache == f"target-cache-{uid}"
            assert final.mtp_state == (f"draft-cache-{uid}", f"hidden-{uid}")


def test_256th_continuous_token_drives_scheduler_finish_and_usage_count():
    """The terminal driver response is the scheduler's finish/usage trigger."""
    from unittest.mock import MagicMock

    from vllm_mlx.request import Request, RequestStatus, SamplingParams
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    _batch_driver, _compute, delivered = _drive_length_boundary(lane_count=1)
    responses = delivered[1]
    terminal = responses[-1]

    scheduler = Scheduler(
        MagicMock(),
        MagicMock(encode=lambda value: list(range(len(value.split())))),
        SchedulerConfig(max_num_seqs=1),
    )
    request = Request(
        request_id="boundary-256",
        prompt="hello",
        sampling_params=SamplingParams(max_tokens=256),
    )
    request.status = RequestStatus.RUNNING
    request.num_prompt_tokens = 1
    for response in responses[:-1]:
        request.append_output_token(response.token)
    scheduler.running[request.request_id] = request
    scheduler.uid_to_request_id[terminal.uid] = request.request_id
    scheduler._decode_tokens = lambda tokens: "x" * len(tokens)  # type: ignore[method-assign]

    outputs, finished = scheduler._process_batch_responses([terminal])

    assert finished == {request.request_id}
    assert len(outputs) == 1
    assert outputs[0].finished is True
    assert outputs[0].finish_reason == "length"
    assert outputs[0].completion_tokens == 256
    assert len(outputs[0].output_token_ids) == 256
    assert scheduler.total_completion_tokens == 256
    assert scheduler.num_requests_processed == 1


def test_join_waits_for_delivery_drain_then_emits_joined_initial_before_proposal():
    runtime, compute, caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1)], runtime)
    batch_driver.next()
    compute.queued_outputs.append([(_draft(111), _target(112))])

    assert _triples(batch_driver.next()) == [(1, 111, None)]
    assert batch_driver.queue_lanes([_spec(2)]) == (2,)
    assert not any(call == ("prepare", 2) for call in compute.calls)

    # Lane 1 still has a committed response queued, so lane 2 cannot attach.
    assert _triples(batch_driver.next()) == [(1, 112, None)]
    assert batch_driver.last_attached_uids == ()
    assert batch_driver.pending_join_uids == (2,)

    joined_initial = batch_driver.next()
    assert _triples(joined_initial) == [(2, 102, None)]
    assert batch_driver.last_attached_uids == (2,)
    assert batch_driver.lane_uids == (1, 2)
    assert caches.calls[-1] == (
        "attach",
        ("target-cache-1",),
        ("target-cache-2",),
    )
    assert sum(call[0] == "propose" for call in compute.calls) == 1

    compute.queued_outputs.append([(_target(113),), (_target(213),)])
    assert [response.uid for response in batch_driver.next()] == [1, 2]
    assert ("propose", (1, 2)) in compute.calls


def test_terminal_detach_is_published_on_final_response_and_survivor_continues():
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create(
        [_spec(1), _spec(2)], runtime, stop_tokens={1: {112}}
    )
    batch_driver.next()
    compute.queued_outputs.append([(_draft(111), _target(112)), (_target(211),)])

    first = batch_driver.next()
    assert _triples(first) == [(1, 111, None), (2, 211, None)]
    assert batch_driver.take_terminal_detaches() == ()
    assert batch_driver.lane_uids == (2,)

    terminal = batch_driver.next()
    assert _triples(terminal) == [(1, 112, "stop")]
    assert terminal[0].prompt_cache == "target-cache-1"
    assert terminal[0].all_tokens == [101, 111, 112]
    assert terminal[0].mtp_state == ("draft-cache-1", "hidden-1")
    assert terminal[0].mtp_cache_tokens == [1]
    packages = batch_driver.take_terminal_detaches()
    assert [package.uid for package in packages] == [1]
    assert batch_driver.take_resumable_detaches() == ()

    compute.queued_outputs.append([(_target(212),)])
    assert _triples(batch_driver.next()) == [(2, 212, None)]
    assert batch_driver.closed is False


def test_fixed_cohort_turnover_holds_survivor_until_its_burst_is_drained():
    runtime, compute, _caches = _runtime(dynamic=False)
    batch_driver = driver.ContinuousMTPDriver.create(
        [_spec(1, max_tokens=3), _spec(2)], runtime
    )
    batch_driver.next()
    compute.queued_outputs.append(
        [(_draft(111), _target(112)), (_draft(211), _target(212))]
    )

    first = batch_driver.next()
    assert _triples(first) == [(1, 111, None), (2, 211, None)]
    assert batch_driver.closed is True
    assert batch_driver.has_work is True
    assert batch_driver.take_resumable_detaches() == ()

    final = batch_driver.next()
    assert _triples(final) == [(1, 112, "length"), (2, 212, None)]
    terminal = batch_driver.take_terminal_detaches()
    assert [package.uid for package in terminal] == [1]
    assert batch_driver.resume_turnover() == (2,)
    assert batch_driver.closed is False
    assert batch_driver.take_resumable_detaches() == ()

    compute.queued_outputs.append([(_target(213),)])
    assert _triples(batch_driver.next()) == [(2, 213, None)]


def test_manual_shutdown_discards_queued_delivery_without_resumable_turnover():
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1)], runtime)
    batch_driver.next()
    compute.queued_outputs.append([(_draft(111), _target(112))])
    batch_driver.next()

    packages = batch_driver.discard_all()
    assert [package.uid for package in packages] == [1]
    assert batch_driver.closed is True
    assert batch_driver.has_pending_responses is False
    assert batch_driver.next() == []
    assert batch_driver.take_resumable_detaches() == ()


def test_remove_uids_discards_cancelled_queue_and_keeps_dynamic_survivor():
    runtime, compute, _caches = _runtime()
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1), _spec(2)], runtime)
    batch_driver.next()
    compute.queued_outputs.append(
        [(_draft(111), _target(112)), (_draft(211), _target(212))]
    )
    assert _triples(batch_driver.next()) == [(1, 111, None), (2, 211, None)]

    removed = batch_driver.remove_uids([1])
    assert [package.uid for package in removed] == [1]
    assert removed[0].terminal is False
    assert removed[0].target_cache == "target-cache-1"
    assert batch_driver.lane_uids == (2,)

    # Lane 1's remaining committed token is discarded; the survivor drains its
    # own queued token before another proposal can run.
    assert _triples(batch_driver.next()) == [(2, 212, None)]
    assert sum(call[0] == "propose" for call in compute.calls) == 1
    compute.queued_outputs.append([(_target(213),)])
    assert _triples(batch_driver.next()) == [(2, 213, None)]


def test_fixed_remove_turns_over_companion_without_transferring_its_ownership():
    runtime, compute, _caches = _runtime(dynamic=False)
    batch_driver = driver.ContinuousMTPDriver.create([_spec(1), _spec(2)], runtime)
    batch_driver.next()

    removed = batch_driver.remove_uids([1])
    assert [package.uid for package in removed] == [1]
    assert batch_driver.closed is True
    assert batch_driver.resume_turnover() == (2,)
    assert batch_driver.take_resumable_detaches() == ()

    compute.queued_outputs.append([(_target(211),)])
    assert _triples(batch_driver.next()) == [(2, 211, None)]
    assert not any(call == ("prepare", 2) for call in compute.calls[2:])
