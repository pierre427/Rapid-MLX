"""Pure-Python contracts for the vendored MTP batch handoff.

The production function is AST-extracted so these tests never import MLX or
construct a model.  Fake arrays exercise only the scheduler state machine.
"""

from __future__ import annotations

import ast
import sys
import types
from collections import deque
from pathlib import Path
from unittest.mock import patch

import pytest

SCHEDULER = Path(__file__).parents[1] / "vllm_mlx" / "scheduler.py"


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Array:
    def __init__(self, values):
        self.values = list(values)
        self.dtype = "uint32"

    def __getitem__(self, index):
        return _Scalar(self.values[index])

    def astype(self, _dtype):
        return self


class _MX:
    uint32 = "uint32"

    @staticmethod
    def array(values, dtype=None):
        del dtype
        return _Array(values)


class _Logger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _extract_function(name: str, *, class_name: str | None = None):
    tree = ast.parse(SCHEDULER.read_text())
    body = tree.body
    if class_name is not None:
        cls = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        body = cls.body
    fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    namespace = {
        "__name__": "vllm_mlx.scheduler_test_probe",
        "__package__": "vllm_mlx",
        "logger": _Logger(),
        "mx": _MX,
    }
    exec(compile(ast.fix_missing_locations(module), str(SCHEDULER), "exec"), namespace)
    return namespace[name]


def _install_with_yields(yields):
    controller = types.ModuleType("vllm_mlx.spec_decode.mtp.draft_k_controller_v2")
    controller.derive_controller_key = lambda model: "test-model"
    generator = types.ModuleType("vllm_mlx.spec_decode.mtp.generator")

    def mtp_generate_step(**_kwargs):
        yield from yields

    generator.mtp_generate_step = mtp_generate_step
    modules = {
        "vllm_mlx.spec_decode": types.ModuleType("vllm_mlx.spec_decode"),
        "vllm_mlx.spec_decode.mtp": types.ModuleType("vllm_mlx.spec_decode.mtp"),
        controller.__name__: controller,
        generator.__name__: generator,
    }
    modules["vllm_mlx.spec_decode"].__path__ = []
    modules["vllm_mlx.spec_decode.mtp"].__path__ = []

    gb = types.SimpleNamespace(
        uids=[7],
        _next_tokens=_Array([500]),
        _next_logprobs=["lp500"],
        tokens=[[]],
        max_tokens=[32],
        prompt_cache=[],
        logits_processors=None,
    )

    def baseline_step():
        current = list(gb._next_tokens.values)
        logprobs = list(gb._next_logprobs)
        for row, token in enumerate(current):
            gb.tokens[row].append(token)
        gb._next_tokens = _Array([token + 100 for token in current])
        gb._next_logprobs = [f"lp{token + 100}" for token in current]
        return current, logprobs

    gb._step = baseline_step
    batch_gen = types.SimpleNamespace(
        _generation_batch=gb,
        stop_tokens=set(),
    )
    model = types.SimpleNamespace(
        mtp_forward=object(),
        make_mtp_cache=object(),
        mtp=object(),
        mtp_max_speculative_tokens=1,
    )
    request = types.SimpleNamespace(
        sampling_params=types.SimpleNamespace(temperature=0.0)
    )
    install = _extract_function("_install_mtp_vendored")
    with patch.dict(sys.modules, modules):
        assert install(
            batch_gen,
            model,
            requests={"req": request},
            uid_to_request_id={7: "req"},
            max_k=1,
        )
    return batch_gen, gb


def test_b1_to_b2_emits_prepared_token_once_without_stale_placeholder():
    batch_gen, gb = _install_with_yields([(600, "lp600", False)])

    assert gb._step()[0] == [500]
    assert gb.tokens[0] == [500]

    assert batch_gen._mtp_vendored_prepare_batch_expansion() is True
    assert gb._next_tokens.values == [600]
    assert gb.tokens[0] == [500]  # prepared, not prematurely published

    # Model BatchGenerator.extend concatenating the newly admitted row.
    gb.uids = [7, 8]
    gb.tokens.append([])
    gb._next_tokens = _Array([600, 800])
    gb._next_logprobs = ["lp600", "lp800"]

    assert gb._step()[0] == [600, 800]
    assert gb.tokens == [[500, 600], [800]]
    assert batch_gen._mtp_vendored_stats["ft_mid_stream_handoff"] == 1


def test_accepted_draft_drains_before_exact_batch_expansion():
    batch_gen, gb = _install_with_yields([(601, "lp601", True), (602, "lp602", False)])

    assert gb._step()[0] == [500]
    assert batch_gen._mtp_vendored_prepare_batch_expansion() is False
    assert gb._step()[0] == [601]
    assert gb.tokens[0] == [500, 601]

    assert batch_gen._mtp_vendored_prepare_batch_expansion() is True
    gb.uids = [7, 8]
    gb.tokens.append([])
    gb._next_tokens = _Array([602, 800])
    gb._next_logprobs = ["lp602", "lp800"]
    assert gb._step()[0] == [602, 800]
    assert gb.tokens[0] == [500, 601, 602]


def test_unprepared_batch_growth_fails_closed_before_baseline_step():
    _batch_gen, gb = _install_with_yields([(600, "lp600", False)])
    assert gb._step()[0] == [500]
    gb.uids = [7, 8]
    gb.tokens.append([])
    gb._next_tokens = _Array([500, 800])
    gb._next_logprobs = ["lp500", "lp800"]

    with pytest.raises(RuntimeError, match="before exact MTP handoff"):
        gb._step()
    assert gb.tokens == [[500], []]


def test_scheduler_defers_waiting_request_until_mtp_hook_is_ready():
    schedule_waiting = _extract_function("_schedule_waiting", class_name="Scheduler")
    request = types.SimpleNamespace(sampling_params=object())
    hook_calls = []
    batch_gen = types.SimpleNamespace(
        _mtp_vendored_prepare_batch_expansion=lambda: hook_calls.append(True) or False
    )
    scheduler = types.SimpleNamespace(
        waiting=deque([request]),
        running={"active": object()},
        config=types.SimpleNamespace(max_num_seqs=2),
        batch_generator=batch_gen,
        _ensure_batch_generator=lambda _params: True,
    )

    assert schedule_waiting(scheduler) == []
    assert list(scheduler.waiting) == [request]
    assert hook_calls == [True]
