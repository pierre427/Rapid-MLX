# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the sorted gather_qmm/gather_mm row-alignment guard."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.patches.switch_layers_gather_pad import install


def _module_with_gather_sort():
    calls = []

    def _original_gather_sort(x, indices):
        calls.append(True)
        *_, M = indices.shape
        indices = indices.flatten()
        order = mx.argsort(indices)
        inv_order = mx.argsort(order)
        return x.flatten(0, -3)[order // M], indices[order], inv_order

    return SimpleNamespace(mx=mx, _gather_sort=_original_gather_sort, calls=calls)


def test_sorted_gather_pad_aligns_large_ragged_row_count():
    module = _module_with_gather_sort()

    assert install(module) is True

    tokens, top_k = 8193, 4
    x = mx.arange(tokens, dtype=mx.float32).reshape(1, tokens, 1)
    x = mx.expand_dims(x, (-2, -3))
    indices = (mx.arange(tokens * top_k, dtype=mx.int32) % 8).reshape(1, tokens, top_k)

    sorted_x, sorted_indices, inv_order = module._gather_sort(x, indices)

    assert module.calls == [True]
    assert indices.size == 32772
    assert sorted_indices.size == 32832
    assert sorted_indices.size % 64 == 0
    assert inv_order.size == indices.size
    assert sorted_x.shape[0] == sorted_indices.size
    assert mx.all(sorted_x[-1] == sorted_x[-2]).item()
    assert sorted_indices[-1].item() == sorted_indices[-2].item()


def test_sorted_gather_pad_leaves_aligned_large_row_count_unchanged():
    module = _module_with_gather_sort()

    assert install(module) is True

    tokens, top_k = 8208, 4
    x = mx.arange(tokens, dtype=mx.float32).reshape(1, tokens, 1)
    x = mx.expand_dims(x, (-2, -3))
    indices = (mx.arange(tokens * top_k, dtype=mx.int32) % 8).reshape(1, tokens, top_k)

    _, sorted_indices, inv_order = module._gather_sort(x, indices)

    assert module.calls == [True]
    assert indices.size == 32832
    assert indices.size % 64 == 0
    assert sorted_indices.size == indices.size
    assert inv_order.size == indices.size


def test_sorted_gather_pad_install_is_idempotent():
    module = _module_with_gather_sort()

    assert install(module) is True
    patched = module._gather_sort

    assert install(module) is False
    assert module._gather_sort is patched
