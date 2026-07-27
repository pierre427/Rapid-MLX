# SPDX-License-Identifier: Apache-2.0
"""Compatibility patch for mlx-lm sorted MoE gathers.

MLX's sorted ``gather_qmm`` / ``gather_mm`` path can silently corrupt affine
MoE outputs when the flattened routed-row count is greater than 32768 and is
not 64-aligned. Padding the sorted rows by duplicating the final row keeps the
expert indices sorted, and mlx-lm's unsort step drops the padding before the
result is reshaped back to the original token/expert grid.
"""

from __future__ import annotations

from types import ModuleType


def install(module: ModuleType | None = None) -> bool:
    """Patch ``mlx_lm.models.switch_layers._gather_sort``.

    Returns ``True`` when this call installs the patch, ``False`` when the
    module is unavailable or was already patched.
    """

    if module is None:
        try:
            from mlx_lm.models import switch_layers as module
        except ImportError:
            return False

    if getattr(module, "_rapid_mlx_sorted_gather_pad_installed", False):
        return False

    mx = module.mx
    original = module._gather_sort

    def _gather_sort(x, indices):
        x, indices, inv_order = original(x, indices)
        n = indices.size
        if n > 32768 and n % 64 != 0:
            pad = 64 - n % 64
            x = mx.concatenate(
                [x, mx.broadcast_to(x[-1:], (pad,) + x.shape[1:])], axis=0
            )
            indices = mx.concatenate(
                [indices, mx.broadcast_to(indices[-1:], (pad,))], axis=0
            )

        return x, indices, inv_order

    module._rapid_mlx_original_gather_sort = original
    module._gather_sort = _gather_sort
    module._rapid_mlx_sorted_gather_pad_installed = True
    return True
