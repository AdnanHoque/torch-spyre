# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LX scratchpad fit predicate for the AIU 1.0 work-division planner.

Each AIU corelet has a 2 MB LX scratchpad — no hardware cache, so any
operand byte the kernel touches must be explicitly resident before
the PT array reads it.

Under the AIU matmul kernel template, A is the *stationary* operand
(held resident across the N-tile loop) and B *streams* through the
data ring chunk-by-chunk. The LX residency constraint is therefore
governed by A_per_core, not by total A+B bytes:

    A_per_core = (M / m) * (K / k) * dtype_bytes

When A_per_core exceeds LX, the kernel must re-fetch A across the
N-tile loop, multiplying effective HMI traffic. Project B Phase 0
measured this as the dominant residual in the cost model:
DSv3 o_proj M=2048 under (1, 8, 4)+kf is 124 ms measured vs. 21 ms
predicted — a ~6× factor consistent with an estimated 14× A re-fetch
multiplier.

The breakdown helper also reports B_per_core for diagnostic use, but
the headline `lx_fits()` predicate gates on A only. A conservative
A+B form is exposed as `lx_fits_conservative()` for callers that want
to also rule out splits where B-side residency could matter (e.g.,
templates where B is stationary instead).
"""

from __future__ import annotations

from dataclasses import dataclass


LX_BYTES_PER_CORE = 2 * 1024 * 1024     # 2 MB per corelet


_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8": 1, "int8": 1}


@dataclass(frozen=True)
class LXBreakdown:
    """Per-core LX residency breakdown for one (shape, split, dtype).

    `a_bytes` is the stationary-operand footprint that drives the
    headline `fits` field. `b_bytes` is reported for diagnostic use
    (templates where B is stationary, or callers that want a
    conservative A+B check).
    """

    a_bytes: int                # M_per × K_per × dtype_bytes (stationary)
    b_bytes: int                # K_per × N_per × dtype_bytes (streaming)
    lx_bytes: int               # capacity (default 2 MB)
    fits: bool                  # a_bytes <= lx_bytes
    fits_conservative: bool     # a_bytes + b_bytes <= lx_bytes
    overage_bytes: int          # max(0, a_bytes - lx_bytes)
    overage_factor: float       # a_bytes / lx_bytes (≥1 means overflow)


def lx_fits(
    shape: tuple[int, int, int],
    split: tuple[int, int, int],
    dtype: str = "fp16",
    lx_bytes: int = LX_BYTES_PER_CORE,
) -> bool:
    """Return True iff stationary-operand (A) footprint fits in LX.

    This is the predicate that matches measured AIU behaviour: under
    the matmul kernel template, A is stationary and B streams, so LX
    overflow happens when A_per_core > LX.
    """
    return lx_breakdown(shape, split, dtype, lx_bytes).fits


def lx_fits_conservative(
    shape: tuple[int, int, int],
    split: tuple[int, int, int],
    dtype: str = "fp16",
    lx_bytes: int = LX_BYTES_PER_CORE,
) -> bool:
    """Return True iff A_per_core + B_per_core fits in LX.

    Use this if you want to rule out splits where B-side residency
    could matter (alternative kernel templates, or as a safety margin).
    """
    return lx_breakdown(shape, split, dtype, lx_bytes).fits_conservative


def lx_breakdown(
    shape: tuple[int, int, int],
    split: tuple[int, int, int],
    dtype: str = "fp16",
    lx_bytes: int = LX_BYTES_PER_CORE,
) -> LXBreakdown:
    """Compute the per-core LX residency breakdown."""
    M, N, K = shape
    m, n, k = split
    if M % m or N % n or K % k:
        # Caller should have filtered by divisibility; if not, the
        # ceil-divide form is what the kernel template would observe.
        M_per = -(-M // m)
        N_per = -(-N // n)
        K_per = -(-K // k)
    else:
        M_per = M // m
        N_per = N // n
        K_per = K // k

    db = _DTYPE_BYTES[dtype]
    a = M_per * K_per * db
    b = K_per * N_per * db
    fits = a <= lx_bytes
    fits_cons = (a + b) <= lx_bytes
    over = max(0, a - lx_bytes)
    return LXBreakdown(
        a_bytes=a, b_bytes=b,
        lx_bytes=lx_bytes,
        fits=fits, fits_conservative=fits_cons,
        overage_bytes=over,
        overage_factor=a / lx_bytes,
    )
