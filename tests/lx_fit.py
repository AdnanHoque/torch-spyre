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
byte the kernel touches must be explicitly resident before use.

The binding LX residency constraint identified by Probe 3 in
`tests/diag_emission_aware_lx_p3_midk_n_sweep.py` is the **PSUM
accumulator**, not operand A. The accumulator must remain LX-resident
across the K-iteration loop because partial products accumulate into
it; A and B can be tile-streamed, but C cannot:

    C_psum_per_core = (M / m) * (N / n) * dtype_psum_bytes  (default fp32)

Probe 3 found a clean inflection at exactly C_psum = LX. Above the
threshold, walls grow ~17 ms per LX overage factor on DSv3 o_proj at
(1, 8, 4)+kf — the catastrophe regime is unmistakable.

Earlier versions of this file gated on operand-A residency
(`A_per = M_per × K_per × dtype_bytes`) following Project B Phase 0's
hypothesis. The hardware probe data refutes that mechanism — A
residency is not the binding constraint. See
`tests/diag_emission_aware_lx_phase0_findings_v2.md` for the full
narrative.

Operand A and B per-core bytes are still reported on the breakdown
for diagnostic use, but the headline `lx_fits()` predicate now gates
on PSUM accumulator size.
"""

from __future__ import annotations

from dataclasses import dataclass


LX_BYTES_PER_CORE = 2 * 1024 * 1024     # 2 MB per corelet


_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8": 1, "int8": 1}


@dataclass(frozen=True)
class LXBreakdown:
    """Per-core LX residency breakdown for one (shape, split, dtype).

    `c_psum_bytes` is the binding constraint: PSUM accumulator must
    stay resident across the K-loop. `a_bytes` and `b_bytes` are
    reported for context but don't gate.
    """

    a_bytes: int                # M_per × K_per × dtype_bytes (operand A)
    b_bytes: int                # K_per × N_per × dtype_bytes (operand B)
    c_psum_bytes: int           # M_per × N_per × dtype_psum_bytes (accumulator)
    lx_bytes: int               # capacity (default 2 MB)
    fits: bool                  # c_psum_bytes <= lx_bytes
    overage_bytes: int          # max(0, c_psum_bytes - lx_bytes)
    overage_factor: float       # c_psum_bytes / lx_bytes (≥1 means overflow)


def lx_fits(
    shape: tuple[int, int, int],
    split: tuple[int, int, int],
    dtype: str = "fp16",
    psum_dtype: str = "fp32",
    lx_bytes: int = LX_BYTES_PER_CORE,
) -> bool:
    """Return True iff per-core PSUM accumulator fits in LX.

    Probe 3 (May 2026) measured a clean inflection at
    M_per × N_per × dtype_psum = LX_BYTES_PER_CORE; above this point
    wall time grows ~17 ms per overage factor, with no detectable
    overhead below. This is the predicate the work-division planner
    needs to stay out of the catastrophic regime.

    The `dtype` argument is retained on the signature for backwards
    compatibility with callers that pass it; only `psum_dtype` and
    the (m, n) factors of `split` actually affect the predicate's
    answer (the K factor is irrelevant since C_psum doesn't depend
    on K).
    """
    return lx_breakdown(
        shape, split, dtype=dtype, psum_dtype=psum_dtype,
        lx_bytes=lx_bytes,
    ).fits


def lx_breakdown(
    shape: tuple[int, int, int],
    split: tuple[int, int, int],
    dtype: str = "fp16",
    psum_dtype: str = "fp32",
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
    db_psum = _DTYPE_BYTES[psum_dtype]

    a = M_per * K_per * db
    b = K_per * N_per * db
    c = M_per * N_per * db_psum

    fits = c <= lx_bytes
    over = max(0, c - lx_bytes)
    return LXBreakdown(
        a_bytes=a, b_bytes=b, c_psum_bytes=c,
        lx_bytes=lx_bytes,
        fits=fits,
        overage_bytes=over,
        overage_factor=c / lx_bytes,
    )
