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

"""Core-emission reorder probe in fully LX-resident regime.

The earlier reorder sweep (`diag_core_emission_sweep.py`) showed flat
results on production-sized shapes. We later isolated two reasons:

  1. HMI is on the same ring as cross-core sharing, so for shapes
     where weights stream from DRAM (every production matmul we
     measured), HMI dominates and reorder can't move total bytes.
  2. The kernel templates already do chunk-based overlapped input
     fetch, hiding ring sharing under compute when sharing IS active.

This probe forces the regime where reorder COULD plausibly matter:
all per-core operands fit in the 2 MB LX scratchpad. Pure ring-share
is the dominant data-movement cost (no HMI streaming on each call).
For the most extreme split (16, 2, 1), the algebraic prediction at
88 GB/s pure ring is:

  M-fast (default): 16 cores share a 1 MB B-band → ~340 us broadcast
  N-fast (reverse): 2 cores share a 32 KB A-row → ~0.4 us broadcast

A 300+ us delta should be measurable above the per-launch noise floor
if reorder actually does what the topology argument predicts AND the
template overlap doesn't fully hide it. If we still see flat results,
the lever is dead at any operand size and we can close out the
project.

Run: python tests/diag_core_emission_lx_fit.py
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre
torch_spyre._autoload()
from torch_spyre import streams as _ts
from torch_spyre._inductor import config as ts_config
from torch_spyre._inductor import core_division as _core_div
from torch_spyre._inductor.codegen import superdsc as _superdsc


WARMUP = 5
ITERS = 20
DTYPE = torch.float16
DTYPE_BYTES = 2

# Shape designed so all four mixed (m, n, 1) splits below have per-core
# operand totals <= 2 MB (the LX scratchpad limit).
M = 128
K = 2048
N = 1024

SPLITS = [(2, 16, 1), (4, 8, 1), (8, 4, 1), (16, 2, 1)]


# ---- planner-pick capture ----------------------------------------------

_captured: list = []
_orig_parse = _superdsc.parse_op_spec


def _hook(op_spec):
    sdsc = _orig_parse(op_spec)
    if _superdsc._is_matmul(op_spec.op):
        _captured.append(op_spec)
    return sdsc


_superdsc.parse_op_spec = _hook  # type: ignore[assignment]


def _split_str(op_spec) -> str:
    parts = []
    for sym, (sz, nc) in op_spec.iteration_space.items():
        try:
            parts.append(f"{int(sz)}x{int(nc)}c")
        except (TypeError, ValueError):
            parts.append(f"?x{nc}c")
    return "[" + ", ".join(parts) + "]"


# ---- force-split machinery ---------------------------------------------

_orig_multi = _core_div.multi_dim_iteration_space_split


def _force_split_factory(target):
    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        prod = target[0] * target[1] * target[2]
        if prod != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms)}
    return _forced


@contextmanager
def _force_split(target):
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


# ---- per-core operand size predictions --------------------------------

@dataclass
class _SplitPrediction:
    target: tuple[int, int, int]
    a_slice_bytes: int
    b_slice_bytes: int
    c_slice_bytes: int
    # For the M-fast (default) emitter: chain of m_split cores share an
    # n-band of B (because cores varying along M axis have same n-coord).
    # For the N-fast (reverse) emitter: chain of n_split cores share an
    # m-band of A (because cores varying along N axis have same m-coord).
    m_fast_chain_len: int
    m_fast_shared_bytes: int
    n_fast_chain_len: int
    n_fast_shared_bytes: int

    @property
    def per_core_total_bytes(self) -> int:
        return self.a_slice_bytes + self.b_slice_bytes + self.c_slice_bytes


def _predict(target):
    m, n, k = target
    A_full = M * K * DTYPE_BYTES
    B_full = K * N * DTYPE_BYTES
    return _SplitPrediction(
        target=target,
        a_slice_bytes=(M // m) * K * DTYPE_BYTES,
        b_slice_bytes=K * (N // n) * DTYPE_BYTES,
        c_slice_bytes=(M // m) * (N // n) * DTYPE_BYTES,
        m_fast_chain_len=m,
        m_fast_shared_bytes=K * (N // n) * DTYPE_BYTES,
        n_fast_chain_len=n,
        n_fast_shared_bytes=(M // m) * K * DTYPE_BYTES,
    )


# ---- bench primitive --------------------------------------------------

def _bench(fn) -> float:
    for _ in range(WARMUP):
        fn()
    _ts.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        _ts.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def _compile_and_bench(target):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()
    cap_start = len(_captured)

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(target):
            mm(a, b)
        _ts.synchronize()
        cap = _captured[cap_start]

        def step():
            with _force_split(target):
                mm(a, b)
        ms = _bench(step)
        return ms, cap, ""
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {str(e)[:60]}"


# ---- main -----------------------------------------------------------

def main() -> int:
    print(f"# Reorder probe in LX-resident regime")
    print(f"# Shape: M={M}, N={N}, K={K}, fp16")
    print(f"# warmup={WARMUP} iters={ITERS}")
    print()

    print("## Per-split LX-fit check + topology predictions\n")
    print("| split | per-core total | M-fast chain × shared | "
          "N-fast chain × shared |")
    print("|---|---:|---:|---:|")
    for target in SPLITS:
        p = _predict(target)
        fits = "✓" if p.per_core_total_bytes <= 2 * 1024 * 1024 else "✗"
        print(f"| {target} | {p.per_core_total_bytes // 1024} KB {fits} | "
              f"{p.m_fast_chain_len} × "
              f"{p.m_fast_shared_bytes // 1024} KB | "
              f"{p.n_fast_chain_len} × "
              f"{p.n_fast_shared_bytes // 1024} KB |")
    print()

    # --- bench each (split, mode) ---
    print("## Bench results\n")
    rows = []
    for target in SPLITS:
        print(f"# {target}")
        ts_config.core_emission_reverse = False
        ms_def, _, err_def = _compile_and_bench(target)
        if err_def:
            print(f"  default:  ERR {err_def}")
            ms_def = None
        else:
            print(f"  default:  {ms_def:.3f} ms")

        ts_config.core_emission_reverse = True
        ms_rev, _, err_rev = _compile_and_bench(target)
        if err_rev:
            print(f"  reverse:  ERR {err_rev}")
            ms_rev = None
        else:
            print(f"  reverse:  {ms_rev:.3f} ms")

        if ms_def is not None and ms_rev is not None:
            delta_ms = ms_def - ms_rev
            speedup = ms_def / ms_rev
            print(f"  delta:    {delta_ms:+.3f} ms  "
                  f"(speedup {speedup:.3f}x)")
        rows.append((target, ms_def, ms_rev))
        print()

    # --- table ---
    print("\n## Side-by-side\n")
    print("| split | per-core total | default ms | reverse ms | delta | "
          "speedup |")
    print("|---|---:|---:|---:|---:|---:|")
    for (target, ms_def, ms_rev) in rows:
        p = _predict(target)
        if ms_def is None or ms_rev is None:
            print(f"| {target} | {p.per_core_total_bytes // 1024} KB | "
                  f"{ms_def or 'err'} | {ms_rev or 'err'} | — | — |")
            continue
        delta = ms_def - ms_rev
        speedup = ms_def / ms_rev
        flag = " ✓" if speedup >= 1.05 else ""
        print(f"| {target} | {p.per_core_total_bytes // 1024} KB | "
              f"{ms_def:.3f} | {ms_rev:.3f} | {delta:+.3f} ms | "
              f"{speedup:.3f}x{flag} |")
    print()

    # --- verdict ---
    print("## Verdict\n")
    valid = [(t, d, r) for (t, d, r) in rows if d is not None and r is not None]
    if not valid:
        print("  No valid measurements.")
        return 1
    deltas = [d - r for (_, d, r) in valid]
    speedups = [d / r for (_, d, r) in valid]
    max_speedup = max(speedups)
    max_delta = max(deltas)
    print(f"  Max delta across splits: {max_delta:+.3f} ms")
    print(f"  Max speedup:             {max_speedup:.3f}x")
    print()

    if max_speedup >= 1.05:
        print("  Reorder shows >=5% movement on at least one LX-fit "
              "split. Topology lever IS exploitable in this regime — "
              "worth pursuing as a planner heuristic for shapes that "
              "fit fully in scratchpad.")
    elif max_speedup >= 1.02:
        print("  Reorder shows 2-5% movement. Marginal. Likely not "
              "worth a heuristic by itself, but useful as a tiebreaker.")
    else:
        print("  Reorder is flat (<2%) even in the LX-fit regime where "
              "ring-share is the dominant data-movement cost. The "
              "kernel templates' overlapped input fetch is hiding ring "
              "topology effects regardless of operand size. The lever "
              "is dead — close out the core-ordering project.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
