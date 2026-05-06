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

"""Inter-op core_id alignment — Phase 0.

Hypothesis: when two consecutive ops have mismatched splits, op 1's
output partitioning across cores doesn't match op 2's input
partitioning, causing a ring shuffle between them.

Test setup: chain matmul1 → matmul2 where matmul2's A input is
matmul1's C output.

  intermediate = A @ B    # op 1: (M, K1) @ (K1, N1) → (M, N1)
  output       = intermediate @ C   # op 2: (M, N1) @ (N1, N2) → (M, N2)

With op 1 fixed at split (8, 4, 1):
  - op 1's output is partitioned (8 m_slices × 4 n_slices)
  - cores 0..7 hold m_slices for n_slice=0, etc.

Op 2 reads (M, N1). Its 'k' dim = N1 (op 1's N dim). For the read to
not require shuffle:
  - op 2's m partitioning should match op 1's m partitioning (same m)
  - op 2's k partitioning should match op 1's n partitioning (same)

If op 1 = (8, 4, 1), the 'naturally matched' op 2 split is (8, ?, 4)
where ?=1 since 8·1·4=32. So matched: op 2 = (8, 1, 4).

If we force op 2 to (4, 8, 1) or other mismatched splits, the
hypothesis predicts higher chained wall than the matched case after
subtracting standalone op 2 cost.

Measurement:
  T_chain(op1, op2) = wall of (op1 → op2) compiled together
  T_solo1 = wall of op1 alone
  T_solo2(op2) = wall of op2 alone

  inter_op_extra(op2) = T_chain - T_solo1 - T_solo2(op2)

If inter_op_extra varies with op 2's split (controlling for the solo
walls), inter-op alignment is a lever. If it's constant, alignment
doesn't help.

Usage:
    python tests/diag_inter_op_alignment.py
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from pathlib import Path
import sys

import torch
from sympy import Symbol

import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402
from torch_spyre._inductor import core_division as _core_div  # noqa: E402


WARMUP = 2
ITERS = 5
DTYPE = torch.float16

_orig_multi = _core_div.multi_dim_iteration_space_split


def _force_split_factory_per_n(targets_by_n):
    """Dispatch the forced split by the value of the N iteration-space dim.

    targets_by_n: dict mapping N value (int) → split tuple (m, n, k).
    Falls back to original planner if N value not in map.
    """
    def _forced(it_space, max_cores, priorities, min_splits=None):
        if len(it_space) != 3:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        # Find the dim whose iteration extent matches one of our keys.
        dims = list(it_space.keys())
        # Iteration space values — look for a match
        space_vals = {str(d): int(it_space[d]) for d in dims}
        for n_val, target in targets_by_n.items():
            if n_val in space_vals.values() and target[0] * target[1] * target[2] == max_cores:
                # Pick this target if the iter space matches expected pattern.
                # Heuristic: assume the second dim ('N' for matmul iter space [M, N, K])
                # is the one we keyed on.
                if len(dims) >= 2 and int(it_space[dims[1]]) == n_val:
                    return {sym: target[i] for i, sym in enumerate(dims)}
        return _orig_multi(it_space, max_cores, priorities, min_splits)
    return _forced


@contextmanager
def _force_splits_per_n(targets_by_n):
    _core_div.multi_dim_iteration_space_split = _force_split_factory_per_n(targets_by_n)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


@contextmanager
def _force_split_single(target):
    """Force ALL ops to the same split."""
    def _forced(it_space, max_cores, priorities, min_splits=None):
        if len(it_space) != 3 or target[0] * target[1] * target[2] != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(it_space.keys())}
    _core_div.multi_dim_iteration_space_split = _forced
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


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


def _bench_solo(M, N, K, split):
    """Time a single matmul (M, N, K) under given split."""
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split_single(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split_single(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:50]}"


def _bench_chained(M, K1, N1, N2, split1, split2):
    """Time chained: (M,K1)@(K1,N1) → (.,N1)@(N1,N2) under given splits."""
    a = torch.randn(M, K1, dtype=DTYPE, device="spyre")
    b = torch.randn(K1, N1, dtype=DTYPE, device="spyre")
    c = torch.randn(N1, N2, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def chain(x, y, z):
        intermediate = x @ y
        return intermediate @ z

    targets_by_n = {N1: split1, N2: split2}
    try:
        with _force_splits_per_n(targets_by_n):
            chain(a, b, c)
        _ts.synchronize()

        def step():
            with _force_splits_per_n(targets_by_n):
                chain(a, b, c)
        return _bench(step), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:50]}"


def main() -> int:
    print("# Inter-op core_id alignment — Phase 0\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")

    # Test cases: (M, K1, N1, N2, label).
    # N1 = N2 set differently to ensure unique iteration spaces for dispatch.
    TEST_CASES = [
        # Llama 70B-style attention output → MLP (kv_proj + o_proj-ish)
        (128, 8192, 1024, 4096, "M=128, kv→o-style"),
        (128, 4096, 8192, 1024, "M=128, o→kv-style"),
        # Larger M
        (256, 4096, 8192, 1024, "M=256, o→kv-style"),
    ]

    # Op 1 = fixed at (8, 4, 1). Op 2 splits we'll vary.
    OP1_SPLIT = (8, 4, 1)
    OP2_SPLITS_TO_TEST = [
        ((8, 1, 4), "(8,1,4) [matched]"),    # matched: m=8 same, k=4 covers op1's n=4
        ((8, 4, 1), "(8,4,1) [partial]"),    # partial: same m, but k≠op1's n
        ((4, 8, 1), "(4,8,1) [mismatched]"), # full mismatch: different m, different k
        ((1, 32, 1), "(1,32,1) [pure-N]"),
        ((32, 1, 1), "(32,1,1) [pure-M]"),
    ]

    print("| case | op2 split | T_solo1 | T_solo2 | T_chain | T_chain - T_solo1 - T_solo2 |")
    print("|---|---|---:|---:|---:|---:|")

    for M, K1, N1, N2, label in TEST_CASES:
        # Solo op 1 wall
        t_solo1, err1 = _bench_solo(M, N1, K1, OP1_SPLIT)
        if err1:
            print(f"| {label} solo1 ERR | — | — | — | — | — |")
            continue

        for op2_split, op2_label in OP2_SPLITS_TO_TEST:
            t_solo2, err2 = _bench_solo(M, N2, N1, op2_split)
            t_chain, errc = _bench_chained(M, K1, N1, N2, OP1_SPLIT, op2_split)
            if err2 or errc:
                print(f"| {label} | {op2_label} | "
                      f"{t_solo1:.2f} | "
                      f"{(t_solo2 if not err2 else 0):.2f} | "
                      f"{(t_chain if not errc else 0):.2f} | "
                      f"ERR: {err2 or errc} |", flush=True)
                continue
            extra = t_chain - t_solo1 - t_solo2
            print(f"| {label} | {op2_label} | "
                  f"{t_solo1:.2f} | {t_solo2:.2f} | {t_chain:.2f} | {extra:+.2f} |",
                  flush=True)
        print(f"|  |  |  |  |  |  |")

    print()
    print("## Reading guide\n")
    print(
        "  - 'Extra' = chained wall minus solo walls. Captures inter-op cost.\n"
        "  - If extra is roughly constant across op2 splits: alignment doesn't matter.\n"
        "  - If extra is much LOWER for matched (8,1,4): alignment is a lever.\n"
        "  - 'extra' may be negative if chained pipelines (overlap of op1 and op2)\n"
        "    — that's fine, what matters is variance across splits.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
