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

"""Probe — does HBM bank parallelism explain the wall-time differences
between splits, or is something else the actual bottleneck?

Hypothesis from earlier analysis: wall time on memory-bound matmul
correlates with the number of DISTINCT concurrent HBM reads. Higher
distinct count → more banks engaged → higher achieved BW → lower wall.

Test design:
  Hold the matmul shape constant (M, N, K). Vary the split (m, n, k)
  to span B-distinct-fragment counts from 1 (pure-M, all 32 cores share
  one B) to 32 (full B-slicing under K-split or pure-N). For each split:
    - Apply k_fast permutation (collapses K-cohort hops to 1, so the
      PSUM-ring cost is controlled and small)
    - Measure wall time

  Per-split metric:
    B_distinct = n · k    (number of distinct B-fragments)
    A_distinct = m · k    (number of distinct A-fragments)

If bank parallelism is the dominant effect: wall ∝ 1 / B_distinct (since
B is the bigger operand for our test shape). The curve should look like
the HBM saturation curve, in reverse.

If something else dominates (ring fanout cost, controller queue depth,
HMI accounting effects), wall might NOT scale with distinct-read count
in a clean way.

Shape: Llama 3.1 8B q_proj (32, 4096, 4096):
  M=32, N=4096, K=4096, fp16
  A = M·K·2 = 256 KB (small)
  B = K·N·2 = 32 MB  (big — B-distinct count should dominate)
  C = M·N·2 = 256 KB (small)
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from contextlib import contextmanager

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402
import torch._inductor.config as _icfg  # noqa: E402

_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/home/adnan/dt-inductor/torch-spyre")

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402

try:
    from torch_spyre._inductor import work_division as _planner  # noqa: E402
except ImportError:
    from torch_spyre._inductor import core_division as _planner  # noqa: E402

from torch_spyre._inductor import config as ts_config  # noqa: E402
from torch_spyre._inductor.codegen import compute_ops as _co  # noqa: E402

WARMUP = 2
ITERS = 10
DTYPE = torch.float16

_orig_multi = _planner.multi_dim_iteration_space_split
_orig_kfast_perm = _co._k_fast_core_id_permutation


def _force_split_factory(target):
    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        if target[0] * target[1] * target[2] != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms)}
    return _forced


@contextmanager
def _force_split(target):
    if target is None:
        yield
        return
    _planner.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _planner.multi_dim_iteration_space_split = _orig_multi


def perm_kfast(m, n, k):
    """K-cohort adjacent on the ring."""
    mn = m * n
    return [(c % k) * mn + (c // k) for c in range(m * n * k)]


@contextmanager
def _force_perm(perm_func, split):
    m, n, k = split
    perm = perm_func(m, n, k)

    def _patched(num_cores, work_slices):
        if num_cores != m * n * k:
            return _orig_kfast_perm(num_cores, work_slices)
        return list(perm)

    _co._k_fast_core_id_permutation = _patched
    prev = ts_config.core_id_k_fast_emission
    ts_config.core_id_k_fast_emission = True
    try:
        yield
    finally:
        _co._k_fast_core_id_permutation = _orig_kfast_perm
        ts_config.core_id_k_fast_emission = prev


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


def measure(M, N, K, split):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_perm(perm_kfast, split), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_perm(perm_kfast, split), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


# Splits chosen to span B-distinct = 1, 2, 4, 8, 16, 32 cleanly,
# plus a few that vary A-distinct independently.
SPLITS = [
    # B-distinct sweep at fixed k=1
    ((32, 1, 1),  1,  32, "pure-M (B-mc 32-way, A unique)"),
    ((16, 2, 1),  2,  16, ""),
    ((8, 4, 1),   4,   8, ""),
    ((4, 8, 1),   8,   4, ""),
    ((2, 16, 1), 16,   2, ""),
    ((1, 32, 1), 32,   1, "pure-N (A-mc 32-way, B unique)"),
    # K-split variants — B-distinct = 32 by k slicing
    ((1, 16, 2), 32,   2, "k=2"),
    ((1, 8, 4),  32,   4, "k=4"),
    ((1, 4, 8),  32,   8, "k=8"),
    ((1, 2, 16), 32,  16, "k=16"),
    ((1, 1, 32), 32,  32, "pure-K (both A and B fully sliced)"),
]


def main():
    print("# Probe — bank parallelism vs distinct-in-flight-reads")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("Shape: Llama 3.1 8B q_proj (32, 4096, 4096)")
    print("  A = 256 KB (small)")
    print("  B = 32 MB  (big — B-distinct count should dominate)")
    print("  C = 256 KB (small)")
    print()
    print("All splits use k_fast permutation to minimize PSUM ring cost.")
    print("Expected if bank parallelism is real: wall ∝ 1/B_distinct")
    print()

    M, N, K = 32, 4096, 4096
    A_bytes = M * K * 2
    B_bytes = K * N * 2
    C_bytes = M * N * 2

    print("| split | B-dist | A-dist | wall ms | B-MB / wall | aggregate "
          "BW est | note |")
    print("|---|---:|---:|---:|---:|---:|---|")

    results = []
    for split, b_dist, a_dist, note in SPLITS:
        m, n, k = split
        wall_ms, err = measure(M, N, K, split)
        if wall_ms is None:
            print(f"| {split} | {b_dist} | {a_dist} | ERR ({err}) | — | — | — |")
            continue

        # Aggregate BW estimate: total bytes / wall
        # Under full multicast: A + B + k·C
        total_bytes = A_bytes + B_bytes + k * C_bytes
        agg_bw_gbs = (total_bytes / 1e9) / (wall_ms / 1e3)  # GB/s

        # B's specific "wall per MB" — this is the actual signal:
        # if B is HBM-bound and bank parallelism scales effective BW with
        # distinct fragments, this should drop sharply with B_dist.
        b_per_ms = (B_bytes / 1024 / 1024) / wall_ms

        print(f"| {split} | {b_dist} | {a_dist} | {wall_ms:.3f} | "
              f"{b_per_ms:.1f} MB/ms | {agg_bw_gbs:.0f} GB/s | {note} |")
        sys.stdout.flush()
        results.append((split, b_dist, a_dist, wall_ms, agg_bw_gbs))

    print()
    print("## Analysis")
    print()

    # Group by B_distinct
    from collections import defaultdict
    by_bdist = defaultdict(list)
    for split, b_dist, a_dist, wall_ms, agg_bw in results:
        by_bdist[b_dist].append((wall_ms, agg_bw))

    print("### Wall time vs B-distinct (median across k splits at same B-distinct)")
    print()
    print("| B-distinct | median wall ms | median agg BW |")
    print("|---:|---:|---:|")
    for b_dist in sorted(by_bdist.keys()):
        walls = [r[0] for r in by_bdist[b_dist]]
        bws = [r[1] for r in by_bdist[b_dist]]
        print(f"| {b_dist} | {statistics.median(walls):.3f} | "
              f"{statistics.median(bws):.0f} GB/s |")

    print()
    print("### Verdict")
    print()
    print("If wall is monotonically decreasing in B-distinct from 1 → 32:")
    print("  → bank parallelism model SUPPORTED. The dominant bottleneck is")
    print("    the number of distinct concurrent reads.")
    print()
    print("If wall flattens or reverses past some B-distinct value:")
    print("  → bus saturates around that distinct-read count. Below that,")
    print("    bank parallelism matters; above, other effects dominate.")
    print()
    print("If wall is unrelated to B-distinct:")
    print("  → bank parallelism is NOT the explanation. The actual mechanism")
    print("    is something else (controller queue, ring fanout, etc.).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
