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

"""Probe 2 verification — ring hop cost re-measurement on clean rebuild.

Replicates the May 2026 Probe 2 measurement (DSv3 o_proj at (1, 16, 2),
varying core-id permutation) on the current clean torch-spyre +
deeptools environment.

Goal: confirm or refute the "5.6 ms/hop" linear-in-K-collab-distance
finding from `diag_emission_aware_lx_phase0_findings_v2.md`.

For each shape × permutation:
- Force the (1, 16, 2) split via _force_split (existing infra)
- Override the core-id permutation by patching
  _k_fast_core_id_permutation to return the desired permutation
- Measure wall time

Permutations tested (all yield 32-element permutations of [0..31]):
  identity:     [0, 1, 2, ..., 31]                — K-collab dist 16
  stride2:      [0, 2, ..., 30, 1, 3, ..., 31]    — K-collab dist 8
  k_fast:       [(c%2)*16 + c//2 for c in 0..31]  — K-collab dist 1
  bit_reverse:  reverse-bit ordering              — K-collab dist 1 (same end as kf)
  reversed:     [31, 30, ..., 0]                  — K-collab dist 16

Shape suite (3 shapes spanning payload size):
  small:  L3-70B q_proj M=128  (1, 16, 2) — small payload
  medium: L3-70B q_proj M=2048 (1, 16, 2) — medium payload
  large:  DSv3 o_proj M=2048   (1, 16, 2) — original Probe 2 shape

If the 5.6 ms/hop finding holds:
  - all three shapes: linear in K_collab_distance
  - slope scales with payload size
  - distance=1 permutations within ~0.2 ms of each other
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

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

WARMUP = 3
ITERS = 8
DTYPE = torch.float16


@dataclass(frozen=True)
class Shape:
    label: str
    M: int
    N: int
    K: int


SHAPES = [
    Shape("L3-70B q_proj M=128",   128,  8192, 8192),
    Shape("L3-70B q_proj M=2048", 2048,  8192, 8192),
    Shape("DSv3 o_proj M=2048",   2048,  7168, 16384),  # original Probe 2 shape
]


def perm_identity(num_cores=32):
    return list(range(num_cores))


def perm_kfast(num_cores, m, n, k):
    mn = m * n
    return [(c % k) * mn + (c // k) for c in range(num_cores)]


def perm_stride2(num_cores=32):
    # stride-2 interleave: even physical IDs first, then odd
    return [c * 2 % num_cores + (c * 2 // num_cores) for c in range(num_cores)]


def perm_bit_reverse(num_cores=32):
    # 5-bit reverse for 32 cores
    def rev(c):
        out = 0
        for i in range(5):
            if c & (1 << i):
                out |= 1 << (4 - i)
        return out
    return [rev(c) for c in range(num_cores)]


def perm_reversed(num_cores=32):
    return list(range(num_cores - 1, -1, -1))


def k_collab_distance(perm, m, n, k):
    """For each chain (group of k_idx values sharing m·n base), compute
    the ring distance between consecutive K-collaborators. Return mean
    over all chains."""
    mn = m * n
    distances = []
    # In the unpermuted (logical) emission, chain c has members at
    # logical IDs c, c + mn, c + 2·mn, ..., c + (k-1)·mn.
    # Under permutation `perm`, physical core c executes the slice that
    # logical core perm[c] would have. So K-collab partners at logical
    # mn-spacing land at physical positions: find (physical_for_logical_c,
    # physical_for_logical_(c+mn)) and measure their ring distance.
    inv = [0] * len(perm)
    for phys, logical in enumerate(perm):
        inv[logical] = phys
    num_cores = len(perm)
    for chain_base in range(mn):
        for ki in range(k - 1):
            l1 = chain_base + ki * mn
            l2 = chain_base + (ki + 1) * mn
            p1 = inv[l1]
            p2 = inv[l2]
            # Ring distance on bidirectional ring
            d = abs(p1 - p2)
            d = min(d, num_cores - d)
            distances.append(d)
    return sum(distances) / len(distances) if distances else 0


_orig_multi = _planner.multi_dim_iteration_space_split


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


_orig_kfast_perm = _co._k_fast_core_id_permutation


@contextmanager
def _force_perm(perm):
    """Replace _k_fast_core_id_permutation with a function returning `perm`."""
    def _patched(num_cores, work_slices):
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


def measure(M, N, K, split, perm):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_perm(perm), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_perm(perm), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


def main():
    print("# Probe 2 verification — ring hop cost on clean rebuild")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32, split=(1,16,2)")
    print()

    split = (1, 16, 2)
    m, n, k = split

    perms = [
        ("identity",   perm_identity()),
        ("reversed",   perm_reversed()),
        ("stride2",    perm_stride2()),
        ("bit_reverse", perm_bit_reverse()),
        ("k_fast",     perm_kfast(32, m, n, k)),
    ]

    print("| shape | (M, N, K) | permutation | K-collab dist (avg) | wall ms |")
    print("|---|---|---|---:|---:|")

    rows = []
    for s in SHAPES:
        for label, perm in perms:
            dist = k_collab_distance(perm, m, n, k)
            ms, err = measure(s.M, s.N, s.K, split, perm)
            wall_str = f"{ms:.2f}" if ms is not None else f"ERR ({err})"
            print(f"| {s.label} | ({s.M},{s.N},{s.K}) | {label} | {dist:.1f} | {wall_str} |")
            sys.stdout.flush()
            rows.append((s, label, dist, ms))

    # Per-shape regression
    print()
    print("## Per-shape: wall ≈ base + slope · K_collab_distance")
    print()
    by_shape = {}
    for (s, label, dist, ms) in rows:
        if ms is None:
            continue
        by_shape.setdefault(s.label, []).append((dist, ms, label))

    print("| shape | base ms | slope ms/hop | distance=1 spread |")
    print("|---|---:|---:|---:|")
    for shape_label, data in by_shape.items():
        if len(data) < 2:
            continue
        # Linear regression
        n_pts = len(data)
        sum_x = sum(d for d, _, _ in data)
        sum_y = sum(y for _, y, _ in data)
        sum_xx = sum(d * d for d, _, _ in data)
        sum_xy = sum(d * y for d, y, _ in data)
        denom = n_pts * sum_xx - sum_x * sum_x
        if denom == 0:
            continue
        slope = (n_pts * sum_xy - sum_x * sum_y) / denom
        base = (sum_y - slope * sum_x) / n_pts

        # distance=1 spread
        d1_walls = [y for d, y, _ in data if abs(d - 1.0) < 0.01]
        if len(d1_walls) >= 2:
            spread = max(d1_walls) - min(d1_walls)
            spread_str = f"{spread:.2f} ms"
        else:
            spread_str = "n/a"

        print(f"| {shape_label} | {base:.2f} | {slope:.2f} | {spread_str} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
