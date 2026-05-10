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

"""Granite per-hop cost probe — disentangle BW-limited vs sync vs contention.

Goal: build a ring-cost model that can predict the kf speedup on
arbitrary Granite shapes, not just the one shape Probe 2 measured.

The model under test:
    wall(distance, payload) = T_baseline(shape)
                            + distance × (T_sync_per_hop
                                          + payload / SFP_BW_eff)

If the model holds:
- slope per shape = T_sync_per_hop + payload / SFP_BW_eff
- linear regression of (slope vs payload) across shapes gives:
    intercept = T_sync_per_hop  (per-hop fixed overhead)
    slope     = 1 / SFP_BW_eff  (effective ring BW including overhead)

Granite 8B linear layers at M=128 span ~14× payload variation:

  shape                 M_per × N_per × 4 = cohort_payload (K-cohort=2)
  kv_proj  (128, 2048)  128 × 128 × 4 =  64 KB
  q_proj   (128, 4096)  128 × 256 × 4 = 128 KB
  o_proj   (128, 4096)  128 × 256 × 4 = 128 KB
  down_proj(128, 4096)  128 × 256 × 4 = 128 KB   (note same N as q_proj)
  gate_proj(128,12800)  128 × 800 × 4 = 400 KB

Plus we add Granite 8B at M=2048 to extend the range to ~4 MB cohort
payload:

  q_proj   (2048,4096)  2048 × 256 × 4 = 2 MB
  gate_proj(2048,12800) 2048 × 800 × 4 = 6.4 MB

For each shape, measure walls under 4 permutations (identity = dist 16,
stride2 = dist 8, bit_reverse = dist 1, k_fast = dist 1). Linear-fit
the slope. Then linear-fit slope-vs-payload.

Subprocess timeout 120s per measurement (Granite 8B M=2048 gate_proj
gets to 60+ ms in past measurements).

Run: python /tmp/granite_perhop_probe.py
"""

from __future__ import annotations

import math
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
    # M=128 layers — primary kf-relevant regime
    Shape("Granite 8B kv_proj M=128",       128,  2048,  4096),
    Shape("Granite 8B q_proj M=128",        128,  4096,  4096),
    Shape("Granite 8B o_proj M=128",        128,  4096,  4096),
    Shape("Granite 8B down_proj M=128",     128,  4096, 12800),
    Shape("Granite 8B gate_proj M=128",     128, 12800,  4096),
    # M=2048 prefill shapes — extend payload range
    Shape("Granite 8B q_proj M=2048",      2048,  4096,  4096),
    Shape("Granite 8B gate_proj M=2048",   2048, 12800,  4096),
]


# Permutations at (1, 16, 2): k=2, m·n = 16.
# Identity: K-collabs at 0 and 16 → distance 16
# stride2: stride-2 interleave → distance 8
# bit_reverse: 5-bit reverse → distance 1 (paired with adjacent)
# k_fast: production permutation → distance 1


def perm_identity():
    return list(range(32))


def perm_stride2():
    # Interleave: even physical IDs first (cores 0,2,...,30), then odd
    out = []
    for c in range(32):
        out.append(c * 2 % 32 + (c * 2 // 32))
    return out


def perm_bit_reverse():
    def rev(c):
        out = 0
        for i in range(5):
            if c & (1 << i):
                out |= 1 << (4 - i)
        return out
    return [rev(c) for c in range(32)]


def perm_kfast(m, n, k):
    mn = m * n
    return [(c % k) * mn + (c // k) for c in range(32)]


def k_collab_distance(perm, m, n, k):
    """Average ring distance between K-collaborators."""
    mn = m * n
    inv = [0] * len(perm)
    for phys, logical in enumerate(perm):
        inv[logical] = phys
    distances = []
    for chain_base in range(mn):
        for ki in range(k - 1):
            l1 = chain_base + ki * mn
            l2 = chain_base + (ki + 1) * mn
            p1 = inv[l1]
            p2 = inv[l2]
            d = abs(p1 - p2)
            d = min(d, len(perm) - d)
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


def cohort_payload_bytes(shape, m, n, k):
    """PSUM bytes per K-cohort chain. Each chain reduces (M_per × N_per) fp32."""
    M_per = shape.M // m
    N_per = shape.N // n
    return M_per * N_per * 4  # fp32 PSUM


def fit_linear(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return ys[0] if ys else 0.0, 0.0, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    rmse = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                          for x, y in zip(xs, ys)) / n)
    return intercept, slope, rmse


def main():
    print("# Granite per-hop cost probe — disentangle BW vs sync vs contention")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32, "
          f"split=(1,16,2)")
    print()

    split = (1, 16, 2)
    m, n, k = split

    perms_all = [
        ("identity",    perm_identity(),    16),
        ("stride2",     perm_stride2(),      8),
        ("bit_reverse", perm_bit_reverse(),  1),
        ("k_fast",      perm_kfast(m, n, k), 1),
    ]

    print("| shape | (M,N,K) | cohort payload | perm | dist | wall ms |")
    print("|---|---|---:|---|---:|---:|")

    rows = []
    for s in SHAPES:
        payload = cohort_payload_bytes(s, m, n, k)
        for label, perm, expected_dist in perms_all:
            dist = k_collab_distance(perm, m, n, k)
            assert abs(dist - expected_dist) < 0.5, (
                f"{label} expected dist {expected_dist}, got {dist}")
            ms, err = measure(s.M, s.N, s.K, split, perm)
            wall_str = f"{ms:.2f}" if ms is not None else f"ERR ({err})"
            payload_str = f"{payload/1024:.0f} KB" if payload < 1024*1024 \
                          else f"{payload/1024/1024:.2f} MB"
            print(f"| {s.label} | ({s.M},{s.N},{s.K}) | {payload_str} | "
                  f"{label} | {dist:.0f} | {wall_str} |")
            sys.stdout.flush()
            rows.append((s, label, dist, ms, payload))

    # Per-shape regression: wall = base + slope · distance
    print("\n## Per-shape slope")
    print()
    print("| shape | payload | base ms | slope ms/hop | RMSE | "
          "kf vs id wall (saving × ms) |")
    print("|---|---:|---:|---:|---:|---|")

    by_shape = {}
    for (s, label, dist, ms, payload) in rows:
        if ms is None:
            continue
        by_shape.setdefault(s.label, []).append(
            (dist, ms, label, payload))

    shape_data_for_payload_fit = []
    for shape_label, data in by_shape.items():
        if len(data) < 2:
            continue
        xs = [d for d, _, _, _ in data]
        ys = [w for _, w, _, _ in data]
        intercept, slope, rmse = fit_linear(xs, ys)
        payload = data[0][3]
        # Find kf and identity walls
        kf_wall = None
        id_wall = None
        for d, w, lbl, _ in data:
            if lbl == "k_fast":
                kf_wall = w
            if lbl == "identity":
                id_wall = w
        kf_id_str = "n/a"
        if kf_wall is not None and id_wall is not None:
            saving = id_wall - kf_wall
            speedup = id_wall / kf_wall if kf_wall > 0 else 0
            kf_id_str = f"{speedup:.2f}× ({saving:+.2f} ms)"
        payload_str = (f"{payload/1024:.0f} KB" if payload < 1024*1024
                       else f"{payload/1024/1024:.2f} MB")
        print(f"| {shape_label} | {payload_str} | {intercept:.2f} | "
              f"{slope:.4f} | {rmse:.3f} | {kf_id_str} |")
        shape_data_for_payload_fit.append((payload, slope))

    # Slope-vs-payload regression: slope = T_sync + payload / SFP_BW
    print("\n## Slope vs payload regression")
    print()
    if len(shape_data_for_payload_fit) >= 2:
        # Linear fit: slope = a + b · payload
        # Where a = T_sync_per_hop (ms), b = 1 / SFP_BW (ms / byte)
        xs = [p for p, _ in shape_data_for_payload_fit]
        ys = [s for _, s in shape_data_for_payload_fit]
        T_sync, inv_BW, rmse = fit_linear(xs, ys)
        bw_GBps = (1.0 / inv_BW) / 1e6 if inv_BW > 0 else float("inf")
        print(f"  slope ≈ {T_sync*1000:.2f} µs + payload × "
              f"{inv_BW*1e6:.4f} ms/MB")
        print(f"  T_sync_per_hop = {T_sync*1000:.2f} µs/hop")
        print(f"  effective SFP-ring BW = {bw_GBps:.2f} GB/s "
              f"(vs spec 35.2 GB/s)")
        print(f"  RMSE = {rmse*1000:.2f} µs")
        print()
        print("  Theoretical SFP ring BW: 32 B/cyc × 1.1 GHz = 35.2 GB/s")
        if bw_GBps > 0:
            print(f"  Measured effective BW = {bw_GBps:.2f} GB/s "
                  f"({bw_GBps/35.2*100:.0f}% of peak)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
