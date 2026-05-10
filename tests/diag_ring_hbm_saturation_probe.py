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

"""§11.4 — HBM saturation curve.

Question: how does effective per-core HBM bandwidth scale as more
cores load concurrently? Naive model: 166 GB/s / N. Real model may
have ring-arbiter / queue effects that make it sub-linear.

Probe design: pure-M split forces every core to read the full B from
HBM (worst-case HMI traffic). Vary SENCORES from 1 to 32. Each core
loads the full (K × N) of B. Per-core HBM traffic stays constant;
total HBM traffic = SENCORES × K × N. Wall growth ÷ SENCORES gives
effective per-core BW.

Shape: M=32, K=8192, N=2048 chosen so:
  per-core B = 8192 × 2048 × 2 = 32 MiB
  per-core compute is light → HMI-bound
  per-core A = 32 × 8192 × 2 = 512 KiB (small, doesn't saturate)
  per-core C = 32 × 2048 × 2 = 128 KiB (small)

This isolates HBM bus contention from PSUM ring traffic.

NOTE: requires SENCORES env to actually limit the number of cores —
this is set via ts_config.sencores, which the planner uses.
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

WARMUP = 3
ITERS = 8
DTYPE = torch.float16

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


def measure(M, N_total, K, n_cores):
    """Measure pure-N split with n_cores cores. Each core handles N/n_cores
    of the N dimension. A is fully shared across all cores."""
    ts_config.sencores = n_cores
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N_total, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    split = (1, n_cores, 1)

    try:
        with _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split(split):
                mm(a, b)

        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


def main():
    print("# Probe §11.4 — HBM saturation curve")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16")
    print()
    print("Shape: pure-N split. M=128, K=8192, N_per=128 fixed per core.")
    print("Vary number of cores. Per-core load constant; total HBM traffic")
    print("scales with n_cores. If HBM is bandwidth-limited at peak, wall")
    print("should grow super-linearly with n_cores once the bus saturates.")
    print()

    M = 128
    K = 8192
    N_per = 128  # fixed per core
    Ns_cores = [1, 2, 4, 8, 16, 32]

    print(
        "| n_cores | N_total | per-core B (KB) | wall ms | "
        "Δwall vs n=1 (ms) | per-core BW (GB/s) |"
    )
    print("|---:|---:|---:|---:|---:|---:|")

    rows = []
    baseline_wall = None
    per_core_b_bytes = K * N_per * 2  # fp16
    per_core_a_bytes = M * K * 2
    per_core_c_bytes = M * N_per * 2
    per_core_total = per_core_b_bytes + per_core_a_bytes + per_core_c_bytes

    for n in Ns_cores:
        N_total = n * N_per
        ms, err = measure(M, N_total, K, n)
        if ms is None:
            print(f"| {n} | {N_total} | {per_core_b_bytes / 1024:.0f} | ERR ({err}) |")
            sys.stdout.flush()
            continue

        if baseline_wall is None:
            baseline_wall = ms
        delta = ms - baseline_wall
        # per-core BW = per_core_bytes / wall_seconds
        per_core_bw = per_core_total / (ms / 1000) / 1e9
        print(
            f"| {n} | {N_total} | {per_core_b_bytes / 1024:.0f} | "
            f"{ms:.3f} | {delta:+.3f} | {per_core_bw:.2f} |"
        )
        sys.stdout.flush()
        rows.append((n, ms, per_core_bw))

    # Restore SENCORES
    ts_config.sencores = 32

    print("\n## HBM saturation analysis")
    print()
    if rows:
        # Compute aggregate BW (sum across cores)
        for n, ms, per_core_bw in rows:
            agg_bw = per_core_bw * n
            print(
                f"  n={n}: per-core {per_core_bw:.2f} GB/s, aggregate {agg_bw:.2f} GB/s"
            )

        print()
        print("  Theoretical HBM peak: 166.4 GB/s (1.3 GHz × 128 B/cycle)")
        max_agg = max(per_core_bw * n for n, _, per_core_bw in rows)
        print(
            f"  Max aggregate BW measured: {max_agg:.2f} GB/s "
            f"({max_agg / 166.4 * 100:.0f}% of peak)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
