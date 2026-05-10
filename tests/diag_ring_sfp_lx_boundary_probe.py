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

"""§11.6 — SFP per-MB inflation across the LX boundary.

Question: §6.2's per-MB ring slope on L3-70B M=2048 was 273 µs/MB,
on DSv3 o_proj M=2048 was 103 µs/MB. The 3× difference is suspected
to be the C_psum > LX overflow penalty kicking in differently. This
probe sweeps N_per across the LX boundary on a single shape family
to isolate.

Shape: M=2048, K=8192, varying N at split (1, 16, 2):
  N_per = N/16  → C_psum_per_core = 2048 × N_per × 4 bytes
  LX is 2 MiB per core
  C_psum = 2 MiB at N_per = 256 → N = 4096

So:
  N=2048  → N_per=128, C_psum=1 MB    (well under LX)
  N=3072  → N_per=192, C_psum=1.5 MB  (under LX)
  N=4096  → N_per=256, C_psum=2 MB    (AT LX boundary)
  N=6144  → N_per=384, C_psum=3 MB    (1.5× over LX)
  N=8192  → N_per=512, C_psum=4 MB    (2× over LX)

For each, measure id and kf walls. The step (id − kf) is the ring-
attributable cost. The slope of step / payload should jump at the
LX boundary if C_psum overflow inflates the per-MB cost.
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

WARMUP = 3
ITERS = 8
DTYPE = torch.float16


def perm_identity():
    return list(range(32))


def perm_kfast(m, n, k):
    mn = m * n
    return [(c % k) * mn + (c // k) for c in range(32)]


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


def main():
    print("# Probe §11.6 — SFP per-MB inflation across LX boundary")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("Shape: (2048, N, 8192) at split (1, 16, 2)")
    print("Sweep N to vary C_psum across the 2 MiB LX boundary.")
    print()

    M = 2048
    K = 8192
    Ns = [2048, 3072, 4096, 6144, 8192]
    split = (1, 16, 2)
    m, n, k_split = split

    print("| N | N_per | C_psum/core | id wall ms | kf wall ms | step ms | step/MB |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    rows = []
    for N in Ns:
        N_per = N // n
        C_psum = M * N_per * 4
        cohort_payload = (
            C_psum  # for (1, 16, 2) M_per=M, so cohort_payload = M × N_per × 4
        )

        id_perm = perm_identity()
        kf_perm = perm_kfast(m, n, k_split)

        id_ms, id_err = measure(M, N, K, split, id_perm)
        kf_ms, kf_err = measure(M, N, K, split, kf_perm)

        if id_ms is None or kf_ms is None:
            err = id_err or kf_err
            c_psum_str = f"{C_psum / 1024 / 1024:.2f} MB"
            lx_marker = " (>LX)" if C_psum > 2 * 1024 * 1024 else ""
            print(f"| {N} | {N_per} | {c_psum_str}{lx_marker} | ERR ({err}) |")
            sys.stdout.flush()
            continue

        step = id_ms - kf_ms
        step_per_mb = step / (cohort_payload / 1024 / 1024) if cohort_payload > 0 else 0
        c_psum_str = f"{C_psum / 1024 / 1024:.2f} MB"
        lx_marker = " (>LX)" if C_psum > 2 * 1024 * 1024 else ""
        print(
            f"| {N} | {N_per} | {c_psum_str}{lx_marker} | "
            f"{id_ms:.3f} | {kf_ms:.3f} | {step:.3f} | "
            f"{step_per_mb:.2f} ms/MB |"
        )
        sys.stdout.flush()
        rows.append((N, N_per, C_psum, id_ms, kf_ms, step, step_per_mb))

    print("\n## LX-boundary inflation analysis")
    print()
    LX_BYTES = 2 * 1024 * 1024
    below = [r for r in rows if r[2] <= LX_BYTES]
    above = [r for r in rows if r[2] > LX_BYTES]
    if below and above:
        avg_per_mb_below = sum(r[6] for r in below) / len(below)
        avg_per_mb_above = sum(r[6] for r in above) / len(above)
        ratio = (
            avg_per_mb_above / avg_per_mb_below
            if avg_per_mb_below > 0
            else float("inf")
        )
        print(f"  avg step/MB BELOW LX: {avg_per_mb_below:.2f} ms/MB")
        print(f"  avg step/MB ABOVE LX: {avg_per_mb_above:.2f} ms/MB")
        print(f"  inflation ratio: {ratio:.2f}× (above/below)")
        if ratio > 1.5:
            print()
            print(
                "  → LX-overflow inflates per-MB ring cost by "
                f"{ratio:.1f}×. The §6.2 per-MB difference between "
                "L3-70B and DSv3 likely comes from this regime."
            )
        elif ratio < 0.7:
            print()
            print(
                "  → LX-overflow REDUCES per-MB ring cost — unexpected. "
                "Maybe HMI dominates more above LX boundary."
            )
        else:
            print()
            print("  → No clean inflation; per-MB cost is similar both sides.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
