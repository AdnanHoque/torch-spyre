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

"""Probe 3 — N-axis sweep at fixed (1, 8, 4)+kf to map mid-k catastrophe.

Phase 0 found: at (1, 8, 4)+kf, DSv3 o_proj M=2048 (N=7168) runs
127 ms but L3-70B kv_proj M=2048 (N=1024) runs only 5.82 ms — 22×
slower for ~7× larger N. Per-core compute is identical (M_per × N_per
× K_per is invariant under any (1, n, k) with n·k = 32 — wait, no,
N_per × K_per shrinks as k grows but per-core MAC count = M_per ×
N_per × K_per = M·N·K/(m·n·k) which IS constant).

Question: at fixed M=2048, K=8192, split=(1, 8, 4)+kf, what N
threshold turns "fast" into "catastrophe"?

  - Per-core MAC count = M × N × K / 32 — scales linearly with N.
    Compute-bound prediction: wall ≈ 0.13 × N / 1024 ms (fp16
    peak ≈ 1 TFLOP/core).
  - The catastrophe at DSv3 o_proj (N=7168) is ~14× over
    compute-bound. If it's a per-N_per-tile re-fetch issue, wall
    should grow super-linearly with N once N exceeds some threshold.

N values (must be divisible by n=8 and stick-aligned at N_per ≥ 64):
  512, 1024, 2048, 4096, 6144, 8192

We compare:
  - (1, 8, 4) + k_fast — the "catastrophe candidate"
  - (32, 1, 1) + identity — pure-M baseline at the same shape

If wall(1,8,4) tracks pure-M wall up to some N then diverges, the
threshold tells us the kernel-template inflection point. If the
divergence is linear with N, it's a per-N_per kernel-template
overhead.

Usage:
    python tests/diag_emission_aware_lx_p3_midk_n_sweep.py
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from pathlib import Path
import sys

import torch

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
from torch_spyre._inductor import config as ts_config  # noqa: E402


WARMUP = 3
ITERS = 12
DTYPE = torch.float16

M_FIXED = 2048
K_FIXED = 8192
N_VALUES = (512, 1024, 2048, 4096, 6144, 8192)


# ---- machinery (mirrors Probe 1) -----------------------------------

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


@contextmanager
def _permutation(name: str):
    prev = ts_config.core_id_permutation
    ts_config.core_id_permutation = name
    try:
        yield
    finally:
        ts_config.core_id_permutation = prev


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


def _compile_and_bench(M, N, K, split, perm):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _permutation(perm), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _permutation(perm), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:80]}"


# ---- main ----------------------------------------------------------

def main() -> int:
    print("# Probe 3 — N-axis sweep at fixed (1, 8, 4)+kf and (32,1,1)+id\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, "
          f"M={M_FIXED}, K={K_FIXED}\n")
    print("Goal: pin down the N threshold where (1,8,4)+kf wall departs\n"
          "from the (32,1,1) baseline — locating the mid-k catastrophe\n"
          "regime by N rather than by k.\n")

    print("| N | N_per @ k=4 | (32,1,1) ms | (1,8,4)+kf ms | ratio | "
          "compute-bound est ms |")
    print("|---:|---:|---:|---:|---:|---:|")

    for N in N_VALUES:
        n = 8
        k = 4
        if N % n != 0:
            continue
        N_per = N // n
        if N_per % 64 != 0:
            continue

        # Compute-bound estimate: per-core MACs = M × N × K / 32
        per_core_macs = M_FIXED * N * K_FIXED / 32
        # peak fp16 ≈ 1 TFLOP/core; pt_util ≈ 1 for these shapes
        compute_ms_est = 2 * per_core_macs / 1e12 * 1e3

        pm_ms, pm_err = _compile_and_bench(
            M_FIXED, N, K_FIXED, (32, 1, 1), "identity")
        kf_ms, kf_err = _compile_and_bench(
            M_FIXED, N, K_FIXED, (1, 8, 4), "k_fast")

        pm_str = f"{pm_ms:.3f}" if pm_ms is not None else f"ERR ({pm_err[:20]})"
        kf_str = f"{kf_ms:.3f}" if kf_ms is not None else f"ERR ({kf_err[:20]})"
        if pm_ms is not None and kf_ms is not None:
            ratio = kf_ms / pm_ms
            ratio_str = f"{ratio:.2f}×"
        else:
            ratio_str = "—"

        print(f"| {N} | {N_per} | {pm_str} | {kf_str} | {ratio_str} | "
              f"{compute_ms_est:.2f} |")

    print()
    print("## Reading guide\n")
    print("If walls track each other: no catastrophe in this regime")
    print("If kf wall has a knee at some N: that's the kernel-template "
          "inflection point")
    print("If kf wall grows linearly with N while (32,1,1) sublinear: "
          "per-N_per overhead in K-split kernel template")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
