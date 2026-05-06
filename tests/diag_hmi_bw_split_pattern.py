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

"""HMI BW gap investigation — does access pattern move the 40 GB/s number?

Earlier pure-M probe (diag_hmi_bw_pure_m.py) found achieved HMI BW
asymptotes to ~40 GB/s on broadcast-B accounting, vs the 67 GB/s
spec headline. Open question: is the gap fixable from torch_spyre
(by picking different splits / access patterns) or is it a
deeptools/runtime ceiling?

This probe tests the same-shape, different-split hypothesis. Total
HMI bytes are identical across split choices (each is M·K + K·N + M·N
under broadcast accounting), but the cores-share-slices pattern
differs:

  (32, 1, 1) — pure-M:  B (= K·N) broadcast to all 32 cores.
                        A split into 32 unique chunks.
  (1, 32, 1) — pure-N:  A (= M·K) broadcast to all 32 cores.
                        B split into 32 unique chunks.
  (1, 16, 2) — k-split: A and B partially shared by k-cluster.
  (1,  8, 4) — k-split: smaller cluster, more sharing per k-slice.
  (16, 2, 1) — m-n:     mixed split, less broadcast either side.

If achieved BW differs across splits → access pattern is a torch_spyre
lever; planner could pick splits to maximize BW.
If BW is flat across splits → ceiling lives in deeptools/runtime.

Usage:
    python tests/diag_hmi_bw_split_pattern.py
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


WARMUP = 3
ITERS = 12
DTYPE = torch.float16
LAUNCH_FLOOR_MS = 3.0
PT_PEAK_TFLOPS_PER_CORE = 1.0


# Shape choices — each (M, N, K) tested across multiple splits.
SHAPES = [
    # Wide-B (HMI-bound): exposes BW clearly because compute is smaller.
    ("wide-B M=128",  128, 8192, 8192),
    ("wide-B M=256",  256, 8192, 8192),
    # Narrower B but still HMI-bound at small M
    ("narrow-B M=128", 128, 4096, 4096),
]

# (label, split, valid_for_min_M)
# split is (m, n, k); product must equal 32.
SPLITS = [
    ("pure-M",     (32, 1, 1)),
    ("pure-N",     (1, 32, 1)),
    ("pure-K",     (1, 1, 32)),
    ("(1,16,2)",   (1, 16, 2)),
    ("(1,8,4)",    (1, 8, 4)),
    ("(2,16,1)",   (2, 16, 1)),
    ("(8,4,1)",    (8, 4, 1)),
    ("(16,2,1)",   (16, 2, 1)),
]


# ---- machinery (copy of diag_hmi_bw_pure_m.py mechanism) -------------

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
    return statistics.median(samples) * 1e3  # ms


def _compile_and_bench(M, N, K, target):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(target):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split(target):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


# ---- analysis --------------------------------------------------------

def _broadcast_hmi_bytes(M, N, K, dtype_bytes=2):
    """HMI bytes under broadcast accounting (all unique chunks, summed).
    Same for any split."""
    return (M * K + K * N + M * N) * dtype_bytes


def _predicted_compute_ms(M, N, K, split, num_cores=32):
    """Per-core compute time at given split, full PT util."""
    m, n, k = split
    M_per, N_per, K_per = M // m, N // n, K // k
    if M_per < 1 or N_per < 1:
        return None
    pt_util = min(1.0, M_per / 8) * min(1.0, N_per / 64)
    if pt_util <= 0:
        return None
    flops_per_core = 2 * M_per * N_per * K_per
    return flops_per_core / (PT_PEAK_TFLOPS_PER_CORE * 1e12 * pt_util) * 1e3


def main() -> int:
    print("# HMI BW gap — does access pattern move the 40 GB/s number?\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")
    print(f"# Implied BW = bytes / (wall - max(LF, compute))")
    print(f"# Total HMI bytes = M*K + K*N + M*N (broadcast model, "
          f"same for all splits)\n")

    print("| shape | (M, N, K) | bytes (MB) | split | wall ms | "
          "compute ms | hmi ms | implied BW (GB/s) |")
    print("|---|---|---:|---|---:|---:|---:|---:|")

    rows = []
    for label, M, N, K in SHAPES:
        bytes_total = _broadcast_hmi_bytes(M, N, K)
        for split_label, split in SPLITS:
            ms, err = _compile_and_bench(M, N, K, split)
            if err:
                print(f"| {label} | ({M},{N},{K}) | "
                      f"{bytes_total / 1e6:.0f} | {split_label} | "
                      f"ERR: {err[:30]} | — | — | — |")
                continue
            compute_ms = _predicted_compute_ms(M, N, K, split)
            non_hmi = max(LAUNCH_FLOOR_MS, compute_ms or 0)
            hmi_ms = ms - non_hmi
            if hmi_ms <= 0:
                bw = float("inf")
            else:
                bw = bytes_total / (hmi_ms * 1e-3) / 1e9
            rows.append((label, M, N, K, split_label, ms, compute_ms,
                         hmi_ms, bw))
            print(f"| {label} | ({M},{N},{K}) | {bytes_total / 1e6:.0f} | "
                  f"{split_label} | {ms:.3f} | "
                  f"{compute_ms or 0:.3f} | {hmi_ms:+.3f} | {bw:.1f} |")
    print()

    # Per-shape: which split gave best/worst BW?
    print("## Per-shape best/worst split for BW\n")
    by_shape = {}
    for r in rows:
        key = (r[0], r[1], r[2], r[3])
        by_shape.setdefault(key, []).append(r)
    for key, shape_rows in by_shape.items():
        valid = [r for r in shape_rows if r[7] > 0 and r[8] != float("inf")]
        if not valid:
            continue
        best = max(valid, key=lambda r: r[8])
        worst = min(valid, key=lambda r: r[8])
        print(f"  {key[0]}: best = {best[4]} ({best[8]:.1f} GB/s), "
              f"worst = {worst[4]} ({worst[8]:.1f} GB/s), "
              f"spread = {best[8] - worst[8]:.1f} GB/s")
    print()

    # Aggregate
    print("## Aggregate stats\n")
    bws = [r[8] for r in rows if r[7] > 0 and r[8] != float("inf")]
    if bws:
        print(f"  rows: {len(bws)}")
        print(f"  median BW: {statistics.median(bws):.1f} GB/s")
        print(f"  range:     {min(bws):.1f} – {max(bws):.1f} GB/s")
        print(f"  spread:    {max(bws) - min(bws):.1f} GB/s")

    print()
    print("## Reading guide\n")
    print(
        "  - If BW spread per shape is < 5 GB/s: access pattern doesn't matter.\n"
        "    The 40 GB/s ceiling lives in the kernel template / runtime.\n"
        "    No torch_spyre lever to close the gap.\n"
        "  - If BW spread > 10 GB/s: access pattern matters. The planner could\n"
        "    bias toward higher-BW splits. torch_spyre lever exists.\n"
        "  - If pure-K consistently wins big: K-split shrinks per-core HMI demand\n"
        "    in a way the broadcast model under-counts.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
