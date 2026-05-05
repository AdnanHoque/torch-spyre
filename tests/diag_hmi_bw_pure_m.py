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

"""Effective HMI bandwidth probe under pure-M (32, 1, 1) split.

The Phase 0 cost model assumes B is HMI-fetched once and ring-
broadcast across all 32 cores under pure-M, giving total HMI bytes
= M*K + K*N + M*N. For some shapes this matches measurement; for
others (DSv3 o_proj-style wide-B) the model under-predicts wall
time by 2-3x.

This probe varies B size (K*N) systematically under pure-M, holding
M small enough that compute is sub-dominant. Extracting effective
HMI time per measurement and fitting against B bytes tells us
whether:

  (a) B is broadcast once (BW ~ 67 GB/s)
  (b) B is partially replicated (effective bytes > broadcast model)
  (c) Effective BW drops at large B (HMI saturates differently)

For each shape the probe forces (32, 1, 1), measures wall time,
subtracts model-predicted compute time, and prints the implied
effective HMI bytes-per-second.

Usage: python tests/diag_hmi_bw_pure_m.py
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


# (label, M, N, K). All forced to (32, 1, 1).
SHAPES = [
    # Very small B
    ("B=8MB  M=64",       64, 1024, 4096),
    ("B=8MB  M=512",      512, 1024, 4096),
    # Small B
    ("B=32MB M=64",       64, 4096, 4096),
    ("B=32MB M=512",      512, 4096, 4096),
    # Medium B
    ("B=128MB M=64",      64, 8192, 8192),
    ("B=128MB M=512",     512, 8192, 8192),
    # Large B (DSv3 o_proj region)
    ("B=235MB M=64",      64, 7168, 16384),
    ("B=235MB M=512",     512, 7168, 16384),
    ("B=256MB M=64",      64, 8192, 16384),
    ("B=256MB M=512",     512, 8192, 16384),
    # Very large B
    ("B=512MB M=64",      64, 16384, 16384),
    ("B=512MB M=512",     512, 16384, 16384),
]


# ---- machinery --------------------------------------------------------

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
    return statistics.median(samples) * 1e3


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
        return None, f"{type(e).__name__}: {str(e)[:100]}"


# ---- analysis helpers -------------------------------------------------

def _predicted_compute_ms(M, N, K, num_cores=32):
    """Compute time per core at pure-M with full PT utilisation."""
    M_per = M // num_cores
    if M_per < 1:
        return None
    pt_util = min(1.0, M_per / 8) * min(1.0, N / 64)
    flops = 2 * M_per * N * K
    return flops / (PT_PEAK_TFLOPS_PER_CORE * 1e12 * pt_util) * 1e3


def _broadcast_hmi_bytes(M, N, K, dtype_bytes=2):
    """HMI bytes assuming B is broadcast once (pure-M idealised)."""
    return (M * K + K * N + M * N) * dtype_bytes


def _per_core_hmi_bytes(M, N, K, dtype_bytes=2):
    """HMI bytes assuming B is fetched per-core (32x replicated)."""
    return (M * K + 32 * K * N + M * N) * dtype_bytes


def main() -> int:
    print("# Effective HMI bandwidth under pure-M\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32, "
          f"split=(32,1,1)\n")
    print(f"# B (broadcast model) = M*K + K*N + M*N bytes")
    print(f"# B (replicated model) = M*K + 32*K*N + M*N bytes")
    print()

    print("| shape | M | N | K | B_bcast (MB) | wall ms | compute ms | "
          "(wall - max(LF, compute)) ms | eff BW (GB/s, bcast) | "
          "eff BW (GB/s, repl) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    rows = []
    for label, M, N, K in SHAPES:
        ms, err = _compile_and_bench(M, N, K, (32, 1, 1))
        if err:
            print(f"| {label} | {M} | {N} | {K} | — | ERR | — | — | — | — |")
            continue

        bcast_bytes = _broadcast_hmi_bytes(M, N, K)
        repl_bytes = _per_core_hmi_bytes(M, N, K)
        compute_ms = _predicted_compute_ms(M, N, K)
        # HMI time by subtraction: wall = max(LF, max(compute, hmi))
        non_hmi = max(LAUNCH_FLOOR_MS, compute_ms or 0)
        hmi_ms = ms - non_hmi
        if hmi_ms <= 0:
            eff_bw_bcast = float("inf")
            eff_bw_repl = float("inf")
        else:
            eff_bw_bcast = bcast_bytes / (hmi_ms * 1e-3) / 1e9
            eff_bw_repl = repl_bytes / (hmi_ms * 1e-3) / 1e9

        rows.append((label, M, N, K, ms, compute_ms, hmi_ms,
                     bcast_bytes, repl_bytes,
                     eff_bw_bcast, eff_bw_repl))
        print(f"| {label} | {M} | {N} | {K} | "
              f"{bcast_bytes / 1e6:.0f} | "
              f"{ms:.3f} | {compute_ms:.3f} | "
              f"{hmi_ms:+.3f} | {eff_bw_bcast:.1f} | {eff_bw_repl:.1f} |")
    print()

    # Summary stats
    print("## Effective HMI BW (subtracted method)\n")
    valid = [r for r in rows if r[6] > 0]
    if valid:
        bcast_bws = [r[9] for r in valid]
        repl_bws = [r[10] for r in valid]
        print(f"  Broadcast-model effective BW: median={statistics.median(bcast_bws):.1f} GB/s, "
              f"min={min(bcast_bws):.1f}, max={max(bcast_bws):.1f}")
        print(f"  Replicated-model effective BW: median={statistics.median(repl_bws):.1f} GB/s, "
              f"min={min(repl_bws):.1f}, max={max(repl_bws):.1f}")
    print()
    print("## Verdict reading guide\n")
    print(
        "  - If broadcast-model BW clusters around 67 GB/s: B IS broadcast,\n"
        "    HMI BW spec is correct. Cost model mis-estimates compute.\n"
        "  - If replicated-model BW clusters around 67 GB/s: B is fetched\n"
        "    per-core under pure-M; cost model needs the 32x B factor.\n"
        "  - If neither, BW depends on shape — fit a regression.\n"
        "  - If broadcast-model BW < 67 GB/s consistently: BW spec is\n"
        "    overstated for this access pattern; lower the constant.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
