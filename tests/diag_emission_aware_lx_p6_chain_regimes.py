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

"""Probe 6 — map the three (m, 1, k) chain-length regimes.

Probe 4 found on DSv3 o_proj M=2048 a non-monotonic wall structure
across the (m, 1, k)+kf family:

  chain=2:  18.6 ms (pipeline)
  chain=4:  18.1 ms (pipeline)
  chain=8:  56.9 ms (sync — jump)
  chain=16: 59.1 ms (sync)
  chain=32: 30.4 ms (allreduce — drop)

Three apparent regimes within the n=1 streaming path. This probe
tests whether the boundaries are universal across shapes by running
the full (m, 1, k) sweep on three production shapes:

  - DSv3 o_proj M=2048    (replicate)
  - Mixtral gate_proj M=2048 (smaller K, smaller N)
  - DSv3 gate_proj M=2048 (larger K, larger N)

For each shape we report measured wall plus a cost-model decomposition
(compute + HMI + LF) so the residual makes the regime structure
visible. The "regime cost" = wall - (compute + HMI + LF) is what the
streaming path adds beyond the predictable terms.

Hypothesis dimensions:

  - If regime boundaries depend on chain length alone (chain ≤ 4 fast,
    8-16 sync, 32 allreduce), they're universal.
  - If they depend on per-chain payload (M_per × N × dtype), bigger
    payloads should hit the sync regime sooner.
  - If they depend on K_per (compute per chain step), shapes with
    bigger K may behave differently.

Usage:
    python tests/diag_emission_aware_lx_p6_chain_regimes.py
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
LAUNCH_FLOOR_MS = 3.0
HMI_BW_GBS = 40.0
SFP_BW_GBS = 32.0
PT_PEAK_TFLOPS = 1.0


SHAPES = [
    ("DSv3 o_proj M=2048",       2048,  7168, 16384),
    ("DSv3 gate_proj M=2048",    2048, 18432,  7168),
    ("Mixtral gate_proj M=2048", 2048, 14336,  4096),
]


# Full (m, 1, k) family + catastrophic control for context.
CONFIGS = [
    ((32, 1, 1), "identity", "pure-M"),
    ((16, 1, 2), "k_fast",   "chain=2"),
    ((8,  1, 4), "k_fast",   "chain=4"),
    ((4,  1, 8), "k_fast",   "chain=8"),
    ((2,  1, 16), "k_fast",  "chain=16"),
    ((1,  1, 32), "k_fast",  "chain=32"),
    ((1,  8, 4),  "k_fast",  "n=8 control"),
]


# ---- machinery ----------------------------------------------------

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
        return None, f"{type(e).__name__}: {str(e)[:60]}"


# ---- cost-model decomposition (no LX overflow penalty applied) ----

def _predict_components(M, N, K, split):
    """Return (compute_ms, hmi_ms, lf_ms) for a clean baseline.

    Uses per-cluster HMI bytes (which we know is correct from Track 2
    Phase 0) and PT util model from hmi_cost_model.
    """
    m, n, k = split
    M_per, N_per, K_per = M // m, N // n, K // k
    pt_util = min(1.0, M_per / 8) * min(1.0, N_per / 64)
    if pt_util <= 0:
        return float("inf"), 0, LAUNCH_FLOOR_MS
    flops = 2 * M_per * N_per * K_per
    compute_ms = flops / (PT_PEAK_TFLOPS * 1e12 * pt_util) * 1e3
    # per-cluster bytes
    hmi_bytes = ((M * K + K * N) // k + M * N) * 2  # fp16
    hmi_ms = hmi_bytes / (HMI_BW_GBS * 1e9) * 1e3
    return compute_ms, hmi_ms, LAUNCH_FLOOR_MS


def _is_valid(M, N, K, split):
    m, n, k = split
    if M % m or N % n or K % k:
        return False
    if (N // n) % 64 != 0:
        return False
    return True


# ---- main ----------------------------------------------------------

def main() -> int:
    print("# Probe 6 — chain-length regime structure within (m, 1, k)+kf\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16\n")
    print("Decomposition: 'predicted base' = max(compute, hmi+LF) using "
          "per-cluster bytes. 'regime cost' = wall − base.\n")

    for label, M, N, K in SHAPES:
        print(f"## {label}  shape=({M}, {N}, {K})\n")
        print("| split | desc | wall ms | compute | hmi | base "
              "(max(c, hmi+LF)) | regime cost | regime |")
        print("|---|---|---:|---:|---:|---:|---:|---|")

        baseline_wall = None  # store pure-M wall for ratio classification
        for split, perm, desc in CONFIGS:
            if not _is_valid(M, N, K, split):
                print(f"| {split} | {desc} | SKIP | — | — | — | — | — |")
                continue
            ms, err = _compile_and_bench(M, N, K, split, perm)
            if ms is None:
                print(f"| {split} | {desc} | ERR ({err[:25]}) | — | — | — | — | — |")
                continue
            compute, hmi, lf = _predict_components(M, N, K, split)
            base = max(compute, hmi + lf)
            regime_cost = ms - base
            if split == (32, 1, 1):
                baseline_wall = ms
            # classify regime: relative to pure-M baseline
            if baseline_wall is not None and baseline_wall > 0:
                ratio = ms / baseline_wall
                if ratio < 1.5:
                    regime = "pipeline"
                elif ratio < 2.5:
                    regime = "allreduce/edge"
                elif ratio < 8:
                    regime = "sync"
                else:
                    regime = "catastrophic"
            else:
                regime = "?"
            print(f"| {split} | {desc} | {ms:.2f} | {compute:.2f} | "
                  f"{hmi:.2f} | {base:.2f} | "
                  f"{regime_cost:+.2f} | {regime} |")
        print()

    print("## Reading guide\n")
    print("Compare the regime cost across chain lengths within each shape.")
    print("If the (chain ≤ 4) → (chain 8, 16) jump is consistent across")
    print("shapes, the regime boundary is universal.")
    print("If different shapes have different boundaries, the trigger")
    print("depends on something shape-specific (M_per × N? K_per? both?).")
    print()
    print("If chain=32 'allreduce' regime cost is smaller than chain=8/16:")
    print("  the kernel template uses a different reduction primitive at")
    print("  chain length = max_cores. Worth understanding architecturally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
