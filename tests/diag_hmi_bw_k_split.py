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

"""Effective HMI bytes & PSUM cost under K-split (kf + id emissions).

Phase 0 left two structural residuals in the cost model:

  1b. K-split shapes are over-predicted by ~95% under k_fast emission.
      Hypothesis: each k-collaborator cluster fetches only its K/k
      slice of B (and A), not the full broadcast. To test, sweep k ∈
      {1, 2, 4} on wide-B shapes with k_fast and back out which
      candidate byte model matches.

  2.  Identity-emission rows (1, 16, 2)+id show opposite-sign errors
      across shapes (DSv3 o_proj +60% under, DSv3 down_proj +100%
      over). Same emission, payload-dependent residual. To probe,
      run both kf and id at the same shape so the wall delta isolates
      the PSUM ring cost.

Strategy: for each shape, measure (32,1,1) baseline + (1,16,2)
{kf, id} + (1,8,4) {kf, id}. Subtracting kf from id at fixed shape
gives PSUM cost in isolation; subtracting (32,1,1) from kf gives the
HMI delta from k-split.

Usage: python tests/diag_hmi_bw_k_split.py
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
HMI_BW_GBS = 40.0  # from pure-M probe
SFP_BW_GBS = 32.0


# Four shapes chosen to cross the residual patterns:
#   - DSv3 o_proj M=128: big-B HMI-bound, the canonical kf win
#   - DSv3 o_proj M=2048: big-B + big PSUM payload (under-pred row)
#   - DSv3 down_proj M=2048: small-K + big payload (over-pred row)
#   - L3-70B kv_proj M=2048: narrow-N control
SHAPES = [
    ("DSv3 o_proj M=128",     128,  7168, 16384),
    ("DSv3 o_proj M=2048",    2048, 7168, 16384),
    ("DSv3 down_proj M=2048", 2048, 7168, 2048),
    ("L3-70B kv_proj M=2048", 2048, 1024, 8192),
]

# (label, split, k_fast). k=1 row is the pure-M baseline.
CONFIGS = [
    ("pure-M",     (32, 1, 1), False),
    ("(1,16,2)kf", (1, 16, 2), True),
    ("(1,16,2)id", (1, 16, 2), False),
    ("(1,8,4)kf",  (1, 8, 4),  True),
    ("(1,8,4)id",  (1, 8, 4),  False),
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


@contextmanager
def _emission(k_fast: bool):
    """Toggle the k_fast core-id permutation for the duration."""
    prev_perm = ts_config.core_id_permutation
    prev_rev = ts_config.core_emission_reverse
    ts_config.core_id_permutation = "k_fast" if k_fast else "identity"
    ts_config.core_emission_reverse = False
    try:
        yield
    finally:
        ts_config.core_id_permutation = prev_perm
        ts_config.core_emission_reverse = prev_rev


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


def _compile_and_bench(M, N, K, split, k_fast):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _emission(k_fast), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _emission(k_fast), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:80]}"


# ---- candidate HMI byte models ----------------------------------------

def _bytes_full_broadcast(M, N, K, k, db=2):
    return (M * K + K * N + M * N) * db


def _bytes_per_cluster(M, N, K, k, db=2):
    return ((M * K + K * N) // k + M * N) * db


def _psum_bytes(M, N, m, n, k, k_fast, db_psum=4):
    if k <= 1:
        return 0
    payload = (M // m) * (N // n) * db_psum
    hops = 1 if k_fast else (m * n)
    return (m * n) * (k - 1) * hops * payload


def _predicted_wall_ms(M, N, K, m, n, k, k_fast, bytes_fn):
    M_per, N_per, K_per = M // m, N // n, K // k
    if M_per < 1 or N_per < 1:
        return None
    pt_util = min(1.0, M_per / 8) * min(1.0, N_per / 64)
    if pt_util <= 0:
        return None
    flops = 2 * M_per * N_per * K_per
    compute_ms = flops / (1e12 * pt_util) * 1e3
    hmi_ms = bytes_fn(M, N, K, k) / (HMI_BW_GBS * 1e9) * 1e3
    psum_ms = _psum_bytes(M, N, m, n, k, k_fast) / (SFP_BW_GBS * 1e9) * 1e3
    return max(compute_ms, hmi_ms + LAUNCH_FLOOR_MS) + psum_ms


# ---- output -----------------------------------------------------------

def main() -> int:
    print("# K-split HMI bytes + PSUM cost probe\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32, HMI_BW={HMI_BW_GBS} GB/s\n")
    print("| shape | config | wall ms | pred-full ms | pred-cluster ms |")
    print("|---|---|---:|---:|---:|")

    results = {}
    for label, M, N, K in SHAPES:
        for cfg_label, split, k_fast in CONFIGS:
            ms, err = _compile_and_bench(M, N, K, split, k_fast)
            if err:
                print(f"| {label} | {cfg_label} | ERR: {err[:30]} | — | — |")
                continue
            m, n, k = split
            pred_full = _predicted_wall_ms(M, N, K, m, n, k, k_fast, _bytes_full_broadcast)
            pred_clus = _predicted_wall_ms(M, N, K, m, n, k, k_fast, _bytes_per_cluster)
            results[(label, cfg_label)] = (ms, pred_full, pred_clus, M, N, K, m, n, k, k_fast)
            print(f"| {label} | {cfg_label} | "
                  f"{ms:.3f} | {pred_full:.3f} | {pred_clus:.3f} |")
    print()

    # ---- per-shape decompositions -----------------------------------
    print("## Per-shape PSUM cost (id − kf at fixed split)\n")
    print("Subtracting kf wall from id wall isolates the m·n-hop PSUM cost:")
    print()
    print("| shape | split | id ms | kf ms | id − kf | model PSUM (kf=1hop) | model PSUM (id=mn hops) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for label, M, N, K in SHAPES:
        for split_label, m, n, k in [("(1,16,2)", 1, 16, 2), ("(1,8,4)", 1, 8, 4)]:
            kf_key = (label, f"{split_label}kf")
            id_key = (label, f"{split_label}id")
            if kf_key not in results or id_key not in results:
                continue
            kf_ms = results[kf_key][0]
            id_ms = results[id_key][0]
            payload = (M // m) * (N // n) * 4
            psum_kf = (m * n) * (k - 1) * 1 * payload / (SFP_BW_GBS * 1e9) * 1e3
            psum_id = (m * n) * (k - 1) * (m * n) * payload / (SFP_BW_GBS * 1e9) * 1e3
            print(f"| {label} | {split_label} | "
                  f"{id_ms:.2f} | {kf_ms:.2f} | {id_ms - kf_ms:+.2f} | "
                  f"{psum_kf:.2f} | {psum_id:.2f} |")
    print()

    # ---- per-shape HMI delta ----------------------------------------
    print("## Per-shape HMI delta (kf − pure-M, with PSUM subtracted)\n")
    print("kf − pure-M wall is dominated by HMI byte change (PSUM is small under kf).")
    print("If per-cluster model is right, kf wall ≈ pure-M wall − (k−1)/k × bytes/BW.")
    print()
    print("| shape | split | pure-M ms | kf ms | pred-full kf ms | pred-cluster kf ms | best fit |")
    print("|---|---|---:|---:|---:|---:|---|")
    for label, M, N, K in SHAPES:
        baseline = results.get((label, "pure-M"))
        if not baseline:
            continue
        for split_label in ["(1,16,2)", "(1,8,4)"]:
            kf_key = (label, f"{split_label}kf")
            if kf_key not in results:
                continue
            kf_ms = results[kf_key][0]
            pred_full = results[kf_key][1]
            pred_clus = results[kf_key][2]
            err_full = abs(pred_full - kf_ms) / kf_ms * 100
            err_clus = abs(pred_clus - kf_ms) / kf_ms * 100
            best = "cluster" if err_clus < err_full else "full"
            print(f"| {label} | {split_label} | "
                  f"{baseline[0]:.2f} | {kf_ms:.2f} | "
                  f"{pred_full:.2f} ({err_full:.0f}%) | "
                  f"{pred_clus:.2f} ({err_clus:.0f}%) | {best} |")
    print()

    print("## Reading guide\n")
    print(
        "  - HMI bytes model: which of full-broadcast vs per-cluster\n"
        "    matches kf walls more closely. If per-cluster wins, the cost\n"
        "    model needs `bytes /= k` for k>1.\n"
        "  - PSUM cost: id − kf gives empirical PSUM ms. Compare to model's\n"
        "    1-hop and m·n-hop predictions. If empirical lies between, the\n"
        "    hops formula is wrong; if either matches, refine that branch.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
