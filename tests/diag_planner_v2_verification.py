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

"""Hardware verification of planner v2 prototype's Tier 2 picks.

Selects 8 representative ops from the prototype's Tier 2 list (≥10%
predicted speedup, pure-M doesn't overflow). For each:

  - Compile + measure pure-M (32, 1, 1) baseline.
  - Compile + measure the v2 pick under k_fast emission.
  - Report measured walls vs cost-model V4 predictions.
  - Compare measured speedup to predicted speedup.

Selection criteria (8 rows):

  Sanity (have validation set data — predictions should match):
    1. L3-70B kv_proj M=2048 (1, 16, 2)+kf   pred 4.00, validation 3.94
    2. DSv3 q_a_proj M=128 (1, 8, 4)+kf      pred 3.23, validation 3.22

  Small-M big predicted speedup (highest risk — small-M HMI BW issue):
    3. L3-70B q_proj M=32   (1, 4, 8)+kf   pred 3.66 vs pure-M 6.38 (1.74×)
    4. DSv3 gate_proj M=32  (1, 4, 8)+kf   pred 4.37 vs pure-M 9.65 (2.21×)

  Medium-M moderate speedup:
    5. L3-70B q_proj M=128  (1, 8, 4)+kf   pred 4.30 vs pure-M 6.46 (1.50×)
    6. L3-70B q_proj M=512  (1,16, 2)+kf   pred 5.52 vs pure-M 6.77 (1.23×)

  Wide-K with K-split:
    7. DSv3 down_proj M=128 (1, 4, 8)+kf   pred 3.86 vs pure-M 6.05 (1.57×)
    8. DSv3 down_proj M=512 (1, 8, 4)+kf   pred 4.72 vs pure-M 6.41 (1.36×)

A v2 pick "validates" if:
  - measured v2 wall is faster than measured pure-M wall, AND
  - cost-model V4 prediction is within ±15% of measured

Usage:
    python tests/diag_planner_v2_verification.py
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
from tests.hmi_cost_model import predict  # noqa: E402


WARMUP = 3
ITERS = 12
DTYPE = torch.float16


# (label, M, N, K, v2_split, predicted_v2_ms_from_prototype)
ROWS = [
    # Sanity checks — already-validated rows from the 30-row Project B set.
    ("L3-70B kv_proj M=2048   sanity",  2048, 1024, 8192,  (1, 16, 2), 4.00),
    ("DSv3 q_a_proj M=128     sanity",   128, 1536, 7168,  (1,  8, 4), 3.23),

    # Small-M big speedup — risky (small-M HMI BW residual).
    ("L3-70B q_proj M=32      big-spd",   32, 8192, 8192,  (1,  4, 8), 3.66),
    ("DSv3 gate_proj M=32     big-spd",   32, 18432, 7168, (1,  4, 8), 4.37),

    # Medium-M moderate speedup.
    ("L3-70B q_proj M=128     mid-spd",  128, 8192, 8192,  (1,  8, 4), 4.30),
    ("L3-70B q_proj M=512     mid-spd",  512, 8192, 8192,  (1, 16, 2), 5.52),

    # Wide-K K-split.
    ("DSv3 down_proj M=128    wide-K",   128, 7168, 18432, (1,  4, 8), 3.86),
    ("DSv3 down_proj M=512    wide-K",   512, 7168, 18432, (1,  8, 4), 4.72),
]


# ---- machinery (mirrors prior probes) -----------------------------

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


# ---- main ----------------------------------------------------------

def main() -> int:
    print("# Planner v2 prototype — hardware verification\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16\n")
    print("Validation rule: v2 pick validates if measured v2 < measured pure-M")
    print("                 AND |cost-model error| ≤ 15%.\n")
    print("| row | shape | v2 split | pure-M ms | v2 ms | "
          "pred v2 ms | err | speedup | result |")
    print("|---|---|---|---:|---:|---:|---:|---:|---|")

    n_validate = 0
    n_partial = 0  # v2 < pure-M but cost-model error > 15%
    n_fail = 0     # v2 >= pure-M (regression)

    for label, M, N, K, v2_split, pred_v2 in ROWS:
        # Pure-M baseline
        pm_ms, pm_err = _compile_and_bench(M, N, K, (32, 1, 1), "identity")
        if pm_ms is None:
            print(f"| {label} | ({M},{N},{K}) | {v2_split} | "
                  f"ERR ({pm_err[:20]}) | — | {pred_v2:.2f} | — | — | SKIP |")
            continue

        # v2 pick
        v2_ms, v2_err = _compile_and_bench(M, N, K, v2_split, "k_fast")
        if v2_ms is None:
            print(f"| {label} | ({M},{N},{K}) | {v2_split} | "
                  f"{pm_ms:.2f} | ERR ({v2_err[:20]}) | {pred_v2:.2f} | "
                  f"— | — | SKIP |")
            continue

        rel_err = (pred_v2 - v2_ms) / v2_ms * 100
        speedup = pm_ms / v2_ms
        if v2_ms >= pm_ms:
            result = "FAIL (slower)"
            n_fail += 1
        elif abs(rel_err) <= 15:
            result = "VALIDATE"
            n_validate += 1
        else:
            result = "PARTIAL (faster but pred off)"
            n_partial += 1

        print(f"| {label} | ({M},{N},{K}) | {v2_split} | "
              f"{pm_ms:.2f} | {v2_ms:.2f} | {pred_v2:.2f} | "
              f"{rel_err:+.1f}% | {speedup:.2f}× | {result} |")

    print()
    print("## Summary\n")
    print(f"  rows considered: {len(ROWS)}")
    print(f"  VALIDATE  (v2 faster, pred ≤ ±15%): {n_validate}")
    print(f"  PARTIAL   (v2 faster, pred off):    {n_partial}")
    print(f"  FAIL      (v2 not faster):          {n_fail}")
    print()
    if n_validate > n_fail:
        print("Preliminary verdict: the v2 prototype's Tier 2 picks largely")
        print("validate. Production planner change is well-supported by HW data.")
    elif n_validate + n_partial > n_fail:
        print("Preliminary verdict: v2 is directionally right but cost-model")
        print("calibration needs work before production. Refine V4 first.")
    else:
        print("Preliminary verdict: v2 picks regress on hardware. Investigate")
        print("residuals before any planner change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
