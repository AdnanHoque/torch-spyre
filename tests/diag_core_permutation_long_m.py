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

"""Long-M generalization probe for block_cyclic permutation.

The broad sweep found a monotonic trend on L3-70B q_proj prefill:
  M=256:  1.015x      M=512:  1.010x
  M=1024: 1.016x      M=2048: 1.024x

Hypothesis: this is output-writeback DRAM banking. If true:
  - the trend should keep growing at M=4096 on L3-70B q_proj
  - it should reproduce on OTHER L3-70B matmul ops at long M
  - it should reproduce on L3-8B (smaller hidden dim) at long M
  - the magnitude should scale with output size M*N (or just M for
    fixed-N comparisons)

If the trend is shape-specific (only L3-70B q_proj's exact dims
matter), it's not a general lever and we should treat the win as a
local pocket.

Design:
  - identity vs block_cyclic on each shape (the candidate)
  - stride2 on K-split shapes (existing winner, for cross-check)
  - bit_reverse on a sample (downside sanity)
  - two trial orders so a 2% finding doesn't need a separate replicator
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
from torch_spyre._inductor import config as ts_config  # noqa: E402
from torch_spyre._inductor import core_division as _core_div  # noqa: E402


WARMUP = 3
ITERS = 15
DTYPE = torch.float16

# (regime, label, M, N, K, split, perms_to_test)
TARGETS = [
    # === Extend the L3-70B q_proj trend ===
    ("trend",  "L3-70B q_proj M=4096", 4096, 8192, 8192, (1, 32, 1),
     ["identity", "block_cyclic", "bit_reverse"]),

    # === Other L3-70B matmuls at long M ===
    # kv_proj has N=1024 = 16 sticks; (1, 32, 1) is invalid. Use (1, 16, 2).
    ("l3-70B", "L3-70B kv_proj M=2048", 2048, 1024, 8192, (1, 16, 2),
     ["identity", "block_cyclic"]),

    ("l3-70B", "L3-70B o_proj M=2048", 2048, 8192, 8192, (1, 32, 1),
     ["identity", "block_cyclic"]),
    ("l3-70B", "L3-70B o_proj M=4096", 4096, 8192, 8192, (1, 32, 1),
     ["identity", "block_cyclic"]),
    ("l3-70B", "L3-70B mlp_gate M=2048", 2048, 28672, 8192, (1, 32, 1),
     ["identity", "block_cyclic"]),
    ("l3-70B", "L3-70B mlp_down M=2048", 2048, 8192, 28672, (1, 32, 1),
     ["identity", "block_cyclic"]),
    ("l3-70B", "L3-70B mlp_down M=4096", 4096, 8192, 28672, (1, 32, 1),
     ["identity", "block_cyclic"]),

    # === L3-8B at long M ===
    ("l3-8B", "L3-8B q_proj M=2048", 2048, 4096, 4096, (1, 32, 1),
     ["identity", "block_cyclic"]),
    ("l3-8B", "L3-8B q_proj M=4096", 4096, 4096, 4096, (1, 32, 1),
     ["identity", "block_cyclic"]),
    ("l3-8B", "L3-8B mlp_down M=2048", 2048, 4096, 14336, (1, 32, 1),
     ["identity", "block_cyclic"]),
    ("l3-8B", "L3-8B mlp_down M=4096", 4096, 4096, 14336, (1, 32, 1),
     ["identity", "block_cyclic"]),

    # === Sanity: K-split with stride2 (known winner — should still hold) ===
    ("kmix",   "L3-70B q_proj K-split (4,1,8)", 128, 8192, 8192, (4, 1, 8),
     ["identity", "stride2"]),
]


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


def _compile_and_bench(M, N, K, target, perm):
    ts_config.core_id_permutation = perm
    ts_config.core_emission_reverse = False
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


def main() -> int:
    print("# Long-M generalization probe — does block_cyclic help broadly?\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")
    print(f"# Each (shape, perm) runs in BOTH trial orders. A real win\n"
          f"# requires (a) consistent direction across orders, (b) mean ≥2%.\n")

    results = {}  # (label, perm, trial) -> ms or None
    for regime, label, M, N, K, target, perms in TARGETS:
        print(f"### [{regime}] {label}  M={M} N={N} K={K}  split={target}")
        for tname, ordered in (("trial1", perms),
                                ("trial2", list(reversed(perms)))):
            print(f"  {tname} (order: {ordered}):")
            for perm in ordered:
                ms, err = _compile_and_bench(M, N, K, target, perm)
                if err:
                    print(f"    {perm:14s}: ERR {err[:80]}")
                    results[(label, perm, tname)] = None
                else:
                    print(f"    {perm:14s}: {ms:.3f} ms")
                    results[(label, perm, tname)] = ms
            print()

    # --- summary table grouped by regime ---
    print("\n## Summary — block_cyclic vs identity (mean of two trial orders)\n")
    print("| regime | shape | split | identity | block_cyclic | mean sp | "
          "consistent? |")
    print("|---|---|---|---:|---:|---:|---|")
    for regime, label, _M, _N, _K, target, perms in TARGETS:
        if "block_cyclic" not in perms:
            continue
        id1 = results.get((label, "identity", "trial1"))
        id2 = results.get((label, "identity", "trial2"))
        bc1 = results.get((label, "block_cyclic", "trial1"))
        bc2 = results.get((label, "block_cyclic", "trial2"))
        if None in (id1, id2, bc1, bc2):
            continue
        sp1 = id1 / bc1
        sp2 = id2 / bc2
        mean = (sp1 + sp2) / 2
        cons = "✓" if (sp1 - 1) * (sp2 - 1) > 0 else "~ flipped"
        ident_med = (id1 + id2) / 2
        bc_med = (bc1 + bc2) / 2
        print(f"| {regime} | {label} | {target} | "
              f"{ident_med:.3f} | {bc_med:.3f} | {mean:.3f}x | {cons} |")
    print()

    # --- confirmed wins ---
    print("## Confirmed wins (≥2% mean AND consistent direction)\n")
    confirmed = []
    for regime, label, _M, _N, _K, target, perms in TARGETS:
        for perm in perms:
            if perm == "identity":
                continue
            id1 = results.get((label, "identity", "trial1"))
            id2 = results.get((label, "identity", "trial2"))
            t1 = results.get((label, perm, "trial1"))
            t2 = results.get((label, perm, "trial2"))
            if None in (id1, id2, t1, t2):
                continue
            sp1, sp2 = id1 / t1, id2 / t2
            mean = (sp1 + sp2) / 2
            if (sp1 - 1) * (sp2 - 1) > 0 and mean >= 1.02:
                confirmed.append((regime, label, perm, mean))
    if confirmed:
        for c in confirmed:
            print(f"  [{c[0]}] {c[1]}  perm={c[2]}  mean speedup {c[3]:.3f}x")
    else:
        print("  None.")
    print()

    # --- output-size scaling check ---
    print("## Output-size scaling check\n")
    print(
        "Hypothesis: block_cyclic speedup correlates with output bytes\n"
        "(M·N·dtype). If true, a scatter plot of (output_bytes, speedup)\n"
        "should show monotonic positive correlation.\n"
    )
    print("| shape | M*N (KB output) | mean speedup |")
    print("|---|---:|---:|")
    pts = []
    for regime, label, M, N, _K, target, perms in TARGETS:
        if "block_cyclic" not in perms:
            continue
        id1 = results.get((label, "identity", "trial1"))
        id2 = results.get((label, "identity", "trial2"))
        bc1 = results.get((label, "block_cyclic", "trial1"))
        bc2 = results.get((label, "block_cyclic", "trial2"))
        if None in (id1, id2, bc1, bc2):
            continue
        sp1 = id1 / bc1
        sp2 = id2 / bc2
        mean = (sp1 + sp2) / 2
        out_kb = (M * N * 2) // 1024
        pts.append((label, out_kb, mean))
        print(f"| {label} | {out_kb} | {mean:.3f}x |")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
