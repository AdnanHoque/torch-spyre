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

"""Phase 0c of the flash-attention-on-Spyre project.

Confirms whether `LX_PLANNING=1` actually pins allowlisted op outputs in
LX scratchpad (skipping DDR roundtrips for downstream consumers). The
flash-attention design relies on this — running max + sum + output kept
in scratchpad across consecutive KV-tile kernels — so we need to verify
the mechanism works as documented before committing to the design.

`OP_OUTPUT_GOOD_FOR_LX_REUSE = ["max", "sum", "clone"]` per
`scratchpad.py:30`. Ops whose names contain these substrings are
candidates for LX pinning when the consumer has matching core_division.

Two tests:

1. **Softmax chain** — `max → sub → exp → sum → realdiv`. Has TWO
   allowlisted producers (max, sum) in the chain. With LX_PLANNING=1,
   the `sub` and `realdiv` should read their respective producers from
   scratchpad instead of DDR.
2. **Pure-mm chain** — `(x @ W1) @ W2`. Has NO allowlisted producers.
   The intermediate goes through DDR regardless of LX_PLANNING. Acts as
   the negative control — LX off vs on should give the same time.

For each, run with `config.lx_planning` set to False and True (with
`torch._dynamo.reset()` between to force recompile). Compare wall time.

Run: python tests/diag_lx_planning.py
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

import torch

import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre  # noqa: F401
from torch_spyre import streams as _ts
from torch_spyre._inductor import config as _ts_cfg


# ---- bench primitive --------------------------------------------------------

WARMUP = 5
ITERS = 30


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


# ---- chains under test -----------------------------------------------------

def _softmax_chain(x: torch.Tensor) -> torch.Tensor:
    """The softmax decomposition pattern from the SDPA path:
    max -> sub -> exp -> sum -> realdiv.

    `max` and `sum` are allowlisted; with LX_PLANNING=1 their outputs
    should be pinned in scratchpad and read from there by `sub` /
    `realdiv` respectively.
    """
    m = torch.amax(x, dim=-1, keepdim=True)
    sub = x - m
    e = torch.exp(sub)
    s = torch.sum(e, dim=-1, keepdim=True)
    return e / s


def _mm_chain(x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    """Negative control: two chained matmuls. No allowlisted producers
    in the chain — LX_PLANNING shouldn't change wall time."""
    h = x @ W1
    return h @ W2


# ---- bench harness ----------------------------------------------------------

@dataclass
class _Row:
    chain: str
    lx_planning: bool
    median_ms: float


def _run_chain(name: str, fn, *inputs) -> dict[bool, float]:
    """Run fn with both LX_PLANNING=False and =True. Returns {lx_on: ms}."""
    out: dict[bool, float] = {}
    for lx in (False, True):
        _ts_cfg.lx_planning = lx
        torch._dynamo.reset()
        compiled = torch.compile(fn, dynamic=False)
        # Warmup forces compile under the current lx_planning setting.
        compiled(*inputs)
        _ts.synchronize()

        ms = _bench(lambda: compiled(*inputs))
        out[lx] = ms
        print(f"  {name}  LX_PLANNING={'on' if lx else 'off'}: {ms:.2f} ms",
              flush=True)
    return out


# ---- shapes -----------------------------------------------------------------

# Score-tensor-shaped input for the softmax chain. (B, H, S, S) for
# attention. Pick S small enough to fit in scratchpad after stick split.
# fp16 stick=64 elems, per-core scratchpad ~2MB.
B, H, S = 1, 32, 64
SOFTMAX_SHAPE = (B, H, S, S)

# Two square matrices for the mm chain.
M, N, K = 128, 4096, 4096
MM_M, MM_N, MM_K = M, N, K  # x @ W1: (M, K) @ (K, N) -> (M, N); W2: (N, M) for chain


def main() -> int:
    rows: list[_Row] = []

    print(f"# softmax chain — input shape {SOFTMAX_SHAPE} fp16", flush=True)
    x_softmax = torch.randn(*SOFTMAX_SHAPE, dtype=torch.float16, device="spyre")
    softmax_results = _run_chain("softmax", _softmax_chain, x_softmax)
    rows.append(_Row("softmax", False, softmax_results[False]))
    rows.append(_Row("softmax", True, softmax_results[True]))

    print(f"\n# mm chain — input shape ({M}, {K}) fp16", flush=True)
    x_mm = torch.randn(M, K, dtype=torch.float16, device="spyre")
    W1 = torch.randn(K, N, dtype=torch.float16, device="spyre")
    W2 = torch.randn(N, M, dtype=torch.float16, device="spyre")
    mm_results = _run_chain("mm chain", _mm_chain, x_mm, W1, W2)
    rows.append(_Row("mm chain", False, mm_results[False]))
    rows.append(_Row("mm chain", True, mm_results[True]))

    # Reset to off before exiting so we don't leak state.
    _ts_cfg.lx_planning = False

    _print_table(rows)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_lx_planning_results.md",
    )
    with open(out_path, "w") as f:
        _print_table(rows, file=f)
    print(f"\n# results written to {out_path}", flush=True)
    return 0


def _print_table(rows: list[_Row], file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("# LX scratchpad pinning diagnostic — flash-attention Phase 0c")
    w("")
    w(f"PyTorch:        {torch.__version__}")
    w(f"torch_spyre:    {getattr(torch_spyre, '__version__', '(editable)')}")
    w(f"warmup iters:   {WARMUP}")
    w(f"measure iters:  {ITERS}")
    w("")
    w("**Hypothesis**: with `LX_PLANNING=1`, allowlisted op outputs "
      "(`max`, `sum`, `clone`) are pinned in LX scratchpad so downstream "
      "consumers read from scratchpad instead of DDR. The softmax chain "
      "has two allowlisted producers; the mm chain has none.")
    w("")
    w("**Expected**:")
    w("- Softmax: LX_PLANNING=on faster than off (DDR roundtrips for "
      "max+sum outputs eliminated)")
    w("- mm chain: LX_PLANNING=on roughly equal to off (negative control)")
    w("")

    by_chain: dict[str, list[_Row]] = {}
    for r in rows:
        by_chain.setdefault(r.chain, []).append(r)

    w("| chain | LX off ms | LX on ms | speedup | verdict |")
    w("|---|---:|---:|---:|---|")
    for chain, group in by_chain.items():
        off = next((r.median_ms for r in group if not r.lx_planning), None)
        on = next((r.median_ms for r in group if r.lx_planning), None)
        if off is None or on is None:
            continue
        speedup = off / on if on > 0 else float("inf")
        if speedup > 1.05:
            verdict = "**LX helps** (DDR avoided)"
        elif speedup < 0.95:
            verdict = "LX *hurts* (regressed)"
        else:
            verdict = "tied (LX has no effect at this shape)"
        w(f"| {chain} | {off:.2f} | {on:.2f} | {speedup:.2f}× | {verdict} |")
    w("")
    w("**Interpretation**: a >1.05× speedup on softmax indicates LX "
      "pinning is actually working. The mm chain should stay near 1.00× "
      "as a control.")


if __name__ == "__main__":
    raise SystemExit(main())
