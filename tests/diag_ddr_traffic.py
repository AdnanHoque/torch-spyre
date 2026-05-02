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

"""Phase 0 of the tile-ordering / scratchpad-locality project.

Hypothesis: matmul on Spyre is bandwidth-bound on LPDDR5 (~200 GB/s) vs
fp16 compute (>150 TFLOPs/s). The Inductor planner controls how cores are
distributed across (M, N, K) iteration dims — i.e., the (m_split, n_split,
k_split) tuple — which determines per-core access patterns and therefore
DDR-bytes-loaded across the kernel.

For C[M,N] = A[M,K] @ B[K,N] with (m, n, k) splits:
  A read traffic = n × |A|   (each N-band reads its M-slice of A)
  B read traffic = m × |B|   (each M-band reads its N-slice of B)
  C write traffic = k × |C|  (k partial outputs per element)

This script forces specific (m, n, k) tuples for a fixed shape, measures
kernel wall-time, and compares against theoretical DDR traffic. If the
correlation holds, tile-ordering at the planner level is a real lever
(probably 2-4× perf headroom on bandwidth-bound shapes). If it doesn't,
Spyre has hidden DDR-traffic mitigation we can't reach from the Inductor
planner.

Run:  python tests/diag_ddr_traffic.py
"""

from __future__ import annotations

import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

# Same four config knobs as the SplitK diagnostic.
import torch._inductor.config as _icfg

_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre  # noqa: F401
from torch_spyre import streams as _ts

from torch_spyre._inductor import core_division as _core_div


# ---- Force-split monkey-patch -----------------------------------------------

# The planner's final allocation step is `multi_dim_iteration_space_split`,
# which takes the iteration space + priority + min_splits and returns the
# concrete `{Symbol: int}` split dict. Forcing a specific split tuple is
# cleanest by replacing this function. For matmul, iteration_space dict
# iteration order is [M, N, K] (per Phase 0 SplitK confirmation), so we can
# apply the forced tuple positionally. Non-matmul ops fall through to the
# original.

_orig_multi = _core_div.multi_dim_iteration_space_split


def _force_split_factory(target: tuple[int, ...]):
    """Returns a multi_dim_iteration_space_split replacement that, for an
    iteration space matching len(target), returns target keyed by symbol
    iteration order. For other iteration spaces, falls through."""

    def _forced(it_space, max_cores, priorities, min_splits):
        syms_in_order = list(it_space.keys())
        if len(syms_in_order) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        # Validate factor product equals max_cores; otherwise fall through to
        # avoid silently mis-sizing.
        prod = 1
        for f in target:
            prod *= f
        if prod != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms_in_order)}

    return _forced


@contextmanager
def _force_split(target: tuple[int, ...]):
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)  # type: ignore[assignment]
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi  # type: ignore[assignment]


# ---- Theoretical DDR traffic ------------------------------------------------

@dataclass
class _TrafficModel:
    a_bytes: int
    b_bytes: int
    c_bytes: int
    total: int

    def fmt_mb(self) -> str:
        return (
            f"A={self.a_bytes / 1e6:.0f}MB B={self.b_bytes / 1e6:.0f}MB "
            f"C={self.c_bytes / 1e6:.0f}MB tot={self.total / 1e6:.0f}MB"
        )


def _traffic_model(M: int, N: int, K: int, m: int, n: int, k: int,
                   dtype_bytes: int = 2) -> _TrafficModel:
    """Theoretical DDR traffic for matmul under given (m, n, k) splits.
    Sees only weight loads + final-output writes; ignores cross-core
    reduction traffic for K-split which the dxp_standalone backend
    handles internally."""
    A = M * K * dtype_bytes
    B = K * N * dtype_bytes
    C = M * N * dtype_bytes
    return _TrafficModel(
        a_bytes=n * A,
        b_bytes=m * B,
        c_bytes=k * C,
        total=n * A + m * B + k * C,
    )


# ---- Bench loop -------------------------------------------------------------

WARMUP = 5
ITERS = 20


def _compile_mm():
    def _mm(a, b):
        return a @ b
    return torch.compile(_mm, dynamic=False)


def _run_one(M: int, N: int, K: int, target: tuple[int, int, int]):
    """Compile + run mm with forced (m, n, k) splits. Returns
    (median_ms, error_or_None)."""
    a = torch.randn(M, K, dtype=torch.float16).to("spyre")
    b = torch.randn(K, N, dtype=torch.float16).to("spyre")

    torch._dynamo.reset()
    err: str | None = None
    median_ms: float | None = None
    try:
        with _force_split(target):
            mm_fn = _compile_mm()
            for _ in range(WARMUP):
                mm_fn(a, b)
            _ts.synchronize()
            samples = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                mm_fn(a, b)
                _ts.synchronize()
                samples.append(time.perf_counter() - t0)
        median_ms = statistics.median(samples) * 1e3
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:120]}"
    return median_ms, err


# ---- Sweeps -----------------------------------------------------------------

# Each entry is (M, N, K, [(m, n, k), ...]) — shape + splits to force.
# All splits must satisfy m·n·k = 32 and stick-alignment. fp16 stick = 64
# elems, so:
#   M / m must be integer (M is non-stick)
#   N / n must be ≥ 64 elems and stick-aligned (N is stick)
#   K / k must be ≥ 64 elems and stick-aligned (K is stick on input A)
SHAPES = [
    # (2048, 4096, 8192) — large prefill matmul. M=2048 splits cleanly.
    # N=64 sticks → N-split factor must divide 64. K=128 sticks → divisor of 128.
    (2048, 4096, 8192, [
        (32, 1, 1),   # M-greedy (current planner default)
        (16, 2, 1),
        (8, 4, 1),
        (4, 8, 1),    # theoretical optimum for this shape
        (2, 16, 1),
        (1, 32, 1),   # N-greedy
        (1, 1, 32),   # K-greedy
    ]),
    # Llama-70B q_proj prefill. M=128 → m_split factors {1, 2, 4, 8, 16, 32}.
    # N=128 sticks, K=128 sticks.
    (128, 8192, 8192, [
        (32, 1, 1),   # default (heuristic-OFF would pick this)
        (16, 2, 1),
        (8, 4, 1),
        (4, 8, 1),
        (2, 16, 1),
        (1, 32, 1),
        (1, 1, 32),   # heuristic-ON = forceK
    ]),
    # Large-K balanced. M=N=1024, K=128 sticks.
    (1024, 1024, 16384, [
        (32, 1, 1),   # default
        (16, 2, 1),
        (8, 4, 1),
        (4, 8, 1),
        (2, 16, 1),
        (1, 16, 2),
        (1, 1, 32),
    ]),
]


@dataclass
class _Row:
    shape: tuple[int, int, int]
    target: tuple[int, int, int]
    traffic: _TrafficModel
    median_ms: float | None
    error: str | None


def _bench_shape(M: int, N: int, K: int, targets: list[tuple[int, int, int]]) -> list[_Row]:
    rows: list[_Row] = []
    for tgt in targets:
        m, n, k = tgt
        traffic = _traffic_model(M, N, K, m, n, k)
        median_ms, err = _run_one(M, N, K, tgt)
        rows.append(_Row(
            shape=(M, N, K), target=tgt, traffic=traffic,
            median_ms=median_ms, error=err,
        ))
        ms_s = "—" if median_ms is None else f"{median_ms:7.2f}"
        tag = "ERR" if err else "OK"
        # Also compute effective DDR bandwidth used per second of kernel time.
        if median_ms:
            bw_gbs = traffic.total / (median_ms * 1e-3) / 1e9
        else:
            bw_gbs = 0.0
        bw_s = f"{bw_gbs:5.1f}" if median_ms else "—"
        print(
            f"# {M:>5}×{N:>5}×{K:>5}  ({m:>2},{n:>2},{k:>2})  "
            f"traf={traffic.total/1e6:7.0f}MB  {ms_s} ms  bw={bw_s}GB/s  {tag}",
            flush=True,
        )
    return rows


def _print_table(all_rows: list[_Row], file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("# Tile-ordering DDR-traffic diagnostic — Phase 0")
    w("")
    w(f"PyTorch:        {torch.__version__}")
    w(f"torch_spyre:    {getattr(torch_spyre, '__version__', '(editable)')}")
    w(f"SENCORES:       {os.environ.get('SENCORES', '32 (default)')}")
    w(f"warmup iters:   {WARMUP}")
    w(f"measure iters:  {ITERS}")
    w("")
    w("**Theoretical DDR traffic** assumes each core independently reads its "
      "A-slice + B-slice + writes its (partial) C-slice, with no inter-core "
      "reuse. A_read = n·|A|, B_read = m·|B|, C_write = k·|C|.")
    w("")
    w("**Effective BW** = traffic / kernel_time. Higher means we're closer to "
      "saturating LPDDR5 (~200 GB/s peak). Lower means kernel-launch / sync / "
      "compute overhead dominates and the matmul isn't actually moving bytes "
      "at peak rate.")
    w("")

    # Group rows by shape for per-shape tables.
    by_shape: dict[tuple, list[_Row]] = {}
    for r in all_rows:
        by_shape.setdefault(r.shape, []).append(r)

    for shape, rows in by_shape.items():
        M, N, K = shape
        w(f"## (M={M}, N={N}, K={K})")
        w("")
        w("| split (m,n,k) | A_read | B_read | C_write | total traf | "
          "median ms | TFLOPs/s | eff BW GB/s | vs default |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        # Find the (32,1,1) default for normalization.
        default_ms = None
        for r in rows:
            if r.target == (32, 1, 1) and r.median_ms is not None:
                default_ms = r.median_ms
                break

        for r in rows:
            if r.error:
                w(f"| {r.target} | — | — | — | "
                  f"{r.traffic.total/1e6:.0f} | — | — | — | — |  err: {r.error}")
                continue
            ms = r.median_ms
            tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12 if ms else 0
            bw = r.traffic.total / (ms * 1e-3) / 1e9 if ms else 0
            speedup = (default_ms / ms) if (default_ms and ms) else 0
            w(
                f"| {r.target} | {r.traffic.a_bytes/1e6:.0f}MB | "
                f"{r.traffic.b_bytes/1e6:.0f}MB | "
                f"{r.traffic.c_bytes/1e6:.0f}MB | "
                f"**{r.traffic.total/1e6:.0f}MB** | "
                f"{ms:.2f} | {tflops:.2f} | {bw:.1f} | {speedup:.2f}× |"
            )
        w("")


def main() -> int:
    all_rows: list[_Row] = []
    for M, N, K, targets in SHAPES:
        print(f"\n# starting shape ({M}, {N}, {K})", flush=True)
        all_rows.extend(_bench_shape(M, N, K, targets))

    _print_table(all_rows)

    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_ddr_traffic_results.md",
    )
    with open(results_path, "w") as f:
        _print_table(all_rows, file=f)
    print(f"\n# results written to {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
