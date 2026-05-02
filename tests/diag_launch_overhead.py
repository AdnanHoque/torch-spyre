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

"""Phase 0b of the flash-attention-on-Spyre project.

Characterizes per-launch overhead on Spyre across kernel-work sizes
relevant to flash-attention tile choices. Method: at a fixed mm shape,
issue N back-to-back identical mm calls in a Python loop and measure
total wall time. Per-call cost = total / N; the asymptotic per-call cost
when N grows large reveals the per-launch overhead floor.

Three shapes representing different fractions of "real work":

1. **Tiny work** — `(M=1, N=512, K=128)`: 130K FLOPs per call, near zero
   compute. Dominated entirely by launch + sync.
2. **Small flash-attention tile** — `(M=64, N=128, K=128)`: 2M FLOPs.
   Roughly the work of one (Q_tile=64) × (KV_tile=128) × D=128 step.
3. **Larger flash-attention tile** — `(M=2048, N=2048, K=128)`: 1G FLOPs.
   Roughly per-Q-tile work for Q_tile=2048 with cores splitting along M.

For each shape we measure `T(N)` for `N ∈ {1, 2, 4, 8, 16, 32, 64}` and
report:

- Per-call wall-time at each N
- Asymptotic per-call cost (large N)
- Ratio T(64) / 64·T(1) — should approach 1.0 if launch overhead amortizes,
  but stays ≪ 1 if there's a fixed first-call cost being averaged out

This gives Phase 1 a concrete answer to "what's the smallest tile that
isn't crippled by launch overhead?".

Run: python tests/diag_launch_overhead.py
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


# ---- bench primitive --------------------------------------------------------

WARMUP = 5
ITERS = 30


def _bench(fn) -> float:
    """Median ms per call to fn(), with per-iter device sync."""
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


# ---- shapes -----------------------------------------------------------------

@dataclass
class _Shape:
    label: str
    M: int
    N: int
    K: int

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N * self.K

    @property
    def output_bytes(self) -> int:
        return self.M * self.N * 2  # fp16


SHAPES = [
    _Shape("tiny work", M=1, N=512, K=128),               # ~130K FLOPs
    _Shape("flash-attention small tile", M=64, N=128, K=128),  # ~2M FLOPs
    _Shape("flash-attention larger tile", M=2048, N=2048, K=128),  # ~1G FLOPs
]

N_VALUES = [1, 2, 4, 8, 16, 32, 64]


# ---- bench loop -------------------------------------------------------------

def _make_call(shape: _Shape):
    """Returns a callable that issues N matmuls back-to-back, all using
    distinct weight tensors so caching can't merge them."""
    a = torch.randn(shape.M, shape.K, dtype=torch.float16, device="spyre")
    # Pre-allocate N distinct weight tensors so each call uses a different B.
    # (Identical-shape weights still hit the same compiled artifact.)
    Bs = [
        torch.randn(shape.K, shape.N, dtype=torch.float16, device="spyre")
        for _ in range(max(N_VALUES))
    ]

    @torch.compile(dynamic=False)
    def _mm(x, y):
        return x @ y

    # Warmup the compile once — first call here triggers compile, subsequent
    # calls hit the artifact cache.
    _mm(a, Bs[0])
    _ts.synchronize()

    def make_loop(n: int):
        def loop():
            for i in range(n):
                _mm(a, Bs[i])
        return loop

    return {n: make_loop(n) for n in N_VALUES}


@dataclass
class _Row:
    shape_label: str
    M: int
    N: int
    K: int
    n_calls: int
    total_ms: float
    per_call_ms: float


def main() -> int:
    rows: list[_Row] = []

    for shape in SHAPES:
        print(f"\n# shape: {shape.label} ({shape.M}x{shape.N}x{shape.K}, "
              f"{shape.flops/1e6:.2f}M FLOPs/call)", flush=True)
        torch._dynamo.reset()
        loops = _make_call(shape)

        for n in N_VALUES:
            ms = _bench(loops[n])
            per_call = ms / n
            rows.append(_Row(
                shape_label=shape.label, M=shape.M, N=shape.N, K=shape.K,
                n_calls=n, total_ms=ms, per_call_ms=per_call,
            ))
            print(f"  N={n:>2}: total {ms:7.2f} ms, per call {per_call:6.3f} ms",
                  flush=True)

    _print_table(rows)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_launch_overhead_results.md",
    )
    with open(out_path, "w") as f:
        _print_table(rows, file=f)
    print(f"\n# results written to {out_path}", flush=True)
    return 0


def _print_table(rows: list[_Row], file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("# Per-launch overhead diagnostic — flash-attention Phase 0b")
    w("")
    w(f"PyTorch:        {torch.__version__}")
    w(f"torch_spyre:    {getattr(torch_spyre, '__version__', '(editable)')}")
    w(f"warmup iters:   {WARMUP}")
    w(f"measure iters:  {ITERS}")
    w(f"per-iter sync:  torch_spyre.streams.synchronize() inside the timed loop")
    w("")
    w("**Method**: at a fixed mm shape, issue N back-to-back compiled mm "
      "calls. Per-call wall time = total / N. The asymptotic value as N "
      "grows large is the per-launch overhead floor for this shape. The "
      "ratio T(N)/T(1) tells us whether per-call overhead has plateaued.")
    w("")

    by_shape: dict[str, list[_Row]] = {}
    for r in rows:
        by_shape.setdefault(r.shape_label, []).append(r)

    for label, group in by_shape.items():
        s = group[0]
        flops_per_call = 2 * s.M * s.N * s.K
        w(f"## {label} — `(M={s.M}, N={s.N}, K={s.K})` "
          f"= {flops_per_call/1e6:.2f}M FLOPs/call")
        w("")
        w("| N | total ms | per call ms | per call vs N=1 |")
        w("|---:|---:|---:|---:|")
        per_call_n1 = next((r.per_call_ms for r in group if r.n_calls == 1), None)
        for r in group:
            ratio = (r.per_call_ms / per_call_n1) if per_call_n1 else float("nan")
            w(f"| {r.n_calls} | {r.total_ms:.2f} | {r.per_call_ms:.3f} | "
              f"{ratio:.2f}× |")
        w("")


if __name__ == "__main__":
    raise SystemExit(main())
