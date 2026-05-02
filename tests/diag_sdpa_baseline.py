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

"""Phase 0a of the flash-attention-on-Spyre project.

Measures the cost of the current SDPA path on Spyre at prefill-relevant
sequence lengths. The current decomposition (decompositions.py:494-584)
materializes a (B, H, S, S) intermediate score tensor and runs as a
sequence of three matmul+pointwise kernels with that tensor flowing
through DDR between them.

For Llama-3-style (H=32 heads, kv_heads=8, head_dim=128) at fp16 the
score tensor is 64*S^2 bytes:

  S = 512   ->   16 MB intermediate
  S = 1024  ->   64 MB
  S = 2048  ->  256 MB  <- per-core memory span limit
  S = 4096  ->    1 GB   <- BW disaster
  S = 8192  ->    4 GB

This bench captures three things per S:

1. End-to-end SDPA wall time — the cost we want to beat
2. Number of Spyre kernels emitted — proxy for launch-overhead
3. Theoretical DDR traffic floor — Q+K+V + score + output bytes

The flash-attention upper bound is total_compute / peak_compute (often
much smaller than the bandwidth-limited current path).

Run: python tests/diag_sdpa_baseline.py
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
from torch_spyre._inductor.codegen import superdsc as _superdsc


# ---- SDSC kernel count capture ----------------------------------------------

_kernel_count = 0
_kernel_ops: list[str] = []
_orig_parse_op_spec = _superdsc.parse_op_spec


def _hook_parse(op_spec):
    global _kernel_count
    _kernel_count += 1
    _kernel_ops.append(op_spec.op)
    return _orig_parse_op_spec(op_spec)


_superdsc.parse_op_spec = _hook_parse  # type: ignore[assignment]


# ---- bench loop -------------------------------------------------------------

WARMUP = 3
ITERS = 20


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


def _theoretical_traffic(B: int, H: int, Hkv: int, S: int, D: int) -> dict[str, int]:
    """Bytes moved through DDR for naive SDPA at fp16."""
    fp16 = 2
    Q = B * H * S * D * fp16
    K = B * Hkv * S * D * fp16
    V = B * Hkv * S * D * fp16
    score = B * H * S * S * fp16   # the materialized intermediate
    out = B * H * S * D * fp16
    # Naive path streams: Q,K to QK matmul; score in DDR; score read for softmax;
    # softmax output to DDR; softmax + V to second matmul; out to DDR.
    # Approximate total weight bytes to DDR assuming no fusion:
    total = Q + K + V + 3 * score + out
    return {
        "Q": Q, "K": K, "V": V,
        "score": score, "out": out, "total_to_DDR": total,
    }


def _theoretical_compute(B: int, H: int, S: int, D: int) -> int:
    """FLOPs for SDPA: 2*B*H*S*S*D for QK matmul + 2*B*H*S*S*D for AV matmul."""
    return 2 * 2 * B * H * S * S * D


# ---- model ----------------------------------------------------------------

# Llama-3-8B style: H=32 heads, kv_heads=8 (GQA), head_dim=128, fp16.
B = 1
H = 32
H_KV = 8
D = 128
DTYPE = torch.float16

S_VALUES = [512, 1024, 2048, 4096]


def _make_qkv(S: int):
    """Q: (B, H, S, D); K, V: (B, Hkv, S, D)."""
    q = torch.randn(B, H, S, D, dtype=DTYPE, device="spyre")
    k = torch.randn(B, H_KV, S, D, dtype=DTYPE, device="spyre")
    v = torch.randn(B, H_KV, S, D, dtype=DTYPE, device="spyre")
    return q, k, v


def _sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=(H_KV != H)
    )


# ---- main -----------------------------------------------------------------

@dataclass
class _Row:
    S: int
    median_ms: float
    n_kernels: int
    kernel_ops: list[str]
    traffic: dict[str, int]
    flops: int


def main() -> int:
    rows: list[_Row] = []

    for S in S_VALUES:
        print(f"\n# S = {S}", flush=True)
        global _kernel_count, _kernel_ops
        _kernel_count = 0
        _kernel_ops = []

        try:
            q, k, v = _make_qkv(S)

            torch._dynamo.reset()
            sdpa = torch.compile(_sdpa, dynamic=False)

            # Warmup also triggers compile, which triggers parse_op_spec calls.
            # Reset counters AFTER compile so we measure the steady-state
            # number of kernels per call (== same as compile-time emission).
            sdpa(q, k, v)
            _ts.synchronize()
            seen_kernels = _kernel_count
            seen_ops = list(_kernel_ops)

            wall_ms = _bench(lambda: sdpa(q, k, v))

            traffic = _theoretical_traffic(B, H, H_KV, S, D)
            flops = _theoretical_compute(B, H, S, D)

            rows.append(_Row(
                S=S, median_ms=wall_ms,
                n_kernels=seen_kernels, kernel_ops=seen_ops,
                traffic=traffic, flops=flops,
            ))

            # Per-iteration BW + TFLOPs/s
            bw_gbs = traffic["total_to_DDR"] / (wall_ms * 1e-3) / 1e9
            tflops = flops / (wall_ms * 1e-3) / 1e12
            print(
                f"  wall: {wall_ms:.2f} ms  "
                f"kernels: {seen_kernels}  "
                f"score (S²) intermediate: {traffic['score']/1e6:.0f} MB  "
                f"total DDR: {traffic['total_to_DDR']/1e6:.0f} MB  "
                f"eff BW: {bw_gbs:.1f} GB/s  "
                f"TFLOPs/s: {tflops:.2f}",
                flush=True,
            )
            print(f"  kernel ops: {seen_ops}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ERR: {type(e).__name__}: {str(e)[:120]}", flush=True)

    _print_table(rows)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_sdpa_baseline_results.md",
    )
    with open(out_path, "w") as f:
        _print_table(rows, file=f)
    print(f"\n# results written to {out_path}", flush=True)
    return 0


def _print_table(rows: list[_Row], file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("# SDPA baseline diagnostic — flash-attention Phase 0a")
    w("")
    w(f"PyTorch:        {torch.__version__}")
    w(f"torch_spyre:    {getattr(torch_spyre, '__version__', '(editable)')}")
    w(f"Model shape:    B={B}, H={H}, H_kv={H_KV}, D={D} (Llama-3-8B GQA)")
    w(f"warmup iters:   {WARMUP}")
    w(f"measure iters:  {ITERS}")
    w(f"is_causal:      True")
    w(f"per-iter sync:  torch_spyre.streams.synchronize() inside the timed loop")
    w("")
    w("**Naive SDPA path** (decompositions.py:494): scale Q+K, QK matmul, "
      "softmax, AV matmul, output transpose. The (B,H,S,S) score intermediate "
      "flows through DDR three times (write, read for softmax, write softmax, "
      "read for AV).")
    w("")
    w("**Flash-attention target**: tile Q-rows; per Q-tile, loop over KV-tiles "
      "with running max + running sum + running output pinned in scratchpad. "
      "Score tensor never materialized in DDR. Per-Q-tile latency dominated "
      "by Q+KV streaming, not S² intermediate.")
    w("")
    w("| S | wall ms | kernels | score (S²) | total DDR | eff BW GB/s | TFLOPs/s | flops/byte AI |")
    w("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        score_mb = r.traffic["score"] / 1e6
        total_mb = r.traffic["total_to_DDR"] / 1e6
        bw = r.traffic["total_to_DDR"] / (r.median_ms * 1e-3) / 1e9
        tflops = r.flops / (r.median_ms * 1e-3) / 1e12
        ai = r.flops / r.traffic["total_to_DDR"]
        w(
            f"| {r.S} | {r.median_ms:.2f} | {r.n_kernels} | "
            f"{score_mb:.0f} MB | {total_mb:.0f} MB | "
            f"{bw:.1f} | {tflops:.2f} | {ai:.1f} |"
        )
    w("")
    w("**Arithmetic intensity** (flops/byte) is the key signal: low AI -> "
      "bandwidth-bound, flash attention helps a lot. AI converges to a "
      "constant for the naive path because total DDR scales like S² (matching "
      "compute scaling), so naive is bandwidth-bound at all S.")


if __name__ == "__main__":
    raise SystemExit(main())
