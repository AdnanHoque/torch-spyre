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

"""FA tiling Phase 0 prototype: validate the algorithm + AIU compile path.

The current spyre__sdpa_overrideable in decompositions.py:494
materializes the full M×M attention matrix, which is 64×2048×2048×2
= 512MB at Llama 70B M=2048. Round-trips that through HMI.

This prototype implements FA-2 with K-tile streaming. For each K-tile,
it computes a partial QK^T, applies online softmax (running max +
sumexp), and accumulates P·V into O. Intermediate S_tile is M×K_tile,
not M×M — stays in LX, no HMI round-trip.

Phase 0 questions:
  1. Does the algorithm produce correct results vs reference attention?
  2. Does torch.compile handle the manually-unrolled K-tile loop on
     AIU without choking?
  3. Is the FA-tiled wall actually faster than the current SDPA path?

If yes to all three, the path forward is to integrate FA tiling into
decompositions.py:494. If no, we learn what's blocking and adjust.

Usage:
    python tests/diag_fa_tiling_phase0.py            # CPU correctness only
    python tests/diag_fa_tiling_phase0.py --aiu       # AIU bench
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import torch


def reference_attn(q, k, v, scale=None):
    """Reference attention: bmm + softmax + bmm. Materializes M×M."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


def fa2_tiled(q, k, v, k_tile_size=128, scale=None):
    """FA-2 algorithm with K-tile streaming, manually unrolled.

    q, k, v: shape (..., M, head_dim).
    Tiles K and V outside the compiled scope (Inductor on AIU can't
    handle dynamic-offset slicing inside the compile graph). The per-
    tile update step IS pure tensor ops.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    M_kv = k.shape[-2]
    n_tiles = (M_kv + k_tile_size - 1) // k_tile_size
    # Pre-split K and V into a list of tile tensors. This happens outside
    # any torch.compile scope (the caller compiles each fa_step instead).
    k_tiles = [k[..., t * k_tile_size:(t + 1) * k_tile_size, :]
               for t in range(n_tiles)]
    v_tiles = [v[..., t * k_tile_size:(t + 1) * k_tile_size, :]
               for t in range(n_tiles)]

    # Running state — initialize per-row max and sumexp
    m_state = torch.full(q.shape[:-1] + (1,), float("-inf"),
                         dtype=torch.float32, device=q.device)
    l_state = torch.zeros(q.shape[:-1] + (1,),
                          dtype=torch.float32, device=q.device)
    o_state = torch.zeros_like(q, dtype=torch.float32)

    for k_tile, v_tile in zip(k_tiles, v_tiles):
        # S_tile = Q · K_tile^T, shape (..., M, k_tile_actual)
        s_tile = torch.matmul(q, k_tile.transpose(-2, -1)) * scale
        s_tile = s_tile.to(torch.float32)

        # Online softmax: update running max
        m_tile = s_tile.amax(dim=-1, keepdim=True)
        m_new = torch.maximum(m_state, m_tile)

        # Probabilities for this tile
        p_tile = torch.exp(s_tile - m_new)

        # Rescale factor for previous output
        rescale = torch.exp(m_state - m_new)

        # Accumulate: O = O * rescale + P_tile · V_tile
        o_state = o_state * rescale + torch.matmul(p_tile, v_tile.to(torch.float32))

        # Accumulate sumexp
        l_tile = p_tile.sum(dim=-1, keepdim=True)
        l_state = l_state * rescale + l_tile

        m_state = m_new

    # Final normalization
    return (o_state / l_state).to(q.dtype)


def correctness_test():
    """Verify FA-2 matches reference attention on a small shape."""
    print("## Correctness test (CPU)\n")
    torch.manual_seed(0)
    # Small shape for fast CPU test
    batch, n_heads, M, head_dim = 1, 2, 256, 64
    q = torch.randn(batch, n_heads, M, head_dim, dtype=torch.float16)
    k = torch.randn(batch, n_heads, M, head_dim, dtype=torch.float16)
    v = torch.randn(batch, n_heads, M, head_dim, dtype=torch.float16)

    ref = reference_attn(q, k, v)
    fa = fa2_tiled(q, k, v, k_tile_size=64)

    max_abs = (ref - fa).abs().max().item()
    # Use cosine similarity for rel — more stable for fp16 attention
    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().float(), fa.flatten().float(), dim=0
    ).item()
    print(f"  shape: {q.shape}")
    print(f"  max abs diff: {max_abs:.6f}")
    print(f"  cosine similarity: {cos:.6f}")
    if max_abs < 0.01 and cos > 0.999:
        print(f"  ✓ PASS (correctness within fp16 tolerance)")
        return True
    else:
        print(f"  ✗ FAIL")
        return False


def aiu_bench(M, n_heads, head_dim, k_tile_size):
    """Benchmark reference vs FA-tiled on AIU."""
    import torch._inductor.config as _icfg
    _icfg.compile_threads = 1
    _icfg.worker_start_method = "fork"
    _icfg.fx_graph_cache = False

    import torch_spyre
    torch_spyre._autoload()
    from torch_spyre import streams as _ts

    WARMUP = 3
    ITERS = 8

    q = torch.randn(1, n_heads, M, head_dim, dtype=torch.float16, device="spyre")
    k = torch.randn(1, n_heads, M, head_dim, dtype=torch.float16, device="spyre")
    v = torch.randn(1, n_heads, M, head_dim, dtype=torch.float16, device="spyre")

    print(f"\n## AIU bench (n_heads={n_heads}, M={M}, head_dim={head_dim}, "
          f"k_tile={k_tile_size})\n")

    # Reference path
    torch._dynamo.reset()
    @torch.compile(dynamic=False)
    def ref_compiled(q, k, v):
        return reference_attn(q, k, v)

    try:
        for _ in range(WARMUP): ref_compiled(q, k, v)
        _ts.synchronize()
        samples = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            ref_compiled(q, k, v)
            _ts.synchronize()
            samples.append(time.perf_counter() - t0)
        ref_ms = statistics.median(samples) * 1e3
        print(f"  reference (full M×M):   {ref_ms:.3f} ms")
    except Exception as e:
        print(f"  reference: ERR {type(e).__name__}: {str(e)[:80]}")
        ref_ms = None

    # FA-tiled path: compile only the per-tile step. Slicing happens in
    # eager Python. Each step takes pre-sliced tile tensors.
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def fa_step(q, k_tile, v_tile, m_state, l_state, o_state, scale):
        """One FA-2 tile update — all fp16. Uses torch.where instead of
        torch.maximum (latter is UnimplementedOp on Spyre backend)."""
        s_tile = torch.matmul(q, k_tile.transpose(-2, -1)) * scale
        m_tile = s_tile.amax(dim=-1, keepdim=True)
        m_new = torch.where(m_tile > m_state, m_tile, m_state)
        p_tile = torch.exp(s_tile - m_new)
        rescale = torch.exp(m_state - m_new)
        o_new = o_state * rescale + torch.matmul(p_tile, v_tile)
        l_new = l_state * rescale + p_tile.sum(dim=-1, keepdim=True)
        return m_new, l_new, o_new

    def fa_outer(q, k, v):
        scale = torch.tensor(1.0 / math.sqrt(q.shape[-1]), dtype=q.dtype,
                             device=q.device)
        M_kv = k.shape[-2]
        n_tiles = (M_kv + k_tile_size - 1) // k_tile_size
        # Stay in fp16 throughout — small running-state precision loss
        # is the trade-off for being lowerable to AIU.
        m = torch.full(q.shape[:-1] + (1,), -65000.0,
                       dtype=q.dtype, device=q.device)  # fp16-safe -inf
        l = torch.zeros(q.shape[:-1] + (1,),
                        dtype=q.dtype, device=q.device)
        o = torch.zeros_like(q)
        for t in range(n_tiles):
            k_tile = k[..., t * k_tile_size:(t + 1) * k_tile_size, :].contiguous()
            v_tile = v[..., t * k_tile_size:(t + 1) * k_tile_size, :].contiguous()
            m, l, o = fa_step(q, k_tile, v_tile, m, l, o, scale)
        return o / l

    try:
        for _ in range(WARMUP): fa_outer(q, k, v)
        _ts.synchronize()
        samples = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            fa_outer(q, k, v)
            _ts.synchronize()
            samples.append(time.perf_counter() - t0)
        fa_ms = statistics.median(samples) * 1e3
        print(f"  FA-tiled (k_tile={k_tile_size}):  {fa_ms:.3f} ms")
    except Exception as e:
        print(f"  FA-tiled: ERR {type(e).__name__}: {str(e)[:80]}")
        fa_ms = None

    if ref_ms and fa_ms:
        print(f"  speedup: {ref_ms / fa_ms:.2f}×")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aiu", action="store_true",
                        help="run AIU benchmarks (default: CPU correctness only)")
    parser.add_argument("--k-tile", type=int, default=128)
    args = parser.parse_args()

    print("# FA tiling Phase 0 prototype\n")
    ok = correctness_test()
    if not ok:
        print("\nCorrectness failed; not running AIU bench.")
        return 1
    if args.aiu:
        # Start with small shape, scale up
        for n_heads, M, head_dim in [
            (8, 256, 128),
            (8, 512, 128),
            (8, 1024, 128),
            (32, 1024, 128),
        ]:
            aiu_bench(M, n_heads, head_dim, args.k_tile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
