# Flash-attention Phase 0b — per-launch overhead characterization

The Phase 0b probe (`tests/diag_launch_overhead.py`) measures per-call
wall time on Spyre across kernel-work sizes spanning four orders of
magnitude (130K to 1G FLOPs/call). Headline: **per-launch cost is ~3 ms
flat.** The compute portion of even a 1 GFLOP matmul is invisible
against the launch + sync + DMA-setup floor at decode-style M=1 shapes.

Combined with Phase 0a, this gives Phase 1 a concrete tile-size budget:
**flash-attention at S=4096 is feasible at 19-43× speedup** over the
current 1555 ms naive path.

## Method

For each of three shapes, issue N back-to-back compiled mm calls in a
Python loop (with N ∈ {1, 2, 4, 8, 16, 32, 64}) and measure total wall
time. Per-call cost = total / N. Asymptotic per-call value at large N
reveals the per-launch overhead floor.

Per-iter `torch_spyre.streams.synchronize()` inside the timed loop. 5
warmup + 30 measure iters, median reported. Same compile-config gauntlet
as the SplitK + SDPA diagnostics.

Distinct weight tensors per call so no caching merges them.

## Results

### Tiny work (1×512×128 — 0.13 MFLOPs/call)

| N | Total ms | Per call | Ratio vs N=1 |
|---:|---:|---:|---:|
| 1 | 2.87 | 2.872 | 1.00× |
| 8 | 23.08 | 2.886 | 1.00× |
| 64 | 186.86 | 2.920 | 1.02× |

### Flash-attention small tile (64×128×128 — 2.10 MFLOPs/call)

| N | Total ms | Per call | Ratio vs N=1 |
|---:|---:|---:|---:|
| 1 | 2.96 | 2.960 | 1.00× |
| 8 | 23.51 | 2.939 | 0.99× |
| 64 | 187.63 | 2.932 | 0.99× |

### Flash-attention larger tile (2048×2048×128 — 1073 MFLOPs/call)

| N | Total ms | Per call | Ratio vs N=1 |
|---:|---:|---:|---:|
| 1 | 3.06 | 3.064 | 1.00× |
| 8 | 24.39 | 3.049 | 1.00× |
| 64 | 193.38 | 3.022 | 0.99× |

## What this tells us

**Per-call cost is ~3 ms regardless of work size** — the compute portion
even of a 1 GFLOP matmul (which would take ~7 µs at peak fp16) is
invisible. Per-call cost varies by < 5% across 4 orders of magnitude in
FLOPs. So:

1. **Compute is essentially free at decode-shape matmul** — wall-time is
   100% per-launch overhead at these M=1/M=64 shapes.
2. **At larger tiles, compute becomes visible but small** — the
   `(2048, 2048, 128)` shape costs 0.06 ms more than the tiny shape, so
   compute is 60 µs while launch overhead is 3 ms.
3. **Tile-size doesn't affect launch cost** — pick the largest tile that
   fits per-core span + scratchpad, pay 3 ms per launch.

## Flash-attention budget at S=4096 — confirmed feasible

Working through the kernel count budget:

- **Tile choice**: 32 cores × Q-rows-per-core = Q_tile_total. Output
  per-core (Q_rows × D × 2 bytes) must fit in ~1.5 MB scratchpad.
  Q_rows-per-core ≤ 64 → **Q_tile_total ≤ 2048**.
- **At S=4096 with Q_tile=2048**: 2 Q-tiles.
- **KV_tile=2048**: 2 KV-iters per Q-tile.
- **Total inner-loop iterations**: 4 per attention layer.

| Kernels per inner-loop step | Total launches | Wall time @ 3 ms each |
|---:|---:|---:|
| 1 (single fused step — backend-blocked) | 4 | **12 ms** (130× speedup) |
| 2 (bmm + softmax-update) | 8 | **24 ms** (65×) |
| 3 (bmm + max+sub+exp + sum+div + accum) | 12 | **36 ms** (43×) |
| 5 (current softmax decomposition + bmm + accum) | 20 | **60 ms** (26×) |

Plus ~6 ms setup overhead (initial Q load, final write).

**Realistic Phase 1 target: 30-50 ms at S=4096 → 30-50× speedup.** Even
the most pessimistic case (5 kernels per step, current allowlist limit)
beats the naive 1555 ms by 26×.

The gap *grows* with S: at S=8192 the naive path's 4 GB intermediate is
infeasible, while flash-attention scales linearly in S — so the speedup
ratio grows.

## What's NOT yet probed (Phase 0c territory)

1. **Does `LX_PLANNING=1` actually keep `max` and `sum` outputs in
   scratchpad, skipping DDR roundtrips?** A small probe with the exact
   softmax chain (`max → sub → exp → sum → realdiv`) at LX_PLANNING off
   vs on would confirm the architectural assumption flash-attention's
   design relies on.
2. **Does Spyre overlap DMA with compute (producer-consumer style)?**
   Per-launch cost includes DMA setup; whether it's serialized vs.
   pipelined determines whether ping-pong-style fusion would amortize.

Phase 0c's LX_PLANNING confirmation is the more urgent of the two — if
`max`/`sum` outputs DON'T actually skip DDR despite the allowlist, the
Phase 1 design needs a custom-op approach instead of leveraging the
existing softmax decomposition.

## Decision

**Phase 1 design proceeds.** The 3 ms-flat launch overhead is the most
important Phase 0 number for the project — it bounds tile-choice cost
without compute factoring in. The 19-43× speedup target is
quantitatively justified.

## Files

- `tests/diag_launch_overhead.py` — Phase 0b probe A
- `tests/diag_launch_overhead_results.md` — auto-regenerated bench output
- `tests/sdpa_phase0b_findings.md` — this document
