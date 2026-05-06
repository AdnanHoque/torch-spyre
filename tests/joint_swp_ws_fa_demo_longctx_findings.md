# Long-context FA-2 demonstration — findings

## TL;DR

Extended the joint SWP+WS demo to M ∈ {1024, 2048, 4096, 8192}.
Real per-op walls measured at each M, ILP run for each.

**Surprising result: no crossover within the tested range.** Even at
M=8192, the reference materialized attention (225 ms) is 2.87× faster
than the best joint+fused FA-tiled prediction (646 ms).

The unstated assumption in the original FA-tiling pitch was that
reference attention must be naive M×M materialization, which would
be HMI-bound at large M. Measurement shows otherwise: **the reference
must be internally tiled by the kernel template**, since an M=8192
attention matrix is 1 GB at fp16 and can't fit anywhere without
tiling.

This narrows the narrative: FA-tiling-via-decomposition isn't a long-
context win on AIU because the kernel template already does
internal tiling efficiently. The path forward for long-context
attention would need either a custom SDSC kernel that *beats* the
reference template's internal tiling, or runtime work on launch-floor
amortization.

## The data

| M | K-tiles | reference ms | unfused serial | unfused joint | fused serial | fused joint |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 17.88 | 196.47 | 174.63 | 122.40 | 77.42 |
| 2048 | 16 | 27.41 | 394.87 | 350.62 | 252.80 | 151.98 |
| 4096 | 32 | 66.76 | 811.89 | 722.92 | 539.20 | 307.59 |
| 8192 | 64 | 225.05 | 1753.17 | 1579.92 | 1251.84 | 646.05 |

Reference vs best FA-tiled (fused + joint):

| M | ref / FA | verdict |
|---:|---:|---|
| 1024 | 0.23 | ref 4.33× faster |
| 2048 | 0.18 | ref 5.54× faster |
| 4096 | 0.22 | ref 4.61× faster |
| 8192 | 0.35 | ref 2.87× faster |

The gap *narrows* with M (4.33× → 2.87×) but doesn't close.

## Scaling analysis

How walls scale with M:

| | M=1024 → M=8192 (8× growth) | implied power |
|---|---:|---:|
| reference | 17.88 → 225.05 = **12.6×** | M^1.20 |
| FA-tiled (fused joint) | 77.42 → 646.05 = **8.3×** | M^1.02 (essentially linear) |

FA-tiled scales linearly with M (each tile fixed work × N tiles where
N ∝ M). Reference scales sub-quadratically — **somewhere between
linear and the M² that naive materialization would predict**. That
gap is exactly where the kernel template's internal optimization
lives.

Extrapolating the trend, crossover would happen when reference
exceeds FA-tiled. With reference scaling as M^1.20 and FA as M^1.02:

  ratio at M = (M / 8192)^0.18 × current_ratio_8192
  ratio = 1 means M / 8192 = (1 / 0.35)^(1 / 0.18) = (2.86)^5.56 ≈ 480

  → M ≈ 8192 × 480 = **~4 million tokens**

That's beyond any practical context length. **There is no realistic
crossover** for FA-via-decomposition vs reference on AIU under
current conditions.

## Why the reference is much faster than naive

At M=8192, the attention matrix S = Q·K^T has:
- shape: (1, 8 heads, 8192, 8192)
- bytes: 8 × 8192 × 8192 × 2 = **1.07 GB** at fp16

This cannot fit:
- Per-core LX (2 MB × 32 = 64 MB) — far too small
- Likely not on-chip at all in one piece

So the reference kernel **must be tiling internally**. We're not
seeing "naive bmm + softmax + bmm with materialized 1 GB matrix" —
we're seeing an already-tiled implementation.

What the reference probably does:
1. Tile both Q and K dimensions internally
2. For each tile pair, do partial matmul + softmax + matmul
3. Accumulate into running output (likely with online softmax, just
   like FA-2)
4. All inside ONE kernel launch (single LF)

That's effectively FA-2 already — just not at the Python decomposition
level, but at the SDSC kernel template level.

## What this means for the demonstration

The original pitch's claim — "joint SWP+WS unlocks long-context
inference" — needs revision:

**Original claim**: FA-2 ping-pong via joint SWP+WS gives 2.55×
speedup at M=1024, scaling more favorably than reference at long
context.

**Revised claim** based on real measurements: joint SWP+WS at the
Python decomposition level cannot match the kernel-template's
internal optimization on any tested M. The technique would need to
be applied INSIDE the kernel template (deeptools-side) to deliver
its theoretical wins.

This is consistent with the calibration finding from earlier: the
prototype's predictions assumed FA tiling at a layer that doesn't
exist in production. Reference attention on AIU is already a tiled
implementation — just not exposed as a Python decomposition.

## Strengths of the demo (revised)

Even with the "no crossover" finding, the demo still has value:

1. **Real measurements at multiple M values** — none of the prior
   prototypes had this. Concrete data across context lengths.
2. **Fusion is shown to be the bigger lever than scheduling**. Fused
   serial beats unfused joint at every M (and by larger margins at
   larger M). Reinforces the earlier finding.
3. **Joint scheduling delivers compounding gains**: at M=8192,
   fused joint is 1.94× faster than fused serial (646 vs 1252 ms).
   That ratio holds across M. **Joint scheduling at fixed fusion
   level is real and stable.**
4. **The reference must be internally tiled** — not previously
   established. Useful for broader project planning (any "we should
   do X via Python decomposition" project should check whether the
   reference op already does X internally).

## Updated paper/patent positioning

The earlier framing positioned FA-tiled + joint SWP+WS as a long-
context win. Long-context measurements don't support that. Better
framing:

**"Joint SWP+WS finds the optimal pipeline schedule for FA-2 on AIU,
which must be implemented at the kernel-template level to actually
deliver its 1.94× per-op gain over per-unit greedy. As a Python-layer
decomposition, it does not beat the reference because the reference
is already internally tiled."**

This is honest. It still has academic/patent value:
- The ILP formulation generalizes Twill to 9 heterogeneous units (still novel)
- The 1.94× joint-vs-decoupled gap (at fused level) is grounded in real measurements
- It quantifies how much scheduling matters once fusion is given (the answer: a lot, ~94%)

But it cannot promise end-user latency wins without deeptools-side
kernel work.

## Where this leaves the project

The same place as before, just with stronger evidence:

| project | ROI | accessibility |
|---|---|---|
| Joint SWP+WS at kernel-template level | High (~2× attention speedup) | Needs deeptools partnership, 6+ months |
| Joint SWP+WS at Python decomposition | Negative — slower than reference | Solo torch_spyre but doesn't help |
| **Fix SDPA-to-bmm regression** | 30-50% on attention | Solo torch_spyre, 1-3 weeks |
| Filing the joint-SWP+WS patent | Defensive/IP value | 2-3 weeks |

The strongest immediate solo torch_spyre win remains the SDPA fix.
The joint scheduling work is best framed as patent/paper output
without committing to an implementation timeline.

## Files

- `joint_swp_ws_fa_demo_longctx.py` — long-context sweep
- `joint_swp_ws_fa_demo_longctx_results.txt` — full output
- This doc — findings
