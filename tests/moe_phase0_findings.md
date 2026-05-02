# MoE grouped-GEMM — Phase 0a findings

The Phase 0a probe (`tests/diag_moe_baseline.py`) measures the cost of
running an MoE FFN layer on Spyre via the naive "E separate matmul calls"
path. Headline: **naive MoE on Spyre is 3× slower than its dense fallback
at top_k=2 and 12× slower at top_k=8**, because per-active-expert wall-
time is dominated by per-kernel launch overhead and scales linearly with
the number of active experts.

This justifies the grouped-GEMM project as the next major work after the
SplitK heuristic. Detailed findings below.

## Method

Two downsized-Mixtral configs at decode M=1:

- (hidden=1024, intermediate=2048, num_experts=8)
- (hidden=1024, intermediate=4096, num_experts=8)

Three measurements per config:

1. **Naive K active experts**: K ∈ {1, 2, 4, 8} sequential `silu(x @ W_gate) *
   (x @ W_up) @ W_down` calls in a Python loop, weighted-summed.
2. **Dense fallback**: column-stacked `(H → E·I)` weights for gate/up and
   `(E·I → H)` for down — one matmul per stage, computes all E experts'
   outputs.

Per-iter sync via `torch_spyre.streams.synchronize()` inside the timed
loop. 5 warmup + 30 measure iters, median reported. Same compile-config
gauntlet as the SplitK / DDR-traffic diagnostics.

## Results

### H=1024, I=2048, E=8

| Variant | Median ms | Per-active-expert ms | vs. naive K=1 |
|---|---:|---:|---:|
| naive K=1 | 15.45 | 15.45 | 1.00× |
| naive K=2 | 30.72 | 15.36 | 1.99× |
| naive K=4 | 61.45 | 15.36 | 3.98× |
| naive K=8 | 123.10 | 15.39 | 7.97× |
| dense fallback (E=8 always run) | **10.12** | — | **0.65×** |

### H=1024, I=4096, E=8

| Variant | Median ms | Per-active-expert ms | vs. naive K=1 |
|---|---:|---:|---:|
| naive K=1 | 15.77 | 15.77 | 1.00× |
| naive K=2 | 31.24 | 15.62 | 1.98× |
| naive K=4 | 62.30 | 15.57 | 3.95× |
| naive K=8 | 123.51 | 15.44 | 7.83× |
| dense fallback (E=8 always run) | **10.91** | — | **0.69×** |

## Two killer observations

### 1. Naive scales perfectly linearly with active experts

Per-active-expert wall-time is constant at ~15.4 ms regardless of K. **Each
new active expert adds a full 15 ms** — pure kernel-launch / dispatch
overhead, no amortization across calls. At Mixtral's top_k=2 each layer
costs ~30 ms; across 32 layers that's ~960 ms per decode token. Unusable
for serving (typical target: <100 ms/token).

### 2. Dense fallback is faster than even a single-expert naive call

Dense computes 8× the arithmetic but runs 35% faster than naive K=1
because it issues 3 kernel launches total instead of 3 per expert.
**Compute is essentially free at decode M=1; launch overhead is
everything.**

Doubling intermediate (2048 → 4096) barely moves wall-time (15.45 → 15.77
ms naive, 10.12 → 10.91 dense). The wall-time floor is determined by
launch + sync, not compute. We're operating at <8% of Spyre's fp16 peak
even on the dense fat matmul.

## Upper-bound analysis for grouped-GEMM

Grouped-GEMM's pitch: "dense's launch profile + oracle compute
correctness." Given the empirical floor of ~10 ms for the dense path,
grouped-GEMM achievable target is also ~10 ms per layer regardless of
top_k, **with the additional benefit of skipping the (E − top_k) / E
compute that dense wastes**:

| Routing config | Naive | Dense fallback | Grouped-GEMM target | Speedup vs. naive |
|---|---:|---:|---:|---:|
| Mixtral 8x7B (E=8, top_k=2) | ~30 ms | ~10 ms | ~10 ms | **3×** |
| DeepSeek-V3 (E=256, top_k=8) | ~120 ms (top-8 only) | dense impractical | ~10 ms | **12×** |
| Phi-MoE (E=16, top_k=2) | ~30 ms | ~10 ms | ~10 ms | 3× |

For DeepSeek-V3 the dense fallback isn't even tractable (256 experts ×
intermediate is enormous), so naive vs. grouped-GEMM is the only real
comparison. **DeepSeek-V3 decode on Spyre is unviable today and only
becomes viable with a grouped-GEMM op.**

## What this *doesn't* answer (Phase 0b territory)

Phase 0a uses downsized dims (H=1024 vs Mixtral real H=4096, I=2048-4096
vs real 14336) for fast iteration. Two open questions for a future Phase
0b:

1. **Does the wall-time floor stay at ~10 ms at real Mixtral dims?** If
   the floor scales with compute rather than overhead at real dims, the
   3-12× gap shrinks. Need a real-dim measurement to validate.
2. **What does token-permute cost?** Permuted-token grouped-GEMM requires
   a `(M, hidden)` gather pre-op + scatter post-op. If that costs 5+ ms
   on Spyre (a real risk given launch overhead dominance), it eats into
   the grouped-GEMM win.

Neither blocks the project — both are bench-harness extensions for the
Phase 1 validation step.

## Decision

**Project proceeds.** Phase 0a's data is conclusive enough at downsized
dims to justify the scope. Real-dim and permute-cost measurements fold
into Phase 1's validation.

## Files

- `tests/diag_moe_baseline.py` — Phase 0a probe
- `tests/diag_moe_baseline_results.md` — auto-regenerated bench output
- `tests/moe_phase0_findings.md` — this document

## Reproducing

```sh
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
cd $DTI_PROJECT_ROOT/torch-spyre
python tests/diag_moe_baseline.py
```

## Phase 0b addendum — real Mixtral dims, framework-overhead isolation, permute probe

Phase 0b extends the bench with three additions:

1. **Real Mixtral 8x7B dims**: H=4096, I=14336, E=8 alongside the
   downsized configs.
2. **Framework-overhead isolation**: an "empty step" measuring just
   `zeros_like + sync` per outer step, and a "single mm" measuring just
   `x @ W` with no SwiGLU pointwise. Together these subtract out the
   per-step overhead from the per-expert numbers.
3. **Token-permute cost probe**: gather + scatter at H=4096 across
   M ∈ {1, 4, 8, 16, 64} to evaluate permuted-token grouped-GEMM
   feasibility.

### Real Mixtral results (H=4096, I=14336, E=8)

| Variant | Median ms | Per-active-expert ms |
|---|---:|---:|
| empty step (zeros_like + sync) | **0.27** | — |
| single mm (no SwiGLU) | **3.87** | — |
| naive K=1 | 18.53 | 18.53 |
| naive K=2 | 36.94 | 18.47 (1.99×) |
| naive K=4 | 73.76 | 18.44 (3.98×) |
| naive K=8 | 147.61 | 18.45 (7.97×) |
| dense fallback (E=8 always run) | **40.23** | — (2.17×) |

### Three things this changes about the Phase 0a story

**1. Framework overhead is negligible (0.27 ms).** The `zeros_like` CPU-
fallback warning we saw is a non-issue. Per-expert ~18ms is essentially
all real Spyre kernel cost. **Concern dismissed.**

**2. Linear scaling persists at real Mixtral dims.** Each active expert
adds 18.5 ms regardless of K. Same regime as downsized. The launch-bound
finding holds at production scale.

**3. Dense fallback is no longer faster than naive at real dims.**
At downsized: dense 10 ms vs naive K=2 30 ms = 3× speedup. At real
Mixtral: dense 40 ms vs naive K=2 37 ms — dense is 9% **slower**. Why?
Dense compute scales with E× while naive scales with K×; at real
intermediate=14336 the `(1, 4096) @ (4096, 114688)` fat matmul is
bandwidth-bound on the 940 MB combined weight matrix. The dense fallback's
launch-amortization advantage was masking E× compute waste at small dims.

### Updated grouped-GEMM target (corrected from Phase 0a)

The Phase 0a estimate of "3× at Mixtral top_k=2" was based on dense's
10 ms floor — which doesn't hold at real dims. The corrected target uses
the real-dim numbers:

| Routing | Naive | Grouped-GEMM target | Speedup |
|---|---:|---:|---:|
| Mixtral 8x7B (E=8, top_k=2) | 37 ms | ~15-20 ms | **~2×** |
| DeepSeek-V3 (E=256, top_k=8) | 148 ms (top-8 only)* | ~30-40 ms | **~5×** |

*For DeepSeek-V3 the per-active-expert cost would scale similarly to
Mixtral — extrapolation, not direct measurement.

Still meaningful, but **the headline is "2-5×" not "3-12×".** Phase 0a
overstated by anchoring on a downsized-dim dense floor that doesn't
generalize.

### Permute probe — token-permuted grouped-GEMM is BLOCKED

| Op | Spyre support |
|---|---|
| `aten::index.Tensor_out` (gather) | ❌ NotImplementedError (eager + compiled) |
| `aten::_index_put_impl_` (scatter) | ❌ NotImplementedError |

Neither op is registered for the Spyre backend. Tested at M ∈ {1, 4, 8,
16, 64} × H=4096 — all unsupported. (M=1 looked like it worked under an
earlier test run because the compile path special-cased it as a slice.
Real gather doesn't work at any M.)

**Implication for Phase 1 design**: permuted-token format requires
gather/scatter ops as a prerequisite. Two paths forward:

- **Block-sparse format** — weights laid out as `(E, H, I)`, kernel
  dispatches cores to different experts. Avoids gather entirely. More
  invasive on the Spyre backend (need core-to-expert dispatch), but no
  op-set prerequisites.
- **Add gather/scatter ops first** — register `aten::index` and
  `aten::scatter`/`aten::index_put` for the Spyre backend (eager + compiled
  paths). Then implement permuted-token grouped-GEMM on top. Smaller
  Phase 1 op design but adds a sequencing dependency.

The block-sparse route is preferred because it avoids the op-set
dependency, and it's also closer to the static-dataflow model Spyre is
designed for. Permuted-token would require dynamic shape dispatch
(M_per_expert known only at runtime) which clashes with the compile model.

### Why dense gets worse at real dims

Each fat matmul in the dense fallback streams through 940 MB of weights
(`4096 × 14336 × 8 × 2 bytes`) per stage. At LPDDR5's ~200 GB/s peak,
that's ~5 ms of pure DMA per stage × 3 stages = ~15 ms of bandwidth-
limited work alone. Plus ~3-4 ms launch overhead each = 30-40 ms total.

The naive K=2 path streams ~234 MB of weights (2 experts' worth × 3
stages each, with per-expert sharing of `x`). Bandwidth-wise that's
cheaper than dense, which is why naive K=2 ≈ dense at real dims.
Grouped-GEMM with proper top_k dispatch streams the same ~234 MB but with
one launch per stage — that's the win mechanism, and it's bandwidth-bound
not launch-bound at real dims.

### Decision (revisited)

**Project still proceeds.** The honest pitch becomes:

> Block-sparse grouped-GEMM op for Spyre, targeting ~2× decode speedup on
> Mixtral 8x7B and ~5× on DeepSeek-V3-class many-small-experts MoE.
> Improves on naive top-k by amortizing kernel launch + DMA setup;
> improves on dense fallback by avoiding (E−top_k)/E compute waste.
> Token-permuted format is blocked at the Spyre op level; block-sparse
> format avoids the dependency.

### What Phase 0b confirms / changes for Phase 1

**Confirms**:
- Linear scaling of naive MoE at real dims (project is justified)
- Framework overhead is not the inflater (numbers are real)

**Changes**:
- Headline win is 2-5×, not 3-12× (corrected)
- Block-sparse format preferred over permuted-token (op-set constraint)
- Bandwidth, not just launch, is the Phase 1 perf model at real dims

### Files added in Phase 0b

- `tests/diag_moe_baseline.py` — extended with real-Mixtral config,
  empty step + single mm overhead probes, and gather/scatter cost probe
- `tests/diag_moe_baseline_results.md` — regenerated with all of the above
