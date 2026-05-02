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
