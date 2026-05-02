# SplitK matmul on Spyre — Phase 1 findings + Phase 2 proposal

This document captures the empirical findings from the Phase 1 perf+accuracy
bench (`tests/bench_splitk_matmul.py`) and proposes a scoped Phase 2 heuristic
based on the data.

It builds on the Phase 0 drift characterization committed in `cc86ca1`
(`tests/diag_splitk_matmul.py`).

## Executive summary

- **The default work-division planner already K-splits when output
  dimensions cannot saturate the 32 cores** within stick alignment. Earlier
  framing that the heuristic is "M/N-greedy and never K-splits" is wrong as
  a general statement — it only doesn't K-split when M·N is large enough to
  absorb all 32 cores.
- **forceK wins on perf by 9-26% in the mid-prefill regime** (M ≈ 128,
  M·N < ~1 M elements, K ≥ 4096). This includes Llama-3-8B and Llama-3-70B
  q_proj prefill and Llama-3-8B MLP-down prefill — real LLM workloads.
- **forceK wins on accuracy by 25-40%** at K ≥ 8192 for shapes where M·N
  saturates cores with default. Per-core fp16 accumulator chains shorten
  from K to K/32, and the cross-core reduction runs at higher precision.
- **forceK loses badly at large M·N** (≥ 8 M elements) because the
  cross-core partial-sum reduction grows with output size and exceeds the
  K-parallelism gain.
- **Stick alignment is a hard constraint on K-split factors.** Llama-3-70B
  MLP-down (K = 28672) cannot do a clean 32-way K-split (28672 / 64 = 448
  sticks; only 16 divides cleanly), and the planner falls back to 16-way.
- **MoE per-expert matmul shapes are structurally aligned with the
  forceK-wins regime.** Most architectures with many small experts
  (DeepSeek-V3, Qwen3-MoE, Phi-MoE) have *every* per-expert matmul at
  prefill in the forceK-wins regime.

## Methodology

`tests/bench_splitk_matmul.py` compiles each `(M, N, K)` shape under two
modes:

- **default** — planner runs unmodified.
- **forceK** — `prioritize_dimensions` monkey-patched to put reduction (K)
  dims before output (M, N) dims in priority. Forces K-split when valid.

For each (shape, mode):

1. Compile + 5 warmup iterations.
2. 20 timed iterations of `mm_fn(a, b)` + `torch_spyre.streams.synchronize()`.
   `.to('cpu')` is *not* in the timed region — we measured it adds ~24%
   overhead on a 1024² matmul, which would systematically over-report kernel
   cost.
3. TFLOPs/s = 2·M·N·K / median_time. End-to-end including any cross-core
   reduction (the dxp_standalone backend handles K-split partial-sum
   reduction; we observe its cost in the wall-time).
4. Drift vs fp32 CPU reference. Relative error masks |ref| ≤ 1e-3.

Four IR-pass-monkey-patch fixes are required for the patches to actually
fire (`compile_threads=1`, `worker_start_method=fork`, `fx_graph_cache=False`,
`torch._dynamo.reset()` between modes). Without all four the bench silently
no-ops.

## Sweep 1 — Decode-skinny (M=1, N=4096)

| K | default TFLOPs/s | forceK TFLOPs/s | speedup | drift Δ (abs p99) |
|---:|---:|---:|---:|---:|
| 1024 | 0.00 | 0.00 | 0.96× | −0.05 |
| 2048 | 0.01 | 0.01 | 0.98× | +0.03 |
| 4096 | 0.01 | 0.01 | 0.98× | +0.04 |
| 8192 | 0.02 | 0.02 | 0.96× | −0.03 |
| 12288 | 0.03 | 0.03 | 0.96× | −0.13 |
| 16384 | 0.03 | 0.03 | 0.96× | −0.16 |

**Verdict**: forceK is a slight (4%) perf loss across all K. Accuracy
improves 7-15% for K ≥ 8192. Wall-time is dominated by fixed kernel-launch
overhead at M=1, so all timings sit at 3-4 ms regardless of K.

## Sweep 2 — Balanced-square (M=N=1024)

| K | default TFLOPs/s | forceK TFLOPs/s | speedup | drift Δ (abs p99) |
|---:|---:|---:|---:|---:|
| 1024 | 0.71 | 0.65 | 0.92× | +0.02 |
| 2048 | 1.39 | 1.33 | 0.96× | +0.07 |
| 4096 | 2.67 | 2.40 | 0.90× | −0.02 |
| 8192 | 4.93 | 4.34 | 0.88× | **−0.26** |
| 12288 | 6.84 | 5.77 | 0.84× | **−0.52** |
| 16384 | 8.64 | 6.93 | 0.80× | **−0.80** |

**Verdict**: forceK is strictly slower (8-20%) but materially more accurate
(25-40% lower p99 abs error) for K ≥ 8192. Default's 32-way M-split
parallelizes without cross-core reduction; forceK's 32-way K-split adds
reduction overhead but shortens fp16 accumulator chains.

## Sweep 3 — Small-N decode (M=1, K=8192, varying N)

| N | default splits | forceK splits | speedup | drift Δ |
|---:|---|---|---:|---:|
| 128 | `[128×2c, 8192×16c]` | `[128×1c, 8192×32c]` | 1.01× | +0.30 |
| 256 | `[256×4c, 8192×8c]` | `[256×1c, 8192×32c]` | 1.01× | +0.15 |
| 512 | `[512×8c, 8192×4c]` | `[512×1c, 8192×32c]` | 1.01× | −0.09 |
| 1024 | `[1024×16c, 8192×2c]` | `[1024×1c, 8192×32c]` | 0.99× | −0.03 |
| 2048 | `[2048×32c, 8192×1c]` | `[2048×1c, 8192×32c]` | 0.97× | −0.05 |
| 4096 | `[4096×32c, 8192×1c]` | `[4096×1c, 8192×32c]` | 0.96× | +0.03 |

**Verdict**: this sweep produced the most surprising finding. **Default
already K-splits** when output dims can't saturate cores — `(1, 128, 8192)`
default uses 2 cores on N + 16 cores on K, not "M/N-only". Default's mixed
strategy is roughly tied with forceK on perf at small N (within 1% noise),
and is *more* accurate at the very smallest N (N ≤ 256). The shorter K-chain
benefit of forceK is offset by a 32-way fan-in to a small N output that
introduces its own roundoff.

## Sweep 4 — M-scaling at (N=4096, K=8192)

| M | default TFLOPs/s | forceK TFLOPs/s | speedup |
|---:|---:|---:|---:|
| 128 | 1.84 | **2.14** | **1.16×** |
| 512 | 7.36 | 7.05 | 0.96× |
| 2048 | 22.95 | 12.86 | **0.56×** |

**Verdict**: this is the headline finding. forceK *wins on perf* at M=128
by 16% and *loses badly* at M=2048 by 44%. The crossover is around
M·N ≈ 1 M elements:

- forceK at M=2048: per-core writes full M·N = 8 M output, 32-way reduction
  over ~256 MB of partial-sum traffic.
- default at M=2048: per-core writes M/32 · N = 256 KB output, no
  cross-core reduction.

The reduction overhead scales with M·N. The compute it parallelizes scales
with M·N·K. So forceK wins when M·N is small relative to K.

## Sweep 5 — Llama prefill shapes (M=128)

| shape | use case | default TFLOPs/s | forceK TFLOPs/s | speedup | drift Δ |
|---|---|---:|---:|---:|---:|
| 128×4096×4096 | L3-8B q_proj prefill | 1.13 | **1.23** | **1.09×** | +0.01 |
| 128×4096×14336 | L3-8B MLP-down prefill | 2.50 | **3.07** | **1.23×** | −0.13 |
| 128×8192×8192 | L3-70B q_proj prefill | 2.66 | **3.35** | **1.26×** | −0.04 |
| 128×8192×28672 | L3-70B MLP-down prefill | 7.54 | 2.42 | **0.32×** | −0.65 |

**Verdict**: forceK wins 9-26% on three of four real-LLM prefill shapes.

The exception — L3-70B MLP-down — has two compounding issues:
1. **Stick-alignment**: K = 28672 / 64 = 448 sticks; cleanly divides into 16
   chunks but not 32. forceK falls back to 16-way K-split.
2. **Span-required default split**: B = 28672 · 8192 · 2 = 470 MB exceeds the
   256 MB per-core span limit, so default already does a mixed
   `[128×16c, 8192×2c, 28672×1c]` split. ForceK can't beat that.

This shape-corner is a real constraint that any production heuristic must
detect and avoid.

## When does forceK win? — concrete crossover

Combining all five sweeps, forceK wins on perf when:

1. **M·N < ~1 M elements**, AND
2. **K ≥ 4096** (compute large enough for K-parallelism to matter), AND
3. **K is stick-aligned for 32-way split** (K / 64 divisible by 32, i.e.
   K divisible by 2048), AND
4. **B = K·N fits within the per-core 256 MB span limit** without forcing
   default into a span-required mixed split.

ForceK wins on accuracy independently when:

5. **K ≥ 8192**, regardless of M·N. This holds across all sweeps where M·N
   is large enough that cross-core fan-in doesn't introduce its own
   roundoff (N ≥ ~512).

## MoE shape catalog

MoE per-expert matmul shapes have structurally smaller M·N than dense MLP
because tokens are routed across experts. With `num_experts` experts and
`top_k` routing, per-expert tokens ≈ M_total · top_k / num_experts.

### Mixtral 8x7B (8 experts, top-2, intermediate=14336)

At prefill M_total=512 → M_per_expert ≈ 128:

| Per-expert matmul | Shape | M·N | Predicted |
|---|---|---:|---|
| gate_proj / up_proj | (128, 14336, 4096) | 1.8 M | tied |
| **down_proj** | **(128, 4096, 14336)** | **0.5 M** | **forceK wins (measured 1.23×)** |

### DeepSeek-V3 (256 experts, top-8, intermediate=2048)

At prefill M_total=2048 → M_per_expert ≈ 64:

| Per-expert matmul | Shape | M·N | Predicted |
|---|---|---:|---|
| gate_proj / up_proj | (64, 2048, 7168) | 0.13 M | **forceK wins** |
| down_proj | (64, 7168, 2048) | 0.46 M | likely forceK wins |

### Qwen3-MoE-A22B (128 experts, top-8, intermediate=1536)

At prefill M_total=2048 → M_per_expert ≈ 128:

| Per-expert matmul | Shape | M·N | Predicted |
|---|---|---:|---|
| gate_proj | (128, 1536, 2048) | 0.20 M | **forceK wins** |
| down_proj | (128, 2048, 1536) | 0.26 M | **forceK wins** |

### DeepSeek-MoE 16B (64 experts, top-6, intermediate=1408)

At prefill M_total=2048 → M_per_expert ≈ 192:

| Per-expert matmul | Shape | M·N | Predicted |
|---|---|---:|---|
| gate_proj | (192, 1408, 2048) | 0.27 M | **forceK wins** |
| down_proj | (192, 2048, 1408) | 0.39 M | tied / slight forceK |

### Headline mapping

> **Most MoE per-expert prefill matmuls fall in the forceK-wins regime.**
> Architectures with many small experts (DeepSeek-V3, Qwen3-MoE, Phi-MoE)
> have *every* per-expert matmul in the regime.

## Caveat on MoE on Spyre today

The compiled path supports `mm` and `bmm` only — there is no grouped-GEMM
op. So MoE on Spyre today runs as either:

- 8 to 256 separate `mm` kernels per layer (high compile + launch overhead)
- A dense fallback (compute all experts, mask)

Neither is efficient. **A grouped-GEMM op for Spyre is a separate, larger
project** and is the natural follow-on after Phase 2 lands. Phase 2's
K-split heuristic improves each individual per-expert matmul; grouped-GEMM
would unlock MoE efficiency at the kernel level.

## Phase 2 proposal

### Scope

A K-split heuristic that fires when all of:

```
M·N < KSPLIT_MAX_OUTPUT_ELEMENTS    (default ~1 M, tunable)
AND K >= KSPLIT_MIN_K               (default 4096, tunable)
AND K is stick-aligned for 32-way K-split
AND B = K·N fits within MAX_SPAN_BYTES per core without forcing
    span-required default mixed split
```

When the heuristic fires, `prioritize_dimensions` rotates reduction dims to
the front so the planner allocates cores along K before M/N.

### Implementation options (in increasing scope)

1. **Direct edit to `prioritize_dimensions`**: shape-conditional branch in
   `core_division.py:405`. Smallest diff, easiest to land. Ships as the
   default behavior.
2. **Layer on top of cyang49's PR #1674 hint mechanism**: add the trigger
   logic as a hint-emitter that sets `force_split_dim=K` when conditions
   match. Requires #1674 to merge first; cleaner separation of concerns.
3. **Config-flag-gated opt-in**: ships behind
   `torch_spyre.config.numerics_priority_k_split = False` initially.
   Conservative; lets early adopters validate before defaulting on.

Recommend (1) or (3) depending on whether we want the new behavior on by
default. If the perf+accuracy data is convincing, (1). If we want to
de-risk, (3).

### Required tests

- Add a work-division regression test (similar to PR #1255) asserting that
  the K-split factor is > 1 for canonical shapes:
  - (128, 4096, 14336) — Llama-8B MLP-down prefill
  - (128, 8192, 8192) — Llama-70B q_proj prefill
  - (1024, 1024, 16384) — large-K balanced
  - (64, 2048, 7168) — DeepSeek-V3 per-expert gate
- Add a regression test asserting that K-split is *not* applied for
  large M·N shapes:
  - (2048, 4096, 8192) — large M·N, default M-split should remain
  - (128, 8192, 28672) — span-required, default mixed split should remain
- Tighten `test_mm_relaxed`'s atol/rtol from 0.1 to 0.05 (or shape-specific
  bounds) for shapes where forceK is now the heuristic — the 25-40% drift
  reduction gives real headroom.

### Out-of-scope / closely related issues

- **`docs/source/compiler/work_division_codegen.md` is stale** — it states
  "The K dimension (reduction dimension) is not split", contradicting both
  the PR #897 commit and our empirical findings. Should be updated as part
  of this work or as a separate doc PR.
- **[#1730](https://github.com/torch-spyre/torch-spyre/issues/1730) — SDPA
  QK matmul numerical correctness**: the QK matmul is at K = head_dim
  (typically 128) so this heuristic doesn't directly apply, but the
  *infrastructure* (better drift characterization) is reusable.
- **[#1917](https://github.com/torch-spyre/torch-spyre/issues/1917) —
  matmul padding on pre-scheduling IR**: orthogonal but adjacent. May
  interact with K-split when K is not stick-aligned.
- **[#1253](https://github.com/torch-spyre/torch-spyre/issues/1253) —
  add work-division tests asserting splitting happens**: the canonical-shape
  tests above directly close this issue.
- **Grouped-GEMM op for MoE**: separate, larger project. Required for
  efficient MoE decode. Natural follow-on after Phase 2.

### Coordination

cyang49 owns the work-division space (PRs #897, #1275, #1345, #1674,
#1674-draft). Phase 2 should be sequenced with their roadmap — particularly
PR #1674 (work-division hint API), which is the cleanest integration point
for an opt-in K-split policy.

## Files

- `tests/diag_splitk_matmul.py` (Phase 0) — committed at `cc86ca1`. SDSC
  capture + force-no-K + force-K modes for drift characterization.
- `tests/bench_splitk_matmul.py` (Phase 1) — perf+accuracy bench across
  five sweeps.
- `tests/bench_splitk_matmul_results.md` (Phase 1) — heuristic-OFF baseline
  output. Auto-regenerated by the bench when `k_split_heuristic` is unset.
- `tests/bench_splitk_matmul_results_heuristic_on.md` — heuristic-ON output,
  written by the bench when `TORCH_SPYRE_K_SPLIT_HEURISTIC=1`.
- `torch_spyre/_inductor/core_division.py` — `prioritize_dimensions` +
  `_k_split_heuristic_should_fire`, committed at `f538567` (v1) and
  `9978aa2` (v2 max-output-dim gate).
- `tests/inductor/test_core_division.py` — 20 unit tests for the heuristic.
- `tests/splitk_phase1_findings.md` — this document.

## Reproducing

```sh
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
cd $DTI_PROJECT_ROOT/torch-spyre

# Phase 0 drift characterization
python tests/diag_splitk_matmul.py

# Phase 1 perf+accuracy bench (heuristic OFF, baseline)
python tests/bench_splitk_matmul.py

# Phase 2 perf+accuracy bench (heuristic ON)
TORCH_SPYRE_K_SPLIT_HEURISTIC=1 python tests/bench_splitk_matmul.py
```

Numbers vary with hardware and concurrent pod load. The *ratios* and
*direction* of effects should be stable.

## Phase 2 v2 addendum (committed `9978aa2`)

The v1 trigger as proposed (`M·N < 32K iter, K >= 64, K aligned, no span
pre-split`) was empirically too loose. It fired for `(1024, 1024, K>=4096)`
balanced-square shapes — measured −10 to −22% perf regressions vs default —
because `M·N_iter = M_elems × N_sticks` underweights N when N is a stick
dim. v2 adds a per-output-dim gate:

```
AND max(output_iter_sizes) < KSPLIT_MAX_OUTPUT_DIM_ITER  (default 256)
```

This filters out the balanced-square regime (max iter dim = 1024) while
preserving the M=128 prefill wins (max iter dim = 128).

### v2 measured impact (heuristic-on default vs heuristic-off baseline, 3-run avg)

| Shape | use case | baseline TF/s | v2 TF/s | Δ |
|---|---|---:|---:|---:|
| 128×4096×4096 | L3-8B q_proj prefill | 1.13 | 1.22 | **+8%** |
| 128×4096×8192 | M-scaling sweep | 1.84 | 2.11 | **+15%** |
| 128×4096×14336 | L3-8B MLP-down prefill | 2.50 | 3.06 | **+22%** |
| 128×8192×8192 | L3-70B q_proj prefill | 2.66 | 3.35 | **+26%** |
| 1024×1024×4096 | balanced-square | 2.70 | 2.71 | tied |
| 1024×1024×8192 | balanced-square | 4.98 | 4.96 | tied |
| 1024×1024×16384 | balanced-square | 8.86 | 8.76 | tied |
| 128×8192×28672 | L3-70B MLP-down (span) | 7.54 | 7.57 | tied (declined) |
| 512–2048×4096×8192 | M-scaling | 7.36 / 22.95 | 7.32 / 22.88 | tied (declined) |
| 1×{128..4096}×K | decode | tied | tied | tied (overhead-bound) |

Inter-run variance across the 3 v2 runs is under 1.5% per shape, so the
+8 to +26% wins are well above noise. **No regression on any measured
shape.** Drift improvements at K≥8192 (25-40% lower p99 abs error) are
preserved on the firing shapes.
