# Cost-model planner — Phase 1.0 findings

The Phase 1.0 probe (`tests/diag_split_gap.py`) measures, for each of 13
production matmul shapes, the gap between the default planner's `(m, n,
k)` factorization choice and the empirical best across all valid
factorizations of `m·n·k = 32`. Headline: **average gap 10.3%, max gap
38.5%, 12 of 13 shapes have positive gap.** This is well above the
threshold needed to justify a cost-model planner, so Phase 1.1
(formalization) is greenlit.

## Method

For each shape:

1. Capture the default planner's chosen split via the SDSC `parse_op_spec`
   hook (same mechanism as the SplitK Phase 0 work).
2. Enumerate all `(m, n, k)` factorizations of 32 satisfying:
   - `M / m` is integer (M is non-stick)
   - `N / n` ≥ 64 elements and stick-aligned (N is stick on output)
   - `K / k` ≥ 64 elements and stick-aligned (K is stick on input A)
3. Force each valid factorization via the
   `multi_dim_iteration_space_split` monkey-patch (same as DDR-traffic
   Phase 0).
4. Measure wall time (3 warmup + 15 timed iters, per-iter
   `torch_spyre.streams.synchronize()`).
5. Compute gap = `(default_ms - best_forced_ms) / default_ms`.

## Results

### Per-shape summary

| Shape | Use case | Default split | Default ms | Best forced | Best ms | Gap |
|---|---|---|---:|---|---:|---:|
| **L3-70B q_proj prefill** | Llama-3-70B prefill | `[128×32c, 8192×1c, 8192×1c]` | 6.54 | `(2, 16, 1)` | **4.02** | **+38.5%** |
| **L3-8B MLP down prefill** | Llama-3-8B MLP | `[128×32c, 4096×1c, 14336×1c]` | 6.03 | `(2, 1, 16)` | **4.20** | **+30.3%** |
| **Mixtral down per-expert** | Mixtral 8x7B | `[128×32c, 4096×1c, 14336×1c]` | 6.07 | `(2, 1, 16)` | **4.23** | **+30.2%** |
| L3-8B q_proj prefill | Llama-3-8B prefill | `[128×32c, 4096×1c, 4096×1c]` | 3.83 | `(1, 32, 1)` | 3.24 | +15.4% |
| L3-70B GQA kv_proj prefill | Llama-3-70B GQA | `[128×32c, 1024×1c, 8192×1c]` | 3.46 | `(2, 16, 1)` | 3.16 | +8.7% |
| L3-8B GQA kv_proj prefill | Llama-3-8B GQA | `[128×32c, 1024×1c, 4096×1c]` | 3.24 | `(2, 16, 1)` | 3.07 | +5.2% |
| DeepSeek-MoE gate | DeepSeek-MoE per-expert | `[192×32c, 1408×1c, 2048×1c]` | 3.23 | `(8, 1, 4)` | 3.15 | +2.5% |
| Qwen3-MoE gate | Qwen3-MoE per-expert | `[128×32c, 1536×1c, 2048×1c]` | 3.15 | `(1, 1, 32)` | 3.10 | +1.8% |
| L3-8B MLP gate/up prefill | Llama-3-8B MLP | `[128×1c, 14336×32c, 4096×1c]` | 3.80 | `(1, 32, 1)` | 3.77 | +0.8% |
| L3-70B GQA TP=8 kv prefill | Llama-3-70B GQA TP=8 | `[128×32c, 128×1c, 8192×1c]` | 3.00 | `(32, 1, 1)` | 3.00 | +0.2% |
| L3-8B q_proj decode | Llama-3-8B decode | `[4096×32c, 4096×1c]` | 3.22 | `(1, 16, 2)` | 3.21 | +0.2% |
| L3-70B GQA TP=8 kv decode | Llama-3-70B decode | `[128×2c, 8192×16c]` | 3.06 | `(1, 1, 32)` | 3.06 | +0.2% |
| L3-70B MLP down prefill | Llama-3-70B MLP | `[128×16c, 8192×2c, 28672×1c]` | 8.01 | `(16, 2, 1)` | 8.03 | -0.3% |

**Aggregate**: 13 shapes measured, average gap **10.3%**, max gap **38.5%**, 12 shapes have positive gap.

## Three patterns the data reveals

### 1. The planner's M-greedy default is consistently suboptimal for prefill

For prefill shapes (M=128 typical), the planner almost always picks
`(32, 1, 1)` — give all 32 cores to M. The empirical best is rarely
this. For 8 of 9 prefill shapes, a different factorization was faster.

The intuition holds: M=128 split 32 ways gives only 4 rows per core,
which is too small to amortize fixed kernel costs. Spilling some cores
to N or K gives bigger per-core work-slices and better total wall-time.

### 2. K-split is sometimes the winner — not just (m, n) re-balancing

For L3-8B MLP-down and Mixtral down (both `(128, 4096, 14336)`),
the empirical best is `(2, 1, 16)` — *2 cores on M, 16 cores on K, 1
core on N*. The SplitK heuristic on `AdnanHoque/diag-splitk-matmul`
catches part of this regime by allowing K-priority, but it forces
`(1, 1, 32)` not `(2, 1, 16)`. The cost model can pick the actual
winner, not just the K-priority binary toggle.

### 3. Decode and TP=8 small-N shapes have negligible gap

M=1 decode shapes and `(128, 128, 8192)` (L3-70B GQA TP=8) all show
~0% gap. These are launch-overhead-dominated — wall time is dictated
by the ~3 ms per-launch floor we measured in flash-attention Phase 0b,
not by which split is picked. A cost model would need to recognize this
regime and not over-optimize.

## Side finding: backend EAR overflow on some K-split factorizations

For L3-70B MLP-down (`(128, 8192, 28672)`), several K-split
factorizations triggered:

```
DtException: EAR overflow detected, file
deeptools/dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:780
```

This is a known Spyre backend constraint (per-tensor address-encoding
limit) that the planner already respects via `must_split_vars`. The
cost model needs to **predict which factorizations the backend rejects**
or empirically test before committing — otherwise we'd score
unimplementable splits as "best" and crash at compile time.

## What this means for Phase 1.1

The cost model needs to predict, per `(m, n, k)` factorization for a
given shape:

1. **Wall time** — the metric we minimize.
2. **Backend feasibility** — some factorizations error out (EAR
   overflow, span limit, scratchpad fit). Must be filtered or
   predicted as ∞-cost.

The decomposition we'll work toward in Phase 1.1:

- Estimate components: `T_compute(m, n, k, M, N, K, dtype)`,
  `T_load(m, n, k, ...)`, `T_store(m, n, k, ...)`,
  `T_launch_floor` (the ~3 ms constant).
- Combine: TBD based on whether Spyre pipelines load/compute/store
  (need a separate probe to test).
- Calibrate: against the data this Phase 1.0 produced + the
  SplitK + DDR-traffic + Stream-K phases' data.

Specifically NOT borrowing TL paper formulas — terminology will fall
out of what we observe empirically.

## Comparison to Phase 0 measurements

This probe extends and validates the DDR-traffic Phase 0 finding:

- **DDR-traffic Phase 0** (committed `88573b8` on `AdnanHoque/diag-splitk-matmul`)
  measured 3 shapes with their full factorization sweep and showed
  inter-split wall-time variance of 1.5-2× per shape.
- **Phase 1.0 here** measures 13 production shapes with the same
  methodology. Confirms the variance is consistent across the
  production catalog and quantifies the planner-vs-best gap.

## Decision

**Phase 1.1 (cost model formalization) is justified.** The 10.3%
average + 30-38% peak gap is well above the threshold I set for
gating the project at this measurement.

## Files

- `tests/diag_split_gap.py` — Phase 1.0 probe
- `tests/diag_split_gap_results.md` — auto-regenerated bench output
  (full per-shape detail tables)
- `tests/cost_model_phase1_0_findings.md` — this document

## Reproducing

```sh
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
cd $DTI_PROJECT_ROOT/torch-spyre
python tests/diag_split_gap.py
```
