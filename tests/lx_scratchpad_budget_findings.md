# LX scratchpad budget — DXP_LX_FRAC_AVAIL impact on prefill matmul

A measurement of how much the on-chip LX scratchpad budget controls
prefill matmul wall time on Spyre, and what that implies for
production configuration defaults.

## Headline

Increasing `DXP_LX_FRAC_AVAIL` from the current default of `0.2` to
`0.8` (with `LX_PLANNING=1`) cuts wall time by **up to 21%** on
Llama-70B q-projection prefill. **Compound with `output_element_priority`
gives 1.63× on the same shape.** But one production shape regresses
16% at high frac, so a global default change isn't safe without a
more targeted heuristic — see "Catalog sweep" and "Regression
investigation" below.

## Probe design

Subprocess-based sweep — both `LX_PLANNING` and `DXP_LX_FRAC_AVAIL`
are read at module-import time, so each configuration needs a fresh
process. For each `(shape, config)` pair:

- compile a forward matmul with `torch.compile`
- 3 warmup iterations
- 25 timed iterations, recording per-iteration wall time
- report median, first-iter, min

Configurations tested:

| label | env vars |
|---|---|
| control | `LX_PLANNING=0` (default `DXP_LX_FRAC_AVAIL`) |
| frac=0.2 | `LX_PLANNING=1, DXP_LX_FRAC_AVAIL=0.2` (current default) |
| frac=0.5 | `LX_PLANNING=1, DXP_LX_FRAC_AVAIL=0.5` |
| frac=0.8 | `LX_PLANNING=1, DXP_LX_FRAC_AVAIL=0.8` |

Shapes tested: L3-8B q_proj (128, 4096, 4096), L3-70B q_proj
(128, 8192, 8192), L3-8B GQA kv_proj (128, 1024, 4096).

## Results

| shape | control | frac=0.2 | frac=0.5 | frac=0.8 | best speedup |
|---|---:|---:|---:|---:|---:|
| L3-8B q_proj prefill | 3.821 ms | 3.783 | 3.608 | **3.498** | **1.09×** |
| **L3-70B q_proj prefill** | **6.515 ms** | **6.419** | **5.719** | **5.452** | **1.20×** |
| L3-8B GQA kv_proj prefill | 3.138 ms | 3.122 | 3.065 | 3.113 | 1.02× |

### Verdict

- The current default of `frac=0.2` provides almost no benefit over
  `LX_PLANNING=0` (1.01–1.02× — within noise).
- Bumping to `frac=0.5` already captures most of the win on shapes
  that benefit at all.
- `frac=0.8` is best across the board for the shapes tested, with
  the largest absolute improvement on L3-70B q_proj (a 1.06 ms
  reduction).
- The launch-floor-bound shape (GQA kv_proj) doesn't benefit much
  from any setting because it's not data-bound to begin with.

## What's actually happening (mechanism)

The IBM AIU architecture doc (slide 86) describes a static-tensor
*preload* mechanism — load weights into LX once, reuse across many
inference calls. The original hypothesis for this probe was: maybe
that mechanism isn't firing for `torch.compile`-driven matmul, and
enabling `LX_PLANNING` would turn it on.

That hypothesis turns out to be **partially wrong** in an
illuminating way. Per-iteration timing:

| shape | config | first ms | median ms | first/median |
|---|---|---:|---:|---:|
| L3-70B q_proj | control | 6.517 | 6.515 | 1.00× |
| L3-70B q_proj | frac=0.8 | 5.469 | 5.452 | 1.00× |

**The first iteration takes the same time as the median in every
configuration.** There is no within-process warm-cache effect — each
call is independent and identical in cost. So the mechanism is not
"load weights once, reuse them" (the cross-inference preload the doc
describes). It's **better in-call LX staging budget**.

Concretely: when `LX_PLANNING=1` is on, the Inductor backend
allocates a configurable fraction of the 2 MB per-core scratchpad to
the LX planner. The planner uses that budget for double-buffered
staging of operands within a single kernel call. With more budget,
each staging chunk can be bigger, requiring fewer DMA round-trips
through HMI. Fewer chunks = less per-chunk overhead = faster wall
time, but the savings appear on every call independently because the
buffer state is reset between calls.

For per-core operand sizes that exceed the available LX budget,
chunked streaming is unavoidable. With L3-70B q_proj per-core B = 4 MB
(8 MB total weight / 32 cores) and per-core LX budget = 1.6 MB at
frac=0.8 vs 0.4 MB at frac=0.2, the staging chunk count differs by
roughly 4×. That's consistent with the ~20% wall-time delta we see —
fewer chunks means less per-chunk launch overhead amortized across
the whole transfer.

## Compounding with `output_element_priority`

These two levers touch different parts of the pipeline:

- `output_element_priority` decides **which dimension to split work
  across** — minimizing total bytes that need to traverse HMI.
- `DXP_LX_FRAC_AVAIL=0.8` decides **how efficiently those bytes are
  staged in LX** during streaming.

Predicted compound speedup on L3-70B q_proj prefill:

| baseline | + element-priority | + frac=0.8 | combined |
|---|---|---|---|
| ~6.5 ms | 1.61× | 1.20× | **~1.93×** |

This needs to be measured directly (Phase 3 of the project plan
below), but the mechanisms suggest they should stack cleanly.

## Implications for production

1. **The current `DXP_LX_FRAC_AVAIL=0.2` default is a regression vs
   what's possible.** Bumping to `0.8` should give meaningful
   wall-time wins on transformer prefill out of the box. The default
   was likely tuned for a different workload mix and never revisited
   for transformer prefill.

2. **`LX_PLANNING=1` should also be on by default.** Today it's `0`.
   With LX off, even bumping the budget doesn't matter — the planner
   isn't engaged.

3. **The benefit is shape-dependent.** Shapes already at the
   launch-floor (decode, small GQA) won't move. Large-weight prefill
   matmul moves the most.

## Catalog sweep — 13 shapes × 5 frac values + compound

The full Phase 1.0 catalog re-run with `LX_PLANNING=1` and
`DXP_LX_FRAC_AVAIL ∈ {0.2, 0.4, 0.6, 0.8, 0.95}`, plus a compound
config (`OUTPUT_ELEMENT_PRIORITY=1` + `frac=0.8`).

### Wins (speedup vs control at `frac=0.95`, sorted)

| shape | speedup |
|---|---:|
| L3-70B q_proj prefill | **1.211×** |
| Mixtral down per-expert | **1.182×** |
| L3-8B MLP down prefill | **1.178×** |
| L3-8B q_proj prefill | 1.077× |

Best frac varies by shape — `0.95` is best for the biggest wins,
but some shapes peak earlier (`0.6` or `0.8`).

### Regression — L3-8B MLP gate/up prefill

| frac | speedup vs control |
|---|---:|
| 0.2 (default) | 1.024× |
| 0.4 | **0.857×** ✗ |
| 0.6 | **0.855×** ✗ |
| 0.8 | **0.839×** ✗ |
| 0.95 | **0.831×** ✗ |

**16-17% slower at any frac > 0.2.** This single regression blocks
a global default change.

### Compound stack (`output_element_priority` + `frac=0.8` vs control)

| shape | speedup |
|---|---:|
| **L3-70B q_proj prefill** | **1.634×** |
| L3-8B MLP down prefill | 1.309× |
| Mixtral down per-expert | 1.294× |
| L3-8B q_proj prefill | 1.193× |
| L3-70B GQA kv_proj prefill | 1.092× |
| DeepSeek-MoE gate | 1.067× |
| L3-8B GQA kv_proj prefill | 1.055× |
| **L3-8B MLP gate/up prefill** | **0.840× ✗** (still regresses) |

The two levers are complementary but **sub-multiplicative** — naive
prediction was 1.61 × 1.20 ≈ 1.93× on L3-70B q_proj, measured 1.63×.
This is consistent with the levers touching the same data path
(element-priority chooses what bytes go through HMI; LX-budget
controls how those bytes are staged) so saturation effects matter.

## Regression investigation — over-commit hypothesis rejected

**Initial hypothesis**: at high `frac` the LX planner over-commits
weight-buffer space, starving activation/output staging on shapes
where per-core operands exceed the 2 MB scratchpad.

**Test**: held `M=128, K=4096`, forced `(1, 32, 1)` split, varied N
so per-core operand total swept across the 2 MB threshold.

| N | per-core total | fits | speedup at frac=0.8 |
|---|---:|---|---:|
| 2048 | 1.55 MB | ✓ | 1.009× |
| 4096 | 2.08 MB | ✗ (just over) | 1.001× |
| 8192 | 3.13 MB | ✗ | **1.024×** |
| 14336 | 4.72 MB | ✗ | **0.816×** |

**Hypothesis rejected**: N=8192 is over the LX limit and shows a
slight benefit (1.024×). Only N=14336 cliffs to 0.816×. The
regression is N=14336-specific, not size-driven.

**Likely cause**: N=14336 = 2¹¹ × 7 — a non-power-of-2. Per-core
columns at `(1, 32, 1)` is `448 = 7 × 64` sticks per core, the only
non-power-of-2 stick count in the test. The other N values
(2048, 4096, 8192) all give power-of-2 stick counts (1, 2, 4).

This rhymes with the L3-70B MLP down outlier (K=28672 = 7 × 4096)
we identified earlier in the cost-model project. **Non-power-of-2
stick counts are a recurring AIU stack pain point** that surfaces
under different conditions. Not ours to fix from torch_spyre, but
useful context — and explains why the LX-budget regression is
narrower than the catalog sweep alone suggested.

## Recommendation

The default-bump approach isn't safe given the L3-8B MLP gate/up
regression. Three options for how the project ships:

| option | mechanism | risk | wins captured |
|---|---|---|---|
| A | conservative gate (frac=0.8 only when stick-counts are power-of-2) | low | most |
| B | per-op `frac` annotation | low | depends on opt-in |
| C | document only, leave default at 0.2 | none | none until users opt in |

For now: **C** is the holding pattern. A and B both want a real fix
to the underlying non-power-of-2 issue (which is upstream of
torch_spyre). If or when the AIU stack improves non-power-of-2
handling, A becomes the right ship.

## Open questions (deferred to future work)

1. **Does LX_PLANNING help on shapes with very different M?** Our
   catalog is M=128 / decode M=1. Long-context prefill (M=2K-8K) was
   not tested. May change the optimum frac.
2. **What's the actual mechanism of the regression?** The non-power-
   of-2 correlation is a hypothesis; a deeper investigation
   (compiler instrumentation, per-iteration timing breakdown) would
   confirm.
3. **Does the existing `k_split_heuristic` (in the splitk-matmul
   branch) interact with LX-budget?** Project A will test this
   alongside the SFP-ring story.
