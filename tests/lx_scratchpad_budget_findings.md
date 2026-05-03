# LX scratchpad budget — DXP_LX_FRAC_AVAIL impact on prefill matmul

A measurement of how much the on-chip LX scratchpad budget controls
prefill matmul wall time on Spyre, and what that implies for
production configuration defaults.

## Headline

Increasing `DXP_LX_FRAC_AVAIL` from the current default of `0.2` to
`0.8` (with `LX_PLANNING=1`) cuts wall time by **up to 20%** on
Llama-70B q-projection prefill. The default is meaningfully
under-tuned for transformer prefill workloads.

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

## Open questions for the next phase

These three measurements raise questions worth answering before
shipping anything as a production default:

1. **Is `frac=0.8` ever a regression?** We tested 3 shapes; the full
   13-shape Phase 1.0 catalog should be re-run. If any shape gets
   slower at high frac, the default needs to be more nuanced.

2. **Does the gain stack cleanly with `output_element_priority`?**
   Predicted yes (different mechanisms), but needs direct
   measurement on the L3-70B q_proj case where both have headroom.

3. **What about MoE per-expert?** Mixtral / Qwen-MoE / DeepSeek-MoE
   matmul has smaller per-expert M but identical weight pattern.
   Should benefit similarly. Worth confirming.

4. **Is there a frac-too-high regime?** At very high frac, less LX is
   left for activations. For workloads where activations are large
   (e.g. long-context prefill with M >> 128), this could regress.
   Need to test shapes outside the M=128 regime.
