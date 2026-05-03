# Spyre matmul perf investigation — session summary

A consolidated summary of the multi-week investigation into work-
division and adjacent levers for Spyre matmul performance. Captures
what was shipped, what was tried and closed, the corrected mechanism
understanding, and the meta-pattern that should shape the next phase.

## Headline contribution

**`output_element_priority` heuristic** — a one-line fix to the
planner's priority ranking that produces:

- **1.61× peak speedup** on Llama-70B q-projection prefill
- **1.12× geomean** across 13 production prefill matmul shapes
- **Zero regressions** anywhere in the catalog

Lives on `AdnanHoque/diag-cost-model-planner` branch, commit
`e0fb7ef`, ready for PR to `main`. Validation: 12 hardware-free unit
tests + on-Spyre smoke test + full catalog sweep. Findings:
[`element_priority_theory.md`](element_priority_theory.md), comparison
table: [`diag_element_priority_compare_results.md`](diag_element_priority_compare_results.md).

The mechanism: the default planner ranks output dims by stick-adjusted
size, which artificially deflates stick-dim N (e.g. N=4096 elements →
64 sticks for fp16) below non-stick M=128. The fix ranks by element
count instead.

## Project ledger

| project | hypothesis | outcome | findings doc |
|---|---|---|---|
| `output_element_priority` | planner ranks output dims wrong | **shipped** — 1.12× geomean | [element_priority_theory.md](element_priority_theory.md) |
| Cost model | predict-best-split would beat greedy | model too coarse — 6% mean regret, 36% max | [cost_model_phase1_0_findings.md](cost_model_phase1_0_findings.md) |
| Core-ordering reorder | dual-ring topology might want different mapping | dead at all scales (1.002× max) | (in `AdnanHoque/diag-core-ordering` branch commits) |
| LX scratchpad budget | `DXP_LX_FRAC_AVAIL` default too low | 1.20× peak, 1.63× compound w/ EP, but 1 shape regresses 16% | [lx_scratchpad_budget_findings.md](lx_scratchpad_budget_findings.md) |
| K-split / PSUM | SFP ring is dedicated, K-split underused | 1.13× on 1 production shape pattern; pure-K never wins | [psum_split_findings.md](psum_split_findings.md) |
| Bidirectional ring | maybe codegen uses only 1 of 2 data rings | no lever exposed at torch_spyre layer; closed by code reading | [bidirectional_ring_findings.md](bidirectional_ring_findings.md) |

## Corrected mechanism understanding

Key facts learned (some from the IBM AIU architecture doc, some from
measurement) that update what was previously hand-wavy:

1. **The on-chip interconnect is two counter-rotating data rings**
   (CW + CCW, 128 B each), plus a separate 32 B SFP ring for psum
   reduction. Dual rings exist but ring-direction selection is
   abstracted away from torch_spyre.

2. **HMI (the DRAM interface) sits on the same data ring as cross-
   core sharing.** This is the single most important fact for
   understanding bottlenecks: cross-core operand sharing competes
   with DRAM streaming for the same ring bandwidth. The original
   "67 GB/s per-link" measurement combined both effects.

3. **Pure ring-share is ~88 GB/s per direction.** Measured by
   isolating LX-resident operands. HMI contention adds ~24% on top
   of pure ring transit when DRAM streaming is concurrent.

4. **Kernel templates are well-engineered.** Overlapped input fetch
   (slide 113), chunked double-buffering, SFP-ring psum routing —
   these absorb a lot of what looked like "planner-layer slack" from
   our projects. The reorder probe being decisively flat was the
   strongest signal of this.

5. **Cross-call weight preload (slide 86 of doc) doesn't fire for
   `torch.compile`-driven matmul.** Our LX-budget probe showed
   first-iter == median across all configs — every kernel call
   re-streams weights from DRAM. The preload mechanism exists in the
   AIU stack but lives in a runtime path torch_spyre's lazy compile
   doesn't trigger. This is a known unexploited lever.

6. **Non-power-of-2 stick counts are a recurring AIU stack pain
   point.** Surfaces as the L3-70B MLP down outlier (K=28672 = 7·4096),
   the L3-8B MLP gate/up regression at high LX frac (N=14336 → 7
   sticks per core), and probably more places we haven't seen. Not
   ours to fix from torch_spyre, but worth flagging.

## Meta-pattern: where the wins live

Six projects investigated, one shipped. The pattern of outcomes is
informative:

| project | layer | outcome |
|---|---|---|
| `output_element_priority` | planner / priority logic | broad win |
| Cost model | planner / predictor | model too coarse |
| Core-ordering | planner / codegen | dead |
| LX scratchpad budget | runtime config | mixed (1 shape regresses) |
| K-split / PSUM | planner / split selection | very narrow |
| Bidirectional ring | runtime / hardware | hidden at our layer |

**The big bug fix at the planner layer (element_priority) was a one-
shot.** Subsequent planner-layer levers are progressively narrower,
and the lower-layer levers (preload, ring direction) are hidden
behind abstractions torch_spyre doesn't control.

**The most productive next phase is probably outside the planner
layer.** Three categories worth considering:

### (a) Runtime / driver pathway
Cross-call weight preload is documented to exist but doesn't fire for
torch.compile. If we can figure out why and enable it, the wins are
large because they apply to every repeated kernel call, not per-shape.
This is a code-reading + runtime-investigation project, not a planner-
optimization project.

### (b) Inductor scheduler / cross-op
Today's planner picks splits per-op. A Llama prefill block has 7 ops
in series; cross-op optimization (HMI-aware scheduling, fusion of
adjacent matmul, scheduled weight prefetch) could exploit slack the
per-op view doesn't see. Hardest to scope, biggest ceiling.

### (c) Op fusion / graph rewrites
Fused FFN, fused QKV-projection, decoder-block fusion. Pattern-rich
territory in GPU-land that translates here. Some of this is already
covered by the parallel flash-attention work.

## Branch / artifact navigation

| branch | what it has |
|---|---|
| `AdnanHoque/diag-cost-model-planner` | `output_element_priority` (shipped), cost model (not shipped), all topology / LX / preload / K-split probes |
| `AdnanHoque/diag-core-ordering` | core-ordering reorder (closed), plus cherry-picked element_priority and ALL the post-EP probes (LX-budget, K-split, K/N ratio, bidirectional ring writeup) |

Most-recent work lives on `AdnanHoque/diag-core-ordering`. The
shipping PR for `output_element_priority` should come from
`AdnanHoque/diag-cost-model-planner` (cleaner branch, no negative-
result accumulations).

## What I'd recommend doing next

In rough order of value-per-effort:

1. **Open the `output_element_priority` PR to main.** This is the
   one shipping deliverable. Don't let it sit in a branch.
2. **Cross-call preload investigation** as the next big project. The
   one project where we have direct evidence the lever exists but
   isn't engaged on our code path. If we can route torch.compile-
   driven matmul through the preload pathway, the wins compound with
   every other shipped lever.
3. **(In parallel)** start scoping cross-op / Inductor-scheduler work
   if the preload investigation has unbounded scope.

Things to deprioritize:
- Further planner-level split-selection projects (diminishing returns
  as shown by the K-split outcome)
- More cost-modeling work (the simple model didn't generalize, and
  more complex models risk overfitting on 13 shapes)
