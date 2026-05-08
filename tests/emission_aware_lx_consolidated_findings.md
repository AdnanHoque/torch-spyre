# Emission-aware LX scheduling — consolidated findings

Status: investigation complete on this branch. Cost-model Fix A
landed; Fixes B/C/D and planner integration are concrete next-step
deliverables but not yet implemented.

This is the throughline document. Per-phase docs (`*_phase*_findings.md`,
`p1_kscan` through `p6_chain_regimes`) carry the per-step detail; this
doc is the narrative.

## What we set out to do

Track 2 Phase 1 of the cost-model investigation found a smoking gun:

> L3-70B kv_proj M=2048 (1, 16, 2). Same shape, same split, same
> 16 MB per-core A footprint (8× over LX). +kf measures 3.94 ms (no
> detectable LX cost), +id measures 10.93 ms (real ~7 ms cost).

That row pair suggested the AIU's `k_fast` core-id permutation might
do more than reduce SFP-ring hops — that adjacent-collaborator
placement might enable per-chain *operand* reuse via the data ring.
If true, K-collaborator chains would form a virtual large-memory
window of `k × LX_per_core`, and the planner could pick splits
whose per-core operand footprint exceeds LX on the assumption
the chain mediates the access. This was the **chain-cooperative
LX residency** hypothesis (M1).

We set out to test M1 against two alternatives (M2 = kernel-template
fast path; M3 = PSUM forward-pipelining), then map the consequences
into the cost model and planner.

## What we actually found

The chain-cooperative LX hypothesis (M1) is **refuted**. The data
ring isn't doing operand multicast; the kf benefit is purely about
SFP-ring K-collaborator distance. But the probes uncovered a
**different** real mechanism that's more practically useful:

> The work-division split space has a hidden M-vs-N asymmetry.
> Splits that divide N (`(1, n>1, k)`) have catastrophic LX-overflow
> cost when per-core PSUM accumulator > 2 MB. Splits that divide M
> instead (`(m, 1, k)`) trigger a streaming-output kernel-template
> path that absorbs the same PSUM overage with no penalty. The
> asymmetry can be 7-15× wall on the same shape with the same
> overage and the same chain length.

This M-vs-N asymmetry has three layers, all empirically grounded:

1. **The binding LX constraint is the PSUM accumulator, not operand
   A.** The original A-side gate (Project B Phase 0, LX Phase 0) was
   checking the wrong operand. Fix A in `tests/lx_fit.py` corrects
   this.
2. **The streaming-output fast path activates when `n = 1`** in any
   `(m, 1, k)` split, not just pure-K. Same C_psum overage, n=1
   absorbs cleanly while n>1 is catastrophic.
3. **The streaming path has three internal regimes** mediated by
   chain length: pipeline (chain ≤ 4, +3 ms cost), sync (chain
   8-16, +25-55 ms cost), allreduce (chain = 32, +14 ms cost). The
   chain=4 → chain=8 boundary is sharp and universal across shapes.

Crossing all three layers, the planner's choice space is no longer a
scalar "pick the split that minimises HMI bytes". It's a structured
decision tree:

  - n>1 + C_psum > LX → catastrophic, never pick
  - n=1 + chain ≤ 4 → pipeline, 3 ms overhead
  - n=1 + 4 < chain < 32 → sync, big overhead
  - n=1 + chain = 32 → allreduce, moderate overhead
  - everything else → close to compute/HMI baseline

There's also a **hardware ceiling** outside this structure: a 256 MB
per-core EAR limit blocks the streaming path for shapes where the B
operand (K × N × dtype) exceeds 256 MB. For the largest models
(Llama 70B+ MLP layers at M=2048), even the streaming path is
unavailable, leaving only catastrophic options.

## The probe sequence

Six probes, run on `AdnanHoque/diag-core-ordering` (which has the
`core_id_permutation` config infrastructure). Branch
`AdnanHoque/emission-aware-lx-phase0` carries the design docs,
findings, and the cost-model artifacts.

| probe | question | finding |
|---|---|---|
| 1 (k-scan) | Does kf wall plateau at chain-LX threshold? | **No.** M1 refuted on 2 shapes. |
| 2 (permutation) | Is the kf benefit cohort-clustering or K-collab distance? | **Pure distance.** ~5.6 ms/hop linear; 3 distance=1 perms within 0.2 ms. |
| 3 (N-axis) | Where does the mid-k catastrophe transition? | **C_psum = LX exactly.** ~17 ms/overage factor above. |
| 4 (m,1,k) | Is the (1,1,32) anomaly a single-chain or n=1 effect? | **n=1 triggers it.** Same overage, n=1 vs n>1: 7× wall. |
| 5 (generality) | Does n=1 fast path generalise to other wide-N shapes? | **Yes, within 256 MB EAR ceiling.** Beyond: blocked. |
| 6 (regimes) | What's the chain-length wall structure within n=1? | **Three regimes.** Pipeline / sync / allreduce, universal boundaries. |

Each probe was designed to disambiguate competing hypotheses, not
just collect data. Probes 1, 2, 3 ran on the cost-model-failure
shapes from the validation set; Probes 4, 5, 6 followed up on the
mid-k anomaly that Probe 1 surfaced.

## Implications for the cost model

`tests/hmi_cost_model.py` had three structural bugs going into this
investigation:

1. **HMI bytes used full-broadcast formula `M·K + K·N + M·N`** for
   all splits. Per-cluster `(M·K + K·N)/k + M·N` is correct under
   K-split. (Track 2 Phase 0 fix.)
2. **LX-fit predicate gated on operand A.** PSUM accumulator is the
   binding constraint. (Fix A, landed in this branch.)
3. **PSUM term used pipe model** with hops_per_send = m·n for
   identity emission. Actual cost is roughly linear in
   K-collaborator ring distance, with calibration constant ~5.6
   ms/hop (shape-dependent slope). (Fix C, not yet implemented.)

Plus:

4. **No PSUM-overflow penalty term.** Catastrophic-regime rows
   under-predict by 46-99%. Calibration: ~17 ms per overage factor
   when n>1; ~3 ms / 25 ms / 14 ms additive when n=1 with chain
   ≤ 4 / 8-16 / =32. (Fix B + Fix D, not yet implemented.)
5. **HMI achieved BW under-modelled at small M.** Implied 128 GB/s
   on DSv3 o_proj M=32 vs cost-model 40 GB/s. Independent residual,
   needs hardware calibration. (Not addressed in this project.)

After Fixes A/B/C/D are layered, the validation set residuals should
drop from V0 21.7% mean error to single-digit on most rows. The
remaining residual is the small-M HMI BW issue.

## Implications for the planner

The planner today picks `(32, 1, 1)` for nearly every matmul. This
is correct for shapes where pure-M fits LX (most decode-regime
matmuls, some prefill). It's catastrophically wrong for some
prefill shapes that the LX-Phase-1 diagnostic flagged as overflow
candidates.

The new candidate space, with preference order:

1. **Pure-M `(32, 1, 1)`** when `M_per × N × dtype_psum ≤ LX`.
   Default. Correct on the bulk of production traffic.
2. **`(16, 1, 2)+kf` or `(8, 1, 4)+kf`** when pure-M overflows but
   the streaming fast path's pipeline regime is reachable
   (chain ≤ 4 and B ≤ 256 MB). 1.3-1.5× pure-M baseline; the
   right second choice.
3. **`(1, 1, 32)+kf`** when the above don't fit; allreduce regime.
   ~2× pure-M baseline; acceptable.
4. **`(1, n>1, k>1)+kf` and `(4-2, 1, k≥8)+kf`**: avoid. Sync regime
   or catastrophic regime.

Plumbing this into `torch_spyre._inductor.core_division` would be
a concrete production change. It needs hardware verification on a
broader shape set than we measured here — but the preference order
is defensible from the data we have.

## What's outside torch_spyre

The 256 MB EAR per-core limit is a hardware/deeptools constraint.
Wide-N prefill on Llama 70B+ MLP layers is in a regime where:

- Pure-M overflows C_psum (catastrophic)
- (m, 1, k) blocked by EAR limit
- (1, n, k>1) catastrophic via PSUM-overflow

There's literally no good split for these shapes today. This is a
deeptools / kernel-template request, not a planner fix. Worth
raising with empirical numbers (Probe 5 has them).

The chain=4 → chain=8 boundary in the streaming fast path is also
a kernel-template (not torch_spyre) detail. The boundary's
universality suggests it's a real codegen path selector. Reading
the SDSC emitter source would settle whether it's
buffer-size-driven, code-path-selector-driven, or topology-driven.

## What's publishable

The contribution is a cost-model term and a planner extension that
no public auto-scheduler models:

> **Per-PE accumulator residency, conditional on which output
> dimension is split, mediated by the kernel template's
> streaming-output behaviour.**

To my knowledge, no published auto-scheduler (Roller, Ansor, AKG,
TVM, MLIR's polyhedral schedulers) models per-PE accumulator
residency at all, let alone conditional on output-dim split
structure. On AIU 1.0 it's a 7-15× wall lever on real production
shapes.

The paper writes itself: characterise the n=1 streaming-output
regime + chain-length sub-regimes, derive the cost-model term, show
the planner wins on production shapes, position relative to public
auto-schedulers' per-operand residency models.

The three-regime structure within the n=1 path (pipeline / sync /
allreduce) is the empirical anchor — it's universal across shapes
and the boundaries are sharp.

## Concrete next-step deliverables

In rough effort order:

1. **Implement Fix B/C/D in `hmi_cost_model.py`** (regime-routed
   PSUM-overflow penalty + ring-distance PSUM term + per-cluster
   bytes from Track 2 Phase 0). Days of work.
2. **Re-run validation** with all four fixes layered. Target:
   single-digit mean error on the 30-row set.
3. **Wire C-PSUM gate + (m, 1, k)+kf candidate space into the
   planner** in `torch_spyre._inductor.core_division`. Production
   change; needs hardware verification before merge.
4. **Surface the EAR ceiling to the deeptools team** with Probe 5
   numbers. Possible feature request: extend EAR limit, or add a
   tile-streaming path at the hardware level for K × N > 256 MB.
5. **Read SDSC emitter source** to characterise the chain=4 →
   chain=8 boundary and the chain=32 allreduce primitive. May
   inform whether the planner can request the allreduce path
   explicitly.

(1) is purely cost-model code with no hardware dependency. (2) is
a re-validation step. (3) is the production lever. (4) and (5) are
cross-team work.

## Branch layout

| branch | role |
|---|---|
| `AdnanHoque/lx-residency-planner-phase0` | LX gate (A-side, wrong predicate, kept as historical artifact) |
| `AdnanHoque/cost-model-track2-kfast-residual` | Per-cluster bytes + 3-mechanism diagnosis |
| `AdnanHoque/emission-aware-lx-phase0` | This investigation; carries Fix A and all probes |
| `AdnanHoque/diag-core-ordering` | Hardware-execution branch; probes ran here |

`emission-aware-lx-phase0` is the canonical branch for this work.
The probe scripts were copied to `diag-core-ordering` for HW runs;
the artifacts (results files + findings docs) live on
`emission-aware-lx-phase0`.

## Quick reference

| file | what |
|---|---|
| `tests/lx_fit.py` | Updated predicate (gates on PSUM accumulator) |
| `tests/diag_lx_overflow_phase1.py` | Validation/production diagnostic with corrected predicate |
| `tests/diag_emission_aware_lx_phase0_findings_v2.md` | Probe 1-3 findings (mechanism diagnosis) |
| `tests/diag_emission_aware_lx_p4_findings.md` | n=1 streaming-output fast path discovery |
| `tests/diag_emission_aware_lx_p5_findings.md` | Generality + EAR ceiling |
| `tests/diag_emission_aware_lx_p6_findings.md` | Three-regime structure within n=1 |
| `tests/diag_lx_overflow_phase1_findings.md` | LX Phase 1 (Fix A) results |

This doc is the index.
