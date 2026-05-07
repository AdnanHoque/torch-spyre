# Emission-aware LX scheduling — Phase 0 scope

Project root question:

> **Does AIU 1.0's k_fast core-id permutation reduce effective HMI
> bytes by routing operand chunks through the inter-core data ring,
> turning a chain of K-collaborators into a virtual large-memory
> window that spans more than per-core LX?**

If yes, the work-division planner has a memory lever it isn't using.
Today's cost models (Roller, TVM auto-scheduler, etc.) treat per-PE
memory as an independent constraint. A chain-cooperative residency
pattern would let the planner pick splits whose per-core working set
overflows LX, on the assumption that the chain mediates the access.

This is solo torch_spyre work — the planner picks the split, the
emission patches the core_id permutation, both touch
`torch_spyre._inductor` and have no deeptools dependency.

## Empirical anchor

The Track 2 Phase 1 finding that motivates this project:

> L3-70B kv_proj M=2048 (1, 16, 2). Same shape, same split, same
> 16 MB per-core A footprint (8× over LX).
> +kf measures 3.94 ms (no detectable LX cost).
> +id measures 10.93 ms (real ~7 ms cost).

The 3 ms LF + 0.74 ms HMI + ~3 ms PSUM (id 16-hop term) covers about
6.7 ms of the +id wall. The +kf wall sits below the 3 ms HMI+LF
floor for the per-cluster bytes formula, and there's no LX
overflow penalty visible. Two facts we must reconcile:

1. The +kf prediction at A_per = 16 MB *should* show LX overflow
   under any naive per-core model (8× re-fetch).
2. It doesn't.

## Three candidate mechanisms

### M1 — data-ring operand multicast (chain-cooperative LX)

Same-k-cohort cores (those needing the same A slice in a (1, n, k)
split) sit at clustered ring positions under k_fast. The Data
QuadRing multicasts A from one HMI fetch to the cohort. Effective
HMI bytes for A are M×K (read once across the full chain) instead of
M×K × num_consumers (re-fetched per consumer). The chain length k
governs how the M×K traffic is divided.

**Distinguishing prediction**: walls scale with `A_per_core / k`,
not `A_per_core`. A k=4 split has 4× the chain-LX of a k=2 split
at the same shape, so wall stays flat across the LX-overflow band
until `A_per_core / k > LX`.

### M2 — kernel-template fast path

Compiler detects k_fast core-id pattern at codegen time and emits a
different M-K-N tile loop where A is streamed K-tile by K-tile,
keeping the working set at `M_per × K_tile` (always fits LX).

**Distinguishing prediction**: only the *exact* k_fast permutation
triggers the fast path. Any other permutation that places K-cohort
cores adjacent (e.g., a cohort-clustering permutation built by hand)
should NOT match k_fast walls — the kernel template doesn't know
about it.

### M3 — PSUM forward-pipelining

Adjacent placement under k_fast lets compute and PSUM accumulation
overlap (core *k_0* receives partial sums from *k_1* before its own
K-loop finishes). This pipelines compute-and-receive, hiding A
re-fetch cost.

**Distinguishing prediction**: the kf benefit collapses as PSUM
payload grows, because pipelined compute can't catch up to PSUM at
high `M_per × N_per`. Already partially observed in
`diag_kfast_high_k_and_zorder_k1_findings.md`, where +kf benefit
shrinks at k=4 vs k=2.

## Phase 0 deliverable

Three measurement probes designed to discriminate M1 / M2 / M3.
Each probe uses the diag-branch infrastructure (`_force_split` +
`_emission` context managers) and the existing PR 1932 / PR 1933
codegen on a branch that has `core_id_permutation` config.

### Probe 1 — chain-LX scaling under +kf

For each of three shapes, sweep k ∈ {1, 2, 4, 8, 16, 32} (with
n = 32/k, m = 1) under k_fast emission. Record wall + back out
implied LX overage relative to chain-LX = k × 2 MB.

Shape matrix:

| shape | (M, N, K) | A_per at k=2 | chain-LX threshold k |
|---|---|---:|---:|
| DSv3 o_proj M=2048 | (2048, 7168, 16384) | 32 MB | k ≥ 16 fits |
| L3-70B kv_proj M=2048 | (2048, 1024, 8192) | 16 MB | k ≥ 8 fits |
| Mixtral kv_proj M=2048 | (2048, 1024, 4096) | 8 MB | k ≥ 4 fits |
| (control) DSv3 q_a_proj M=128 | (128, 1536, 7168) | 0.7 MB | always fits |

**M1 prediction**: wall stays flat or decreasing as k increases past
the chain-LX threshold (overflow disappears once chain-LX > A_per).
There should be a discernible inflection where overage transitions
through 1.0.

**M2 prediction**: wall stays flat across all k (kernel template
streams regardless of overage).

**M3 prediction**: wall increases at high k because PSUM payload per
chain grows with N_per = N/n = N×k/32, and PSUM cost scales with
payload.

Two of these have very different curve shapes; the data should
disambiguate cleanly.

### Probe 2 — permutation discriminator

For one shape from Probe 1, run (1, 16, 2) under three emission
configs:

- **id default**: K-collaborators 16 hops apart (control)
- **id + manual cohort-clustering permutation**: arbitrary permutation
  that places same-k-cohort cores adjacent (positions 0..15 for k=0,
  16..31 for k=1) — *not* the k_fast permutation but with the same
  cohort property
- **kf**: k_fast permutation

The diag-branch arbitrary-permutation infrastructure (see
`diag_2d_direction_probe.py`) supports specifying permutations
directly.

**M1 prediction**: id-default slow, both permutations fast (and ≈
each other).
**M2 prediction**: id-default slow, manual permutation slow (kernel
template doesn't recognise it), kf fast.
**M3 prediction**: id-default slow, both permutations match
according to PSUM hop count alone (kf=1 hop, manual depending on
arrangement).

Probe 2 cleanly separates M1 from M2.

### Probe 3 — chain-LX overage threshold

For (1, 16, 2)+kf at varying M (32, 128, 256, 512, 1024, 2048,
4096), record A_per_core and wall. The chain-LX (per M1) is
2 × LX = 4 MB. So:

| M | A_per | overage_factor (per M1) | M1 prediction |
|---:|---:|---:|---|
| 32 | 256 KB | 0.06× | flat |
| 128 | 1 MB | 0.25× | flat |
| 256 | 2 MB | 0.5× | flat |
| 512 | 4 MB | 1.0× | inflection here |
| 1024 | 8 MB | 2.0× | wall starts climbing |
| 2048 | 16 MB | 4.0× | clear overhead |
| 4096 | 32 MB | 8.0× | strong overhead |

**M1 prediction**: knee at M = 512 (overage = 1×). Wall flat below,
climbing above.
**M2 prediction**: monotonic with M (compute scales), no knee.
**M3 prediction**: monotonic with M (PSUM payload scales), no knee.

Existing validation has 2 of 7 rows (M=128 and M=512). Probe 3
fills in the M-axis to pin down where the chain-LX threshold sits.

## What "success" looks like

Phase 0 closes if the three probes agree on which mechanism is
operative. Expected outcomes:

- **M1 confirmed** → Phase 1 = extend cost model with chain-residency
  term; Phase 2 = identify production shapes where K-split+kf wins
  through chain-LX; Phase 3 = explore k=4, k=8, k=16 splits the
  current planner ignores.
- **M2 confirmed** → Phase 1 = document the kernel template's
  fast-path conditions in the cost model; smaller scope, no novel
  research, but useful for planner accuracy.
- **M3 confirmed** → Phase 1 = the PSUM aggregate-link work from
  Track 2 Phase 1 already covers most of this; close project and
  return to Roller-on-AIU.
- **Mixed signals** → triage which is dominant, write a more
  targeted Phase 0.5 to isolate.

## What's at stake (research framing)

If M1 is the mechanism, the contribution is a new constraint axis
for accelerator auto-schedulers:

> **Per-chain memory residency as a function of split topology and
> emission permutation, with the chain length serving as a tunable
> memory-multiplication factor.**

Public auto-schedulers (Roller, AKG, TVM Ansor) and accelerator
planners (TPU, Cerebras, Groq) all model memory as a per-PE
constraint. None of them treats *which-PE-talks-to-which-PE* as a
memory-system parameter. The AIU's 5-ring topology with the
matmul-template's PSUM chain provides a natural place where the
arrangement *is* the memory budget.

The published contribution would be:

1. The chain-cooperative residency mechanism on AIU 1.0.
2. A cost model that captures it (probably: HMI bytes are charged
   per-chain, not per-core, when same-k-cohort permutations apply).
3. A planner that uses chain length as a memory-headroom lever.
4. Empirical wins on production transformer-block shapes that
   today's planner can't pick because they violate per-core LX.

If M2 or M3 is the mechanism, the result is still useful (planner
fix, cost-model accuracy lift) but lower-novelty.

## Files (planned)

- `tests/diag_emission_aware_lx_p1_kscan.py` — Probe 1
- `tests/diag_emission_aware_lx_p2_permutation.py` — Probe 2 (stub
  until we have arbitrary-permutation API in this branch lineage)
- `tests/diag_emission_aware_lx_p3_overage_threshold.py` — Probe 3
- `tests/diag_emission_aware_lx_phase0_findings.md` — written after
  measurements come back

## Cross-references

- `tests/diag_kfast_residual_phase1_findings.md` — Track 2 Phase 1,
  the smoking-gun row pair
- `tests/diag_lx_overflow_phase0_findings.md` — LX gate scope
- `tests/diag_pr1932_top10_replication_findings.md` — top-10 +kf
  replication that anchors the kf-emission machinery
- `tests/diag_kfast_high_k_and_zorder_k1_findings.md` — earlier
  k>2 measurements; M3 evidence
