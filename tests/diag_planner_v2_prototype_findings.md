# Planner v2 prototype findings

Companion to `diag_planner_v2_prototype.py`. The prototype runs
offline against the same shape sweep as `diag_lx_overflow_phase1.py`
(5 production models × 4 M values × 6 matmul ops = 120 ops),
enumerates work-division candidates including the (m, 1, k)+kf
streaming family identified by Probes 4-6, ranks them via the V4
cost model, and reports where v2 would change the planner's pick.

## TL;DR

| Tier | Description | # ops | predicted savings |
|---|---|---:|---:|
| 1 — must-change | pure-M overflows C_psum | 6 | 0 ms (uncharacterised regime) |
| 2 — verify | ≥10% predicted speedup | **52** | **+94 ms total** |
| 3 — noise | <10% predicted speedup | 30 | skip |
| pure-M kept | v2 agrees with current planner | 32 | — |

Production-actionable: the 52 Tier 2 ops are candidates for
hardware verification. Median per-op savings is 1-2 ms; a few small-M
shapes show 1.5-2× speedup.

The prototype does NOT modify production code. Anywhere v2 picks a
non-default split is hardware-verifiable through the existing
`_force_split` test infrastructure used by Probes 1-6.

## Tier 1 — pure-M overflows C_psum

These shapes hit catastrophic LX overflow under the planner's
current pure-M default. v2 picks pure-N for all of them — but the
cost model gives pure-N the same prediction as pure-M (both are
k=1, both have C_psum > LX, and the catastrophic regime was
empirically characterised only at k>1).

| model | M | op | shape | pure-M | v2 pick |
|---|---:|---|---|---|---|
| L3 8B | 2048 | gate_proj | (2048, 14336, 4096) | overflows | pure-N |
| L3 8B | 2048 | up_proj | (2048, 14336, 4096) | overflows | pure-N |
| Mixtral | 2048 | gate_proj | (2048, 14336, 4096) | overflows | pure-N |
| Mixtral | 2048 | up_proj | (2048, 14336, 4096) | overflows | pure-N |
| DSv3 | 2048 | gate_proj | (2048, 18432, 7168) | overflows | pure-N |
| DSv3 | 2048 | up_proj | (2048, 18432, 7168) | overflows | pure-N |

**Open question**: does pure-N (1, 32, 1) on a C_psum-overflowing
shape run fast (no chain) or is there a separate catastrophic
regime for k=1 with n>1? Probe 4-6 only characterised k>1. A
follow-up measurement on these specific shapes would close it.

The Llama 70B+ MLP shapes are absent from this tier because they
hit the EAR ceiling (B operand > 256 MB), making most splits
uncompilable. Those are deeptools-territory.

## Tier 2 — verify-worthy: ≥10% predicted speedup

52 ops where v2 predicts ≥10% wall reduction without C_psum
overflow. Highlights:

- **Small-M (M = 32-128) + wide-N shapes**: v2 picks `(1, 4, 8)+kf`
  or `(1, 8, 4)+kf`, giving 1.5-2× predicted speedup. The mechanism
  is K-split improving PT utilisation (pure-M at M=32 has
  M_per=1, PT util = 0.125; K-split has M_per=M, util = 1.0).
- **Medium-M (M = 512) shapes**: v2 picks `(1, 16, 2)+kf`,
  predicted speedup 1.10-1.36×. Mechanism is per-cluster HMI byte
  reduction.

Confidence varies by row:

- **Rows we have validation data for** (DSv3 q_a_proj M=128 +kf,
  L3-70B kv_proj at various M): V4 cost-model error is ≤ 6%, so
  predictions are trustworthy.
- **Rows we don't** (most of Tier 2): predictions use cost-model
  V4 with 16% mean error. Hardware verification before any
  production rollout.

The biggest predicted speedups (1.5-2× at M=32) are exactly where
the V4 cost model has its biggest residual (small-M HMI BW under-
modelled). They might be smaller in practice — possibly
significantly. But they shouldn't go away entirely; the PT-util
mechanism is structurally real, and at small M K-split's full
M_per is a large advantage over pure-M's 1-row-per-core.

## Tier 3 — within noise

30 ops with <10% predicted speedup. Skip from any production
recommendation; the cost model can't distinguish at this
resolution.

## What's not in this prototype

- **Layout / memory-binding constraints**: the production planner
  rejects splits that violate memory layout or per-tensor
  hardware-span limits. The prototype doesn't enforce these. Real
  planner integration may reject some v2 picks.
- **Compile-time validity**: the prototype filters by divisibility,
  stick alignment, and the EAR ceiling, but doesn't actually
  compile each candidate. Some may fail SDSC emission for reasons
  we haven't catalogued.
- **The (m, 1, k)+kf streaming candidate space is partially
  there**: the prototype's enumerator includes (m, 1, k) but the
  cost model V4 only applies the n=1 regime treatment when the
  candidate has n=1 explicitly. Most Tier 2 picks are n>1 K-split
  (e.g., `(1, 8, 4)+kf`), which uses regular pipe-PSUM. The n=1
  family ((m, 1, k>1)+kf) is mostly tied or worse than n>1 K-split
  in the cost model, even though Probe 4-6 showed it's the safer
  choice for high-overage cases.

## How to validate Tier 2 picks on hardware

For each Tier 2 row of interest:

1. Use `_force_split(target_split)` and `_permutation("k_fast")`
   from the existing diag-branch infrastructure (see
   `diag_emission_aware_lx_p4_streaming_path.py` for the pattern).
2. Compile with the v2 pick + the existing planner's pure-M.
3. Compare measured walls. If v2 prediction matches measurement
   within ±15% AND v2 wall < pure-M wall, the row validates.
4. After 5-10 validated rows, propose planner integration to the
   torch_spyre team with the empirical evidence.

A single-shape, full-Tier-2 verification run takes 15-30 minutes
on hardware. Doing it on a representative subset (one shape per
shape-family) should be feasible in 1-2 hours.

## Path to production

The prototype is a **pre-PR design artifact**, not a planner
replacement. Path to a production change:

1. **Measure**: hardware-verify Tier 2 picks on a representative
   subset (5-10 ops). Produce a measured-wall table.
2. **Refine**: if any verified picks are slower than predicted,
   refine the cost model's V4 calibration before proceeding.
3. **Plan integration**: design a planner change that takes the
   shape, enumerates candidates per the prototype, and ranks via
   cost-model V4. Wire into `torch_spyre._inductor.core_division`
   as an extension to `multi_dim_iteration_space_split`.
4. **Verify production**: run the existing planner test suite
   plus a transformer-block end-to-end benchmark. Ensure no
   regression on shapes where v2 keeps pure-M.
5. **Land**: the planner change is a modest PR (~200 lines)
   plus the cost-model code already on this branch.

## Files

- `tests/diag_planner_v2_prototype.py` — prototype
- `tests/diag_planner_v2_prototype_results.txt` — tiered output
- `tests/cost_model_v4_findings.md` — V4 cost-model details
- `tests/emission_aware_lx_consolidated_findings.md` — overall
  project consolidation
- This doc
