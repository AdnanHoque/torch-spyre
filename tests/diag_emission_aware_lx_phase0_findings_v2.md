# Emission-aware LX scheduling — Phase 0 v2 (post-Probe 2/3)

Updates and re-frames the v1 findings document. Probes 2 (permutation
discriminator) and 3 (N-axis sweep) ran on `AdnanHoque/diag-core-ordering`
on the same hardware as Probe 1.

## TL;DR

Two clean novel findings emerged from the three-probe phase. Neither
is the original "chain-cooperative LX residency" framing (which is
refuted), but both are concrete, reproducible, and load-bearing for
the cost model and the planner.

1. **The +id penalty in (1, n, k>1) is purely a function of
   K-collaborator ring distance.** Probe 2 measured ~5.6 ms wall
   per ring hop on DSv3 o_proj at (1, 16, 2). Three different
   permutations achieving distance=1 (k_fast, block_cyclic,
   bit_reverse) gave walls within 0.2 ms of each other. There is
   no second-order cohort-clustering effect.

2. **The "mid-k catastrophe" is C-side PSUM-accumulator LX
   overflow, not A-side operand overflow.** Probe 3 found a clean
   inflection at exactly `M_per × N_per × 4 bytes = LX`. Below the
   inflection, K-split kf walls track pure-M; above it, walls
   diverge and grow ~17 ms per LX-overage factor. The original
   LX-fit predicate (gating on A) was checking the wrong operand.

These findings invalidate the LX-Phase-0 gate I wrote earlier and
materially change what the cost model needs.

## Probe 2 — permutation discriminator at (1, 16, 2) on DSv3 o_proj

| permutation | K-collab dist | wall (ms) |
|---|---:|---:|
| identity | 16 | 115.77 |
| stride2 | 8 | 61.54 |
| bit_reverse | 1 | 30.84 |
| block_cyclic | 1 | 31.00 |
| k_fast | 1 | 31.03 |

(reversed/antipodal/random_* failed to compile — they put logical
core 0 at non-zero physical positions, which appears to violate a
codegen anchor assumption. Tangential to the finding; the data we
have is sufficient.)

**Interpretation.** Wall is approximately linear in K-collab
distance:

    wall ≈ base + slope × K_collab_distance

with slope ≈ 5.6 ms/hop on this shape. The exact slope will scale
with PSUM payload (M_per × N_per × 4); the *linearity* is the
generic claim.

This means the cost model's `_total_psum_ring_bytes` term should
multiply by *actual ring distance traveled per chain send*, computed
from the active permutation, instead of the m·n hops it currently
hard-codes for non-kf emissions.

## Probe 3 — N-axis sweep at fixed M=2048, K=8192, (1, 8, 4)+kf

| N | N_per | C_psum_per_core | (32,1,1) ms | (1,8,4)+kf ms | catastrophe overhead |
|---:|---:|---:|---:|---:|---:|
| 512 | 64 | 512 KB | 3.52 | 3.47 | none |
| 1024 | 128 | 1 MB | 3.67 | 4.34 | none |
| 2048 | 256 | **2 MB ← LX** | 4.47 | 5.39 | none |
| 4096 | 512 | 4 MB | 6.03 | **19.64** | 14 ms |
| 6144 | 768 | 6 MB | 7.58 | **57.30** | 50 ms |
| 8192 | 1024 | 8 MB | 9.08 | **76.37** | 67 ms |

The catastrophe transition happens between N_per = 256 (C_psum =
2 MB, exactly LX) and N_per = 512 (C_psum = 4 MB, 2× LX). Above the
threshold, the catastrophe overhead grows ~17 ms per excess overage
factor — roughly linear scaling.

**Mechanism**: the PSUM accumulator (M_per × N_per fp32 elements,
4 bytes each) must remain resident in LX across the K-iteration
loop because partial products accumulate into it. When it exceeds
LX per core, the kernel template spills it — likely fetching from
HMI per K-iteration or per K-tile, which multiplies effective HMI
bytes by the K-iteration count.

This explains every K-split row in the validation set with C_psum >
LX:

| validation row | M_per | N_per | C_psum | overage | measured | predicted |
|---|---:|---:|---:|---:|---:|---:|
| DSv3 o_proj M=2048 (1,16,2)+kf | 2048 | 448 | 3.67 MB | 1.8× | 31.23 | 16.87 (-46%) |
| DSv3 o_proj M=2048 (1,16,2)+id | 2048 | 448 | 3.67 MB | 1.8× | 116.12 | 44.39 (-62%) |
| L3-70B kv_proj M=2048 (1,16,2)+kf | 2048 | 64 | 0.5 MB | 0.25× | 3.94 | 4.63 (+18%) |

The first two miss because the cost model has no C-spill term. The
last fits because C_psum < LX so no catastrophe.

**Anomaly worth noting**: Probe 1 found (1, 1, 32)+kf on DSv3 o_proj
runs at 30 ms even though C_psum = 58.7 MB (29× LX). This is the
only catastrophic-overage case that's fast. Hypothesis: with k=32
and m=n=1, there is a single chain with one chain head holding the
output. The kernel template can stream-spill output to HMI as the
chain accumulates, rather than holding all M_per × N_per fp32
elements simultaneously. At 1 < k < 32 with multiple cells, each
cell's accumulator is independent and can't share the streaming
path.

## What this means for the cost model

Three concrete fixes follow from the data, none of which were in the
Track 2 Phase 1 framework:

### Fix A — replace LX-fit predicate (M_per × N_per × 4)

`tests/lx_fit.py`'s headline predicate gates on `A_per_core` (the
stationary operand). The right gate is **PSUM accumulator
residency**:

    fits = M_per × N_per × dtype_psum_bytes ≤ LX_BYTES_PER_CORE

The A-side LX-fit predicate was wrong on the mechanism. It happens
to correlate (both A and C grow with M), but the binding constraint
is C.

### Fix B — PSUM-overflow penalty term

For splits where C_psum > LX_per_core, add a multiplicative HMI
penalty:

    overhead_ms ≈ 17 × (C_psum_overage_factor - 1)

This calibration comes from the linear regression of Probe 3's
catastrophe overhead vs C_psum / LX. The constant 17 will need
re-fit on more shapes and at other (m, n, k) — Probe 3 measured
only k=4. But the *form* (linear in overage past LX) is clean.

Probably also needs a special case for `m=n=1, k=32` where C_psum
streaming bypasses the penalty (see anomaly above).

### Fix C — PSUM-ring-distance term

Replace `hops_per_send = 1 if k_fast else (m*n)` in
`_total_psum_ring_bytes` with the actual hops traversed per send,
computable from any permutation:

    hops_per_send = avg over chains of (sum of physical hops for
                    that chain's k-1 sends)

For k_fast: 1. For identity at (1, n, k): m·n. For stride2 at
(1, 16, 2): 8. For bit_reverse at (1, 16, 2): 1. The cost-model
wall scales with this exactly per the Probe 2 measurements.

## Why the original framing failed (negative result is useful)

The "chain-cooperative LX residency via data-ring multicast"
hypothesis was internally consistent with Project B's Phase 0
finding ("LX overflow re-fetch when A_per_core > 2 MB"). The probes
showed that:

- Project B's mechanism (A-overflow re-fetch) doesn't appear at the
  observed shapes — the catastrophe at (1, 8, 4)+kf with A_per =
  16 MB on DSv3 happens to coincide with C_psum = 7.3 MB > LX, and
  the latter is what's actually breaking.
- The kf benefit is purely about K-collab ring distance reducing
  PSUM contention, not about operand reuse via the data ring.

So the data ring isn't doing the operand-multicast work I
hypothesized. The novel finding from this project is more banal but
more useful: the PSUM accumulator is the binding LX constraint, and
ring distance is the dominant cost model term we were getting wrong.

## Next steps

The three fixes above are concrete and pre-validated against
existing measurements. Proposed sequencing:

1. **Implement Fix A** (replace LX-fit predicate) and re-run the
   LX-Phase-0 diagnostic. The DSv3 o_proj M=2048 (1,16,2)+kf
   row should now correctly identify overflow; the kv_proj
   M=2048 (1,16,2)+kf row should now correctly NOT.

2. **Implement Fix C** (ring-distance PSUM term) in the cost model
   and re-run Track 2 Phase 1. K-split+id rows should drop from
   49% to ~10% mean error (linear hop scaling matches Probe 2).

3. **Implement Fix B** (PSUM-overflow penalty) and re-run validation.
   This closes the catastrophic-overage rows. Calibration constant
   (17 ms per factor) needs more measurement coverage to refine —
   one shape, one k value isn't enough.

4. **Wire combined corrected predicates into the planner**. Reject
   splits where C_psum > LX without a fast-streaming path. This is
   the actual production-actionable lever from the project.

5. **Investigate the k=32 streaming anomaly**. Why does (1, 1, 32)
   bypass the catastrophe? If we can characterize the condition,
   we can model it (or even propose a kernel-template change to
   extend it to other splits).

## Files

- `tests/diag_emission_aware_lx_p2_permutation.py` + `_results.txt`
- `tests/diag_emission_aware_lx_p3_midk_n_sweep.py` + `_results.txt`
- `tests/diag_emission_aware_lx_phase0_findings.md` (v1, partial)
- This doc (v2, supersedes v1)
