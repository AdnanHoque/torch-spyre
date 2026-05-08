# LX residency planner — Phase 1 findings (Fix A applied)

Companion to `diag_lx_overflow_phase1.py`, which re-runs the Phase 0
scan with the corrected predicate (PSUM-accumulator residency
instead of operand-A residency). The corrected predicate is
implemented in `tests/lx_fit.py`.

## TL;DR

Fix A (PSUM-side gate) reframes the LX-fit picture cleanly:

- The heuristic in PR 1933 is **safe** — none of its 15 firing
  cases overflow C_psum. The Phase 0 conclusion that the heuristic
  was rejected on 2 rows was an artifact of the wrong predicate.
- The validation set's catastrophic-under-prediction rows are
  **all and only** the C_psum-overflow rows. Two of three overflow
  rows under-predict by 46–62%; the third (down_proj M=2048 +id)
  has a separate PSUM-ring-distance issue.
- **14 production matmuls overflow C_psum under pure-M at M=2048**,
  mostly MLP gate/up projections on Llama 70B/405B/Mixtral. These
  are not in the validation set; they're the prefill-regime shapes
  the planner currently runs without seeing the catastrophe.

## Old gate vs new gate disagreements (Section A summary)

The two predicates classify production matmuls very differently:

|  | A-overflow count | C-overflow count |
|---|---:|---:|
| pure-M (32, 1, 1) | 3 | **15** |
| heuristic-firing (1, n, k>1) | 2 | **0** |

Direction of disagreement:

- **14 cases**: A-side gate says fits, C-side says overflow. These
  are wide-N matmuls at M=2048 (gate_proj, up_proj on Llama 70B/405B/
  Mixtral; q_proj/o_proj on Llama 405B; gate/up/q/o on DeepSeek V3).
  The old gate would have green-lit catastrophic regimes.
- **2 cases**: A-side says overflow, C-side says fits. These are
  L3-70B and L3-405B kv_proj at M=512 under (1, 16, 2)+kf. The old
  gate would have rejected these — but the validation row L3-70B
  kv_proj M=512 +kf measures **3.17 ms** (faster than pure-M 3.36),
  confirming there's no actual catastrophe. The C-side gate
  correctly accepts.

The C-side gate is right empirically on every disagreement we have
data for.

## Validation residual partition (Section B summary)

| C-PSUM status | n | mean \|err\| | max \|err\| | under-pred count |
|---|---:|---:|---:|---:|
| fits | 27 | 16.5% | 94.8% | 2 / 27 |
| overflows | 3 | 69.1% | 99.4% | 2 / 3 |

The three C-overflow rows in the validation set:

| row | C_psum overage | rel err | direction |
|---|---:|---:|---|
| DSv3 o_proj M=2048 (1,16,2)+kf | 1.75× | -46.0% | under |
| DSv3 o_proj M=2048 (1,16,2)+id | 1.75× | -61.8% | under |
| DSv3 down_proj M=2048 (1,16,2)+id | 1.75× | +99.4% | over |

The two **under**-predictions are the catastrophic regime — exactly
where the C-spill mechanism causes the cost model to undershoot.
Fix B (the PSUM-overflow penalty calibrated at ~17 ms per overage
factor) would close these.

The **over**-prediction on down_proj +id is a different residual:
the cost model's pipe-PSUM term at id mode (m·n × hops) overstates
the actual ring time. Fix C (replace pipe-PSUM with actual ring
distance per Probe 2) addresses it.

So the residual story now decomposes cleanly:

| residual class | mechanism | fix |
|---|---|---|
| Catastrophic under-pred on overflow rows | C-spill not modelled | Fix B |
| Over-pred on +id rows (with or without overflow) | PSUM pipe term wrong | Fix C |
| Small-M over-pred on +kf rows | HMI achieved-BW too low at small M | (HW BW calibration) |
| All other rows | within ±15% | already-acceptable |

## Production planner impact (Section C summary)

If we wired the C-PSUM gate into PR 1933's heuristic, **zero
heuristic-fired splits would be rejected** — the heuristic is
already safe.

If we wired it as a *general* check before the planner picks pure-M
on M=2048 prefill matmuls, 14 cases would surface as "consider an
alternative split". Promising candidates would be (1, n, k>1)
splits where C_psum drops below LX and HMI per-cluster bytes drop
proportionally — which is why these shapes might benefit from the
heuristic if its `n_sticks ≥ 32` gate were relaxed at M=2048.

Sample candidates worth measuring (not in current validation):

- L3-70B gate_proj M=2048 (2048, 28672, 8192): pure-M C_psum =
  7 MB (3.5× LX), catastrophic. Try (1, 16, 2)+kf: M_per=2048,
  N_per=1792, C_psum = 14 MB (7× LX) — *worse*. (1, 32, 1)+kf is
  not k_fast since k=1. Try (32, 1, 1) only is the planner's pick.
  Hmm — these wide-N shapes have *no* split that puts C_psum below
  LX at M=2048 because M_per × N_per = M × N / (m × n) is
  invariant under any split that doesn't reduce the M dimension.
  Pure-M at m=32 minimises C_psum to (M/32) × N × 4 = 64 × N × 4.
  For N = 28672 that's 7 MB. To fit LX we'd need m × n_per to give
  M_per × N_per × 4 ≤ 2 MB, i.e., M × N ≤ 512K elements. For
  M=2048, N=28672: M × N = 58M elements; we need to divide by 117
  cores. That's not possible with 32 cores. **Wide-N
  prefill is structurally LX-overflowing; the planner has no
  good choice.**

This is a real architectural finding: at large M and large N, the
PSUM accumulator's residency requirement outstrips per-core LX no
matter how you split the work-division. The kernel template must
have a fast streaming-output path for this regime — which is
exactly the (1, 1, 32) anomaly Probe 1 surfaced (one chain head,
one streaming output). If we can characterise the conditions under
which the kernel template streams output efficiently, we can extend
the planner's options.

## Reframing of the project

The original "emission-aware LX scheduling" framing was about
cohort-clustering and chain-cooperative residency — both refuted.
But what Phase 0/1 has produced is more practical:

**The cost model and planner both need to gate on PSUM accumulator
residency, not operand-A residency.** That's a single concrete
change with a clean validation outcome.

Alongside it:

- **Fix B** (calibrated overflow penalty) closes the catastrophic
  rows.
- **Fix C** (ring-distance PSUM term) closes the +id over-pred rows.
- **Wide-N prefill is structurally underserved**: the planner has
  no LX-fitting split for many M=2048 MLP layers. The kernel
  template must have a streaming-output fast path; characterising
  it is the next architectural lever.

## Files

- `tests/lx_fit.py` — updated predicate (gates on C_psum)
- `tests/diag_lx_overflow_phase1.py` — new diagnostic
- `tests/diag_lx_overflow_phase0.py` — historical (A-side gate)
- This doc — Phase 1 findings
- `tests/diag_emission_aware_lx_phase0_findings_v2.md` — Probe
  1/2/3 findings that motivated Fix A
