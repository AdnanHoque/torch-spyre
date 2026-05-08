# Planner v2 verification findings — most picks regress on hardware

Companion to `diag_planner_v2_verification.py`. Hardware-verified
8 representative Tier 2 picks from the planner v2 prototype against
the production planner's pure-M default.

## TL;DR

**Verdict: v2 prototype is not ready for planner integration.**
5 of 8 picks regress on hardware — the cost model dramatically
under-predicts wall on medium-M and wide-K K-split shapes.

What does survive contact with hardware:

- **Small-M big-speedup picks validate.** L3-70B q_proj M=32 +kf
  measured 1.49× speedup (predicted 1.74×); DSv3 gate_proj M=32 +kf
  measured 1.76× (predicted 2.21×). PT-utilization at small M is a
  real lever — pure-M's M_per=1 wastes the array, K-split with
  M_per=M fills it.

- **DSv3 q_a_proj M=128 (1,8,4)+kf validates** at 1.09× speedup.
  This is the row PR 1933's heuristic already targets; cost-model
  V4 prediction matches measurement to 0.3%.

What fails:

- **Medium-M K-split (M=128, M=512) regresses 15-41%.** Cost model
  says these should be faster than pure-M; hardware says they're
  slower. L3-70B q_proj M=512 (1,16,2)+kf predicted 5.5 ms,
  measured 11.0 ms (-50% off).
- **Wide-K K-split regresses catastrophically.** DSv3 down_proj
  M=512 (1,8,4)+kf measured 35.5 ms vs predicted 4.7 ms (-87% off).
  Pure-M is 4× faster on this row.
- **Even the validation-set "sanity" row marginally fails.** L3-70B
  kv_proj M=2048 (1,16,2)+kf measured 3.95 ms, pure-M 3.65 ms —
  pure-M is faster today even though validation set had +kf 3.94 vs
  pure-M 3.67. Marginal but the kf benefit is smaller than the
  validation set suggested.

## Full data

| row | shape | v2 split | pure-M | v2 | pred v2 | err | speedup | result |
|---|---|---|---:|---:|---:|---:|---:|---|
| L3-70B kv_proj M=2048 sanity | (2048,1024,8192) | (1,16,2)+kf | 3.65 | 3.95 | 4.00 | +1.3% | 0.92× | **FAIL** |
| DSv3 q_a_proj M=128 sanity | (128,1536,7168) | (1,8,4)+kf | 3.50 | 3.22 | 3.23 | +0.3% | 1.09× | VALIDATE |
| L3-70B q_proj M=32 big-spd | (32,8192,8192) | (1,4,8)+kf | 6.27 | 4.20 | 3.66 | -12.8% | 1.49× | VALIDATE |
| DSv3 gate_proj M=32 big-spd | (32,18432,7168) | (1,4,8)+kf | 9.51 | 5.41 | 4.37 | -19.3% | 1.76× | PARTIAL |
| L3-70B q_proj M=128 mid-spd | (128,8192,8192) | (1,8,4)+kf | 6.47 | 7.58 | 4.30 | -43.3% | 0.85× | **FAIL** |
| L3-70B q_proj M=512 mid-spd | (512,8192,8192) | (1,16,2)+kf | 6.46 | 11.04 | 5.52 | -50.0% | 0.59× | **FAIL** |
| DSv3 down_proj M=128 wide-K | (128,7168,18432) | (1,4,8)+kf | 9.70 | 11.16 | 3.86 | -65.4% | 0.87× | **FAIL** |
| DSv3 down_proj M=512 wide-K | (512,7168,18432) | (1,8,4)+kf | 9.24 | 35.49 | 4.72 | -86.7% | 0.26× | **FAIL** |

## Where the cost model went wrong

The cost-model V4 (Fixes A/B/C/D) was calibrated against the 30-row
Project B validation set. That set:

- Used a narrow shape mix (4-5 ops on 4-5 models)
- Included only a few K-split rows
- Was dominated by decode-regime shapes

The verification probe added shapes outside that narrow mix —
specifically, **q_proj** at M=128/512 (wider N=8192 than the
validation set's kv_proj at N=1024) and **down_proj** at M=128/512
(huge K=18432). Both regimes show wall behaviors the cost model
wasn't trained to predict.

Concrete things missing from V4:

1. **Real-shape K-split penalty at medium M.** Even with C_psum
   fitting LX (no Fix B trigger) and small chain length (Fix D
   pipeline regime adds only 3 ms), measured walls are 70-90%
   higher than predicted. Mechanism unknown — possibly K-tile
   re-iteration overhead, similar to the +id K-dependent residual
   we saw earlier.
2. **Catastrophic regime at large K_per under K-split + kf.** DSv3
   down_proj M=512 at K_per=4608, M_per=512: measured 35 ms,
   predicted 5 ms. The mid-k catastrophe characterised in Probe 1
   was at large M (=2048) with C_psum > LX. Now we see a similar
   catastrophe at medium M (=512) with C_psum *fitting* LX. The
   trigger is wider than C_psum overflow.

## What this means

**For the planner integration**: don't proceed on the Tier 2 list.
The picks are not reliably faster than pure-M.

**For the cost model**: V4 over-promises on shapes outside the
calibration set. Before any production change, the model needs:
- Broader calibration (more shapes, more M values)
- A characterised penalty for the medium-M K-split regression
- Verification of the big-speedup small-M predictions on more
  shapes to know whether they generalise

**For the research narrative**: The strongest finding holds. PT
utilisation at small M is a real, measurable lever — both verified
big-speedup rows showed 1.5-1.8× speedup vs pure-M. The
M-vs-N-asymmetry mechanism (Probe 4) is intact. The streaming-
output regime structure (Probe 6) is intact. What's broken is the
cost model's ability to predict K-split wall outside the
calibration set.

## Refined plan

The verification turned a "ready-to-integrate" picture into a
"more characterisation needed" picture. Before any planner change,
sequential work to do:

1. **Probe 7 (proposed): characterise the medium-M K-split
   regression.** Run a sweep across M ∈ {64, 128, 256, 512, 1024,
   2048} at fixed (1, n, k)+kf splits on 2-3 shapes. Compare
   measured wall to V4 prediction. Find where the model breaks
   down and characterise the missing term.

2. **Probe 8 (proposed): characterise the wide-K K-split
   catastrophe.** Run K-sweep at fixed (M, N, split) on shapes
   like DSv3 down_proj. The K_per dependence we suspected from
   Probes 4-6 may be the culprit; this would pin it down.

3. **Re-run validation** on a broader shape set (say 100 rows
   instead of 30) once 1-2 are characterised. Refit cost-model
   constants.

4. **Then re-run this verification** on Tier 2 picks. If most
   validate, proceed to planner integration. If still mixed,
   restrict the planner change to validated shape regimes only.

The small-M big-speedup wins (verified at 1.5-1.8×) are
production-relevant on their own. A narrower planner change —
"prefer (1, n, k>1)+kf only for M ≤ 64 on wide-N shapes" — would
be a defensible incremental change without needing to fix the
medium-M cost model first.

## Files

- `tests/diag_planner_v2_verification.py` — verification probe
- `tests/diag_planner_v2_verification_results.txt` — raw output
- This doc
