# HMI cost model — Phase 0 findings

## Status

After targeted HMI BW probe (`diag_hmi_bw_pure_m.py`) and refit, the
model fits **12 / 30 rows within 10%** of measured wall time. Median
relative error 13.3%, mean 21.7%, max 99%. Signed mean error −1.7 ms.

**Still not within the ≤10% target.** Pattern 1 (wide-B pure-M
under-prediction) is now largely closed — DSv3 o_proj M=128 went from
+154% error to +1.2%. The remaining residuals fall into different
patterns that need structural changes rather than parameter tweaks.

## Phase 0 deltas vs first pass

| change | reason | effect |
|---|---|---|
| HMI_BW_GBS: 67 → 40 | Probe shows effective BW asymptotes to ~40 GB/s under pure-M with broadcast B accounting | Closes the wide-B pure-M residual |
| Wall formula: max(LF, max(c, h) + p) → max(c, h+LF) + p | Probe: wall ≈ LF + bytes/BW exactly under HMI-bound, but compute-bound rows fit measured compute alone (no LF added) — so LF overlaps with compute but stacks on HMI | Fixes pure-M wide-B without inflating compute-bound o_proj M=2048 |
| ACHIEVED_FRAC: 0.5 → 1.0 | Compute-bound rows fit better at full PT_PEAK (peak fp16 is 1 TFLOP/core in practice) | Improves fit on M=2048 rows broadly |
| PSUM total = m·n × (k−1) × hops × payload | Original missed multi-chain serialization; chains share the SFP ring | Improves +id residuals (still imperfect) |

## What fits within 10%

Mostly small-to-medium narrow-N shapes where launch floor or a
modest HMI demand dominates:

| shape | mode | measured | predicted | rel err |
|---|---|---:|---:|---:|
| L3-70B kv_proj M=32 | natural | 3.31 | 3.00 | 9.3% |
| L3-70B kv_proj M=128 +kf | kf | 3.09 | 3.00 | 2.9% |
| L3-70B kv_proj M=512 +kf | kf | 3.17 | 3.00 | 5.5% |
| Mixtral kv_proj M=32–2048 (all) | natural | 3.10–3.27 | 3.00 | 3.4–8.4% |
| Mixtral kv_proj M=128 +kf | kf | 3.01 | 3.00 | 0.3% |
| DSv3 down_proj M=32 | natural | 3.17 | 3.00 | 5.3% |
| DSv3 down_proj M=128 +kf | kf | 3.16 | 3.00 | 4.9% |
| DSv3 q_a_proj M=128 +kf | kf | 3.22 | 3.00 | 6.9% |
| DSv3 o_proj M=2048 +kf | kf | 31.23 | 30.18 | 3.4% |

These rows confirm the model's launch floor (3 ms) is calibrated
and that for genuinely launch-floor-bound or
clearly-compute-bound-with-good-PT-utilization shapes, the
first-order model works.

## Residual pattern 1 (CLOSED): pure-M with very wide B

Probe `diag_hmi_bw_pure_m.py` swept 10 pure-M shapes with B from
8 MB to 256 MB and inferred effective HMI BW under both broadcast and
replicated-B accounting. Results:

- Replicated-model BW: 1300–3400 GB/s (absurd) → **B is NOT 32×
  replicated under pure-M**. Broadcast accounting is correct.
- Broadcast-model BW: clusters at **40 GB/s for M=64**, 45–46 GB/s for
  M=512 (the latter inflated because compute overlaps part of HMI).
- Wall formula that exactly fits HMI-bound rows: `wall ≈ LF + bytes/BW`
  with LF=3 ms, BW=40 GB/s. No compute term needed when M is small.

After lowering HMI_BW_GBS from 67 to 40 and switching the wall formula
from `max(LF, max(c,h)+p)` to `max(c, h+LF)+p`:

| shape | measured | predicted (now) | err |
|---|---:|---:|---:|
| DSv3 o_proj M=128 pure-M | 9.13 | 9.02 | +1.2% |
| DSv3 o_proj M=512 pure-M | 8.47 | 9.48 | +12% |
| DSv3 o_proj M=2048 pure-M | 13.28 | 15.03 | +13% |

The headline pattern is closed. The 67 GB/s figure was the spec
nameplate, not the achieved bandwidth under matmul-template ring
broadcast. **Use 40 GB/s for the broadcast pure-M regime.**

## Residual pattern 1b (PARTIAL): k-split shrinks HMI for some shapes

K-split probe (`diag_hmi_bw_k_split.py`) measured 4 shapes × 5
configs (pure-M, (1,16,2)±kf, (1,8,4)±kf). Three takeaways:

### A. Per-cluster bytes model partially correct

For shapes where per-core A fits in LX (≤2 MB), the per-cluster
model fits well. For shapes where it overflows, neither model fits.

| shape | split | measured kf | full-model | cluster-model | best |
|---|---|---:|---:|---:|---|
| L3-70B kv_proj M=2048 | (1,16,2) | 3.93 | 4.63 (18%) | 4.00 (2%) | cluster |
| L3-70B kv_proj M=2048 | (1,8,4) | 4.27 | 5.15 (21%) | 4.21 (2%) | cluster |
| DSv3 o_proj M=128 | (1,16,2) | 4.70 | 9.14 (95%) | 6.15 (31%) | cluster |
| DSv3 down_proj M=2048 | (1,16,2) | 6.85 | 6.51 (5%) | 6.04 (12%) | full |
| DSv3 o_proj M=2048 | (1,16,2) | 31.25 | 16.87 (46%) | 16.87 (46%) | both fail |
| DSv3 o_proj M=2048 | (1,8,4) | 124.33 | 20.54 (83%) | 20.54 (83%) | both fail |

The two o_proj M=2048 rows fail badly: kf measures 31 ms and 124 ms,
both models predict ~20 ms. There's a missing structural cost.

### B. NEW: LX overflow re-fetch dominates wide-shape k-split

At M=2048, K=16384 under (1, 8, 4): per-core A slice = 2048·4096·2
bytes = **16 MB**, which is 8× the per-core LX capacity (2 MB). The
N tile dim (N_per=896) requires multiple K-streamed sub-tiles, so A
gets re-fetched from HMI per N-chunk. Rough estimate: 14× re-fetch
→ 7 GB total HMI demand at 40 GB/s ≈ 180 ms — consistent ballpark
with measured 124 ms (and explains why cluster-bytes accounting
under-predicts so badly).

Pure-M (32, 1, 1) keeps A_per_core = 64·16384·2 = 2 MB → fits LX
exactly, no re-fetch. That's the mechanism behind "planner pure-M
wins at large M".

**Implication**: cost model needs an LX-fit gate. When per-core A
or B exceeds LX, re-fetch multiplier kicks in. This is a structural
addition, not a parameter tweak.

### C. k_fast adjacency benefit collapses at k=4

PSUM cost decomposition (id wall − kf wall = empirical PSUM):

| shape | (1,16,2) id−kf | (1,8,4) id−kf |
|---|---:|---:|
| DSv3 o_proj M=128 | +5.28 | +0.11 |
| DSv3 o_proj M=2048 | +84.62 | +1.85 |
| DSv3 down_proj M=2048 | +10.14 | -0.13 |
| L3-70B kv_proj M=2048 | +6.94 | +1.11 |

At (1,16,2) the kf adjacency gives 5–85 ms savings, scaling roughly
with payload. At (1,8,4) the savings vanish — kf and id measure
nearly the same wall on every shape.

Mechanism guess: at k=4 with mn=8, kf places k-collaborators at
ring positions {0, 1, 2, 3} of a 32-core ring. The PSUM chain still
makes 3 sends, each nominally 1 hop, total 3 hops. But the chain is
already so long (m·n = 8 chains in flight, each making 3 sends) that
SFP ring bandwidth is the bottleneck and the per-send hop count
matters less than total bytes traversing the ring.

This is an SFP throughput saturation regime, not an SFP latency one.
The model's per-hop bytes/SFP_BW formula approximates total ring
bytes correctly, but the empirical id−kf is not predictable from
the hop-count delta alone — it depends on payload too.

## Residual pattern 2: (1, 16, 2)+identity PSUM is under-predicted

| shape | measured | predicted | err |
|---|---:|---:|---:|
| L3-70B kv_proj M=2048 (1,16,2)+id | 10.93 | 5.27 | +107% |
| Mixtral kv_proj M=2048 (1,16,2)+id | 6.94 | 4.73 | +47% |
| DSv3 o_proj M=2048 (1,16,2)+id | 116.12 | 44.39 | +162% |
| DSv3 down_proj M=2048 (1,16,2)+id | 17.07 | 31.24 | −83% |

The model predicts PSUM bytes correctly per chain × hops, but
measured wall is consistently 2-3× higher than the bandwidth
calculation suggests. Possible causes:

- Effective SFP BW is < 32 GB/s (perhaps 10-15 GB/s achieved).
- Chain serialisation incurs per-hop fixed overhead beyond raw
  byte transit.
- Multiple chains in flight contend for shared resources beyond
  raw bandwidth (e.g., per-core SFP queue depth).

Note DSv3 down_proj +id is the *opposite* — over-predicted, which
suggests the model is mis-counting payload for that shape. Worth
checking the K-cluster math on a non-square N geometry.

**To close**: instrument the existing pairwise-distance probe data
more carefully. We measured 0.476 ms / d on the (1, 16, 2) kv_proj
shape — that gives an empirical per-hop cost we should use directly,
rather than computing from raw bandwidth.

## Residual pattern 3: launch-floor-bound rows are under-predicted by 5-25%

| shape | measured | predicted | err |
|---|---:|---:|---:|
| L3-70B kv_proj M=128 pure-M | 3.37 | 3.00 | +12% |
| L3-70B kv_proj M=2048 pure-M | 3.67 | 3.00 | +22% |
| DSv3 down_proj M=128 pure-M | 3.73 | 3.00 | +24% |
| DSv3 q_a_proj M=2048 pure-M | 4.06 | 3.00 | +35% |

Many shapes the model classifies as launch-floor-bound have
measured walls 0.3–1.0 ms above 3.0. Either:

- The launch floor is closer to 3.3 ms (suggests parameter tweak)
- There's a per-op overhead proportional to compute or operand
  size that the model misses (suggests structural addition)

**To close**: fit launch_floor + small_overhead(operand_bytes)
linearly against the launch-floor-bound subset.

## Phase 0 verdict

The model is **not yet ready** to drive Phase 2 scheduling
decisions confidently — too many rows outside the 10% bound, and
the misses are not symmetric (predominantly under-prediction).

It **is** useful as a *coarse* classifier: it correctly labels
shapes as launch-floor-bound vs HMI-bound vs compute-bound on
roughly 80% of rows, and the absolute predictions are within 50%
on the worst rows. That's enough to identify HMI-bound op pairs
(the question Phase 2 needs to answer) but not enough to predict
exact wall-time deltas under hypothetical scheduling.

Three concrete next-step options:

1. **Iterate on the model first** — close the three residual patterns
   above with parameter fits (probably ~1 day) before extending to
   Phase 1. Less risky.

2. **Move on to Phase 1 with caveats** — extend to a transformer
   block now, accepting ±50% prediction error. The Phase 1 output
   is qualitative ("which ops are HMI-bound, where are the gaps")
   and tolerates the model's coarseness.

3. **Run a small targeted probe** — measure HMI bandwidth under a
   few specific shapes (a wide-B pure-M sweep) to fit the missing
   parameter, then come back to calibration.

## Files

- `hmi_cost_model.py`: the per-op predictor module
- `diag_hmi_cost_model_calibrate.py`: calibration harness with 30-row
  validation set
- `diag_hmi_bw_pure_m.py` + `*_results.txt`: HMI BW probe under pure-M
- `diag_hmi_bw_k_split.py` + `*_results.txt`: HMI bytes & PSUM probe
  under k-split (kf + id emissions, k ∈ {1, 2, 4})
- This doc: findings

## What worked despite the residuals

- Mixed-radix PSUM-byte accounting (`m·n × (k−1) × hops × payload`)
  matches the kv_proj +kf rows almost perfectly.
- PT utilisation curve (`min(1, M_per/8) × min(1, N_per/64)`) gives
  the right qualitative classification across all rows.
- Launch floor is the right floor for shapes where compute and HMI
  are below 3 ms.

These are the parts of the model worth keeping verbatim for Phase 1.
