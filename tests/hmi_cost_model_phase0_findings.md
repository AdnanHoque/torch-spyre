# HMI cost model — Phase 0 findings

## Status

First-pass model in `hmi_cost_model.py` fits **10 / 30 rows within
10%** of measured wall time. Median relative error 14.1%, mean 22.8%,
max 83%. Signed mean error −3.3 ms (model under-predicts on average).

**Not within the ≤10% target the scope doc set for Phase 0.** The
residuals cluster into three structural patterns that point to
specific model improvements rather than a wholesale rewrite. Each
pattern is documented below with what we'd need to measure to close
the gap.

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

## Residual pattern 1: pure-M with very wide B is under-predicted

| shape | measured | predicted | err | classification |
|---|---:|---:|---:|---|
| DSv3 o_proj M=128 pure-M | 9.13 | 3.60 | +154% | HMI-bound |
| DSv3 o_proj M=512 pure-M | 8.47 | 3.87 | +119% | HMI-bound |
| DSv3 o_proj M=32 pure-M | 4.84 | 3.53 | +37% | HMI-bound |

All have huge B = K·N (e.g. 235 MB for o_proj). Model assumes B is
fetched from HMI exactly once and ring-broadcast across all 32
cores, giving HMI = M·K + K·N + M·N bytes. Measured wall is 2-3×
predicted, suggesting the broadcast-once assumption is wrong in
practice — perhaps the kernel template fetches B in chunks per core
without full ring sharing, or HMI throughput drops when 32 cores
are concurrently demanding.

**To close**: a probe that measures HMI bandwidth directly under
pure-M, varying B size, would tell us whether the effective per-op
BW is lower than 67 GB/s, or whether B replication (32× streaming)
is happening for some shapes.

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
- This doc: findings

## What worked despite the residuals

- Mixed-radix PSUM-byte accounting (`m·n × (k−1) × hops × payload`)
  matches the kv_proj +kf rows almost perfectly.
- PT utilisation curve (`min(1, M_per/8) × min(1, N_per/64)`) gives
  the right qualitative classification across all rows.
- Launch floor is the right floor for shapes where compute and HMI
  are below 3 ms.

These are the parts of the model worth keeping verbatim for Phase 1.
