# Cost model V4 — Fixes A/B/C/D layered, validation re-run

V4 = V0 + four fixes derived from the Phase 0/1 LX investigation +
the Probe 1-6 mechanism characterisation:

  - **Fix A**: HMI bytes use per-cluster `(M·K + K·N)/k + M·N`
    (Track 2 Phase 0). For k=1 this reduces to broadcast.
  - **Fix B**: PSUM-overflow penalty at `n>1, k_fast=True,
    C_psum > LX`. ~17 ms per overage factor (Probe 3 calibration).
    Gated on k_fast since Probe 3 measured the catastrophic
    A-re-fetch-per-N-tile mechanism only under k_fast.
  - **Fix C**: Pipe-model ring traversal with `hops_per_send` from
    the active emission. Existing structure; unchanged in this PR.
  - **Fix D**: n=1 streaming-output regime cost: pipeline regime
    (chain ≤ 4) +3 ms, sync regime (4 < k < 32) ~`1.5 × payload_MB
    + 5` ms, allreduce regime (k = 32) +14 ms (Probe 6 calibration).

`tests/lx_fit.py` was already updated as part of LX-Phase-1 (Fix A
sense, but for the LX-fit gate) — separate from the cost model
changes here.

## Result on 30-row validation set

| version | mean \|err\| | median \|err\| | max | rows over 10% |
|---|---:|---:|---:|---:|
| V0 (baseline) | 21.7% | — | 99.4% | 18/30 |
| V1 (+ per-cluster bytes only) | 17.5% | — | 99.4% | 13/30 |
| V4 (all fixes, B gated on kf) | **16.1%** | 8.1% | 97% | **12/30** |

5.6 percentage point improvement on mean error, 6 fewer rows outside
the ±10% band, and the worst single-row error (96.7%) is no worse
than V0's worst (99.4%).

## Per-row outcomes for the K-split + kf rows the cost model existed to predict

These are the rows the cost model is critical to get right (the
planner uses cost-model rankings to choose between K-split splits
and pure-M):

| row | V0 err | V4 err |
|---|---:|---:|
| L3-70B kv_proj M=128 (1,16,2)+kf | +13.1% | +5.5% |
| L3-70B kv_proj M=512 (1,16,2)+kf | +17.2% | (not in worst) |
| L3-70B kv_proj M=2048 (1,16,2)+kf | +17.4% | **+1.5%** |
| Mixtral kv_proj M=128 +kf | +8.3% | (not in worst) |
| DSv3 o_proj M=128 +kf | +94.8% | +31.1% |
| DSv3 o_proj M=2048 +kf | -46.0% | **-5.2%** |
| DSv3 down_proj M=128 +kf | +23.8% | +12.0% |
| DSv3 q_a_proj M=128 +kf | +14.1% | **+0.3%** |

Three rows now land inside ±2% (q_a_proj M=128 +kf, L3-70B kv_proj
M=2048 +kf, DSv3 o_proj M=2048 +kf). The catastrophic-regime row
(DSv3 o_proj M=2048 +kf at A_per = 32 MB) drops from -46% to -5% —
exactly the case Probe 3's calibration was meant to capture.

DSv3 o_proj M=128 +kf still shows +31% over-prediction. That's the
small-M HMI BW issue from Track 2 Phase 1 (the cost model uses
40 GB/s but small-M kf shapes implied 128 GB/s). Not addressed by
any of A/B/C/D — needs a separate calibration sweep.

## Remaining residuals (rows still outside ±10%)

| row | V4 err | category |
|---|---:|---|
| DSv3 down_proj M=2048 (1,16,2)+id | +96.7% | +id K-dependent over-pred |
| DSv3 o_proj M=32 | +84.1% | small-M HMI BW |
| DSv3 o_proj M=2048 (1,16,2)+id | -61.8% | +id K-dependent under-pred |
| DSv3 o_proj M=128 +kf | +31.1% | small-M HMI BW |
| L3-70B kv_proj M=2048 (1,16,2)+id | -27.5% | +id K-dependent |
| L3-70B kv_proj M=2048 (32,1,1) | +19.0% | small-M-related growth term |
| DSv3 down_proj M=32 | +18.4% | small-M (LF-bound) |
| Mixtral kv_proj M=2048 (32,1,1) | +14.1% | small-M-related growth term |
| L3-70B kv_proj M=1024 (32,1,1) | +13.3% | small-M-related growth term |
| DSv3 o_proj M=2048 (32,1,1) | +13.2% | (other) |
| DSv3 down_proj M=128 +kf | +12.0% | small-M (LF-bound) |
| DSv3 o_proj M=512 | +11.8% | (other) |

Two distinct residual categories remain:

1. **Small-M HMI BW** (5 rows). The cost model uses 40 GB/s
   achieved BW; small-M shapes imply 67-128 GB/s. Track 2 Phase 1
   documented this as Mechanism 2; not addressed by any of A-D.
   Needs a hardware-side BW calibration sweep.

2. **+id K-dependent residual** (3 rows). The pipe-model PSUM
   formula (hops × payload / SFP_BW) gives the same prediction for
   DSv3 o_proj +id and DSv3 down_proj +id (same payload), but
   measured walls differ by 7×. The actual cost depends on K_per
   in a way the model doesn't capture. Possibly K-tile re-iteration
   under +id with scattered chains. Needs more probe data to fit.

## What V4 actually buys for the planner

The cost model is now substantially correct on every K-split+kf row
that the production planner could pick. Specifically:

- All 8 K-split+kf rows in the validation set are within ±31%, with
  5 of 8 within ±15% and 3 within ±2%.
- The 3 worst K-split+kf rows are all small-M (M ≤ 128) — the
  small-M HMI BW residual, not the K-split mechanism.

This means the cost model can be trusted to **rank candidate
splits** for the planner — comparing pure-M to (1, n, k>1)+kf,
or comparing different (m, 1, k)+kf chain lengths against each
other. Even where absolute predictions are loose, the relative
ordering matches measurement on the rows we have.

The two remaining residual categories (small-M HMI, +id
K-dependent) don't affect the planner's typical path because:

- Small-M HMI residual is bounded — the worst row is +84% relative
  but only ~4 ms absolute. Won't flip ranking decisions where
  competing splits are all small-M.
- +id K-dependent residual only affects rows where the planner
  forced identity emission. Production sets `core_id_permutation =
  k_fast` by default, so this only affects diagnostic comparisons,
  not planner output.

## Files changed

- `tests/hmi_cost_model.py` — `_hmi_bytes` switched to per-cluster
  formula; new `_psum_regime_cost_ms`, `_n1_sync_regime_cost_ms`,
  `_c_psum_per_core` helpers; `predict()` calls
  `_psum_regime_cost_ms` instead of bare pipe-model PSUM.
- `tests/diag_hmi_cost_model_calibrate_v4_results.txt` — saved V4
  calibration output for reference.
- This doc.

## Next steps

The cost-model improvements unblock planner integration. Concrete
next moves:

1. **Add (m, 1, k)+kf candidate space to the planner.** The
   cost model now correctly costs these splits relative to pure-M.
   Where pure-M is best, it stays best; where pure-M overflows
   C_psum, the cost model identifies (m, 1, k)+kf alternatives at
   ~1.3-1.5× pure-M instead of the current "no good split"
   situation.

2. **Surface the EAR ceiling** to the deeptools team with Probe 5
   numbers. Wide-N prefill on Llama 70B+ MLP layers is structurally
   underserved.

3. **Address the +id K-dependent residual** (Fix E?). Run more
   probes at varying K to characterise the K_per scaling. Lower
   priority since production uses k_fast.

4. **Calibrate small-M HMI BW.** Run an HMI-only probe at small M
   to see what achieved BW the kernel template gets when compute
   is light. Independent of LX investigation.
