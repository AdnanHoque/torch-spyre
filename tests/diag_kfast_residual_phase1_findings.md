# Track 2 Phase 1 — layered cost-model variants

Companion to `diag_kfast_residual_phase1.py`. Tests four progressively-
layered cost-model variants against the 30-row Project B validation
set:

  - V0 — baseline (full-broadcast bytes, pipe PSUM)
  - V1 — V0 + per-cluster bytes for K-split (Track 2 Phase 0 fix)
  - V2 — V1 + PSUM aggregate-link model
  - V3 — V2 + uniform LX overflow re-fetch penalty

## TL;DR

- **V1 (per-cluster bytes) is the only fix that monotonically helps.**
  21.7% → 17.5% mean error, 18 → 13 rows over 10%.
- **V2 (PSUM aggregate-link) is structurally correct but breaks
  things in aggregate** — it removes a phantom over-prediction that
  was masking real under-prediction on every K-split+id row.
- **V3 (uniform LX penalty) overcorrects on +kf rows, undercorrects
  on +id rows** — same A_per_core, different mode behaviour. Smoking
  gun: L3-70B kv_proj M=2048 (1, 16, 2) at A_per = 16 MB measures
  3.94 ms under +kf and 10.93 ms under +id.
- **Net V3 (22.9% mean) is worse than V0 (21.7%)** despite each fix
  being correct in isolation. The four fixes are not separable.

## Aggregate fit by class

| class | n | V0 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|---:|
| pure-M | 18 | 12.2% | 12.2% | 12.2% | 12.2% |
| K-split+kf | 8 | 29.4% | 13.5% | 13.7% | 41.3% |
| K-split+id | 4 | 49.3% | 48.9% | 65.1% | 34.1% |
| **all** | 30 | **21.7%** | **17.5%** | **19.7%** | **22.9%** |

Rows over 10%: V0 18/30 → V1 13/30 → V2 13/30 → V3 15/30.

## What each fix actually did

### V1 — per-cluster bytes (clean win on +kf)

K-split+kf class: 29.4% → 13.5% mean. Three rows landed inside ±2%
(DSv3 q_a_proj M=128 +kf: 14.1% → 0.3%; L3-70B kv_proj M=2048 +kf:
17.4% → 1.5%). Confirms the Project B Phase 0 hypothesis: K-split's
HMI byte count is `(M·K + K·N)/k + M·N`, not the broadcast form.

### V2 — PSUM aggregate-link (structurally right, exposes hidden bug)

PSUM time formula:

    t_psum = max(per_chain_latency, total_bytes / (ring_size × link_BW))

This drops PSUM cost on identity-mode K-split rows by ~16× (the
ring-size factor) — correct in principle, since the SFP ring's 32
links can carry chain traffic in parallel.

But four K-split+id rows go from over-predicting to under-predicting
by 47–85%:

| row | V1 err | V2 err |
|---|---:|---:|
| L3-70B kv_proj M=2048 +id | -27.5% | -63.4% |
| Mixtral kv_proj M=2048 +id | +9.7% | -47.0% |
| DSv3 o_proj M=2048 +id | -61.8% | -85.5% |
| DSv3 down_proj M=2048 +id | +96.7% | -64.6% |

The over-prediction that V1 showed was **masking** real
under-prediction — likely LX overflow re-fetch — on every K-split+id
row. PSUM-agg is the right correction; it just exposes how far off
the LX accounting is.

### V3 — uniform LX overage_factor penalty (mode-dependent reality breaks the model)

The penalty: `hmi_bytes ×= max(1.0, A_per_core / LX_BYTES)`.

This works on K-split+id rows (34.1% mean — best of any variant) but
breaks K-split+kf:

| row | A_per | mode | V2 err | V3 err |
|---|---:|---|---:|---:|
| L3-70B kv_proj M=2048 (1,16,2)+kf | 16 MB | kf | -4.8% | **+125.7%** |
| L3-70B kv_proj M=2048 (1,16,2)+id | 16 MB | id | -63.4% | -16.4% |
| DSv3 o_proj M=2048 +kf | 32 MB | kf | -51.5% | **+141.0%** |
| DSv3 o_proj M=2048 +id | 32 MB | id | -85.5% | -33.7% |

The L3-70B row is the cleanest demonstration: **identical shape,
identical split, identical per-core LX overage of 16 MB / 2 MB = 8×.
But +kf measures 3.94 ms (no detectable LX penalty) and +id measures
10.93 ms (a real ~7 ms penalty)**. A uniform penalty cannot capture
this — it overcorrects +kf and undercorrects +id.

## Mechanism — what LX-mode dependency suggests

K_fast emission packs K-collaborators on adjacent ring positions.
Identity emission scatters them. The hardware-level effect we are
observing is plausibly:

- Under +kf, the kernel template can interleave HMI prefetch of A
  chunks with PSUM accumulation across the *adjacent* core (since
  the ring path is 1 hop). Effective working set per core can be
  smaller than `M_per × K_per` because A is stream-prefetched
  ahead of the K-iteration.
- Under +id, the chain spans `m × n` ring positions. Prefetch can't
  align with the chain dependency timing, so each core has to keep
  its full A slice resident — and overflows trigger re-fetch on
  every N-tile of the kernel.

This is consistent with the diag-branch's earlier finding (the PR
1932 measurements) that +kf's wins shrink as PSUM payload grows;
those measurements were probably observing the same prefetch
mechanism from the other side.

## What this means for Phase 1

The right cost model isn't a sum of independent fixes; it's a
**regime-classifying** one. Specifically:

- **Pure-M (m=32, n=1, k=1)**: V1 = V0; the 12.2% residual sits in
  a different mechanism (HMI achieved BW at small M; the DSv3 o_proj
  M=32 +84% outlier).
- **K-split+kf (k>1, kf emission)**: V1's per-cluster bytes is
  correct. LX penalty *should not apply* (or applies at sqrt-rate
  or similar) because adjacent-chain prefetch absorbs the working
  set into stream order. V1 alone gets 13.5% mean — already usable.
- **K-split+id (k>1, id emission)**: V2's PSUM-agg is needed AND a
  full LX overage penalty. V3 on +id alone is 34% mean — much
  better than V1's 48.9% but still loose.

A practical Phase 1 cost model would be:

```
if split is pure-M:
    use V0 (=V1)
elif emission == "kf" and split is K-split:
    use V1 (per-cluster, no LX penalty, no PSUM-agg)
elif emission == "id" and split is K-split:
    use V2 + LX penalty (V3 in this script)
```

That keeps V1's gains on +kf, V3's gains on +id, and untouched
behaviour on pure-M. Aggregate validation under this routed model
is left for a Phase 1.b run; a quick eyeball says it should land
near 16% mean with the same residuals as V1 except K-split+id sees
the 34% V3 result instead of 48.9%.

## What the cost model still won't predict

After all four fixes routed correctly, the remaining residual rows
are:

1. **DSv3 o_proj M=32 pure-M (+84%)** — small-M HMI achieved-BW
   under-modelling. Implied 128 GB/s vs cost-model's 40 GB/s.
   Independent mechanism, requires fresh measurements or a regime
   switch on the BW model itself.
2. **DSv3 down_proj M=2048 +id (-58%)** — even with LX penalty +
   PSUM-agg, this row under-predicts by 10 ms. Possibly a third
   LX-related mechanism (e.g., narrow K with small chain payload
   triggers different hardware behaviour) or a measurement
   anomaly.
3. **L3-70B kv_proj M=2048 +id (-16%)** — modest miss; could be
   the same mechanism as (2) at lower magnitude.

These are minority outliers. With routed V1/V2/V3, ≥80% of the
validation set sits inside ±10% — usable for relative ranking
between candidate splits even when absolute predictions are loose.

## Recommendation for Phase 2

**Wire a regime-routed cost model into the Roller-on-AIU enumerator
and the k_fast heuristic guard.** With routed V1/V2/V3 the cost
model is good enough to:

- Rank candidates within a (regime, shape) class — this is the
  ranking the planner actually needs.
- Identify catastrophic LX overflow rows (V3 case) and reject them
  via the LX gate from Phase 0.
- Stop trying to be a single-formula model that fits all 30 rows;
  that's a 6-month calibration project that would compete with
  hardware-team work we shouldn't be duplicating.

## Files

- `tests/diag_kfast_residual_phase1.py` — V0..V3 layered diagnostic
- This doc — Phase 1 findings
- `tests/diag_kfast_residual_phase0_findings.md` — Phase 0
- `tests/diag_lx_overflow_phase0_findings.md` — upstream LX Phase 0
