# Track 2 Phase 0 — k_fast / k-split residual investigation

Companion to `diag_kfast_residual_phase0.py`. Started from the LX
Phase 0 finding that the cost model over-predicts wall by up to 2×
on small-M k_fast rows that fit LX, and asked: what residual term
explains it?

## TL;DR

**Per-cluster HMI bytes** is a real fix on `K-split+kf` rows
(29.4% → 13.5% mean error, with three rows landing inside ±2%) but
does not close the broader story. After that fix, the 30-row
validation set has **three distinct residual mechanisms** still
unaccounted for:

1. **PSUM-term over-prediction on identity-mode K-split** (e.g.
   DSv3 down_proj M=2048 (1,16,2)+id: 99% over-pred). Suggests the
   SFP-ring aggregate-bandwidth model is wrong: model treats the
   ring as one 32 GB/s pipe; the ring has 32 links so aggregate
   throughput when chains saturate links in parallel is closer to
   32 × 32 = ~1024 GB/s.

2. **HMI-BW under-prediction at small M on wide-K wide-N shapes**
   (DSv3 o_proj M=32 pure-M: 84% over-pred). Implied achieved BW
   from measurement is ~128 GB/s, well above the model's 40 GB/s
   floor and even above the 67 GB/s spec headline.

3. **Catastrophic LX overflow** (DSv3 o_proj M=2048 +id at A_per =
   32 MB: 62% under-pred; +kf: 46% under-pred). Already known from
   the LX Phase 0; the per-cluster fix doesn't address it.

These are independent levers. Per-cluster bytes (for kf), PSUM
re-modeling (for id), achieved-BW calibration (for small-M pure-M),
and LX overflow penalty (for catastrophic rows) need to be combined
for the cost model to be load-bearing across the full validation set.

## Per-cluster bytes hypothesis — confirmed for kf, weak for id

The hypothesis was that the cost model's HMI-byte formula
`M·K + K·N + M·N` (full broadcast) is wrong under K-split because
each K-cluster only fetches its own K_per chunk. The per-cluster
form is `(M·K + K·N) / k + M·N`.

By split class:

| class | n | broadcast mean | pcl mean | broadcast max | pcl max |
|---|---:|---:|---:|---:|---:|
| pure-M | 18 | 12.2% | 12.2% | 84.1% | 84.1% |
| K-split+kf | 8 | 29.4% | 13.5% | 94.8% | 46.0% |
| K-split+id | 4 | 49.3% | 48.9% | 99.4% | 96.7% |
| **all** | 30 | **21.7%** | **17.5%** | 99.4% | 96.7% |

The per-cluster fix lands cleanly on `K-split+kf` rows where the
remaining residual is dominated by other factors. It barely moves
`K-split+id` rows because those rows' dominant term is PSUM ring
traffic, not HMI bytes (see next section).

Per-cluster does not affect pure-M rows (they have k=1, so the two
formulas collapse to the same answer).

## Three residual mechanisms after the per-cluster fix

### (1) PSUM-term over-prediction — `K-split+id`

The standout: DSv3 down_proj M=2048 (1, 16, 2) + id. Predicted
33.57 ms, measured 17.07 ms (+96.7%). Decomposing the prediction:

- compute (M_per=2048, N_per=448, K_per=1024): 1.88 ms
- HMI per-cluster: 1.21 ms (+ LF 3.00 ms = 4.21 ms)
- PSUM: 16 chains × 1 send × 16 hops × 3.67 MB = **940 MB / 32 GB/s = 29.4 ms** ← dominant

Wall = max(compute, hmi+LF) + psum = max(1.88, 4.21) + 29.4 = 33.57 ms

The PSUM term assumes the SFP ring is a single 32 GB/s pipe shared
across all chains. But the ring is 32 cores × 32 GB/s/link =
~1024 GB/s aggregate when chains saturate disjoint links. For
DSv3 down_proj at (1, 16, 2)+id, the 16 chains each travel 16 hops
in parallel — under aggregate-throughput, the 940 MB takes
~0.9 ms, not 29.4 ms. That would put the prediction at ~5 ms,
much closer to the measured 17 ms (the remaining 12 ms is likely
LX-overflow re-fetch since A_per = 4 MB > 2 MB).

The same mechanism appears on every `K-split+id` row to varying
degrees. PR 1932's k_fast emission *also* benefits from this
correction — under k_fast, hops_per_send = 1 (collaborators are
adjacent), so the model already predicts low PSUM time for kf and
doesn't get bitten as hard.

**Fix sketch**: change PSUM time to bytes / (link_count ×
link_bandwidth) where link_count is `min(num_chains, ring_size)`
or similar. Or model PSUM as a per-chain latency limit
(hops × payload / link_BW) and take the max over chains since
they run in parallel.

### (2) HMI-BW under-prediction at small M — pure-M wide-K wide-N

DSv3 o_proj M=32 (32, 1, 1) pure-M: predicted 8.91 ms, measured
4.84 ms (+84.1%). HMI byte count is dominated by the K·N term
(234 MB out of 235 MB total — B is 7168 × 16384 × 2 = 234 MB).

If the model's BW were correct, the measurement would imply
HMI time = 4.84 - LF = 1.84 ms, BW = 235 MB / 1.84 ms = **128 GB/s**.
That's well above the 67 GB/s spec headline.

Possible mechanisms:

- The K·N broadcast read uses ring multicast — the HMI port serves
  one fetch and the ring distributes to 32 cores. Effective BW
  measured at the HMI port is the model's 40 GB/s, but the
  *useful* BW (data delivered to all consumers) is much higher.
- At small M, compute is so light that HMI saturation isn't reached
  — the kernel template might prefetch only the slice it needs.
- The 40 GB/s value was calibrated against shapes where HMI is the
  binding constraint; at smaller M, a different regime applies.

**Fix sketch**: model achieved HMI BW as a function of compute/HMI
ratio, or distinguish "broadcast-ring-multicast bytes" (counted
once, served at higher effective BW) from "per-core unique bytes"
(counted per consumer). The cost model already uses ring-share
accounting for *bytes*; it needs the same idea for *bandwidth*.

### (3) Catastrophic LX overflow — known from LX Phase 0

DSv3 o_proj M=2048 (1, 16, 2) + id: A_per = 32 MB (16× LX). Pred
44.39, measured 116.12 (-61.8%).
DSv3 o_proj M=2048 (1, 16, 2) + kf: same shape, kf emission. Pred
16.87, measured 31.23 (-46.0%).

Per-cluster bytes leaves these rows wrong by 30–60% because the
true cost is operand re-fetch from HMI (an LX-overflow penalty
multiplier on HMI bytes), not the per-cluster bytes count.

The LX gate from `tests/lx_fit.py` correctly identifies these as
overflow rows. Adding a multiplicative re-fetch penalty on HMI
bytes (overage_factor × HMI_bytes) would close most of this.

## What this changes about the project

The Track 2 thread starts here. The per-cluster bytes diff is
**a one-line cost-model change** with measurable lift; the PSUM
re-model is the bigger lever (closes the largest remaining single-
row error in the validation set, and the PSUM mechanism affects
every K-split row in production too, including identity emission
which the planner uses by default when k_fast isn't applicable).

A useful sequencing for Track 2 Phase 1 would be:

1. **PSUM aggregate-bandwidth fix** (likely the largest lever).
   Re-validate against the 30-row set; check whether `K-split+id`
   rows drop into the ±10% band.
2. **HMI achieved-BW model** (second largest lever, all `pure-M`
   rows benefit). Measurements may be needed to calibrate.
3. **Per-cluster bytes** ship together with PSUM fix since they
   touch adjacent code paths.
4. **LX overflow penalty** (Track 2 / LX-Phase-1 territory).

The LX-Phase-0 finding stands: the LX gate alone is narrow. But
**combined with PSUM + HMI + per-cluster fixes, the cost model
is genuinely repairable** — Phase 0 originally targeted "23% →
60% top-1 accuracy"; with all four fixes layered the validation
mean error path is 21.7% → 17.5% (per-cluster) → ~10% projected
after PSUM → ~7% projected after HMI → bounded by LX overflow on
3 rows that the gate would simply reject.

## Files

- `tests/diag_kfast_residual_phase0.py` — diagnostic script
- This doc — findings
- (cross-ref) `tests/diag_lx_overflow_phase0_findings.md` —
  upstream LX residency Phase 0
- (cross-ref) `tests/hmi_cost_model_phase0_findings.md` —
  Project B Phase 0, where the per-cluster bytes formula was
  first noted
