# Peak efficiency probe — findings

Companion to `diag_peak_efficiency_probe.py`. Raw probe output
captured at `/tmp/peak_efficiency_results.txt`. Branch:
`AdnanHoque/feat-k-fast-combined`.

## Headline

Across 14 production matmul shapes, mean efficiency is **~76% of
theoretical peak**, with **K-split + k_fast** (the path PR 1986 adds)
hitting **80.1% mean** in its M=32-128 firing zone. This is in the
same band as well-tuned GPU matmul kernels on comparable shapes, so
the heuristic is already extracting most of the available PT and HBM
headroom — further work-division tuning has diminishing returns.

## Methodology

- **Theoretical peak** is taken as `max(compute_bound, hbm_bound)` —
  whichever side of the roofline the shape lives on.
- **Compute bound:** `2 · M · N · K / 72.1 TFLOPS`, where 72.1 TFLOPS
  is the published fp16 PT peak and the leading 2 counts FMA as two
  ops per MAC.
- **HBM bound:** `(M·K + K·N + k·M·N) · 2 bytes / 166 GB/s`. Full A
  and B multicast is the verified Spyre traffic pattern; the third
  term covers the `k` partial PSUMs that have to land on C when the
  K dimension is split `k>1` ways.
- **% peak** = `theoretical_ms / observed_ms · 100`. Observed ms is
  the device wall time averaged over 8 iterations after 2 warmups,
  fp16, SENCORES=32.
- The probe evaluates both K-split families (with k_fast emitted) and
  mixed-MN families (with identity permutation) for each shape, and
  reports the **faster wall time per shape** along with which split
  family produced it.

## Per-shape results

| shape | M | N | K | best split | best ms | compute ms | hbm ms | theor ms | % peak | bound |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| L3.1-8B q_proj M=32 | 32 | 4096 | 4096 | (2, 8, 2) | 0.268 | 0.015 | 0.207 | 0.207 | 77.1% | HBM |
| Granite-8B q_proj M=32 | 32 | 4096 | 4096 | (2, 8, 2) | 0.268 | 0.015 | 0.207 | 0.207 | 77.1% | HBM |
| Granite-8B gate M=32 | 32 | 12800 | 4096 | (2, 8, 2) | 0.742 | 0.047 | 0.643 | 0.643 | 86.6% | HBM |
| L3.1-8B q_proj M=128 | 128 | 4096 | 4096 | (2, 8, 2) | 0.287 | 0.060 | 0.221 | 0.221 | 76.9% | HBM |
| Granite-8B gate M=128 | 128 | 12800 | 4096 | (2, 8, 2) | 0.813 | 0.186 | 0.677 | 0.677 | 83.4% | HBM |
| Granite-8B down M=128 | 128 | 4096 | 12800 | (2, 8, 2) | 0.802 | 0.186 | 0.664 | 0.664 | 82.8% | HBM |
| L3.2-3B gate M=128 | 128 | 8192 | 3072 | (4, 4, 2) | 0.433 | 0.089 | 0.333 | 0.333 | 77.0% | HBM |
| L3.1-8B q_proj M=512 | 512 | 4096 | 4096 | (4, 8, 1) | 0.356 | 0.238 | 0.253 | 0.253 | 71.0% | HBM |
| Granite-8B gate M=512 | 512 | 12800 | 4096 | (4, 8, 1) | 1.064 | 0.745 | 0.736 | 0.745 | 70.0% | compute |
| DSv3 q_b_proj M=512 | 512 | 24576 | 1536 | (8, 4, 1) | 1.049 | 0.536 | 0.616 | 0.616 | 58.7% | HBM |
| Mixtral gate M=1024 | 1024 | 16384 | 6144 | (8, 4, 1) | 3.496 | 2.859 | 1.491 | 2.859 | 81.8% | compute |
| Qwen-14B kv M=2048 | 2048 | 2048 | 5120 | (8, 4, 1) | 0.841 | 0.596 | 0.303 | 0.596 | 70.8% | compute |
| L3.1-70B q M=2048 | 2048 | 8192 | 8192 | (8, 4, 1) | 5.022 | 3.812 | 1.213 | 3.812 | 75.9% | compute |
| Granite-8B gate M=2048 | 2048 | 12800 | 4096 | (4, 8, 1) | 4.195 | 2.978 | 1.049 | 2.978 | 71.0% | compute |

## Aggregates

| group | shapes | mean %peak | median %peak | min | max |
|---|---:|---:|---:|---:|---:|
| family K | 7 | 80.1% | 77.1% | 76.9% | 86.6% |
| family MN | 7 | 71.3% | 71.0% | 58.7% | 81.8% |
| bound: HBM | 9 | 76.7% | 77.1% | 58.7% | 86.6% |
| bound: compute | 5 | 73.9% | 71.0% | 70.0% | 81.8% |

The K-split family (k_fast firing zone) sits ~9pp above the mixed-MN
family on this suite. The HBM-bound and compute-bound buckets are
within ~3pp of each other, which says the gap to peak is not a
single dominant inefficiency on one side of the roofline — both
sides leave similar headroom on the table.

## Where the 20-30% gap goes

The candidates below are the architectural overheads not folded into
the simple roofline. The PSUM-ring share is now quantified directly in
the next section ("PSUM ring cost contribution"): under k_fast it
contributes ~6% of the remaining gap on median, ~11% on mean. The
bulk of the gap therefore lives in the non-ring items — launch
overhead, HBM-below-peak, and PT pipeline fill.

- **Fixed kernel launch / scheduler overhead** that does not scale
  with problem size — most visible on the smallest shapes.
- **PSUM ring reduction on K-split**: `k-1` ring hops per output
  tile to combine partial PSUMs, not accounted for in the simple
  HBM roofline. Quantified in the next section — small residual
  after k_fast.
- **PT array pipeline fill latency** — the systolic array spends a
  prefix and suffix of cycles partially populated; the `2·M·N·K`
  count assumes a fully steady-state pipe.
- **HBM efficiency below catalog peak** — the ~166 GB/s per-direction
  figure is an upper bound; sustained read+write traffic is
  typically lower, and effective aggregate throughput (~225 GB/s
  measured) is gated by burst sizes and access patterns.
- **C writeback to HBM for `k>1`** — partial PSUMs have to be
  materialized on chip then merged out; the `k·M·N` term in the HBM
  bound captures the volume but not the serialization.

## PSUM ring cost contribution

Companion probe: `tests/diag_psum_ring_cost_probe.py`. Raw results at
`/tmp/psum_ring_cost_results.txt`. **PSUM ring under k_fast is ~6% of
the remaining peak gap on median, ~11% on mean.** The 20-30% gap to
peak (previous section) is mostly NOT ring — it is launch overhead,
HBM-below-peak, and PT pipeline fill.

### Methodology

- For each (shape, K-split), measure wall with identity (K-cohort at
  physical ring distance `mn`) and with k_fast (K-cohort at distance
  1). `Δ = wall_id − wall_kf` is the ring-cost difference, equal to
  `(mn − 1) · ring_cost_per_unit_distance`.
- Residual ring under k_fast is estimated as
  `kf_ring_est = Δ / (mn − 1)` — the per-hop cost scaled to the
  single hop k_fast still pays.
- `gap_to_peak = wall_kf − theoretical_peak_ms` and
  `ring/gap = kf_ring_est / gap_to_peak × 100` gives the share of
  the remaining peak gap attributable to PSUM ring under k_fast.

### Per-shape results

| shape | split | mn | id wall ms | kf wall ms | Δ ms | kf ring est | peak ms | gap to peak | ring/gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L3.1-8B q_proj M=32 | (2, 8, 2) | 16 | 0.284 | 0.271 | 0.013 | 0.001 | 0.207 | 0.064 | 1.3% |
| Granite-8B q_proj M=32 | (2, 8, 2) | 16 | 0.279 | 0.268 | 0.011 | 0.001 | 0.207 | 0.061 | 1.2% |
| Granite-8B gate M=32 | (2, 8, 2) | 16 | 0.755 | 0.737 | 0.018 | 0.001 | 0.643 | 0.094 | 1.3% |
| L3.1-8B q_proj M=128 | (2, 8, 2) | 16 | 0.368 | 0.289 | 0.079 | 0.005 | 0.221 | 0.068 | 7.7% |
| Granite-8B gate M=128 | (2, 8, 2) | 16 | 1.641 | 0.810 | 0.831 | 0.055 | 0.677 | 0.133 | 41.8% |
| Granite-8B down M=128 | (2, 8, 2) | 16 | 1.016 | 0.801 | 0.215 | 0.014 | 0.664 | 0.137 | 10.5% |
| L3.2-3B gate M=128 | (4, 4, 2) | 16 | 0.869 | 0.431 | 0.438 | 0.029 | 0.333 | 0.097 | 30.0% |
| L3.1-8B q_proj M=32 k4 | (1, 8, 4) | 8 | 0.286 | 0.294 | -0.009 | -0.001 | 0.210 | 0.084 | -1.5% |
| L3.1-8B q_proj M=128 k4 | (1, 8, 4) | 8 | 0.362 | 0.325 | 0.037 | 0.005 | 0.234 | 0.092 | 5.8% |

### Aggregates

| metric | value |
|---|---:|
| mean kf ring cost | 12 μs |
| mean peak gap | 92 μs |
| mean ring/gap | 10.9% |
| median ring/gap | 5.8% |

### Interpretation

- k_fast has already absorbed the bulk of ring cost. The Δ column
  (5 μs to 831 μs) measures the savings k_fast delivers *over*
  identity placement at the same (m, n, k) — that is the ring cost
  k_fast has eliminated, not the cost that remains.
- Residual ring cost under k_fast is tiny in absolute terms — mean
  12 μs per kernel — and the median ring/gap of 5.8% says that for
  half of the suite ring is essentially negligible on top of the
  rest of the gap.
- **Outlier**: Granite-8B gate M=128 at **41.8%** is the only shape
  where residual ring is a meaningful fraction of the remaining
  gap. L3.2-3B gate M=128 at **30.0%** is similar. Both are wide-N
  gate shapes with many output tiles per core, where the single
  remaining hop is still amortized over a lot of PSUM traffic.
- For the other seven shapes, further ring-side optimization (PSUM
  batching, mkfast-composed variants, more aggressive cohort
  placement) has diminishing returns — there is simply not much
  ring time left to recover.
- This empirically validates the priority of HBM-efficiency and
  launch-overhead optimizations over additional ring tuning: the
  big remaining contributors to the 20-30% gap are off the ring.

### Implication for future optimizations

Ring-aware optimizations had a big payoff with k_fast — mean Δ of
~200 μs saved per kernel, max 831 μs (Granite-8B gate M=128). With
that win banked, further pure-ring optimizations have small absolute
upside: a residual mean of 12 μs is the ceiling on what more ring
tuning can recover on this suite. The 20-30% gap is dominated by HBM
bandwidth efficiency below catalog peak, kernel launch / scheduling
overhead, and PT pipeline fill. Next-round optimization candidates
should attack those: ring-aware activation fusion (eliminates HBM
round-trips on the activation side), Q/K/V multicast fusion (cuts
per-kernel launch overhead by collapsing three matmuls into one
dispatch), or RIU CW/CCW direction separation (raises effective HBM
bandwidth toward the catalog 166 GB/s/direction figure).

## Best and worst cases

- **Best overall:** Granite-8B gate at M=32 — **86.6%** of peak. HBM-
  bound, small-M, narrow-K with wide-N: exactly the regime the
  k_fast permutation is built for.
- **Best compute-bound:** Mixtral gate at M=1024 — **81.8%** of peak.
  Says the PT array sustains ~59 TFLOPS effective on a large-M
  prefill shape, which is a healthy fraction of the 72.1 TFLOPS peak.
- **Worst overall:** DSv3 q_b_proj at M=512 — **58.7%** of peak.
  Wide-N narrow-K (`N=24576, K=1536`) hits the HBM roofline but
  underperforms it by ~17pp vs the rest of the HBM-bound bucket.
  Worth a targeted drilldown — possible culprits are B-tile reuse
  in this aspect ratio, or `(8, 4, 1)` not being the empirical
  optimum here.

## Strategic interpretation

- **PR 1986's K-split + k_fast path is already at the efficiency
  band that well-tuned matmul kernels on commodity GPUs achieve**
  (~50-80% on similar shapes). 80% mean in the firing zone is a
  strong number, not a starting point.
- **Mixed-MN family at 71% mean is solid for large-M prefill** but
  carries more headroom than the K-split path; further mixed-MN
  tuning is a higher-leverage knob than re-tuning K-splits.
- **Diminishing returns on heuristic tuning at the aggregate
  level** — to move the suite mean meaningfully above ~76% we would
  need to attack one of the architectural overheads above (PSUM
  reduction cost, pipeline fill, HBM access patterns), not pick
  different `(m, n, k)` splits.
- **DSv3 q_b_proj is the outlier worth investigating** — a single
  shape that is ~17pp below its bucket median. Cheap to probe with
  a full sweep of valid splits at this exact `(M, N, K)`.
- **GPU efficiency comparison:** for context, NVIDIA cuBLAS hgemm on
  comparable mid-size fp16 shapes typically lands in the 50-80% of
  peak band depending on shape, and only the largest fully-tiled
  GEMMs reach ~90%. Spyre at 80% on the K-split firing zone is in
  the same regime, not below it.
