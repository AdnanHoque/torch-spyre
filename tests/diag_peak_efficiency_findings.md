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

- **Fixed kernel launch / scheduler overhead** that does not scale
  with problem size — most visible on the smallest shapes.
- **PSUM ring reduction on K-split**: `k-1` ring hops per output
  tile to combine partial PSUMs, not accounted for in the simple
  HBM roofline.
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
