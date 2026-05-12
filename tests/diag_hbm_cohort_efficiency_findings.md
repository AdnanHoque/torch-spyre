# HBM cohort efficiency — findings

Consolidates the HBM saturation and multicast-cohort story from the
recent probes. Companion probes:
`tests/diag_ring_hbm_saturation_probe.py`,
`tests/diag_riu_contention_probe.py`,
`tests/diag_psum_ring_cost_probe.py`. Broader context:
`tests/diag_peak_efficiency_findings.md`. Branch:
`AdnanHoque/feat-k-fast-combined`.

## Headline

Achieved HBM bandwidth on Spyre matmul depends sharply on **multicast
cohort size**. Aggregate bus utilization grows with the number of
distinct concurrent HBM requests, not with the number of cores
fetching. Pure splits (32-way cohorts) collapse to single-channel
bandwidth (~50 GB/s achieved). Mixed-MN splits (4-8 way cohorts)
achieve 2-3x more by keeping multiple HBM banks busy in parallel. This
bank-parallelism utilization — not RIU CW/CCW contention and not PT
utilization — is the binding constraint on the remaining 20% gap to
theoretical peak.

## The saturation curve

From `diag_ring_hbm_saturation_results.txt`. Pure-N split, per-core
load fixed at 2048 KB, vary n_cores. Wall and aggregate BW as the bus
fills up:

| n_cores | per-core BW (GB/s) | aggregate BW (GB/s) | Δwall vs n=1 (ms) |
|---:|---:|---:|---:|
| 1 | 23.74 | 23.74 | +0.000 |
| 2 | 24.73 | 49.45 | -0.007 |
| 4 | 24.35 | 97.40 | -0.004 |
| 8 | 16.44 | 131.51 | +0.079 |
| 16 | 9.47 | 151.46 | +0.269 |
| 32 | 7.05 | 225.72 | +0.421 |

Theoretical HBM peak: 166 GB/s (1.3 GHz x 128 B/cycle). Max aggregate
measured: 225.72 GB/s, i.e. 136% of the nominal peak — explained by
ring multicast hiding bytes from the bus accounting (one HBM read fans
out to N cores, the bus is charged once but N cores see the data).

## The mechanism

- HBM has multiple internal channels/banks (typically 8-32 on parts of
  this generation). Requests to different banks happen in parallel;
  requests to the same bank queue serially.
- Aggregate BW = single-channel BW x number of channels actively in
  use. A single bank delivers roughly 25 GB/s on this part; eight
  banks in parallel deliver ~200 GB/s.
- When N cores request the **same** data (multicast cohort of size N),
  they all hit ONE bank. The bus serves the line once, ring fans it
  out to N cores. Other banks are idle. Aggregate BW caps at one bank.
- When N cores request **distinct** data (unique addresses), the
  requests spread across N banks. All banks active in parallel.
  Aggregate BW saturates the bus.
- Net: 32-way multicast yields ~25 GB/s per fragment (single bank);
  32 distinct requests yield ~225 GB/s aggregate (bus saturated).
- The wall-time penalty for a pure 32-way cohort is therefore not "the
  bus is slow" but "we only used 1/8 of the bus".

## Observed on real shapes (M=512)

From `/tmp/riu_contention_results.txt`, Llama 3.1 8B q_proj
(M=512, N=4096, K=4096). All splits move the same theoretical traffic
(42 MB), so wall-time differences are pure cohort effects:

| split | mc pattern | distinct fragments | wall ms | achieved BW (GB/s) |
|---|---|---:|---:|---:|
| (32, 1, 1) | A unique / B mc=32 | 1 (B) + 32 (A) but A is per-core | 0.852 | 49 |
| (1, 32, 1) | A mc=32 / B unique | 32 (B) + 1 (A) | 0.658 | 64 |
| (4, 8, 1) | A mc=8 / B mc=4 | 4 (B) + 8 (A) = 12 | 0.355 | 118 |
| (8, 4, 1) | A mc=4 / B mc=8 | 8 (B) + 4 (A) = 12 | 0.412 | 102 |

Granite-8B q_proj at the same M=512 shape replicates this within
~1 μs (0.852 / 0.657 / 0.355 / 0.410), so the cohort effect is
shape-property not implementation noise. The 2.4x wall-time gap
between (32, 1, 1) and (4, 8, 1) on identical traffic volume is
entirely from how many HBM banks the request pattern lights up at
once.

## Cohort-efficiency rule of thumb

- **4-8 way cohorts** achieve near-peak per-fragment BW. Multiple
  banks active, multicast still amortizing the bytes-per-line.
- **16+ way cohorts** collapse toward single-bank rate. Few distinct
  in-flight requests, most banks idle.
- The planner should prefer splits whose A-multicast and B-multicast
  cohorts both sit in the 4-8 range, i.e. **maximize the count of
  distinct in-flight HBM fragments** rather than maximize multicast
  fan-out for byte-amortization. The fan-out wins less than the bank
  parallelism does.

## What this rules out / supersedes

- **RIU CW/CCW direction separation** was the earlier hypothesis
  (`diag_riu_contention_probe.py`) and is empirically rejected on the
  M=512 shapes. If ring direction contention were the binding
  constraint, mixed-MN splits (which use both CW and CCW
  simultaneously) would be **slower** than pure splits (which use one
  direction). Observed: mixed-MN is 2.4x **faster** on identical
  traffic. The probe's `wall_mixed ~ max(pure-M, pure-N)` outcome
  confirms ring directions are already operating independently — the
  wall-time differences come from HBM bank parallelism, not RIU.
- **PT utilization** was another earlier framing and is also rejected.
  All four splits in the M=512 table above have identical per-core PT
  cycles (~524K cycles); compute time alone is identical across
  splits. The 2.4x wall-time gap is on the HBM side, not the PT side.
- **PSUM ring cost** under k_fast contributes only ~6% median /
  ~11% mean to the 20-30% peak gap
  (`diag_psum_ring_cost_probe.py`, summarized in
  `diag_peak_efficiency_findings.md`). The dominant share of the
  remaining gap is HBM bank-parallelism utilization, not ring.

## Implication for future optimization

- The remaining 20-30% gap to peak is attacked by **cohort-size-aware
  planner heuristics**, not by ring-aware permutations or RIU
  direction-separation work in deeptools.
- Concretely: pick splits that maximize the count of distinct
  in-flight HBM fragments (A-multicast cohort size + B-multicast
  cohort size, capped near 8 each), subject to PT and LX constraints
  that the current planner already enforces.
- This is a planner-side change in `torch_spyre/_inductor/work_division.py`,
  not a deeptools-side change — it is actionable from torch-spyre
  alone, with no compiler-side dependency.
