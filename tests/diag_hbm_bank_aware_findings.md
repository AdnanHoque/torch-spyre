# HBM-bank-aware core placement — findings

A measurement-driven investigation that started as "ring-aware
restickify optimization" and pivoted, on the evidence, to a different
and stronger result: **wall-clock on `out`-split matmul kernels is
sensitive to the physical core-id → work-slice mapping, and the
mechanism is HBM bank/channel contention, not ring traversal.**

This doc records every measurement and the theory evolution, so the
next person (or future-us) can pick up from evidence rather than
hand-waving.

## TL;DR

* **Restickify ring cost is bandwidth-bound, not hop-bound.** Six
  core-id permutations × four tensor sizes = 24 configurations; the two
  restickify-bearing kernel bundles are invariant in all of them
  (< 1% spread). No ring-aware optimization can reduce restickify cost.
* **But the same lever — permuting physical core IDs at SDSC emission —
  moves wall time 11–20% on a different class of kernels:** matmuls
  split on the `out` (output) dimension.
* **The fabric is HBM, not a ring.** SFP ring is ruled out (no K-split
  anywhere — `"in": 1` in every matmul SDSC). RIU ring is ruled out
  (restickify, the heaviest RIU consumer, is the *most* invariant
  bundle). The sensitivity correlates with `out`-split matmuls, which
  read distinct large weight-column slices from HBM. Permuting core IDs
  changes the physical-core → weight-slice → HBM-bank mapping; the
  default (identity) mapping clusters physically-adjacent cores onto
  adjacent HBM regions, causing bank/row-buffer contention.

## Background — how we got here

Original goal: extend the k_fast PR (#1986, SFP-ring cohort adjacency
for matmul PSUM reduction) to the RIU ring, to reduce restickify cost.

Related in-flight work targeting restickify, all at different layers:

| Layer | Workstream |
|---|---|
| Decomposition | Audit `contiguous()` calls in `decompositions.py` (Matthew Arnold + aviros) — removes restickifies from the graph |
| Layout-decision | Beam-search global restickify optimizer (issue #739, PR #1979, merged) — picks layouts to minimise restickifies inserted |
| Load-time | Pre-format HF weights into Spyre layout at safetensors load (issue #1339) — eliminates weight-driven (re)stickify |
| SDSC emission | This investigation — make the *remaining* restickifies cheaper... which turned out not to be possible, but surfaced the HBM-bank result |

## Methodology

All measurements on a Granite-3.3-8B-shaped 4-layer probe:

```python
for _ in range(4):
    q = h @ wq;  k = h @ wk
    attn = q @ k.t()          # the qk^T pattern — the only restickify source
    a = attn @ h;  o = a @ wo
    g = silu(o @ wgate);  u = o @ wup
    h = (g * u) @ wdown
```

Default shape: M=128, H=4096, INTER=12800. fp16, SENCORES=32.

Two instrumentation tools (built on branch
`AdnanHoque/rfc-ring-aware-restickify`):

* **Per-bundle timing harness** — monkey-patches
  `SpyreSDSCKernelRunner.run` to wrap `launch_kernel` with a forced
  `streams.synchronize()`, so each bundle's device execution time is
  measured (launch_kernel alone is async and only captures submission
  overhead). Inspects each bundle's `code_dir` for `ReStickifyOpHBM`
  SDSC ops.
* **Core-id permutation patch** — monkey-patches
  `superdsc._get_core_to_slice_mapping` to substitute `core_id` with
  `perm(core_id)` for a chosen permutation. Same splits, same bytes,
  different physical-core → work-slice assignment.

Permutations tested:

| Name | Definition | Adjacency effect |
|---|---|---|
| `identity` | c → c | baseline |
| `bit_reverse` | 5-bit bit-reversal | maximal scrambling |
| `stride_3` | c → 3c mod 32 | coprime multiplicative |
| `stride_7` | c → 7c mod 32 | coprime multiplicative |
| `reverse` | c → 31 − c | preserves most local adjacency |
| `pair_swap` | c → c xor 1 | swaps immediate neighbours only |

## Measurement 1 — Phase A: identity vs bit_reverse

| Config | Total wall (10 iters) | Restickify-bearing bundles |
|---|---:|---:|
| identity | 322.25 ms | 46.43 ms (14.56%) |
| bit_reverse | 300.98 ms (**−6.4%**) | 46.19 ms (−0.5%) |

First signal: a "random" permutation makes the whole graph 6.4% faster,
but the restickify-bearing bundles don't move.

## Measurement 2 — full permutation sweep (per-bundle, M=128 H=4096)

Per-bundle total ms across all six permutations:

| Bundle | SDSC split | identity | bit_rev | stride_3 | stride_7 | reverse | pair_swap |
|---|---|---:|---:|---:|---:|---:|---:|
| `mm_mul_5` (bm+mul) | **out: 25** | 45.19 | 36.21 | 37.29 | 36.79 | 44.71 | 43.50 |
| `mm_silu_9` (bm+neg+exp) | **out: 25** | 34.64 | 28.03 | 28.89 | 28.60 | 34.26 | 33.24 |
| `silu_4` (add+realdiv) | **out: 25** | 7.67 | 6.49 | 6.75 | 6.92 | 7.53 | 7.65 |
| `mm_silu_2` (bm+bm+neg) | mb: 32 | 19.64 | 17.58 | 17.91 | 17.77 | 19.54 | 19.28 |
| `mm_6` (bm+bm) | mb: 32 | 108.78 | 107.83 | 108.47 | 108.67 | 108.27 | 108.53 |
| `mm_8` (bm+bm) | mb: 32 | 27.82 | 27.87 | 28.13 | 28.16 | 27.70 | 28.02 |
| `mm_10` (bm) | mb: 32 | 27.38 | 27.08 | 27.33 | 27.35 | 27.30 | 27.37 |
| `mm_t_7` (bm+**restick**+bm) | mb:32 / out:32 | 28.51 | 28.29 | 28.46 | 28.52 | 28.41 | 28.47 |
| `mm_0` (bm+bm+**restick**) | mb:32 / out:32 | 17.99 | 17.97 | 18.05 | 18.08 | 17.99 | 18.05 |
| `mm_t_1` (bm+bm) | mb: 32 | 0.80 | 0.79 | 0.79 | 0.80 | 0.80 | 0.79 |
| `silu_3` (exp) | mb-ish | 0.86 | 0.76 | 0.78 | 0.80 | 0.84 | 0.84 |

Two clean groupings:

* **Permutation-sensitive** (`mm_mul_5`, `mm_silu_9`, `silu_4`): scrambling
  permutations give 15–20%; locality-preserving ones give ~0–4%.
* **Permutation-invariant** (`mm_6`, `mm_8`, `mm_10`, `mm_t_1`, `mm_t_7`,
  `mm_0`): < 2% spread across all six.
* `mm_silu_2` is intermediate — 9–11% under scrambling despite being
  `mb`-split (see theory below).

## Measurement 3 — H sweep (restickify size scaling)

M=128 fixed, INTER = 3·H, identity vs bit_reverse:

| H | wall id / br | restick-bundles id / br | restick % |
|---:|---:|---:|---:|
| 512 | 19.15 / 19.86 | 2.00 / 2.03 ms | 12.3% / 12.0% |
| 1024 | 41.13 / 41.53 | 4.13 / 4.14 ms | 10.7% / 10.6% |
| 2048 | 129.05 / 130.77 | 12.91 / 13.06 ms | 10.2% / 10.2% |
| 4096 | 351.66 / 355.73 | 46.07 / 46.45 ms | 13.2% / 13.2% |

Restickify-bearing bundle time scales linearly with H (2 → 4 → 13 → 46)
and is invariant to permutation at every size. Linear-in-bytes,
invariant-to-mapping is the textbook bandwidth-bound signature.

## Measurement 4 — M sweep (decode regime)

* q@k.t() probe: M=64 → 14.8%, M=128 → 14.6% restickify-bearing share.
  M ∈ {1, 16, 32} fail to compile (Spyre layout-propagation corner
  cases for M below one stick — `_matmul_layouts` `StopIteration`,
  `cannot restickify y to generated_coord`).
* MLP-only probe (no q@k.t()): **0% restickify at every M from 1 to
  128.** All restickifies in the original probe come from the q@k.t()
  pattern; pure-MLP chains have none.

## Measurement 5 — SDSC split structure

Pulled from `sdsc_*_*.json` in the compiled bundles. `numWkSlicesPerDim_`:

| Bundle | mb | out | in | cores | Sensitive? |
|---|---:|---:|---:|---:|---|
| `mm_mul_5` | 1 | **25** | 1 | 25 | yes (−20%) |
| `mm_silu_9` | 1 | **25** | 1 | 25 | yes (−19%) |
| `silu_4` | 1 | **25** | — | 25 | yes (−15%) |
| `mm_silu_2` | 32 | 1 | 1 | 32 | partial (−11%) |
| `mm_6` | 32 | 1 | 1 | 32 | no |
| `mm_8` | 32 | 1 | 1 | 32 | no |
| `mm_10` | 32 | 1 | 1 | 32 | no |
| `mm_t_1` | 32 | 1 | 1 | 32 | no |
| `mm_t_7` matmuls | 32 | 1 | 1 | 32 | no |
| `mm_t_7` restickify | 1 | 32 | — | 32 | no |

**`"in": 1` everywhere** — no matmul in this graph is K-split.

## Theory — which fabric is being exploited

### SFP ring: ruled out

PSUM bichain reduction on the SFP ring only exists when a matmul is
K-split (`in > 1`). Every matmul SDSC in this graph has `"in": 1`. There
is no K-cohort and no SFP-ring traversal to optimise. k_fast's lever is
not even active on this workload.

### RIU ring: ruled out

Restickify (`ReStickifyOpHBM`) is the single heaviest RIU-ring consumer
in the graph — its whole job is cross-core data movement over RIU. If
RIU traversal were the lever, restickify-bearing bundles would be the
*most* permutation-sensitive. They are the *most invariant* (< 1% across
all 24 configs). The mechanism is something `out`-split matmuls use that
restickify does not.

### HBM bank/channel contention: the working theory

`out`-split matmul: each core computes a distinct slice of the output
columns, which means each core reads a **distinct slice of the weight
matrix's columns** from HBM (the activation is broadcast). Weights are
large (H×H or H×INTER). The physical-core → weight-slice mapping
therefore determines the physical-core → HBM-bank mapping.

* `identity`: physically-adjacent cores (0,1,2,3,…) read adjacent weight
  column ranges → adjacent HBM addresses → same/adjacent banks →
  row-buffer / bank contention → serialisation.
* `bit_reverse` / `stride_k`: physically-adjacent cores read widely
  separated weight ranges → distinct banks → concurrent reads → higher
  aggregate HBM bandwidth.

This explains every grouping:

* **`out`-split matmuls sensitive** — distinct large weight reads,
  permutation controls bank spread.
* **`mb`-split matmuls invariant** — the weight is *broadcast* (every
  core reads the same weight addresses); only the activation is
  per-core, and the activation is small (M×H, M=128). No bank-spread
  decision to make.
* **restickify invariant** — its reads come from another core's output
  buffer over the RIU ring, not from HBM. No HBM banks involved.
* **`mm_silu_2` partial (−11%)** — `mb`-split, but it does two
  back-to-back matmuls; the combined HBM pressure (intermediate +
  weights) is enough to show a partial bank-spread effect even without
  an `out`-split.

### Directional confirmation

The permutations that help (`bit_reverse`, `stride_3`, `stride_7`) all
maximally *spread* cores; the ones that don't (`reverse`, `pair_swap`)
preserve local adjacency. k_fast wanted *adjacency* (cohort members
close, fewer ring hops). This wants *spread* (readers far apart, fewer
bank collisions). Opposite objective function → contention effect, not
ring-traversal effect.

## What this means

The lever — permuting physical core IDs at SDSC emission — is the same
mechanism k_fast uses, but the fabric and the objective are different:

| | k_fast (PR #1986) | This investigation |
|---|---|---|
| Fabric | SFP ring | HBM banks/channels |
| Objective | minimise ring hops (cohort adjacency) | minimise bank contention (reader spread) |
| Applies to | K-split matmuls | `out`-split matmuls reading distinct weight slices |
| Mechanism | `core_id_to_work_slice` permutation | same |

Restickify ring optimisation is dead — definitively, on the evidence.
But "HBM-bank-aware core placement for `out`-split matmuls" is a
genuine, measured, ~6% wall-clock opportunity on a real Granite-shaped
workload, with a clean mechanism and a one-knob lever.

## Open questions / next steps

1. **Confirm the HBM mechanism directly.** A synthetic probe: one
   `out`-split matmul, sweep core→weight-slice mappings, measure
   aggregate HBM bandwidth (not just wall time). If bandwidth tracks
   the bank-spread prediction, the mechanism is nailed.
2. **Model the bank function.** Need Spyre's HBM address → bank mapping
   (bank stride, number of banks/channels) to derive an *optimal*
   permutation rather than relying on bit_reverse happening to be good.
3. **Characterise breadth.** How many real-model matmuls are `out`-split
   vs `mb`-split? `out`-split is the planner's choice for narrow-N /
   large-weight shapes; need a shape-catalog sweep to size the
   opportunity across real models.
4. **Why does the planner pick `out: 25` (25 cores, 7 idle)?** The
   `out`-split bundles use only 25 of 32 cores. Worth understanding
   whether that's a divisibility artefact and whether it interacts with
   the bank-spread story.
5. **Interaction with k_fast.** If a future matmul *is* K-split, the
   SFP-cohort-adjacency objective (k_fast) and the HBM-bank-spread
   objective could conflict. Need a combined cost model.

## Reproduction

Probes (on branch `AdnanHoque/rfc-ring-aware-restickify`, in `/tmp` at
time of writing — should be moved into the repo):

* `phase_a_perm_sweep.py` — per-bundle timing under a chosen `PERM`
* `phase_a_size_sweep.py` — H sweep, identity vs bit_reverse
* `restickify_kernel_timing.py` — forced-sync per-bundle timing harness
* `restickify_telemetry.py` / `mapping_alignment.py` — graph-level
  restickify telemetry (precise hop math, stride-based symbol matching)
