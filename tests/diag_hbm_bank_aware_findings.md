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

* **Restickify cost is bandwidth-bound, not hop-bound — for both
  activation and weight restickify.** Six core-id permutations × four
  tensor sizes = 24 configurations; the activation-restickify-bearing
  bundles are invariant in all of them (< 1% spread). A later probe
  (Measurement 7) confirmed the same for *weight* restickify, and found
  the permutation lever structurally cannot even be applied to it. No
  ring-aware optimization can reduce restickify cost.
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

The permutations that help (`bit_reverse`, `stride_3`, `stride_7`,
`cluster`) all maximally *spread* cores; the ones that don't (`reverse`,
`pair_swap`) preserve local adjacency. k_fast wanted *adjacency* (cohort
members close, fewer ring hops). This wants *spread* (readers far apart,
fewer bank collisions). Opposite objective function → contention effect,
not ring-traversal effect.

## Measurement 6 — direct HBM bandwidth probe (isolated `out`-split matmul)

Isolated `x(128,4096) @ W(4096,N)`, no surrounding ops. Effective HBM
bandwidth = bytes-touched / per-call wall time. Sweep N and permutation.

First the **`mb`-split vs `out`-split transition**, root-caused by
diffing the compiled SDSC for N=8192 vs N=12800:

| N | N/64 sticks | split (verified from SDSC) | regime |
|---:|---:|---|---|
| 1536–8192 | 24–128 | `mb`: 32 | M is bigger output dim → M-split |
| 12800+ | 200+ | `out`: 25/32/… | N-in-sticks > M=128 → N-split |

* **`mb`-split is broadcast-bound (~37 GB/s, flat, permutation-invariant).**
  Per-call time scales perfectly linearly with weight bytes from N=1536
  to N=8192. Every core needs the full weight; it goes over the on-chip
  broadcast fabric whose bandwidth is the ceiling. No per-core slice →
  no permutation lever.
* **`out`-split is HBM-bandwidth-bound (~70–145 GB/s, permutation-sensitive).**
  Each core reads its own distinct weight column-slice straight from HBM.
  Aggregate bandwidth = bank parallelism = a function of the
  core→slice→address→bank mapping.

Then the **core-count effect within the `out`-split regime** (all N here
have N/64 > 128, so all are genuinely `out`-split):

| N | `out`-split | identity | cluster | bit_reverse | headroom |
|---:|---:|---:|---:|---:|---:|
| 10240 | **32** | 118.7 | 119.3 | 117.4 | ~0% |
| 12288 | **32** | 72.3 | 73.6 | 72.3 | ~2% |
| 16384 | **32** | 124.4 | 124.9 | 121.7 | ~0% |
| 24576 | **32** | 70.4 | 70.0 | 70.3 | ~0% |
| 9600 | 30 | 112.8 | 118.4 | 119.5 | **+6%** |
| 13312 | 26 | 108.5 | 140.7 | 130.5 | **+30%** |
| 12800 | 25 | 106.9 | 145.6 | 134.3 | **+36%** |
| 11264 | 22 | 102.0 | 132.3 | 144.0 | **+41%** |

(GB/s effective; "headroom" = best non-identity vs identity.)

**Every `out:32` case is flat. Every non-power-of-2 core count
(`out:22/25/26`) swings 30–41%.** `out:30` is mildly sensitive (+6%) —
30 is "close to" 32.

Two further measured details with design implications:

1. **The optimal permutation is core-count-dependent.** `bit_reverse`
   wins at `out:22` (144 GB/s); `cluster` wins at `out:25` and `out:26`.
   The fix is not "apply a fixed permutation" — it is "compute the
   bank-aligned permutation for *this* core count".
2. **A secondary sticks-per-core effect exists even within `out:32`.**
   N=12288 (6 sticks/core) and N=24576 (12 sticks/core) sit at ~70 GB/s
   while N=10240 (5/core) and N=16384 (8/core) reach ~120 GB/s. This is
   a separate intra-core access-pattern effect, out of scope for the
   permutation lever but worth noting.

## Measurement 7 — weight restickify

Measurements 1–5 covered *activation* restickify (the `ReStickifyOpHBM`
from the q@k.t() pattern). A separate probe tested *weight* restickify —
the restickify inserted when a matmul's weight tensor is not already in
the layout the matmul wants. A maintainer flagged this as a cost that
"happens during all matmuls", and since it is a large HBM read it was a
plausible second target for the bank lever.

Workload: `x(M,K) @ W.t()` with `W` shaped `(N,K)` reliably forces a
weight `ReStickifyOpHBM` on the `(N,K)` weight tensor (~100 MB at
N=12800).

Findings:

* **Never its own bundle.** The planner always fuses the weight
  restickify into the consuming matmul's bundle (`ReStickifyOpHBM` +
  `batchmatmul`). With two consumers of the same weight it is duplicated
  per consumer, still fused — never hoisted standalone, so it cannot be
  timed in isolation.
* **Same non-power-of-2 stranding on paper.** Its SDSC split is
  `mb:<ncores>, out:1`, with core count = the largest divisor of N/64
  that is ≤ 32 — the same stranding as `out`-split matmuls. So it looked
  like a candidate.
* **But permutation-invariant.** At N=12800 the matmul *alone* swings
  +86% under `bit_reverse` (~0.83 ms absolute); the fused
  restickify+matmul bundle swings only +24% (~0.68 ms absolute) — the
  *same* absolute swing, the percentage just shrinks by dilution. The
  restickify contributes ~1.8 ms/call of essentially flat time and ~0 ms
  of additional permutation sensitivity.
* **The lever structurally cannot be applied.** At non-power-of-2 core
  counts (`mb:22/25`) every non-identity permutation that isn't closed
  on `[0,ncores)` maps a core_id outside the used set → bundler SIGABRT
  (`"Workslice information for coreId=23 was not found"`). The
  permutation infrastructure assumes a full-32 bijection; restickify
  ops use sub-32 core sets.

**Conclusion:** weight restickify is bandwidth-bound and
permutation-invariant, exactly like activation restickify. It is a real,
sizable HBM cost (~1.8 ms/call for a 100 MB weight, dominating the
bundle), but it is a large *sequential streaming* read that saturates
bandwidth regardless of the core → slice mapping. The bank-contention
lever needs many cores issuing *distinct concurrent* reads whose address
spread you control — that is the `out`-split matmul pattern, not a
restickify. Weight restickify should be attacked by *eliminating* it
(load-time weight pre-formatting, issue #1339; layout-decision
optimizer, PR #1979), not by core-id permutation.

## Measurement 8 — forced K-split on M=1 decode

The scope/impact claim below rests on decode (M=1) staying `out`-split.
A probe checked the alternative: force a K-split on the M=1 down_proj
shape (Granite-3.3-8B, M=1, K=12800, N=4096), bypassing the planner gate
that normally keeps it `out:32`.

| Forced split | per-call | Δ vs `out:32` baseline (0.737 ms) |
|---|---:|---:|
| `out:32, in:1` (baseline) | 0.737 ms | — |
| `out:16, in:2` | 0.794 ms | **+7.7%** |
| `out:8, in:4` | 0.785 ms | **+6.5%** |
| `out:4, in:8` | 0.854 ms | **+15.9%** |

Every forced K-split is *slower*, and latency degrades monotonically
with reduction depth. At M=1 the per-slice partial product is a single
PT row — the matmul compute is trivially cheap, so a K-split only adds
an SFP-ring PSUM reduction with nothing to amortise it against. This
confirms decode should stay `out`-split: the HBM-bank lever, not
K-split, is the right tool for the decode regime.

## What this means

The lever — permuting physical core IDs at SDSC emission — is the same
mechanism k_fast uses, but the fabric and the objective are different:

| | k_fast (PR #1986) | This investigation |
|---|---|---|
| Fabric | SFP ring | HBM banks/channels |
| Objective | minimise ring hops (cohort adjacency) | minimise bank contention (reader spread) |
| Applies to | K-split matmuls | `out`-split matmuls with non-power-of-2 core counts |
| Mechanism | `core_id_to_work_slice` permutation | same |

**The lever is now precisely characterised, by measurement:**

* Restickify ring optimisation is dead — RIU is bandwidth-bound, and
  restickify-bearing bundles are invariant across all permutations and
  sizes tested. This holds for both activation restickify
  (Measurements 1–5) and weight restickify (Measurement 7); the lever
  structurally cannot even be applied to restickify SDSCs at
  non-power-of-2 core counts.
* `mb`-split matmuls are broadcast-bound — not a target.
* **`out`-split matmuls with a power-of-2 core count** already have a
  near-optimal default mapping — not a target.
* **`out`-split matmuls with a non-power-of-2 core count** lose **30–41%
  of HBM bandwidth** to bank misalignment under the default identity
  mapping. A bank-aware permutation recovers it. **This is the target.**

Trigger condition (cheap to detect at SDSC emission): the op is
`out`-split AND `numCoresUsed_` is not a power of two.

Scope/impact: `out`-split is the planner's choice whenever N-in-sticks
exceeds M — i.e. **all of decode** (M=1 → every matmul is N-split) and
the large-N projections in prefill. The non-power-of-2 core count
happens whenever N/64 has no divisor of 32 — and **Granite-3.3-8B's
INTER=12800 lands exactly on the `out:25` bad case** (+36% bandwidth
left on the table). This is a real production shape, not synthetic.

## Open questions / next steps

1. ~~**Confirm the HBM mechanism directly.**~~ **DONE — Measurement 6.**
   Mechanism confirmed: `out`-split matmuls with non-power-of-2 core
   counts lose 30–41% HBM bandwidth to bank misalignment.
2. **Model the bank function.** The optimal permutation is core-count-
   dependent (`bit_reverse` best at `out:22`, `cluster` best at
   `out:25/26`). Need Spyre's HBM address → bank mapping (bank stride,
   number of banks/channels) to *derive* the bank-aligned permutation
   for any core count rather than picking from a fixed menu. This is
   the key blocker for a principled optimisation.
3. **Characterise breadth.** How many real-model matmuls hit the bad
   case (`out`-split AND non-power-of-2 core count)? Need a shape-
   catalog sweep across production models. Granite-3.3-8B INTER=12800
   is confirmed bad (`out:25`); need DSv3, Llama, Mixtral dims checked.
4. **Why does the planner pick `out: 25` (25 cores, 7 idle)?** The
   `out`-split core count is the largest divisor of N-in-sticks that is
   ≤ 32. For N/64 with no good divisor near 32, this strands cores AND
   creates the bank misalignment. Worth asking whether the planner
   should prefer a *worse* core count that is bank-friendlier — i.e.
   the bank-alignment objective may belong in core division, not just
   in core-id emission.
5. **Secondary sticks-per-core effect.** Even `out:32` shows a 70 vs
   120 GB/s split by sticks-per-core (6/12 slow, 5/8 fast). Separate
   from the permutation lever, but a second HBM-access-pattern effect
   worth its own investigation.
6. **Interaction with k_fast.** If a future matmul *is* K-split, the
   SFP-cohort-adjacency objective (k_fast) and the HBM-bank-spread
   objective could conflict. Need a combined cost model. See
   "Interaction with the Split-K PR" below.

## Interaction with the Split-K PR (#1986)

k_fast / Split-K and this HBM-bank lever are **non-overlapping today**
and **complementary, not conflicting, after #1986 merges:**

* **No overlap on the current workload.** Every matmul SDSC in the probe
  graph is `"in": 1` — nothing is K-split, so k_fast's SFP-cohort lever
  is inert here. This HBM-bank lever only fires on `out`-split matmuls.
  An op is either `out`-split or `in`-split for a given dim; the two
  levers target disjoint kernel populations.
* **#1986 changes *which* matmuls are K-split, not the bank result.**
  k_fast's M-range gate is `1 <= rows_per_core <= 2*_PT_ROWS` (where
  `rows_per_core = M / max_cores`), and its wide-N gate
  `rows_per_core > _PT_ROWS/2 and n_sticks >= max_cores` only rejects
  wide-N when **M > 128**. So for M in roughly `[32, 128]`, k_fast grabs
  *wide-N* matmuls too and moves them from `out`-split into `in`-split.
  That removes a substantial chunk of the **M <= 128 prefill** out-split
  population — not just narrow-N shapes. What it leaves untouched:
  * **All of decode (M=1):** `rows_per_core = 1/max_cores < 1` → k_fast's
    first gate rejects it. Decode matmuls stay `out`-split. This is the
    HBM lever's core territory and #1986 does not contest it.
  * **M > 128 wide-N prefill:** the wide-N gate rejects it → stays
    `out`-split.
* **The shared mechanism is the merge risk.** Both levers are the same
  `_get_core_to_slice_mapping` / `core_id → work_slice` permutation. If a
  matmul is *both* K-split (wants SFP cohort adjacency) *and* reads
  distinct weight slices per cohort (wants HBM bank spread), the two
  objectives pull the permutation in opposite directions — k_fast wants
  adjacency, this wants spread. That case does not exist in the current
  graph, but a combined cost model is needed before both land. Whoever
  merges second must not blindly overwrite the other's mapping.
* **Recommended sequencing.** #1986 should merge first (it is further
  along and has the cost-model scaffolding). The HBM-bank permutation
  should be built *on top of* #1986's mapping API, as a second term in
  the same cost model, gated on the `out`-split + non-power-of-2 trigger
  so it never touches the kernels k_fast owns.

## Reproduction

Probes (in `/tmp` at time of writing — should be moved into the repo):

* `hbm_bandwidth_probe.py` — isolated `out`-split matmul
  `x(M,K) @ W(K,N)`, monkey-patches `_get_core_to_slice_mapping` for the
  permutation and `SpyreSDSCKernelRunner.run` for forced-sync timing;
  reports effective HBM bandwidth. Env: `PERM`, `PROBE_N/M/K`.
* `hbm_outsplit_sweep.sh` — drives `hbm_bandwidth_probe.py` over the
  guaranteed-`out`-split N set × {identity, cluster, bit_reverse};
  produces the Measurement 6 core-count table.
* `hbm_corecount_sweep.py` — earlier non-power-of-2 sweep (also covers
  the `mb`-split sizes that turned out broadcast-bound).
* `diff_kernels.py` — compiles `x@W` for a given N, dumps
  `numWkSlicesPerDim_` / `numCoresUsed_` / `coreIdToWkSlice_` /
  `segment_size.json`; used to root-cause the `mb`→`out` transition.
* `weight_restickify_probe.py` — forces a weight `ReStickifyOpHBM` via
  `x @ W.t()`, sweeps permutations on the fused bundle; Measurement 7.
* `down_proj_kfast_probe.py` / `down_proj_kfast_force.py` — Measurement 8:
  k_fast on/off and forced-K-split sweeps on the M=1 decode down_proj
  shape.
* `phase_a_perm_sweep.py` — per-bundle timing under a chosen `PERM`
* `phase_a_size_sweep.py` — H sweep, identity vs bit_reverse
* `restickify_kernel_timing.py` — forced-sync per-bundle timing harness
* `restickify_telemetry.py` / `mapping_alignment.py` — graph-level
  restickify telemetry (precise hop math, stride-based symbol matching)
