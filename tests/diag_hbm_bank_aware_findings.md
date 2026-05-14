# Core-id permutation for out-split matmuls — negative result

This investigation chased two permutation-based optimizations: first
"ring-aware restickify", then "HBM-bank-aware core placement". **Both are
negative.** The HBM-bank result looked positive for a while and even had
a quantified headline number — but that number was traced to a
measurement artifact in the probe.

This doc records the artifact prominently so it is not repeated, the
theory evolution that led there, and the structural facts that *do*
survive — including one genuinely unexplained effect still worth chasing.

## TL;DR

* **The core-id → work-slice permutation lever does not change out-split
  matmul bandwidth.** Valid (bijective) permutations are flat — < 2%
  spread — at every core count tested, power-of-2 or not.
* **The earlier "30–41% headroom from a bank-aware permutation" result
  was an artifact.** The probe's permutations were `% 32` maps. The
  out-split slice map is `Mod(core_id, C)`; for a non-power-of-2 core
  count `C`, `Mod(perm32(c), C)` is **not a bijection** — it silently
  skips ~20% of the output slices and double-computes others. The probe
  never checked numerical correctness, and buffer reuse across compiles
  left stale-but-correct values in the un-written slices. The "speedup"
  was a kernel computing ~20% *less*.
* **Restickify ring optimization** (the original goal) is also negative:
  bandwidth-bound, permutation-invariant — and at non-power-of-2 core
  counts the permutation infra can't even be applied (it aborts the
  bundler).
* **What survives:** the structural `mb`-split vs `out`-split distinction,
  the restickify measurements (those bundles were power-of-2, so the
  `% 32` perms happened to be valid there), the forced-K-split-on-decode
  result, the AIU HMI topology — and **one real, characterized effect**:
  out-split bandwidth swings ~70 vs ~120 GB/s with sticks-per-core,
  predicted by `oddpart(sticks_per_core) ∈ {3,7}`. Measured under identity,
  uncontaminated, K-independent, 12/12 fit — and potentially actionable in
  the planner's core-count selection.

## The artifact — why Measurements 1, 2, and 6 are invalid

The out-split slice map the planner emits is, verified from the compiled
SDSC:

```
iteration_space = {mb: 128, out: 12800, in: 4096}
dim_splits      = {mb: 1, out: 25, in: 1}     num_cores = 25
core_id -> slice:  Mod(core_id, 25)
```

The original probe (`hbm_bandwidth_probe.py`) applied permutations by
substituting `core_id -> perm(core_id)` into that expression. But every
permutation it defined — `bit_reverse`, `stride_3`, `stride_7`,
`cluster`, `reverse`, `pair_swap` — was a map on `[0, 32)`. For a
power-of-2 core count (`out:32`) those are valid bijections. For a
**non-power-of-2** count (`out:22/25/26/30`), `Mod(perm32(c), C)` is
**not a bijection on `[0, C)`** — some slices get two cores, others get
none.

`coreIdToWkSlice_` dumped from the compiled bundle under the `cluster`
patch at N=12800 (C=25) proves it directly:

```
25 cores, out-slice per core:
  [0,8,16,24,1,9,17,0,2,10,18,1,3,11,19,2,4,12,20,3,5,13,21,4,6]
  unique slices: 20 / 25      missing: {7, 14, 15, 22, 23}
```

Five output column-bands are **never computed**. The probe did not check
correctness, and across its repeated compiles the output buffer was
reused — so the missing slices still held correct values from a previous
(identity) run. The kernel "ran 20% faster" because it did 20% less work.

### The verification

`verify_measurement6.py` re-ran N=12800 with explicit bijection checks
and a correctness check:

| PERM | bijection on [0,25)? | slices computed | per-call | eff BW | verdict |
|---|---|---:|---:|---:|---|
| identity | yes | 25 | 1.016 ms | 107.5 GB/s | OK |
| stride:2 | yes | 25 | 1.035 ms | 105.4 GB/s | OK |
| stride:3 | yes | 25 | 1.022 ms | 106.8 GB/s | OK |
| stride:7 | yes | 25 | 1.019 ms | 107.1 GB/s | OK |
| cluster | **no** | **20** | 0.755 ms | 144.6 GB/s | skips 5 slices |
| bit_reverse | **no** | **20** | 0.818 ms | 133.5 GB/s | skips 5 slices |

(The `cluster`/`bit_reverse` numbers match Measurement 6's 145.6 / 134.3
— they are the same artifact.)

**Every valid bijection is flat (105–107 GB/s, ≤ 2% spread).** The
"non-power-of-2 core counts are special" story collapses: power-of-2
counts looked flat because the `% 32` perms were valid bijections there
*and* the lever does nothing; non-power-of-2 counts looked
permutation-sensitive *only* because the perms were silently invalid.

**Corrected finding: the `core_id → work_slice` permutation has no effect
on out-split matmul bandwidth, at any core count. The lever does not
exist.**

## Retracted

The following are withdrawn — all depended on `% 32` permutations applied
to non-power-of-2 (`out:22/25/26/30`) bundles:

* **Measurement 1** (Phase A, identity vs bit_reverse, "−6.4%"
  whole-graph) — the graph's `out:25` bundles skipped work; the run also
  produced wrong output.
* **Measurement 2** ("permutation-sensitive" vs "permutation-invariant"
  groupings) — the "sensitive" bundles were exactly the `out:25` ones;
  they were computing less, not running faster.
* **Measurement 6's core-count table** and the "30–41% headroom" claim.

## What survives

### `mb`-split vs `out`-split is structural

Independent of permutations — read straight from the SDSC. The planner
splits the larger output dim: `mb`-split when M is bigger, `out`-split
when N-in-sticks > M.

* **`mb`-split is broadcast-bound** (~37 GB/s, flat, permutation-invariant
  — and `% 32` perms are valid bijections on `mb:32`, so this is clean).
  Every core needs the full weight; it comes over the on-chip broadcast
  fabric (the HMI's hardware multicast). No per-core slice → no lever.
* **`out`-split is HBM-bandwidth-bound** (~70–145 GB/s). Each core reads
  its own distinct weight column-band from HBM through the shared HMI.

### Restickify is bandwidth-bound (Measurements 3, 4, 5, 7)

These measurements stand: the restickify-bearing bundles were `mb:32` /
`out:32` — power-of-2 — so the `% 32` permutations *were* valid bijections
there, and the bundles were genuinely permutation-invariant.

* **Activation restickify** (qk^T pattern): restickify-bearing bundle time
  scales linearly with H and is invariant to permutation at every size —
  the textbook bandwidth-bound signature.
* **Weight restickify** (Measurement 7): always fused into the consuming
  matmul's bundle, never standalone. Permutation-invariant — the matmul
  alone swings under (invalid) perms but the fused restickify contributes
  ~1.8 ms/call of flat time. And at non-power-of-2 core counts the
  permutation infra aborts the bundler outright (`"Workslice information
  for coreId=23 was not found"`).
* Conclusion: restickify of either kind is bandwidth-bound. Attack it by
  *eliminating* it (load-time weight pre-formatting, issue #1339;
  layout-decision optimizer, PR #1979), not by core-id permutation.

### Forced K-split hurts decode (Measurement 8)

About split *type*, not permutations — unaffected. Forcing a K-split on
the M=1 down_proj shape (Granite-3.3-8B, M=1, K=12800, N=4096) is slower
at every ratio:

| Forced split | per-call | Δ vs `out:32` baseline (0.737 ms) |
|---|---:|---:|
| `out:16, in:2` | 0.794 ms | +7.7% |
| `out:8, in:4` | 0.785 ms | +6.5% |
| `out:4, in:8` | 0.854 ms | +15.9% |

At M=1 the per-slice partial product is a single PT row — the matmul
compute is trivially cheap, so a K-split only adds an SFP-ring PSUM
reduction with nothing to amortise it against. Decode should stay
`out`-split.

### AIU HMI topology (from the Rapid Core ISA)

The ISA does not contain a DRAM bank/channel address function (it defers
HMI internals to separate HMI / Transport-Layer docs). But the topology
it does give reframes the would-be mechanism:

* All 32 cores share **one HMI** — 2×128 B ring ports, **256 B/cycle full
  duplex**. Path: core → L3LU → RIU → QuadRing → HMI → DRAM. No per-core
  memory channel.
* So any out-split bandwidth contention is at the **single shared HMI**
  (its 256 B/cycle ceiling, its flow-control buffering across 32
  competing cores), not "DRAM bank contention" as originally theorised.
  The honest identity number (~107 GB/s) is ~42% of the HMI's
  ~256 GB/s ceiling.
* Unicast loads (`LDM`) — the out-split path — are capped at 4 outstanding
  requests per core. Multicast loads (`LDGM`) — the `mb`-split path — are
  the hardware broadcast: "HMI synchronizes ring multicasting enabling all
  32 cores to simultaneously read the same data."

## The real finding — sticks-per-core bandwidth

The one effect that survives, and the only positive result here. Measured
under *identity* (no permutation — uncontaminated by the artifact above),
on isolated `out:32` matmuls, splits confirmed via `diff_kernels.py`.

Dense sweep, M=128, K=4096, `sticks_per_core = (N/64)/32`:

| sticks/core | eff BW | | sticks/core | eff BW |
|---:|---:|---|---:|---:|
| 5  | 117 GB/s | | 11 | 118 GB/s |
| 6  | 73 GB/s  | | 12 | 71 GB/s  |
| 7  | 67 GB/s  | | 13 | 120 GB/s |
| 8  | 124 GB/s | | 14 | 72 GB/s  |
| 9  | 117 GB/s | | 15 | 123 GB/s |
| 10 | 124 GB/s | | 16 | 118 GB/s |

**Predictor: `oddpart(sticks_per_core) ∈ {3,7}` → slow (~70 GB/s);
otherwise fast (~120 GB/s).** 12/12 fit. The slow set `{6,7,12,14}` is
exactly `{3,7}·2ᵏ` — closed under doubling the band width, so the effect
depends on a *modular phase* of the band geometry, not its absolute byte
size. **K-independent**: fast stays ~115–125 and slow ~70–73 across
K ∈ {2048, 4096, 8192}.

`out:32` only holds for `sticks_per_core` 5–16 (at M=128); below 5 the
planner picks `mb`-split, at 17+ it picks `mb16/out2`.

Caveat — M-dependence is unproven: at M=256, `s=9` (normally fast) dropped
to 61 GB/s, but per-call also doubled with weight bytes unchanged, so
M=256 is a weight-refetch / compute-bound regime, not a clean HMI test.

### Mechanism (hypothesis, not confirmed)

Best guess: a 4-way interleave at the shared HMI — 4 outstanding requests
per core (per the ISA), or 4 HBM bank/pseudo-channel phases. When
`oddpart(s)` is small and `≡ 3 (mod 4)`, consecutive cores' contiguous
bands (start offset `core·s·128 B`) land in a conflicting phase pattern
that serialises at the single shared ring port. Unverified.

### Why this could be actionable

`sticks_per_core = (N/64) / num_cores`, and the planner *chooses*
`num_cores`. For N=14336 it picks `out:32` → s=7 → **slow**; `out:28`
would give s=8 → **fast** — potentially ~1.7× the bandwidth with 4 fewer
cores. So a planner heuristic could deliberately pick a smaller core
count to land on a fast `sticks_per_core`. The lever is in core-count
*selection*, not core-id placement. Viability is under test.

## Lessons for the next person

* A probe that permutes `core_id → work_slice` mappings **must** validate
  the result is a bijection on `[0, C)` *and* check numerical correctness
  against a reference. Buffer reuse across compiles will silently mask
  skipped writes — a wrong kernel can look like a fast kernel.
* The `core_id_to_work_slice` permutation does not move HBM bandwidth. Do
  not rebuild on Measurements 1 / 2 / 6.
* If you want to reduce out-split HBM cost, the lever is *not* core
  placement. The remaining open question is the sticks-per-core sizing
  effect above.

## Reproduction

Probes in `/tmp` at time of writing (should be moved into the repo):

* `verify_measurement6.py` — the verification: re-runs an out-split matmul
  with explicit bijection + correctness checks. **This is the probe that
  exposed the artifact.**
* `dump_cluster_mapping.py` — dumps `coreIdToWkSlice_` under a
  non-bijective permutation; shows the 5 un-written slices directly.
* `inspect_mapping.py` — dumps the raw inputs/outputs of
  `_get_core_to_slice_mapping` for an out-split matmul.
* `hbm_hmi_model_probe.py` — bijection-validated permutation probe
  (permutations parameterised by the actual core count C). Confirms valid
  bijections are flat.
* `diff_kernels.py` — dumps `numWkSlicesPerDim_` / `numCoresUsed_` /
  `coreIdToWkSlice_` for a given N; used to confirm actual splits.
* **FLAWED — do not trust:** `hbm_bandwidth_probe.py`,
  `hbm_outsplit_sweep.sh`, `hbm_corecount_sweep.py`,
  `phase_a_perm_sweep.py`, `phase_a_size_sweep.py` — these used `% 32`
  permutations with no bijection or correctness check and produced the
  retracted Measurements 1 / 2 / 6.
* Still valid: `restickify_kernel_timing.py`, `weight_restickify_probe.py`,
  `down_proj_kfast_probe.py` / `down_proj_kfast_force.py` (Measurements
  3–5, 7, 8 — power-of-2 bundles or non-permutation experiments).
