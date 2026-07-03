# dl-dsc Production-Path LX-to-LX Relayout — Verdict

**Scope:** did the production **dl-dsc scatter** path fire on device, what
SPEEDUP does its LX-to-LX relayout buy, and what is its effective ring
bandwidth ρ — measured head-to-head against the **`stcdp_range`** carrier
already measured at **~36 GB/s** effective (13.1 MB move = 362 µs, slope
method, R² ≥ 0.9999; see `LX_TO_LX_SPEED.md`).

The two paths drive the **same physical `STCDPOpLx`** on-chip move; they
differ only in **how the move is inserted**:

- **`stcdp_range`** — the frontend emits an explicit `OnChipMoveSTCDPOpLx`
  range payload.
- **dl-dsc** (this doc) — the production dl-dsc scatter frontend records
  producer/consumer coordinate metadata (`coreIdToWkSlice_`); the backend
  resident-scatter sizing fix reads the coordinate mismatch and
  **auto-synthesizes** a `-LxRelayout` SuperDsc emitting `STCDPOpLx`, sized
  to the resident piece. No explicit frontend movement row.

Numbers are tagged **measured** (device A/B), **modeled** (backend source /
op-equivalence), or **bound/inferred** (decomposition against the
`stcdp_range` absolute).

Workload: fused SwiGLU `[1, 512, 4096]` (FMS `get_module("swiglu")`), params
re-homed to `torch.empty(device="spyre")` (no H2D wedge). Harvest env,
32 cores, solo device.

---

## 1. Did it fire? — YES (measured, HIGH confidence)

The dl-dsc scatter is device-verified in the emitted SDSC. `REL=1`
(`SPYRE_LX_PLANNER_RELAYOUT=1`) vs `REL=0` baseline differs on exactly **two
of six** sub-bundles of the fused SwiGLU:

- **`sdsc_5`** — the down-proj `batchmatmul`
  `N_={mb:512, out:4096, in:12800}` → `Y[512,4096] = A[512,12800] @ W[12800,4096]`.
  Under `REL=1` the input activation **A flips `component_ hbm→lx`** and its
  producer `coreIdToWkSlice_` is populated **in-split ×32** (`in:32` distinct,
  `mb:1`). The consumer compute distribution is **`mb8 × out4`**. Producer
  (`in`) ≠ consumer (`mb`) **and** A is LX-pinned ⇒ the backend condition
  (`SdscRelayoutInsertion.cpp:135-137`: LX-pinned input **and**
  `allocCoords.coreIdToWkSlice_ != sdsc->coreIdToWkSlice_`) is
  **deterministically TRUE** ⇒ exactly one `STCDPOpLx` **`-LxRelayout`**
  SuperDsc is synthesized.
- **`sdsc_3`** — the `mul`/silu. Its **output flips `hbm→lx`** and its
  coords match the consumer ⇒ **residency only, no move**.
- **Baseline `REL=0`** — both edges are HBM, `coreIdToWkSlice_` empty,
  `lxRelayoutClassifications_` empty. A clean discriminator.

Firing is confirmed **three independent ways**:

1. **Frontend SDSC metadata** — the `hbm→lx` flip + in-split×32 producer
   coords on A (above).
2. **The deterministic backend code path** — given that exact coordinate
   mismatch, `insertRelayoutSdsc` has no branch that skips the move.
3. **Artifact delta** — the `REL=1` `bundle.mlir` **dropped ~40 lines of HBM
   address constants** (the spilled-A offsets, stride 2048): direct evidence
   A left HBM.

The literal debug log `"Lx space found, inserting stcdpLx"` is behind
`#ifdef DEBUG_RELAYOUT` and **compiled out** of the fix build — it is *not*
greppable and was *not* relied on.

**Bytes moved:** A = 512 × 12800 × 2 = **13,107,200 B = 13.1 MB** —
**byte-for-byte identical** to the `stcdp_range` tensor.

---

## 2. SPEEDUP — measured, N = 20

Steady state (warmup absorbs a one-time ~60 s cold-start "lost completion"
wedge). Sub-1 % spread; the two distributions are **disjoint** (REL0 device
min 8.835 ms > REL1 device max 8.745 ms), so the delta is not noise.

| metric | `REL=0` (HBM) | `REL=1` (dl-dsc) | speedup | Δ |
|---|---|---|---|---|
| **Device** | 8.861 ms | 8.714 ms | **1.017×** | **−147 µs** |
| **Wall**   | 9.181 ms | 9.045 ms | **1.015×** | **−136 µs** |

The **−147 µs** is the net benefit of keeping the down-proj LHS on-chip
(the 13.1 MB relayout move, `sdsc_5`) **plus** the mul output on-chip
(residency, `sdsc_3`) instead of round-tripping HBM. That is **1.66 %** of
the 8.86 ms fused SwiGLU.

---

## 3. Effective ρ (LX-to-LX) — ~36 GB/s (modeled by op-equivalence; not independently re-measured here)

**ρ ≈ 36 GB/s.** The dl-dsc move invokes the **identical `STCDPOpLx`
primitive** on the **identical 13.1 MB tensor**, over the **same RIU BiRing**,
as `stcdp_range`. `stcdp_range`'s standalone-execute slope method isolated
that op at **362 µs / 13.1 MB = 36.2 GB/s** (R² ≥ 0.9999). Same op + same
ring + same bytes ⇒ **same ρ to first order** — this is the honest prior, and
it is **modeled by equivalence, not an independent dl-dsc measurement**.

**Why the slope method cannot isolate the dl-dsc move here (measurement
limit, not a mechanism difference).** `stcdp_range` exposes the move as an
**explicit, duplicable `sdsc_execute` line** in `bundle.mlir`, so K extra
copies can be spliced in and `Δtime/Δcount` cancels all fixed overhead. The
dl-dsc relayout is **backend-synthesized *inside* `sdsc_5`'s execute** during
`dxp_standalone`; `bundle.mlir` holds only the six whole-SDSC `sdsc_execute`
lines, so there is **no standalone line to duplicate**. This is the direct
structural consequence of the insertion locus (backend auto-synth vs frontend
range payload) — not evidence of a different move.

**Best independent bound (inferred, uses the `stcdp_range` absolute).** The
measured **+147 µs net A/B** is consistent with a **362 µs on-chip move**
(36 GB/s on 13.1 MB) **displacing a ~509 µs HBM relayout** of the same
13.1 MB — i.e. **~1.41× at the edge**. Caveat: the +147 µs also folds in the
`sdsc_3` **mul-output residency** (no move), so attributing the full delta to
A's move edge is a **loose upper attribution**; the 509 µs / 1.41× figure is
**inferred, not independently re-measured**.

**Second-order caveat (chunking).** `BurstEfficiency` is chunk-size- and
multicast-degree-sensitive (`LX_TO_LX_SPEED.md §1`). The dl-dsc move has a
**different chunking** than `stcdp_range` (§4): its per-core resident piece is
`getRelayoutPieceSize` = 13.1 MB / num_pieces(`mb8×in1`=8) = **1.638 MB/core**,
and it adds an **out-split ×4 broadcast** the pure permutation lacked. A
larger multicast degree tends to **lower** effective ρ on logical bytes. So
**36 GB/s is the prior**; a real measured gap (were the move separable) would
indicate a piece-size/chunking difference, **not a different mechanism**.

---

## 4. Side-by-side: dl-dsc vs `stcdp_range`

| aspect | `stcdp_range` (measured) | dl-dsc (this doc) |
|---|---|---|
| Physical op | `STCDPOpLx` | **same** `STCDPOpLx` |
| SuperDsc name | frontend `OnChipMoveSTCDPOpLx` | backend `…-LxRelayout` (`-inp`/`-out`/`-probe`) |
| Insertion | explicit **frontend** range payload | **backend** auto-synth from producer/consumer `coreIdToWkSlice_` mismatch (`in`-split → `mb`-split) |
| Unique bytes | 13.1 MB | **same 13.1 MB** |
| Chunk pattern | pure **5-chunk strided permutation** | coordinate-derived scatter (32 `in`-pieces → 8 `mb`-pieces) **plus out-split ×4 broadcast** |
| Sizing | range-payload chunking | **resident-piece fix**: `getRelayoutPieceSize` → **1.638 MB/core** (`ceil(form_size / num_pieces)`) |
| Ring / cores | RIU BiRing, 32 cores | **same** RIU BiRing, 32 cores |
| ρ effective | **36 GB/s (measured, R²≥0.9999)** | **~36 GB/s (modeled by equivalence)** — chunking caveat |
| Move isolable by slope? | **yes** (explicit `sdsc_execute` line) | **no** (fused inside `sdsc_5` execute) |
| Speedup vehicle | standalone rho probe | **fused SwiGLU A/B: −147 µs device / 1.017×** |

**Same physical move?** Yes — same `STCDPOpLx`, same ring, same 13.1 MB
unique tensor, same producer→consumer strided permutation. **Same bytes?**
Same 13.1 MB *unique*; dl-dsc additionally **replicates across the out-split
(×4)**, so its physical traffic is **≥** the pure permutation's. **Same ρ,
different speedup?** ρ is the same primitive (~36 GB/s prior); the *speedups
are measured on different vehicles* — `stcdp_range` gave a standalone rho, the
dl-dsc number is the **end-to-end fused-SwiGLU A/B (−147 µs)**.

**Where the difference comes from — the mechanism.** The **backend
resident-scatter sizing fix** derives the move from the coordinate mismatch
and sizes the LX residency probe to the **resident piece (1.638 MB/core)**,
not the whole range; the **production dl-dsc scatter frontend** supplies only
the coordinate metadata, not an explicit movement row. The backend
auto-synthesis is a **compile-time** step; the emitted runtime op is the same
`STCDPOpLx`.

---

## 5. What this means

**(a) The production dl-dsc LX-to-LX transfer is the same ~36 GB/s effective
(strided) primitive.** Same op, same ring, same 13.1 MB — the transfer rate
is `stcdp_range`'s measured 36 GB/s to first order (**modeled by
equivalence**), with a second-order chunking caveat (§3): the resident-piece
sizing (1.638 MB/core) + out-split ×4 broadcast differ from the range
payload, and `BurstEfficiency` is chunk/multicast-sensitive, so dl-dsc's ρ
could sit **at or slightly below** 36 GB/s on logical bytes. It is **not
faster per byte** — it is the same ring primitive.

**(b) The win is removing the HBM round-trip, not a faster wire.** The
**measured −147 µs (1.017×)** comes from keeping the down-proj LHS (13.1 MB
on-chip relayout) and the mul output (residency) on-chip instead of spilling
to and reloading from HBM. Consistent with the `stcdp_range` finding: the
on-chip ring move (~362 µs) is the *minority* of the cost it displaces — here
it undercuts a **~509 µs HBM relayout** (**inferred/bound**), a **~1.41×** win
at that edge, which nets to **1.66 %** of the whole fused SwiGLU.

**(c) The insertion mechanism adds no runtime overhead; it moves work to
compile time.** dl-dsc's backend auto-synthesis (reading the coordinate
mismatch, sizing via `getRelayoutPieceSize`) is a **compile-time** cost. At
runtime it emits the **same `STCDPOpLx`** as the explicit `stcdp_range`
payload, so there is **no added per-move runtime overhead** from the insertion
locus — **modeled as neutral** (the two paths were not run head-to-head on the
identical move to measure a runtime delta, and no such delta is expected from
op-identity). If anything, sizing the residency probe to the **resident
piece** rather than the full range **avoids over-provisioning** LX; the net
runtime effect of that sizing on the move rate is **unmeasured** and modeled
neutral.

**Bottom line.** The production dl-dsc path **fires** (device-verified, 3
ways), delivers a **measured −147 µs / 1.017× device** (−136 µs / 1.015×
wall) on the fused SwiGLU, and carries the **same ~36 GB/s effective
`STCDPOpLx`** LX-to-LX primitive as `stcdp_range`. The speedup is the
**HBM-round-trip removal**, not a faster ring; the frontend-metadata +
backend-coordinate-synthesis insertion is a **compile-time** mechanism that
adds **no runtime overhead** over the explicit range carrier.

---

## Provenance

- **Measured (device, harvest env, solo, 32 cores):** N=20 A/B,
  `SPYRE_LX_PLANNER_RELAYOUT` 1 vs 0; device 8.861→8.714 ms, wall
  9.181→9.045 ms; distributions disjoint. Firing verified in the emitted
  SDSC (`sdsc_5` A `hbm→lx` + in-split×32 producer coords; `sdsc_3` output
  `hbm→lx`) and the `bundle.mlir` ~40-line HBM-constant drop. Driver:
  `…/scratchpad/dldsc_driver.py`.
- **Modeled (backend source, read-only):**
  `SdscRelayoutInsertion.cpp:119` (`insertRelayoutSdsc`), `:135-137`
  (LX-pinned + `coreIdToWkSlice_` mismatch), `:161` (`…-Relayout` SuperDsc),
  `:219-220` (`OpFuncs::STCDPOpLx`), `:29-45` (`getRelayoutPieceSize`,
  per-core piece `ceil(form_size / num_pieces)`); frontend
  `lx_relayout.py:15-22` (records metadata, no movement rows), `:137`
  (`_classify_coordinate_topology`), `:220` (`_core_id_to_device_slice`),
  `:306` (`_record_plan`); config gate `config.lx_planner_relayout`
  (`_inductor/config.py:45`). Fix build:
  `build/deeptools/dxp/libdxp.so` (`getRelayoutPieceSize`×1,
  `insertRelayoutSdsc`×6, `-LxRelayout`×4).
- **Bound / inferred (uses the `stcdp_range` absolute):** ρ ≈ 36 GB/s by
  op-equivalence (`stcdp_range` 362 µs / 13.1 MB, R²≥0.9999, `LX_TO_LX_SPEED.md`);
  ~509 µs displaced HBM relayout / ~1.41× edge from the +147 µs net A/B — a
  loose upper attribution (also folds in the `sdsc_3` residency edge), **not**
  an independent dl-dsc slope measurement (the move is fused inside `sdsc_5`
  and has no duplicable standalone execute line).
- **Not fabricated:** the dl-dsc move was not slope-isolated on device; its ρ
  is stated as a modeled prior with an explicit chunking caveat, and the
  edge-decomposition as an inferred bound.

Written to `/home/adnan/dt-inductor/torch-spyre/carousel-rfc-impl/DLDSC_LX_SPEED.md`.