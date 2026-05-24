# Tiled On-Chip Producer→Consumer Pipeline Design (the >4k regime)

The unified design for a **tiled, double-buffered, on-chip producer→consumer
pipeline** that keeps a cross-core activation handoff resident in LX even when the
producer's output slice plus the consumer's input slice exceed the 2 MB/core LX
scratchpad (the >4k regime). The proven single-move recipe is in
`CoreToCoreDataMovementRecipe.md`; the move-tiling correction is in
`StreamingImplementationPlan.md`. This document folds the nine research digests
(architecture, schedule-IR, deeptools-source, inductor-source, docs, and the GPU
precedent) into one design and one honest strategic recommendation.

Every claim is grounded in a digest or source file; inferences are flagged
**[INFER]**. Where an older deeptools stage-note conflicted with current source, the
current source wins (called out in §10).

---

## 1. Problem recap — the endpoint-residency gap

The proven cross-core ring `STCDPOpLx` keeps a producer→consumer activation handoff in
LX with zero HBM traffic (`CoreToCoreDataMovementRecipe.md` §8). But **move-tiling
alone does not close the >4k regime.** `build_streamed_bridge` tiles only the
*transfer* buffer; the producer DL op still writes its **full** per-core output slice
and the consumer DL op still reads its **full** per-core input slice
(`StreamingImplementationPlan.md` CORRECTION). At hidden S=8192 each per-core slice is
~4 MB; producer-output + consumer-input is ~8 MB ≫ the 2 MB/core LX. `apply_lx_flip`
stamps both endpoints LX-resident at buffers spaced only 128 KB apart, so the
producer's full slice overwrites the consumer buffer → corruption
(`onchip_realize.py:realize_streamed_handoff`; `PerformanceResults.md` "Streaming
(>4k) status").

Usable LX is **~1.6 MB, not 2 MB** — the backend reserves 20 %
(`DXP_LX_FRAC_AVAIL=0.2`, so `0.8 × 2 MB ≈ 1.6 MB`;
`docs/source/compiler/scratchpad_planning.md` §1; `config.py`). The single 2-region
move covers ≤4k (1 MB/core × 2 = 2 MB, zero DL headroom); past 4k it cannot fit.

**Closing >4k requires the producer and consumer to be tiled too** — produce tile *k*,
hand it off, consume tile *k*, so only a tile is ever LX-resident.

## 2. The design — a fused tiled producer→move→consumer pipeline

Tile the producer, perform a per-tile cross-core `STCDPOpLx` move, then run the
consumer, all in one fused loop:

```text
for k in 0 .. K-1:
    P_k : producer computes output rows [k*T, (k+1)*T)  -> prod_buf   (DL op, tiled)
    M_k : STCDPOpLx moves prod_buf -> cons_buf cross-core             (data-op)
    C_k : consumer computes on rows [k*T, (k+1)*T) from cons_buf      (DL op, tiled)
```

With double-buffering (two tile-buffer pairs indexed `k & 1`), producer(k+1),
ring-move(k), and consumer(k-1) overlap — a depth-2, 3-stage software pipeline. Only a
tile plus its double-buffer twin is live per buffer, so live LX collapses to a flat
**`4 T + W`** (four tile buffers — producer ×2, consumer ×2, cross-core forces distinct
src/dst — plus the DL ops' own working LX `W`), regardless of S. At
`STREAM_TILE_BYTES = 128 KB`, `4 T = 512 KB`, leaving ~1.1 MB for `W`. The handoff
becomes **size-unbounded**: only K (= ⌈slice/T⌉, hence ring transactions and pipeline
iterations) grows with S; the live LX footprint does not
(`design_tiled_pipeline.md` §2.2).

Tile the **non-stick (mb/row) dim** so every stick stays whole — never the out-chunk,
which risks a sub-stick (`StreamingImplementationPlan.md` §6;
`docs/source/user_guide/tensors_and_layouts.md` §4.1).

**Precedent.** This is the FlashAttention / CUTLASS warp-specialized template
made concrete: tile the intermediate in on-chip memory, never materialize it to HBM,
overlap producer (DMA) and consumer (compute) stages. On Spyre the
"block-and-tile + double-buffer L1" model is the **intended** programming model —
RaPiD states the compiler "blocks and tiles the program loops … balancing the
scratch-pad capacity and available bandwidth" and hides DMA latency "by
double-buffering data in the L1 scratchpad overlapped in time with computation"
(`kb_scour_tiling.md` §2, `sources/rapid.pdf` p.7). The chunked-prefill-attention
schedule already streams K/V through LX with `numBuffers_=-1`
(`kbib_arch.md` §5, `schedule-ir-spec.md`). See the GPU mapping table in §11.

## 3. The building blocks already exist in the Schedule IR

The authoritative current-source view (`kbib_schedule.md`, `kbib_arch.md`) is that the
production `dsc2::ScheduleTree` already models *every staging primitive* a tiled,
double-buffered, LX-resident pipeline needs — **for one op at a time**:

| Primitive | What it gives us | Source |
|---|---|---|
| `LoopNode` (nested, static/symbolic) | tile loop nests over an op's own dims | `dsc2-schedule-tree.md`; `kbib_schedule.md` §1a |
| `AllocateNode.numBuffers_` = 1/2/-1 | single / ping-pong / streaming overlap | `dsc2-schedule-tree.md` L118-120; `kbib_arch.md` §2.1 |
| `TransferNode.unitTimeTransferChunkSize_` | windowed per-tile transfers | `dsc2-schedule-tree.md` L76-80; `kbib_schedule.md` §1b |
| `TransferNode direction="local"` | LX↔LX (cross-core) transfers | `schedule-ir-spec.md`; `kb_scour_tiling.md` §3 |
| `BlockNode` | interleave transfer / compute / sync in a loop body | `dsc2-schedule-tree.md`; `kbib_schedule.md` §1d |
| `coreIdToGTRInfo_` / `ring_multicast` | cross-core multicast | `kbib_schedule.md` §1f |
| `implicit_sync_on_streaming_buffer` | auto-generated boundary syncs | `kbib_arch.md` §2.2 |

The matmul K-loop (`B_lx`/`A_lx` `num_buffers=2`, single-buffer `C_lx` accumulator) and
the chunked-prefill-attention loop (`K_lx`/`V_lx` `num_buffers=-1`) **are**
double-buffered tiled pipelines that keep a tile LX-resident
(`schedule-ir-spec.md` §"Default Matmul", §"Chunked Prefill Attention";
`kbib_schedule.md` §1c). The crucial limit: **each of those examples is the staging
plan for ONE compute op.** Don't hand-roll ping-pong or boundary syncs — drive
`numBuffers_` and let the compiler generate the syncs (`kbib_arch.md` §6.2). The gap
is not loops or buffering; it is combining two ops in one region (§4).

## 4. The precise gap (authoritative current deeptools source)

`design_tiled_deeptools.md` reads the live deeptools source and finds the blocker is
**structural, not a missing buffer trick**: one schedule tree = one compute op. Three
hard blocks, each cited to source:

- **(a) One DL op per core.** `coreIdToDsc_` binds each `coreId` to a **scalar** dsc
  index, and the merge path hard-asserts `DT_CHECK(seenDLDsc == false)`
  (`dcg_manager.cpp:821`). A second DL step per core trips the assert.
- **(b) DL ops materialize the FULL per-core output — no windowed DL invocation.**
  Data-ops carry `dimToStartCordinate`/`dimToSize_` per dim (this is what move-tiling
  uses), but a DL op's extent is its blocking hierarchy `B_`/`T_`/`P_`
  (`designSpaceConfig.cpp:getPcfgLoopCount` 1450-1496) — there is no "produce only rows
  [k*T,(k+1)*T)" notion. You cannot ask the producer DL op to emit tile k only, or the
  consumer DL op to consume tile k only.
- **(c) Multi-DSC is data-parallel ONLY.** Multiple DSCs in a SuperDSC are admitted
  *only when each is its part of the SAME work* (same op sharded across cores). A
  producer and a consumer are explicitly the unsupported "work A / work B" case:
  `DT_CHECK_MSG(isSameDscGroup(mySDsc), "Expect DSCs in the same group")`
  (`L3DlOpsScheduler.cpp:7052-7059`); `createChunkLoops` builds the loop order from
  `dscs_.at(0)` and assumes all DSCs share it.

The existing streaming engine — `createChunkLoops` / MVLOOP — is strictly **intra-op**
and **stages through HBM** (`isHbmPinned()`; "transfer node inside the outermost loop",
`dlOps.cpp:405-410`). It is the very round-trip the handoff removes, and it cannot feed
one op's tile from another op's tile. **K-unrolling does NOT shortcut this**: K DL steps
trip (a); a producer+consumer pair trips (c) even at K=1; and the DL op still writes its
full slice (b), so unrolling the *moves* without windowing the *compute* leaves endpoint
residency unchanged (`design_tiled_deeptools.md` "Is K-UNROLLING a viable shortcut").
This matches the codex finding that the tiled pipeline compiles via K-unroll but is
**not value-correct on hardware** (`kb_scour_scheduling.md`), down-weighted but
consistent.

## 5. The ask — "Ask 3A: fused tiled handoff block"

Closing >4k requires a **third, larger deeptools ask** beyond the two already
identified (the Foundation gate that wires the mixed-bundle import, and the
layout-changing transpose). It is a genuinely new *scheduling* capability, built FROM
the existing IR primitives (§3) but combining them across ops
(`design_tiled_deeptools.md` "Ask 3A"):

1. **Pipeline-group mode** — relax the same-group gate (`L3DlOpsScheduler.cpp:7059`) so
   `dscs_` may hold distinct ops with a declared producer→consumer dependency, and the
   scheduler builds a shared outer loop whose body is [producer-tile, move,
   consumer-tile] instead of assuming DSC 0's loop order.
2. **More than one DL op per core** — relax `DT_CHECK(seenDLDsc == false)`
   (`dcg_manager.cpp:821`) and let `coreIdToDsc_` (or a new `coreIdToDscList_`) bind
   multiple DL indices per core; emit a DL pcfg per scheduled `dldsc_idx`.
3. **Windowed DL-op invocation** — give the DL op a tile window analogous to the
   data-op `dimToStartCordinate`/`dimToSize_`, so producer/consumer compute runs over
   rows [k*T, (k+1)*T) per iteration. **This is the genuinely new primitive** — there is
   no DL-op windowing today.
4. **An outer tile loop owning multi-op bodies** — either extend `coreIdToDscSchedule`
   with a trip-count field, or build a `dsc2::LoopNode` whose body legally owns COMPUTE
   nodes from two ops. Today every node has a single `ownerDsc` (`dsc2.h:470`); whether
   that is load-bearing across codegen is the **open question** — "extend LoopNode" vs
   "new fused-op IR" is the difference between a medium and a large RFC, and needs a
   deeptools-owner read of `moveNode`/`fillLoopOffsetsAndAddresses`/DVS lowering.

Because the user is on the inductor team, this is an **RFC handoff**, not inductor work.

**Acceptance test.** An S=8192 same-stick handoff (producer add → cross-core STCDP →
consumer bmm/elementwise) compiles HBM-free and runs value-correct through (gated)
deeptools, with only ≤ a tile (≤ LX live) at any time, and the senprog shows an outer
loop wrapping [producer-tile, `L3_LDU`/`L3_STU` move, consumer-tile].

## 6. Same-stick only — the transpose frontier

The pipeline targets **same-stick** handoffs (the proven, non-faulting path).
`STCDPOpLx` requires identical `stickDimOrder_` on src and dst; a layout-changing tile
move would need `ReStickifyOpWithPTLx`, which **faults Compute-CB** on device
(`CoreToCoreDataMovementRecipe.md` §10). The layout-change is a separate frontier.

The DNNDaSher model (authoritative, `kbib_arch.md` §1) reframes that frontier
favorably. Our same-stick STCDP is an **Inter-DataStick shuffle** — whole DataSticks
moved cross-core over the ring, the cheap and intended primitive. A layout-change is an
**Intra-DataStick shuffle**, realized by repurposing the idle 8×8 MPE arrays as a
**2D-Compute-Array transpose** (an `L×L` transpose in `2L` memory transactions — a
published, hardware-sanctioned technique). **[INFER]** Because that transpose-on-PE
path is a known-good org technique, the `ReStickifyOpWithPTLx` Compute-CB fault is
**likely an integration / descriptor bug, not a hardware wall** — so the transpose
frontier is plausibly addressable. The right next step there is to re-examine the
integration: it consumes `stickDimOrder_` / `layoutDimOrder_` / `coreIdToWkSlice_`,
which torch-spyre already emits (`kbib_arch.md` §1.5, §6.2).

## 7. Inductor side

No temporal tiling is emitted today. Work-division is **spatial-only**: it assigns
`op.op_it_space_splits` (a `{dim: n_cores}` map, product ≤ 32), with no iteration count
or tile loop (`work_division.py:apply_splits`; `design_tiled_inductor.md`;
`docs/source/compiler/work_division_planning.md`). Every coord-fold is
`sdscFoldProps_=[{factor_:1,"time"}]`, `"temporal":0`; a core computes its whole
~4 MB slice in one SDSC pass. The scratchpad allocator sizes whole tensors
(`size_per_core = dev_size // num_cores`, no partial residency; `scratchpad.py`). The
work-division 256 MB span guard is the *addressable* span, not LX — nothing sizes
splits to fit the 2 MB LX (`kb_scour_docs.md` §0b).

So torch-spyre would emit the **multi-op tiled plan** (the fused handoff block) for
`L3DlOpsScheduler`; the **realize pass** (`onchip_realize`, the seam documented in
`CoreToCoreDataMovementRecipe.md` §6b) is where it plugs in. But **pure-inductor cannot
do it**: emitting the plan is necessary, but executing a tiled, windowed,
multi-DL-op-per-core fused block needs Ask 3A in deeptools (§4-5). Note the docs flag a
documented-but-unimplemented alternative for cross-bundle LX persistence —
"non-terminal kernel" hints (`scratchpad_planning.md` §7.4) — but the conservative,
device-validated position is that LX does not persist across an `sdsc_execute`
boundary, so the handoff must stay inside one SuperDSC (`kb_scour_docs.md` §1.3).

## 8. Strategic recommendation (the honest headline)

**>4k is where the on-chip handoff's relative win is SMALLEST.** The matmul cost grows
O(N³) while the handoff grows O(N²), so the handoff becomes a smaller *fraction* of
total time even as its absolute magnitude grows. Measured speedup *tapers*: 1.22× @1024
→ 1.19× @2048 → 1.13× @4096 (`PerformanceResults.md`;
`CoreToCoreDataMovementRecipe.md` §9). Ask 3A is a **large** deeptools effort aimed at
the **lowest-relative-payoff** regime.

Recommendation:

1. **Ship the ≤4k single 2-region move** (proven, value-correct on device). It covers
   the 1024–4096 sweet spot and the real attention QK^T→softmax edge (**1.29× at
   seq=512**, `PerformanceResults.md`).
2. **Gate >4k handoffs to HBM** (graceful fallback) for now — the realize pass already
   fail-closes when the footprint exceeds LX (`onchip_realize.py`).
3. **Document Ask 3A as a future RFC** (§4-5). Pursue it **only if** a concrete >4k
   bandwidth-bound case (very-long-context decode, large MoE activations) shows the
   **absolute** saving justifies the deeptools cost. Decide by the absolute byte saving
   in a real bandwidth-bound workload, not by the relative ratio.
4. **Keep `build_streamed_bridge` as a partial building block** (the move-tiling piece);
   do not device-pursue it until Ask 3A lands — a naive >4k test fails on endpoint
   overlap, not buffer reuse (`StreamingImplementationPlan.md` CORRECTION).

Also note: **the transpose-as-integration-bug (§6) may be a higher-ROI frontier than
>4k tiling** — it unblocks the layout-changing bucket (the majority of high-value
pre-matmul edges; `CoreToCoreDataMovementRecipe.md` §12.2) across *all* sizes, whereas
Ask 3A only extends the lowest-relative-payoff regime.

## 9. Open questions and inferences

Open (need a deeptools-owner read or a device measurement):

- Whether a `dsc2::LoopNode` can legally own COMPUTE nodes from two `ownerDsc`s with a
  modest change, or whether single-`ownerDsc` is load-bearing across codegen — gates
  "extend LoopNode" vs "new fused-op IR" (§5 item 4).
- Whether the tiled DL op's working set `W` stays O(tile). The size-unbounded claim
  (§2) assumes O(tile) scratch; if `W` is O(slice) the `4 T + W` budget re-introduces an
  S-dependent term (still better than full residency, but bounded).
- Whether the AOT schedule actually **overlaps** a DL step with an l3lu/l3su step on the
  same core (there is no runtime warp scheduler). If every schedule step implies a full
  barrier, the double-buffer buys only LX, not latency hiding.
- The exact `after_sync`/`before_sync` placement for the 2-slot ping-pong that prevents
  WAR/RAW hazards while still permitting P/M/C co-issue (the hardest schedule detail).
- Whether the high-value >4k same-stick edges (very-long-context decode, large MoE
  activations) actually exceed the 2 MB/core endpoint budget at production sizes — the
  cost/benefit gate for committing Ask 3A.

Inferences (flagged): the transpose Compute-CB fault being an integration bug (§6,
`kbib_arch.md` §6.2); `W` staying O(tile) (§2.2); AOT step overlap depending on
deeptools issuing to distinct units without a forced barrier (§2). Move-tiling alone
being fully expressible today (no new mechanism) is structurally confirmed.

## 10. Where a codex finding conflicted with current source

The codex stage-note digests (`kb_scour_tiling.md`, `kb_scour_scheduling.md`) are
*older deeptools* and were used for ideas/history only, never as current capability.
One material conflict, resolved in favor of current source:

- Codex (`kb_scour_scheduling.md`) reports the tiled producer→move→consumer pipeline as
  **expressible today by K-unrolling** (compile-clean, just not value-correct). The
  **current deeptools source** (`design_tiled_deeptools.md`) shows it is **not
  expressible at all** — K-unrolling trips three hard asserts (`seenDLDsc`,
  `isSameDscGroup`, and the absence of windowed DL invocation). **Taken: current
  source.** K-unrolling is not a shortcut; Ask 3A is required.

Codex's capacity reasoning (tiling defeats the 2 MB wall; 64×64 tiles; same-layout
moves are `production_valid=true`) is *consistent* with current source and was used as
corroboration, not as authority.

## 11. GPU-precedent mapping

| Pipeline element | CUDA / CUTLASS / FlashAttention analog | Tightness |
|---|---|---|
| Whole tiled P→M→C loop kept in LX | FlashAttention: tile the intermediate in SRAM, never materialize to HBM, fuse | tight |
| Producer-tile / move / consume-tile per k | CUTLASS warp-specialized mainloop (TMA producer warps fill SMEM, MMA consumer warps drain) | tight — but Spyre spells the schedule AOT, no runtime warp scheduler |
| Double-buffer ping-pong (4 T) | multistage SMEM pipeline (2-stage `cp.async`/TMA double-buffer) | tight, modulo AOT overlap |
| Per-tile `STCDPOpLx`, `memId`=owner | per-stage `cp.async.bulk` into a peer SM's Distributed Shared Memory; `%cluster_ctarank` ≙ `memId` | tight |
| `L3_MVLOOPCNT` over `L3_LDU`/`L3_STU` | looped TMA / `cp.async.bulk` in the mainloop | tight |
| `coreIdToDscSchedule` after/before-sync | `mbarrier` / `barrier.cluster.arrive/wait` | tight, but AOT-frozen vs runtime |
| Tile OUTPUT rows, never the reduction | FlashAttention outer loop over query-row tiles | tight |
| Unbounded activation at flat 4 T LX | persistent/megakernel keeping only the live stage in SRAM | tight in spirit |

Where it breaks (`NvidiaGpuEquivalent.md` §4): ring topology vs near-uniform DSM
crossbar (byte-hops is a first-class cost on Spyre); purely **AOT** schedule (we
synthesize `coreIdToDscSchedule` by hand); explicit per-core data-ops vs implicit warp
roles; 2 MB LX/core but only 32 cores and **no L2/cache to spill to** — which makes the
LX-residency win even more load-bearing than on a GPU.

## 12. Bottom line

The tiled producer→consumer pipeline is the correct *shape* for the >4k gap, and every
staging primitive it needs already exists in the Schedule IR — but only for one op at a
time. The blocker is structural: one schedule tree = one compute op, with no windowed
DL invocation and a data-parallel-only multi-DSC gate. Closing >4k needs **Ask 3A**, a
large new deeptools scheduling capability for the **lowest-relative-payoff** regime.
The recommendation is therefore to **ship the proven ≤4k move, gate >4k to HBM, and
hold Ask 3A as a future RFC** pursued only against a concrete absolute-saving case — and
to weigh the transpose-as-integration-bug frontier (§6), which unblocks the
layout-changing majority across all sizes, as the higher-ROI direction.
