# On-Chip Restickify — Remaining Threads: Scoping & Handoffs

Scoping document for the open threads of the on-chip core-to-core LX
data-movement work. It turns each remaining frontier into a crisp, actionable
ask with a **Problem / Ask / Acceptance criteria / Owner / Effort** block, and
sorts the asks into three buckets: **inductor work we do ourselves**,
**deeptools RFC handoffs**, and **out-of-scope production-stack handoffs**.

This is the formal scoping that follows `SessionFindings-2026-05.md` (the
frontier map). It goes deeper than that summary: each thread is grounded in the
sibling design docs and in the inductor code already landed for it. Read the
proven state first (§1) — it is what motivates the asks. Inferences are flagged
**[INFER]**; device measurements are MEASURED unless labeled otherwise.

## Authors

- Adnan Hoque

---

## 1. The proven state (what motivates the asks)

The on-chip handoff is not a proposal — its core mechanism is landed on silicon
and reachable through the real compiler. The asks below extend a working
capability; they are not a request to prove the idea.

**Per-edge same-stick handoff is device-proven and value-correct.** A
producer→consumer activation slice moves core-to-core over the RIU ring
(`STCDPOpLx`), staying LX-resident with **zero HBM round trip**, value-correct on
device, with the mandatory remove-the-senprog negative control passing in every
case (`CoreToCoreDataMovementRecipe.md` §8/§11; `PerformanceResults.md`).

| Edge / workload | shape | on-chip vs HBM | source |
|---|---|---|---|
| add-mm `(a+b.t()+c.t())@d` | 512–4096 | **1.13–1.28x** (peaks mid-range) | `PerformanceResults.md` |
| Attention prefill QK^T→softmax | seq=512, bh=32, hd=128 | **1.29x**, value-correct | `PerformanceResults.md` |
| Attention prefill, 3-way | `[1,32,512,128]` | **1.28x** (spyre+kernel) | this work |

**It is compiler-driven, not a splice.** `torch.compile` itself emits the mixed
SuperDSC (`SPYRE_ONCHIP_HANDOFF_REALIZE=1`) and the (gated) `dxp` accepts it:
the add-mm 2048 end-to-end run is value-correct at the baseline error with a
clean negative control, no manual splice or artifact redirect
(`CoreToCoreDataMovementRecipe.md` §6b; `PerformanceResults.md`
"Compiler-driven E2E").

**The first REAL-MODEL edge has landed.** The Granite block's
`batchmatmul → mul` same-stick reshard runs on-chip and **value-correct on
device** for the symmetric **25→25** geometry the current granite graph actually
plans (producer and consumer both shard `out:25` across 25 cores). This is the
first on-chip handoff proven on a real model component rather than a micro-graph
or a synthetic round trip. **[INFER]** the brief records this landing as
post-dating `SessionFindings-2026-05.md` and the project memory, both of which
still describe the granite edge as "not landed"; the symmetric 25→25 result
supersedes that status.

**The asymmetric N→M mechanism is also device-proven** on a synthetic
**32→8→32** round trip (value-correct, genuine cross-core ring, negative control
clean; `tier0-tier1-onchip` @14a3cbd). Owner assignment is load-bearing — a
single-stage 8→32 with guessed owners was 93.8% wrong; the round trip is
owner-consistent.

**Bottom line:** the same-stick on-chip handoff is proven per-edge, proven
compiler-driven, and has its first real-model edge landed on device. The five
threads below are what it takes to generalize and scale from there.

---

## 2. Bucket summary

| # | Thread | Owner | Effort |
|---|---|---|---|
| 1 | Layout-changing transpose | **deeptools RFC** | large (vendor) |
| 2 | >4k tiled on-chip pipeline | **deeptools RFC** | large (vendor) |
| 3 | Asymmetric reshard owner-overlap + DCG-PieceInfo contract | **mostly inductor** (+ 1 small deeptools/contract item) | ~250 LOC inductor + small alignment |
| 4 | sendnn 3-way comparison | **out of scope** (production-stack + deeptools) | n/a (handoff) |
| 5 | Dynamic addressing (MoE routing) | inductor + runtime (future) | future |

The split is sharp: **inductor owns generalizing the proven same-stick path**
(Thread 3, plus productionizing the realize pass). **deeptools owns the two
genuinely-missing vendor primitives** (Thread 1 transpose, Thread 2 fused tiled
block) and **one small contract item** (Thread 3's PieceInfo exposure).
**sendnn (Thread 4) is below the torch-spyre boundary entirely.** Thread 5 is a
named future direction.

---

## 3. Thread 1 — Layout-changing transpose (deeptools RFC)

**Problem.** A layout-changing cross-core move — where the stick orientation
itself changes (e.g. `out_↔mb_`) — cannot be done on-chip today. The only
deeptools op that transposes sticks, `ReStickifyOpWithPTLx` (PTLx), **faults
Compute-CB (`0x7b1b`) on device** for our geometry. This session reversed the
verdict from "integration bug" to "deeptools/hardware wall"
(`TransposeFaultDeepDive.md`):

- Fix #1 (split into a genuine local per-core transpose + a separate cross-core
  STCDP reshard) landed perfectly **in the descriptor**, but the transpose
  compute senprog is **byte-for-byte identical** to the faulting one (same
  LXLU 2048 / L0SU 32768 / PT 64 loop bounds). The loop bound is a *product*
  over the output piece dims; resharding which dim is the 2048-band vs the
  64-band only redistributes the same product, so the reshard is invisible to
  the per-core compute program. The fix could not have removed a fault that
  lives in that program (`TransposeFaultDeepDive.md` §1).
- PTLx's **only sanctioned geometry** in all of deeptools is a 4-dim
  `{j_,i_,out_,mb_}` matmul-internal `out_↔j_` transpose with a tiny 128-deep
  per-core piece. Our op is a **flat 2-dim `{mb_,out_}` `out_↔mb_` transpose
  with a 2048-deep per-core piece** — off the trodden path
  (`TransposeFaultDeepDive.md` §2.1).
- It has **no execution test anywhere** — no SDSC, DDL, senulator, or device
  test ever *executes* a PTLx; the single stock usage is a codegen-only test
  (`TransposeFaultDeepDive.md` §3).
- deeptools itself routes around it: PTLx is **unconditionally disabled on
  Sen1.5+** (`dsm.cpp:12875`).

**The ask.** Either:

- **(1a)** make `ReStickifyOpWithPTLx` execute correctly for a flat 2-dim
  `{mb_,out_}` `out_↔mb_` transpose of **arbitrary per-core depth** on
  RCUDD1A/DD2, **and add a senulator/device execution test** (none exists
  today); or
- **(1b)** declare that geometry unsupported and provide a **sanctioned
  layout-changing cross-core primitive** — the "remote-fragment-aware
  coordinate remap" / consumer-endpoint adapter (the gather→transpose→scatter
  chain) — so inductor can emit Q/K/V and pre-matmul transposes without an HBM
  round trip.

**Minimal-repro plan to attach to the RFC.** Build a splice whose per-core
transpose piece matches the only geometry the op is tested at — a single tile
per core, per-core `out_:64 × mb_:64` (depth **64**, not 2048) — by using a tiny
M or by adding an outer loop dim so each PT invocation transposes one 64×64
tile. Then A/B it against the 2048-deep piece, both with the mandatory
remove-the-senprog negative control, run solo (single shared accelerator):

- If the **64-deep runs clean** but the **2048-deep faults** → the wall is the
  deep-loop PT-transpose codegen: a concrete, reportable deeptools bug with a
  minimal repro.
- If **even the 64-deep faults** → the op is non-functional on this device for
  *any* flat `out_↔mb_` geometry (the stronger wall).

Attach the resulting senprog and the §1.2 smoking-gun table to the RFC
(`TransposeFaultDeepDive.md` §5).

**Acceptance criteria.** A flat 2-dim `{mb_,out_}` `out_↔mb_` cross-core
transpose handoff compiles HBM-free and runs **value-correct on device** through
stock deeptools (no forced descriptor overrides), with a covering
senulator/device execution test added to the deeptools tree. For (1b),
equivalently: the layout-changing edge runs value-correct via the sanctioned
adapter primitive.

**Owner.** deeptools (they own the PT-transpose codegen and its hardware
contract). The user is on the inductor team; per the standing scope note this is
a handoff, not a patch.

**Effort.** Large / vendor-side. Reverse-engineering the op's validated envelope
against an op with no execution test belongs in deeptools.

---

## 4. Thread 2 — >4k tiled on-chip pipeline (deeptools RFC)

**Problem.** The proven single 2-region move fits handoffs up to ~4k hidden
(1 MB/core × 2 = 2 MB, realistically ~1.6 MB usable after the 20% backend
reserve). Past 4k the producer-output + consumer-input slices exceed LX.
**Move-tiling alone does not close this** — it tiles only the transfer buffer,
while the producer/consumer DL ops still need full-slice LX residency, and the
realize path flips both endpoints to 128 KB-spaced buffers → overlap/corruption
(`Research-2MB-Streaming.md` CORRECTION; `TiledOnChipPipelineDesign.md` §1;
`PerformanceResults.md` "Streaming (>4k) status").

Closing >4k requires a **fused tiled producer→move→consumer pipeline** so that
only a tile (plus its double-buffer twin) is ever LX-resident — collapsing live
LX to a flat `4T + W` independent of S. Every staging primitive this needs
already exists in the Schedule IR (`LoopNode`, `AllocateNode.numBuffers_`,
windowed `TransferNode`, local LX↔LX transfers), **but only for one op at a
time**. The blocker is structural — *one schedule tree = one compute op*
(`TiledOnChipPipelineDesign.md` §3-4), with three hard blocks each cited to
deeptools source:

- **(a) One DL op per core.** `coreIdToDsc_` binds each `coreId` to a *scalar*
  dsc index; the merge path hard-asserts `DT_CHECK(seenDLDsc == false)`
  (`dcg_manager.cpp:821`). A second DL step per core trips the assert.
- **(b) No windowed DL invocation.** A DL op materializes its *full* per-core
  output; its extent is the blocking hierarchy `B_/T_/P_`
  (`designSpaceConfig.cpp:getPcfgLoopCount`), with no "produce only rows
  [k*T,(k+1)*T)" notion.
- **(c) Multi-DSC is data-parallel only.** A producer + consumer pair is the
  unsupported "work A / work B" case:
  `DT_CHECK_MSG(isSameDscGroup(mySDsc), ...)`
  (`L3DlOpsScheduler.cpp:7052-7059`); `createChunkLoops` builds the loop order
  from `dscs_.at(0)` and assumes all DSCs share it.

K-unrolling does **not** shortcut this: K DL steps trip (a); a producer+consumer
pair trips (c) even at K=1; and the DL op still writes its full slice (b)
(`TiledOnChipPipelineDesign.md` §4, §10).

**The ask — the "fused tiled handoff block".** A multi-op fused tiled schedule
region, built FROM the existing IR primitives but combining them across ops
(`TiledOnChipPipelineDesign.md` §5, "Ask 3A"):

1. **Pipeline-group mode** — relax the same-group gate
   (`L3DlOpsScheduler.cpp:7059`) so `dscs_` may hold distinct ops with a
   declared producer→consumer dependency, and the scheduler builds a shared
   outer loop whose body is `[producer-tile, move, consumer-tile]`.
2. **More than one DL op per core** — relax `DT_CHECK(seenDLDsc == false)`
   (`dcg_manager.cpp:821`); bind multiple DL indices per core (e.g. a new
   `coreIdToDscList_`).
3. **Windowed DL-op invocation** — give the DL op a tile window analogous to
   the data-op `dimToStartCordinate`/`dimToSize_`, so producer/consumer compute
   runs over rows `[k*T,(k+1)*T)` per iteration. **This is the genuinely new
   primitive** — there is no DL-op windowing today.
4. **An outer tile loop owning multi-op bodies** — extend `LoopNode` (or add a
   fused-op IR node) so a loop body legally owns COMPUTE nodes from two
   `ownerDsc`s. Whether single-`ownerDsc` is load-bearing across codegen is the
   open question that separates a medium from a large RFC, and needs a
   deeptools-owner read of `moveNode`/`fillLoopOffsetsAndAddresses`/DVS
   lowering.

Tile the **non-stick (mb/row) dim** so every stick stays whole.

**Acceptance criteria.** An S=8192 same-stick handoff (producer add → cross-core
STCDP → consumer bmm/elementwise) compiles HBM-free and runs value-correct
through (gated) deeptools, with only ≤ a tile (≤ LX) live at any time, and the
senprog shows an outer loop wrapping `[producer-tile, L3_LDU/L3_STU move,
consumer-tile]` (`TiledOnChipPipelineDesign.md` §5).

**Owner.** deeptools. This is a new *scheduling* capability, not inductor work —
inductor can emit the multi-op tiled plan into the realize pass, but executing a
tiled, windowed, multi-DL-op-per-core fused block needs the deeptools surface.

**Effort.** Large / vendor-side.

**Cost/benefit caveat.** **>4k is the lowest-relative-payoff regime.** The
matmul cost grows O(N³) while the handoff grows O(N²), so the handoff is a
smaller *fraction* of total time as S grows; measured speedup tapers
(1.22x @1024 → 1.19x @2048 → 1.13x @4096). Ask 3A is a large deeptools effort
aimed at the smallest *relative* win. The recommendation (§9) is to **ship the
≤4k move, gate >4k to HBM, and hold this ask** as a future RFC pursued only
against a concrete *absolute*-byte-saving case (very-long-context decode, large
MoE activations) (`TiledOnChipPipelineDesign.md` §8).

---

## 5. Thread 3 — Asymmetric reshard owner-overlap + DCG-PieceInfo contract (mostly INDUCTOR)

This is the **next concrete inductor step**, and it is mostly pure-inductor with
one small deeptools/contract item. The asymmetric N→M same-stick reshard is the
generalization that turns the symmetric proven path into a general one across
real-model block edges with unequal, non-aligned shard boundaries.

**Status framing (important).** Asymmetric reshard is a **generality item, not a
current blocker.** The current granite graph plans the **symmetric 25→25** edge,
which is **device-landed value-correct** (§1). The 8→25 asymmetric edge it was
originally scoped against was a **cached `k_fast` work-division artifact**, not
what the live graph plans. **[INFER]** this supersedes the
`AsymmetricReshardPlan.md` / `SessionFindings-2026-05.md` framing, which were
written when the granite edge appeared to be 8→25; the brief records the live
graph as symmetric. The asymmetric work below still matters for the general case
(other models, other work-division plans) — it is just no longer on the granite
critical path.

**Problem.** Generalizing from symmetric (32→32, or the reversed `i↔31-i`
mirror) to asymmetric N→M (unequal, non-aligned piece boundaries on the same
stick) is **pure-inductor in the data-movement engine**: the STCDP frontend
*already* computes the producer-piece × consumer-piece overlap cells and the
src→dst ring moves. `DcgFE::createSubPieces(STCDPOpLx*)` loops every output
piece × every input piece, intersects rectangles (`doesPiecesOverlap`), and
registers one LX→LX sub-move per non-empty overlap keyed by src `memId` → dst
`memId` (`AsymmetricReshardPlan.md` §0-1). Inductor's only data-movement job is
to **feed it the producer and consumer pieces at their native (unequal) sizes**
and let DCG do the cells. The builders for this are already landed
(`build_asymmetric_reshard_bridge` / `_partition_pieces` /
`realize_asymmetric_handoff`, `tier0-tier1-onchip` @14a3cbd) and device-proven
on the synthetic 32→8→32 round trip (§1).

Two real-model gaps remain — one a contract item, one an alignment constraint:

**(3a) The static SDSC `labeledDs` carry EMPTY PieceInfo.** In the cached
granite SDSC JSONs, the per-core placement is **not** in the static
`labeledDs.PieceInfo` — it is computed by DCG internally and only surfaces in the
compiled `scheduleTree_` allocate-node (`startAddressCoreCorelet_.data_` per-core
bases + `coordInfo` fold factors). So inductor must **derive** owners and piece
sizes from the schedule tree, which is fragile: the granite splice does exactly
this (`reproduction/granite/derive_edge.py`, `tier0-tier1-onchip` @8916823) —
grouping per-core bases to recover the bands, asserting the folds reconcile, and
fail-closing on gap/overlap. Guessing owners without this fails badly (the 93.8%
-wrong single-stage 8→32, §1).

**(3b) dxp's PCFGToDataflowIR requires producer band-owners to be a SUBSET of
the consumer's cores.** The mixed SuperDSC lives on the consumer, whose DL op
fixes the active corelet set. dxp's `PCFGToDataflowIR` (`senpcfgs_`) **rejects**
an STCDP cell sourced from a core outside that set — granite's native bmm bands
on `[0,4,..28]` overflow the 25-core consumer set `[0..24]`. The granite splice
works around this by remapping producer bands into `[0, num_cores)`
(`prod_owners = [k * (num_cores // n_bands)]`) and guards it with an explicit
check in the bridge: `build_asymmetric_reshard_bridge` raises if any owner ≥
`num_cores` (`tier0-tier1-onchip` @8916823).

**The ask.**

- **(3a-ask, small deeptools/contract item):** a **clean inductor-readable
  contract for DCG-computed pieces** — a stable way to read the per-core owners
  and piece sizes that DCG computes, so inductor does not have to scrape the
  compiled `scheduleTree_` allocate-node. Today the static `labeledDs` PieceInfo
  is empty and the schedule-tree derivation is fragile.
- **(3b-ask, inductor OR deeptools):** **either** inductor work-division aligns
  producer band-owners into the consumer's core set (the remap the granite
  splice already does, productionized into the work-division / realize pass),
  **or** deeptools relaxes the `PCFGToDataflowIR` subset constraint so cells may
  source from any core. The inductor-side remap is the cheaper path and keeps
  the dependency inside our boundary.
- **(3-inductor, the bulk):** productionize `build_asymmetric_reshard_bridge` /
  `realize_asymmetric_handoff` into the realize pass with edge detection and the
  owner-derivation/alignment above — ~250 LOC pure inductor composing with the
  existing symmetric path (`AsymmetricReshardPlan.md` §2, §5). Same-stick only;
  layout-changing stays Thread 1.

**Acceptance criteria.**

- A real asymmetric N→M same-stick edge (a graph that genuinely plans unequal
  shards) compiles and runs **value-correct on device** through the realize pass
  plus the existing dxp gate, with the cross-core ring signature present
  (`L3_LDU`/`L3_STU`) and the negative control clean.
- The owner/piece geometry is obtained through the (3a) contract or a documented,
  asserted derivation — not a guess — and the offline structural gate (cells
  partition the stick with no gap/overlap, reconstruct the consumer partition)
  passes (`AsymmetricReshardPlan.md` §4).
- Producer band-owners are aligned into the consumer's core set (3b), verified
  by the bridge's owner-span guard.

**Owner.** **Mostly inductor** (the realize-pass generalization, the
work-division owner-alignment in 3b). One **small deeptools/contract item**
(3a — the PieceInfo exposure). 3b's relax-the-constraint alternative is a
deeptools option but the inductor-side remap is preferred.

**Effort.** ~250 LOC pure inductor + the small owner-alignment work; the 3a
contract is a small deeptools ask.

---

## 6. Thread 4 — sendnn 3-way comparison (OUT of inductor scope)

**Problem.** A clean 3-way comparison (on-chip vs HBM-restickify vs the
sendnn / torch_sendnn production stack) on the same edges is **doubly-blocked
below the torch-spyre boundary**. The 2.10 rebuild (isolated `/tmp/torch-spyre-210`
plus `/tmp/sendnn210-venv`) cleared the `torch.ops.spyre.overwrite` registration
the 2.11 build had broken, so sendnn loads — but then:

- **Blocker A (generate path):** sendnn `generate` hits `auto_functionalized_v2`
  unhandled in the **torch_sendnn FX converter** (the FMS KV-cache store HOP) —
  a production-stack handler gap.
- **Blocker B (prefill-only path):** prefill-only clears A but hits a deeptools
  **`Dsm::stickify` `map::at` SIGABRT**.

The sendnn standalone attention op runs but is **memcpy-dominated** (compute does
not surface as named kernels) → no clean compute A/B
(`SessionFindings-2026-05.md` §3).

**The ask.** Handoff to the **torch_sendnn + deeptools teams**: add the
`auto_functionalized_v2` HOP handler in the torch_sendnn converter, and fix the
DSM `stickify` `map::at` crash. Both are below the torch-spyre boundary.

**Acceptance criteria.** sendnn `generate` and prefill-only both run the granite
block, producing a sendnn baseline leg for the 3-way comparison on the same
edges; the attention compute is isolable enough for a fair on-chip-vs-sendnn A/B.

**Owner.** **Out of inductor scope** — production stack (torch_sendnn) +
deeptools, per the standing scope note. Until then, the granite tsp-HBM numbers
(prefill-forward **50.21 ms** / generate **~2.17 s**, MEASURED) are the available
baseline leg, and the clean signal is per-edge tsp-HBM vs tsp-on-chip.

**Effort.** n/a to inductor (handoff).

---

## 7. Thread 5 — Dynamic addressing (MoE routing) (FUTURE)

**Problem.** MoE token dispatch/combine is a **same-stick** all-to-all (so *not*
blocked by the transpose wall), but the destination core is **data-dependent**:
it needs a **runtime `memId`**, whereas the proven STCDP path uses static,
compile-time addressing (`memId` is fixed in the PieceInfo at compile time)
(`SessionFindings-2026-05.md` §2.4; `PerformanceResults.md` "Device findings").

**The ask (future).** A **runtime-addressed STCDP** — a same-stick cross-core
move whose destination `memId` is resolved at runtime from the router's
data-dependent token→expert assignment. Scope and feasibility to be worked in a
later session; recorded here for completeness.

**Acceptance criteria.** (Deferred.) A same-stick handoff whose per-core
destination is selected by a runtime index runs value-correct on device.

**Owner.** inductor + runtime (future); depends on deeptools support for
runtime-resolved `memId`. **[INFER]** the runtime/deeptools split is not yet
characterized — the static-vs-runtime addressing boundary needs a deeptools read
before this can be scoped firmly.

**Effort.** Future / unscoped.

---

## 8. The inductor vs deeptools vs out-of-scope split (consolidated)

**INDUCTOR work — we do ourselves:**

- **Thread 3 (the bulk):** asymmetric N→M reshard generalization — productionize
  `build_asymmetric_reshard_bridge`/`realize_asymmetric_handoff` into the realize
  pass with edge detection (~250 LOC; the data-movement engine is already
  deeptools-complete via `createSubPieces`).
- **Thread 3b (owner alignment):** align producer band-owners into the
  consumer's core set in work-division / realize (the preferred path over a
  deeptools constraint relax).
- **Realize-pass productionization** generally: per-size LX allocation (landed),
  sharding-match, and the same-core/same-stick + symmetric path already
  compiler-driven and device-proven.

**DEEPTOOLS RFC asks — handoffs:**

- **Thread 1:** make PTLx execute for flat 2-dim `out_↔mb_` arbitrary-depth (+
  add an execution test), OR provide a sanctioned layout-changing cross-core
  primitive.
- **Thread 2:** the fused tiled handoff block (pipeline-group mode + >1 DL/core
  - windowed DL + outer multi-op loop).
- **Thread 3a (small contract item):** a clean inductor-readable contract for
  DCG-computed pieces (the static `labeledDs` PieceInfo is empty today).

**OUT-OF-SCOPE handoffs — below the torch-spyre boundary:**

- **Thread 4 (sendnn):** torch_sendnn converter `auto_functionalized_v2` handler
  - the deeptools DSM `stickify` crash.

**FUTURE:**

- **Thread 5 (dynamic addressing):** runtime-addressed STCDP for MoE routing.

---

## 9. Prioritization & sequencing recommendation

Ordered by ROI given the proven state and the inductor-team scope:

1. **Thread 3 — asymmetric reshard generalization + owner-alignment (INDUCTOR,
   do now).** It is pure-inductor in the data-movement engine, ~250 LOC, the
   builders are landed and device-proven on a synthetic round trip, and it
   directly broadens real-model coverage. Pair it with the **3a PieceInfo
   contract** ask to deeptools (small, decouples us from scraping the schedule
   tree) and the **3b owner-alignment** done inductor-side. Sequence it on the
   already-landed symmetric 25→25 granite edge as the reference.

2. **Thread 1 — layout-changing transpose (DEEPTOOLS RFC, file next).** This
   unblocks the layout-changing bucket — the majority of high-value pre-matmul
   edges (Q/K/V, pre-matmul reorientation) — **across all sizes**, so it is a
   higher-ROI frontier than >4k tiling. File the RFC with the minimal-repro
   (64-deep vs 2048-deep) result attached so deeptools has a concrete bug or a
   concrete "unsupported geometry, provide the adapter" decision.

3. **Thread 2 — >4k tiled pipeline (DEEPTOOLS RFC, hold).** Document it as a
   future RFC but **do not pursue until a concrete absolute-byte-saving case
   justifies it** — it is a large deeptools effort aimed at the lowest-relative-
   payoff regime. Ship the ≤4k move and gate >4k to HBM (the realize pass already
   fail-closes). Decide by absolute byte saving in a real bandwidth-bound
   workload, not the relative ratio.

4. **Thread 4 — sendnn 3-way (HANDOFF, parallel/independent).** Hand off to the
   torch_sendnn + deeptools teams; it does not block inductor work. Use the
   per-edge tsp-HBM vs tsp-on-chip signal meanwhile.

5. **Thread 5 — dynamic addressing (FUTURE).** Revisit when MoE routing is the
   target; needs a deeptools read on runtime `memId` first.

**One-line strategic read:** do the inductor generalization (Thread 3) now, file
the transpose RFC (Thread 1) next because it unblocks the most value across all
sizes, and hold the >4k tiled block (Thread 2) as a cost/benefit-gated future
ask. sendnn (Thread 4) and dynamic addressing (Thread 5) are out-of-scope and
future respectively.

---

## Source docs

| Doc | Role |
|---|---|
| `SessionFindings-2026-05.md` | the frontier map this scoping formalizes |
| `TransposeFaultDeepDive.md` | Thread 1 evidence + the minimal-repro plan |
| `TiledOnChipPipelineDesign.md` | Thread 2 design + the three structural blocks + Ask 3A |
| `Research-2MB-Streaming.md` | the LX-budget analysis behind Thread 2 |
| `AsymmetricReshardPlan.md` | Thread 3 file-by-file plan + overlap-cell algorithm |
| `PerformanceResults.md` | all device measurements (proven state, §1) |
| `CoreToCoreDataMovementRecipe.md` | the proven single-move recipe + §5 dxp gate + §6b realize pass |
| `NNNN-OnChipRestickifyRFC.md` | the why/what (tier model, Foundation/Transform contracts) |
