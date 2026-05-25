# Can the >4k Tiled On-Chip Pipeline Be Inductor-Only? — Definitive Verdict

**Question.** The shipping on-chip handoff is whole-tensor and single-shot: it
keeps a cross-core activation handoff resident in LX, but only when the producer's
output slice plus the consumer's input slice fit in the 2 MB/core LX budget
(the ≤4k regime). For >4k the tensor no longer fits, so the handoff must be
*streamed* — tiled and double-buffered through LX a window at a time. The open
question: can that **tiled producer→consumer pipeline** be built **purely on the
inductor side, with no deeptools change**, or is a new deeptools scheduling
capability genuinely required?

This document consolidates two independent investigations that attacked the two —
and only two — inductor-side levers. Both read current source only (no device, no
edits) against the `pr-copy-elision-option1` deeptools tree. Each cites
`file:line`; inferences are flagged `[INFER]`.

- **Lever 1 — op-fusion:** collapse producer + consumer into one op so the
  existing intra-op tiler streams them.
- **Lever 2 — existing-machinery:** drive existing deeptools knobs
  (`numBuffers_`, `dataStageParam_`, the temporal axis, non-terminal-kernel
  hints, cross-bundle LX persistence) to keep a cross-*op* handoff on-chip and
  tiled, without fusing.

## TL;DR verdict

**There is no inductor-only path to the >4k tiled pipeline. A deeptools change is
genuinely required.** Both levers are closed, for the *same* root reason, and the
prior conclusion in [TiledOnChipPipelineDesign.md](TiledOnChipPipelineDesign.md)
(the "Ask 3A" fused tiled handoff block is a new deeptools capability) **stands**.

Crucially, this is **not** the limitation the shipping on-chip handoff already
solved. That handoff routes the cross-core move through the **data-op channel**
(an `STCDPOpLx` in `datadscs_`) between two separate compute ops, each of which is
still exactly one compute function — so it never touches the wall below. The
tiled pipeline is a **second, orthogonal** capability: *streaming* that handoff,
which requires windowed compute invocation that neither lever supplies.

---

## Lever 1 — op-fusion: REFUTED

Hypothesis: the blocker is "producer and consumer are two ops; a multi-op
SuperDSC trips `isSameDscGroup`/`seenDLDsc`." If inductor fuses them into one op,
that restriction is satisfied by construction and the existing intra-op tiler
(`createChunkLoops` + `numBuffers_=2`) handles the rest — no deeptools change.

**Finding: fusing does not dodge the blocker; it relocates it to a harder
single-DSC wall.**

- **The DSC language contract permits one compute function per op.**
  `designSpaceConfig.h:188-192`: (i) every core does identical work; (ii) **only
  one compute function in the op** (SFP/PE/PT); (iii) a single nested loop
  sequence; (iv) inputs derived from a single primary op. A producer-tile and a
  consumer-tile separated by a cross-core move are two compute functions — not
  representable as one DSC.
- **Multi-`computeOp_` exists but is epilogue-only.** `computeOp_` is a
  `std::vector<ComputeOpInfo>` (`designSpaceConfig.h:213`), and the shipping
  multi-entry cases are fused **epilogues** — matmul/bmm + ADD/BIASADD/RELU6/
  BATCHNORM applied inline to the primary op's PSUM/output, in the **same loop**,
  on the **same blocking**, with **no cross-core move between stages**
  (`dlOps.cpp:98-110,277,354-360`). The only cross-core path inside a
  multi-`computeOp_` op is the matmul **PSUM-reduction ring** (`dlOps.cpp:280-290`,
  asserted to be the matmul accumulation tensor), not a general activation handoff.
- **No windowed DL invocation.** Even granting a hypothetical second `computeOp_`,
  a DL op always materializes its *full* per-core output; the chunk loop is built
  from `dscs_.at(0)` alone (`L3DlOpsScheduler.cpp:3553-3582`) and tiles one op's
  operands **HBM↔LX**, not LX↔LX between two compute stages. There is no
  `dimToStartCordinate`/`dimToSize_` for compute, so the consumer stage cannot be
  asked to run over "rows [k·T,(k+1)·T)" of a producer tile.
- **No outer loop owning two compute stages.** Every schedule node has a single
  `ownerDsc` (`dsc2.h:470`); a loop body cannot legally own COMPUTE nodes from two
  stages with a cross-core move interleaved.

The `isSameDscGroup`/`seenDLDsc` asserts the multi-op angle tripped are *replaced*
by the single-DSC one-compute-function contract. The wall moves; it does not fall.

**Inductor-side is the easy half and does not help:** inductor can already
synthesize a SuperDSC body carrying multiple `opFuncsUsed_`, multiple `datadscs_`,
and a multi-step `coreIdToDscSchedule` (the mixed-fold machinery in
`onchip_realize.py` / `bundle.py:fold_onchip_handoff`). The JSON can be *written*;
what cannot be written is a *legal, value-correct* schedule, because the deeptools
consumer enforces "one compute function per op" and lacks windowed DL invocation.

## Lever 2 — existing-machinery: prior NO UPHELD

Hypothesis: without fusing, inductor drives existing deeptools knobs to keep a
cross-op handoff on-chip and tiled. Four crux questions, each challenged:

- **Q1 — non-terminal kernel hints / cross-bundle LX persistence:**
  **unimplemented in both trees.** The §7.4 route exists only as a doc note
  (`scratchpad_planning.md:226-235`: "preserving LX state across the boundary …
  **requires runtime scheduler support and compiler liveness tracking across
  bundle boundaries**"). Grep for `non-terminal|nonterminal|preserveLX|lxPersist|
  contextSwitch` across `torch_spyre/` and `deeptools/` returns **zero** kernel/
  bundle hits. Inductor has nothing to emit; deeptools has nothing to honor.
  Realizing it *is* the deeptools/runtime change we are trying to avoid.
- **Q2 — inductor emits `temporal>0` + `numBuffers_=2` + `dataStageParam_`:**
  **no.** Inductor hardcodes `"temporal": 0` (`compute_ops.py:90,143`) and emits
  zero `temporal/numBuffers/dataStage` tokens (`superdsc.py`). `numBuffers_`
  (1=none, 2=double, −1=stream; `dsc2.h:978`) is set *inside* the deeptools
  scheduler's `createAllocateNode` (`L3DlOpsScheduler.cpp:450`), only when
  `isHbmPinned()` (`:1160`); temporal folds are *derived* by the scheduler
  (`:6625-6631`), not authored in JSON. And the one existing temporal path is
  intra-op and **spills to HBM** (`dlOps.cpp:405-410`) — the very round-trip the
  handoff removes. `[INFER, strong]`
- **Q3 — K-unroll/multi-op guards real on current source:** **all confirmed, no
  flag.** `DT_CHECK(seenDLDsc == false)` one-DL-per-core (`dcg_manager.cpp:821`,
  reset `:676`); scalar `coreIdToDsc_` = `map<int, DesignSpaceConfig*>`
  (`superdsc.h:67`), "a DSC can only be used in one schedule step"
  (`L3DlOpsScheduler.cpp:388`). Refinement: `isSameDscGroup` itself just returns
  `true` (`:49-52`) — enforcement is **structural**, via `dscs_.at(0)` in
  `createChunkLoops` (`:3559`) and `getCoreSplitDimensions` (`:334-352`).
- **Q4 — does LX persist across `sdsc_execute`?** **yes, measured** (PF /
  single-user VF, the de-facto mode). `deeprt.cpp:207` clears `lxTrackPerCore` —
  but that is the *planner* resetting its LX tracking and conservatively evicting
  to HBM at the SDSC boundary, not a hardware wipe. So keeping the handoff on-chip
  is a scheduling choice: the mixed SuperDSC is *one* route, and (for a same-shard
  handoff) an LX-planner change (don't-evict + coordinate LX addresses across
  consecutive OpSpecs, measured to work on stock dxp) is cleaner.

The temporal/double-buffer machinery and LX residency are computed *inside* the
deeptools scheduler, are intra-op only, and stage through HBM. Inductor's SuperDSC
emit cannot author any of it. This lever is dead.

---

## The unifying root cause

Both levers reduce to the same missing primitive:

> **One DSC = one compute function. There is no windowed DL-op invocation, and no
> outer loop owning multiple compute stages with a cross-core move between them.**

| Angle | What blocks it | Source |
|---|---|---|
| Two ops, one region (prior) | `seenDLDsc` + structural same-group (multi-DSC is data-parallel only) | `dcg_manager.cpp:821`; `L3DlOpsScheduler.cpp:334,3559` |
| One fused op (Lever 1) | one compute function per op; multi-`computeOp_` is epilogue-only; no windowed DL; single `ownerDsc` | `designSpaceConfig.h:190`; `dlOps.cpp:98-110,277`; `dsc2.h:470` |
| Existing machinery (Lever 2) | temporal/`numBuffers_` are scheduler-computed + intra-op + HBM-staged; planner evicts LX to HBM across SDSC (LX itself *persists* in PF / single-user VF — measured); persistence hint unimplemented | `compute_ops.py:90,143`; `L3DlOpsScheduler.cpp:450`; `deeprt.cpp:207`; `scratchpad_planning.md:226-235` |

Op-fusion changes *which* assert you hit; the existing-machinery angle finds the
knobs are scheduler-internal. Neither supplies the windowed-multi-stage-compute
capability. The required surface is the previously-named **Ask 3A**: a fused tiled
handoff block (pipeline-group mode + >1 DL/core + windowed DL invocation + an
outer loop owning two ops' compute nodes) — a deeptools change, RFC handoff.

## What this does NOT close (scoping the negative result)

The negative result is bounded; these remain true and useful:

1. **The shipping ≤4k whole-tensor handoff is unaffected.** It routes the move
   through a data-op (`STCDPOpLx`) in a mixed SuperDSC between two separate
   compute ops, keeping exactly one DL op per core (`onchip_realize.py`,
   `onchip_bridge.py`). It satisfies the one-compute-function contract trivially
   and is value-correct on device today.
2. **Same-loop pointwise epilogue fusion is free.** matmul/bmm + add/relu6/biasadd
   on the same output, same blocking, no cross-core move, already fuses as
   multi-`computeOp_` with no deeptools change (`dlOps.cpp:98-110`). It is
   orthogonal to the >4k gap (no handoff, no tiling-for-residency).
3. **Strategic note (unchanged):** >4k is the lowest-relative-payoff regime —
   matmul O(N³) compute dwarfs the O(N²) handoff. Ship the proven ≤4k two-region
   move, gate >4k to HBM, and pursue Ask 3A only against a concrete
   absolute-saving case.

## Caveat

Both investigations are source-read analyses, not device experiments. Two agents
reading current source independently reached the same structural conclusion via
different levers, which is strong corroboration — but weight the architectural
read against device validation before treating it as final.

## Source index

- `designSpaceConfig.h:188-192,213` — DSC language: one compute function, identical
  work, single loop nest, single primary op; `computeOp_` is a vector.
- `dlOps.cpp:98-110,277,354-360,280-290,405-410` — multi-`computeOp_` = inline
  epilogue; RING dt is PSUM reduction; temporal transfer spills to HBM.
- `dsc2.h:470,978` — single `ownerDsc`; `numBuffers_` enum.
- `L3DlOpsScheduler.cpp:49-52,334-352,388,450,1160,3553-3582,6625-6631` —
  `isSameDscGroup` returns true; structural same-group via `dscs_.at(0)`; chunk
  loop / `numBuffers_` / temporal folds are scheduler-internal, HBM-pinned.
- `dcg_manager.cpp:676,821` — `DT_CHECK(seenDLDsc == false)`, one DL per core.
- `superdsc.h:29-44,67` — scalar `DscScheduleStep`; `coreIdToDsc_` map.
- `deeprt.cpp:207` — planner clears `lxTrackPerCore` and evicts LX to HBM per
  SDSC; LX itself *persists* across the boundary in PF / single-user VF (measured),
  so this is a scheduling choice, not a hardware wipe.
- `compute_ops.py:53,90,143` — inductor hardcodes `temporal: 0`.
- `op_spec.py:47-64`, `spyre_kernel.py:414-457`, `superdsc.py:510-609`,
  `compute_ops.py:208-424` — inductor: one node → one OpSpec → one SDSCSpec → one
  DSC / one computeOp.
- `onchip_realize.py`, `onchip_bridge.py`, `bundle.py:fold_onchip_handoff` —
  shipping mixed-fold proves the JSON can carry a folded move (not a 2nd compute).
- `scratchpad_planning.md:226-235,27-36` — §7.4 non-terminal-kernel persistence is
  a documented future runtime+compiler change; planner's LX-persistence assumption
  is flagged a correctness gap.
- [TiledOnChipPipelineDesign.md](TiledOnChipPipelineDesign.md) §1,§4,§5,§8 — prior
  multi-op-region conclusion and Ask 3A.
