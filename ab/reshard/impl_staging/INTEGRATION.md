# Core-to-Core Reshard Substrate -- Integration Doc

## What this substrate is
A genuine core-to-core (LX -> RIU BiRing -> LX, no HBM round-trip) data-movement
substrate for the ONE non-co-assignable edge in the Granite SwiGLU MLP
(`down_proj(silu(gate(x))*up(x))`, 1x512x4096, hidden 12800): the `mul`-output ->
`down_proj` edge. `down_proj` reduces over `K=12800`, which IS the `mul`'s split
dim, so the activation must be gathered core-to-core instead of round-tripping HBM.
This is a **2-D `{mb:4,out:8}` (8x4=32-core) co-split producer feeding a K-reduction
consumer** -- the orthogonal-split reshard flash-ws never emits on device.

## Architecture: flash-ws spine + core-to-core realize body
- **Spine (flash-ws, reused as-is):** `onchip_handoff.plan_onchip_handoffs` /
  `_plan_edge` / `_consumer_to_producer_symbol_map` / `_ownership_identical` (the
  divergent-ownership detector + cost), `restickify_cost.build_transfer_plan` /
  `ring_distance` (orthogonal 8x8 grid hop math), `restickify_ring.decode_op_splits`,
  and the config-gated wiring slot at `passes.py:260`. flash-ws **fail-closes** on
  exactly this edge (`onchip_realize.py:1346 prod_split != cons_split`; every
  realizer hardcodes `src_split_dim == dst_split_dim`), so it costs the edge but
  cannot realize it.
- **Realize body (core-to-core, the only working 2-D authoring):**
  `torch_spyre/_inductor/reshard/{pieces,substrate,cells}.py`. `pieces` builds the
  2-D Piece lists (row-band x col-band; `swiglu_producer_owner = mb + 4*out`,
  `swiglu_consumer_owner` over the K-band each down-proj core reads). `substrate`
  emits the 2-D `STCDPOpLx` bridge (`build_asymmetric_reshard_bridge`), the LX
  flips (`apply_lx_flip`, with `backGapCore_` clear), the 2MB/core liveness gate
  (`allocate_lx_bases`), and the splice. `cells.assert_partition` is the offline
  cell-coverage gate (necessary, not sufficient -- see Acceptance).

## The three grafts (additive, default-OFF, fail-closed)
1. **Planner predicate** `onchip_handoff._is_reduction_input_edge` -- detects the
   producer-splits-the-consumer's-reduced-dim signature (consumer is a `pt` matmul;
   its reduced `in_`/K maps via `symbol_map` onto the producer split dim;
   `consumer_splits['in']==1`). Slots into `_plan_edge`; pure observer.
2. **2-D realizer** `onchip_realize.realize_reduction_reshard` -- unlike the flash-ws
   realizers it accepts `prod_split != cons_split`; its body delegates to the
   `reshard.substrate` stack (no reimplementation). Gated by
   `config.onchip_reduction_reshard` (env `SPYRE_ONCHIP_REDUCTION_RESHARD`,
   default False), mirroring `onchip_handoff_planner/_realize`.
3. **Bundle splice** (mutating) beside `codegen/bundle.realize_onchip_handoff` --
   flips both edge tensors to LX and inserts the **standalone pure-data-op STCDP
   SDSC** (`substrate.splice_reshard_standalone` -> `build_standalone_dataop_sdsc`).
   This routes through `dxp.cpp:255` (`dscs_==0 && dataOpdscs_>0 -> dcg.runDcg`);
   the mixed fold is **rejected** at `SdscTree.cpp:152` ("Datadsc not allowed, use
   dldsc"). Do NOT use `splice_reshard` (mixed) for the serial P2 path.

## The EBR blocker (deeptools)
Decode of `/tmp/c2c-perband/debug/sdsc_2`: the reshard senprog is a **genuine ring
move** -- `L3_STU=248 == L3_LDU=248` (UNIRINGDTU, EAR-routed, dest core from the
`out_` coord, **correct**, carry NO EBR). The broken `3200*core` dest column lives
ONLY on the 40 `L3_STMU` (RINGDTHBMU = ring-store-WITH-HBM-writeback) HBM-mirror
stores. Carrier flow: `dsm.cpp:6764` populate (`foldCoords.at(0)=coreId` ->
`3200*core`) -> `csi.ebrInit_` -> `dlOps.cpp:919 setDestStAddr` ->
`dcgbeCodegen.cpp:2720` EBR initValue -> `L3_STMU` src3. **The scoped carrier
one-liner is BLOCKED:** the reshard SDSC has flattened the producer `{mb:4,out:8}`
band to `{mb:32,out:1}`, so `core//mb_split` is not derivable at `dsm.cpp:6764`. The
real fix is upstream in `dsm/translators/perfDscToSdsc/perfDscToSdsc.cpp` (~2099):
fold the `[out,mb]` HBM tensor `startAddressCoreCorelet_` by OUT band rather than
linear core, preserving the producer split across the SDSC boundary -- a multi-site
translator change requiring a dxp rebuild. flash-ws does NOT solve this (planner-
only/fail-closed; its 1-D pure-LX bridges make `core==column-band` -> `3200*core`
correct by construction and never exercise the EBR leg). The gather
(`InputFetchNeighbor`) does NOT dodge it -- same `computeMulticastOptMetadata`
chokepoint. **Whether the EBR fix is the WHOLE fix is a device question** (LIVE-vs-
DEAD mirror): if device `max_err` stays high after the fix, the mirror was dead and
the real bug is the ring landing (LAR/LBR threading) or producer-LX persistence.

## Warp-spec layer (additive, on the value-correct reshard only)
K-chunked software pipeline on the ONE reshard edge: `warpspec_pipeline_schedule`
(prologue/steady/epilogue, G K-chunks, double-buffered ping/pong), G clamped so each
per-core slice >= 1 stick (64 fp16) and `G | K` (at G=4 on K=12800: 3200-stick
chunks, 0.8MB/bank x2 = 1.6MB <= 2MB LX -- solves the 3.2MB>2MB sizing risk).
`G==1` degenerates to the exact serial `mixed_schedule`, so flag-off is byte-
identical. **Field-order hazard (load-bearing):** the on-disk JSON row is
`[datadsc, dldsc, before_sync, after_sync]` per the parser (`superdsc.cpp:744-762`,
authoritative); the `L3DlOpsScheduler.cpp:378` comment and the `mixed_schedule`
docstring saying `[after,before]` are stale/swapped -- emit before-then-after. Do
NOT set both indices on one row (folds the gather inside that compute's inner loop);
keep chunk g's gather and chunk g-1's compute on separate rows. The warp-spec body
is intrinsically mixed (cannot use the standalone dodge) -> depends on the sec.5 dxp
import-gate patch (already built at `/home/adnan/dt-inductor/build/deeptools-onchip/dxp`).

## Acceptance (load-bearing lesson)
`assert_partition` cell-coverage (9/9) and senprog ring-presence are **necessary but
INSUFFICIENT** -- both passed historically while output was ~0 (the 2-D EBR and the
fused `backGapCore_` bugs sailed through). **Acceptance MUST be on-device `max_err`
vs CPU eager (fp16-noise level) + a negative control** (disable the flag / strip the
reshard senprog -> the run must change), run SOLO on the single shared accelerator.