# Granite Block LX Communication Classes

Date: 2026-06-29

Branch: `ah/comms-collectives`

Reproduction runbook:
`docs/results/granite_e2e/comms_collectives_reproduce_runbook_20260630.md`

## Goal

Use the dl-dsc LX relayout path as the base for removing the remaining
activation HBM spills in a Granite block.  PR1 covers resident scatter relayout;
this branch records and prototypes the next communication classes separately
from the merge-ready PR.

## Current Baseline From Artifacts

The latest corrected-env Granite prefill artifacts measured:

| variant | kernel ms/iter | median wall ms |
|---|---:|---:|
| baseline, relayout off | 14.6977 | 34.8575 |
| full Torch LX + backend LX=1 | 12.3391 | 32.4521 |
| comms branch, collectives enabled | 12.3147 | 32.5027 |

The run environment matters.  Torch must see `DXP_LX_FRAC_AVAIL=0` while the
DXP subprocess must see `DXP_LX_FRAC_AVAIL=1` through the split wrapper recorded
in the reproduction runbook.

The latest collectives-enabled run was:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_splitenv_20260630_040302
```

It passed, but it did **not** emit any `lxRelayoutClassifications_` fields in
the generated SDSCs.  The current collectives prototype therefore did not fire
on the full Granite block.  The measured speedup is still the established
full-LX/boundary-clone win, not a new collective-lowering win.

## 2026-06-30 Current-Source DXP Replay

The latest validation rebuilt Deeptools from the current
`ah/comms-collectives` checkout and used a fresh split wrapper:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/tools/dxp-split-wrapper-current/dxp_standalone
```

Do not use the older `pr_lx_scatter_20260629_170114` wrapper for this branch;
it points at an older DXP binary.

Three current-source runs are archived:

| run | result | conclusion |
|---|---|---|
| `granite_prefill_collectives_current_dxp_20260630_060731` | import failure | runtime library path loaded stale `libdvs.so` |
| `granite_prefill_collectives_current_dxp_runtimefix_20260630_061024` | DXP pure-DL failure | scheduled pure DL SDSCs were incorrectly routed through the mixed data-op path |
| `granite_prefill_collectives_current_dxp_routingfix_20260630_061426` | attention DDC failure | real current-source relayout gap reproduced |

The runtime import failure was fixed by putting installed Spyre runtime
libraries before inherited `dt-inductor` libraries:

```bash
export LD_LIBRARY_PATH=/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/spyre-comms/lib:/home/adnan/dt-inductor/build/libaiupti/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

The pure-DL DXP failure was fixed by tightening the mixed-SDSC route in
`dxp/dxp.cpp`: only SDSCs with both a populated `coreIdToDscSchedule` and
nonempty `dataOpdscs_` should call `runDcgForDataOpsDlOps`.  Scheduled pure DL
SDSCs must stay on `runDcgForDlOpsStandalone`.

After that routing fix, the run reached the attention operand relayout gap:

```text
DtException: Unexpected corelet cardinality mismatch for nodes
allocate-Tensor1_lx and transfer_lds0_src:lxlu_dst:sfp
```

A debug replay of the failed attention SuperDSC is archived at:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/debug_relayout_replay_20260630_062207
```

The replay shows Deeptools master-style automatic relayout insertion is active:

```text
Inserting relayout for: Tensor0
Lx space found, inserting stcdpLx
Inserting relayout for: Tensor0
Lx space found, inserting stcdpLx
Inserting relayout for: Tensor1
Lx space not found, inserting stcdpHBM
```

So current Deeptools can synthesize LX relayouts for resident `Tensor0`
scatter-like mismatches, but the value-side attention operand `Tensor1` does
not fit the resident-relayout model.  Torch classifies that operand as:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
read_index = 1
labeledDs_[1] = Tensor1, dsType = KERNEL, memOrg = {lx}
```

This confirms the current gap is communication-class specific.  Resident
scatter works.  The attention operand needs staged/all-gather-like input
movement instead of full consumer-view materialization.

The current artifact comparison is:

| metric | baseline off | full Torch LX | comms collectives-on |
|---|---:|---:|
| SDSC count | 44 | 47 | 47 |
| `ReStickifyOpHBM` rows | 5 | 5 | 5 |
| SDSCs with `lxRelayoutClassifications_` | 0 | 0 | 0 |
| LX allocate rows | 53 | 66 | 66 |
| HBM allocate rows | 61 | 54 | 54 |

So the current speedup does not come from removing the named
`ReStickifyOpHBM` rows.  It comes from keeping more intermediate allocations in
LX inside the fused chains.  The five explicit HBM restickifies remain and are
the next gap to address.

## Implemented Class: Resident Scatter

This covers edges where:

- a producer owns final tensor slices in LX;
- a consumer needs the same tensor with a different per-core resident view;
- the post-relayout resident pieces fit in LX.

Torch records producer residency in the consumer input allocation coordinates.
Deeptools sees that the input tensor distribution differs from the consumer
compute split and inserts an `STCDPOpLx`-based relayout before compute.

This is a materialized resident-view class: the relayout creates the consumer
view in LX before the consumer op.

## Remaining Granite Class: Matmul Operand Broadcast

The remaining non-weight Granite prefill gap is the attention value-side matmul
operand.  The producer shards the operand across cores, while the consumer
matmul split needs a loop-scoped stream of pieces from those producer shards.

Artifact signature:

- edge: `buf21 -> buf22`
- current classification: `matmul_operand_broadcast`
- communication pattern: `all_gather_replicate`
- observed failing SDSC: attention value-side `batchmatmul`

This is not the same as resident scatter.  If we force the current resident
relayout mechanism, Deeptools attempts to materialize a full post-relayout
operand per consumer core.  The artifact repro showed about 4 MiB per core,
which fails LX capacity and is the wrong lowering shape even if capacity were
available.

The intended contract is:

1. Torch classifies the edge as a non-primary matmul operand collective.
2. dl-dsc coordinates describe the producer tensor distribution and consumer
   operand/compute requirement.
3. Deeptools synthesizes scheduled ring movement in or around the matmul operand
   transfer loop.
4. Working-set reduction decides staging granularity; the communication class
   remains explicit so cost and lowering can reason about it.

## Newly Confirmed Gap: Explicit Layout Restickify HBM

The corrected artifacts show that all variants still contain five
`ReStickifyOpHBM` rows:

| source | shape | row | split | scope |
|---|---:|---|---|---|
| `arg2_1`, attention QKV projection weight | `[6144,4096]` | `sdsc_fused_linear_rms_norm_0/sdsc_7.json` | `{mb:32,out:1}` | weight prelayout, out of scope |
| `mul_6` / `buf13`, computed attention activation | `[1,32,512,128]` logical target | `sdsc_fused__scaled_dot_product.../sdsc_9.json` | `{mb:32,x:1,out:1}` | in scope |
| `arg5_1`, attention output projection weight | `[4096,4096]` | `sdsc_fused__scaled_dot_product..._add_linear.../sdsc_0.json` | `{mb:32,out:1}` | weight prelayout, out of scope |
| `arg7_1`, fused FFN gate/up projection weight | `[25600,4096]` | `sdsc_fused__scaled_dot_product..._add_linear.../sdsc_10.json` | `{mb:25,out:1}` | weight prelayout, out of scope |
| `arg8_1`, FFN down-projection weight | `[4096,12800]` | `sdsc_fused_add_linear_mul_3/sdsc_0.json` | `{mb:1,out:25}` | weight prelayout, out of scope |

This distinction is important.  The benchmark uses empty Spyre parameters for
speed and reproducibility, but those tensors are still model parameters in the
compiled graph.  Their restickifies are weight-layout preparation problems.
They should be handled by offline/preload weight layout work, not by this
communication-class branch.

The only remaining non-weight restickify in this run is the computed attention
activation restickify.  That is the row this branch should continue to
investigate.

Latest confirmation from
`/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_selective_relayout_retry_20260630_044850`
shows the same split:

```text
ReStickifyOpHBM rows: 5
layout_restickify_weight classifications: 4
disabled runtime relayout reservations: buf14:buf46, buf22:buf21
```

The four weight rows are recorded as:

```text
kind = layout_restickify_weight
communication_pattern = offline_weight_prelayout
unsupported_reason = graph-input/parameter restickify is owned by offline weight prelayout, not runtime LX relayout
```

That means they are deliberately excluded from the runtime communication scope.
The remaining runtime issue is the computed attention activation restickify,
currently emitted as an LX input to HBM output row, plus the dependent matmul
operand broadcast that prevents that activation from staying fully resident.

After adding explicit computed-restickify classification, the latest run is:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_layout_restickify_class_20260630_050148
kernel_ms_per_iter: 12.0335
median wall ms: 32.5332
```

Generated SuperDSC classification counts:

| class | count | realized |
|---|---:|---|
| `scatter` | 14 | yes |
| `layout_restickify_weight` | 4 | no, offline weight prelayout |
| `layout_restickify_activation` | 1 | no, needs LX layout-restickify contract |
| `matmul_operand_broadcast` | 1 | no, full resident reservation does not fit |

The one activation class appears on:

```text
sdsc_fused__scaled_dot_product.../sdsc_10.json
source: buf46
kind: layout_restickify_activation
communication_pattern: layout_transform_then_operand_broadcast
unsupported_reason: computed activation restickify needs an LX layout restickify contract plus loop-scoped matmul operand lowering
```

The one remaining operand broadcast class appears on:

```text
sdsc_fused__scaled_dot_product.../sdsc_18.json
source: buf21
kind: matmul_operand_broadcast
communication_pattern: all_gather_replicate
unsupported_reason: backend relayout reservation did not fit in scratchpad
```

The generated SDSC gives the key handoff fields:

```text
sdsc_18 root op: 18_batchmatmul
consumer work division: {x:1, mb:32, out:1, in:1}
classification read_index: 1
labeledDs_[1]: Tensor1, dsType=KERNEL, memOrg={hbm,lx}
```

After the reservation-skip change, a realized-collective probe generated the
intended Torch-side contract:

```text
run root:
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_realized_probe_20260630_054408

sdsc_18 Tensor1:
kind: matmul_operand_broadcast
realized: true
read_index: 1
labeledDs_[1]: Tensor1, dsType=KERNEL, memOrg={lx}
allocate-Tensor1_lx coordinates: producer out-slice ownership per core
```

That run then failed in DXP/DDC with:

```text
Unexpected corelet cardinality mismatch for nodes
allocate-Tensor1_lx and transfer_lds0_src:lxlu_dst:sfp
```

That first failure was produced through the older split wrapper from
`pr_lx_scatter_20260629_170114`, so it was only provisional evidence.  The
current-source replay above retested the same class with a DXP binary rebuilt
from the current `ah/comms-collectives` Deeptools checkout.  The same backend
gap remains, with more precise evidence: Deeptools inserts resident
`STCDPOpLx` relayouts for `Tensor0`, then cannot handle the `Tensor1`
all-gather-like matmul operand without falling back to HBM and hitting the DDC
corelet mismatch.

This matters because the existing Deeptools InputFetchNeighbor path is currently
hard-coded around the DL `INPUT` data-structure type.  The Granite operand gap
is a non-primary matmul operand, so the consumer-side data-structure is
`KERNEL`, not `INPUT`.  The backend already has `STCDPOpLx` ring movement and
input-neighbor scheduling machinery, but that machinery is not yet generalized
to "fetch this specific matmul input ordinal / ds type."  That is the current
blocking backend gap for using staged all-gather-like operand movement here.

The latest prototype generalized enough of that path to route the non-`INPUT`
`KERNEL` operand through loop-scoped `STCDPOpLx` input-neighbor lowering.  That
moved the failure past the earlier DDC/corelet mismatch and into DCC legality:

| attempt | result |
|---|---|
| full resident materialization | invalid LX immediate, `LX_MODLRFIMM :: lrfimm:-4161536` |
| chunked input-neighbor with original chunking | `dtTable=4096`, too large/timeouts |
| grouped input-neighbor with `x=16` chunk override | DCC reaches lowering but fails `Max IBUFF(256) Current IBUFF(745)` for L3LU |
| disabling subpiece reuse | worsens to `Current IBUFF(1729)` |
| forcing DCC O3 | no improvement; same `Current IBUFF(745)` |
| compact grouped L3LU cap=16 | worsens to `Current IBUFF(1377)` |
| compact grouped L3LU cap=4 | fails with `Current IBUFF(1346)` |
| compact grouped L3LU cap=2 | DXP/DCC timeout after 180s, still `dtTable=256` |
| staged input-neighbor, 4 producer cores per row | passes DXP replay; 8 data-op stages, each `dtTable=4 inpSP=4 outSP=4 maxL3SU=1 maxL3LU=4` |
| staged input-neighbor with compact coordinate-sorted L3SU/L3LU ordering | passes DXP replay; same 8 staged rows |

The latest Deeptools DCC fix uses `dataOpdscs_.at(datadscIdx)` for the staged
data-op descriptor lookup.  After that fix, all-gather/replicate compile
passes.  The stage-size compile sweep passes for
`DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=4`, `8`, `16`, and `32`.

The useful conclusion is that this is now past "can the schema express the
edge?" and into "can the backend synthesize a compact legal collective
program?"  One giant input-neighbor row flattens the all-gather into too many
receive fragments.  Small grouping caps do not fix the shape: they either still
emit too much L3LU control flow or make DCC compile time unacceptable.  Splitting
the same logical all-gather into legal loop-scoped movement phases resolves the
known DCC IBUFF failure for the isolated attention bundle.  The artifact is:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/attention_broadcast_replay_staged4_20260630_101812
/home/adnan/codex-isolated/comms_collectives_20260629/runs/attention_broadcast_replay_staged4_sorted_20260630_102627
```

This is not yet the final collective abstraction.  It is the first legal backend
shape for the Granite `matmul_operand_broadcast` class: multiple small
`STCDPOpLx` rows scheduled before the consumer compute row, instead of one
oversized input-neighbor row.

The first full Granite execution attempt with staged-four append ordering
compiled all required rows, then failed at runtime with a PCIe bus-fence during
the first block iteration:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_staged4_20260630_102000
RAS::PCI::BusFence, code 0xa35e
```

This means the `matmul_operand_broadcast` path has advanced from compile-time
blocked to runtime-safety blocked.  Full Granite currently bus-fences at
runtime.  The next diagnostic is to run the full block with
`DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=32` to test the larger-stage compiled
shape before changing the lowering again.

CDX validation also requires a clean runtime `LD_LIBRARY_PATH`: keep
`/opt/ibm/spyre/deeptools/lib`, `/opt/ibm/spyre/runtime/lib`, and
`/opt/ibm/spyre/spyre-comms/lib` ahead of inherited `dt-inductor`, flex, or
older Deeptools library paths.

## Backend Integration Shape For Operand Broadcast

The remaining runtime classes should not be forced through the current
resident-relayout insertion path.  That path is intentionally materialized:

```text
DXP: runDsmRelayout(sdsc, executionStep, memTrackers, relayout_sdscs)
  -> SdscRelayoutInsertion.cpp
  -> insert an STCDPOpLx relayout before the consumer
  -> reserve a full post-relayout consumer LX view
  -> fall back to HBM relayout if LX does not fit
```

That is correct for resident scatter, but wrong for non-primary matmul operands.
For `buf21 -> buf22` and the dependent `buf46 -> buf14` path, the consumer
needs a loop-scoped stream of operand pieces.  Materializing the entire
consumer-side operand view costs about MiBs per core and either fails LX capacity
or falls back to HBM, which defeats this branch's purpose.

Deeptools already has a closer lowering entrypoint:

```text
DcgManager::runDcgForInputFetchNeighbor(SuperDsc& main, SuperDsc* pre)
  -> DcgFE::generatePcfgIRForDataOpInpFetch(main, pre, ...)
  -> fillDataDSCForInputFetchNeighbor(...)
  -> STCDPOpLx with producer/consumer subpieces and chunk traffic
```

So the next backend patch should be DXP wiring, not a new ring primitive:

1. Read top-level `lxRelayoutClassifications_`.
2. For `kind == matmul_operand_broadcast`, skip resident
   `SdscRelayoutInsertion` for that input.
3. Match the classification to the consumer input using `read_index`.
4. Generalize InputFetchNeighbor from hard-coded `DsTypes::INPUT` to the
   classified operand's actual ds type, e.g. `DsTypes::KERNEL` for the Granite
   value-side matmul.
5. Locate the producer SuperDSC or producer coordinate metadata that owns the
   LX-resident operand.
6. Call the existing InputFetchNeighbor generation path, or synthesize its
   equivalent `STCDPOpLx` data-op internally from the dl-dsc coordinate maps.
7. Preserve the resulting `STCDPOpLx` schedule as a loop-scoped input movement
   associated with the consumer batchmatmul.
8. Reject, rather than silently HBM-fallback, if the producer/consumer metadata
   is insufficient for this path.

An isolated prototype for this direction is archived here:

```text
docs/results/granite_e2e/dldsc_backend_patches/inputfetch_neighbor_consumer_dstype_prototype.patch
```

That patch generalizes parts of the InputFetchNeighbor path from hard-coded
`DsTypes::INPUT` toward a consumer operand ds type such as `DsTypes::KERNEL`.
It is not production-ready: the prototype inferred the consumer ds type from a
data-op schedule index, which is not guaranteed to equal the consumer read
index.  The production contract should carry an explicit consumer `read_index`
or ds-type selector from Torch's classification metadata into backend lowering.

This keeps the contract aligned with the North Star:

- Torch classifies and costs the communication class.
- dl-dsc coordinates describe producer residency and consumer demand.
- Deeptools synthesizes physical movement and chunk scheduling.
- Later work can overlap this movement with compute instead of merely placing it
  before compute.

## Primitive Audit

The expanded objective is to understand and eventually implement the on-chip
communication classes needed by Granite and nearby transformer workloads:
broadcast, multicast, gather, all-gather, reduce, and all-reduce.  Current
source inspection shows three different support layers:

| primitive | evidence in Deeptools | usable through current dl-dsc LX relayout path? | status for this branch |
|---|---|---|---|
| scatter / permutation | `dxp/SdscRelayoutInsertion.cpp` inserts `STCDPOpLx` for mismatched LX input coordinates | yes, for resident post-relayout views that fit in LX | implemented and validated for PR1-style resident scatter |
| broadcast / multicast | `STCDPOpLx` carries `prodConsList`; DCG computes multicast metadata via `computeMulticastOptMetadata`, `promoteToMode3`, and GTR multicast lowering | partly; resident relayout can express one-to-many pieces, but materializes the destination view | exists, but Granite operand broadcast needs staged input-neighbor lowering, not resident materialization |
| gather | `GatherOpHBM` exists in data-op/DSM/DCG paths | no; current named gather path is HBM-oriented, not LX-to-LX relayout | not implemented for this branch's on-chip path |
| all-gather | multi-AIU optimizer has `OpFuncs::AllGather`; input-neighbor fetch can generate many producer-to-consumer subpieces with `STCDPOpLx` | not yet from Torch dl-dsc relayout metadata | Granite `matmul_operand_broadcast` is the first on-chip all-gather-like target |
| reduce | DL compute ops include `SUM`, `SUM_NONSTICK`, `MAX`, etc.; ISA has `REDUCE`; psum ring paths exist for reductions | yes as compute/reduction, not as a standalone relayout primitive | existing compute path, not the current Granite spill |
| all-reduce | multi-AIU collective code has `OpFuncs::AllReduce` and DSM collective implementations | no single-AIU LX relayout hook from this branch yet | future primitive; not needed for current Granite prefill spill evidence |

The important conclusion is that "the primitive exists" is not enough.  For this
branch, the primitive must be reachable from Torch's dl-dsc coordinate contract
and must avoid HBM fallback.  Today only resident scatter satisfies that end to
end.  The next Granite-relevant class is the all-gather-like staged operand
broadcast for attention batchmatmul inputs.

The first backend schema gap is also clear: Torch emits
`lxRelayoutClassifications_`, but Deeptools `SuperDsc` did not import or
preserve that field.  The local Deeptools `ah/comms-collectives` patch now adds
a small scalar metadata map so DXP can branch on `kind` without re-deriving the
class from raw coordinates.

These rows are not direct producer-to-consumer LX distribution mismatches by the
time `plan_lx_relayouts()` runs.  They have already been materialized as
explicit `spyre.restickify` / `ReStickifyOpHBM` nodes during the stick-layout
pipeline:

```text
propagate_spyre_tensor_layouts
optimize_restickify_locations
finalize_layouts
insert_restickify
...
span_reduction
work_distribution
scratchpad allocation / LX planner
```

That placement matters.  The dldsc LX relayout path in Deeptools can currently
see an LX input whose tensor distribution differs from the consumer compute
distribution.  It cannot replace an HBM `ReStickifyOpHBM` node that Torch has
already inserted before scratchpad planning.

This is also not the same class as resident scatter.  The in-scope row is a
layout/stick restickify: the tensor's physical stick/layout form changes, not
only the core that owns each already-formed piece.

Current Deeptools `SdscRelayoutInsertion.cpp` also treats relayout input and
output LDS layout/stick metadata as the consumer form.  That is sufficient for
same-layout redistribution, but not enough to describe a true LX layout
restickify with different pre-layout and post-layout forms.  A production
layout-restickify solution needs an explicit pre/post layout contract, or a
backend LX restickify primitive that can derive that contract from dl-dsc.

## Branch State

This branch now records non-primary matmul operand mismatches as classified but
unrealized LX relayout plans:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
realized = false
```

That metadata intentionally does not populate the existing realized scatter
input map.  This avoids asking Deeptools to run the resident-scatter lowering on
a collective edge.

The full Granite block currently does not emit those classifications because
the relevant edges are hidden behind already-inserted HBM restickify nodes.  The
classifier is still useful for synthetic and narrower graph shapes, but the
full-block path requires earlier intervention.

## 2026-06-30 All-Gather Runtime Finding

The first `matmul_operand_broadcast` prototype made the attention value-side
operand visible as an all-gather-like LX relayout edge.  DXP compile succeeded
after fixing the mixed data-op PCFG indexing bug:

```text
dcg/dcg_fe/pcfg_gen/pcfg_gen.cpp
  mySDscMain.dataOpdscs_.at(datadscIdx)
```

but the full Granite prefill run fenced the AIU at runtime:

```text
CDX stage32 compact broadcast:
/home/adnan-cdx/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_stage32_20260630_112057
PROC_RC=255
RAS::PCI::BusFence

CLC stage32 non-compact broadcast:
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_noncompact_stage32_20260630_113822
PROC_RC=255
RAS::PCI::BusFence
```

The data-op dump that exposed the issue is:

```text
/home/adnan-cdx/codex-isolated/comms_collectives_20260629/runs/attention_dataop_dump_stage32_20260630_112740/dumps/18_batchmatmul-dataop-0.json
```

It showed an attention operand with logical size:

```text
in=512, out=128, x=32, fp16 => 4,194,304 bytes
```

The attempted all-gather materialized that full operand on every consumer core.
That is not a safe resident LX relayout.  The compact path also produced
whole-operand ring spans (`addrSpan=4194304`) for shard transfers, and the
non-compact path still required a full replicated destination view.  The right
communication class for this case is tiled or loop-scoped operand movement, not
full resident all-gather.

The Torch classifier now has a resident collective size guard:

```text
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVE_MAX_BYTES=1048576
```

With the guard enabled, the same Granite prefill run is value/runtime clean:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_20260630_114425
returncode=0
shape=[1,512,4096]
wall median=24.495 ms
trace kernel_ms_per_iter=12.0539
```

Fresh replay after resetting `adnan-clc-spyre-dev-pf` and adding the frontend
`ReStickifyOpLx` SDSC contract test also passes:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_clc_20260630_122458
returncode=0
shape=[1,512,4096]
wall median=23.9255 ms
trace kernel_ms_per_iter=12.0628
tests/inductor/test_lx_relayout_dldsc.py: 14 passed in 0.16s
```

The post-guard SDSC metadata proves the unsafe collective is now classified but
not realized:

```text
sdsc_18.json
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
estimated_tensor_bytes = 4194304
realized = false
unsupported_reason = resident all-gather would replicate 4194304 bytes per consumer core,
                     exceeding SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVE_MAX_BYTES=1048576;
                     needs tiled/streamed lowering
```

This narrows the communication taxonomy:

| class | resident full-materialization status | Granite status |
|---|---|---|
| scatter/permutation | safe when destination view fits LX | PR1 validated |
| small broadcast/multicast/all-gather | possible for small tensors; still experimental | guarded by max resident bytes |
| attention-sized all-gather | unsafe as resident replication | needs tiled/loop-scoped movement |
| layout restickify activation | not a pure communication move | Torch can now emit `ReStickifyOpLx` for safe LX-to-LX transforms; Granite's remaining instance still feeds a non-primary matmul operand and therefore waits on loop-scoped operand movement |

The detailed guarded-run spill inventory is archived in
`comms_collectives_guarded_spill_inventory_20260630.md` and companion CSV
`comms_collectives_guarded_spill_inventory_20260630.csv`.  Independent
inspection found the same counts:

```text
SDSC JSONs: 47
ReStickifyOpHBM rows: 5
lxRelayoutClassifications_: 20
realized scatter relayouts: 14
unrealized classes: 4 offline weights, 1 computed activation restickify,
                    1 unsafe resident all-gather
```

The current Deeptools `ah/comms-collectives` diff should be treated as an
exploration branch.  The candidate pieces to preserve are:

- scheduled data-op indexing with `dataOpdscs_.at(datadscIdx)`;
- consumer-DS-type plumbing for relayout operands that are not ordinary
  `INPUT`;
- mixed scheduled routing through `runDcgForDataOpsDlOps`;
- the DXP skeleton that recognizes loop-scoped matmul operand broadcast.

The diagnostic/risky pieces should stay recorded but not be treated as
production-ready yet:

- `DXP_DUMP_RELAYOUT_DATAOPS_DIR` data-op dump hook;
- hidden broadcast staging/compactness env toggles;
- broad grouped multi-destination `STCDPOpLx` changes before destination
  address selection, GTR/GTRIMM behavior, ordering, and consumer deduplication
  are proven safe.

## Next Implementation Step

Torch now has the narrow frontend pieces for LX restickify:

- `ReStickifyOpLx` is a distinct op constant.
- SDSC padding/back-gap handling treats `ReStickifyOpLx` like
  `ReStickifyOpHBM`.
- `spyre_kernel.py` emits `ReStickifyOpLx` when a restickify data op has both
  input and output allocated in LX.
- `plan_lx_relayouts` realizes computed activation layout transforms only when
  they do not feed a non-primary matmul operand.  This avoids accidentally
  enabling the unsafe resident all-gather path.

The remaining Granite HBM row is therefore not solved by one more Torch
allocator tweak.  It is a chain:

```text
resident scatter -> computed activation layout restickify -> non-primary matmul operand broadcast
```

The next prototype should target that chain without falling back to full
resident replication:

1. Ignore graph-input/parameter weight restickifies; those belong to offline
   weight prelayout/preload work.
2. Identify the computed activation restickify whose input and output can both
   be LX resident.
3. Preserve the original producer tensor layout and the consumer-required target
   layout.
4. Emit a dl-dsc contract that exposes an LX input with explicit producer
   coordinates and a post-layout consumer requirement, without materializing an
   HBM `ReStickifyOpHBM` node.
5. Lower the layout transform through Deeptools `ReStickifyOpLx`, not generic
   scatter.  Deeptools has low-level `ReStickifyOpLx` support, but the current
   DXP relayout insertion path does not synthesize it from
   `layout_restickify_activation` metadata.
6. After layout-restickify is working, return to the attention value-side
   `matmul_operand_broadcast` edge and lower it as loop-scoped movement rather
   than post-relayout full materialization.

Concrete frontend/backend gap:

- Torch emits safe standalone `layout_restickify_activation` transforms as
  realized metadata, but the Granite attention case remains unrealized because
  it is also a non-primary matmul operand broadcast.
- The existing allocation-coordinate mechanism can describe producer ownership,
  but not the stick-layout transform by itself.  The contract needs explicit
  pre-layout and post-layout fields.
- Deeptools has `ReStickifyOpLx` primitives, but no DXP handler that consumes
  the `layout_restickify_activation` class and creates an LX restickify data op.
- The output of that LX restickify then feeds a matmul operand, so the dependent
  all-gather/broadcast still needs loop-scoped operand movement.

## 2026-06-30 Staged Broadcast Follow-Up

After the after-sync experiment, we found one concrete DCG metadata bug in the
compact staged broadcast path.  The generated transfer table for the attention
operand had sub-stick producer shards:

```text
producer shard out = 4
stick size out = 64
```

`finalizeBurstInfo()` used integer division for the stick adjustment, which
made `maxBurst=0`, `numTransactions_=inf`, and `trCost=inf`.  The exploratory
Deeptools branch now ceil-divides and clamps those values to at least one.

Compile-only replay after that fix:

```text
DUMP=/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_dataop_dump_ceilburst_20260630_125425
18_batchmatmul-dataop-{0..7}.json
entries=4
inf=0
maxBurst={32: 4}
numTransactions={512: 4}
cMemIDs=[32]
cIDXs=[1]
```

That removed the invalid metadata but did not make the hardware program safe.
The compact grouped-output path still bus-fenced:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_staged4_ceilburst_clc_nofmspath_20260630_125846
return code 255
RAS::PCI::BusFence code 0xa35e
```

We then split two concepts that had been overloaded in the backend prototype:

- `compactInputNeighborBroadcast`: whether DCG uses grouped compact output
  pieces and compact ordering/span behavior;
- `partialOutputCoverage`: whether the data-op is one stage of a larger
  loop-scoped operand movement, so one row is not expected to cover the whole
  output LDS piece by itself.

With that marker, `DXP_LX_RELAYOUT_BROADCAST_COMPACT=0` now routes through
InputFetchNeighbor instead of ordinary STCDP coverage checking.  DXP replay
passes:

```text
DUMP=/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_dataop_dump_noncompact_partialroute_20260630_130859
18_batchmatmul-dataop-{0..7}.json
entries=4
inf=0
maxBurst={1: 4}
numTransactions={16384: 4}
cMemIDs=[32]
cIDXs=[32]
```

The noncompact path also bus-fenced on hardware, both with and without Kineto:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_noncompact_partialroute_clc_20260630_130928
return code 255
RAS::PCI::BusFence code 0xa35e

/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_noncompact_partialroute_noprofile_clc_20260630_131206
return code 255
RAS::PCI::BusFence code 0xa35e
```

This rules out Kineto/AIUPTI as the primary cause.  It also tells us the
problem is not only grouped multi-address output pieces.  The remaining gap is
the communication contract itself: the current staged path still asks the
backend to prepare a full matmul operand region before consumer compute.  The
attention operand requires a true matmul-loop-scoped movement, where the
transfer is tied to the consumer matmul tile/loop that immediately consumes it.

### Current Granite Spill Taxonomy

Latest independent inventory compared the guarded passing run and the forced
staged-broadcast run:

```text
guarded success:
  /home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_clc_20260630_122458
  returncode=0
  kernel_ms_per_iter=12.0627818

forced staged broadcast:
  /home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_staged4_ceilburst_clc_nofmspath_20260630_125846
  returncode=255
  RAS::PCI::BusFence code 0xa35e
```

The current scatter PR removes the realized scatter/permutation class.  The
remaining rows are:

| class | count | status |
|---|---:|---|
| scatter/permutation | 14 | realized and value-correct in guarded run |
| weight prelayout/restickify | 4 | intentionally out of scope; offline preload work |
| computed activation layout restickify | 1 | still HBM because it feeds a non-primary matmul operand |
| attention matmul operand broadcast/all-gather | 1 | classified; guarded off unless forced; forced path bus-fences |
| gather/reduce/all-reduce | 0 | not observed in this Granite prefill block inventory |

Concrete attention edge:

```text
sdsc_18.json
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
estimated_tensor_bytes = 4194304
guarded: realized=false, Tensor1_hbm
forced staged: realized=true, Tensor1_lx, runtime unsafe
```

The next implementation should therefore stop treating this as a resident
all-gather.  It should extend the dl-dsc/LX relayout contract with a
loop-scoped operand movement primitive that the matmul lowering can consume
inside the matmul schedule, rather than as pre-compute whole-operand setup.

## 2026-06-30 Latest CLC And Paired IFN Status

Latest CLC reset after the risky broadcast/all-gather runs:

```text
pod=adnan-clc-spyre-dev-pf
IP=10.128.18.230
NODE=p1-worker-43
VFIO=/dev/vfio/25
stale python/dxp/senprog/aiu processes: none
```

Guarded Granite remains the safe success point recorded above:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_clc_20260630_122458
returncode=0
kernel_ms_per_iter=12.0627818
```

Current paired InputFetchNeighbor DXP replay for attention `sdsc_18`
(`matmul_operand_broadcast` / `all_gather_replicate`) is still negative.  Both
paired compact and paired noncompact forms reach DCC, but fail L3LU IBUFF:

| paired IFN form | DXP/DCC result |
|---|---|
| compact | `Max IBUFF(256) Current IBUFF(651)` |
| noncompact | `Max IBUFF(256) Current IBUFF(745)` |

Changing compact IFN coalescing through `1`, `2`, `4`, and `8` did not alter the
compact result.  The current conclusion is unchanged: the attention operand
cannot be solved as whole-operand resident broadcast/all-gather setup; it still
needs a true matmul-loop-scoped movement contract.

## 2026-06-30 Paired IFN And Staged No-Sync Follow-Up

After recreating `adnan-clc-spyre-dev-pf`, the current clean CLC pod state is:

```text
pod=adnan-clc-spyre-dev-pf
IP=10.128.18.230
NODE=p1-worker-43
AIU=/dev/vfio/25
stale benchmark/python/dxp processes: none
```

The restored paired compact InputFetchNeighbor replay for the Granite attention
`sdsc_18` all-gather/replicate edge generated full 32-producer coverage, not
the earlier diagnostic stage-only artifact:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_paired_ifn_affine_restored_clc_20260630_142351
rc=134
Max IBUFF(256) Current IBUFF(651)
dataops/18_batchmatmul-dataop-0.json
coreIdsUsed_count=32
coreIDForRing_unique=[0..31]
coreIDForRing_list_count=16384
list_len_hist={1: 15872, 31: 512}
```

`DXP_LX_RELAYOUT_IFN_AFFINE_DTKEYS=1` did not reduce the problematic L3LU
shape.  The dumped PCFG still has 8192 instances each of
`coreIDForRingCondAndVal`, `GTRAndBurstCondAndVal`, `destStartCondAndVal`, and
`bigStAddrOffsets`.  The IBUFF pressure is therefore not just destination-base
condition encoding; it is the per-producer ring/GTR conditional expansion for
whole-operand all-gather.

Two additional DXP-only controls were run:

| run | result | conclusion |
|---|---|---|
| `dxp_paired_ifn_affine_unicast_forced_clc_20260630_143338` | `rc=134`, no IBUFF, `coreids_ring.size() == 1` | forcing unicast removes multicast-group emission but grouped multi-destination pieces are not legal unicast transfers |
| `dxp_paired_ifn_noncompact_unicast_forced_clc_20260630_143411` | `rc=134`, no IBUFF, `coreids_ring.size() == 1` | noncompact does not make this paired IFN unicast shape legal |

The staged non-paired path was then tested with schedule-level after-sync
disabled:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_nonpaired_stage32_nostagedsync_clc_20260630_143615
DXP_LX_RELAYOUT_BROADCAST_STAGE_CORES=32
DXP_LX_RELAYOUT_BROADCAST_COMPACT=1
DXP_LX_RELAYOUT_BROADCAST_NO_STAGED_SYNC=1
rc=0
IBUFF=none
```

This proves DXP can still compile the staged data-op bundle without the broad
schedule-level staged sync.  However, the corresponding full Granite hardware
run still bus-fenced:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_stage32_nostagedsync_dev_optfirst_20260630_144557
returncode=255
RAS::PCI::BusFence code 0xa35e
```

This rules out the schedule-level `after_sync` as the primary runtime failure
cause.  The staged whole-operand all-gather shape itself is not runtime-safe for
the Granite attention operand, even when the DXP gate passes.

Runtime note: after CLC pod recreation, the isolated branch `_C.so` and the
installed comms library could not be loaded together with a single obvious flex
runtime.  `/home/adnan/dt-inductor/sentient/runtime/lib/libflex.so` satisfies
`RuntimeEntry::toPriority`, while `/opt/ibm/spyre/runtime/lib/libflex.so`
satisfies comms-side `RuntimeContext::getDefaultStream` on CLC.  On dev,
`/opt/ibm/spyre/runtime/lib` first allowed the branch to import and reach the
hardware fence.  Record the runtime path in every run artifact; otherwise
import failures can be mistaken for relayout failures.

Current architectural conclusion:

```text
scatter/permutation: works and is value-correct
resident whole-operand attention all-gather: DXP/runtimes are not viable
next required primitive: matmul-loop-scoped operand movement, tied to the
consumer matmul schedule/tile, not precomputed as a full resident operand
```

## 2026-06-30 Frontend Contract Refinement

Torch now has an explicit experimental realization mode for non-scatter
collectives:

```bash
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVE_REALIZATION=loop_scoped
```

The default remains `resident`, preserving the existing conservative behavior.
When collectives are enabled and this mode is `loop_scoped`, the frontend can
mark a `matmul_operand_broadcast` edge as realized without claiming that the
full replicated operand should be resident in LX.  The emitted plan records:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
realization_strategy = loop_scoped_input_fetch
```

This separates the logical contract from the unsafe prototype that forced the
resident byte cap high.  Deeptools still owns the physical lowering and runtime
safety of the movement; this frontend change only makes the handoff precise.

Focused validation on the recreated CLC pod:

```text
pod=adnan-clc-spyre-dev-pf
IP=10.128.18.231
AIU=/dev/vfio/25
command:
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  LD_LIBRARY_PATH=/home/adnan/opt-newer/runtime/lib:/home/adnan/opt-newer/spyre-comms/lib:/home/adnan/opt-newer/deeptools/lib:/home/adnan/opt-newer/senlib/lib:/home/adnan/dt-inductor/build/libaiupti/lib:/home/adnan/dt-inductor/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-} \
  /home/adnan/dt-inductor/.venv/bin/python -m pytest tests/inductor/test_lx_relayout_dldsc.py -q
result:
  14 passed in 4.18s
```

The latest DXP-only metadata control used:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_metadata_control_clc_20260630_150250
```

It replayed the latest staged/no-sync attention bundle, confirmed `sdsc_18`
still carries `matmul_operand_broadcast`, and passed DXP with `rc=0`.  This
does not change the hardware conclusion: the same class still needs a
runtime-safe loop-scoped backend realization before rerunning full Granite with
the collective forced on.

## 2026-06-30 Subpiece-Reuse Diagnostic

After adding a Deeptools diagnostic knob:

```bash
export DXP_LX_RELAYOUT_BROADCAST_SUBPIECE_REUSE=0
```

CLC rebuilt `dxp_standalone` successfully from the current `ah/comms-collectives`
Deeptools tree:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/deeptools/build-dxp-comms-current/dxp/dxp_standalone
```

The DXP-only control run is:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_subpiece_reuse_controls_clc_20260630_151848
bundle:
/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_stage32_nostagedsync_dev_optfirst_20260630_144557/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_g_umxry2
```

Results:

```text
paired_compact_affine rc=134
  Max IBUFF(256) Current IBUFF(651)

paired_compact_affine_unicast_reuse rc=134
  DtException: coreids_ring.size() == 1

paired_compact_affine_unicast_noreuse rc=134
  DtException: coreids_ring.size() == 1

paired_noncompact_affine_unicast_noreuse rc=134
  Max IBUFF(256) Current IBUFF(1481)
  Max IBUFF(256) Current IBUFF(1412)
```

Conclusion: disabling STCDP subpiece reuse does not make the grouped input
neighbor path legal.  The compact unicast variant still fails because the DCC
ring lowering expects a multi-core ring for that path; the noncompact variant
still overflows IBUFF.  This closes off another whole-operand broadcast variant
and strengthens the architectural conclusion: attention needs a true
loop-scoped matmul operand movement lowering, not a precomputed resident or
staged whole-operand all-gather.

## Next Backend Hook: DL-Scheduler LX Neighbor

The best next hook is not another DXP-generated staged `DataOpDsc` row.  The
existing Deeptools DL scheduler already has an LX-neighbor path that places a
dummy `NO_COMPONENT -> LX` transfer marker inside the consumer schedule tree:

```text
dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp
  isLabeledDsLXNeighbor
  createAllocationAndTransfer
  createSynchronization
```

That path is already loop-scoped: the marker is inserted in the innermost
consumer matmul loop before the tile/subchunk compute, and the scheduler adds
the L3LU-to-LXLU soft synchronization around that marker.

The movement payload should still reuse the existing `STCDPOpLx` /
InputFetchNeighbor machinery:

```text
dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp
  createPcfgForInputFetchNeighbor
  createSubPieces

dcg/dcg_fe/pcfg_gen/stcdpOp.cpp
  transformToPcfgSTCDPLxUnrolled
```

So the next implementation shape is:

1. Use Torch's `matmul_operand_broadcast` classification and `read_index` to
   identify the consumer operand.
2. In Deeptools, mark that consumer operand as an LX-neighbor input instead of
   synthesizing whole-operand staged data rows.
3. Let `L3DlOpsScheduler` insert the per-tile transfer marker in the matmul
   schedule.
4. Let the existing IFN/STCDP generator produce the ring movement for that
   tile-scoped marker.

This is closer to the North Star contract:

```text
Torch: classify/cost the communication class and emit tensor-vs-compute
       coordinates.
Deeptools: synthesize legal loop-scoped ring movement and schedule it.
```

It also explains why the staged data-row family kept failing: it tried to solve
a tile-local operand fetch as a full-operand materialization problem.

### Guarded IFN-With-DL Probe

A guarded Deeptools probe tried to make paired input-fetch schedules also run
normal DL PCFG generation:

```bash
export DXP_LX_RELAYOUT_IFN_WITH_DLOP=1
```

Run:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_ifn_with_dlop_controls_clc_20260630_153441
```

Results:

```text
paired_compact_affine_old rc=134
  Max IBUFF(256) Current IBUFF(651)

paired_compact_affine_with_dlop rc=134
  DtException: unit already set for associated schedule step
  dcc/src/Stitcher/ModuleStitcher.cpp line 279
```

This is progress as a diagnostic, not as a fix.  It confirms that simply
generating both the IFN PCFG and the DL PCFG for one paired schedule step is not
enough: DCC currently sees both modules as trying to populate the same
unit/step slot.  The production backend design needs one of:

1. A true single combined IFN+DL module for the paired schedule step.
2. A DCC stitcher contract that allows the IFN movement unit and DL compute unit
   to coexist in the same logical consumer step without colliding.
3. A scheduler-level representation where the LX-neighbor transfer marker is
   emitted as part of the DL module, rather than as an independently stitched
   data module.

The third option is still the cleanest direction.

A second marker-only control tried to use the scheduled data-op only as the
`isLabeledDsLXNeighbor()` marker and skip independent IFN PCFG generation:

```bash
export DXP_LX_RELAYOUT_IFN_WITH_DLOP=1
export DXP_LX_RELAYOUT_IFN_DLOP_MARKER_ONLY=1
```

Run:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_ifn_dlop_marker_only_clc_20260630_153749
```

Result:

```text
rc=134
std::out_of_range: vector::_M_range_check: __n (which is 0) >= this->size()
```

That control fails because the DL scheduler path expects IFN metadata that is
normally populated by `generatePcfgIRForDataOpInpFetch`.  Therefore the next
real patch cannot simply skip IFN generation.  It needs to preserve the IFN
metadata construction while avoiding an independently stitched IFN module for
the same schedule step.

### Metadata-Only IFN + DCC Skip Probe

The next guarded probe split IFN metadata generation from independent IFN
module stitching:

```bash
export DXP_LX_RELAYOUT_IFN_WITH_DLOP=1
export DXP_LX_RELAYOUT_IFN_METADATA_ONLY=1
```

Changes tested:

```text
dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp
  build IFN subpieces/traffic/GTR metadata, but skip createPcfgsSTCDPOp

dcc/src/Conversion/PCFGToDataflowIR/PCFGToDFManager.cpp
  map metadata-only IFN dataops to the DL module instead of converting an
  empty data-op PCFG
```

This moved past the earlier DCC failures.  The new failure is program
verification:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_ifn_metadata_dccskip_clc_20260630_154401
rc=134
LX_MODLRFIMM :: lrfimm:-4161536 src0:0
```

Trying to clear the consumer allocation node failed earlier in DDC:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_ifn_metadata_dccskip_allocreset_clc_20260630_154546
rc=134
DtException: allocNode, dsc/dsc2.cpp line 3999
```

Rewriting the consumer allocation start address to a local placeholder kept DDC
alive but did not fix program verification:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_ifn_metadata_dccskip_alloczero_clc_20260630_154802
rc=134
LX_MODLRFIMM :: lrfimm:-4161536 src0:0
```

Interpretation: the metadata-only direction is the right control point, but the
current IFN/LX-neighbor scheduler still treats the attention operand as a full
4 MB logical LX allocation when computing DL LRF immediates.  The remaining
backend gap is chunk-local address folding for the neighbor operand:

```text
build IFN metadata
skip standalone IFN module stitching
mark the consumer operand as LX-neighbor
allocate/fold only the current matmul chunk in LX for DL address generation
```

Without that final chunk-local address model, DXP can avoid the old IBUFF and
stitching failures but still emits an out-of-range LX immediate.

### 2026-06-30 Communication-Class Checkpoint

The current branch now treats LX relayout opportunities as communication
classes rather than one generic "relayout" bucket:

| class | meaning | PR1 status | Granite status |
|---|---|---|---|
| `scatter` | each consumer slice is sourced from one producer slice, with no fan-in or fan-out | supported by the DLDSC LX relayout PR path | this is the class that produced the measured Granite block speedup |
| `broadcast` / `multicast` | one producer slice feeds multiple consumer slices | classified only unless routed through a specific backend collective | not the dominant remaining Granite blocker in the current attention traces |
| `gather` | one consumer slice is assembled from multiple producer slices | not supported by PR1 | attention `sdsc_10 Tensor1` falls here |
| `all_gather` | every consumer needs all producer slices | partially prototyped through loop-scoped IFN/STCDP | attention operand broadcast experiments hit DCC/runtime gaps |
| `reduce` / `all_reduce` | fan-in plus arithmetic reduction | not a pure byte movement relayout | future collective work, not PR1 |

The most important new concrete case is the attention value-side sub-stick
gather:

```text
consumer: sdsc_10 / 10_batchmatmul
input: Tensor1 LX
producer distribution: 32 producer out fragments
consumer compute split: {x:16, mb:1, out:2, in:1}
stick dim: out
stick size: 64
fragment width: out=16
communication_class: gather
```

Each consumer out-half needs a 256-wide `out` region assembled from sixteen
16-wide producer fragments.  That is not the same class as PR1's whole-stick
scatter/permutation case.  The direct DLDSC/STCDP path currently fails because
`STCDPOpLx` assumes each transfer fragment is at least one full stick along the
stick dimension:

```text
stcdpOp.cpp
DT_CHECK(inpSP.dimToSize["out"] >= stickDim)

observed: inpSP.out = 16, stickDim = 64
```

Relaxing that assertion is not a valid fix.  The downstream ring transfer path
does not carry the missing intra-stick metadata:

```text
transfer dim
source intra-stick offset
destination intra-stick offset
element count
```

`setPlacementInfoSubPiece()` also rounds stick-coordinate differences with
`ceil(diff / stickDim)`, which loses the needed `diff % stickDim` offset.  Once
the assertion is bypassed, later lowering either fails coverage/addressing or
emits invalid codegen.  The bounded transfer count shows this is not a
combinatorial explosion; it is an underspecified partial-stick movement.

Two viable research routes remain:

1. **DLDSC/STCDP staged route:** move legal whole sticks over the ring into
   staging LX, then use a local partial-stick LX operation to extract/assemble
   the consumer chunk.  This keeps the compact DLDSC coordinate contract but
   requires a real local assembly step and chunk-local address folding.
2. **Explicit byte-range route:** describe the exact sub-stick source/dest byte
   ranges in the frontend.  A prototype metadata builder represents the
   `out=16` FP16 fragment as a 32-byte movement instead of a 128-byte stick.
   This is the clearest diagnostic carrier for the sub-stick gather, but it can
   grow large for broad collectives and still needs backend range-aware
   scheduling/lowering.

The near-term implementation question is therefore not "can we classify the
edge?"  We can.  The remaining backend question is whether production wants to
realize `gather`/`all_gather` by extending compact DLDSC/STCDP staging, or by
accepting an explicit byte-range carrier for the cases where STCDP's inferred
whole-stick model is too coarse.

### 2026-07-01 Implementation Lane Checkpoint

Three parallel probes clarified the next boundary.

#### DLDSC/STCDP Internal Range Path

CDX workspace:

```text
/home/adnan-cdx/codex-isolated/comms_collectives_staged_substick_agent_20260630_225637/deeptools
```

The prototype added internal sub-stick metadata:

```text
SubStickRangeInfo
transferInfo::subStickRanges
STCDPOpLx::enableSubStickRanges
```

`insertSubPieces()` now records:

```text
transfer dim
source intra-stick offset
destination intra-stick offset
element count
producer memId
consumer memId
```

Replay moved past the old front-end STCDP assertion:

```text
old failure: inpSP.dimToSize["out"] >= stickDim
observed:    16 >= 64
```

The new boundary is lower in the backend:

```text
DXP_STCDP_SUBSTICK_RANGES reached STCDP ring-DT lowering for core=0
unit=10 dtKey=0 dim=out srcOffset=0 dstOffset=0 count=16 producer=0
consumer=0.

SenPcfgRingDtNode/DcgBE ring lowering only carries stick-addressed
start/stride values and has no src/dst intra-stick offset fields.
```

Artifact:

```text
/home/adnan-cdx/codex-isolated/comms_collectives_staged_substick_agent_20260630_225637/runs/staged_substick_ranges_20260701_005730
```

Interpretation: compact DLDSC/STCDP remains a viable architecture, but the
next required backend work is deeper than `dtTable_` metadata.  Either
`SenPcfgRingDtNode` plus `DcgBE::pcfgringDTToInstr()` need explicit
intra-stick source/destination offsets, or Deeptools needs a lower-level ring
sequence that moves partial-stick ranges without rounding LX addresses to
`addr / bytesPerStick`.

Follow-up on the same CDX workspace checked the direct ring-offset option.  It
is not a small backend-only extension in the current model:

```text
dsc/pcfg.h
  SenPcfgRingDtNode has stick-addressed start/stride/ring metadata.

dcg/dcg_be/dcgbeCodegen.cpp
  pcfg ring codegen divides LX start address by bytesPerStick before LAR init.

dsc/isa.cpp
  L3 ring opcode type 21 carries src0/LAR, src1/LBR or drm, node, burst,
  group, be.  It has no source intra-stick offset, destination intra-stick
  offset, or sub-stick element-count field.
```

Replay artifact:

```text
/home/adnan-cdx/codex-isolated/comms_collectives_staged_substick_agent_20260630_225637/runs/staged_substick_ring_offsets_20260701_011119
```

Conclusion: direct partial-stick ring movement needs deeper ISA/ProgIR/codegen
support.  For the compact DLDSC direction, the practical next architecture is
whole-stick ring staging followed by local LX partial-stick assemble/extract.

#### Explicit Byte-Range Path

CLC workspace:

```text
/home/adnan/codex-isolated/explicit_range_agent_20260630/deeptools
```

The explicit byte-range prototype now lowers a small list of ranges, not just
one range.  The four-range replay represents one full 128-byte consumer stick
assembled from four 16-wide FP16 producer fragments:

```text
parsedRangeCount: 4
range[0] bytesPerMove=32 srcCore=3 dstCore=19 dstLx=36864
range[1] bytesPerMove=32 srcCore=4 dstCore=19 dstLx=36896
range[2] bytesPerMove=32 srcCore=5 dstCore=19 dstLx=36928
range[3] bytesPerMove=32 srcCore=6 dstCore=19 dstLx=36960
dtTableCount: 4
coreIDtoDtKey_L3SU: cores 3,4,5,6 each own one dt key
coreIDtoDtKey_L3LU: core 19 owns dt keys 0 1 2 3
```

Replay:

```text
/home/adnan/codex-isolated/explicit_range_agent_20260630/runs/explicit_range_four_replay_20260701_005411
rc=0
```

Runtime-facing validation is still blocked:

```text
dxp_standalone --bundle ... -b senulator
rc=134
DtException: minStartAddr % dscGlobal.sysDef.bytesPerStick == 0
file dsc/senulatorProg.cpp line 324
```

Interpretation: explicit ranges are the fastest way to describe and DXP-lower
the sub-stick gather.  The cost is a larger frontend/backend contract: physical
movement ranges, byte addresses, producer/consumer cores, synthetic LDS
metadata, and schedule rows must all be enumerated.  That is useful for a
research carrier but riskier as the long-term collective contract.

Follow-up on the same CLC workspace moved the explicit byte-range prototype
through senulator/backend acceptance for the four-range synthetic bundle:

```bash
DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO=1
DXP_ENABLE_COMPILE_TIME_CORRECTION=1
dxp_standalone --bundle -d .../bundle_input -b senulator
```

Artifact:

```text
/home/adnan/codex-isolated/explicit_range_agent_20260630/runs/explicit_range_four_senresolve_ctc_20260701_012808
```

Result:

```text
result_senulator_backend.txt: rc=0
semantic coverage: [(0,32), (32,64), (64,96), (96,128)]
```

This proves backend/senulator codegen acceptance for the intended four
explicit byte ranges.  It does not yet prove patterned runtime memory
correctness.  The prototype now touches JSON import, transfer materialization,
generic verifier expectations, DXP bundle-symbol handling, senulator metadata
export, and compile-time correction control, which reinforces the earlier
scaling concern.

#### 4-Head Attention Script Probe

dev-pf workspace:

```text
/home/adnan/codex-isolated/attention_4h_probe_20260701_005035
```

Script commit:

```text
git@github.ibm.com:aviros/test-spyre-scripts.git
05deb9702654f73781b457ed052a3ff69316670f
```

Runs:

| lane | result |
|---|---|
| baseline current comms | fails before scratchpad/SDSC |
| scatter planner + Deeptools | same failure before scratchpad/SDSC |
| archived scatter baseline fallback | same failure before scratchpad/SDSC |

Failure:

```text
torch._inductor.exc.InductorError:
NotImplementedError: buf10 (Pointwise): no mechanism to resolve stick incompatibility
```

No scratchpad allocator plan, `plan_solver` summary, or `sdsc_*.json` files
were produced.  Therefore this run did not reproduce the pasted baseline that
showed allocator/SDSC behavior.  The scatter planner cannot affect this failure
yet because the compile stops before scratchpad planning and SDSC generation.

Follow-up in the same dev-pf workspace identified the issue:

```text
buf10 = running_max = torch.maximum(real_max, block_max)
source: test_flash_4_head.py:117

buf4 = real_max.amax(dim=-1)
buf9 = torch.amax(scores, dim=-1)
```

Current main fails because it lacks singleton-stick reduction restickify:

```text
buf4 STL ...: No mechanism to gather elements from multiple sticks into single stick
```

Local history shows the support existed in:

```text
0e928a8 Support singleton-stick reduction restickify
```

and was later reverted by:

```text
59b1086
```

Sibling scripts avoid the same issue with a script-level workaround:

```python
scores.transpose(-1, -2).contiguous()  # avoid stick reduction
```

Using an isolated runtime monkey patch that restored only the singleton-stick
branch, both baseline and scatter-planner lanes got past `buf10`, reached
scratchpad/SDSC, emitted 89 SDSC JSON files, and then failed later in DXP:

```text
DtException: Could not find any suitable dimension mapping
```

Baseline with singleton patch:

```text
/home/adnan/codex-isolated/attention_4h_probe_20260701_005035/runs/baseline_singleton_restickify_patch_20260701_011409
scratchpad limit: 1638 KB
SDSC files: 89
allocations: 154 hbm, 72 lx
ReStickifyOpHBM count: 20
```

Scatter planner with singleton patch:

```text
/home/adnan/codex-isolated/attention_4h_probe_20260701_005035/runs/scatter_singleton_restickify_patch_20260701_011645
scratchpad limit: 2048 KB
planned LX relayout edges: 8
SDSC files: 89
allocations: 120 hbm, 106 lx
ReStickifyOpHBM count: 20
STCDPOpLx count: 0
```

Realized scatter edges:

```text
buf4 -> buf10
buf4 -> buf11
buf5 -> buf7
buf1 -> buf13
buf12 -> buf13
buf10 -> buf14
buf20 -> buf21
```

Still not handled by scatter:

```text
buf27 -> buf7
kind: layout_restickify_activation
realized: false
reason: computed activation restickify needs loop-scoped matmul operand lowering
```

Interpretation: the scatter planner does change attention allocation behavior
once singleton-stick restickify lets compilation reach scratchpad/SDSC.  It
does not remove the remaining layout-restickify activation spill, and execution
is still blocked by a later DXP dimension-mapping failure.
