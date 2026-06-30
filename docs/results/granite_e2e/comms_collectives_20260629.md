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

## Next Implementation Step

The next prototype should target the computed activation layout-restickify HBM
row first:

1. Ignore graph-input/parameter weight restickifies; those belong to offline
   weight prelayout/preload work.
2. Identify the computed activation restickify whose input and output can both
   be LX resident.
3. Preserve the original producer tensor layout and the consumer-required target
   layout.
4. Emit a dl-dsc contract that exposes an LX input with explicit producer
   coordinates and a post-layout consumer requirement, without materializing an
   HBM `ReStickifyOpHBM` node.
5. Extend Deeptools relayout insertion only if needed to support different
   pre-layout and post-layout forms for `STCDPOpLx`.
6. After layout-restickify is working, return to the attention value-side
   `matmul_operand_broadcast` edge and lower it as loop-scoped movement rather
   than post-relayout full materialization.
