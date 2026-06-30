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
3. Locate the producer SuperDSC that owns the LX-resident operand.
4. Call the existing InputFetchNeighbor generation path with
   `(consumer_sdsс, producer_sdsc)`.
5. Preserve the resulting `STCDPOpLx` schedule as a loop-scoped input movement
   associated with the consumer batchmatmul.
6. Reject, rather than silently HBM-fallback, if the producer/consumer metadata
   is insufficient for this path.

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
