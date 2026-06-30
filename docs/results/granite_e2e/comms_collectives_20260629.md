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

| kernel area | SDSC row | split | current form |
|---|---|---|---|
| attention score path | `sdsc_9` | `{mb:32,x:1,out:1}` | HBM -> HBM |
| attention + MLP fused kernel input | `sdsc_0` | `{mb:32,out:1}` | HBM -> HBM |
| fused SwiGLU / down-proj handoff | `sdsc_10` | `{mb:25,out:1}` | HBM -> HBM |
| residual/downstream linear handoff | `sdsc_0` | `{mb:1,out:25}` | HBM -> HBM |
| final norm / output projection handoff | `sdsc_7` | `{mb:32,out:1}` | HBM -> HBM |

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

This is also not the same class as resident scatter.  The remaining rows are
layout/stick restickifies: the tensor's physical stick/layout form changes, not
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

The next prototype should target explicit layout-restickify HBM rows first:

1. Identify restickify-plan entries whose input and output can both be LX
   resident.
2. Preserve the original producer tensor layout and the consumer-required target
   layout.
3. Emit a dl-dsc contract that exposes an LX input with explicit producer
   coordinates and a post-layout consumer requirement, without materializing an
   HBM `ReStickifyOpHBM` node.
4. Extend Deeptools relayout insertion only if needed to support different
   pre-layout and post-layout forms for `STCDPOpLx`.
5. After layout-restickify is working, return to the attention value-side
   `matmul_operand_broadcast` edge and lower it as loop-scoped movement rather
   than post-relayout full materialization.
