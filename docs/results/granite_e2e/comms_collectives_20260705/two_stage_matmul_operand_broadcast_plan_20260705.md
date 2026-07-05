# Two-Stage Matmul Operand Broadcast Plan - 2026-07-05

## Why This Exists

The Granite attention RHS spill is not plain same-layout broadcast.

It is:

```text
activation LX layout
  -> all-gather / broadcast across cores
  -> local layout conversion into PT KERNEL operand layout
  -> batchmatmul consumes KERNEL operand
```

The direct fused prototype emits ring transfers, but writes directly into the KERNEL operand view and is value-wrong. The full resident staged prototype is value-correct on small synthetic cases, but materializes the full RHS per core and exceeds LX capacity for Granite.

The production-shaped path should keep the value-correct decomposition and make it loop/tile-scoped.

## Required Schedule Shape

For each matmul operand tile:

```text
allocate source-layout staging LX
allocate final KERNEL operand LX
ring all-gather cross-core source-layout chunks into staging slots
local copy same-core chunks into staging slots if restickify requires uniform input
sync ring/local movement complete
local restickify/layout-convert staging -> final KERNEL operand view
sync final KERNEL view complete
matmul consumes final KERNEL operand view
```

Same-core copies must not use ring. A diagnostic self-ring experiment caused a PCI bus fence.

## Deeptools Work

1. Extend the relayout plan metadata with an explicit strategy:

```text
realization_strategy = loop_scoped_gather_restickify
```

The plan must expose:

- source LX tensor layout and stick shape
- target KERNEL tensor layout and stick shape
- per-consumer tile/chunk shape
- producer chunk count per gather group
- source-core to destination-core movement plan

2. Keep DXP relayout insertion metadata-only for this path.

Do not insert full resident data-op SDSCs for Granite attention. Do not allocate full gather/final tensors with `MemTrackBundle::checkAndAddDs`.

3. In `L3DlOpsScheduler.cpp`, add a dedicated matmul operand broadcast path.

Do not route this through the generic INPUT-neighbor path. The current scheduler assumes LX-neighbor tensors are ordinary inputs, while this case is a KERNEL operand with layout conversion.

4. Add an auxiliary loop-scoped staging allocation.

The staging buffer is sized from the active tile, not from full tensor `lxSize_`:

```text
stage_bytes =
    producer_chunks_for_this_consumer_tile
  * bytes_per_source_layout_tile_chunk
```

The staging layout must match the source activation tensor layout.

5. Ring movement only handles cross-core pieces.

The existing `lxNeighborRingTransfers_` side-table is a reasonable prototype carrier for cross-core transfers, but `sourceCoreId == destinationCoreId` must be handled by local LX copy or by direct source reads in restickify.

6. Run local layout conversion into the final KERNEL view.

Preferred carrier:

- `ReStickifyOpLx` or `ReStickifyOpWithPTLx` semantics
- invoked as a schedule-local stage, not as full resident mixed data-op materialization

Fallback if restickify requires a contiguous staged input:

- copy same-core chunks into their staging slots first
- restickify uniformly from staging

## Torch Work

Torch should continue to describe the logical contract:

- producer tensor distribution
- consumer compute distribution
- source and target layout metadata
- communication classification: `matmul_operand_broadcast`

Torch should not emit physical ring schedules for this path.

## Risks

- Hidden fake `LabeledDsInfo` for staging may leak into normal input-neighbor/unit-view logic.
- A new explicit auxiliary allocation descriptor is cleaner but touches more scheduler code.
- Restickify is currently easiest as a data-op path; making it schedule-local may need a wrapper node/type.
- Accidentally sizing staging from full tensor shape recreates the Granite capacity failure.
- Sync has two required boundaries: ring/local movement before restickify, and restickify before PT consume.

## Acceptance

1. Synthetic `M16/M32/M64` matmul operand broadcast is value-correct.
2. Same-core pieces are copied locally or read directly, never sent through ring.
3. Granite attention RHS relayout removes the HBM handoff without full RHS materialization.
4. Kernel timing is compared against the current scatter-only PR1 baseline and the disabled baseline.

