# DLDSC LX Collective Class Status - 2026-07-05

## Summary

DLDSC coordinate metadata is expressive enough to describe ownership changes, fanout, and fanin. The current implementation status is not uniform across all collective classes, because some classes are pure movement while others require arithmetic or layout/form conversion.

The key distinction:

- Same-layout LX movement is mostly a movement and allocation problem.
- Matmul operand broadcast is movement plus layout conversion into a PT `KERNEL` operand view.
- Reductions are arithmetic collectives, not pure relayout.

## Status Table

| Class | DLDSC representation | Evidence today | Current status | Main remaining gap |
|---|---|---|---|---|
| Scatter | Producer and consumer cover same tensor with different unique destination ownership. | PR1 scatter path and Granite attention/SwiGLU experiments. | Works for the covered same-layout case. | Generalize coverage, keep capacity guards. |
| Broadcast / multicast | One producer shard maps to multiple consumer cores for the same logical tensor/layout. | Same-layout 4-way RHS broadcast/all-gather synthetic runs pass. | Partially working for same-layout fanout. | Add a dedicated multicast test and productionize metadata/allocator contract. |
| Gather | Multiple producer shards map to one consumer core for the same logical tensor/layout. | Synthetic gather passes when destination base is frontend-safe; dynamic backend destination can corrupt. | Mechanically possible, not production-safe yet. | Destination allocation ownership and non-overlap contract. |
| All-gather | Every consumer receives all producer shards. | Same-layout small RHS all-gather passes; wider source-core case has metadata/fold gaps. | Partial for same-layout. | Source address metadata and fold/cardinality handling for wider producer groups. |
| Matmul operand broadcast | All-gather source activation shards, then convert into PT `KERNEL` operand layout. | Full resident staged path is value-correct synthetically; loop-scoped direct KERNEL ring emits traffic but does not yet prove value correctness. | Structural lowering works; loop-scoped KERNEL conversion remains the backend gap. | Need two-stage loop-scoped source-layout gather plus local KERNEL layout conversion. |
| Form-changing restickify | Coordinates plus source/destination stick/layout metadata. | `ReStickifyOpLx` exists; full staged synthetic path can use it; standalone attention replays hit DDL/op mapping gaps. | Partial. | Backend support for `ReStickifyOpLx` in the target schedule shape, or a streaming equivalent. |
| Reduce | Many producer values combine arithmetically into one output shard. | No positive single-AIU LX collective evidence in this branch. | Not covered by relayout alone. | Requires op, axes, dtype/accumulation, identity, and scheduling of arithmetic fan-in. |
| All-reduce | Reduce plus redistribute/broadcast result. | No positive single-AIU LX collective evidence in this branch. | Not covered by relayout alone. | Build reduce, then broadcast result; likely a separate arithmetic collective feature. |

## Evidence Snapshot

Same-layout broadcast/all-gather:

```text
min_matmul_auto_relayout_nosplit_scale1_backend02_20260703_142134
ALLCLOSE True
MAX_DIFF 0.03125

min_matmul_auto_relayout_nosplit_scale2_backend02_20260703_142049
ALLCLOSE True
MAX_DIFF 0.0625
```

Same-layout gather:

```text
min_lx_gather_srcsplit2_forcedbase_run.log
ALLCLOSE True
MAX_DIFF 0.25

min_lx_gather_srcsplit4_run.log
ALLCLOSE True
MAX_DIFF 0.25
```

Matmul operand broadcast structural probes:

```text
kernel_neighbor_skipviews_M16_001835
DXP emits L3 ring send/recv nodes.
ALLCLOSE False
MISMATCH 3829 / 4096

kernel_neighbor_direct_shift_M16_002402
Direct all-gather still writes the wrong PT KERNEL view.
ALLCLOSE False
MISMATCH 4096 / 4096

kernel_neighbor_direct_selfring_shift_M16_003720
Same-core-as-ring workaround is unsafe.
RAS::PCI::BusFence
```

Treat these as backend layout-conversion probes, not as flash-attention correctness conclusions. The current flash attention value path has an independent zero-stride/broadcast-view lowering issue: `TensorArg` does not preserve `stride_map`, and SDSC generation recomputes dense strides from `device_size`. That can make an `unsqueeze` broadcast dimension incorrectly participate in linear address calculation even when relayout is disabled. Until that baseline bug is fixed elsewhere, flash value correctness is not a clean oracle for this communication track.

Full resident staged matmul operand path:

```text
staged_geometric_offset_M16_231542
ALLCLOSE True
MAX_DIFF 0.001953125
```

That staged result proves the decomposition is correct:

1. Gather source-layout chunks.
2. Restickify/layout-convert locally into the KERNEL operand view.

It is not directly Granite-production-safe because full resident materialization can exceed per-core LX capacity for attention RHS tensors.

## Design Consequence

The next useful implementation step is not another direct address formula for fused ring-to-KERNEL writes. We should preserve the value-correct staged decomposition and make it loop/tile-scoped:

1. Allocate a non-overlapping loop-local staging buffer for source-layout gathered chunks.
2. Use ring movement only for cross-core pieces.
3. Use a real local LX copy for same-core pieces.
4. Run local layout conversion into the PT KERNEL operand view for the current tile.
5. Bind the matmul operand reader to that tile-scoped KERNEL view.

This keeps the frontend/backend contract aligned with the DLDSC model:

- Torch describes tensor distribution and consumer compute distribution.
- Deeptools derives the movement cardinality and synthesizes physical ring/local movement.
- Layout-changing cases carry enough metadata for backend layout conversion, not just core ownership.
