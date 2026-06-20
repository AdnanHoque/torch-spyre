# LX Coordinate Remap PR Scope

This note records the intended production split for LX coordinate remap.  The
goal of the first PR is to package the value-correct pass we have today, prove
it gives an end-to-end kernel-time win, and leave broader communication classes
as explicit follow-up work.

## PR 1: Exact Resharding

PR 1 covers exact, non-reducing, non-replicating movement between different
`PerCoreView`s when the same logical 128-byte stick already exists in a
producer LX region and the consumer wants that stick on a different core.

Supported communication classes:

- Same-view persistence remains owned by main's `LX_PLANNER`.
- One-to-one cross-core coordinate remap.
- Disjoint scatter where one producer-owned region is split across unique
  consumer owners.
- Disjoint gather/concat where multiple producer owners provide unique logical
  pieces to one consumer owner.

Unsupported communication classes:

- Fan-out or multicast, where the same logical stick must be copied to multiple
  consumer cores.
- Reductions, including split-K partial accumulation.
- Layout-changing restickify or stick-transpose movement.
- Streaming/tiled producer-consumer handoff for regions that do not fit in LX.
- Weight preload or weight restickify removal.

The key PR 1 artifact is the fused FMS SwiGLU prefill run.  It proves the
projection output can stay in LX, move through `LXCoordinateRemapOp`, and feed
the pointwise SwiGLU chain without reloading the projection halves from HBM.

## Follow-Ups

PR 2 should target read-only activation fan-out for the final
`mul -> down_projection` edge.  In matmul terms, this is operand multicast:
multiple output-column core groups need the same activation tile.  This is not
a reduction, but it is a different communication class than PR 1 because the
destination ownership is intentionally duplicated.

PR 3 should target streaming or tiled activation handoff only if PR 2 cannot
fit the activation tile with an acceptable matmul split.  Streaming is a
scheduling feature, not just a larger coordinate-remap movement list.

Weight `ReStickifyOpHBM` removal is intentionally out of scope for this pass.
That work belongs to model preload/layout handling and should not block PR 1.

Warp specialization should start only after the movement baseline is stable and
the benchmark artifacts show the remaining bottleneck is scheduling/resource
overlap rather than avoidable HBM traffic.

## PR 1 Acceptance

- Unit tests and DXP gates remain green.
- FMS fused SwiGLU prefill keeps the archived kernel-time win using
  trace-derived `kernel_ms_per_iter`.
- Jamie-style SDSC tables show the projection-half HBM round trips removed.
- `onchip_move.jsonl` records planned edges and explicit fallback reasons.
- A Granite-block fake-weight benchmark is run to determine whether exact
  resharding applies beyond isolated SwiGLU, especially in MLP and attention
  subgraphs.
- Published conclusions keep weight restickifies and fan-out as residual,
  out-of-scope costs.
