# Coordinate-Only SHUFFLE Status

## Two Separate Claims

1. The grouped K-side all-gather is feasible using LX ring movement.
2. Current Deeptools can derive and materialize it from DLDSC coordinates alone.

The evidence proves the first claim. For the second claim, the answer depends
on what "current Deeptools" means:

- the unmodified `704c19f8fb` backend does not derive the movement;
- an isolated 141-line bounded-SHUFFLE materializer derives and lowers the
  exact movement from the same Variant A coordinates and explicit endpoints.

It does **not** prove that coordinates are fundamentally inexpressive. The
logical distribution can be represented as redundant residency. The observed
gap is between that representation and current physical materialization.

## Target Movement

For each of four attention heads:

```text
8 source cores x one distinct 128 KiB K shard
    -> grouped all-gather
8 destination cores x one complete 1 MiB K operand
```

At shard granularity this is:

```text
4 groups x 8 source shards x 8 destinations = 256 placements
32 local placements + 224 remote placements
```

## Experiments

| Experiment | Representation | Current result | Conclusion |
|---|---|---|---|
| Variant A | Redundant output coordinates assign the complete-head result to eight cores | Scheduler rejects `0_shuffle` | Current SHUFFLE mapping does not accept this expanding replicated distribution |
| Variant B | Nonphysical replication dimension with input cardinality 1 and output cardinality 8 | Same scheduler rejection | Making replication explicit does not activate an all-gather path |
| Bounded SHUFFLE patch | Recognize the expansion and emit bounded transfer rows | Reaches DCG, then fails output subpiece coverage | Physical coverage validation also assumes unsupported output semantics |
| Exact bounded SHUFFLE patch | Preserve independent S1/S2 layouts and emit one bounded row per source shard | DXP/DCG/DCC pass; 256 placements, no HBM or restickify | The coordinate contract is structurally sufficient once the missing materializer is implemented |
| Explicit DataDSC control | Manually enumerate eight bounded `STCDPOpLx` rows | Structurally lowers all 256 placements with no HBM | Existing ring primitives can carry the required traffic |
| Custom staged materializer | `ReStickifyOpLx` plus grouped `STCDPOpLx` placement | Value-correct and removes the K HBM handoff | The operation is implementable when materialization is supplied explicitly |

The clean Variant A and Variant B replay failure is:

```text
Scheduler failed to find a suitable op mapping for sdsc: 0_shuffle
```

The exact backend reference is:

```text
Deeptools 704c19f8fb7f0cc972f20404f9dd0010895a35e2
```

## Backend Delta Proven By The Isolated Patch

The exact-backend experiment demonstrates the minimum structural behavior that
stock Deeptools lacks:

1. **Replicated SHUFFLE materialization**
   Derive the grouped one-to-many placements from redundant source/output
   coordinate residency.
2. **Explicit expanded destination consumption**
   Use the frontend-provided S2 address and 1 MiB/core extent rather than hidden
   dynamic allocation.
3. **Independent source and destination physical layouts**
   Support the compact S1 stride and complete-K S2 stride directly, or insert a
   well-defined local `ReStickifyOpLx` stage.
4. **Coverage and synchronization semantics**
   Accept replicated output coverage and ensure all placements finish before
   the score BMM consumes S2.
5. **Fail-closed capacity behavior**
   Preserve the HBM path when S1, S2, and other live tensors cannot fit without
   overlap.

The patch proves items 1, 2, 3, and structural ordering/coverage for the
isolated fixture. Full-workload S2 lifetime validation, capacity fallback,
patterned AIU correctness, and performance are still required before this is a
production result.

## Current Production Interpretation

The strongest defensible statement is now:

> DLDSC coordinates can describe the desired source and consumer distributions,
> and an isolated backend patch successfully derives and lowers the grouped
> all-gather from those coordinates. Stock Deeptools still lacks that bounded,
> replication-aware materializer, while production integration additionally
> requires a tracked, nonoverlapping S2 lifetime and fail-closed capacity
> behavior.

Do not summarize this result as either "coordinates cannot express all-gather"
or "coordinates already make all-gather work." Neither statement matches the
evidence.
