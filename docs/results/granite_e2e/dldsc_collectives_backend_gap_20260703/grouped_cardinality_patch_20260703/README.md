# Grouped matmul operand broadcast cardinality patch

This artifact records the first Granite attention RHS collective replay after
backend-derived grouping was added for `matmul_operand_broadcast` plans.

## Source state

- Torch branch: `ah/comms-collectives`
- Deeptools branch: `ah/comms-collectives`
- Replay bundle: Granite block attention SDSC bundle from the July 3 fail-closed run.

## Result

The backend plan now derives the grouped collective from tensor distribution
metadata instead of falling back to one global all-to-all group.

For `8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`:

- `group_count = 2`
- `producer_chunks_per_group = 16`
- `consumer_replicas_per_group = 16`
- `replication_factor = 16`
- `logical_transfer_count = 512`

This replaces the previous global expansion shape of 1024 logical transfers.
The two groups map producer cores `0..15` to consumer cores `0..15`, and
producer cores `16..31` to consumer cores `16..31`.

## Remaining backend gap

Replay still fails at:

```text
DtException: op->inpSP_.at(inpSPIdx).dimToSize_.at(dimNameOuter) >= stickDim,
file .../dcg/dcg_fe/pcfg_gen/stcdpOp.cpp line 4342
```

That failure is expected for this diagnostic path. The collective is a grouped
sub-stick KERNEL operand assembly: producer chunks are narrower than the target
KERNEL stick. Resident `STCDPOpLx` materialization requires each input piece to
already cover the stick dimension, so it cannot implement this class directly.

The next implementation step is the loop-scoped KERNEL LX-neighbor route: fill a
KERNEL LX chunk for the matmul operand inside the matmul schedule and let the
existing matmul KERNEL load path bind it, instead of materializing a resident
replicated tensor before the DL op.

## Verification

Focused Deeptools unit test:

```text
./util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
22 tests passed
```

Full DXP standalone rebuild also passed before replay.
