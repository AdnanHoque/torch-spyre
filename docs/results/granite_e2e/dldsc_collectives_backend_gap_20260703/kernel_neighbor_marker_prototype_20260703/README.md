# KERNEL-neighbor marker prototype checkpoint

This artifact records the first env-gated prototype that avoids the unsafe
resident `STCDPOpLx` materialization path for the Granite attention RHS
`matmul_operand_broadcast` collective.

## Prototype env

```text
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1
DXP_LX_FRAC_AVAIL=1
```

## What changed

The prototype:

1. Keeps the grouped collective plan from DLDSC metadata.
2. Marks the KERNEL operand as an LX-neighbor operand from
   `lxRelayoutClassifications_` instead of requiring a scheduled resident
   DataDSC row.
3. Clears the producer allocation for that consumer KERNEL LDS so the scheduler
   can create a loop-scoped consumer allocation.
4. Registers the scheduler-created LX-neighbor allocation with memory allocation
   metadata.
5. Allows layout discovery from `primaryDsInfo_` when the allocation node is
   intentionally absent before scheduler allocation.

## Result

The failure moved past the old resident-STCDP sub-stick check:

```text
stcdpOp.cpp line 4342: input piece must cover stick dimension
```

and also past the missing-allocation-node crash. The current failure is:

```text
DtException: Memory allocation must be valid to commit.
... L3DlOpsScheduler.cpp line 1433
```

## Interpretation

This is the expected next backend gap. The current scheduler-created KERNEL
allocation is still too resident-like. For `8_batchmatmul`, the RHS KERNEL core
stage is large (`mb=32, out=256, in=128` for the consumer slice), so simply
creating an LX allocation for the consumer KERNEL operand does not fit.

The production route needs true loop-scoped KERNEL chunk assembly: the backend
should fetch/assemble only the current KERNEL chunk from grouped producer shards
and bind that chunk into the existing matmul KERNEL load path. It should not
materialize the full per-core RHS in LX.

## Plan JSON

The grouped collective plan remains correct:

- `group_count = 2`
- `producer_chunks_per_group = 16`
- `consumer_replicas_per_group = 16`
- `logical_transfer_count = 512`

## Next backend work

1. Carry grouped source-shard metadata onto the KERNEL-neighbor transfer marker.
2. Size the KERNEL-neighbor allocation from the active matmul transfer/chunk loop,
   not the full consumer KERNEL data stage.
3. Lower the `NO_COMPONENT -> LX` KERNEL-neighbor marker into real L3 ring reads
   from producer LX shards.
4. Preserve mixed HBM+KERNEL-neighbor scheduling without the diagnostic guard.
