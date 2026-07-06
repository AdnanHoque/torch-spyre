# Staged Matmul Operand Broadcast Lowering Plan - 2026-07-06

This plan records the next implementation slice for value-correct DLDSC all-gather/broadcast-style movement in Granite and flash attention.

The immediate target is not flash numerical correctness. Flash currently has an independent zero-stride/broadcast view lowering issue. The target here is proving that the communication lowering is correct on non-zero-stride cases, then using flash/Granite as structural stress tests.

## Problem Class

The active hard edge is:

```text
producer LX/source-layout tensor
  -> consumer batchmatmul Tensor1 / KERNEL RHS operand
```

Torch metadata classifies this as:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
materialization_pattern = all_gather_replicate_with_layout_conversion
requires_layout_conversion = true
staged_destination.scope = matmul_transfer_loop
```

This is a copy-style collective plus layout conversion. It is not reduce/all-reduce.

## What Is Already Present

Torch already emits the logical contract in DLDSC metadata:

- source and consumer core maps;
- operand/read index;
- source LX tensor metadata;
- target KERNEL tensor metadata;
- layout transform metadata;
- staged destination scope.

Deeptools already has these pieces:

- `util/LayoutAllgatherRestickify.*` can classify and expand logical transfers.
- `dxp/SdscRelayoutInsertion.cpp` consumes LX relayout classifications.
- `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp` can attach `lxNeighborRingTransfers_` to a DL `TransferNode`.
- `dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp` can emit ring send/recv nodes from those transfer records.
- `dcg/dcg_fe/pcfg_gen/restickifyOp.cpp` already lowers `ReStickifyOpLx` data-op DSCs.

The newest Deeptools checkpoint records the missing stage explicitly and fails closed when the direct shortcut would be value-unsafe:

```text
Adnan-Hoque1/deeptools ah/comms-collectives
f23ab8b85 dcg: fail closed for staged matmul operand conversion
```

## Why The Direct Shortcut Is Wrong

The direct KERNEL-neighbor path tries to move producer-layout LX bytes directly into the final matmul KERNEL operand address space. Synthetic row-pattern tests showed that path corrupts the row mapping.

The source layout and target KERNEL layout are different. The ring move cannot silently be both a gather and a restickify.

Correct decomposition:

```text
1. Gather/all-gather producer shards into source-layout LX staging.
2. Locally convert source-layout staging to the consumer KERNEL RHS layout.
3. Let batchmatmul consume the converted KERNEL tile.
```

## Recommended Patch Sequence

### Patch 1: Backend unit fixture for staged semantics

Add a small Deeptools fixture before touching full flash/Granite:

- 2 or 4 producer cores.
- 2 or 4 consumer cores.
- source LX layout intentionally differs from KERNEL RHS layout.
- expected result is row-pattern exact after gather plus local conversion.

Assertions:

- direct KERNEL-neighbor lowering is rejected when `requires_layout_conversion=true`;
- staged plan records a source-layout staging allocation;
- local conversion appears before the DL matmul in the schedule;
- no HBM relayout is inserted for the operand.

Candidate files:

```text
util/test/LayoutAllgatherRestickify_unit_test.cpp
dxp/test/dxp_unittest.cpp
dxp/test/test_core_work_div_incompt/
```

### Patch 2: Add source-layout staging representation

Extend the current `TransferNode::StagedLayoutConversionInfo` or add a parallel internal staging descriptor so the scheduler can distinguish:

```text
ring destination: source-layout LX staging
final destination: matmul KERNEL RHS tile
```

Candidate files:

```text
dsc/dsc2.h
dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp
```

Important invariant: ring transfer destination must not be the final PT/KERNEL layout allocation when `requires_layout_conversion=true`.

### Patch 3: Create staged transfer nodes

Update `populateMatmulOperandBroadcastRingTransfers(...)` so layout-converting plans populate transfers to staging, not the KERNEL operand.
