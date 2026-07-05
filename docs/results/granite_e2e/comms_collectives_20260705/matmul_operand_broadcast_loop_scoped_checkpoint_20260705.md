# Matmul Operand Broadcast: Loop-Scoped Checkpoint

This checkpoint records the current state of the remaining non-weight Granite/attention HBM spill class after validating the PR1 scatter path and same-layout collectives.

## Status

PR1 scatter is still valid for the class it targets: an LX-resident producer tensor and an LX-consuming op have different core ownership, but the tensor layout/stick format is already compatible. Torch emits the tensor-distribution-vs-compute metadata in the DLDSC, and Deeptools derives the core-to-core movement.

The remaining Granite/attention RHS handoff is broader. It is not just scatter. It is:

```text
producer LX shard
  -> grouped all-gather / broadcast across consumer cores
  -> local layout conversion / restickify into the matmul KERNEL operand layout
  -> batchmatmul consumes the KERNEL operand
```

The communication class is therefore `all_gather_replicate + layout_conversion`, not plain `scatter`.

## What Has Been Proven

1. Same-layout scatter works for the PR1 class.
2. Same-layout 4-way RHS broadcast/all-gather passes synthetic checks.
3. Same-layout gather can pass when the destination LX base is explicitly safe.
4. Full resident `STCDPOpLx -> ReStickifyOpLx` is value-correct on small synthetic runs.
5. Full resident gather/restickify fails Granite-scale attention because it materializes the full RHS operand per consumer core and cannot reserve enough LX.
6. Direct ring-to-KERNEL is capacity-friendly but value-wrong. It writes bytes into the final KERNEL region without applying the required source-layout-to-KERNEL-layout conversion.
7. Same-core transfers must not be represented as self-ring traffic. Self-ring diagnostic traffic fenced the device. Same-core cases should be direct/no-op when aliasing is valid, or use the existing LXLU/LXSU local copy path when a distinct LX destination must be materialized.

## Concrete Evidence

Representative run roots on CDX:

- `/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/granite_eagerfms_staged_backend02_geometric_20260704_232053`
- `/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/granite_eagerfms_staged_finalfresh_backend02_20260704_232511`
- `/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506`

Granite full resident staged attempt failed with:

```text
matmul_operand_broadcast gather/restickify materialization failed:
unable to allocate final matmul operand LX region on core 0
```

Small synthetic full resident staged path had value-correct evidence from prior run logs:

```text
ALLCLOSE True
MAX_DIFF 0.001953125
```

Direct ring-to-final-KERNEL experiments emitted ring traffic but produced wrong values, confirming that ring movement alone is insufficient for KERNEL operands.

The Granite attention plan artifact shows why direct address remapping is not enough:

```text
source_lx_tensor:      layout=[mb,x,out],  stick=[x]
target_kernel_tensor: layout=[mb,out,in],  stick=[out]
consumer compute:     split includes x and out
```

So the handoff changes both ownership and stick/layout form. A same-layout all-gather can be realized as ring movement, but this edge also needs a local source-layout-to-KERNEL-layout conversion before the matmul can consume the operand.

## Local-Copy Diagnostic

After the direct ring-to-final-KERNEL path failed, we tested whether the problem was simply that same-core pieces were being dropped. A diagnostic variant added an explicit same-core local-copy path using the existing `LX -> LXLUSUFIFO -> LX` mechanism while keeping the direct KERNEL all-gather experiment enabled.

Run root:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506/kernel_neighbor_direct_localcopy_mixed_M16_013136
```

Diagnostic env:

```text
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_DIRECT_ALLGATHER_DIAGNOSTIC=1
DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1
DEEPTOOLS_LX_NEIGHBOR_MAX_RUN=8
DEEPTOOLS_LX_NEIGHBOR_VIEW_PROBE=1
```

Result:

```text
ALLCLOSE False
MAX_DIFF 4.0
MISMATCH 4085 / 4096
ROWMAP_OUT0 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ROWMAP_REF0 [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]
```

Interpretation: explicit local copying is not sufficient. The direct KERNEL path is wrong because it still bypasses the required loop-scoped layout conversion/restickify into the KERNEL operand format. The diagnostic code was reverted after this test.

## Correct Backend Hook

The coarse DXP `dataOpdscs_` path can schedule `STCDPOpLx` and `ReStickifyOpLx` before a DL op, but that is the wrong granularity for Granite-scale attention. It materializes too much at once.

The correct hook is in the DL schedule:

- `L3DlOpsScheduler` inserts the artificial LX-neighbor transfer inside the innermost matmul loop.
- `populateMatmulOperandBroadcastRingTransfersAfterAllocation` has access to allocated LX bases after the scheduler assigns addresses.
- `dsc2Pcfg.cpp::buildLxNeighborRingTransfers` already lowers loop-scoped ring send/recv nodes.

The missing piece is a staged loop-scoped realization:

```text
for each matmul tile / consumer core:
  1. ring-gather producer sticks into a small source-layout temp LX tile
  2. locally restickify/layout-convert that temp tile into the final KERNEL operand allocation
  3. run the matmul tile using the now-correct KERNEL operand bytes
```

## Required Deeptools Shape

The next backend change should add explicit staged metadata to the loop-scoped transfer node, not another coarse mixed-SDSC data-op row.

Recommended representation:

- keep `lxNeighborRingTransfers_` for cross-core ring sends/receives into the temp LX tile;
- add a small staged-restickify payload on the same transfer node describing:
  - source/temp LX base and byte range;
  - destination KERNEL LX base and byte range;
  - source logical layout;
  - target KERNEL logical layout;
  - producer chunk / consumer replica / group metadata;
  - whether the same-core case is direct/no-op or requires LXLU/LXSU copy.

Recommended lowering:

- emit the existing `RINGDATATRANSFER` nodes first;
- then emit a local layout-conversion/restickify sequence before returning to the matmul compute path;
- do not use self-ring for local copies.

Allocator note: this temp should be represented as a real scheduled LX allocation near the loop, but it should not replace the consumer LDS allocation. The closest existing pattern is paged-index staging, which creates a loop-near allocation, checks `allocAllMem`, and falls back to smaller buffering when the full resident form does not fit. The matmul operand temp needs the same style of allocation discipline, but for an activation/KERNEL operand tile.

## Why This Matters

This preserves the North Star contract:

1. Torch/Inductor describes the tensor distribution and consumer compute distribution.
2. The planner classifies the edge as `all_gather_replicate + layout_conversion`.
3. Deeptools owns the physical synthesis: ring movement, local layout conversion, address assignment, and legal scheduling.
4. Later scheduling work can overlap gather/restickify for tile `i+1` with compute for tile `i`.

## Current Bottom Line

The remaining Granite/attention spill is not blocked by the DLDSC coordinate contract. It is blocked by the lack of a loop-scoped staged backend primitive for `all_gather_replicate + layout_conversion` into KERNEL operands.
