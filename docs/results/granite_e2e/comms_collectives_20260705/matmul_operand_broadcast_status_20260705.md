# Matmul Operand Broadcast / All-Gather Relayout Status - 2026-07-05

## Scope

This note records the current exploratory state for the non-weight Granite spill class where an LX-resident activation shard is consumed as a matmul KERNEL operand with a different tensor/compute distribution.

Communication class:

- `matmul_operand_broadcast`
- equivalent high-level class: grouped all-gather / broadcast with local layout conversion
- representative synthetic edge: `1_batchmatmul` `Tensor1`
- representative Granite/attention class: activation `ReStickifyOpHBM -> batchmatmul` RHS handoff

## Current Findings

The DLDSC metadata path is now sufficient for Deeptools to identify the useful edge:

- source LX tensor distribution is present
- target KERNEL tensor distribution is present
- consumer compute split is present
- backend movement plan expands to logical all-gather transfers

The latest CDX prototype also gets through the first backend gap: DXP emits L3 ring send/recv PCFG nodes for the synthetic matmul operand broadcast case.

Key run:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506/kernel_neighbor_skipviews_M16_001835
```

Important evidence:

```text
[matmul_operand_ring_populate] attaching to prefinalize template=transfer_lds1_src:no_component_dst:no_component_lx_neighbor lds=Tensor1
[lx_neighbor_pcfg_dsc2] node=transfer_lds1_src:no_component_dst:no_component_lx_neighbor transfers=1024 core=0 unit=l3lu
[lx_neighbor_pcfg_dsc2] emit recv ...
[lx_neighbor_pcfg_dsc2] node=transfer_lds1_src:no_component_dst:no_component_lx_neighbor transfers=1024 core=0 unit=l3su
[lx_neighbor_pcfg_dsc2] emit send ...
```

So the current blocker is no longer "can DXP accept or emit ring movement?" It can. The blocker is value correctness for the layout-converting KERNEL operand path.

## What Was Fixed In The Prototype

The Deeptools exploratory branch currently has local fixes for:

- Matching `matmul_operand_broadcast` classifications even when `lds.lxSize_` is unset for layout-converting KERNEL operands.
- Seeding a destination allocation from DLDSC target metadata when there is no same-LDS folded source allocation.
- Avoiding a `setRelevantCompCoreCl()` crash when a schedule node has explicit L3 relevance before the default `NO_COMPONENT` core/corelet map is installed.
- Treating the artificial LX-neighbor ring node as a dependency/placement node, not a normal `NO_COMPONENT -> LX` data view.
- Emitting `RINGDATATRANSFER` PCFG nodes for the ring transfer list.

## Current Value-Correctness Blocker

Two diagnostic runs isolate the remaining issue.

### 1. Fixed-address carousel emits ring traffic but aliases the source region

Run:

```text
kernel_neighbor_skipviews_M16_001835
```

Result:

```text
ALLCLOSE False
MAX_DIFF 3.0
MISMATCH 3829 / 4096
ROWMAP_OUT0 [0.0, 0.0, ..., 3.0, 3.25, 3.5, 3.75]
```

Interpretation:

The destination KERNEL allocation base is `0`, the same as the producer LX source shard. The ring carousel overwrites source data that later phases still need.

### 2. Diagnostic destination shift removes aliasing but exposes layout mismatch

Run:

```text
kernel_neighbor_destshift_M16_002150
```

Diagnostic env:

```text
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_AVOID_SOURCE_ALIAS_DIAGNOSTIC=1
```

Result:

```text
ALLCLOSE False
MAX_DIFF 4668.0
MISMATCH 4096 / 4096
```

Interpretation:

Avoiding aliasing is necessary but not sufficient. The direct/fused ring writes are not yet using the same physical layout that the downstream PT matmul KERNEL operand reader expects.

### 3. Direct all-gather diagnostic confirms this is not only carousel ordering

Run:

```text
kernel_neighbor_direct_shift_M16_002402
```

Diagnostic env:

```text
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_DIRECT_ALLGATHER_DIAGNOSTIC=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_AVOID_SOURCE_ALIAS_DIAGNOSTIC=1
```

Result:

```text
ALLCLOSE False
MAX_DIFF 4.0
MISMATCH 4096 / 4096
ROWMAP_OUT0 [0.1875, 0.4375, 0.6875, 0.9375, ...]
```

Interpretation:

Even direct source-to-destination transfers are chunk-rotated/mislaid relative to the matmul KERNEL read view. The remaining problem is therefore the layout-converting placement contract, not simply the ring route.

### 4. Same-core-as-ring is not a valid local-copy workaround

Run:

```text
kernel_neighbor_direct_selfring_shift_M16_003720
```

Diagnostic env:

```text
DEEPTOOLS_LX_NEIGHBOR_ALLOW_SELF_RING_DIAGNOSTIC=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_DIRECT_ALLGATHER_DIAGNOSTIC=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_AVOID_SOURCE_ALIAS_DIAGNOSTIC=1
```

Result:

```text
RAS::PCI::BusFence
message="PCIe bus master fence"
```

Interpretation:

The ring path should not be used for source-core == destination-core transfers. Once the destination is moved away from the source shard to avoid aliasing, same-core pieces need an actual local LX copy or a two-stage data-op path. Treating a local copy as a self-ring transfer is unsafe.

## Granite S=512 Checkpoint

An earlier Granite block checkpoint with the loop-scoped matmul operand broadcast path classified two attention-side RHS edges:

```text
10_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
18_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
```

The archived timing from that run was:

| Variant | kernel_ms_per_iter | wall median ms |
|---|---:|---:|
| relayout disabled | 14.7258 | 27.6074 |
| relayout enabled, backend LX frac 0.2 | 13.8213 | 26.5205 |

This is about `1.07x` kernel speedup for that checkpoint. It is useful evidence that removing these attention-side HBM handoffs can matter, but it should not be treated as the final performance result while the synthetic KERNEL operand all-gather path is still value-wrong.

## Design Implication

The `matmul_operand_broadcast` class is not the same as PR1 scatter.

PR1 scatter:

- same logical tensor view
- source and destination slices cover the same stick layout
- backend can derive one direct LX move per cell

This class:

- source is an activation tensor view
- destination is a matmul KERNEL operand view
- all consumer cores need replicated source chunks
- a local layout conversion/restickify is needed before PT consumes the operand
- destination must not alias the source LX shard

The clean production design is likely a two-stage loop-scoped path:

1. Ring all-gather source-layout chunks into a non-overlapping loop-local staging region.
2. Perform local LX-to-LX layout conversion/restickify into the KERNEL operand view consumed by PT.

The older full resident staged path was value-correct on small synthetic cases but fails Granite capacity because it materializes the full KERNEL operand per core. The next implementation should keep that value-correct decomposition but make it loop/tile-scoped.

## Next Steps

1. Add an explicit non-overlap allocation contract for relayout destination buffers, not a post-allocation diagnostic shift.
2. Stop trying to fold all-gather and KERNEL restickify into one address formula until the KERNEL physical view is formally derived.
3. Prototype loop-scoped `source-layout all-gather -> local ReStickifyOpLx/KERNEL layout` for the synthetic M16/M32/M64 cases.
4. Once synthetic correctness passes, replay the Granite attention RHS spill and measure whether it removes the HBM round trip without exceeding LX.
