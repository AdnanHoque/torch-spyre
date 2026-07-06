# Synthetic Matmul Operand Multicast Pass

This directory archives the first passing value probe for grouped multicast on
the staged matmul-operand LX relayout path.

## Branches

- Torch branch: `gather-restickify`
- Torch SHA: `c9e0e9ae`
- Deeptools branch: `gather-restickify`
- Deeptools SHA: `b8c8a8a4e`
- Pod: `adnan-cdx-spyre-dev-pf`

## What Changed

Two small changes moved this probe from a blocked dense all-gather path to the
supported loop-scoped matmul operand path:

1. Torch now lets grouped `multicast` topologies use the same RHS matmul operand
   contract as `broadcast` and `all_gather`.
2. Deeptools now derives destination shard metadata for this path from
   `computeCoreIdToWkSlice_` when the target tensor's resident
   `coreIdToWkSlice_` is compact.

The second point matters because KERNEL operands are loop-scoped to consumer
compute cores. A compact allocation map may only describe the resident tensor
storage, while `computeCoreIdToWkSlice_` describes all consumer cores that need
the operand during matmul.

## Result

Return code: `0`

Stdout:

```text
SUCCESS synthetic_value_matmul_operand_broadcast torch.Size([1, 4, 256, 64])
```

Backend plan:

```text
backend_plans/3_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
```

Plan summary:

- `artifact_kind`: `matmul_operand_broadcast_backend_plan`
- `kind`: `matmul_operand_broadcast`
- `communication_pattern`: `all_gather_replicate`
- `realization_strategy`: `gather_then_restickify`
- `physical_lowering_status`: `lowered_gather_then_restickify`
- `logical_transfer_count`: `32`
- `group_count`: `8`
- `replication_factor`: `4`
- `realized`: `true`

## Interpretation

This proves the current staged matmul-operand relayout substrate can handle a
grouped multicast shape:

```text
8 producer shards -> 32 consumer cores
1 source shard fans out to 4 consumer cores per group
```

This is not generic resident all-gather. It is the safer scoped form:

```text
matmul_operand_broadcast -> grouped fanout -> local ReStickifyOpLx -> matmul KERNEL operand
```

The dense resident layout-allgather path remains intentionally blocked for
Granite/attention-sized tensors unless explicitly enabled for diagnostics.

