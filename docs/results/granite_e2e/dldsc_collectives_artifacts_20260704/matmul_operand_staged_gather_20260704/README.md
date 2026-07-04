# Matmul Operand Broadcast: Staged Gather + Restickify Evidence (2026-07-04)

This artifact records the July 4 diagnostic for the Granite/attention communication class where an LX-resident producer becomes a downstream `batchmatmul` RHS/KERNEL operand. This is an all-gather/replicate communication pattern plus a KERNEL layout conversion.

## Result

The direct loop-scoped KERNEL-neighbor write is not value-correct for M >= 16 in the synthetic row-pattern probe. It emits ring traffic at the right schedule point, but it writes directly into the consumer KERNEL operand address space and skips the local layout conversion that the PT matmul operand expects.

The staged path is value-correct:

```text
DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
```

This path inserts two DataOp rows before the consumer matmul:

1. `STCDPOpLx` gather/all-gather from producer LX shards into a temporary LX layout.
2. `ReStickifyOpLx` local conversion into the consumer KERNEL operand layout.

## Synthetic Coverage

| Variant | Shape | Result | Evidence |
| --- | --- | --- | --- |
| Direct KERNEL-neighbor, max run 8 | M=16,K=64,N=256 | FAIL, 2048/4096 mismatches | `logs/kernel_neighbor_maxrun8_M16_run.log` |
| Direct KERNEL-neighbor, out-major | M=32,K=64,N=256 | FAIL, 6144/8192 mismatches | `logs/kernel_neighbor_outmajor_M32_run.log` |
| Staged gather+restickify | M=16,K=64,N=256 | PASS, allclose true | `logs/gather_restickify_M16_run.log` |
| Staged gather+restickify | M=32,K=64,N=256 | PASS, allclose true | `logs/gather_restickify_M32_run.log` |
| Staged gather+restickify | M=64,K=64,N=256 | PASS, allclose true | `logs/gather_restickify_M64_run.log` |

## Key Row-Map Evidence

Direct KERNEL-neighbor M16 output row map:

```text
ROWMAP_OUT0 [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
ROWMAP_REF0 [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]
```

Staged gather+restickify M64 output row map matches the reference exactly for the printed rows.

## Granite Smoke Note

Attempted one-layer Granite S512 smoke with the staged path, but current Torch/FMS environments fail before DXP planning with:

```text
Unsupported: Spyre backend does not support: All inputs to an op must have same element arrangement, op: mul, args: "buf0": ElementArrangement.DL16_TO_FP32, "buf4": ElementArrangement.STANDARD
```

No backend plans are emitted in that failure. This is a separate Torch lowering/environment compatibility issue, not evidence against the staged gather+restickify lowering.

## Interpretation

This communication class is not a pure scatter or plain all-gather. For matmul RHS/KERNEL operands, correctness requires both:

- cross-core movement to replicate producer shards, and
- local layout conversion into the PT/KERNEL operand format.

The direct KERNEL-neighbor experiment proved schedule placement and ring emission, but it skipped the layout conversion. The two-stage DataOp path proves the value-correct contract shape for this class.
