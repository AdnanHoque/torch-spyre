# Granite Structural Gap After Matmul Operand Multicast Fix

This directory archives two Granite block structural compile probes on CDX after
adding grouped multicast support to the matmul operand relayout path.

## Branches

- Torch branch: `gather-restickify`
- Torch SHA: `c9e0e9ae`
- Deeptools branch: `gather-restickify`
- Deeptools SHA: `b8c8a8a4e`
- Pod: `adnan-cdx-spyre-dev-pf`

## Probe Mode

The probe uses `benchmarks/granite_block_layer_probe.py` with one local
modification: `_sync(value)` is skipped after the block call. This keeps the
run structural: it exercises Torch lowering, SDSC generation, DXP/DCC lowering,
and backend plan emission without entering the full AIU runtime sync path that
previously hung.

These are not performance runs and not value-correctness runs.

## S512 Prefill

Shape: `B=1, S=512, H=4096`, causal prefill.

Result: failed in Deeptools before DCC completion.

Key error:

```text
matmul_operand_broadcast gather/restickify materialization failed:
unable to allocate final matmul operand LX shard on core 0 bytes 2097152
```

Plan evidence:

- `8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`
- kind: `matmul_operand_broadcast`
- logical transfers: `512`
- group count: `2`
- replication factor: `16`
- physical lowering status in artifact: `blocked`

Interpretation: S512 needs a final scoped operand shard that is too large for
the current dense materialization strategy. This is a working-set/capacity gap,
not a coordinate classification gap.

## S256 Prefill

Shape: `B=1, S=256, H=4096`, causal prefill.

Result: Deeptools lowered both matmul operand relayouts, then DCC failed.

Key error:

```text
Require larger IBUFF
Max IBUFF(128) Current IBUFF(153)
error: Unable to lower successfully the module for sdsc: 16_batchmatmul
```

Plan evidence:

- `8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`
- `16_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`
- both kind: `matmul_operand_broadcast`
- both strategy: `gather_then_restickify`
- both physical lowering: `lowered_gather_then_restickify`

Interpretation: S256 proves the current backend can lower this Granite
attention communication class through staged LX gather/restickify, but the
resulting DCC program exceeds IBUFF for the second matmul. The next backend
work is reducing the generated transfer/restickify schedule footprint or
chunking/pipelining it so IBUFF stays under the hardware limit.

## Current Read

The multicast fix is real and value-correct on the synthetic probe, and it
keeps flash attention structurally HBM-free. Full Granite now exposes the next
two blockers:

- capacity at S512;
- DCC instruction-buffer pressure at S256.

This supports the working-set-reduction direction rather than adding more
copy-only communication classes first.

