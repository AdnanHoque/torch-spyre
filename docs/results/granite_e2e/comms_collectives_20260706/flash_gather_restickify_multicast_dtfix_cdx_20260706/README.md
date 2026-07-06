# Flash Attention Structural Probe After Multicast Fix

This directory archives a compile-only structural probe of `test_flash.py` on
the clean `gather-restickify` branches after adding grouped multicast support
to the matmul operand relayout path.

## Branches

- Torch branch: `gather-restickify`
- Torch SHA: `c9e0e9ae`
- Deeptools branch: `gather-restickify`
- Deeptools SHA: `b8c8a8a4e`
- Pod: `adnan-cdx-spyre-dev-pf`

## Probe Mode

The run used the same structural harness as the earlier flash evidence:

- `PATCH_MODE=no_h2d,skip_cpu_ref`
- `Tensor.to(device="spyre")` is patched to allocate empty Spyre tensors.
- CPU reference and `assert_close` are skipped.

This avoids the known baseline flash value issue and only validates compile,
SDSC generation, DXP lowering, and relayout artifact emission.

## Result

Return code: `0`

Stdout:

```text
[runtime_patch] no_h2d Tensor.to(device=spyre) compile probe enabled
[runtime_patch] assert_close skipped for compile probe
SUCCESS
```

Summary:

- SDSC files: `550`
- `ReStickifyOpHBM_total`: `0`
- `ReStickifyOpLx_total`: `160`
- Relayout metadata SDSCs: `550`
- Backend matmul operand plans: `32`
- Plan kind: `matmul_operand_broadcast`
- Realization strategy: `gather_then_restickify`
- Physical lowering: `lowered_gather_then_restickify`
- Representative logical transfers per plan: `256`
- Representative group count: `4`
- Representative replication factor: `8`

## Interpretation

The updated branch still removes the flash attention HBM restickify path in the
structural compile probe. All 32 matmul operand relayouts lower through the
staged path:

```text
source LX shards -> grouped fanout -> local ReStickifyOpLx -> KERNEL matmul operand
```

This is structural evidence only. End-to-end value correctness for this flash
script is still blocked by the known unrelated baseline zero-stride/broadcast
view issue.

