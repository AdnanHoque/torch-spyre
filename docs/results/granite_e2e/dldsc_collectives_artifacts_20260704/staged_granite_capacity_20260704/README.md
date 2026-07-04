# Staged Matmul Operand Granite Capacity Check - 2026-07-04

This bundle records the first Granite S512 prefill replay that reached DXP with `DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1` using the correct runbook FMS checkout:

`/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/decode_regression_rev_ab_20260610_163300/foundation-model-stack-eager_spyre`

## What Passed

The synthetic row-pattern matmul operand probe passes after replacing the debug source-stick offset knob with a geometric source subpiece address:

- M16: `ALLCLOSE True`, `MISMATCH 0 / 4096`
- M32: `ALLCLOSE True`, `MISMATCH 0 / 8192`
- M64: `ALLCLOSE True`, `MISMATCH 0 / 16384`

This validates the two-stage carrier for small controlled cases:

1. `STCDPOpLx` grouped gather/all-gather into temporary LX.
2. `ReStickifyOpLx` local layout conversion into the matmul operand layout.

## Granite Finding

Granite S512 reaches DXP but fails on the first attention matmul operand edge:

- SuperDSC: `10_batchmatmul`
- plan: `10_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`
- source distribution: 32 `mb` shards
- consumer distribution: `{x:16, out:2}`
- communication pattern: `all_gather_replicate_with_layout_conversion`
- logical transfers: 512

Two resident materialization attempts failed:

1. Reuse imported final address: fails because the final address overlaps an existing source LX allocation.
2. Allocate a fresh final region: fails because the resident converted KERNEL operand is too large for LX.

The capacity math is the core result. The consumer KERNEL RHS has dimensions `{mb:32, out:512, in:128}` and the consumer split is `{out:2}`. A fully resident converted RHS per consumer core is roughly:

```text
32 * (512 / 2) * 128 * 2 bytes = 2 MiB per core
```

That exceeds the ~1.6 MiB usable LX budget before temporary gather pieces. Therefore the next viable backend design is not resident materialization; it is loop/tile-scoped staged movement and local restickify bound to the matmul transfer loop.

## Archived Files

- `plans/10_batchmatmul_Tensor1_plan_initial_overlap.json`
- `plans/10_batchmatmul_Tensor1_plan_fresh_final.json`
- `results/initial_overlap_result.json`
- `results/fresh_final_result.json`
- `logs/initial_overlap_run_tail.txt`
- `logs/fresh_final_run_tail.txt`
- `patches/deeptools_staged_matmul_operand_experiment.diff`
- `patches/deeptools_staged_matmul_operand_experiment.diffstat.txt`
