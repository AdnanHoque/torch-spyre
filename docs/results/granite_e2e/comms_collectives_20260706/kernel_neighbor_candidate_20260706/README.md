# Granite/Flash Matmul Operand Kernel-Neighbor Candidate - 2026-07-06

This artifact records the CDX exploration that moved Granite attention matmul operand communication from dense resident gather/restickify to loop-scoped LX input-neighbor fetch.

## Code Under Test

- Torch: `AdnanHoque/torch-spyre:gather-restickify` at `c9e0e9ae039303714b531742ea96c6c05b54faf0`
- Deeptools: `Adnan-Hoque1/deeptools:gather-restickify` at `e3e265d22c7283054dd36e147a7e7ec919606441`
- Pod used for these artifacts: `adnan-cdx-spyre-dev-pf`
- CLC was intentionally not used; it was reserved for Claude.

## What Changed

The passing path uses the existing matmul operand broadcast classification from Torch, but asks Deeptools to realize it as a loop-scoped KERNEL-neighbor operand instead of dense resident materialization.

Deeptools changes in this candidate:

1. Seed KERNEL-neighbor destination allocation coordinates from the consumer SuperDSC `coreIdToWkSlice_`, not the producer/source allocation. This gives DDC the full consumer loop-coordinate contract for offset calculation.
2. Permit mixed HBM-pinned tensors and LX input-neighbor tensors only when `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1` is active. Generic mixed HBM+IFN remains guarded.

## Result Summary

| Run | Result | Meaning |
| --- | --- | --- |
| `dense_gather_restickify_s256_fail` | fail | Dense gather/restickify lowers both plans but DCC hits IBUFF overflow. |
| `dense_gather_restickify_s512_fail` | fail | Dense gather/restickify tries to allocate a too-large final resident operand shard. |
| `chunk2_s256_fail` | fail | Smaller dense chunks reduce IBUFF slightly but still exceed the limit. |
| `chunk1_s256_fail` | fail | Too-small chunks duplicate control overhead and make IBUFF worse. |
| `kernel_neighbor_s256_guard_fail` | fail | Kernel-neighbor classification works, but old backend guard rejects mixed HBM+IFN. |
| `kernel_neighbor_s256_diag_pass` | pass | With diagnostic guard bypass plus consumer-coordinate seed, S256 compiles. |
| `kernel_neighbor_s256_prod_pass` | pass | Production-shaped path compiles without diagnostic bypass. |
| `kernel_neighbor_s512_prod_pass` | pass | S512 compiles without dense final operand allocation. |
| `flash_kernel_neighbor_compile_pass` | pass | Flash compile probe emits 32 loop-scoped matmul operand plans. |

## Passing Granite Evidence

### S256

- Run: `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s256_matmul_kernel_neighbor_prod_20260706_134348`
- Return code: `0`
- Plans:

16_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json `loop_scoped_input_fetch` `lowered_loop_scoped_kernel_neighbor` transfers=512 group=2 repl=16<br>8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json `loop_scoped_input_fetch` `lowered_loop_scoped_kernel_neighbor` transfers=256 group=4 repl=8

### S512

- Run: `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_kernel_neighbor_clean_candidate_20260706_134737`
- Return code: `0`
- Plans:

16_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json `loop_scoped_input_fetch` `lowered_loop_scoped_kernel_neighbor` transfers=1024 group=1 repl=32<br>8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json `loop_scoped_input_fetch` `lowered_loop_scoped_kernel_neighbor` transfers=512 group=2 repl=16

## Flash Compile Evidence

- Run: `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_kernel_neighbor_clean_candidate_20260706_134926`
- Return code: `0`
- Plan count: `32`
- Strategy counts: `{'loop_scoped_input_fetch': 32}`
- Lowering counts: `{'lowered_loop_scoped_kernel_neighbor': 32}`

This is compile-only. Runtime value correctness for the flash script is still blocked by the independent zero-stride/broadcast view issue in the baseline path, not by this communication lowering.

## Interpretation

Dense all-gather materialization is the wrong default for large attention matmul operands. It either consumes too much LX for a final resident operand or emits too much address/control logic for DCC. The loop-scoped KERNEL-neighbor path is the scalable direction for these operands: keep the destination operand scoped to the matmul loop and fetch the needed producer pieces through LX-neighbor transfers.

## Reproduction Notes

Use split LX availability:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

and a `dxp_standalone` wrapper that forwards `DXP_BACKEND_LX_FRAC_AVAIL` to the subprocess as `DXP_LX_FRAC_AVAIL`.

Key feature flags for the passing path:

```bash
export SPYRE_LX_PLANNING=1
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
export SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
export SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
unset DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY
unset DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC
```

Each run directory contains the generated `run.sh` when available, `structural_summary.json`, logs, return code, and backend plan JSONs.
