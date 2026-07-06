# Remaining Granite HBM Rows After DLDSC Relayout - 2026-07-06

Artifact branch base SHA before this note: `513c7a9934a3bdc6b337259e7cd5eea790ff516d`.

This note classifies the remaining `ReStickifyOpHBM` rows in the current Granite S512 profiled run after enabling the DLDSC relayout / kernel-neighbor matmul operand path.

Run root:

`/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_kernel_neighbor_profile_profpy212_20260706_145100`

## Current Read

The enabled run still contains four `ReStickifyOpHBM` rows. They are all adjacent to linear `batchmatmul` projections and match the weight-prelayout class that is out of scope for this communication-collectives work. The in-scope attention activation handoff has been converted to `ReStickifyOpLx` plus `matmul_operand_broadcast` metadata and is no longer an HBM restickify.

## Remaining HBM Rows

| Kernel directory | File | Op | Nearby consumer | Classification |
| --- | --- | --- | --- | --- |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_wkgi00mv` | `sdsc_0.json` | `0_ReStickifyOpHBM` | `sdsc_1.json` / `1_batchmatmul` | Linear projection weight/input-layout restickify, out of scope for this pass |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_l0vnwvyf` | `sdsc_0.json` | `0_ReStickifyOpHBM` | `sdsc_1.json` / `1_batchmatmul` | MLP first projection weight restickify, out of scope |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_l0vnwvyf` | `sdsc_4.json` | `4_ReStickifyOpHBM` | `sdsc_5.json` / `5_batchmatmul` | MLP down-projection weight restickify, out of scope |
| `sdsc_fused_linear_rms_norm_0_i4dogoht` | `sdsc_6.json` | `6_ReStickifyOpHBM` | `sdsc_7.json` / `7_batchmatmul` | Output/linear projection weight restickify, out of scope |

## In-Scope Spill Already Removed

The attention value-side handoff in:

`sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_35tt9oeh`

now uses:

- `sdsc_7.json`: `7_ReStickifyOpLx`
- `sdsc_8.json`: `8_batchmatmul`
- backend plan: `8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`
- backend plan: `16_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`

This is the row responsible for the attention handoff kernel-group improvement in the profiled comparison:

- disabled attention handoff kernel group: about `5.862 ms` total over 5 events
- enabled attention handoff kernel group: about `2.822 ms` total over 5 events
- local handoff speedup: about `2.08x`

## Interpretation

For the current Granite S512 profiled run, the remaining visible `ReStickifyOpHBM` rows do not point to a missing copy-only communication primitive. They point to weight layout/preload work. That matches the explicit scope boundary for this branch: remove non-weight HBM spills using DLDSC relayout and collectives, while leaving weight prelayout to the separate weight-preloading effort.

The next communication primitive work should therefore not target these four rows. It should target new workload edges that remain HBM-backed because they require a communication class not yet covered, especially reduce/all-reduce or generic resident gather/all-gather outside the already validated matmul operand path.
