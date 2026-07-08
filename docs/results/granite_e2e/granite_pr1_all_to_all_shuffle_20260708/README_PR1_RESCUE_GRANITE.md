# Granite PR1 Rescue Placement Device Run

Run root:
`/home/adnan/codex-isolated/pr1_rescue_compare_20260708/runs/granite_rescue_device_20260708_200414`

Torch tree:
`/home/adnan/codex-isolated/pr1_rescue_compare_20260708/torch-spyre-rescue-device`

This tree is a device-runnable transplant of the allocator-native LX relayout rescue implementation into the known-good PR1 AIU checkout. It preserves the benchmark GraphEditor compatibility fix and uses the known-good `_C.so` rather than rebuilding against mismatched runtime headers.

## Timing

| Variant | Kernel ms/iter | Wall median ms | Speedup |
|---|---:|---:|---:|
| baseline off | 14.7002 | 28.0849 | 1.000x |
| rescue + full Torch LX/backend=1 | 12.0353 | 24.9376 | 1.221x kernel / 1.126x wall |

Primary metric is archived Kineto trace-derived `kernel_ms_per_iter` from `trace_summary.json`.

## Env

Baseline:

```bash
DXP_LX_FRAC_AVAIL=0.2
```

Rescue/optimized:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1
LX_BOUNDARY_CLONES=1
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=1
PATH=/home/adnan/codex-isolated/pr1_rescue_compare_20260708/tools/dxp-split-wrapper:$PATH
```

The wrapper makes Torch see full LX (`DXP_LX_FRAC_AVAIL=0`) while Deeptools sees backend scratch space (`DXP_LX_FRAC_AVAIL=1`).

## Granite SDSC Regions

Baseline Granite emits four fused regions:

1. `sdsc_fused_linear_rms_norm_0_*`: input RMSNorm + QKV projection.
2. `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_*`: attention core, including QK/softmax/value BMM.
3. `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_*`: attention output projection + residual/RMSNorm.
4. `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_*`: MLP/SwiGLU.

Optimized rescue emits four corresponding regions, with some fusion naming changes:

1. `sdsc_fused_linear_rms_norm_0_*`
2. `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_*`
3. `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_*`
4. `sdsc_fused_add_linear_mul_3_*`

Jamie-style summaries are in:
`sdsc_reports/`

## PR1-Classified Edges Removed/Improved

### 1. Attention value-side handoff into value BMM

Granite op mapping: attention core, softmax/value-side BMM.

Baseline evidence:

- Region: `baseline_off/...sdsc_fused__scaled_dot_product...view_1__tq1btul`
- `sdsc_7`: `ReStickifyOpHBM`, `INPUT (hbm), OUTPUT (hbm)`
- `sdsc_8`: `batchmatmul`, first input `0_hbm`

Optimized evidence:

- Region: `rescue_full_torch_lx_backend1/...sdsc_fused__scaled_dot_product...view_1_a_5_pfbj`
- `sdsc_9`: `ReStickifyOpHBM`, `INPUT (lx), OUTPUT (hbm)`
- `sdsc_10`: `batchmatmul`, first input `0_lx`

Communication class: PR1 `all_to_all_shuffle` / DLDSC relayout metadata. The consumer BMM now sees the activation operand as LX-resident instead of forcing the same handoff through HBM.

### 2. Attention output projection input

Granite op mapping: attention result into output projection matmul.

Baseline evidence:

- Region: `baseline_off/...sdsc_fused__scaled_dot_product...add_linear_mul_rms_norm_transpose_view_2_e4x0j0sy`
- `sdsc_0`: `ReStickifyOpHBM`, `INPUT (hbm), OUTPUT (hbm)`
- `sdsc_1`: `batchmatmul`, first input `0_hbm`

Optimized evidence:

- Region: `rescue_full_torch_lx_backend1/...sdsc_fused__scaled_dot_product...add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_4jydowcu`
- `sdsc_1`: `batchmatmul`, first input `0_lx`

Communication class: PR1 `all_to_all_shuffle` / DLDSC relayout metadata. The output projection consumes the activation from LX instead of an HBM-backed handoff.

## Not Removed By PR1

PR1 does not remove the MLP/SwiGLU activation spills. In the optimized run, the fused MLP-start region still has:

- `sdsc_10`: `ReStickifyOpHBM` for front projection input/weight-side layout.
- `sdsc_11`: `batchmatmul` with HBM operands.
- `sdsc_12`: `silu`, `INPUT (hbm), OUTPUT (lx)`.
- `sdsc_13`: `mul`, with one remaining HBM input and HBM output.

Those are PR2/all-gather-restickify or future WSR-shaped work, not PR1 all-to-all shuffle.

Weight restickifies are also out of scope for this pass.
