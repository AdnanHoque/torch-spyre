# Granite S512 Kernel-Neighbor Wall/SDSC Comparison

This run compares relayout disabled vs the kernel-neighbor candidate on a one-layer Granite causal prefill shape `[1,512,4096]`. Timing is wall-sync only; the matching profiler overlay was ABI-incompatible, so this is not a Kineto kernel-time claim.

## Timing

| variant | median wall ms | measured iterations |
|---|---:|---|
| `disabled` | 28.787 | `[29.825, 28.777, 28.891, 28.787, 28.714]` |
| `kernel_neighbor` | 28.501 | `[29.117, 28.501, 28.396, 28.187, 28.941]` |

Wall speedup: `1.010x` (0.286 ms).

## SDSC Delta

| variant | ReStickifyOpHBM rows | ReStickifyOpLx rows | backend plans |
|---|---:|---:|---:|
| `disabled` | 5 | 0 | 0 |
| `kernel_neighbor` | 4 | 1 | 2 |

The solved edge in this candidate is the attention value-side matmul operand handoff inside the first attention kernel. In the disabled run it appears as `sdsc_7: ReStickifyOpHBM`; in the enabled run the same row is `sdsc_7: ReStickifyOpLx`, with two backend `matmul_operand_broadcast` plans lowered as loop-scoped kernel-neighbor movement.

Remaining `ReStickifyOpHBM` rows are activation/layout handoffs outside that solved attention operand edge:
- `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_i8_4ggpw/sdsc_0.json`: `0_ReStickifyOpHBM`, split `{'mb': 32, 'out': 1}`
- `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_21f9_3dl/sdsc_0.json`: `0_ReStickifyOpHBM`, split `{'mb': 25, 'out': 1}`
- `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_21f9_3dl/sdsc_4.json`: `4_ReStickifyOpHBM`, split `{'mb': 1, 'out': 25}`
- `sdsc_fused_linear_rms_norm_0_d7zujvzt/sdsc_6.json`: `6_ReStickifyOpHBM`, split `{'mb': 32, 'out': 1}`

These are not weight preloads in this run; they are `OUTPUT -> KERNEL` activation/layout restickifies that need the next communication classes or layout-changing LX restickify support.
