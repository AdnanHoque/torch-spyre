# Remaining HBM Rows: Weight Classification

The kernel-neighbor candidate removes the non-weight attention activation handoff that previously appeared as `sdsc_7: ReStickifyOpHBM` in the first attention kernel. The enabled run has `sdsc_7: ReStickifyOpLx` instead.

The four remaining `ReStickifyOpHBM` rows in the enabled one-layer Granite S512 run are weight/prelayout rows, not computed activation spills:

| kernel | row | mapped buffer | shape | interpretation |
|---|---|---|---|---|
| `sdsc_fused_linear_rms_norm_0` | `sdsc_6` | `buf45` | `[6144,4096]` | QKV/front attention projection weight |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2` | `sdsc_0` | `buf47` | `[4096,4096]` | attention output projection weight |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3` | `sdsc_0` | `buf48` | `[25600,4096]` | MLP gate/up projection weight |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3` | `sdsc_4` | `buf49` | `[4096,12800]` | MLP down projection weight |

Evidence comes from the debug compile at:

`/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_kernel_neighbor_debuglog2_20260706_142118`

Relevant debug lines:

```text
lx_pinning: buf45 (restickify) -> op not allowed
lx_pinning: buf47 (restickify) -> op not allowed
lx_pinning: buf48 (restickify) -> op not allowed
lx_pinning: buf49 (restickify) -> op not allowed
kernel_store: buf45, shape=[6144, 4096]
kernel_store: buf47, shape=[4096, 4096]
kernel_store: buf48, shape=[25600, 4096]
kernel_store: buf49, shape=[4096, 12800]
```

These rows are intentionally out of scope for the communication pass because weight preloading/prelayout is owned by a separate optimization lane. The in-scope activation communication proof point is the HBM-to-LX replacement in the attention kernel.
