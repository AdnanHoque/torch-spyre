# Focused attention-bundle sync timeout

Date: 2026-07-06

This is the smallest AIU reproducer from the current Granite path.

It imports the generated Granite compiled module, creates a Spyre runtime context, runs only:

1. `sdsc_fused_linear_rms_norm_0`
2. `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1`
3. `torch.accelerator.synchronize()`

The attention bundle contains the staged `matmul_operand_broadcast` plans. Runtime flags include:

```bash
DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_MAX_CHUNKS=1
```

Result:

- return code: `124` from the 180s timeout wrapper
- RMS bundle returns immediately
- attention bundle returns immediately
- `torch.accelerator.synchronize()` after attention hits `RuntimeStream::synchronize() still waiting after 60000ms: in_flight_=1 device=0`
- no `AFTER torch.accelerator.synchronize` is printed

Interpretation:

The relayout-bearing attention bundle itself leaves unresolved runtime work. The downstream Granite bundle is not required to reproduce the lost-completion behavior.
