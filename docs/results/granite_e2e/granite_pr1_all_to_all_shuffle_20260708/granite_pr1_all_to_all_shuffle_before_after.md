# Granite Block PR1 `all_to_all_shuffle` Before/After

Run root:
`/home/adnan/codex-isolated/pr1_rescue_compare_20260708/runs/granite_rescue_device_20260708_200414`

Torch tree used for the optimized run:
`/home/adnan/codex-isolated/pr1_rescue_compare_20260708/torch-spyre-rescue-device`

This report classifies the fresh whole-Granite-block prefill run and maps the PR1 `all_to_all_shuffle` edges to Granite operations. The SDSC tables were generated with Jamie's `summarize-sdsc` helper and are under `sdsc_reports/`.

## Timing

| Variant | Feature state | Kernel ms/iter | Wall median ms | Speedup |
|---|---|---:|---:|---:|
| `baseline_off` | `SPYRE_LX_PLANNER_RELAYOUT` off | 14.7002 | 28.0849 | 1.000x |
| `rescue_full_torch_lx_backend1` | PR1 `all_to_all_shuffle` on | 12.0353 | 24.9376 | 1.221x kernel / 1.126x wall |

Primary metric: archived Kineto trace-derived `kernel_ms_per_iter` from `trace_summary.json`.

## Whole-Block Region Map

| Granite region | Baseline SDSC directory | Optimized SDSC directory | PR1 target? |
|---|---|---|---|
| Input RMSNorm + QKV projection | `sdsc_fused_linear_rms_norm_0_ny_j4_e_` | `sdsc_fused_linear_rms_norm_0_7q4p6065` | No |
| Attention core: QK scores, softmax chain, value BMM | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1__tq1btul` | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_a_5_pfbj` | Yes, one score-side BMM input |
| Attention output projection + residual/RMSNorm + fused MLP start | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_e4x0j0sy` | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_4jydowcu` | Yes, output projection input |
| MLP/SwiGLU tail/down projection | `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_col3nu6t` | `sdsc_fused_add_linear_mul_3_fqyn6bbt` | No |

## PR1 Target 1: Attention Scores

This is the first attention matmul:

```python
S = Q @ K.T
P = softmax(S)
C = P @ V
```

The PR1-targeted edge is the `Q` input to `S = Q @ K.T`.

### Before

SDSC evidence:

- Directory: `baseline_off/.../sdsc_fused__scaled_dot_product...view_1__tq1btul`
- `sdsc_7`: `ReStickifyOpHBM`, `INPUT (hbm), OUTPUT (hbm)`
- `sdsc_8`: `batchmatmul`, first input `0_hbm`, second input `1_hbm`, output `2_hbm`

Residency:

| Formula object | Before location |
|---|---|
| `Q` input to `S = Q @ K.T` | HBM, after an HBM restickify |
| `K` input to `S = Q @ K.T` | HBM |
| `S` output | HBM |

### After

SDSC evidence:

- Directory: `rescue_full_torch_lx_backend1/.../sdsc_fused__scaled_dot_product...view_1_a_5_pfbj`
- `sdsc_9`: `ReStickifyOpHBM`, `INPUT (lx), OUTPUT (hbm)` still exists structurally
- `sdsc_10`: `batchmatmul`, first input `0_lx`, second input `1_hbm`, output `2_hbm`

Residency:

| Formula object | After location |
|---|---|
| `Q` input to `S = Q @ K.T` | LX, after PR1 `all_to_all_shuffle` materializes the consumer view |
| `K` input to `S = Q @ K.T` | HBM |
| `S` output | HBM |

What changed: `Q` no longer takes an HBM round trip before the score matmul. Torch emits the producer/consumer coordinate contract, and Deeptools realizes the on-chip `all_to_all_shuffle` so the score matmul can read `Q` directly from LX.

## PR1 Target 2: Attention Output Projection

This is the handoff from the attention context into the output projection:

```python
C = P @ V
O = C @ W_o
Y = X + O
N = rms_norm(Y)
```

The PR1-targeted edge is the `C` input to `O = C @ W_o`.

### Before

SDSC evidence:

- Directory: `baseline_off/.../sdsc_fused__scaled_dot_product...add_linear_mul_rms_norm_transpose_view_2_e4x0j0sy`
- `sdsc_0`: `ReStickifyOpHBM`, `INPUT (hbm), OUTPUT (hbm)`
- `sdsc_1`: `batchmatmul`, first input `0_hbm`, second input `1_hbm`, output `2_hbm`

Residency:

| Formula object | Before location |
|---|---|
| `C` input to `O = C @ W_o` | HBM, after an HBM restickify |
| `W_o` input to `O = C @ W_o` | HBM |
| `O` output | HBM |

### After

SDSC evidence:

- Directory: `rescue_full_torch_lx_backend1/.../sdsc_fused__scaled_dot_product...add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_4jydowcu`
- `sdsc_1`: `batchmatmul`, first input `0_lx`, second input `1_hbm`, output `2_hbm`

Residency:

| Formula object | After location |
|---|---|
| `C` input to `O = C @ W_o` | LX, after PR1 `all_to_all_shuffle` materializes the consumer view |
| `W_o` input to `O = C @ W_o` | HBM |
| `O` output | HBM |

What changed: `C` no longer spills through HBM before the output projection. The output projection matmul consumes the attention context from LX.

## Not Targeted By PR1 In This Whole-Block Run

### Input RMSNorm + QKV projection

Baseline and optimized both still show the QKV projection matmul with HBM inputs:

- Baseline: `sdsc_fused_linear_rms_norm_0_ny_j4_e_`, `sdsc_7 batchmatmul`, `INPUT (hbm), INPUT (hbm), OUTPUT (hbm)`
- Optimized: `sdsc_fused_linear_rms_norm_0_7q4p6065`, `sdsc_8 batchmatmul`, `INPUT (hbm), INPUT (hbm), OUTPUT (hbm)`

This is not a PR1 `all_to_all_shuffle` edge in this run.

### MLP/SwiGLU

PR1 does not remove the MLP/SwiGLU activation spills. The optimized run still has HBM-backed MLP rows:

- Optimized MLP start region: `sdsc_10 ReStickifyOpHBM`, `sdsc_11 batchmatmul`, `sdsc_12 silu`, `sdsc_13 mul`
- Optimized MLP tail/down-proj region: `sdsc_0 ReStickifyOpHBM`, `sdsc_1 batchmatmul`

These are PR2/all-gather-restickify or later WSR-shaped cases, not PR1 `all_to_all_shuffle`.

## Summary Of What PR1 Bought Us

PR1 removes HBM-backed activation reads on two attention-side matmul inputs:

1. Score-side attention BMM first activation operand: `0_hbm -> 0_lx`.
2. Attention context into output projection: `0_hbm -> 0_lx`.

That is enough to reproduce the whole-block prefill speedup:

```text
kernel_ms_per_iter: 14.7002 -> 12.0353
kernel speedup:     1.221x
wall median:        28.0849 -> 24.9376
wall speedup:       1.126x
```

## Jamie-Style SDSC Summary Artifacts

Full SDSC summaries generated with Jamie's `summarize-sdsc` helper are committed in:

`sdsc_reports/`

Key files:

- Baseline attention core: `baseline_off_block_prefill_cache_inductor-spyre_sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1__tq1btul_.md`
- Optimized attention core: `rescue_full_torch_lx_backend1_block_prefill_cache_inductor-spyre_sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_a_5_pfbj_.md`
- Baseline attention output projection: `baseline_off_block_prefill_cache_inductor-spyre_sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_e4x0j0sy_.md`
- Optimized attention output projection: `rescue_full_torch_lx_backend1_block_prefill_cache_inductor-spyre_sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_4jydowcu_.md`
- Baseline MLP/SwiGLU: `baseline_off_block_prefill_cache_inductor-spyre_sdsc_fused_add_linear_mul_silu_split_with_sizes_3_col3nu6t_.md`
- Optimized MLP/SwiGLU tail: `rescue_full_torch_lx_backend1_block_prefill_cache_inductor-spyre_sdsc_fused_add_linear_mul_3_fqyn6bbt_.md`
