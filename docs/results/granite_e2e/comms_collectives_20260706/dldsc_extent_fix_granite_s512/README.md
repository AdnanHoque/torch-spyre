# Granite S512 DLDSC Relayout Extent Fix

Date: 2026-07-06

## Summary

This records the current `ah/comms-collectives` progress for Granite S512 causal prefill using the DLDSC relayout path.

The important change is a Deeptools substrate fix: generic LX relayout piece construction now derives logical piece extents from the concrete inner DLDSC extent instead of the outer SuperDSC extent. The Granite attention relayout path can produce an outer SuperDSC `N_` value of `-1`; using that value caused DXP replay failure before the useful on-chip movement path could run.

The fixed Deeptools branch is:

```text
Adnan-Hoque1/deeptools:ah/comms-collectives
HEAD: 3095cdb33 [DXP] derive LX relayout pieces from DLDSC extents
```

## Validation

Focused tests on `adnan-cdx-spyre-dev-pf`:

```text
dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
2/2 passed

util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
27/27 passed
```

The exact logs are archived next to this file:

```text
dxp_unit_corework_after_remote_branch.log
util_layout_after_remote_branch.log
```

## Granite Run Roots

Primary run:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_dldsc_extent_fix_20260706_162134
```

Profiler retry with `acc_events=True` in the local bench harness:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_dldsc_extent_fix_acc_events_20260706_162423
```

Both runs are Granite block causal prefill:

```text
input_shape = [1, 512, 4096]
attn_name = sdpa_causal
iters = 5
warmups = 1
fused_weights = true
```

## Timing

Use these wall numbers as a sanity check, not as a final kernel-time claim.

| Run root | Variant | rc | Median wall ms | Trace kernel ms / iter |
|---|---:|---:|---:|---:|
| `granite_s512_dldsc_extent_fix_20260706_162134` | disabled | 0 | 30.4257869720459 | 0.0 |
| `granite_s512_dldsc_extent_fix_20260706_162134` | enabled | 0 | 28.591156005859375 | 0.0 |
| `granite_s512_dldsc_extent_fix_acc_events_20260706_162423` | disabled | 0 | 29.234886169433594 | 0.0 |
| `granite_s512_dldsc_extent_fix_acc_events_20260706_162423` | enabled | 0 | 28.9614200592041 | 0.0 |

Wall-time speedup in the primary run is about `1.064x`.

Important profiler caveat: the fresh Kineto traces currently contain no `cat == "kernel"` events, so `trace_summary.json` reports `kernel_ms_per_iter = 0.0`. This is a profiler/harness issue, not proof of zero kernel work. Do not publish a kernel-time speedup from these fresh traces until the event extraction path is fixed.

## SDSC Before/After

The useful structural change is in the first scaled-dot-product attention kernel. Baseline has a non-weight HBM relayout between the attention pointwise chain and the value-side batchmatmul:

```text
disabled/block_prefill/cache/inductor-spyre/
  sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_l0sb_cr1/
    sdsc_7.json
    op = 7_ReStickifyOpHBM
```

Enabled replaces that HBM relayout with an on-chip LX relayout:

```text
enabled/block_prefill/cache/inductor-spyre/
  sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_hoxo4vfb/
    sdsc_7.json
    op = 7_ReStickifyOpLx
```

That handoff feeds:

```text
sdsc_8.json
op = 8_batchmatmul
split = {in:1, mb:1, out:2, x:16}
```

The removed non-weight spill is an attention activation/layout handoff. Its communication class is:

```text
layout all-gather / matmul operand broadcast
realized as gather-then-restickify through ReStickifyOpLx
```

The enabled run also emits backend movement plans:

```text
enabled/backend_plans/8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
kind = matmul_operand_broadcast
logical_transfer_count = 512
realization = loop-scoped KERNEL-neighbor matmul operand

enabled/backend_plans/16_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
kind = matmul_operand_broadcast
logical_transfer_count = 1024
realization = loop-scoped KERNEL-neighbor matmul operand
```

## HBM Restickify Rows

Primary run, disabled:

| Row | Classification |
|---|---|
| attention `sdsc_7.json`: `7_ReStickifyOpHBM` | non-weight activation/layout spill, removed by enabled run |
| `sdsc_fused_linear_rms_norm_0_4ahdyy5g/sdsc_6.json`: `6_ReStickifyOpHBM` | weight-related |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_zfusx4_e/sdsc_0.json`: `0_ReStickifyOpHBM` | weight-related |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_qgfph3zp/sdsc_0.json`: `0_ReStickifyOpHBM` | weight-related |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_qgfph3zp/sdsc_4.json`: `4_ReStickifyOpHBM` | weight-related |

Primary run, enabled:

| Row | Classification |
|---|---|
| attention `sdsc_7.json`: `7_ReStickifyOpLx` | on-chip replacement for the baseline attention activation/layout HBM spill |
| `sdsc_fused_linear_rms_norm_0_eihz7of0/sdsc_6.json`: `6_ReStickifyOpHBM` | weight-related |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_qnaxp71u/sdsc_0.json`: `0_ReStickifyOpHBM` | weight-related |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_bdquhnb2/sdsc_0.json`: `0_ReStickifyOpHBM` | weight-related |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_bdquhnb2/sdsc_4.json`: `4_ReStickifyOpHBM` | weight-related |

## Current Communication Class Status

| Class | Current evidence |
|---|---|
| Scatter/permutation | PR1 class; not the newly eliminated Granite spill here. |
| Broadcast/multicast | Partially represented by `matmul_operand_broadcast`; still needs broader synthetic and workload coverage. |
| Gather/all-gather | The Granite attention spill is effectively a layout all-gather into a matmul operand, with local restickify. |
| Reduce/all-reduce | Not solved by this copy-only relayout path; needs arithmetic reduction support. |
| Form-changing restickify | Covered in this path through `ReStickifyOpLx` for the attention handoff. |

## Remaining Gaps

1. Fix or re-enable trace-derived kernel timing in the current profiling path. Fresh traces are structurally useful but not timing-authoritative.
2. Run the inductor-team flash attention script with the same DLDSC/Deeptools stack and archive before/after SDSCs.
3. Generalize the passing all-gather/restickify path into explicit coverage for broadcast, multicast, gather, and all-gather cases.
4. Treat reduce and all-reduce separately because they require arithmetic, not only movement.
5. Keep weight restickifies out of this PR scope; those should be handled by offline weight preloading/layout work.

