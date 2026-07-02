# DLDSC Remaining HBM Spill Classes - 2026-07-02

## Scope

This note classifies the remaining non-weight HBM spill evidence in the
Granite/flash DLDSC artifacts. It uses only artifact docs and pod run roots.

Weight/preload restickifies are out of scope. In this note, "in scope" means a
computed activation or runtime operand handoff that currently or historically
round-tripped through HBM and should be expressible as on-chip DLDSC movement.

## Current Best Flash Evidence

The current best CLC run is:

```text
/home/adnan/codex-isolated/dldsc_flash_new_deeptools_20260702_101806/runs/
  flash_validation_deeptools_00a37826_ldfix_20260702_102707
```

Observed run facts:

| Metric | Value |
| --- | ---: |
| return code | 0 |
| wall time | 243s |
| SDSC files | 550 |
| `layout_allgather_restickify` plan files | 32 |
| `ReStickifyOpHBM` rows | 0 |
| `ReStickifyOpLx` rows | 32 |
| `lxRelayoutClassifications_` rows | 32 |

SDSC op counts from `sdsc_files.txt`:

| op | count |
| --- | ---: |
| `mul` | 128 |
| `add` | 96 |
| `sub` | 64 |
| `batchmatmul` | 64 |
| `exp` | 64 |
| `max` | 34 |
| `sum` | 32 |
| `ReStickifyOpLx` | 32 |
| `maximum` | 32 |
| `identity` | 3 |
| `realdiv` | 1 |

Each of the 32 batchmatmul classifications has:

```text
communication_class=all_gather
communication_pattern=layout_allgather_restickify
transfer_count=256
max_fanout=8
max_fanin=8
```

Representative current artifacts:

```text
/home/adnan/codex-isolated/dldsc_flash_new_deeptools_20260702_101806/runs/
  flash_validation_deeptools_00a37826_ldfix_20260702_102707/cache/
  inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_e0o6o5ky/
  sdsc_105.json

/home/adnan/codex-isolated/dldsc_flash_new_deeptools_20260702_101806/runs/
  flash_validation_deeptools_00a37826_ldfix_20260702_102707/
  105_batchmatmul_Tensor1_0_layout_allgather_restickify_plan.json
```

Interpretation: the previously visible flash activation HBM spill class is not
remaining in this run. It is now represented as 32 LX restickifies plus 32
all-gather layout-restickify contracts, and the run exits successfully.

## Flash Baseline Contrast

The older checked-in flash summary is:

```text
docs/results/granite_e2e/flash_contract_20260702/flash_contract_summary.json
```

Its baseline run was:

```text
/home/adnan/codex-isolated/dldsc_runtime_validation_20260702_075517/runs/
  latest_after_zero_stick_optimized_20260702_084153
```

That baseline had:

| Metric | Baseline | Current best CLC |
| --- | ---: | ---: |
| return code | 0 | 0 |
| wall time | 435s | 243s |
| SDSC files | 550 | 550 |
| `ReStickifyOpHBM` rows | 32 | 0 |
| `ReStickifyOpLx` rows | 0 | 32 |
| layout-all-gather classifications | 0 | 32 |
| layout-all-gather plans | 0 | 32 |

The eliminated baseline class is therefore:

```text
class: all_gather
pattern: layout_allgather_restickify
old HBM symptom: 32 ReStickifyOpHBM rows
current state: 0 ReStickifyOpHBM rows; 32 ReStickifyOpLx rows
```

## Classification Table

| Communication class | Remaining non-weight HBM spill evidence | Counts and paths | Current read |
| --- | --- | --- | --- |
| scatter | No counted remaining non-weight HBM spill in inspected current artifacts. | Current flash CLC run has 0 `ReStickifyOpHBM` and no scatter classifications. | One-to-one scatter is not the remaining flash/Granite blocker in these artifacts. |
| broadcast | No separate pure-broadcast HBM spill count was found. | The Granite AV operand issue is described in the reduced artifact as "broadcast/all-gather"; see all_gather row. | Treat as part of the matmul operand all-gather/broadcast materialization gap, not as a standalone broadcast count. |
| multicast | No remaining multicast HBM spill count was found. | No inspected run summary or SDSC classification reports `communication_class=multicast` as a remaining HBM spill. | Backend tests cover multicast-style movement, but these Granite/flash artifacts do not expose it as the residual HBM spill class. |
| gather | No remaining gather HBM spill count was found. | No inspected run summary or SDSC classification reports `communication_class=gather` as a remaining HBM spill. | Gather is not the visible residual class here. |
| all_gather | Yes, historically for flash layout/restickify; current best flash run removes it. Granite still has a separate AV operand broadcast/all-gather materialization issue. | Flash baseline: 32 `ReStickifyOpHBM` rows in `docs/results/granite_e2e/flash_contract_20260702/flash_contract_summary.json`. Current CLC: 32 all-gather classifications, 32 plan files, 0 `ReStickifyOpHBM`. Granite reduced AV proof: `/home/adnan/codex-isolated/comms_collectives_20260629/runs/buf21_small_fit_collective_20260630_031142/BOUNDARY_RESULT.md`. | Flash class is fixed in the current CLC evidence. Granite AV remains the non-flash class to solve for full prefill speedup; the artifact calls it matmul operand broadcast/all-gather, not scatter. |
| reduce/all_reduce | No remaining DLDSC HBM spill class found. | Current flash has `sum`/`max` compute rows, but no reduce/all_reduce relayout classifications. Existing docs say reduce/all_reduce are outside this copy-relayout path. | Not in scope for this DLDSC relayout lane; value-combining collectives need separate semantics. |
| layout/restickify | Current flash has no remaining HBM restickify spill. Older Granite one-layer prefill has one in-scope computed activation LX-to-HBM restickify row; other HBM-to-HBM restickifies are out of scope. | Flash current CLC: 0 `ReStickifyOpHBM`, 32 `ReStickifyOpLx`. Granite in-scope row: `/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_layout_restickify_class_20260630_050148/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_jm0s3oi3/sdsc_9.json` (`lx -> hbm`, has LX core residency). | Flash layout/restickify is no longer remaining in the current best run. The older Granite activation row is still useful as a pre-flash-contract baseline symptom. |

## Granite Non-Flash Evidence

The strongest Granite-specific non-flash artifact is the reduced `buf21`
boundary run:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/
  buf21_small_fit_collective_20260630_031142/BOUNDARY_RESULT.md
```

It preserves the important attention value operand mismatch:

```text
Tensor1 producer residency: out sharded across 32 cores
Consumer AV matmul compute: mb sharded across 32 cores
Tensor1 layout dimensions: out,in,x
```

The reduced full Tensor1 operand is only 4096 bytes, and DXP compiles it
successfully. The note's conclusion is that full Granite prefill fails because
current relayout insertion materializes too much of the value operand per
consumer core instead of staging/broadcasting it through the matmul transfer
loop.

Concrete count evidence:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/
  buf21_small_fit_collective_20260630_031142/buf21_batchmatmul.json
```

contains one reduced `16_batchmatmul` SDSC for this boundary case.

Related stage sweep:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/
  dxp_nonpaired_stage_sweep_clc_20260630_135558/summary.txt
```

shows the same bundle can be staged with:

| stage | dataops | rc |
| ---: | ---: | ---: |
| 1 | 32 | 0 |
| 2 | 16 | 0 |
| 4 | 8 | 0 |
| 8 | 4 | 0 |
| 16 | 2 | 0 |
| 32 | 1 | 0 |

This is the remaining Granite communication shape with concrete non-flash
evidence. It should be treated as an all-gather/broadcast matmul-operand
materialization problem, not as another scatter relayout.

## Older Granite Restickify Rows

The older one-layer Granite prefill CLC run:

```text
/home/adnan/codex-isolated/comms_collectives_20260629/runs/
  granite_prefill_layout_restickify_class_20260630_050148
```

has a passing block run and profiler summary:

```text
block_prefill/result.json
block_prefill/summary.md
block_prefill/trace_summary.json
```

The `result.json` cache summary reports five `ReStickifyOpHBM` rows across the
four generated kernels. Raw SDSC inspection separates them as:

| Row | Components | Scope |
| --- | --- | --- |
| `.../sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_jm0s3oi3/sdsc_9.json` | `lx -> hbm`, has LX core residency | in-scope computed activation layout/restickify symptom |
| `.../sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_pn321070/sdsc_0.json` | `hbm -> hbm`, no LX core residency | out-of-scope weight/preload-style restickify |
| `.../sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_pn321070/sdsc_10.json` | `hbm -> hbm`, no LX core residency | out-of-scope weight/preload-style restickify |
| `.../sdsc_fused_add_linear_mul_3_q_g6wv_l/sdsc_0.json` | `hbm -> hbm`, no LX core residency | out-of-scope weight/preload-style restickify |
| `.../sdsc_fused_linear_rms_norm_0_vw3dbz1e/sdsc_7.json` | `hbm -> hbm`, no LX core residency | out-of-scope weight/preload-style restickify |

Those four HBM-to-HBM rows should not be counted as remaining non-weight HBM
spills for the DLDSC communication substrate. They belong to the weight/preload
layout lane.

## Bottom Line

Current best evidence says the flash activation layout/restickify HBM spill is
gone: 32 old `ReStickifyOpHBM` rows are now 32 `ReStickifyOpLx` rows with 32
all-gather layout-restickify classifications and a passing CLC run.

The remaining non-flash Granite communication class with concrete evidence is
the attention value matmul operand broadcast/all-gather materialization issue.
The reduced `buf21` artifacts prove the shape and capacity boundary, but the
full Granite prefill path still needs a staged matmul-operand movement strategy
rather than full per-consumer materialization.
