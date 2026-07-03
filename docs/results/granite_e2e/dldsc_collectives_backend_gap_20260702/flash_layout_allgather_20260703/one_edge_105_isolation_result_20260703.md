# Flash Layout-Allgather One-Edge Isolation - 2026-07-03

## Purpose

This isolates the current `layout_allgather_restickify` correctness failure by enabling exactly one backend-materialized relayout edge in the full flash attention bundle.

The env filter used for this diagnostic was:

```bash
DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_ONLY_SDSC=105_batchmatmul
```

That keeps the Torch classification/planning path unchanged, but asks the experimental Deeptools path to materialize only the `105_batchmatmul` layout-allgather relayout. All other planned layout-allgather relayouts are skipped by this diagnostic filter.

## Run

- Pod: `adnan-cdx-spyre-dev-pf`
- Run directory: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_allgather_only105_20260703_025006`
- Script: `test-spyre-scripts/test_flash.py`
- DXP split-env wrapper: `DXP_LX_FRAC_AVAIL=0` for Torch, `DXP_BACKEND_LX_FRAC_AVAIL=1` for DXP subprocess
- Backend plan count emitted: 1
- Emitted plan: `105_batchmatmul_Tensor1_0_layout_allgather_restickify_plan.json`

## Result

DXP compilation and AIU runtime launch succeeded. The failure is numerical correctness after execution, not a DXP import or runtime crash.

```text
[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_amax_full_zeros_zeros_like_0 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_allgather_only105_20260703_025006/cache/inductor-spyre/sdsc_fused_amax_full_zeros_zeros_like_0_bwu6bs69
[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_allgather_only105_20260703_025006/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_7etom2h_
[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_div_unsqueeze_2 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_allgather_only105_20260703_025006/cache/inductor-spyre/sdsc_fused_div_unsqueeze_2_c9_1l0ck
Mismatched elements: 15210624 / 16777216 (90.7%)
Greatest absolute difference: inf at index (0, 0, 1, 0) (up to 0.1 allowed)
Greatest relative difference: nan at index (0, 0, 1, 0) (up to 0.1 allowed)
```

## Interpretation

One active layout-allgather relayout edge is enough to corrupt the final flash result:

- one-edge run: `15210624 / 16777216` mismatches, `90.7%`, with `inf`
- prior all-32-edge run: roughly `99.2%` mismatches, with `inf`

This narrows the bug away from purely cumulative interaction across the 32 generated relayouts. The descriptor or physicalization for a single `ReStickifyOpLx -> batchmatmul KERNEL operand` layout-allgather edge is still value-wrong.

The next useful isolation is to make a deterministic standalone value test for this exact 32-core grouped all-gather, because existing `DataOpStandalone --ddsc-test-spec` cannot directly express this flash coordinate mapping and `senpcfg -f` cannot load multi-core LX input maps in this path.

## Included Artifacts

- `one_edge_105_command.sh`: exact run command and env
- `one_edge_105_plan.json`: Torch-emitted backend plan for this edge
- `deeptools_allgather_one_edge_filter_experiment.patch`: current Deeptools experimental patch including the diagnostic filter
- `deeptools_allgather_one_edge_filter_experiment_diff_stat.txt`: patch stat
