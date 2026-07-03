## test_flash_staged_restickify_allgather_logicalcoords_20260703_013412

run_dir: /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_logicalcoords_20260703_013412
exit_code: 1
backend_plan_count: 32

### Kernel runner / correctness excerpt
```text
43700:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_amax_full_zeros_zeros_like_0 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_logicalcoords_20260703_013412/cache/inductor-spyre/sdsc_fused_amax_full_zeros_zeros_like_0_at34us7v
43701:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_logicalcoords_20260703_013412/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_dc_h77i7
43702:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_div_unsqueeze_2 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_logicalcoords_20260703_013412/cache/inductor-spyre/sdsc_fused_div_unsqueeze_2_wka98xfs
43708:AssertionError: Tensor-likes are not close!
43710:Mismatched elements: 16646918 / 16777216 (99.2%)
43711:Greatest absolute difference: inf at index (0, 0, 0, 0) (up to 0.1 allowed)
43712:Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.1 allowed)
```

## test_flash_staged_restickify_allgather_dataop_schedule_20260703_014630

run_dir: /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_dataop_schedule_20260703_014630
exit_code: 1
backend_plan_count: 32

### Kernel runner / correctness excerpt
```text
43700:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_amax_full_zeros_zeros_like_0 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_dataop_schedule_20260703_014630/cache/inductor-spyre/sdsc_fused_amax_full_zeros_zeros_like_0__wyzpvx2
43701:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_dataop_schedule_20260703_014630/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_5nsqio27
43702:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_div_unsqueeze_2 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_dataop_schedule_20260703_014630/cache/inductor-spyre/sdsc_fused_div_unsqueeze_2_wenjlhuk
43708:AssertionError: Tensor-likes are not close!
43710:Mismatched elements: 16646939 / 16777216 (99.2%)
43711:Greatest absolute difference: inf at index (0, 0, 0, 0) (up to 0.1 allowed)
43712:Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.1 allowed)
```

## test_flash_staged_restickify_allgather_source_layout_fix_20260703_015219

run_dir: /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_source_layout_fix_20260703_015219
exit_code: 1
backend_plan_count: 32

### Kernel runner / correctness excerpt
```text
43700:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_amax_full_zeros_zeros_like_0 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_source_layout_fix_20260703_015219/cache/inductor-spyre/sdsc_fused_amax_full_zeros_zeros_like_0_yuepssui
43701:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_source_layout_fix_20260703_015219/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_arl3utyu
43702:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_div_unsqueeze_2 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_staged_restickify_allgather_source_layout_fix_20260703_015219/cache/inductor-spyre/sdsc_fused_div_unsqueeze_2_eit1wvnw
43708:AssertionError: Tensor-likes are not close!
43710:Mismatched elements: 16646923 / 16777216 (99.2%)
43711:Greatest absolute difference: inf at index (0, 0, 0, 0) (up to 0.1 allowed)
43712:Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.1 allowed)
```

## test_flash_direct_allgather_dataop_schedule_20260703_015843

run_dir: /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_allgather_dataop_schedule_20260703_015843
exit_code: 1
backend_plan_count: 32

### Kernel runner / correctness excerpt
```text
43700:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_amax_full_zeros_zeros_like_0 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_allgather_dataop_schedule_20260703_015843/cache/inductor-spyre/sdsc_fused_amax_full_zeros_zeros_like_0_e6vz8l4g
43701:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_allgather_dataop_schedule_20260703_015843/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_0kky8g70
43702:[INFO] [spyre.inductor.kernel_runner] RUN: sdsc_fused_div_unsqueeze_2 /home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_allgather_dataop_schedule_20260703_015843/cache/inductor-spyre/sdsc_fused_div_unsqueeze_2_typgrfwe
43708:AssertionError: Tensor-likes are not close!
43710:Mismatched elements: 16646933 / 16777216 (99.2%)
43711:Greatest absolute difference: inf at index (0, 0, 0, 0) (up to 0.1 allowed)
43712:Greatest relative difference: nan at index (0, 0, 0, 0) (up to 0.1 allowed)
```

