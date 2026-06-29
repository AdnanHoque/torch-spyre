# PR1 Scatter Review And Artifact Update - 2026-06-29

This is the dev-pf artifact update for the `pr-lx-relayout-scatter` Torch branch and the `pr-lx-relayout-dldsc-scatter` Deeptools patch. It records the reviewer trim, the benchmark evidence, the HBM spill mapping, and the local run hacks needed to reproduce the result from a fresh pod.

## Branches And SHAs

| repo | branch | sha |
|---|---|---|
| Torch | AdnanHoque/torch-spyre:pr-lx-relayout-scatter | fbe79b8862aaebd80bca204a2c7b24a831e9102d |
| Deeptools | Adnan-Hoque1/deeptools:pr-lx-relayout-dldsc-scatter | b8c09743c46505b4cac46b434b9eb3243ae0b685 |

## Reviewer Pass Summary

- Torch PR branch was trimmed from 776 insertions to 666 insertions by removing unused unsupported-movement/future-work metadata from the PR payload. The communication taxonomy is kept in artifact docs instead.
- Torch PR branch is code-only: no docs are present in the diff. It has one signed-off commit authored/committed as `Adnan Hoque <adnan.hoque1@ibm.com>`.
- Deeptools patch remains one signed-off commit, 67 insertions across `dxp/SdscRelayoutInsertion.cpp` and `ddc/ddc_fold.cpp`; no debug instrumentation or benchmark-only code is included.

## Focused Validation

| gate | result |
|---|---|
| Torch unit tests | `tests/inductor/test_lx_relayout_dldsc.py`: 5 passed in 6.99s after the reservation fix |
| Deeptools standalone fixture | `dxp/test/test_core_work_div_incompt` accepted by patched `dxp_standalone` with `DXP_LX_FRAC_AVAIL=1` |
| Granite causal prefill | `B=1,S=512,E=4096` full Torch LX + backend split wrapper + local GraphEditor fix passes and produces 1.223x kernel speedup |

## Performance Summary

| variant | status | kernel_ms_per_iter | median_wall_ms | kernel_speedup | wall_speedup |
|---|---|---|---|---|---|
| baseline_off | pass | 14.697693 | 34.857512 | 1.000000x | 1.000000x |
| boundary_relayout_lxfrac_0p2_graphfix | pass | 14.581861 | 34.120083 | 1.007944x | 1.021613x |
| boundary_full_torch_lx_backend1_graphfix | pass | 12.014579 | 31.895638 | 1.223322x | 1.092861x |

## Communication-Class Counts

| comm_class | baseline_off | boundary_relayout_lxfrac_0p2_graphfix | boundary_full_torch_lx_backend1_graphfix |
|---|---|---|---|
| hbm_input_roundtrip_candidate | 5 | 5 | 3 |
| hbm_kernel_operand | 5 | 5 | 5 |
| hbm_output_spill | 5 | 5 | 5 |
| lx_input_same_view | 1 | 1 | 1 |
| lx_output | 1 | 1 | 1 |
| missing_matmul_operand_collective | 1 | 1 | 1 |
| scatter | 0 | 0 | 2 |

## Key Before/After SDSC Rows

| variant | kernel | sdsc | n_shape | compute_split | tensor | role | component | layout | coord_core_map_len | comm_class | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_off | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1 | sdsc_8 | x_=512,mb_=32,out_=512,in_=128 | out:2,x:16 | Tensor0 | INPUT | hbm | mb,in,x | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| baseline_off | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1 | sdsc_8 | x_=512,mb_=32,out_=512,in_=128 | out:2,x:16 | Tensor2 | OUTPUT | hbm | out,x,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| baseline_off | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1 | sdsc_16 | x_=32,mb_=512,out_=128,in_=512 | mb:32 | Tensor1 | KERNEL | hbm | out,in,x | 0 | missing_matmul_operand_collective | attention value operand remains HBM; needs all-gather/replicate-style class, not PR1 scatter |
| baseline_off | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2 | sdsc_1 | mb_=512,out_=4096,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | hbm | in,mb | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| baseline_off | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2 | sdsc_1 | mb_=512,out_=4096,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| baseline_off | sdsc_fused_add_linear_mul_silu_split_with_sizes_3 | sdsc_1 | mb_=512,out_=25600,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| baseline_off | sdsc_fused_add_linear_mul_silu_split_with_sizes_3 | sdsc_1 | mb_=512,out_=25600,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| baseline_off | sdsc_fused_add_linear_mul_silu_split_with_sizes_3 | sdsc_5 | mb_=512,out_=4096,in_=12800 | mb:8,out:4 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| baseline_off | sdsc_fused_add_linear_mul_silu_split_with_sizes_3 | sdsc_5 | mb_=512,out_=4096,in_=12800 | mb:8,out:4 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| baseline_off | sdsc_fused_linear_rms_norm_0 | sdsc_7 | mb_=512,out_=6144,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| baseline_off | sdsc_fused_linear_rms_norm_0 | sdsc_7 | mb_=512,out_=6144,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| boundary_full_torch_lx_backend1_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1 | sdsc_10 | x_=512,mb_=32,out_=512,in_=128 | out:2,x:16 | Tensor0 | INPUT | lx | mb,in,x | 32 | scatter | producer LX residency differs from consumer compute; backend inserts resident LX relayout |
| boundary_full_torch_lx_backend1_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1 | sdsc_10 | x_=512,mb_=32,out_=512,in_=128 | out:2,x:16 | Tensor2 | OUTPUT | hbm | out,x,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| boundary_full_torch_lx_backend1_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1 | sdsc_18 | x_=32,mb_=512,out_=128,in_=512 | mb:32 | Tensor1 | KERNEL | hbm | out,in,x | 0 | missing_matmul_operand_collective | attention value operand remains HBM; needs all-gather/replicate-style class, not PR1 scatter |
| boundary_full_torch_lx_backend1_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2 | sdsc_1 | mb_=512,out_=4096,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | lx | in,mb | 32 | scatter | producer LX residency differs from consumer compute; backend inserts resident LX relayout |
| boundary_full_torch_lx_backend1_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2 | sdsc_1 | mb_=512,out_=4096,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| boundary_full_torch_lx_backend1_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2 | sdsc_11 | mb_=512,out_=25600,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| boundary_full_torch_lx_backend1_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2 | sdsc_11 | mb_=512,out_=25600,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| boundary_full_torch_lx_backend1_graphfix | sdsc_fused_add_linear_mul_3 | sdsc_1 | mb_=512,out_=4096,in_=12800 | mb:8,out:4 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| boundary_full_torch_lx_backend1_graphfix | sdsc_fused_add_linear_mul_3 | sdsc_1 | mb_=512,out_=4096,in_=12800 | mb:8,out:4 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| boundary_full_torch_lx_backend1_graphfix | sdsc_fused_linear_rms_norm_0 | sdsc_8 | mb_=512,out_=6144,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| boundary_full_torch_lx_backend1_graphfix | sdsc_fused_linear_rms_norm_0 | sdsc_8 | mb_=512,out_=6144,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| boundary_relayout_lxfrac_0p2_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1 | sdsc_10 | x_=512,mb_=32,out_=512,in_=128 | out:2,x:16 | Tensor0 | INPUT | hbm | mb,in,x | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| boundary_relayout_lxfrac_0p2_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1 | sdsc_10 | x_=512,mb_=32,out_=512,in_=128 | out:2,x:16 | Tensor2 | OUTPUT | hbm | out,x,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| boundary_relayout_lxfrac_0p2_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1 | sdsc_18 | x_=32,mb_=512,out_=128,in_=512 | mb:32 | Tensor1 | KERNEL | hbm | out,in,x | 0 | missing_matmul_operand_collective | attention value operand remains HBM; needs all-gather/replicate-style class, not PR1 scatter |
| boundary_relayout_lxfrac_0p2_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2 | sdsc_1 | mb_=512,out_=4096,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | hbm | in,mb | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| boundary_relayout_lxfrac_0p2_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2 | sdsc_1 | mb_=512,out_=4096,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| boundary_relayout_lxfrac_0p2_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2 | sdsc_11 | mb_=512,out_=25600,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| boundary_relayout_lxfrac_0p2_graphfix | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2 | sdsc_11 | mb_=512,out_=25600,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| boundary_relayout_lxfrac_0p2_graphfix | sdsc_fused_add_linear_mul_3_8qgzwno_ | sdsc_1 | mb_=512,out_=4096,in_=12800 | mb:8,out:4 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| boundary_relayout_lxfrac_0p2_graphfix | sdsc_fused_add_linear_mul_3_8qgzwno_ | sdsc_1 | mb_=512,out_=4096,in_=12800 | mb:8,out:4 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| boundary_relayout_lxfrac_0p2_graphfix | sdsc_fused_linear_rms_norm_0 | sdsc_8 | mb_=512,out_=6144,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input is read from HBM |
| boundary_relayout_lxfrac_0p2_graphfix | sdsc_fused_linear_rms_norm_0 | sdsc_8 | mb_=512,out_=6144,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |

## Which HBM Spills Were Removed

- PR1 removes two resident input round trips in this Granite prefill artifact. In the baseline these are `INPUT (hbm)` rows on `batchmatmul` consumers; in the optimized SDSCs they become `INPUT (lx)` rows with non-empty allocation `coreIdToWkSlice_`, which is the dl-dsc coordinate contract that triggers Deeptools internal `LxRelayout` / `STCDPOpLx`.
- The two proven removals are: the attention fused region `sdsc_10` input (`mb/in/x` layout, `out:2,x:16` compute split), and the post-attention/projection fused region `sdsc_1` input (`in/mb` layout, `mb:4,out:8` compute split). These are activation handoffs, not static weights.
- Output HBM materializations still remain in this PR1 artifact, and the static weight restickifies remain intentionally out of scope because separate preloading work should own them.
- The remaining high-value attention gap is the value-side matmul operand, classified here as `missing_matmul_operand_collective`. It is not a resident scatter case: consumer cores need a replicated/all-gathered operand view, so a future communication class is required.

## Local Benchmarking Hacks

- Use empty/fake Spyre weights from `spyre-granite-e2e-bench` so benchmarking measures module performance without loading real model weights.
- Overlay the profiler-enabled `torch_spyre._C.so` from `/home/adnan/dt-inductor/torch-spyre/torch_spyre/_C.so`.
- Set `TORCH_DEVICE_BACKEND_AUTOLOAD=0`; import `torch_spyre` and call `_autoload()` explicitly from the benchmark path.
- Put `/opt/ibm/spyre/runtime/lib` before `/opt/ibm/spyre/spyre-comms/lib` in `LD_LIBRARY_PATH`.
- Use the DXP split wrapper for full frontend LX tests: Torch sees `DXP_LX_FRAC_AVAIL=0`, while DXP sees `DXP_BACKEND_LX_FRAC_AVAIL=1` remapped to `DXP_LX_FRAC_AVAIL=1`. Direct backend `DXP_LX_FRAC_AVAIL=0` fails on both pods with initial chunk/LX capacity errors.
- Apply the local-only GraphEditor `ReinterpretView` wrapper-preservation fix for the boundary-clone Granite path; the patch is archived next to this note and is not part of the PR branch.

## Future Communication Primitive Roadmap

- PR1: resident scatter, where each destination resident slice can be materialized from producer-owned LX slices.
- PR2 candidate: matmul operand broadcast/all-gather/replicate for attention value operands and similar non-primary matmul inputs. This needs a contract that says the consumer needs an operand view replicated or gathered across a different split, not a full resident materialization per core.
- Later: reduction-aware movement, where movement must occur after partial reductions; layout-changing restickify/reformat movement; and overlapped scheduling so movement can pipeline with compute instead of acting as a blocking pre-consumer materialization.
