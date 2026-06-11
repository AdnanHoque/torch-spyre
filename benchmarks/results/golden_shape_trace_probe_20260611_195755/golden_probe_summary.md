# Golden Granite Shape And Trace Probe

- started: `2026-06-11T20:14:21Z`
- finished: `2026-06-11T20:16:23Z`
- run root: `/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/golden_shape_trace_probe_20260611_195755/direct_full`

## What This Probe Checks

- Golden-shape rows exercise the shape/split inventory as standalone matmul/BMM kernels.
- Fused-context rows put the same MLP/attention families behind Granite-like fusion.
- Neighborhood rows perturb the main target shapes to catch overfitting to one point.
- Trace-conformance checks whether emitted kernel names are in Antoni's e2e trace inventory.

## Results

| group | task | role | rc | kernel ms | PT util | expected split | emitted split | split match | emitted kernel dirs | trace-name subset | trace weight |
|---|---|---|---:|---:|---:|---|---|---:|---|---:|---:|
| golden-shape | `golden_prefill_qk_t_scores_512x32x128_x_512` | prefill QK^T scores | 0 | 1.646 | 1.810 | `x4,mb1,out8,in1` | `x1,mb4,out8,in1` | no | `sdsc_fused_bmm_0_k30vlt_2` | no | 0 |
| golden-shape | `golden_prefill_attn_v_32x512x512_x_128` | prefill attn @ V | 0 | 0.778 | 3.827 | `x1,mb32,out1,in1` | `x1,mb32,out1,in1` | yes | `sdsc_fused_bmm_0` | no | 0 |
| golden-shape | `golden_prefill_mlp_gate_up_w1_w3_1x512x4096_x_12800` | prefill MLP gate+up (w1,w3) | 0 | 1.973 | 37.740 | `mb4,out8,in1` | `mb4,out8,in1` | yes | `sdsc_fused_0_c` | no | 0 |
| golden-shape | `golden_prefill_mlp_down_w2_1x512x12800_x_4096` | prefill MLP down (w2) | 0 | 1.611 | 46.226 | `mb4,out8,in1` | `mb8,out4,in1` | no | `sdsc_fused_0` | no | 0 |
| golden-shape | `golden_prefill_q_o_proj_1x512x4096_x_4096` | prefill Q/O proj | 0 | 0.547 | 43.604 | `mb4,out8,in1` | `mb8,out4,in1` | no | `sdsc_fused_0` | no | 0 |
| golden-shape | `golden_prefill_k_v_proj_gqa_1x512x4096_x_1024` | prefill K,V proj (GQA) | 0 | 0.126 | 47.182 | `mb4,out8,in1` | `mb8,out4,in1` | no | `sdsc_fused_0_1287u8_0` | no | 0 |
| golden-shape | `golden_decode_qk_t_scores_64x32x128_x_576` | decode QK^T scores | 0 | 0.237 | 1.770 | `x32,mb1,out1,in1` | `x32,mb1,out1,in1` | yes | `sdsc_fused_bmm_0` | no | 0 |
| golden-shape | `golden_decode_attn_v_32x64x576_x_128` | decode attn @ V | 0 | 0.337 | 1.243 | `x1,mb16,out2,in1` | `x1,mb32,out1,in1` | no | `sdsc_fused_bmm_0` | no | 0 |
| golden-shape | `golden_decode_mlp_gate_up_w1_w3_1x64x4096_x_12800` | decode MLP gate+up (w1,w3) | 0 | 0.740 | 12.580 | `mb4,out8,in1` | `mb4,out8,in1` | yes | `sdsc_fused_0` | no | 0 |
| golden-shape | `golden_decode_q_o_proj_1x64x4096_x_4096` | decode Q/O proj | 0 | 0.246 | 12.090 | `mb4,out8,in1` | `mb4,out8,in1` | yes | `sdsc_fused_0` | no | 0 |
| golden-shape | `golden_decode_mlp_down_w2_1x64x12800_x_4096` | decode MLP down (w2) | 0 | 0.765 | 12.175 | `mb4,out8,in1` | `mb4,out8,in1` | yes | `sdsc_fused_0` | no | 0 |
| golden-shape | `golden_decode_k_v_proj_gqa_1x64x4096_x_1024` | decode K,V proj (GQA) | 0 | 0.071 | 10.513 | `mb4,out8,in1` | `mb4,out8,in1` | yes | `sdsc_fused_0` | no | 0 |
| fused-context | `fused_prefill_mlp_gate_rms_linear_silu` | prefill MLP gate/up in RMSNorm+linear+SiLU context | 0 | 1.660 | 44.872 | `` | `mb32,out1` | no | `sdsc_fused_mul_rms_norm_silu_0` | no | 0 |
| fused-context | `fused_prefill_full_glu_chain` | prefill full GLU chain | 0 | 5.061 | 44.144 | `` | `mb8,out4,in1` | no | `sdsc_fused_add_2, sdsc_fused_add_mul_rms_norm_0, sdsc_fused_mul_silu_1` | no | 0 |
| fused-context | `fused_decode_mlp_gate_rms_linear_silu` | decode padded MLP gate/up in add+RMSNorm+linear+SiLU context | 0 | 0.810 | 11.487 | `` | `mb32,out1` | no | `sdsc_fused_add_0, sdsc_fused_mul_rms_norm_silu_1` | no | 0 |
| fused-context | `fused_decode_full_glu_chain` | decode padded full GLU chain | 0 | 2.327 | 12.000 | `` | `mb4,out8,in1` | no | `sdsc_fused_add_2, sdsc_fused_add_mul_rms_norm_0, sdsc_fused_mul_silu_1` | no | 0 |
| fused-context | `fused_decode_attention_cat_qk` | decode attention cat-cache QK score context | 1 | N/A | N/A | `` | `` | no | `` | no | 0 |
| neighborhood | `neighborhood_prefill_mlp_gate_m256` | prefill MLP gate/up neighborhood M=256 | 0 | 1.291 | 28.854 | `` | `mb4,out8,in1` | no | `sdsc_fused_0` | no | 0 |
| neighborhood | `neighborhood_prefill_mlp_gate_m512` | prefill MLP gate/up neighborhood M=512 | 0 | 1.965 | 37.903 | `` | `mb4,out8,in1` | no | `sdsc_fused_0` | no | 0 |
| neighborhood | `neighborhood_prefill_mlp_gate_m1024` | prefill MLP gate/up neighborhood M=1024 | 0 | 3.784 | 39.362 | `` | `mb4,out8,in1` | no | `sdsc_fused_0` | no | 0 |
| neighborhood | `neighborhood_decode_mlp_gate_m1` | decode MLP gate/up neighborhood M=1 | 0 | 0.858 | 0.169 | `` | `out25,in1` | no | `sdsc_fused_0` | no | 0 |
| neighborhood | `neighborhood_decode_mlp_gate_m64` | decode MLP gate/up neighborhood M=64 | 0 | 0.732 | 12.721 | `` | `mb4,out8,in1` | no | `sdsc_fused_0` | no | 0 |
| neighborhood | `neighborhood_decode_attention_qk_n512` | decode QK score neighborhood cache N=512 | 0 | 0.393 | 0.947 | `` | `x1,mb32,out1` | no | `sdsc_fused_bmm_transpose_0` | no | 0 |
| neighborhood | `neighborhood_decode_attention_qk_n576` | decode QK score neighborhood cache N=576 | 0 | 0.540 | 0.776 | `` | `x1,mb32,out1` | no | `sdsc_fused_bmm_transpose_0` | no | 0 |
| neighborhood | `neighborhood_decode_attention_qk_n1024` | decode QK score neighborhood cache N=1024 | 0 | 0.963 | 0.774 | `` | `x1,mb32,out1` | no | `sdsc_fused_bmm_transpose_0` | no | 0 |

## Read

- Failed tasks: **1**.
- Tasks that completed but hit perf-suite custom-op PT post-processing failure: **0**.
- Golden-shape split mismatches: **5**.
- Tasks whose kernel names hit Antoni's trace inventory: **0**.

A standalone golden shape is useful for optimizer targeting, but a low trace-name hit rate means the result should not be treated as e2e-representative by itself. Fused-context rows are the bridge: if those rows move differently from the standalone rows, tuning only to the shape table can mislead us.

## Artifacts

- Per-task `perf.txt`, stdout/stderr, `result.json`, and cache/export directories live under the task subdirectories.
- `all_results.json` contains the full machine-readable summary.
