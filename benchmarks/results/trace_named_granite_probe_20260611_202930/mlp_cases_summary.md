# Trace-Named Granite Probe

| case | target prefix | rc | matched | kernel ms | emitted normalized kernels | first splits |
|---|---|---:|---:|---:|---|---|
| `prefill_mlp_gate` | `sdsc_fused_linear_mul_rms_norm_silu` | 0 | True | 3.502 | `sdsc_fused_linear_mul_rms_norm_silu_0_g5_eyoj9` | `sdsc_fused_linear_mul_rms_norm_silu_0_g5_eyoj9:{'mb': 32, 'out': 1}; sdsc_fused_linear_mul_rms_norm_silu_0_g5_eyoj9:{'mb': 32, 'out': 1}; sdsc_fused_linear_mul_rms_norm_silu_0_g5_eyoj9:{'out': 32, 'x': 1}; sdsc_fused_linear_mul_rms_norm_silu_0_g5_eyoj9:{'out': 32, 'x': 1}` |
| `decode_mlp_gate_add` | `sdsc_fused_add_linear_mul_rms_norm_silu` | 0 | False | 2.626 | `sdsc_fused_add_0, sdsc_fused_linear_mul_rms_norm_silu_1` | `sdsc_fused_add_0:{'mb': 32, 'out': 1}; sdsc_fused_linear_mul_rms_norm_silu_1:{'mb': 32, 'out': 1}; sdsc_fused_linear_mul_rms_norm_silu_1:{'mb': 32, 'out': 1}; sdsc_fused_linear_mul_rms_norm_silu_1:{'out': 32, 'x': 1}` |
| `prefill_mlp_down` | `sdsc_fused_add_linear_mul` | 0 | True | 4.035 | `sdsc_fused_add_linear_mul_0_8` | `sdsc_fused_add_linear_mul_0_8:{'mb': 1, 'out': 25}; sdsc_fused_add_linear_mul_0_8:{'mb': 8, 'out': 4, 'in': 1}; sdsc_fused_add_linear_mul_0_8:{'mb': 8, 'out': 4}; sdsc_fused_add_linear_mul_0_8:{'mb': 8, 'out': 4}` |
| `decode_mlp_down_silu` | `sdsc_fused_add_linear_mul_silu` | 0 | False | 2.756 | `sdsc_fused_add_1, sdsc_fused_linear_mul_silu_0` | `sdsc_fused_add_1:{'mb': 4, 'out': 8}; sdsc_fused_linear_mul_silu_0:{'mb': 1, 'out': 25}; sdsc_fused_linear_mul_silu_0:{'mb': 1, 'out': 25}; sdsc_fused_linear_mul_silu_0:{'mb': 1, 'out': 25}` |
