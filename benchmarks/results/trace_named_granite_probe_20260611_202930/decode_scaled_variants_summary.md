# Trace-Named Granite Probe

| case | target prefix | rc | matched | kernel ms | emitted normalized kernels | first splits |
|---|---|---:|---:|---:|---|---|
| `decode_mlp_gate_scaled_add` | `sdsc_fused_add_linear_mul_rms_norm_silu` | 0 | False | 2.638 | `sdsc_fused_add_mul_rms_norm_0, sdsc_fused_linear_silu_1` | `sdsc_fused_add_mul_rms_norm_0:{'mb': 32, 'out': 1}; sdsc_fused_add_mul_rms_norm_0:{'mb': 32, 'out': 1}; sdsc_fused_add_mul_rms_norm_0:{'mb': 32, 'out': 1}; sdsc_fused_add_mul_rms_norm_0:{'mb': 32, 'out': 1}` |
| `decode_mlp_down_silu_scaled_add` | `sdsc_fused_add_linear_mul_silu` | 0 | False | 2.758 | `sdsc_fused_add_1_toz_u9if, sdsc_fused_linear_mul_silu_0` | `sdsc_fused_add_1_toz_u9if:{'mb': 4, 'out': 8}; sdsc_fused_linear_mul_silu_0:{'mb': 1, 'out': 25}; sdsc_fused_linear_mul_silu_0:{'mb': 1, 'out': 25}; sdsc_fused_linear_mul_silu_0:{'mb': 1, 'out': 25}` |
