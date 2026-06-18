# Granite Block Probe vs Antoni Trace

This note compares the local Granite block probe in `benchmarks/granite_block_probe.py` with Antoni's Granite e2e Kineto traces:

- `aviros-spyre-test_1673899.1780607390507520214.pt.trace.json`
- `aviros-spyre-test_52053.1781021126249019455.pt.trace.json`

The trace files provide launch kernel names and counts. They do not include the full SDSC subop list, so exact gate/up co-residency in Antoni's run requires the generated `sdsc_*.json` cache from that run. The local probe does emit SDSC JSONs, so we can inspect subops there directly.

## Prefill Kernel Name Match

The local Granite block prefill probe emits the same main prefill kernel families seen in Antoni's trace.

| role | Antoni trace kernel | local Granite block kernel | read |
|---|---|---|---|
| attention input norm + Q/K/V projections | `sdsc_fused_linear_mul_rms_norm_sum_unsqueeze_view_0` | `sdsc_fused_linear_mul_rms_norm_sum_unsqueeze_view_0` | exact family match |
| prefill SDPA / attention core | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_1` | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_1` | exact family match |
| attention output + residual / next norm setup | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_transpose_unsqueeze_view_2` | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_transpose_unsqueeze_view_2` | exact family match |
| MLP gate/up projection + SiLU | `sdsc_fused_linear_mul_rms_norm_silu_3` | `sdsc_fused_linear_mul_rms_norm_silu_3` | exact family match |
| MLP down projection + residual | `sdsc_fused_add_linear_mul_4` | `sdsc_fused_add_linear_mul_4` | exact family match |

## Are Gate And Up Fused?

For the local Granite block prefill probe, yes. The emitted `sdsc_fused_linear_mul_rms_norm_silu_3` SDSC contains two `batchmatmul` subops:

| local SDSC | subop | split |
|---|---|---|
| `sdsc_fused_linear_mul_rms_norm_silu_3` | `5_batchmatmul` | `{'mb': 4, 'out': 8, 'in': 1}` |
| `sdsc_fused_linear_mul_rms_norm_silu_3` | `11_batchmatmul` | `{'mb': 4, 'out': 8, 'in': 1}` |

That is the expected gate/up pair, with SiLU/mul fused in the same kernel family.

For Antoni's trace, the kernel name `sdsc_fused_linear_mul_rms_norm_silu_3` strongly indicates the same fused MLP gate/up family, and the launch name matches our probe. The trace alone does not expose subops, so the exact statement "two batchmatmuls are in the same Antoni SDSC" should be made only after checking Antoni's SDSC JSON.

## Decode Probe Match And Limits

The local decode probes are useful for no-regression checks, but they are not a perfect launch-name reproduction of Antoni's full e2e decode trace.

The attention decode probe emits fused SDPA families with the same general shape and naming style as Antoni's decode trace, including `scaled_dot_product_fused_attention_overrideable` kernels and fused projection/norm helpers. The MLP decode probe emits fused MLP kernels, but with shorter probe-local names:

| probe area | local decode kernel family | relation to Antoni trace |
|---|---|---|
| MLP decode | `sdsc_fused_linear_rms_norm_silu_0` | fused MLP/norm/silu probe family, not exact Antoni e2e launch name |
| MLP decode | `sdsc_fused_add_linear_mul_1` | fused add/linear/mul probe family, not exact Antoni e2e launch name |
| attention decode | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_*` | same fused SDPA family, exact suffixes differ by cache/update path |

Use these decode probes as an A/B guard for the cost-model change, not as a claim that the local probe reproduces every Antoni e2e decode launch name exactly.

## Measured A/B Summary

All measurements were taken on the pod using the same probe harness, with isolated cache directories. The absolute wall times include runtime and synchronization effects, so the important signal is the paired A/B.

| probe | baseline `cost-model-physics` | general-tile cost model | read |
|---|---:|---:|---|
| full block prefill, `M=512` | `524.610 ms` | `493.336 ms` | prefill improves about `1.06x` |
| MLP decode, `M=64` | `8.858 ms` | `8.985 ms` | effectively neutral; same splits |
| attention decode, `M=64` | `74.993 ms` | `75.188 ms` | effectively neutral; same splits |

The attention decode runs both emitted a runtime-stream timeout warning before completing. Since the warning appeared symmetrically on both sides and the emitted splits matched, this is still useful as a no-regression comparison, but the absolute attention decode wall time should not be over-interpreted.

Raw result lines are archived in `raw_probe_results.txt`.

