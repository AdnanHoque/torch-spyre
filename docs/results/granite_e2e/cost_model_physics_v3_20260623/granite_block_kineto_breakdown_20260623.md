# Granite Block Kineto And Split Breakdown

Run root:

```text
/home/adnan/dt-inductor/profiler_runs/granite_block_cost_model_isolated_20260623_224926
```

This compares:

- baseline: current main plus the same local Granite compile prerequisite used on both sides
- candidate: the same baseline plus the cost-model `work_division.py`

The benchmark is the one-layer FMS Granite block from
`https://github.ibm.com/Adnan-Hoque1/spyre-granite-e2e-bench`.
Coordinate-remap content in that repo is unrelated to this measurement.

Split notation in this file uses planner names:

```text
SDSC x   -> b = batch/head-like split
SDSC mb  -> m = token/row split
SDSC out -> n = output-column split
SDSC in  -> k = reduction split
```

## Kineto Summary

The profiler build records kernel event names as path labels, not SDSC names.
The per-kernel table below maps Kineto launch-order buckets back to the
generated SDSC inventory.

| regime | baseline wall median ms | candidate wall median ms | baseline kernel ms/iter | candidate kernel ms/iter | kernel speedup |
|---|---:|---:|---:|---:|---:|
| prefill | 27.867 | 28.467 | 16.149 | 16.574 | 0.974x |
| decode_expand | 18.868 | 14.880 | 14.765 | 10.621 | 1.390x |

The local block profile supports the decode story strongly. Prefill is flat in
this empty-weight one-block harness, so Antoni's full e2e run remains the
aggregate source of truth for the prefill improvement.

## Prefill Kernel Buckets

| launch | kernel name | function / role | baseline key split(s) | baseline mean ms | candidate key split(s) | candidate mean ms | delta ms |
|---:|---|---|---|---:|---|---:|---:|
| 1 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1` | SDPA value path: attention probabilities @ V plus surrounding attention pointwise/reduction work | `attn@V bmm: {b:1,m:32,n:1,k:1}` | 1.901 | `attn@V bmm: {b:1,m:32,n:1,k:1}` | 1.895 | -0.006 |
| 2 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2` | Attention output path: O projection fused with residual / RMS-norm / layout work | `O-proj: {m:4,n:8,k:1}` | 1.773 | `O-proj: {m:4,n:8,k:1}` | 1.990 | +0.217 |
| 3 | `sdsc_fused_add_linear_mul_silu_split_with_sizes_3` | MLP block: fused gate/up projection, SiLU/mul, and down projection | `gate+up: {m:4,n:8,k:1}; down: {m:4,n:8,k:1}` | 1.450 | `gate+up: {m:4,n:8,k:1}; down: {m:8,n:4,k:1}` | 1.446 | -0.004 |
| 4 | `sdsc_fused_linear_rms_norm_0` | Input norm + fused QKV projection | `QKV: {m:4,n:8,k:1}` | 11.025 | `QKV: {m:4,n:8,k:1}` | 11.243 | +0.218 |

## Decode Kernel Buckets

| launch | kernel name | function / role | baseline key split(s) | baseline mean ms | candidate key split(s) | candidate mean ms | delta ms |
|---:|---|---|---|---:|---|---:|---:|
| 1 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_linear_split_with_sizes_transpose_unsqueeze_view_4` | Attention value path plus O projection in decode context | `attn@V: {b:1,m:32,n:1,k:1}; O-proj: {m:32,n:1,k:1}` | 1.336 | `attn@V: {b:8,m:4,n:1,k:1}; O-proj: {m:4,n:8,k:1}` | 1.283 | -0.053 |
| 2 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_transpose_unsqueeze_view_2` | QK^T attention-score matmul and layout work | `QK^T: {b:32,m:1,n:1,k:1}` | 0.006 | `QK^T: {b:4,m:4,n:1,k:2}` | 0.006 | -0.000 |
| 3 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_mul_split_with_sizes_sum_transpose_unsqueeze_view_1` | Attention postprocessing / reduction / transpose bucket | no matmul in this SDSC bucket | 2.655 | no matmul in this SDSC bucket | 0.570 | -2.085 |
| 4 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_unsqueeze_3` | Attention pointwise softmax/normalization bucket | no matmul in this SDSC bucket | 0.049 | no matmul in this SDSC bucket | 0.049 | +0.000 |
| 5 | `sdsc_fused_add_linear_mul_rms_norm_silu_split_with_sizes_5` | MLP entry: residual/RMS work fused with gate/up projection and SiLU | `gate+up: {m:4,n:8,k:1}` | 1.488 | `gate+up: {m:4,n:8,k:1}` | 0.979 | -0.508 |
| 6 | `sdsc_fused_add_linear_mul_silu_split_with_sizes_6` | MLP exit: down projection plus residual / pointwise work | `down: {m:32,n:1,k:1}` | 5.197 | `down: {m:4,n:8,k:1}` | 5.194 | -0.003 |
| 7 | `sdsc_fused_linear_mul_rms_norm_split_with_sizes_sum_unsqueeze_view_0` | Input norm + fused QKV projection | `QKV: {m:1,n:32,k:1}` | 4.035 | `QKV: {m:4,n:8,k:1}` | 2.540 | -1.495 |

## Actual SDSC Matmul Picks

This is the source of truth for what actually ran. These rows are extracted
from the emitted `sdsc_*.json` files in each Granite block cache. The CPU
cost-function table in the next section is only a diagnostic for why the branch
wants different choices; when CPU and SDSC disagree, use this SDSC table.

| regime | function / role | shape `B x M x N x K` | shared RHS | baseline SDSC pick | candidate SDSC pick | changed? |
|---|---|---|---:|---|---|---:|
| decode | O projection | `1x64x4096x4096` | true | `{m:32,n:1,k:1}` | `{m:4,n:8,k:1}` | true |
| decode | MLP down projection | `1x64x4096x12800` | true | `{m:32,n:1,k:1}` | `{m:4,n:8,k:1}` | true |
| decode | fused QKV projection | `1x64x6144x4096` | true | `{m:1,n:32,k:1}` | `{m:4,n:8,k:1}` | true |
| decode | MLP gate/up projection | `1x64x25600x4096` | true | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | false |
| decode | attention @ V | `32x64x128x576` | false | `{b:1,m:32,n:1,k:1}` | `{b:8,m:4,n:1,k:1}` | true |
| decode | QK^T attention scores | `64x32x576x128` | false | `{b:32,m:1,n:1,k:1}` | `{b:4,m:4,n:1,k:2}` | true |
| prefill | O projection | `1x512x4096x4096` | true | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | false |
| prefill | MLP down projection | `1x512x4096x12800` | true | `{m:4,n:8,k:1}` | `{m:8,n:4,k:1}` | true |
| prefill | fused QKV projection | `1x512x6144x4096` | true | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | false |
| prefill | MLP gate/up projection | `1x512x25600x4096` | true | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | false |
| prefill | attention @ V | `32x512x128x512` | false | `{b:1,m:32,n:1,k:1}` | `{b:1,m:32,n:1,k:1}` | false |
| prefill | QK^T attention scores | `512x32x512x128` | false | `{b:4,m:1,n:8,k:1}` | `{b:16,m:1,n:2,k:1}` | true |

## CPU Split-Choice Diff On Granite Block Shapes

This is CPU-side only. It extracts the matmul/BMM shapes present in the Granite
block probe SDSCs and runs each branch cost function over legal split divisors.
It does not time kernels or override the emitted SDSC result. The emitted split
columns are repeated from the SDSC source-of-truth table above; the CPU pick
columns show the direct inner cost-function choice on the same logical shape.

Raw artifact on the pod:

```text
/home/adnan/dt-inductor/profiler_runs/granite_block_cost_model_isolated_20260623_224926/cpu_split_choice_diff_20260623_235519/cpu_split_choice_diff.md
```

| regime | function / role | shape `B x M x N x K` | shared RHS | emitted baseline | emitted candidate | CPU main pick | CPU PR pick | changed? |
|---|---|---|---:|---|---|---|---|---:|
| decode | O projection | `1x64x4096x4096` | true | `{m:32,n:1,k:1}` | `{m:4,n:8,k:1}` | `{m:1,n:8,k:2}` | `{m:4,n:8,k:1}` | true |
| decode | MLP down projection | `1x64x4096x12800` | true | `{m:32,n:1,k:1}` | `{m:4,n:8,k:1}` | `{m:1,n:8,k:2}` | `{m:4,n:8,k:1}` | true |
| decode | fused QKV projection | `1x64x6144x4096` | true | `{m:1,n:32,k:1}` | `{m:4,n:8,k:1}` | `{m:1,n:8,k:2}` | `{m:4,n:8,k:1}` | true |
| decode | MLP gate/up projection | `1x64x25600x4096` | true | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | false |
| decode | attention @ V | `32x64x128x576` | false | `{b:1,m:32,n:1,k:1}` | `{b:8,m:4,n:1,k:1}` | `{b:1,m:1,n:2,k:1}` | `{b:8,m:4,n:1,k:1}` | true |
| decode | QK^T attention scores | `64x32x576x128` | false | `{b:32,m:1,n:1,k:1}` | `{b:4,m:4,n:1,k:2}` | `{b:1,m:1,n:9,k:1}` | `{b:8,m:2,n:1,k:2}` | true |
| prefill | O projection | `1x512x4096x4096` | true | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | false |
| prefill | MLP down projection | `1x512x4096x12800` | true | `{m:4,n:8,k:1}` | `{m:8,n:4,k:1}` | `{m:4,n:8,k:1}` | `{m:8,n:4,k:1}` | true |
| prefill | fused QKV projection | `1x512x6144x4096` | true | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | false |
| prefill | MLP gate/up projection | `1x512x25600x4096` | true | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | `{m:4,n:8,k:1}` | false |
| prefill | attention @ V | `32x512x128x512` | false | `{b:1,m:32,n:1,k:1}` | `{b:1,m:32,n:1,k:1}` | `{b:1,m:8,n:2,k:1}` | `{b:1,m:32,n:1,k:1}` | true |
| prefill | QK^T attention scores | `512x32x512x128` | false | `{b:4,m:1,n:8,k:1}` | `{b:16,m:1,n:2,k:1}` | `{b:1,m:1,n:8,k:1}` | `{b:2,m:2,n:8,k:1}` | true |

## Readout

The decode gain comes from moving bad pure-M or awkward output-only decode
splits into healthier split families:

- shared-weight decode projections move toward `{m:4,n:8,k:1}`;
- decode attention @ V moves from pure `m` splitting to batch/head plus `m`;
- decode QK^T gains batch/head, `m`, and `k` parallelism;
- fused QKV decode moves from `{m:1,n:32,k:1}` to `{m:4,n:8,k:1}`.

The prefill story is more mixed in this local block harness. Most heavy
projection choices are preserved. The candidate changes MLP down from
`{m:4,n:8,k:1}` to `{m:8,n:4,k:1}` and changes the QK^T attention-score split,
but the trace-derived prefill kernel total is flat/slightly slower in this
empty-weight one-block probe. Antoni's full Granite e2e run remains the
aggregate source of truth for the reported prefill win.
