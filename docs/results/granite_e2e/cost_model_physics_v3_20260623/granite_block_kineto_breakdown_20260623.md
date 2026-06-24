# Granite Block Kineto And Split Breakdown

Run root:

```text
/home/adnan/dt-inductor/profiler_runs/granite_block_cost_model_isolated_20260623_224926
```

This compares:

- cost model main: current main plus the same local Granite compile prerequisite used on both sides
- cost model improved: the same main setup plus the cost-model `work_division.py`

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

| regime | cost model main wall median ms | cost model improved wall median ms | cost model main kernel ms/iter | cost model improved kernel ms/iter | kernel speedup |
|---|---:|---:|---:|---:|---:|
| prefill | 27.867 | 28.467 | 16.149 | 16.574 | 0.974x |
| decode_expand | 18.868 | 14.880 | 14.765 | 10.621 | 1.390x |

The local block profile supports the decode story strongly. Prefill is flat in
this empty-weight one-block harness, so Antoni's full e2e run remains the
aggregate source of truth for the prefill improvement.

## Prefill Kernel Buckets

| launch | kernel name | function / role | cost model main key split(s) | cost model main mean ms | cost model improved key split(s) | cost model improved mean ms | delta ms |
|---:|---|---|---|---:|---|---:|---:|
| 1 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1` | SDPA value path: attention probabilities @ V plus surrounding attention pointwise/reduction work | `attn@V bmm: {b:1,m:32,n:1,k:1}` | 1.901 | `attn@V bmm: {b:1,m:32,n:1,k:1}` | 1.895 | -0.006 |
| 2 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2` | Attention output path: O projection fused with residual / RMS-norm / layout work | `O-proj: {m:4,n:8,k:1}` | 1.773 | `O-proj: {m:4,n:8,k:1}` | 1.990 | +0.217 |
| 3 | `sdsc_fused_add_linear_mul_silu_split_with_sizes_3` | MLP block: fused gate/up projection, SiLU/mul, and down projection | `gate+up: {m:4,n:8,k:1}; down: {m:4,n:8,k:1}` | 1.450 | `gate+up: {m:4,n:8,k:1}; down: {m:8,n:4,k:1}` | 1.446 | -0.004 |
| 4 | `sdsc_fused_linear_rms_norm_0` | Input norm + fused QKV projection | `QKV: {m:4,n:8,k:1}` | 11.025 | `QKV: {m:4,n:8,k:1}` | 11.243 | +0.218 |

## Decode Kernel Buckets

| launch | kernel name | function / role | cost model main key split(s) | cost model main mean ms | cost model improved key split(s) | cost model improved mean ms | delta ms |
|---:|---|---|---|---:|---|---:|---:|
| 1 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_linear_split_with_sizes_transpose_unsqueeze_view_4` | Attention value path plus O projection in decode context | `attn@V: {b:1,m:32,n:1,k:1}; O-proj: {m:32,n:1,k:1}` | 1.336 | `attn@V: {b:8,m:4,n:1,k:1}; O-proj: {m:4,n:8,k:1}` | 1.283 | -0.053 |
| 2 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_transpose_unsqueeze_view_2` | QK^T attention-score matmul and layout work | `QK^T: {b:32,m:1,n:1,k:1}` | 0.006 | `QK^T: {b:4,m:4,n:1,k:2}` | 0.006 | -0.000 |
| 3 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_mul_split_with_sizes_sum_transpose_unsqueeze_view_1` | Attention postprocessing / reduction / transpose bucket | no matmul in this SDSC bucket | 2.655 | no matmul in this SDSC bucket | 0.570 | -2.085 |
| 4 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_unsqueeze_3` | Attention pointwise softmax/normalization bucket | no matmul in this SDSC bucket | 0.049 | no matmul in this SDSC bucket | 0.049 | +0.000 |
| 5 | `sdsc_fused_add_linear_mul_rms_norm_silu_split_with_sizes_5` | MLP entry: residual/RMS work fused with gate/up projection and SiLU | `gate+up: {m:4,n:8,k:1}` | 1.488 | `gate+up: {m:4,n:8,k:1}` | 0.979 | -0.508 |
| 6 | `sdsc_fused_add_linear_mul_silu_split_with_sizes_6` | MLP exit: down projection plus residual / pointwise work | `down: {m:32,n:1,k:1}` | 5.197 | `down: {m:4,n:8,k:1}` | 5.194 | -0.003 |
| 7 | `sdsc_fused_linear_mul_rms_norm_split_with_sizes_sum_unsqueeze_view_0` | Input norm + fused QKV projection | `QKV: {m:1,n:32,k:1}` | 4.035 | `QKV: {m:4,n:8,k:1}` | 2.540 | -1.495 |

## Attention Matmul Deep Dive

This section isolates the SDPA matmuls because they are the clearest example
of why the improved model helps decode without blindly applying the same choice
to prefill. The evidence below comes from the emitted SDSC JSON fields
`N_`, `numWkSlicesPerDim_`, and `dataStageParam_` in this run root.

### Logical Q/K/V Shapes

The Granite block uses hidden size 4096, 32 query heads, 8 KV heads, and
head dimension 128. The attention BMM path expands the grouped KV heads to the
32 query-head space before the BMMs.

| regime | Q shape after projection | K shape used by attention | V shape used by attention | logical attention BMMs |
|---|---|---|---|---|
| prefill | `[1, 32, 512, 128]` | `[1, 32, 512, 128]` after KV-head expansion | `[1, 32, 512, 128]` after KV-head expansion | QK^T: `[32,512,128] @ [32,128,512] -> [32,512,512]`; attn@V: `[32,512,512] @ [32,512,128] -> [32,512,128]` |
| decode | `[1, 32, 64, 128]` | `[1, 32, 576, 128]` after cache append and KV-head expansion | `[1, 32, 576, 128]` after cache append and KV-head expansion | QK^T: `[32,64,128] @ [32,128,576] -> [32,64,576]`; attn@V: `[32,64,576] @ [32,576,128] -> [32,64,128]` |

The emitted SDSC axis labels are compiler iteration-space labels, not always
the literal PyTorch logical order. In the tables below, `b/m/n/k` means
`SDSC x/mb/out/in` as in the rest of this file. For attention @ V, `b` is the
expanded head-batch axis. For QK^T, the emitted axes are transposed: `x` is
query-length-like and `mb` is head-like. This is why the SDSC shape can look
like `64x32x576x128` for decode QK^T even though the logical BMM is
`[32,64,128] @ [32,128,576]`.

### Emitted SDSC Evidence

| regime | attention matmul | logical BMM | emitted SDSC shape `b x m x n x k` | cost model main split | cost model main per-core tile | cost model improved split | cost model improved per-core tile |
|---|---|---|---|---|---|---|---|
| prefill | attention @ V | `[32,512,512] @ [32,512,128]` | `32x512x128x512` | `{b:1,m:32,n:1,k:1}` | `{b:32,m:16,n:128,k:512}` | `{b:1,m:32,n:1,k:1}` | `{b:32,m:16,n:128,k:512}` |
| prefill | QK^T attention scores | `[32,512,128] @ [32,128,512]` | `512x32x512x128` | `{b:4,m:1,n:8,k:1}` | `{b:128,m:32,n:64,k:128}` | `{b:16,m:1,n:2,k:1}` | `{b:32,m:32,n:256,k:128}` |
| decode | attention @ V | `[32,64,576] @ [32,576,128]` | `32x64x128x576` | `{b:1,m:32,n:1,k:1}` | `{b:32,m:2,n:128,k:576}` | `{b:8,m:4,n:1,k:1}` | `{b:4,m:16,n:128,k:576}` |
| decode | QK^T attention scores | `[32,64,128] @ [32,128,576]` | `64x32x576x128` | `{b:32,m:1,n:1,k:1}` | `{b:2,m:32,n:576,k:128}` | `{b:4,m:4,n:1,k:2}` | `{b:16,m:8,n:576,k:64}` |

The decode attention @ V row is the cleanest hard proof of the batch/head
parallelism story. Cost model main uses all 32 cores by splitting only `m`, but
that leaves each core with only 2 token rows. The improved model still uses
32 cores, but it moves the split to `{b:8,m:4,n:1,k:1}`. That gives each core
16 token rows and uses the independent head-batch axis for the remaining
parallelism. The win is not "more cores"; it is a healthier per-core tile.

Decode QK^T has a similar shape problem but with the emitted axes transposed.
Main splits only the emitted `b`/`x` axis. The improved model spreads work over
the emitted outer axis, the head-like `m` axis, and a small `k=2` reduction
split. That avoids a pure outer-axis schedule and keeps the per-core K tile at
64 elements, exactly one 64-wide stick.

### Cost Model Terms That Matter

The relevant implementation is `_matmul_split_cost` in
`torch_spyre/_inductor/work_division.py`. The attention changes are driven by
general true-BMM terms, not by op-name checks.

| term | formula shape | attention effect |
|---|---|---|
| PT efficiency | compute time is derated when `M / m` is too short to fill PT passes | discourages decode tiles that starve the PT array |
| `m_tile_underfill_us` | `log2(16 / (M / m)) * 30` when `M / m < 16` | directly rejects decode attention @ V main's 2-row/core tile; the improved split reaches 16 rows/core |
| `m_lane_underuse_us` | tie-break toward enough `m` lanes to stream rows over stationary weights | helps QK^T decode avoid a pure outer-axis split with no head/token lane split |
| true-BMM HBM fanout | for true BMM, fanout is `n`, not `max(m,n)` | makes head/batch parallelism cheaper than unnecessary output-column fanout |
| true-BMM batch split | `log2(b) * 10`, additive | allows batch/head splitting when it prevents M underfill instead of globally punishing it |
| true-BMM PSUM | `(k - 1) * output_elems_per_core * 1e-4` | lets small K-splits such as QK^T decode `k=2` win when they improve the tile shape |
| large-M true-BMM value guard | when `M` is large, `M/m >= 16`, and `K >> N`, penalize unnecessary `n` split | keeps prefill attention @ V at `{b:1,m:32,n:1,k:1}` because it already has 16 rows/core and splitting tiny `N=128` would add layout/fusion fallout |

This is why prefill and decode do different things. In prefill attention @ V,
`M=512` and the emitted split `{m:32}` already gives `512 / 32 = 16` rows per
core, so the PT is fed and the large-M value guard prevents an unnecessary
`n` split. In decode attention @ V, `M=64` with `{m:32}` gives only
`64 / 32 = 2` rows per core, so the underfill term pushes the planner to use
batch/head parallelism instead: `{b:8,m:4,n:1,k:1}` gives `64 / 4 = 16` rows
per core.

For QK^T, the main improvement is reducing avoidable output-column fanout while
using independent outer/head/K parallelism. Prefill QK^T shifts from
`{b:4,m:1,n:8,k:1}` to `{b:16,m:1,n:2,k:1}`; the emitted per-core output
tile becomes wider, but the true-BMM HBM fanout from `n` drops from 8 to 2.
Decode QK^T shifts from `{b:32,m:1,n:1,k:1}` to `{b:4,m:4,n:1,k:2}`, which
adds head-like `m` parallelism and a legal one-stick K split instead of relying
on a single outer-axis split.

### Relation To The Profile

The attention SDSC changes are visible in the decode fused context:

- attention @ V changes from `{b:1,m:32,n:1,k:1}` to `{b:8,m:4,n:1,k:1}`;
- QK^T changes from `{b:32,m:1,n:1,k:1}` to `{b:4,m:4,n:1,k:2}`;
- the fused QKV input projection in the same decode block changes from
  `{m:1,n:32,k:1}` to `{m:4,n:8,k:1}`.

The local Kineto buckets show the largest decode reductions in the surrounding
fused attention/QKV region: the attention postprocessing bucket moves from
2.655 ms to 0.570 ms, and the QKV bucket moves from 4.035 ms to 2.540 ms.
Those buckets include pointwise/layout work as well as matmul-adjacent effects,
so the safe conclusion is that the improved attention split/layout choices
correlate with the fused-region reduction; the SDSC table above is the source
of truth for the actual matmul split changes.

## Actual SDSC Matmul Picks

This is the source of truth for what actually ran. These rows are extracted
from the emitted `sdsc_*.json` files in each Granite block cache. The CPU
cost-function table in the next section is only a diagnostic for why the branch
wants different choices; when CPU and SDSC disagree, use this SDSC table.

| regime | function / role | shape `B x M x N x K` | shared RHS | cost model main emitted SDSC kernel/op | cost model main SDSC pick | cost model improved emitted SDSC kernel/op | cost model improved SDSC pick | changed? |
|---|---|---|---:|---|---|---|---|---:|
| decode | O projection | `1x64x4096x4096` | true | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_cat_clone_expand_linear_mul_rms_norm_split_with_sizes_transpose_unsqueeze_view_4` / `6_batchmatmul` | `{m:32,n:1,k:1}` | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_cat_clone_expand_linear_mul_rms_norm_split_with_sizes_transpose_unsqueeze_view_4` / `6_batchmatmul` | `{m:4,n:8,k:1}` | true |
| decode | MLP down projection | `1x64x4096x12800` | true | `sdsc_fused_linear_mul_rms_norm_silu_split_with_sizes_5` / `9_batchmatmul` | `{m:32,n:1,k:1}` | `sdsc_fused_linear_mul_rms_norm_silu_split_with_sizes_5` / `9_batchmatmul` | `{m:4,n:8,k:1}` | true |
| decode | fused QKV projection | `1x64x6144x4096` | true | `sdsc_fused_linear_mul_rms_norm_split_with_sizes_sum_unsqueeze_view_0` / `7_batchmatmul` | `{m:1,n:32,k:1}` | `sdsc_fused_linear_mul_rms_norm_split_with_sizes_sum_unsqueeze_view_0` / `7_batchmatmul` | `{m:4,n:8,k:1}` | true |
| decode | MLP gate/up projection | `1x64x25600x4096` | true | `sdsc_fused_linear_mul_rms_norm_silu_split_with_sizes_5` / `2_batchmatmul` | `{m:4,n:8,k:1}` | `sdsc_fused_linear_mul_rms_norm_silu_split_with_sizes_5` / `2_batchmatmul` | `{m:4,n:8,k:1}` | false |
| decode | attention @ V | `32x64x128x576` | false | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_cat_clone_expand_linear_mul_rms_norm_split_with_sizes_transpose_unsqueeze_view_4` / `3_batchmatmul` | `{b:1,m:32,n:1,k:1}` | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_cat_clone_expand_linear_mul_rms_norm_split_with_sizes_transpose_unsqueeze_view_4` / `3_batchmatmul` | `{b:8,m:4,n:1,k:1}` | true |
| decode | QK^T attention scores | `64x32x576x128` | false | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_transpose_unsqueeze_view_2` / `5_batchmatmul` | `{b:32,m:1,n:1,k:1}` | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_transpose_unsqueeze_view_2` / `5_batchmatmul` | `{b:4,m:4,n:1,k:2}` | true |
| prefill | O projection | `1x512x4096x4096` | true | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_split_with_sizes_transpose_unsqueeze_view_2` / `10_batchmatmul` | `{m:4,n:8,k:1}` | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_split_with_sizes_transpose_unsqueeze_view_2` / `10_batchmatmul` | `{m:4,n:8,k:1}` | false |
| prefill | MLP down projection | `1x512x4096x12800` | true | `sdsc_fused_linear_mul_rms_norm_silu_split_with_sizes_3` / `9_batchmatmul` | `{m:4,n:8,k:1}` | `sdsc_fused_linear_mul_rms_norm_silu_split_with_sizes_3` / `9_batchmatmul` | `{m:8,n:4,k:1}` | true |
| prefill | fused QKV projection | `1x512x6144x4096` | true | `sdsc_fused_linear_mul_rms_norm_split_with_sizes_sum_unsqueeze_view_0_` / `7_batchmatmul` | `{m:4,n:8,k:1}` | `sdsc_fused_linear_mul_rms_norm_split_with_sizes_sum_unsqueeze_view_0` / `7_batchmatmul` | `{m:4,n:8,k:1}` | false |
| prefill | MLP gate/up projection | `1x512x25600x4096` | true | `sdsc_fused_linear_mul_rms_norm_silu_split_with_sizes_3` / `2_batchmatmul` | `{m:4,n:8,k:1}` | `sdsc_fused_linear_mul_rms_norm_silu_split_with_sizes_3` / `2_batchmatmul` | `{m:4,n:8,k:1}` | false |
| prefill | attention @ V | `32x512x128x512` | false | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_split_with_sizes_transpose_unsqueeze_view_2` / `7_batchmatmul` | `{b:1,m:32,n:1,k:1}` | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_split_with_sizes_transpose_unsqueeze_view_2` / `7_batchmatmul` | `{b:1,m:32,n:1,k:1}` | false |
| prefill | QK^T attention scores | `512x32x512x128` | false | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1` / `6_batchmatmul` | `{b:4,m:1,n:8,k:1}` | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1` / `6_batchmatmul` | `{b:16,m:1,n:2,k:1}` | true |

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

| regime | function / role | shape `B x M x N x K` | shared RHS | emitted cost model main | emitted cost model improved | CPU cost model main pick | CPU cost model improved pick | changed? |
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
projection choices are preserved. The cost model improved run changes MLP down from
`{m:4,n:8,k:1}` to `{m:8,n:4,k:1}` and changes the QK^T attention-score split,
but the trace-derived prefill kernel total is flat/slightly slower in this
empty-weight one-block probe. Antoni's full Granite e2e run remains the
aggregate source of truth for the reported prefill win.
