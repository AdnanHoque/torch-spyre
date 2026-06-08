# Static Unit-Batch BMM Canonicalization, From First Principles

This note explains the code path behind implementation commit `9e10e95`
(`Optimize static unit-batch BMM prefill MLP`) on
`pr-mlp-prefill-explained`. The same implementation is present in the branch
history at `e544a05`; later commits add documentation, provenance, and benchmark
artifacts only.

It is intentionally about the compiler representation, not a new benchmark run.
The numbers here are copied from the archived reports under
`docs/pr-mlp-prefill-explained/artifacts`.

## Problem Statement

The prefill MLP shape in the archived probe is:

```text
mlp [[1, 512, 4096]]
```

Inside the MLP, the expensive work is three wide projections:

```text
gate: [1, M, K] @ weight
up:   [1, M, K] @ weight
down: [1, M, I] @ weight
```

For a standard transformer MLP, the weights are shared across the batch. A
mathematically natural representation is:

```text
[1, M, K] @ [K, N] -> [1, M, N]
```

PyTorch and Inductor can also expose an equivalent unit-batch BMM shape:

```text
[1, M, K] @ [1, K, N] -> [1, M, N]
```

The leading dimension is statically `1`. It is not a real batched/MoE dimension
with different weights per batch member. It is only a shape-carrying axis that
keeps the user-visible output rank intact.

The old torch-spyre lowering treated this static unit-batch BMM too literally
after the earlier shared-weight unit-BMM work. The math was correct, but the
layout presented to Spyre/DeepTools did not recover the same sendnn-like
unit-BMM schedule used for the better shared-weight path. That left the full
prefill MLP with an avoidable kernel gap even after the standalone projection
was mostly fixed.

## Why Layout Matters

A matmul is not only the equation:

```text
C[m, n] = sum_k A[m, k] * B[k, n]
```

For Spyre codegen it also becomes an `OpSpec` with:

- an iteration space, such as rows, columns, and reduction;
- tensor device sizes and device coordinates;
- a work split across cores;
- layout ordering before the final stick dimension.

For Granite-sized prefill MLP projection, the useful work is a wide stream of
`M=512` activation rows over `K=4096` into a large `N` projection. The earlier
PR chain already established the good shape for a shared 2D weight:

```text
input:  [unit, in, mb, x]
weight: [in, out]
output: [unit, out, mb, x]
```

Here `unit` is the logical batch dimension of size 1, `mb` is the M/token row
dimension, `in`/`out` are K/N stick tile dimensions, and `x` is the in-stick
lane. Keeping the unit dimension visible lets the same shared-weight unit-BMM
layout preservation and cost-model choices apply. If that size-1 axis is
flattened or squeezed away too early, the generated code can still compute the
same tensor but present a weaker loop/layout shape to the backend.

## What Changed In `temp_passes.py`

The follow-up implementation extended the existing temporary Inductor passes in
two narrow ways.

First, the earlier `mm` to `bmm` rewrite already handled a shared 2D weight:

```text
view([1, M, K] -> [M, K])
mm([M, K], [K, N])
view([M, N] -> [1, M, N])
```

It rewrote that into:

```text
bmm([1, M, K], expand([K, N] -> [1, K, N]))
```

When the only batch dimension is statically `1`, the new node receives FX custom
metadata:

```text
"_spyre_shared_weight_unit_bmm": {"batch_dim": 0}
```

That metadata is the durable marker saying: this looks like a BMM, but it should
use the shared-weight unit-BMM layout rule, not the true batched-weight rule.

Second, `9e10e95` added direct marking for plain static unit-batch BMM nodes.
The helper checks that the lhs, rhs, and output are all rank 3, that all leading
dimensions are statically `1`, and that the matrix dimensions line up:

```text
lhs: [1, M, K]
rhs: [1, K, N]
out: [1, M, N]
```

Only then does it attach the same custom metadata. This is deliberately
conservative: `B > 1` stays on the normal batched/MoE-style BMM path.

The pass still preserves the separate higher-rank matmul cleanup: if Inductor
has collapsed multiple batch dimensions into a 3D `aten.bmm`, the pass can
replace it with `spyre.batched_matmul` because `aten.bmm` requires exactly rank
3 and would trip fake tensor updating for higher-rank inputs.

## How The Marker Gets Lowered

The marker travels through three compiler layers:

1. FX pass metadata:

```text
node.meta["custom"]["_spyre_shared_weight_unit_bmm"] = {"batch_dim": 0}
```

2. Lowering metadata:

```text
op_info["shared_weight_unit_bmm"] = {"batch_dim": 0}
```

3. Spyre kernel `OpSpec` metadata:

```text
OpSpec(..., op_info={"shared_weight_unit_bmm": {"batch_dim": 0}}, ...)
```

The important point is that the pass does not change user semantics. It only
attaches enough information for lowering and codegen to distinguish:

```text
static unit batch: [1,M,K] @ [1,K,N] -> [1,M,N]
true batched BMM:  [B,M,K] @ [B,K,N] -> [B,M,N], B > 1
```

That distinction is lost if codegen only sees "rank-3 BMM".

## What Changed In `spyre_kernel.py`

`spyre_kernel.py` has two jobs in this fix.

First, it recognizes unit-BMM candidates from sizes when the metadata is not
already present. The helper accepts both forms:

```text
[1, M, K] @ [K, N]    -> [1, M, N]
[1, M, K] @ [1, K, N] -> [1, M, N]
```

It rejects non-rank-3 lhs/output, non-unit batch, mismatched `M/K/N`, and true
batched weights. This lets the earlier shared-weight unit-BMM path and the new
static-unit-BMM path converge before OpSpec creation.

Second, it preserves or reconstructs the unit dimension while building the
`OpSpec`. The key function is `_preserve_shared_weight_unit_bmm_dim(...)`.
It runs only when:

- `op_info` contains `shared_weight_unit_bmm`;
- the op is a batch matmul reduction;
- the current iteration space has the expected matmul shape;
- `batch_dim` is `0`.

If each relevant TensorArg still has exactly one size-1, coordinate-0
non-stick dimension, codegen renames that coordinate to a fresh symbol:

```text
_spyre_bmm_unit
```

Then it reorders the non-stick dimensions so the unit dimension comes first and
the stick dimension remains last.

## Recovering `_spyre_bmm_unit` After Squeeze

The subtle case is the full perf-suite MLP. By the time codegen sees some
TensorArgs, the unit axis can already have been squeezed out of the layout even
though the logical operation is still `[1,M,K] @ [1,K,N] -> [1,M,N]`.

`9e10e95` handles that by synthesizing the missing unit layout dimension:

1. Look at the input and output TensorArgs, not the weight TensorArg.
2. If neither has a non-stick size-1, coordinate-0 unit dimension, insert one
   immediately before the stick dimension.
3. Give that inserted coordinate value `0` and a size of `1`.
4. Mark it with the same `_spyre_bmm_unit` symbol used for preserved unit axes.
5. Reorder dimensions into the sendnn-like unit-BMM order.

The archived generated-code evidence shows the recovered layout:

```text
op_info={'shared_weight_unit_bmm': {'batch_dim': 0}}

input device_size=[1, 64, 512, 64]
input device_coordinates=[_spyre_bmm_unit, floor(c2/64), c0, Mod(c2, 64)]

output device_size=[1, 200, 512, 64]
output device_coordinates=[_spyre_bmm_unit, floor(c1/64), c0, Mod(c1, 64)]
```

That is the crux of the fix. The compiler recovers the logical unit axis as a
layout axis, even when an earlier view/squeeze erased it from the physical
TensorArg shape.

`spyre_kernel.py` also keeps the stride-map invariant intact after
`align_tensors(...)` may reorder coordinates. That matters because once a
synthetic or preserved unit dimension is introduced, `stride_map[d]` still has
to refer to the host-element stride for device dimension `d`.

## Composition With The Earlier Chain

This follow-up does not replace the earlier `pr-mlp-fix` chain. It composes with
it.

The earlier chain did three important things:

- converted shared 2D-weight prefill projection into a unit-BMM representation;
- preserved the sendnn-like unit-BMM layout for `[1,M,K] @ [K,N]`;
- adjusted the work-division/cost-model behavior so, after M has enough work,
  remaining split capacity is spent usefully on the wide N/output dimension.

That earlier chain fixed the isolated wide projection. The remaining full MLP
gap came from a different representation of the same basic prefill operation:
the perf-suite MLP was exposing `[1,M,K] @ [1,K,N]`, not only `[1,M,K] @ [K,N]`.

`9e10e95` canonicalizes that static rank-3 form into the same metadata and
layout path. So the chain becomes:

```text
FX shape pattern
  -> shared_weight_unit_bmm marker
  -> lowering op_info
  -> OpSpec unit-axis preservation/recovery
  -> shared-weight unit-BMM layout
  -> existing cost-model/work-division behavior
```

The final result is not a special MLP-only hack. It is a narrower equivalence
class for one shape family:

```text
[1,M,K] @ [K,N] and [1,M,K] @ [1,K,N]
```

both lower as shared-weight unit-BMM, while true `B > 1` BMM remains separate.

## Why This Improves Prefill MLP

The full prefill MLP has two fused gate/up BMMs plus elementwise work, followed
by a down BMM. Before this follow-up, those BMMs were mathematically correct but
arrived at the backend with an awkward static unit-batch layout. That left
torch-spyre slower than sendnn on kernel time even after the isolated projection
had improved.

After canonicalization, the full MLP uses the same healthier layout family as
the fixed standalone projection. The archived report shows the torch-spyre MLP
kernel time moving from `11.702 ms` to `6.169 ms`, while the paired sendnn time
is `5.788 ms`.

The win is therefore representational:

- no new math;
- no device-specific benchmark tuning in this docs change;
- one static unit-batch shape is mapped back to the layout path already known to
  feed the backend better.

## Quantified Isolated Chain

All rows below are archived numbers. They should be read as evidence for the
individual compiler links, not as fresh measurements from this docs commit.

| chain / probe | shape | before | after | gain | comparison / note | artifact |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Earlier shared-weight unit-BMM plus cost model, main vs PR projection | `matmul [[1,512,4096],[4096,12800]]` | `3.749 ms` kernel, `29.794%` PT | `1.023 ms` kernel, `72.799%` PT | `3.66x` kernel | PR spyre time `3.159 ms` vs main `5.790 ms`, `1.83x` | `artifacts/outputs/pr_mlp_fix_shape_aware_summary.md` |
| Diagnostic sendnn-order layout, standalone projection | `[[1,512,4096],[4096,12800]]` | `2.469 ms` kernel | `1.021 ms` kernel | `2.42x` kernel | prior sendnn `0.952 ms`, near parity | `artifacts/outputs/jamie_unit_bmm_probe.md` |
| Diagnostic sendnn-order layout, shared-weight full prefill MLP | `mlp [[1,512,4096]]` | `7.745 ms` kernel | `3.958 ms` kernel | `1.96x` kernel | prior sendnn `5.879 ms`; diagnostic beat that prior run | `artifacts/outputs/jamie_unit_bmm_probe.md` |
| Static unit-batch BMM canonicalization, full perf-suite MLP | `mlp [[1,512,4096]]` | `11.702 ms` kernel | `6.169 ms` kernel | `1.90x` kernel | paired sendnn `5.788 ms`; torch-spyre/sendnn `1.066x` | `artifacts/outputs/static_unit_batch_bmm_canonicalization/report_tsp_sendnn.txt` |
| Static unit-batch BMM canonicalization, standalone projection check | `matmul [[1,512,4096],[4096,12800]]` | n/a | `1.018 ms` kernel | n/a | paired sendnn `0.960 ms`; torch-spyre/sendnn `1.060x` | `artifacts/outputs/static_unit_batch_bmm_canonicalization/static_unit_batch_bmm_canonicalization_report.md` |

## Validation Status

The archived implementation report records the following non-device unit-test
validation:

```text
python -m pytest tests/inductor/test_temp_passes.py \
  tests/inductor/test_coarse_tiling.py \
  tests/inductor/test_work_division.py -q

126 passed
```

No device benchmarks were run for this write-up. The performance numbers above
come from the archived pod reports already checked into this artifact tree.

A fresh bs1/sl512 main-vs-PR sweep was archived after this note was drafted at
`artifacts/outputs/prefill_bs1_sl512_main_vs_pr_sendnn_20260608_072036/`.

## Boundary

This fix is intentionally not the true batched/MoE BMM fix. If `B > 1`, the
leading dimension is no longer a synthetic/shared unit axis. It means separate
batch slices, potentially with separate weights, and it needs its own lowering
and scheduling work.
