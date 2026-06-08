# Jamie Unit-BMM Probe

## Bottom Line

Jamie's temp_passes.py hunch is relevant, with a correction: temp_passes.py is not where the singleton batch dim is lost. It creates a BMM-shaped FX graph. The singleton is lost later when Inductor/torch-spyre dependency coordinates and OpSpecs are simplified.

The actionable win is not just "keep the size-1 dim"; it is "keep the size-1 BMM dim and emit sendnn-like primary layout order."

## What Changed Claude's Conclusion

Claude was right that the decode MLP gap was a benchmark artifact for standard shared-weight transformer MLP. But the prefill shared-weight gap is not exhausted as a DeepTools-only/array-fill issue.

An isolated torch-spyre-only diagnostic patch that preserves the logical BMM singleton and orders the layout like sendnn changed:

| case | baseline tsp | diagnostic tsp | sendnn prior | read |
| --- | ---: | ---: | ---: | --- |
| standalone prefill projection `[[1,512,4096],[4096,12800]]` | 2.469 ms | 1.021 ms | 0.952 ms | near parity |
| shared-weight prefill MLP `[[1,512,4096]]` | 7.745 ms | 3.958 ms | 5.879 ms | diagnostic beats prior sendnn |

The full-MLP parent harness returned nonzero because it claimed no perf files were found, but the child completed and wrote:

`/home/adnan-cdx/codex-isolated/matmul-mlp-gap-current-20260606-070000/profiler_runs/shared_mlp_unit_bmm_sendnn_order_20260606_073056/perf/shared_weight_mlp_torch-spyre_shape_1_512_4096_.txt`

## temp_passes.py Read

`torch_spyre/_inductor/temp_passes.py::_unflatten_mm_to_bmm` matches:

`view(3D -> 2D) -> mm(2D, 2D) -> view(2D -> 3D)`

and rewrites it to:

`bmm(lhs_3d, unsqueeze/expand(rhs_2d))`

So for shared-weight matmul, FX still has `[1, M, K] @ [1, K, N] -> [1, M, N]`.

The singleton disappears downstream:

- `torch._inductor.dependencies.extract_read_writes()` calls `index_vars_squeeze()`.
- `torch._inductor.ir.SqueezeView.squeezer()` removes `size == 1` dimensions.
- `torch_spyre/_inductor/views.py::compute_coordinates()` also skips dims with `size == 1`.
- `SpyreKernel.create_op_spec()` consumes the squeezed iteration space.
- `superdsc.parse_op_spec()` labels dimensions from the resulting rank, so the SDSC becomes `mb/out/in` instead of `x/mb/out/in`.

## Probe Results

Current torch-spyre SDSC for standalone projection:

```text
N_ = { mb: 512, out: 12800, in: 4096 }
INPUT  layoutDimOrder = [in, mb]
KERNEL layoutDimOrder = [in, out]
OUTPUT layoutDimOrder = [out, mb]
```

Naively preserving the singleton was not enough:

| diagnostic layout | kernel_ms |
| --- | ---: |
| `x,in,mb` / `x,out,mb` | 3.394 ms |
| `in,mb,x` / `out,mb,x` | 3.972 ms |
| `mb,in,x` / `mb,out,x` | 1.021 ms |

The winning diagnostic emitted:

```text
N_ = { x: 1, mb: 512, out: 12800, in: 4096 }
INPUT  layoutDimOrder = [mb, in, x]
KERNEL layoutDimOrder = [in, out]
OUTPUT layoutDimOrder = [mb, out, x]
work slices = { x: 1, mb: 4, out: 8, in: 1 }
```

This matches the important part of Jamie's sendnn screenshot: the singleton batch dimension survives in the BMM SDSC, and `mb` precedes the stickified matrix dimension in primary layout order.

## Patch Status

The patch is a diagnostic only, isolated to:

`/home/adnan-cdx/codex-isolated/matmul-mlp-gap-current-20260606-070000/torch-spyre/torch_spyre/_inductor/spyre_kernel.py`

It is flag-gated behind:

`SPYRE_PRESERVE_UNIT_BMM_DIM=1`

It should not be upstreamed as-is. A production fix should carry explicit logical-BMM metadata from `temp_passes.py` or lowering into pre-scheduling/codegen, then preserve/reorder the singleton only for this shared-weight BMM case. It also needs tests for generated OpSpec/SDSC layout, plus non-regression coverage for true 2D matmul and real batched/MoE BMM.

## Next Direction

Do not discard this as a DeepTools-only prefill issue. The immediate torch-spyre path is:

1. Replace the env-gated hack with explicit metadata from `_unflatten_mm_to_bmm()`.
2. Restore the singleton dim before OpSpec simplification for shared-weight BMM.
3. Preserve sendnn-like layout order: input `[mb, in, unit]`, output `[mb, out, unit]`, kernel `[in, out]`.
4. Add focused tests that inspect generated OpSpec/SDSC.
5. Re-run shared prefill MLP against sendnn in one clean paired run.

