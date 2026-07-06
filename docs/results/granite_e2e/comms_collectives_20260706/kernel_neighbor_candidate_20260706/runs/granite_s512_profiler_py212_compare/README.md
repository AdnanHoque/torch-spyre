# Granite S512 Profiler Comparison: DLDSC Kernel-Neighbor Relayout

This run uses the scratch profiler-enabled Torch-Spyre extension rebuilt against the py212 Torch runtime. It compares the same one-layer Granite causal prefill S512 probe with LX relayout disabled versus the current kernel-neighbor candidate enabled.

## Run Identity

| Field | Disabled | Enabled |
|---|---|---|
| Run | granite_s512_profile_profpy212_relayout_disabled_20260706_145218 | granite_s512_kernel_neighbor_profile_profpy212_20260706_145100 |
| Return code | 0 | 0 |
| Shape | B=1, S=512, E=4096 | B=1, S=512, E=4096 |
| Case | causal prefill | causal prefill |
| Iterations | 5 active, 1 warmup | 5 active, 1 warmup |

## Timings

| Metric | Disabled | Enabled | Speedup |
|---|---:|---:|---:|
| Wall median ms | 29.468 | 29.418 | 1.002x |
| Kernel ms / iter | 12.565 | 11.875 | 1.058x |
| Memory ms / iter | 0.959 | 0.884 | 1.085x |
| Attention handoff kernel total ms | 5.862 | 2.822 | 2.077x |

## SDSC Effect

The in-scope attention activation handoff changes from sdsc_7 ReStickifyOpHBM to sdsc_7 ReStickifyOpLx. The remaining HBM restickifies in the enabled Granite block are weight/prelayout rows, not computed activation spills, and are out of scope for this communication pass.

The enabled run emits two backend matmul operand plans under enabled/backend_plans_sample/. They are matmul_operand_broadcast contracts lowered as loop-scoped KERNEL-neighbor movement.

## Caveats

This is a kernel-profiler run using a scratch profiler-enabled extension tree. It is valid for kernel timing, but the profiler build itself is not part of the production branches. The CUDA CUPTI warning in the log is unrelated to AIU events; the trace contains AIU kernel events and nonzero kernel timings.
