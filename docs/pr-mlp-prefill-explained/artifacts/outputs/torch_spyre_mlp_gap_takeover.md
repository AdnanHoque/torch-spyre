# Torch-Spyre/SendNN MLP Gap Takeover

Date: 2026-06-06

## Current Verdict

Claude's updated conclusion is mostly right, with one metric caveat.

- The old "decode MLP gap" is a benchmark artifact for standard transformer MLPs. The built-in `mlp` benchmark uses rank-3 per-batch weights and forces the batched/MoE-style BMM path.
- A real shared-weight decode MLP does not show a torch-spyre kernel gap. In the isolated current-source probe, torch-spyre is slightly faster than sendnn on `kernel_ms`.
- Shared-weight prefill MLP still has a real gap, but it is about `1.32x`, not the old `1.9x`/`2.7x` decode-batched story.
- Per-SDSC/per-key rows are diagnostic only. Use raw/report `kernel_ms.mean_ms` for headline parity. `Spyre-kernel_times Total` is a mean of per-key rows in the harness, not a sum.

## Key Measurements

All torch-spyre current-source runs used isolated snapshot:

`/home/adnan-cdx/codex-isolated/matmul-mlp-gap-current-20260606-070000`

Shared-weight MLP custom op:

`/home/adnan-cdx/codex-isolated/matmul-mlp-gap-current-20260606-070000/probes/shared_weight_mlp_matmul_op.py`

| case | shape | torch-spyre kernel_ms | sendnn kernel_ms | tsp/sendnn |
|---|---:|---:|---:|---:|
| shared MLP decode | `[4, 1, 4096]` | 2.125 | 2.437 | 0.872x |
| shared MLP prefill | `[1, 512, 4096]` | 7.745 | 5.879 | 1.317x |

Prefill sweep, same shared-weight op:

| M | torch-spyre kernel_ms | sendnn kernel_ms | tsp/sendnn |
|---:|---:|---:|---:|
| 64 | 2.323 | 2.137 | 1.087x |
| 128 | 2.678 | 2.271 | 1.179x |
| 256 | 3.981 | 2.943 | 1.353x |
| 512 | 7.745 | 5.879 | 1.317x |

Standalone gate/up projection at `M=512,N=12800`:

| run | torch-spyre kernel_ms | sendnn kernel_ms | note |
|---|---:|---:|---|
| focused shared artifact | 2.469 | 0.952 | prior VS Code run |
| current isolated rerun | 2.648 | 0.950 | rerun with more measured iterations |

## What Changed In The Model

- Built-in `benchmark.py` MLP creates rank-3 weights:
  - `gate`: `[batch, emb, intermediate]`
  - `up`: `[batch, emb, intermediate]`
  - `down`: `[batch, intermediate, emb]`
- Standard transformer MLP should use rank-2 shared weights:
  - `gate`: `[emb, intermediate]`
  - `up`: `[emb, intermediate]`
  - `down`: `[intermediate, emb]`
- The shared-weight expression still lowers through `aten.bmm` after unsqueeze/expand, so the string `bmm` alone is not the artifact. The discriminator is whether the weight storage is genuinely rank-3 per batch or rank-2 shared/expanded.

## Critique Of The Older DeepTools/Prefetch Verdict

- The two-matmul probe refutes the "inter-SDSC bubble" as the dominant decode issue. Flat bandwidth across 1/2/3 batched matmuls means combining SDSCs is not the missing 2x.
- The path evidence is still useful but should be scoped narrowly:
  - `dxp_standalone --bundle` does not directly call DSM ProgramCombine/weight-preload machinery.
  - That does not prove all Inductor-side levers are exhausted.
- `can_fuse_vertical/horizontal=False` does not mean SiLU/mul are separately launched. Spyre's post-pass builds `sdsc_fused_mul_silu_0` for gate/up/SiLU/mul, with down projection separate.

## Next Actionable Probe

The remaining real target is shared-weight prefill under-fill.

Best next discriminator:

1. Generate shared-prefill SDSCs with sendnn-shaped schedule metadata: `coreletFold=2`, `numCoreletsUsed=2`, tile-local LX intermediates, and BMM output/corelet split shaped like sendnn.
2. Separately keep the current split but isolate/remove the large fill/memset-looking path from profiling.

Interpretation:

- If only fill-free moves, chase array-fill/memset behavior.
- If only sendnn-shaped corelet/LX moves, chase schedule/locality/PT under-fill.
- If neither moves, the residual is likely inside DeepTools BMM schedule quality for the current SDSC contract.

## Artifact Paths

- Current key torch-spyre run:
  `/home/adnan-cdx/codex-isolated/matmul-mlp-gap-current-20260606-070000/profiler_runs/current_tsp_keypoints_20260606_065514`
- Current gate rerun:
  `/home/adnan-cdx/codex-isolated/matmul-mlp-gap-current-20260606-070000/profiler_runs/current_gate_rerun_20260606_070053`
- Shared-weight sweep:
  `/home/adnan-cdx/codex-isolated/matmul-mlp-gap-20260606-055706/profiler_runs/shared_weight_mlp_prefill_sweep_20260606_064331`
- Earlier shared-weight sendnn comparisons:
  `/home/adnan-cdx/codex-isolated/matmul-mlp-gap-20260606-055706/profiler_runs/shared_weight_mlp_matmul_tsp_20260606_063605`
