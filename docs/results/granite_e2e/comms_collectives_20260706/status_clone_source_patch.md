# Granite DLDSC Collectives Status - Computed Clone Source Patch

Date: 2026-07-06
Torch branch: `ah/comms-collectives`
Torch commit recorded for this update: `e4ae1053e54f7e6673ac26a022f89a9f0c965e0c`
Deeptools branch under test: `ah/comms-collectives`
Deeptools base commit in CLC workroot: `352919bf3f9c0efb2430568c667111aeb0a99e95` plus local prototype backend edits

## Why This Update Exists

The S512 Granite block run proved one attention activation handoff can now stay on chip through `ReStickifyOpLx` plus a lowered DLDSC `matmul_operand_broadcast` plan. However, the SDSC bundle still contained another activation-like attention handoff:

```text
attention sdsc_15 identity/clone -> attention sdsc_16 batchmatmul Tensor1
```

That edge already had a logical `matmul_operand_broadcast` / `all_gather_replicate` classification, but its source clone was not LX-pinned. The consumer still showed:

```text
sdsc_16 Tensor1 KERNEL allocation: allocate-Tensor1_hbm
source_lx_tensor: missing
```

The root cause was allocator eligibility. Computed-source restickifies were allowed into the relayout research lane, but computed-source clone/layout steps were still rejected as `op not allowed` unless the broad `LX_BOUNDARY_CLONES` feature was enabled. That is too coarse for this goal because boundary/input clones are a separate graph-boundary feature; this edge is an internal computed activation clone.

## Code Change

`torch_spyre/_inductor/scratchpad/allocator.py`

- Factored computed-source detection into `_computed_source_unary_output`.
- Kept computed-source `restickify` eligibility unchanged.
- Added `_clone_output_good_for_lx_relayout`:
  - only active when `SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1`;
  - only for `clone` ops;
  - only when at least one read dependency is another `ComputedBuffer`;
  - graph inputs/weights remain excluded.

`tests/inductor/test_lx_relayout_dldsc.py`

- Added tests proving computed-source clones are eligible.
- Added tests proving graph-input clones are not eligible.

Validation:

```text
python3 -m pytest tests/inductor/test_lx_relayout_dldsc.py -q
18 passed, 1 warning
```

## Current Granite S512 Evidence Before This Patch

Successful enabled run on CLC:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_after_allgather_fix_20260705_234646
```

Baseline retry:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_disabled_baseline_retry1
```

Timing from those artifacts:

| Variant | kernel ms/iter | wall median ms | root ReStickifyOpHBM | root ReStickifyOpLx | backend plans |
|---|---:|---:|---:|---:|---:|
| baseline retry | 12.5466 | 30.5395 | 5 | 0 | 0 |
| relayout enabled after all-gather fix | 12.1038 | 30.8068 | 4 | 1 | 1 |

Interpretation:

- Kernel-time improved by about 3.7% in this specific S512 block run.
- Wall time did not improve in this run.
- One explicit HBM restickify became `ReStickifyOpLx`.
- The backend lowered one `matmul_operand_broadcast` plan as `all_gather_replicate` with `physical_lowering_status=lowered_loop_scoped_kernel_neighbor`.

## What Was Removed

The removed explicit HBM row is in the first attention kernel:

```text
sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1
baseline: sdsc_7 ReStickifyOpHBM
enabled:  sdsc_7 ReStickifyOpLx
```

This is the attention value-side handoff into a downstream batchmatmul operand. It is an activation communication class, not a weight preload issue.

## Remaining Explicit HBM Restickify Rows

The enabled run still has four root `ReStickifyOpHBM` rows:

```text
sdsc_fused_linear_rms_norm_0/sdsc_6
sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2/sdsc_0
sdsc_fused_add_linear_mul_silu_split_with_sizes_3/sdsc_0
sdsc_fused_add_linear_mul_silu_split_with_sizes_3/sdsc_4
```

The restickify insertion log identifies the source of these rows as graph inputs:

```text
buf45 input arg2_1
buf47 input arg5_1
buf48 input arg7_1
buf49 input arg8_1
```

Those are currently treated as preload/prelayout work and are out of scope for the DLDSC activation-collectives lane. The current feature should not chase those unless the scope changes.

## Remaining Activation-Like Handoff

The next in-scope activation handoff is not counted as a root `ReStickifyOpHBM` row. It appears as an HBM-backed matmul KERNEL operand:

```text
attention sdsc_15 identity/clone -> attention sdsc_16 batchmatmul Tensor1
```

Evidence from `sdsc_16.json` before this patch:

```text
kind: matmul_operand_broadcast
communication_class: all_gather
communication_pattern: all_gather_replicate
producer_op: clone
consumer_op: batchmatmul
target_kernel_tensor allocation_name: allocate-Tensor1_hbm
source_lx_tensor: missing
```

The `e4ae105` patch targets this exact gap by allowing the computed-source clone to become an LX relayout source.

## Hardware Verification Still Needed

A follow-up Granite S512 run is needed to verify that the patch changes the `sdsc_15 -> sdsc_16` handoff from HBM-backed Tensor1 to an LX-backed source plus lowered backend plan.

Attempted CLC reruns were blocked by device/runtime state:

- Boundary-clone rerun was interrupted while synchronizing a device copy before SDSC artifacts were emitted.
- Clone-source rerun failed opening `/dev/vfio/25` with `Device or resource busy`.
- CLC hot reset attempted against `0000:3f:00.0` but aborted with `RISCV config not found`.

Do not treat the clone-source patch as hardware-proven yet. It is unit-tested and pushed, but the next step is a clean pod/device rerun.

## Communication Class Matrix From Current Evidence

| Class | Current evidence | Status |
|---|---|---|
| scatter/permutation | Topology tests pass. Not the active S512 Granite blocker. | frontend-classified, not newly hardware-proven here |
| broadcast/multicast | Topology tests pass. No separate real Granite lowered plan observed in this run. | frontend-classified, backend proof pending |
| gather | Prior synthetic LX gather artifact exists from earlier work. No standalone Granite gather plan in this run. | partial evidence only |
| all-gather/replicate | Attention `matmul_operand_broadcast` lowered once in S512 Granite; focused flash artifacts lowered 32 plans. | active path works for at least one Granite edge |
| reduce/all-reduce | Local reductions exist as compute ops, but no DLDSC communication reduce/all-reduce plan observed. | not implemented/proven |

## Reproduction Notes

Use the run environment from:

```text
.../relayout_enabled_after_allgather_fix_20260705_234646/env.txt
```

Important knobs:

```text
SPYRE_LX_PLANNER_RELAYOUT=1
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
PATCH_MODE=no_h2d,skip_cpu_ref
```

When cloning an old run env into a new pod, do not blindly reuse stale AIU/VFIO variables. Confirm the live pod device first:

```text
ls -l /dev/vfio
env | grep -E "PCIDEVICE_IBM_COM_AIU_PF|AIU_WORLD_RANK_0"
```
