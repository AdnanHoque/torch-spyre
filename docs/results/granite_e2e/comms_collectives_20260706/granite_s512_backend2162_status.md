# Granite S512 DLDSC Collectives Status - 2026-07-06

## Source State

- Torch branch: `AdnanHoque/torch-spyre ah/comms-collectives`
- Torch commit: `bf61f811`
- Deeptools fork branch: `Adnan-Hoque1/deeptools ah/comms-collectives`
- Deeptools commit: `2162efb3e`
- Pod: `adnan-clc-spyre-dev-pf`
- Workroot: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404`
- Run root: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_clone_source_patch_backend2162_20260706_003302`

## Workload

- FMS one-layer Granite block
- Case: causal prefill
- Input shape: `[1, 512, 4096]`
- Attention: `sdpa_causal`
- Weights: empty Spyre tensors; real values are irrelevant for compiler/profiler shape testing
- Profile mode: Kineto trace, `iters=5`, `warmups=1`, no profile memory

## Result

- Return code: `0`
- Wall median: `30.447006 ms`
- Kernel time per iter: `11.918180 ms`
- Memory time per iter: `0.422779 ms`

For comparison, the previous enabled run before the clone-source eligibility patch was:

- Run: `relayout_enabled_after_allgather_fix_20260705_234646`
- Kernel time per iter: `12.103802 ms`
- Backend plans: `1`

The latest run adds one more realized backend plan and reduces trace kernel time by about `1.5%` versus that previous enabled run. Compared with the clean disabled baseline retry (`12.546643 ms/kernel_iter`), the latest enabled run is about `1.053x` faster in kernel time.

## Realized On-Chip Movement

Two activation-side attention handoffs are now realized by DLDSC/LX relayout:

| edge | SDSC evidence | communication class | backend evidence |
|---|---|---|---|
| Attention pointwise/context output into value-side BMM operand | `sdsc_7 ReStickifyOpLx -> sdsc_8 batchmatmul Tensor1` in `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_*` | all-gather / replicate | `8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`, `512` logical transfers, `lowered_loop_scoped_kernel_neighbor`, `realized=true` |
| Attention clone/identity output into later BMM operand | `sdsc_15 identity/clone -> sdsc_16 batchmatmul Tensor1` in the same attention kernel | all-gather / replicate | `16_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`, `1024` logical transfers, `lowered_loop_scoped_kernel_neighbor`, `realized=true` |

The second row is the new coverage from the Torch clone-source eligibility patch. Before that patch, this edge was classified but its consumer tensor stayed HBM-backed; now Deeptools receives an LX `Tensor1` target and emits the second ring-transfer plan.

## Remaining Explicit HBM Restickify Rows

The remaining `ReStickifyOpHBM` rows in this run are weight-format restickifies by shape and graph position. These are intentionally out of scope for this communication-collectives track because weight preload/layout work is expected to handle them offline.

| kernel | file | logical size | interpretation |
|---|---|---:|---|
| `sdsc_fused_linear_rms_norm_0_*` | `sdsc_6.json` | `6144 x 4096` | QKV projection weight layout |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_*` | `sdsc_0.json` | `4096 x 4096` | attention output projection weight layout |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_*` | `sdsc_0.json` | `25600 x 4096` | fused SwiGLU gate/up projection weight layout |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_*` | `sdsc_4.json` | `4096 x 12800` | SwiGLU down projection weight layout |

No remaining explicit `ReStickifyOpHBM` row in this run is currently classified as an in-scope non-weight activation spill. A separate residency scan still shows many compute tensors with `hbm+lx` metadata, so this should not be read as proof that every intermediate avoids every HBM-backed allocation/write. The strong claim here is narrower: the explicit HBM restickify round trips left in the Granite S512 bundle are weight-format rows.


## Residency Caveat

The SDSC `labeledDs_` metadata can show `hbm+lx` for compute tensors even when the explicit HBM restickify has been removed. That can mean a graph boundary, a weight/input backing allocation, or dual residency metadata. It is not by itself the same as an HBM relayout round trip. Before making a stronger claim than the table above, inspect generated senprog/DXP transfer artifacts or profiler memory events for the specific tensor edge.

## Flash Attention Cross-Check

A separate successful DEV-pod flash run already shows the same direction on the standalone flash workload:

- Root: `/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_current_main_backend1_20260705_191932`
- Return code: `0`
- SDSCs: `550`
- HBM restickify: `0`
- LX restickify: `33 files / 97 occurrences`
- Backend plans: `32`
- Plan kind: `matmul_operand_broadcast`
- Communication pattern: `all_gather_replicate`

Later flash attempts were interrupted or device-busy, so this is SDSC/classification evidence rather than a fresh timing conclusion.

## Current Communication-Class Coverage

| class | current evidence | status |
|---|---|---|
| scatter/permutation | PR1 scatter path and DLDSC metadata path | supported for direct ownership remap |
| broadcast/multicast | represented as `matmul_operand_broadcast`; backend can lower realized LX operand plans | partly supported for matmul operand broadcast/replicate cases |
| all-gather/replicate | Granite attention and flash attention plans lower through loop-scoped kernel-neighbor ring transfers | working for current matmul operand cases |
| gather | not independently proven in this run | still needs focused workload/SDSC case |
| reduce/all-reduce | not present as a remaining non-weight Granite spill in this run | still future primitive work; arithmetic combination is not just data movement |
| form-changing restickify | explicit `ReStickifyOpLx` works for the attention handoff above | partially supported; broader layout-changing cases still need focused tests |

## Reproduction Notes

Use the same split LX capacity setup that previously unblocked Granite:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
export PATH=$ROOT/tools/dxp-split-wrapper:$ROOT/deeptools/build-deeptools/dxp:$PATH
```

The wrapper maps `DXP_BACKEND_LX_FRAC_AVAIL` to `DXP_LX_FRAC_AVAIL` only for the Deeptools subprocess. Torch sees full frontend LX planning, while DXP receives backend scratch space for inserted ring-transfer chunks.

Key feature flags for this run:

```bash
export SPYRE_LX_PLANNING=1
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
export SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
export SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
```

Run command:

```bash
/home/adnan/dt-inductor/.venv/bin/python benchmarks/granite_block_layer_probe.py   --fms-root "$ROOT/foundation-model-stack"   --run-root "$RUN_ROOT"   --case prefill   --compile-block   --attn-name sdpa_causal   --iters 5   --warmups 1   --profile   --no-profile-memory
```

## Next Gaps

1. Add focused gather and reduce/all-reduce workloads rather than infer support from absence in this Granite run.
2. Make the matmul operand broadcast/all-gather backend path less prototype-shaped: remove diagnostic env coupling, reduce debug-only DDC changes, and add small Deeptools unit coverage for the exact coordinate cases.
3. Decide whether the `matmul_operand_broadcast` naming should split into more precise taxonomy (`broadcast`, `multicast`, `all_gather_replicate`) now that it is carrying multiple cardinalities.
4. Re-run the flash script with a clean device for timing once the current completed SDSC evidence is archived.
