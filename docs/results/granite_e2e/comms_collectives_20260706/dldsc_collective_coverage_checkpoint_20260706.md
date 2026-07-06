# DLDSC LX Collective Coverage Checkpoint - 2026-07-06

This checkpoint records the current state of the Granite/flash communication-collectives work on the artifact branch. It is intentionally narrow: it separates what Torch can describe, what Deeptools can lower today, and what has actually been exercised on AIU.

## Source State

Latest artifact branch:

- Torch artifact branch: `AdnanHoque/torch-spyre ah/comms-collectives`
- Artifact branch head at time of update: `1b078d4`
- Deeptools collectives branch under test: `Adnan-Hoque1/deeptools ah/comms-collectives`
- Deeptools commit under test: `2162efb3e`

CDX probe:

- Pod: `adnan-cdx-spyre-dev-pf`
- Probe root: `/home/adnan-cdx/codex-isolated/comms_collectives_gather_reduce_probe_20260706_005444`
- Torch checkout in probe: `db16aab3`
- Deeptools checkout in probe: `2162efb3e`
- Probe report: `/home/adnan-cdx/codex-isolated/comms_collectives_gather_reduce_probe_20260706_005444/probe_report.md`

Granite S512 profiled run:

- Pod: `adnan-clc-spyre-dev-pf`
- Workroot: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404`
- Enabled run: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_clone_source_patch_backend2162_20260706_003302`
- Disabled same-branch control: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_disabled_current_backend2162_retry_20260706_005549`

## Current Result Summary

Granite block causal prefill, shape `[1, 512, 4096]`, empty Spyre weights, one FMS layer:

| variant | kernel ms/iter | wall median ms | explicit `ReStickifyOpHBM` rows | explicit `ReStickifyOpLx` rows | backend plans |
|---|---:|---:|---:|---:|---:|
| relayout disabled | 12.5480 | 30.8380 | 5 | 0 | 0 |
| relayout enabled | 11.9182 | 30.4470 | 4 | 1 | 2 |

Observed speedup:

- Kernel-time speedup: `1.053x`
- Wall-time speedup: `1.013x`

The enabled Granite run proves two activation-side attention handoffs are realized through on-chip movement:

| edge | enabled evidence | class | backend plan |
|---|---|---|---|
| attention pointwise/context output into value-side BMM operand | `sdsc_7 ReStickifyOpLx -> sdsc_8 batchmatmul Tensor1` | all-gather / replicate | `8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`, 512 logical transfers |
| attention clone/identity output into later BMM operand | `sdsc_15 identity/clone -> sdsc_16 batchmatmul Tensor1` | all-gather / replicate | `16_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`, 1024 logical transfers |

Both backend plans report `physical_lowering_status=lowered_loop_scoped_kernel_neighbor` and `realized=true`.

## Remaining Explicit HBM Restickifies

The remaining explicit `ReStickifyOpHBM` rows in the enabled Granite run are weight-format rows by shape and graph position:

| kernel | file | logical size | classification |
|---|---|---:|---|
| `sdsc_fused_linear_rms_norm_0_*` | `sdsc_6.json` | `6144 x 4096` | QKV projection weight |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_*` | `sdsc_0.json` | `4096 x 4096` | attention output projection weight |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_*` | `sdsc_0.json` | `25600 x 4096` | fused SwiGLU gate/up projection weight |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_*` | `sdsc_4.json` | `4096 x 12800` | SwiGLU down projection weight |

These are out of scope for the activation communication-collectives lane because weight prelayout/preload is expected to solve them offline.

Important caveat: SDSC `labeledDs_` can still show `hbm+lx` for compute tensors. That metadata alone is not proof of an explicit HBM relayout round trip. The stronger claim we can make from the current artifacts is that the remaining explicit `ReStickifyOpHBM` rows in the profiled Granite bundle are weight-format rows.

## Class Coverage

| communication class | Torch representation | Deeptools lowering | AIU evidence | next gap |
|---|---|---|---|---|
| scatter / permutation | `lx_relayout.py` classifies one-to-one coordinate movement | generic coordinate-mismatch `STCDPOpLx` path covers direct copy relayouts | prior PR1/scatter artifacts; not the active Granite blocker in the latest S512 run | keep capacity and non-overlap guards tight |
| broadcast | topology classification exists; matmul RHS cases can become `matmul_operand_broadcast` | generic fixture covers full-to-sliced copy; specialized matmul RHS path lowers ring transfers | Granite and flash matmul operand plans prove the specialized path | productionize generic fanout replay outside matmul RHS |
| multicast | topology classification exists as one-to-many subset | Deeptools has older STCDP/DCG multicast machinery, but no dedicated DLDSC relayout carrier tied to Torch metadata | not independently proven in this run | add a focused DLDSC replay with explicit subset receivers |
| gather | topology classification exists as many-to-one copy movement | Deeptools unit fixture covers sliced producer to full consumer; destination allocation contract is still thin | not independently proven on a useful Granite edge | define destination ownership/non-overlap contract and replay on a useful edge |
| all-gather / replicate | topology classification exists; matmul RHS can become `all_gather_replicate`; form-changing path can become `layout_allgather_restickify` | specialized matmul RHS path emits loop-scoped kernel-neighbor ring transfers | Granite S512 has 2 realized plans; flash structural run has 32 plans and zero HBM restickifies | finish broader two-stage source-layout gather plus local KERNEL layout conversion |
| reduce | skipped by LX relayout planner when producer has partial reduction state | not implemented in this DLDSC relayout path | none | new arithmetic collective contract is needed |
| all-reduce | not generated by the current relayout planner | not implemented in this DLDSC relayout path | none | build reduce first, then redistribute/broadcast the result |

## Flash Attention Cross-Check

The strongest completed standalone flash structural run is:

- Pod: `adnan-spyre-dev-pf`
- Root: `/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_current_main_backend1_20260705_191932`
- Return code: `0`
- SDSCs: `550`
- HBM restickify count: `0`
- LX restickify count: `33 files / 97 occurrences`
- Backend plans: `32`
- Plan kind: `matmul_operand_broadcast`
- Communication pattern: `all_gather_replicate`

This is useful SDSC/classification evidence, not yet a fresh timing conclusion. Later flash attempts were interrupted or device-busy.

## Interpretation

The current branch is no longer just a scatter prototype. It has a working specialized all-gather/replicate route for matmul RHS-style operands, and that route removes in-scope attention activation handoffs in the Granite S512 block. The remaining explicit HBM restickifies in that profiled Granite run are weight-format rows.

The next hard implementation target is not generic reduce. It is broadening the all-gather path so it handles grouped/form-changing matmul operands cleanly, including source-layout gather and local KERNEL layout conversion. Reduce and all-reduce should be treated as a separate arithmetic-collective feature because they combine values, not just move bytes.

## Reproduction Reminders

Use the split frontend/backend LX capacity setup:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
export PATH=$ROOT/tools/dxp-split-wrapper:$ROOT/deeptools/build-deeptools/dxp:$PATH
```

The wrapper maps `DXP_BACKEND_LX_FRAC_AVAIL` to `DXP_LX_FRAC_AVAIL` only for the Deeptools subprocess. Torch sees full frontend LX planning; DXP receives backend scratch space for inserted ring-transfer chunks.

Core relayout flags used for the successful Granite run:

```bash
export SPYRE_LX_PLANNING=1
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
export SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
export SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
```

For wedged devices, first kill stale Python/DXP/senprog users of `/dev/vfio/*`. If that does not clear it, delete/recreate the pod. `aiu_dd2_hot_reset` may fail on these pods with `RISCV config not found`, so pod recreation has been the reliable fallback.
