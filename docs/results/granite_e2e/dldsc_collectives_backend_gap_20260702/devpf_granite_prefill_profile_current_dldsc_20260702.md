# Granite Causal Prefill DLDSC Profile Attempt - dev-pf - 2026-07-02

## Summary

- Pod: `adnan-spyre-dev-pf`
- Workspace: `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155`
- Run dir: `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/granite_prefill_profile_current_dldsc_20260702_162328`
- Torch branch/SHA: `ah/comms-collectives` / `fc3c6686f22eaf5112b2adbe814420f6fdfa7567`
- Deeptools branch/SHA: `ah/comms-collectives` / `3d54e87eb404b54c0ba74b98d6caa83945b2ef5b`
- DXP binary: `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/build-deeptools/dxp/dxp_standalone`
- Granite harness: `/home/adnan/codex-isolated/comms_collectives_20260629/spyre-granite-e2e-bench`, SHA `76cd51426ba1de6e99dd8fbf613cb0f32b71e87f`
- Local runtime setup: copied `_C.so` from `/home/adnan/codex-isolated/dldsc_collectives_validate_20260702_112302/torch-spyre/torch_spyre/_C.so`, SHA256 `b449a232ec1c07046eb64153d9672447242734005a3f822678f665aabe835c99`.

## Command

Full command/env is archived at:

- `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/granite_prefill_profile_current_dldsc_20260702_162328/command.txt`

Key relayout env:

```text
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=1
SPYRE_LX_PLANNER_RELAYOUT=1
LX_BOUNDARY_CLONES=1
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
```

Workload command shape:

```text
granite_block_layer_probe.py --case prefill --compile-block --attn-name sdpa_causal --iters 1 --warmups 1 --profile --no-profile-memory
```

## Result

- Exit code: `255`
- Attempt wall time: `111 s`
- Profile trace: none emitted; abort happened during warmup before the profiled iteration.
- `kernel_ms_per_iter`: unavailable.
- `wall_ms` per measured iteration: unavailable.
- Correctness: not reached.
- Fallback status: no non-weight activation `ReStickifyOpHBM` rows in emitted SDSCs; weight/prelayout HBM restickifies remain.

## Exact Blocker

This is not the known GraphEditor `ReinterpretView` issue and not a library-load failure. It reached runtime and launched kernels, then failed with a PCIe bus fence during warmup:

```text
RuntimeStream::synchronize() still waiting after 60000ms: in_flight_=1 device=0
RuntimeStream::synchronize() device=0 completed after 60000ms
RAS::PCI::BusFence code 0xa35e
```

The launched kernels before failure were:

```text
sdsc_fused_linear_rms_norm_0
sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1
sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2
```

The device was still queryable after the failure with `aiu-query-devices`, so this run did not leave the pod in an obviously unrecoverable state.

## Generated Artifacts

- SDSC JSON files: `47`
- `bundle.mlir` files: `4`
- Backend plan files: `2`
- Trace files: `0`
- Probe `result.json`: not emitted, because the process aborted via native runtime exception before Python cleanup.

Generated summaries:

- `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/granite_prefill_profile_current_dldsc_20260702_162328/sdsc_op_summary.json`
- `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/granite_prefill_profile_current_dldsc_20260702_162328/sdsc_sequence_summary.json`
- `/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/granite_prefill_profile_current_dldsc_20260702_162328/aiu_query_after.txt`

## SDSC Restickify Evidence

Operation counts:

```json
{
  "ReStickifyOpHBM": 4,
  "ReStickifyOpLx": 1,
  "add": 5,
  "batchmatmul": 6,
  "exp": 1,
  "identity": 6,
  "max": 1,
  "mean": 2,
  "mul": 13,
  "realdiv": 1,
  "rsqrt": 2,
  "silu": 1,
  "sub": 1,
  "sum": 1,
  "sumnonstick": 2
}
```

The attention kernel contains the non-weight activation relayout as `ReStickifyOpLx`:

```text
sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_ekis0ivm
  sdsc_9.json ReStickifyOpLx cores=32
  N: mb=32, x=512, out=128
```

The four remaining HBM restickifies are projection-weight-shaped prelayouts, not the attention activation spill:

```text
sdsc_fused_linear_rms_norm_0_kelp4spx/sdsc_7.json
  ReStickifyOpHBM N: mb=6144, out=4096

sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_x1vqlb0v/sdsc_0.json
  ReStickifyOpHBM N: mb=4096, out=4096

sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_x1vqlb0v/sdsc_10.json
  ReStickifyOpHBM N: mb=25600, out=4096

sdsc_fused_add_linear_mul_3_s2jivkqz/sdsc_0.json
  ReStickifyOpHBM N: mb=4096, out=12800
```

Interpretation: for this emitted block, the DLDSC path keeps the attention activation restickify on LX, but the block still fails at runtime before we can measure speed or correctness. The remaining HBM restickifies are weight/projection prelayout rows, which are out of scope for this communication primitive effort and expected to be handled by separate weight preloading/prelayout work.

## Backend Plan Artifacts

Plan files:

```text
/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/granite_prefill_profile_current_dldsc_20260702_162328/backend_plans/10_batchmatmul_Tensor1_0_layout_allgather_restickify_plan.json
/home/adnan/codex-isolated/dldsc_collectives_devpf_20260702_154155/runs/granite_prefill_profile_current_dldsc_20260702_162328/backend_plans/18_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
```

The `layout_allgather_restickify` artifact was emitted as realized, but its movement fields are empty (`logical_transfer_count=0`, `group_count=0`). That is consistent with the active CDX diagnosis: the current backend path recognizes the contract but does not correctly materialize the `dimension_rename` source-to-consumer coordinate mapping.

The `matmul_operand_broadcast` artifact remains metadata-only/blocked with `logical_transfer_count=1024`, so it is not the physical path that completed this run.

## Classification

- GraphEditor `ReinterpretView`: not observed.
- Runtime/library setup: passed far enough to compile and launch kernels.
- Capacity failure: not observed; no `initial chunk parameters must fit in LX` failure, so no 0.2 retry was run.
- Current blocker: hardware/runtime unsafe relayout realization, likely the same backend physical mapping/scheduling class being debugged on CDX for standalone flash correctness.
