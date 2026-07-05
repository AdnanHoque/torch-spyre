# Granite S512 Relayout CLC Archive - 2026-07-05

This note records the current Granite S512 block-prefill relayout state from the
CLC pod-local experiment root. It intentionally covers only the isolated
artifact repos on `adnan-clc-spyre-dev-pf` in namespace `a6-quantization`.

## Scope

- Pod: `adnan-clc-spyre-dev-pf`
- Namespace: `a6-quantization`
- Isolated root: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404`
- Latest archive: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840`
- Pair runbook: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/run_pair.sh`
- Archive UTC window: `2026-07-05T08:19:40Z` to `2026-07-05T08:30:13Z`

The latest runbook and outputs were present, so the status below is a minimal
verification of that archive: return codes, summary files, generated SDSC JSON,
backend plan JSON, and the DXP replay artifact were inspected. No PR branch was
checked out or modified.

## Repository State

| Repo | Remote / branch | SHA | Status |
|---|---|---|---|
| torch-spyre | `AdnanHoque/torch-spyre:ah/comms-collectives` | `8960d88af18e31033a75e36450d8b6efcf9cf301` | clean before this artifact |
| deeptools | `Adnan-Hoque1/deeptools:ah/comms-collectives` | `352919bf3f9c0efb2430568c667111aeb0a99e95` | pre-existing dirty file: ` M util/LayoutAllgatherRestickify.cpp` |
| spyre-granite-e2e-bench | `main` | `76cd51426ba1de6e99dd8fbf613cb0f32b71e87f` | clean |
| foundation-model-stack | detached | `b4f36b5af526b938db506a17dcd32d468a7a91d8` | clean |

Binary paths used by the archive:

- DXP split wrapper: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/tools/dxp-split-wrapper/dxp_standalone`
- DXP real binary: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/deeptools/build-deeptools/dxp/dxp_standalone`
- DXP real binary mtime: `2026-07-05 07:21:50.904661279 +0000`

## Key Environment

Common pinned paths and frontend/backend split:

- `DEEPTOOLS_PATH=/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/deeptools`
- `DEEPTOOLS_INSTALL_DIR=/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/deeptools`
- `PYTHONPATH=/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/torch-spyre:/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/torch-spyre/tests/inductor:/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/foundation-model-stack:/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/spyre-granite-e2e-bench:/home/adnan/dt-inductor/sentient/runtime/lib`
- `DXP_LX_FRAC_AVAIL=0`
- `DXP_BACKEND_LX_FRAC_AVAIL=1`
- `SPYRE_LX_PLANNING=1`

The split wrapper keeps Torch/frontend at `DXP_LX_FRAC_AVAIL=0`; for the DXP
subprocess it exports `DXP_LX_FRAC_AVAIL=$DXP_BACKEND_LX_FRAC_AVAIL`, so DXP
sees LX frac `1`.

Relayout disabled baseline retry:

- `SPYRE_LX_PLANNER_RELAYOUT=0`
- `SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=0`
- `SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=0`
- `SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=0`
- `SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=0`
- `TORCHINDUCTOR_CACHE_DIR=/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_disabled_baseline_retry1/block_prefill/cache`
- Full env: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_disabled_baseline_retry1/env.txt`

Relayout enabled collectives backend1:

- `SPYRE_LX_PLANNER_RELAYOUT=1`
- `SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1`
- `SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1`
- `SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1`
- `SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1`
- `DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1`
- `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1`
- `TORCHINDUCTOR_CACHE_DIR=/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_collectives_backend1/block_prefill/cache`
- Full env: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_collectives_backend1/env.txt`

## Results

`ReStickify` counts are reported as root SDSC rows / raw string occurrences
across generated `sdsc_*.json` files.

| Variant | RC | Process wall s | Median wall ms | Kernel ms/iter | Memory ms/iter | ReStickifyOpHBM | ReStickifyOpLx | Backend plans | Run dir |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| relayout disabled baseline retry | `0` | `93.133` | `30.539512634277344` | `12.546643199999998` | `0.306815` | `5 / 15` | `0 / 0` | `0` | `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_disabled_baseline_retry1` |
| relayout enabled collectives backend1 | `1` | `83.592` | `None` | `None` | `None` | `1 / 3` | `1 / 3` | `1` | `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_collectives_backend1` |
| disabled baseline first attempt | `1` | `100.615` | `None` | `None` | `None` | `5 / 15` | `0 / 0` | `0` | `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_disabled_baseline` |

The first disabled attempt compiled but hit a runtime hardware/stream error:
`Compute CB hardware error detected`, followed by `StreamInErrorState`. The retry
passed and is the baseline row to compare against the enabled run.

## HBM Row Classification

The disabled baseline retry generated five root `ReStickifyOpHBM` rows:

| Row | Shape / split | Classification | Notes |
|---|---|---|---|
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_eh0s3ycu/sdsc_7.json` -> `7_ReStickifyOpHBM` | `N={mb:32,x:512,out:128}`, split `{mb:32,x:1,out:1}` | activation | Attention RHS layout restickify; allocates `Tensor0_hbm` as `['mb','out','x']` and `Tensor1_hbm` as `['mb','x','out']`. |
| `sdsc_fused_linear_rms_norm_0_j28ddriw/sdsc_6.json` -> `6_ReStickifyOpHBM` | `N={mb:6144,out:4096}`, split `{mb:32,out:1}` | weight/prelayout | QKV/front projection KERNEL prelayout, matrix-shaped row with KERNEL layout `['out','mb']`. |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_syjxsuyn/sdsc_0.json` -> `0_ReStickifyOpHBM` | `N={mb:4096,out:4096}`, split `{mb:32,out:1}` | weight/prelayout | Attention output projection prelayout, matrix-shaped row with KERNEL layout `['out','mb']`. |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_x2z9cury/sdsc_0.json` -> `0_ReStickifyOpHBM` | `N={mb:25600,out:4096}`, split `{mb:25,out:1}` | weight/prelayout | FFN gate/up projection prelayout, matrix-shaped row with KERNEL layout `['out','mb']`. |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3_x2z9cury/sdsc_4.json` -> `4_ReStickifyOpHBM` | `N={mb:4096,out:12800}`, split `{mb:1,out:25}` | weight/prelayout | FFN down projection prelayout, matrix-shaped row with KERNEL layout `['out','mb']`. |

The enabled run generated only two kernel groups before aborting. In that partial
set:

- The attention activation row above becomes
  `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_t_frrns5/sdsc_7.json` -> `7_ReStickifyOpLx`, with LX allocations for both Tensor0 and Tensor1.
- The only remaining generated HBM row is
  `sdsc_fused_linear_rms_norm_0_b9mtplfq/sdsc_6.json` -> `6_ReStickifyOpHBM`, the QKV/front projection weight/prelayout row.
- The later disabled-baseline weight/prelayout rows are not reached because the
  enabled run aborts during the first attention bundle compile.

## Backend Plan And Blocker

Enabled run backend plan:

- Plan: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_collectives_backend1/backend_plans/8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`
- `kind=matmul_operand_broadcast`
- `communication_pattern=all_gather_replicate`
- `sdsc_name=8_batchmatmul`
- `logical_transfer_count=512`
- `group_count=2`
- `producer_chunks_per_group=16`
- `consumer_replicas_per_group=16`
- `physical_lowering_status=lowered_loop_scoped_kernel_neighbor`
- `realization_strategy=loop_scoped_input_fetch`
- `mutation_point=Dxp::insertRelayoutSdsc after LX pinned input coordinate mismatch`

Current enabled-run blocker:

```text
terminate called after throwing an instance of 'DtException'
  what():  DtException: Unable to map graph within architecture constraints: The initial chunk parameters must fit in LX for SuperDSC: 8_batchmatmul, file /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 1701
```

This is not the earlier artifact JSON crash. The current enabled path emits the
matmul operand broadcast plan, then DXP aborts because `8_batchmatmul` initial
chunk parameters do not fit in LX.

DXP replay verifies the same blocker without rerunning the full Python probe:

- Replay dir: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/dxp_replay_enabled_failing_bundle_backend1`
- Replay RC: `134`
- Replay wall seconds: `2.083`
- Replay stderr: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/dxp_replay_enabled_failing_bundle_backend1/stderr.log`

## Exact Reproduction Commands

Run the archived pair runner from the host, targeting only the CLC pod:

```bash
oc exec -n a6-quantization adnan-clc-spyre-dev-pf -- bash -lc 'bash /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/run_pair.sh'
```

The exact disabled retry command captured in the archive is:

```bash
cd /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/spyre-granite-e2e-bench
/home/adnan/dt-inductor/.venv/bin/python3 benchmarks/granite_block_layer_probe.py --fms-root /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/foundation-model-stack --run-root /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_disabled_baseline_retry1 --case prefill --seq-len 512 --batch 1 --hidden 4096 --compile-block --attn-name sdpa_causal --iters 5 --warmups 1 --profile --profile-dir /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_disabled_baseline_retry1/block_prefill/profile --no-profile-memory
```

The exact enabled command captured in the archive is:

```bash
cd /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/spyre-granite-e2e-bench
/home/adnan/dt-inductor/.venv/bin/python3 benchmarks/granite_block_layer_probe.py --fms-root /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/foundation-model-stack --run-root /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_collectives_backend1 --case prefill --seq-len 512 --batch 1 --hidden 4096 --compile-block --attn-name sdpa_causal --iters 5 --warmups 1 --profile --profile-dir /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_collectives_backend1/block_prefill/profile --no-profile-memory
```

The exact DXP replay command captured in the archive is:

```bash
cd /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_collectives_backend1/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_t_frrns5
dxp_standalone --bundle -d /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_collectives_backend1/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_t_frrns5
```

Full environment snapshots are in each run directory's `env.txt`; these are the
source of truth for reruns because the pod image inherits a large Spyre runtime
environment.
