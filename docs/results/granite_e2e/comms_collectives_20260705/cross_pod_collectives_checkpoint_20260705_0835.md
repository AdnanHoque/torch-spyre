# Cross-Pod DLDSC Collectives Checkpoint - 2026-07-05 08:35 UTC

This checkpoint records the state after running the DLDSC collectives lane across the three AIU pods. It is intentionally an artifact-branch note, not a production PR description.

## Executive State

The logical DLDSC classification path is working for the current flash/Granite attention RHS class: the backend sees `matmul_operand_broadcast` / `all_gather_replicate` plans and can replace HBM restickify rows with LX restickify rows in compile/runtime-smoke runs.

The feature is not value-correct end-to-end yet for the Granite/attention RHS matmul operand class. The current blocker is the physical realization of an all-gathered activation shard into the PT matmul KERNEL operand layout without full RHS materialization.

What is proven today:

- Flash script path exists and compiles/runs in smoke mode.
- Relayout-enabled flash removes the HBM restickify rows from generated SDSCs and emits backend plans.
- Granite S512 baseline is reproducible.
- Granite relayout-enabled path emits a real `matmul_operand_broadcast` plan and removes/replaces the first attention HBM restickify before failing.
- CDX synthetic M4 diagnostics show that fake fold metadata is not the correct fix for the KERNEL operand path.

What remains unproven:

- Value correctness for flash with collectives enabled. Current flash runbook uses `PATCH_MODE=no_h2d,skip_cpu_ref` and skips `assert_close`.
- Granite relayout-enabled profiled completion.
- Loop-scoped all-gather plus local layout conversion into KERNEL operand form.

## Pods Used

| Pod | Device | Role | Result |
|---|---:|---|---|
| `adnan-spyre-dev-pf` | `/dev/vfio/31` | flash attention compile/runtime-smoke verification | relayout-on passed smoke and replaced HBM restickifies with LX restickifies; correctness skipped |
| `adnan-clc-spyre-dev-pf` | `/dev/vfio/73` | Granite S512 baseline/relayout archive | baseline passed; relayout-on fails at `8_batchmatmul` LX chunk-fit blocker |
| `adnan-cdx-spyre-dev-pf` | `/dev/vfio/80` | synthetic M4 backend correctness debugging | identity fold seed removed `map::at`, then failed normal PT output fold; not a viable fix |

## DEV Flash Attention Result

Summary path:

```text
/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/dldsc_flash_collectives_lane_summary_20260705_083256/README.md
```

Script verified:

```text
/tmp/test-spyre-scripts/test_flash.py
repo: git@github.ibm.com:aviros/test-spyre-scripts.git
branch: main
sha: afda166e58b23519d0b4ca871350b011b56d91a3
```

Repository state:

```text
Torch:      ah/comms-collectives @ 8960d88af18e31033a75e36450d8b6efcf9cf301, clean
Deeptools: ah/comms-collectives @ 352919bf3f9c0efb2430568c667111aeb0a99e95, dirty util/LayoutAllgatherRestickify.cpp
DXP wrapper: /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/tools/dxp-split-wrapper/dxp_standalone
```

Results:

| Variant | Run dir | RC | SDSCs | HBM restickify | LX restickify | Backend plans | Correctness |
|---|---|---:|---:|---:|---:|---:|---|
| baseline relayout off | `/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/baseline_relayout_off_backend1_20260705_082034` | 0 | 550 | 33 files / 97 occurrences | 0 | 0 | skipped |
| relayout collectives on | `/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_backend1_20260705_082639` | 0 | 550 | 0 | 33 files / 97 occurrences | 32 | skipped |

Interpretation:

The flash script exercises the intended SDSC transformation: with relayout enabled, HBM restickifies disappear and LX restickifies plus backend `matmul_operand_broadcast` plans appear. This is not correctness evidence because the runbook skipped H2D and `assert_close`.

Next flash step:

Run a correctness-enabled version of the script after the backend KERNEL operand path is fixed, then collect profiler traces. Until then, flash is useful for artifact shape and DXP/runtime smoke only.

## CLC Granite S512 Result

Summary path:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/summary.md
```

Repository state:

```text
Torch:      ah/comms-collectives @ 8960d88af18e31033a75e36450d8b6efcf9cf301, clean
Deeptools: ah/comms-collectives @ 352919bf3f9c0efb2430568c667111aeb0a99e95, dirty util/LayoutAllgatherRestickify.cpp
```

Results:

| Variant | RC | process wall s | median wall ms | kernel ms/iter | memory ms/iter | ReStickifyOpHBM | ReStickifyOpLx | backend plans |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relayout disabled baseline retry | 0 | 93.133 | 30.5395 | 12.5466 | 0.306815 | 5 / 15 | 0 / 0 | 0 |
| relayout enabled collectives backend1 | 1 | 83.592 | none | none | none | 1 / 3 | 1 / 3 | 1 |

The first disabled attempt hit a transient runtime hardware state (`Compute CB hardware error detected`, then `StreamInErrorState`); the retry passed and is the baseline row above.

Enabled backend plan:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_collectives_backend1/backend_plans/8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
```

Plan properties:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
sdsc = 8_batchmatmul
logical_transfers = 512
groups = 2
producer_chunks_per_group = 16
consumer_replicas_per_group = 16
physical_lowering_status = lowered_loop_scoped_kernel_neighbor
realization_strategy = loop_scoped_input_fetch
```

Current Granite blocker:

```text
DtException: Unable to map graph within architecture constraints: The initial chunk parameters must fit in LX for SuperDSC: 8_batchmatmul
file: .../deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 1701
```

DXP replay reproduced the same blocker with backend LX frac 1:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/dxp_replay_enabled_failing_bundle_backend1/
return code = 134
```

Interpretation:

This confirms the Granite path reaches the useful attention RHS class and emits a backend-derived all-gather plan. It also confirms that full or incorrectly staged materialization exceeds available LX for the Granite attention bundle. The next backend implementation must be genuinely loop/tile scoped.

## CDX Synthetic M4 Backend Probe

Active CDX root:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
```

Synthetic run root:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506
```

Relevant runs:

| Run | Result | Meaning |
|---|---|---|
| `kernel_neighbor_carousel_M4_212148` | `ALLCLOSE True`, `MISMATCH 0 / 1024` | archived value-correct small case from older backend realization state |
| `normal_view_M4_075722` | `std::out_of_range map::at` in DDC fold | normal DDC path missing loop distribution params for artificial LX-neighbor transfer |
| `normal_view_fold_seed_M4_082113` | no `map::at`; then `Solution cannot be found` for `transfer_lds1_src:lxlu_dst:ptrow0` | identity fold seed removes missing-map crash but fails fill-data fold solving for KERNEL transfer |
| `normal_view_fill_guard_M4_082344` | artificial KERNEL transfer gets past fill-data; then normal `compute_ptrow0_fma16` output fold fails | decoupled fill-data guard still perturbs/fails normal PT coordinate solving |
| `compact_ring_fold_seed_M4_082602` | same normal PT output fold failure | compact-ring fallback is also broken under the identity fold seed change |

CDX experimental Deeptools edits made during this checkpoint:

- Added identity `loopParamsAfterDistribution` records before returning from the LX-neighbor fold guard in `dsc/dsc2.cpp`.
- Decoupled the `lx_neighbor_fill_data_guard` from `DEEPTOOLS_LX_NEIGHBOR_FORCE_VIEW_GUARD_SKIP` in `ddc/ddcv1.cpp` and `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp`.

These edits are not a solution. They are negative evidence. They show that treating the artificial KERNEL neighbor transfer as a mostly-normal fold with patched metadata is the wrong direction.

The archived passing M4 case attached the ring transfer to:

```text
transfer_lds1_src:no_component_dst:lx_lx_local
```

The current failing path creates or reasons about:

```text
transfer_lds1_src:lxlu_dst:ptrow<N>
```

That `lxlu -> ptrow` transfer is then interpreted by DDC/fill as a normal KERNEL data view, causing fold/placement failures or value-wrong output.

## Current Technical Conclusion

The substrate path is now clear:

1. DLDSC metadata can classify the edge.
2. Deeptools can synthesize logical all-gather transfer plans.
3. The remaining hard part is not cardinality; it is physical realization and scheduling of the matmul KERNEL operand.

The correct next implementation should not keep patching DDC with fake folds. It should introduce an explicit two-stage backend realization for this class:

```text
source activation LX layout
  -> ring all-gather into loop-local source-layout staging
  -> same-core local copy/read for local chunks, never self-ring
  -> local ReStickifyOpLx / layout conversion into final KERNEL operand tile
  -> PT matmul consumes final KERNEL tile
```

Acceptance for the next implementation:

- synthetic M4/M16/M64 value-correct without `DEEPTOOLS_LX_NEIGHBOR_FORCE_VIEW_GUARD_SKIP`;
- no self-ring for same-core pieces;
- no full RHS materialization for Granite attention;
- Granite S512 relayout-enabled run reaches profiling and reduces non-weight HBM restickifies beyond the scatter baseline;
- flash script runs correctness-enabled once backend correctness is established.
