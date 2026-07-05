# Backend All-Gather Diagnostic Checkpoint - 2026-07-05

This checkpoint records the current state of the DLDSC LX-relayout collectives effort after the CDX synthetic matmul-operand all-gather diagnostics, the CLC Granite S512 spill classification, and the DEV flash compile-probe.

It is an artifact-branch note, not a production design. It intentionally avoids broad claims: the inductor-side DLDSC contract can now describe the relevant communication classes, but the backend realization for loop-scoped all-gather into a matmul KERNEL operand is not value-correct yet.

## Current Scope Answer

For the latest Granite S512 artifacts, the in-scope non-weight HBM spill is the attention activation layout-restickify/all-gather edge. The remaining HBM root restickifies in the latest disabled baseline are weights and are out of scope for this lane because weight preloading/prelayout should handle them separately.

Communication class status from current artifacts:

| Class | Observed in current Granite/flash artifacts | Current status |
|---|---|---|
| scatter / disjoint 1:1 remap | Not the active Granite spill class | Production scatter path is separate and not the blocker here |
| broadcast / multicast | Not standalone in Granite S512; matmul operand path has broadcast-like replication | Needs backend KERNEL-neighbor realization |
| gather | Not observed as a standalone remaining spill | Not proven by current artifacts |
| all-gather | Yes: `matmul_operand_broadcast`, `communication_pattern=all_gather_replicate` | Classified by Torch/DLDSC; backend realization incomplete |
| layout-restickify | Yes: attention activation relayout | HBM restickify converts to LX in compile-probe, but E2E value path blocked by backend all-gather realization |
| reduce / all-reduce | Not observed in this Granite S512 spill set | Future work |

## CLC Granite S512 Classification

Read-only classification was done on pod `adnan-clc-spyre-dev-pf` under:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840
```

Latest disabled baseline has five HBM root restickifies:

- Four are weight restickifies and are out of scope:
  - QKV/linear weight, shape `6144x4096`
  - attention output weight, shape `4096x4096`
  - MLP up/gate weight, shape `25600x4096`
  - MLP down weight, shape `4096x12800`
- One is in scope:
  - attention activation layout restickify, shape `{mb=32,x=512,out=128}`
  - disabled path: `ReStickifyOpHBM`
  - relayout path: `ReStickifyOpLx`
  - communication class: layout-restickify plus all-gather-replicate into a matmul operand

The latest enabled Granite path emits a backend plan:

```text
relayout_enabled_collectives_backend1/backend_plans/8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
```

Important plan fields:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
logical_transfers = 512
groups = 2
producer_chunks_per_group = 16
consumer_replicas_per_group = 16
physical_lowering_status = lowered_loop_scoped_kernel_neighbor
realization_strategy = loop_scoped_input_fetch
```

Current CLC failure:

```text
Unable to map graph within architecture constraints:
The initial chunk parameters must fit in LX for SuperDSC: 8_batchmatmul
```

Interpretation: frontend classification and DLDSC metadata are present. Backend currently tries a realization that is not a valid loop-scoped/staged all-gather for this KERNEL operand.

## DEV Flash Compile-Probe

Read-only verification was done on pod `adnan-spyre-dev-pf` under:

```text
/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129
```

Latest dirs:

```text
runs/dldsc_flash_collectives_lane_summary_20260705_083256
runs/baseline_relayout_off_backend1_20260705_082034
runs/relayout_on_backend1_20260705_082639
```

Counts:

| Variant | ReStickifyOpHBM files | ReStickifyOpHBM occurrences | ReStickifyOpLx | Backend plans |
|---|---:|---:|---:|---:|
| baseline relayout off | 33 | 97 | 0 | 0 |
| relayout on | 0 | 0 | 33 files / 97 occurrences | 32 |

This is compile-probe evidence only. The run used:

```text
PATCH_MODE=no_h2d,skip_cpu_ref
```

So it proves SDSC lowering/conversion from HBM restickify to LX restickify for the flash compile path. It does not prove numerical correctness or profiler speedup.

## CDX Synthetic M4 Backend Diagnostics

Active CDX workspace:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
```

Synthetic run root:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506
```

Known-good baseline without relayout:

```text
baseline_no_relayout_M4_044702
ALLCLOSE True
MAX_DIFF 0.0
MISMATCH 0 / 1024
ROWMAP_OUT0 [0.0, 0.25, 0.5, 0.75]
```

### Diagnostic results

| Run | Change tested | Result | Interpretation |
|---|---|---|---|
| `two_stage_no_skip_redistrib_M4_085923` | Let KERNEL-neighbor `lxlu -> ptrow` transfer build DDC fold instead of skipping | Fails with `Not enough elements to distribute`, owner `transfer_lds1_src:lxlu_dst:ptrow0`, `corelet_slice` wants 256 but elemArr has 128 | DDC is applying PT corelet split to a source-layout staging view; KERNEL-neighbor path needs a fold special case |
| `two_stage_neutral_corelet_M4_090528` | Neutralize synthetic `corelet_slice` only for two-stage KERNEL-neighbor transfer | Moves to `fillDataInfo` failure on `l3_lx_kernel` / `xrf_kernel` | Fold progressed; address/fill-data still expects normal KERNEL transfer semantics |
| `two_stage_fold_fill_guard_M4_090855` | Also guard KERNEL-neighbor fill-data address scaling under two-stage diagnostic | Moves to PT output fold failure on `compute_ptrow0_fma16`, `allocate_lds2_ptarf` | Forcing transfer fold construction contaminates normal PT output fold state; this is not the clean path |
| `two_stage_order_ring_first_oldguard_M4_091118` | Keep old transfer-fold skip, emit ring PCFG before local PCFG | Runs but output all zeros | Ordering alone is insufficient; local LXLU/LXSU stage is not value-correct |
| `two_stage_schedule_before_consumer_M4_092039` | Move synthetic PCFG-only schedule from bundle front to before last DLDSC step | Runs but output all zeros | The synthetic local-copy realization still does not put values where PT consumes them, or it is not synchronized as assumed |

The latest value failure:

```text
RUN_DIR=/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506/two_stage_schedule_before_consumer_M4_092039
ALLCLOSE False
MAX_DIFF 0.75
MISMATCH 1013 / 1024
ROWMAP_OUT0 [0.0, 0.0, 0.0, 0.0]
ROWMAP_REF0 [0.0, 0.25, 0.5, 0.75]
```

The synthetic transfer has both ring and local stages:

```text
transfer_lds1_src:no_component_dst:lx_lx_local
ring=1024
local=1024
```

But the final values are still zero, including same-core chunks. That points away from ring-only bugs and toward the local KERNEL-neighbor copy/conversion carrier or its schedule/dependency.

## Current Backend Interpretation

The current exploratory backend has two partial paths:

1. DDC-folded `lxlu -> ptrow` transfer path:
   - can be made to pass the first KERNEL-neighbor input fold with a diagnostic corelet bypass and fill-data guard;
   - then fails normal PT output fold construction;
   - conclusion: this path is too invasive unless Deeptools gets a principled KERNEL-neighbor coordinate contract.

2. Transfer-fold-skipped plus synthetic ring/local PCFG path:
   - compiles and runs;
   - emits ring and LXLU/LXSU local stages;
   - still produces zeros;
   - conclusion: current local stage is not a valid/staged/synchronized local restickify into the exact KERNEL operand view consumed by PT.

The likely production backend shape is therefore not “append a free-standing LXLU/LXSU copy pair and hope PT sees it.” It needs one of:

- a real loop-scoped `ReStickifyOpLx`/DL op carrier inserted at the right schedule point;
- an explicit backend local-copy/restickify primitive with well-defined staging allocation, final destination allocation, and schedule dependency before PT operand fetch;
- or a KERNEL-neighbor extension in Deeptools that treats the all-gather plus local layout conversion as part of the matmul operand transfer loop, with DDC/DDC-fill support instead of diagnostic guards.

## Current Experimental Patch Location

The CDX Deeptools checkout contains unpushed exploratory edits here:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/repos/deeptools-comms-clean
```

A diagnostic diff was archived at:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506/patches/two_stage_kernel_neighbor_diagnostic_20260705.diff
```

This diff is intentionally not production-ready. It is evidence for what was tried and why the next implementation should avoid re-discovering the same dead ends.

## Next Recommended Step

Do not keep extending the current free-standing local LXLU/LXSU stage unless we first prove it can perform a simple same-core LX-to-LX copy before a PT consumer.

The next smallest useful probes are:

1. Same-core local-copy litmus:
   - no ring;
   - one core;
   - producer writes patterned LX data;
   - local stage copies source LX address to destination LX address;
   - consumer reads destination.
   - If this fails, abandon the LXLU/LXSU standalone carrier.

2. Existing-carrier probe:
   - express the local layout-conversion leg through existing `ReStickifyOpLx` or `STCDPOpLx` machinery, where LXLU/LXSU FIFO sequencing is already known to work inside an op PCFG.
   - Keep ring all-gather staging separate.

3. Schedule-contract probe:
   - export the post-DXP SuperDSC or generated schedule with the PCFG-only step and verify it is ordered after the producer DLDSC and before the consumer DLDSC on every participating core.

4. Only after the synthetic M4 path is value-correct:
   - retry Granite S512 with split frontend/backend LX frac env;
   - retry flash with real H2D and CPU reference enabled;
   - then collect profiler traces.


## Repro Commands

Build CDX DXP:

```bash
kubectl exec -n a6-quantization adnan-cdx-spyre-dev-pf -- bash -lc '
set -euo pipefail
/home/adnan-cdx/dt-inductor-mixed/.venv/bin/cmake --build /home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/build-deeptools-comms-clean-fast --target dxp_standalone -j8
'
```

Run the latest old-guard two-stage M4 diagnostic:

```bash
kubectl exec -n a6-quantization adnan-cdx-spyre-dev-pf -- bash -lc '
set -euo pipefail
ROOT=/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
RUN=$ROOT/runs/min_stable_matmul_operand_broadcast_20260704_100506
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_TWO_STAGE_DIAGNOSTIC=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_EMIT_LOCAL_RESTICKIFY_STAGE=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_DIRECT_ALLGATHER_DIAGNOSTIC=1
export DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1
export DEEPTOOLS_LX_NEIGHBOR_FORCE_VIEW_GUARD_SKIP=1
export DEEPTOOLS_LX_NEIGHBOR_VIEW_PROBE=1
export DEEPTOOLS_LX_NEIGHBOR_FOLD_GUARD_DIAGNOSTIC=1
CASE=two_stage_next_M4_$(date +%H%M%S)
bash "$RUN/run_one_64diag.sh" "$CASE" mul_one 1 4 64 256 >/tmp/${CASE}.driver 2>&1 || true
echo RUN_DIR=$RUN/$CASE
grep -E "ALLCLOSE|MAX_DIFF|MISMATCH|ROWMAP|DtException|map::at|Solution cannot|lx_neighbor" "$RUN/$CASE/run.log" | sed -n "1,240p" || true
'
```

Copy the latest diagnostic patch from CDX if needed:

```bash
kubectl exec -n a6-quantization adnan-cdx-spyre-dev-pf -- bash -lc '
ls -l /home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506/patches/two_stage_kernel_neighbor_diagnostic_20260705.diff
'
```
