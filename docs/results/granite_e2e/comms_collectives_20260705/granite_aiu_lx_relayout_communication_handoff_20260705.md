# Granite / AIU LX Relayout Communication Handoff - 2026-07-05

This is a first-principles handoff for the Granite/AIU LX relayout
communication work. It is written for a human or agent with no prior context.
It describes the problem, the current state, the approaches that were tried,
where the code and artifacts live, how to reproduce the current results, and
what to do next.

This is an artifact-branch document. Keep broad notes and run artifacts on
`AdnanHoque/torch-spyre:ah/comms-collectives`. Do not move this material onto
production PR branches unless Adnan explicitly asks.

## One-Page Objective And Current Status

Objective:

Remove in-scope non-weight HBM round trips in Granite and flash attention by
using DLDSC-described LX-to-LX movement, local LX layout conversion, and backend
scheduled communication on AIU. The useful long-term contract is:

```text
Torch describes logical tensor residency and consumer need.
Deeptools derives legal physical movement, local conversion, and schedule.
```

Why this matters:

- AIU cores compute on per-core LX scratchpad data.
- HBM is much larger, but using HBM as a handoff between producer and consumer
  creates avoidable activation traffic.
- The current Granite attention RHS blocker is not a plain copy. It is grouped
  all-gather of activation shards plus local conversion into the PT matmul
  `KERNEL` operand layout.

Current high-level status:

| Area | State |
|---|---|
| PR1 scatter lane | Production-shaped first class. It covers same-layout tensor ownership mismatch: producer and consumer need the same logical/layout form, but on different cores. |
| Granite attention RHS | Still blocked. Classified as `matmul_operand_broadcast` / `all_gather_replicate`, but backend physical realization is not value-correct and Granite-scale enabled runs hit LX chunk-fit failure. |
| Flash attention smoke | Relayout-on flash compile/runtime smoke can replace explicit `ReStickifyOpHBM` with `ReStickifyOpLx` and emit backend plans, but current smoke uses `PATCH_MODE=no_h2d,skip_cpu_ref`; it is not value correctness. |
| Synthetic M4/M16 | Baseline and some archived staged/full-resident cases are value-correct. Current loop-scoped all-gather into `KERNEL` operand either crashes in DDC/fold metadata paths or runs with wrong/all-zero values. |
| Backend direction | Stop extending free-standing fake `lxlu -> ptrow` folds unless a local-copy litmus proves the carrier. Prefer a real loop-scoped two-stage realization: source-layout ring gather, same-core local copy/read, local `ReStickifyOpLx`-style conversion to final `KERNEL` tile. |

The immediate next useful experiment is a small same-core local LX-to-LX copy
litmus before PT consumption. If that fails, abandon the standalone LXLU/LXSU
carrier for this class and route local conversion through `ReStickifyOpLx`,
`STCDPOpLx`, or a backend-owned schedule-local primitive with complete fold and
coordinate metadata.

## First-Principles Model

Read an AIU DLDSC program in this lane as core-local computations connected by
storage and movement constraints:

- AIU cores own compute tiles. A work division maps logical dimensions such as
  `mb`, `x`, `in`, and `out` onto cores.
- LX is per-core scratchpad. It is fast, local, and capacity limited.
- HBM is off-core memory. HBM activation handoffs are the spill class this work
  is trying to remove.
- The ring moves data between AIU cores. It is for cross-core LX-to-LX traffic.
  It is not a valid same-core copy mechanism; a self-ring diagnostic fenced a
  device.
- PT matmul consumes `KERNEL` operand views. A source activation in LX can still
  be unusable by PT until it is laid out and stickified in the exact physical
  form the matmul reader expects.

The important DLDSC distinction is tensor distribution versus compute
distribution:

- Tensor distribution says where source tensor coordinates currently live:
  core id, coordinate slice, memory component, layout order, stick order, and
  allocation facts.
- Compute distribution says which logical output/compute coordinates the
  consumer core is responsible for.
- A relayout edge exists when a consumer needs a logical coordinate range that
  is not already present on that consumer core in the required physical layout.

The active Granite/attention class is:

```text
source activation LX layout
  -> grouped all-gather / broadcast across consumer cores
  -> same-core local copy or direct read for local chunks
  -> local source-layout-to-KERNEL-layout conversion
  -> PT batchmatmul consumes final KERNEL operand tile
```

This is why direct ring writes into the final `KERNEL` address range are wrong:
they move bytes, but they skip the source-layout-to-`KERNEL` layout conversion.

## Branch, Pod, And Repo Map

Use the pod-local workspace unless it disappears:

```text
namespace: a6-quantization
pod:       adnan-cdx-spyre-dev-pf
root:      /home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
```

Torch artifact checkout:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/repos/torch-spyre
branch: ah/comms-collectives
remote: git@github.com-adnan-cdx-spyre-dev-pf:AdnanHoque/torch-spyre.git
head before this doc commit: e8d9b6e104911473f515108aeb5e083c5c13371c
```

Deeptools experiment checkout:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/repos/deeptools-comms-clean
branch: ah/comms-collectives
remote: git@github.ibm.com:Adnan-Hoque1/deeptools.git
current observed head: fa30750e1 [DXP] preserve matmul operand source target layout contract
```

Do not revert Deeptools dirty files. The dirty state is intentional exploratory
work. Current observed dirty files include:

```text
dcc/src/Conversion/PCFGToDataflowIR/PCFGToDFManager.cpp
dcc/src/Conversion/PCFGToDataflowIR/PCFGToDataflowIR.cpp
dcc/src/Conversion/SentientToProgIR/Utils.cpp
dcc/src/Stitcher/ModuleStitcher.cpp
dcg/dcg_fe/pcfg_gen/dlOps.cpp
dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp
dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp
dcg/dcg_fe/transfer_compute/transfer_compute.cpp
dcg/dcg_manager/dcg_manager.cpp
ddc/ddc_fold.cpp
ddc/ddcv1.cpp
dsc/dsc2.cpp
dsc/dsc2.h
dsc/dsc2Pcfg.cpp
dxp/SdscRelayoutInsertion.cpp
util/LayoutAllgatherRestickify.cpp
```

Important build/runtime paths:

```text
DXP build:
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/build-deeptools-comms-clean-fast

DXP split wrapper:
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/tools/dxp-split-wrapper/dxp_standalone

Synthetic M4/M16 root:
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506
```

Other pods used by the current artifact set:

| Pod | Device | Role |
|---|---:|---|
| `adnan-cdx-spyre-dev-pf` | `/dev/vfio/80` | Active synthetic M4/M16 RHS operand broadcast and backend diagnostics |
| `adnan-spyre-dev-pf` | `/dev/vfio/31` | Flash attention smoke/value follow-up runs |
| `adnan-clc-spyre-dev-pf` | `/dev/vfio/73` | Granite S512 baseline and relayout archive |

## Current Code And Progress Locations

Torch-side code paths:

| Path | Role |
|---|---|
| `torch_spyre/_inductor/config.py` | Frontend env flags for relayout and collectives. |
| `torch_spyre/_inductor/lx_relayout.py` | DLDSC tensor-vs-compute relayout classification and contract emission. |
| `torch_spyre/_inductor/layout_allgather_restickify.py` | Pure-Python contract/classifier helpers for `layout_allgather_restickify` and `matmul_operand_broadcast`. |
| `torch_spyre/_inductor/codegen/bundle.py` | Filters/propagates relayout classification metadata into bundle output. |
| `torch_spyre/_inductor/codegen/superdsc.py` | SuperDSC/DLDSC emission context. |
| `torch_spyre/_inductor/codegen/compute_ops.py` | Consumer operand metadata and layout facts. |
| `torch_spyre/_inductor/scratchpad/allocator.py` | LX allocation/planning interaction. |
| `torch_spyre/_inductor/spyre_kernel.py` | Runtime/bundle planning path and DLDSC classification plumbing. |

Deeptools-side code paths:

| Path | Role |
|---|---|
| `util/LayoutAllgatherRestickify.cpp` | Validates/synthesizes `layout_allgather_restickify` and `matmul_operand_broadcast` movement plans. |
| `dxp/SdscRelayoutInsertion.cpp` | DXP hook that sees DLDSC metadata, emits plan artifacts, and prototypes STCDP/restickify/kernel-neighbor insertion strategies. |
| `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp` | Active loop-scoped matmul operand broadcast and KERNEL-neighbor scheduling experiments. |
| `dsc/dsc2Pcfg.cpp` | Lowers LX-neighbor ring transfer side tables into PCFG send/recv nodes. |
| `dsc/dsc2.cpp`, `dsc/dsc2.h` | Distribution/fold metadata experiments for synthetic transfer nodes. |
| `ddc/ddc_fold.cpp`, `ddc/ddcv1.cpp` | DDC fold/fill paths that currently fail or need complete metadata for artificial transfers. |
| `dcg/dcg_fe/transfer_compute/transfer_compute.cpp` | Existing `STCDPOpLx` and `ReStickifyOpLx` data-op mechanics. |

Key artifact docs already on the branch:

```text
docs/results/granite_e2e/comms_collectives_handoff_20260705.md
docs/results/granite_e2e/comms_collectives_20260705/collective_class_status_20260705.md
docs/results/granite_e2e/comms_collectives_20260705/backend_allgather_diagnostic_checkpoint_20260705.md
docs/results/granite_e2e/comms_collectives_20260705/cross_pod_collectives_checkpoint_20260705_0835.md
docs/results/granite_e2e/comms_collectives_20260705/m4_backend_checkpoint_20260705.md
docs/results/granite_e2e/comms_collectives_20260705/matmul_operand_broadcast_loop_scoped_checkpoint_20260705.md
docs/results/granite_e2e/comms_collectives_20260705/two_stage_matmul_operand_broadcast_plan_20260705.md
```

## Approaches Considered

### Coordinate-Remap Data-DSC

What it was:

Torch explicitly computed source/destination coordinate rows and inserted a
coordinate-remap data op or mixed SDSC carrier between producer and consumer.

What worked:

- Proved that avoiding HBM activation handoffs can improve Granite block
  timing.
- Produced concrete source/destination row artifacts that were easy to inspect.
- Was useful for early debugging because the frontend controlled exact movement
  rows.

What failed or aged out:

- Data-DSC support was deprecated in the SuperDSC bundle direction.
- It pushed too much physical movement scheduling into Torch.
- It did not scale cleanly to fanout, fanin, all-gather, local layout
  conversion, or future arithmetic collectives.

Conclusion:

Keep the evidence. Do not use this as the production contract.

### `STCDPOpLx` / Ranged Carrier

What it was:

Use existing LX movement vocabulary and `STCDPOpLx`-style data-op mechanics as a
carrier for ranges or grouped movement.

What worked:

- It reused an existing backend concept rather than inventing a public physical
  movement op.
- Small full-resident `STCDPOpLx -> ReStickifyOpLx` staged cases can be
  value-correct.
- It is still a plausible backend building block for the local or full-resident
  path.

What failed:

- Coarse data-op insertion materializes too much for Granite attention RHS.
- A range-transfer list is still a physical carrier, not the full logical
  tensor-vs-compute contract.
- KERNEL operands require final layout/stick conversion, not only byte movement.

Conclusion:

Use `STCDPOpLx` as a possible internal primitive, not as the top-level Torch
contract for this feature.

### DLDSC Backend Relayout Contract

What it is:

Torch emits DLDSC metadata that describes tensor distribution, consumer compute
distribution, and source/target layout metadata. Deeptools classifies
incompatibilities and inserts/schedules movement.

What worked:

- Same-layout scatter is production-shaped and belongs in PR1.
- DLDSC coordinate maps can express scatter, broadcast/multicast, gather, and
  all-gather cardinality when relevant split-1 dimensions are explicitly present
  as slice `0`.
- The flash/Granite attention RHS class is now visible as
  `matmul_operand_broadcast` / `all_gather_replicate` in backend plan artifacts.

What is incomplete:

- The current backend does not yet have a value-correct loop-scoped realization
  for all-gather plus local layout conversion into a PT `KERNEL` operand.
- Destination allocation, same-core local movement, fold metadata, and schedule
  dependencies must be handled as one backend-owned contract.

Conclusion:

This is the desired long-term contract. The remaining work is physical backend
realization, not basic frontend classification.

### PR-Style Explicit Movement Versus DLDSC Metadata

The production scatter PR should stay narrow. It should demonstrate a clean
DLDSC tensor-vs-compute mismatch contract for same-layout scatter. It should not
carry broad artifact docs, full Granite run output, or every exploratory
backend carrier.

Explicit movement prototypes were still useful:

- They identified which movement classes appear in Granite.
- They proved some small staged paths can be value-correct.
- They exposed backend gaps in fold, allocation, and KERNEL operand binding.

But production direction is not "Torch emits ring schedules." Production
direction is "Torch emits enough logical metadata for Deeptools to synthesize
ring/local movement, layout conversion, and schedule."

### Direct KERNEL-Neighbor / Loop-Scoped Prototype

What it was:

Try to make the Granite attention RHS path capacity-safe by avoiding
full-resident materialization and inserting movement inside the matmul operand
loop.

What worked:

- Backend can emit L3 ring send/recv PCFG nodes.
- Archived small synthetic M4 carousel run was value-correct:

```text
kernel_neighbor_carousel_M4_212148
ALLCLOSE True
MISMATCH 0 / 1024
```

- Diagnostics narrowed failures to DDC/fold/fill metadata and KERNEL operand
  local-copy/layout binding.

What failed:

- Direct ring writes into final `KERNEL` layout are value-wrong.
- Same-core self-ring is unsafe.
- Free-standing synthetic `transfer_lds1_src:lxlu_dst:ptrow*` nodes crash in
  DDC/fold paths unless guarded, and broad guards produce value-wrong output.
- Recent two-stage old-guard runs emit ring/local stages but still produce all
  zeros.

Conclusion:

Keep the loop-scoped objective, but replace the fragile free-standing local
copy/fold path with a real schedule-local carrier or complete metadata contract.

## Communication Taxonomy

| Class | First-principles meaning | Current status | Main gap |
|---|---|---|---|
| Scatter | One source shard moves to one unique destination shard with compatible layout. | PR1 production-shaped same-layout class. | Keep scope tight, add capacity guards and tests. |
| Broadcast / multicast | One source shard feeds multiple destination cores. | Partial same-layout synthetic evidence. | Dedicated production metadata/allocation contract and efficient fanout. |
| Gather | Multiple source shards feed one destination core. | Partial synthetic evidence when destination allocation is safe. | Backend-owned non-overlap/destination allocation. |
| All-gather | Many or all source shards are made available to many or all consumers. | Classified for attention RHS as `all_gather_replicate`; same-layout small cases have evidence. | Loop/tile-scoped staging, same-core handling, fold/cardinality metadata. |
| Reduce | Multiple source values are combined arithmetically into one output shard. | Not covered by relayout-only work. | Needs reduction op, axes, dtype, identity, accumulation, and schedule. |
| All-reduce | Reduce plus redistribute/broadcast result. | Not covered. | Build reduce first, then distribute result. |
| Local layout conversion | Same values, same or local core, different physical stick/layout form. | `ReStickifyOpLx` exists and small staged paths can work. | Schedule-local conversion into PT `KERNEL` operand without full RHS materialization. |

Granite attention RHS is a composite:

```text
all-gather/broadcast + local layout conversion into KERNEL
```

Treating it as a plain all-gather or scatter loses the KERNEL operand layout
semantics.

## What PR1 Covers And What Remains

PR1 covers:

- Same-layout scatter/direct remap.
- Producer tensor distribution and consumer compute distribution differ by core
  ownership.
- The tensor layout/stick form is already compatible with the consumer.
- Deeptools derives and inserts LX relayout from DLDSC metadata.
- This is the right first production step because it proves the metadata
  contract without solving every collective.

PR1 does not cover:

- Broadcast/multicast fanout as a production feature.
- Gather/all-gather fanin/fanout with grouped replicas.
- Matmul RHS `KERNEL` operand layout conversion.
- Loop-scoped source-layout staging plus local restickify.
- Arithmetic collectives: reduce and all-reduce.
- Weight restickifies, which belong to preload/prelayout, not this activation
  communication lane.

The remaining Granite blocker is:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
consumer_operand_ds_type = KERNEL
```

## Current Evidence And Artifacts

### Granite S512

Latest CLC archive root:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840
```

Summary path:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/summary.md
```

Current result:

| Variant | RC | Kernel ms/iter | HBM restickify | LX restickify | Backend plans |
|---|---:|---:|---:|---:|---:|
| relayout disabled baseline retry | 0 | 12.5466 | 5 files / 15 occurrences | 0 | 0 |
| relayout enabled collectives backend1 | 1 | none | 1 file / 3 occurrences | 1 file / 3 occurrences | 1 |

Enabled backend plan:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_collectives_backend1/backend_plans/8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
```

Plan facts:

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

Current failure:

```text
DtException: Unable to map graph within architecture constraints:
The initial chunk parameters must fit in LX for SuperDSC: 8_batchmatmul
```

Interpretation:

Granite reaches the right attention RHS class and emits a backend-derived plan,
but the current realization still exceeds or misuses LX capacity. The next
backend implementation must be genuinely loop/tile scoped.

### Flash Attention

DEV root:

```text
/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129
```

Latest smoke result examples:

```text
/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_verify_backend1_20260705_075332
/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_verify_backend1_20260705_174127
```

Observed shape:

| Run | RC | SDSCs | Backend plans | Matmul operand plans | HBM restickify | LX restickify |
|---|---:|---:|---:|---:|---:|---:|
| archived `075332` | 0 | 550 | 32 | 32 | 0 | 33 files / 97 occurrences |
| fresh `174127` | 0 | 550 | 32 | 32 | 0 | 33 files / 97 occurrences |

Important caveat:

```text
PATCH_MODE=no_h2d,skip_cpu_ref
```

This proves structural compile/runtime mechanics and HBM-to-LX restickify
replacement in a smoke configuration. It does not prove value correctness.

### CDX Synthetic M4/M16

Run root:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506
```

Useful known runs:

| Run | Result | Meaning |
|---|---|---|
| `baseline_no_relayout_M4_044702` | `ALLCLOSE True`, `MISMATCH 0 / 1024` | HBM/no-relayout control is correct. |
| `kernel_neighbor_carousel_M4_212148` | `ALLCLOSE True`, `MISMATCH 0 / 1024` | Archived small passing carousel state. |
| `single_stage_prefer_finalized_M4_174304` | `std::out_of_range map::at` | DDC fold fails on artificial `lxlu -> ptrow` transfer. |
| `single_stage_foldskip_M4_174558` | `ALLCLOSE False`, `MISMATCH 949 / 1024` | Skipping fold compiles/runs but is value-wrong. |
| `single_stage_missing_loop_guard_M4_175509` | later `map::at` in loop-element-offset computation | Earlier fold guard moved failure forward; metadata is still incomplete. |
| `two_stage_schedule_before_consumer_M4_092039` | `ALLCLOSE False`, `MISMATCH 1013 / 1024`, row output all zeros | Ring/local stages are present, but the local KERNEL path is not value-correct. |

Most important negative lesson:

Fake folds and broad guards are not a solution. The missing metadata is part of
how the PT consumer finds the correct KERNEL operand view. If a guard lets the
program run but values are wrong, the guard is hiding a real semantic problem.

## Environment Knobs And Runbook Notes

Frontend relayout flags:

| Env var | Typical value | Meaning |
|---|---:|---|
| `SPYRE_LX_PLANNING` | `1` | Enable LX planning path. |
| `SPYRE_LX_PLANNER_RELAYOUT` | `1` for enabled, `0` for baseline | Top-level frontend relayout switch. |
| `SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES` | `1` | Enable collective classification/plans. |
| `SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY` | `1` | Enable layout-all-gather restickify metadata. |
| `SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT` | `1` | Emit matmul operand broadcast contracts. |
| `SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS` | `1` | Allow HBM restickify replacement with LX restickify where classified. |
| `LX_BOUNDARY_CLONES` | `1` | Boundary clone plumbing needed by current relayout probes. |

Backend diagnostic flags:

| Env var | Use |
|---|---|
| `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1` | Try loop-scoped KERNEL-neighbor matmul operand path. |
| `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_TWO_STAGE_DIAGNOSTIC=1` | Try two-stage diagnostic path where available. |
| `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_EMIT_LOCAL_RESTICKIFY_STAGE=1` | Emit local stage in two-stage diagnostic variants. |
| `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_DIRECT_ALLGATHER_DIAGNOSTIC=1` | Direct all-gather diagnostic, known value-wrong for KERNEL conversion. |
| `DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1` | Diagnostic escape hatch for mixed HBM/input-neighbor cases. |
| `DEEPTOOLS_LX_NEIGHBOR_VIEW_PROBE=1` | Extra logging for LX-neighbor view/debug state. |
| `DEEPTOOLS_LX_NEIGHBOR_FORCE_VIEW_GUARD_SKIP=1` | Diagnostic guard path. Avoid treating passing compile as correctness. |
| `DEEPTOOLS_LX_NEIGHBOR_FOLD_GUARD_DIAGNOSTIC=1` | Diagnostic fold guard. Broad fold skips are value-risky. |

Split LX capacity env:

The wrapper:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/tools/dxp-split-wrapper/dxp_standalone
```

maps:

```text
DXP_BACKEND_LX_FRAC_AVAIL -> DXP_LX_FRAC_AVAIL
```

This split matters because the same `DXP_LX_FRAC_AVAIL` knob has different
practical uses in frontend and backend:

- For Torch/frontend, `DXP_LX_FRAC_AVAIL=0` lets frontend planning assume full
  LX.
- For DXP/backend, `DXP_LX_FRAC_AVAIL=0` means no backend chunk space and causes
  failures such as "initial chunk parameters must fit in LX".

Use this shape for current relayout experiments:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
export PATH=/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/tools/dxp-split-wrapper:$PATH
```

For quick sweeps, try backend fractions `0.2` and `1.0` first.

## Reproduction Commands

### Check CDX Branch State

```bash
oc exec -n a6-quantization adnan-cdx-spyre-dev-pf -- bash -lc '
set -euo pipefail
ROOT=/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
for d in "$ROOT/repos/torch-spyre" "$ROOT/repos/deeptools-comms-clean"; do
  echo "== $d =="
  cd "$d"
  git status --short --branch
  git rev-parse HEAD
done
'
```

### Build CDX DXP

```bash
oc exec -n a6-quantization adnan-cdx-spyre-dev-pf -- bash -lc '
set -euo pipefail
/home/adnan-cdx/dt-inductor-mixed/.venv/bin/cmake \
  --build /home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/build-deeptools-comms-clean-fast \
  --target dxp_standalone \
  -j8
'
```

### Run The Current M4 Diagnostic

```bash
oc exec -n a6-quantization adnan-cdx-spyre-dev-pf -- bash -lc '
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

CASE=handoff_two_stage_M4_$(date +%H%M%S)
bash "$RUN/run_one_64diag.sh" "$CASE" mul_one 1 4 64 256 >/tmp/${CASE}.driver 2>&1 || true
echo RUN_DIR=$RUN/$CASE
grep -E "ALLCLOSE|MAX_DIFF|MISMATCH|ROWMAP|DtException|map::at|Solution cannot|lx_neighbor|matmul_operand" "$RUN/$CASE/run.log" | sed -n "1,240p" || true
'
```

Acceptance for this class is not "it compiles." For M4, require:

```text
ALLCLOSE True
MISMATCH 0 / 1024
ROWMAP_OUT0 matches ROWMAP_REF0
```

Then scale to M16/M64 and source split variants.

### Run Granite S512 Baseline And Enabled Paths

Use the CLC root:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404
```

Relayout-off baseline shape:

```bash
oc exec -n a6-quantization adnan-clc-spyre-dev-pf -- bash -lc '
set -euo pipefail
ROOT=/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404
RUN=$ROOT/runs/granite_s512_split_backend1_relayout_off_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"

export PATH="$ROOT/tools/dxp-split-wrapper:$ROOT/deeptools/build-deeptools/dxp:$PATH"
export PYTHONPATH="$ROOT/torch-spyre:$ROOT/torch-spyre/tests/inductor:$ROOT/foundation-model-stack:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="$RUN/block_prefill/cache"
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
export SPYRE_LX_PLANNING=1
export SPYRE_LX_PLANNER_RELAYOUT=0
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=0
export SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=0
export SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=0
export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=0
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR="$RUN/backend_plans"
export DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR="$RUN/backend_plans"
export SPYRE_LX_RELAYOUT_PLAN_DIR="$RUN/lx_relayout_plans"

cd "$ROOT/spyre-granite-e2e-bench"
/home/adnan/dt-inductor/.venv/bin/python3 benchmarks/granite_block_layer_probe.py \
  --fms-root "$ROOT/foundation-model-stack" \
  --run-root "$RUN" \
  --case prefill \
  --seq-len 512 \
  --batch 1 \
  --hidden 4096 \
  --compile-block \
  --attn-name sdpa_causal \
  --iters 5 \
  --warmups 1 \
  --profile \
  --profile-dir "$RUN/block_prefill/profile" \
  --no-profile-memory
echo RUN=$RUN
'
```

Relayout-on enabled shape:

```bash
oc exec -n a6-quantization adnan-clc-spyre-dev-pf -- bash -lc '
set -euo pipefail
ROOT=/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404
RUN=$ROOT/runs/granite_s512_split_backend1_relayout_on_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"

export PATH="$ROOT/tools/dxp-split-wrapper:$ROOT/deeptools/build-deeptools/dxp:$PATH"
export PYTHONPATH="$ROOT/torch-spyre:$ROOT/torch-spyre/tests/inductor:$ROOT/foundation-model-stack:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="$RUN/block_prefill/cache"
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
export SPYRE_LX_PLANNING=1
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
export SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
export SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
export LX_BOUNDARY_CLONES=1
export DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR="$RUN/backend_plans"
export DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR="$RUN/backend_plans"
export SPYRE_LX_RELAYOUT_PLAN_DIR="$RUN/lx_relayout_plans"

cd "$ROOT/spyre-granite-e2e-bench"
/home/adnan/dt-inductor/.venv/bin/python3 benchmarks/granite_block_layer_probe.py \
  --fms-root "$ROOT/foundation-model-stack" \
  --run-root "$RUN" \
  --case prefill \
  --seq-len 512 \
  --batch 1 \
  --hidden 4096 \
  --compile-block \
  --attn-name sdpa_causal \
  --iters 5 \
  --warmups 1 \
  --profile \
  --profile-dir "$RUN/block_prefill/profile" \
  --no-profile-memory || true
echo RUN=$RUN
'
```

Expected current enabled failure:

```text
8_batchmatmul
The initial chunk parameters must fit in LX for SuperDSC
```

### Run DEV Flash Smoke

Use only as compile/runtime smoke until correctness is restored:

```bash
oc exec -n a6-quantization adnan-spyre-dev-pf -- bash -lc '
set -euo pipefail
ROOT=/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404
RUN=/home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_verify_backend1_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"

export PATH="$ROOT/tools/dxp-split-wrapper:$ROOT/deeptools/build-deeptools/dxp:$PATH"
export PYTHONPATH="$ROOT/torch-spyre:$ROOT/torch-spyre/tests/inductor:$ROOT/foundation-model-stack:/home/adnan/dt-inductor/sentient/runtime/lib"
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache"
export PATCH_MODE=no_h2d,skip_cpu_ref
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
export SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
export SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
export SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
export LX_BOUNDARY_CLONES=1
export DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR="$RUN/backend_plans"
export DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR="$RUN/backend_plans"

timeout 900 /home/adnan/dt-inductor/.venv/bin/python3 /tmp/test-spyre-scripts/test_flash.py || true
echo RUN=$RUN
'
```

Do not call this value-correct unless `PATCH_MODE=no_h2d,skip_cpu_ref` is
removed and the CPU/reference checks pass.

## Current Blockers

1. `matmul_operand_broadcast` physical realization is incomplete.
   - Backend sees the right plan.
   - Ring traffic can be emitted.
   - The final PT `KERNEL` view is still wrong or unpopulated.

2. Free-standing artificial transfer nodes lack complete DDC semantics.
   - Failing names include `transfer_lds1_src:lxlu_dst:ptrow0`.
   - Failures include `std::out_of_range map::at`, missing loop distribution,
     and later loop-element-offset computation failures.
   - Skipping folds can run but produces wrong values.

3. The local same-core path is not proven.
   - Same-core traffic must not use self-ring.
   - Diagnostic ring/local two-stage runs still produced zeros.
   - A standalone local LX-to-LX copy litmus is needed.

4. Granite enabled run is capacity-blocked.
   - Current enabled relayout reaches `8_batchmatmul` and fails initial chunk
     fit.
   - This is consistent with an implementation that is not genuinely tile
     scoped.

5. Flash smoke skips correctness.
   - It is useful for SDSC shape and DXP/runtime mechanics only.

## Next Planned Experiments

Do these in order:

1. Preserve the current Deeptools dirty state.
   - Commit to Adnan's Deeptools fork or archive a patch before broad changes.
   - Do not revert exploratory dirty files.

2. Build a same-core local-copy litmus.
   - One producer core.
   - Patterned LX source.
   - Local LX-to-LX copy or restickify stage.
   - PT/matmul or a simpler consumer reads the destination.
   - No ring.
   - If this fails, stop using the standalone LXLU/LXSU local carrier for this
     class.

3. Try an existing-carrier local conversion probe.
   - Express local layout conversion through `ReStickifyOpLx`,
     `ReStickifyOpWithPTLx`, or `STCDPOpLx` mechanics that already know how to
     sequence LXLU/LXSU/fold state.
   - Keep cross-core ring all-gather separate and loop scoped.

4. Implement the intended two-stage loop-scoped path:

```text
for each matmul tile / consumer core:
  allocate small source-layout staging LX tile
  ring gather cross-core source chunks into staging
  copy/read same-core chunks locally
  local restickify/layout-convert staging -> final KERNEL operand tile
  matmul consumes final KERNEL operand tile
```

5. Revalidate synthetic M4.
   - Require `ALLCLOSE True`.
   - Require `MISMATCH 0 / 1024`.
   - Require `ROWMAP_OUT0` to match the reference row pattern.

6. Scale to M16/M64 and source split variants.

7. Replay flash with correctness enabled.
   - Remove `PATCH_MODE=no_h2d,skip_cpu_ref`.
   - Confirm `assert_close` and H2D/ref paths.

8. Rerun Granite S512.
   - Use split `DXP_LX_FRAC_AVAIL=0` and `DXP_BACKEND_LX_FRAC_AVAIL=1`.
   - Compare trace-derived `kernel_ms_per_iter`, not only wall time.
   - Classify remaining HBM rows as weight/prelayout versus activation.

## Acceptance Criteria

Do not claim the feature is complete until these are true:

- PR1 scatter remains isolated and production-clean.
- Synthetic M4/M16/M64 matmul operand broadcast is value-correct.
- Same-core chunks are local copies/direct reads, never self-ring.
- No full RHS materialization is required for Granite attention.
- Flash runs with correctness enabled and passes.
- Granite S512 relayout-enabled run completes and shows trace-derived kernel
  timing, with correctness and fallback status recorded.
- Any remaining `ReStickifyOpHBM` rows are classified as weight/prelayout or
  another explicit out-of-scope class.

## Handoff Guardrails

- Work on `ah/comms-collectives` for artifact docs only.
- Do not modify PR branches from this handoff thread.
- Do not revert unrelated user or agent changes.
- Do not treat dirty Deeptools exploratory edits as disposable.
- Do not publish speedups from wall time alone.
- Record Torch SHA, Deeptools SHA, dirty state, DXP binary path, wrapper path,
  env flags, run roots, backend plan JSON, stdout/stderr tails, and profiler
  trace summaries for every serious run.
- Weight restickifies are out of scope for this lane.
- Reductions and all-reductions are future arithmetic collective work, not a
  pure relayout extension.
