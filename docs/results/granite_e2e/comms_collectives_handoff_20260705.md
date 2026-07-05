# DLDSC LX-Relayout Communications Handoff - 2026-07-05

This is the first-principles handoff for the current Granite DLDSC/LX-relayout communications work. It is written for a new human or LLM agent with no prior context. It records the goal, the architecture, the approaches already tried, the current branch/pod state, the exact reproducible runs, and the next useful fixes.

## 1. North-Star Goal And Scope

The long-term goal is to remove in-scope non-weight HBM round trips in Granite by using DLDSC-described LX-to-LX relayout and on-chip communication.

In scope:

- classify communication edges that arise when a producer tensor distribution and a consumer compute distribution do not match;
- express the logical contract through DLDSC coordinate metadata;
- let Deeptools synthesize legal physical movement through LX/ring/local-copy mechanisms;
- cover the communication classes that appear in Granite:
  - scatter / disjoint 1:1 remap;
  - broadcast / multicast / 1:many fanout;
  - gather / all-gather / many:1 or many:many fanin/fanout;
  - layout-changing LX restickify;
  - reduce / all-reduce as later arithmetic collectives;
- verify with synthetic unit-style probes, flash attention probes, and Granite block prefill probes.

Out of scope for this lane:

- weight restickifies and weight preload/prelayout;
- working-set-reduction / streaming policy for large full-resident tensors, except where needed to avoid accidentally materializing a full RHS operand;
- production PR polishing for non-scatter collectives;
- claiming speedup from wall time alone.

The practical target for this artifact branch is broader than the production scatter PR. This branch is an experiment and progress log. The production PR branches should stay lean.

## 2. Branch, Repo, Pod, And Path Map

Use live pod state. Do not use stale local clones unless a path in this document explicitly says it is an archived artifact.

### Pods And Devices

Namespace:

```bash
a6-quantization
```

Pods:

| Pod | Device | Main role |
|---|---:|---|
| `adnan-cdx-spyre-dev-pf` | `/dev/vfio/80` | Active synthetic RHS operand broadcast and flash-attention work |
| `adnan-spyre-dev-pf` | `/dev/vfio/31` | Granite block / historical runbook reproduction |
| `adnan-clc-spyre-dev-pf` | `/dev/vfio/73` | Parallel Deeptools/Torch experiments |

Check pod health:

```bash
kubectl get pods -n a6-quantization | rg 'adnan|NAME'
```

### Current CDX Root

The active CDX workspace for this thread is:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
```

Important subpaths:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/repos/torch-spyre
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/repos/deeptools-comms-clean
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/build-deeptools-comms-clean-fast
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/tools/dxp-split-wrapper/dxp_standalone
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506
```

### Torch Repo

Repo:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/repos/torch-spyre
```

Branch:

```text
ah/comms-collectives
```

Remote:

```text
git@github.com-adnan-cdx-spyre-dev-pf:AdnanHoque/torch-spyre.git
```

At the time this handoff was written, the branch head before this doc commit was:

```text
42d8a2932a1b554d445f782421152357784d278a
```

This branch is the artifact/progress branch. It contains docs and run artifacts. Do not treat it as the production PR branch.

### Deeptools Repo

Repo:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/repos/deeptools-comms-clean
```

Branch:

```text
ah/comms-collectives
```

Remote:

```text
git@github.ibm.com:Adnan-Hoque1/deeptools.git
```

At the time this handoff was written, the branch head was:

```text
fa30750e129909fb8df8ce7a04891a32544014e1
```

Important: the Deeptools checkout is intentionally dirty with exploratory backend work. Do not revert it. Current dirty files include:

```text
dcc/src/Conversion/PCFGToDataflowIR/PCFGToDFManager.cpp
dcc/src/Conversion/PCFGToDataflowIR/PCFGToDataflowIR.cpp
dcc/src/Conversion/SentientToProgIR/Utils.cpp
dcg/dcg_fe/pcfg_gen/dlOps.cpp
dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp
dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp
dcg/dcg_manager/dcg_manager.cpp
ddc/ddc_fold.cpp
dsc/dsc2.cpp
dsc/dsc2.h
dsc/dsc2Pcfg.cpp
dxp/SdscRelayoutInsertion.cpp
```

## 3. Architecture Summary

The model we are converging on is:

1. Torch/Inductor chooses work divisions.
2. Torch emits DLDSC metadata that describes:
   - tensor distribution: which logical tensor coordinates are resident on which core/LX memory;
   - consumer compute distribution: which logical output/compute coordinates each core computes;
   - source and target layout/stick metadata when layout changes are possible.
3. The planner classifies and costs mismatch edges so work-division choices can eventually become reshard-aware.
4. Deeptools synthesizes the physical movement:
   - ring movement for cross-core LX transfers;
   - local LX copy for same-core materialization when aliasing is not valid;
   - local layout conversion/restickify when source layout and consumer operand layout differ;
   - legal scheduling around compute.

The guiding contract is not "Torch emits physical ring schedules." The intended contract is "Torch exposes enough coordinate/layout metadata for Deeptools to derive and schedule the movement." The artifact branch has also explored explicit physical carriers because those were useful to prove what was missing, but the direction is DLDSC coordinate metadata plus backend synthesis.

Current focus:

```text
activation LX shard
  -> grouped all-gather / broadcast across consumer cores
  -> local layout conversion into matmul KERNEL operand layout
  -> batchmatmul consumes KERNEL operand
```

This is the Granite/attention RHS handoff class. It is not the same as plain scatter.

## 4. Approaches Considered

### A. Explicit Coordinate-Remap Data-Op Path

Earlier work inserted an explicit coordinate-remap movement op between producer and consumer. Torch computed the exact source/destination cells, emitted a mixed SDSC containing movement rows, and Deeptools imported/lowered those rows.

What worked:

- proved that eliminating HBM activation handoffs can produce real speedups;
- gave explicit before/after SDSC artifacts;
- was easy to reason about for specific edges because the frontend had full control.

What did not work long-term:

- data-dsc support was deprecated in the SuperDSC bundle direction;
- it put too much physical movement scheduling into Torch;
- scaling to broadcast/gather/reduce would have made Torch own too much backend mechanics;
- the backend team preferred DLDSC coordinate incompatibility plus backend synthesis.

Why we pivoted:

The longer-term contract should use DLDSC tensor-vs-compute coordinate metadata and let Deeptools synthesize the movement.

### B. `STCDPOpLx` / Ranged Carrier

We also prototyped reusing or extending existing `STCDPOpLx`-style LX transfer support.

What worked:

- avoided inventing a wholly new public movement op;
- aligned better with existing Deeptools vocabulary for LX movement.

What did not work alone:

- a simple list of range transfers is still a physical carrier, not a full logical contract;
- layout-changing KERNEL operands need more than byte copies;
- one-to-many / many-to-one / many-to-many collectives need allocation and scheduling policy, not just transfer descriptors;
- coarse full-resident `STCDPOpLx -> ReStickifyOpLx` can be value-correct on small synthetic cases but fails Granite-scale capacity when it materializes the full RHS operand.

Why we pivoted:

`STCDPOpLx` remains useful as a backend building block, but the higher-level feature should be driven by DLDSC coordinate mismatch and class-aware backend lowering.

### C. Deeptools DLDSC Backend Relayout Insertion

The current production direction is Deeptools relayout insertion from DLDSC metadata. Torch emits tensor distribution and compute distribution; Deeptools detects incompatibility and inserts movement internally.

What worked:

- the existing Deeptools master path can recognize simple tensor-vs-compute incompatibility;
- cardinality probes showed the DLDSC representation can express scatter, one-to-many, many-to-one, and many-to-many cases when coordinate maps are dense over relevant dimensions;
- PR1-style scatter is aligned with this design.

What still needs work:

- the useful Granite/attention RHS case is not just ownership mismatch; it is grouped all-gather plus layout conversion into PT KERNEL operand form;
- backend allocation must avoid source/destination aliasing;
- the full-resident staged solution is too large for Granite attention;
- the loop-scoped KERNEL operand path currently emits ring traffic but is not value-correct.

### D. Current Kernel-Neighbor Carousel Approach

The active CDX thread tries to make the Granite/attention RHS class loop-scoped instead of full-resident.

The intended decomposition is:

```text
for each matmul tile / consumer core:
  1. ring-gather source-layout chunks into a non-overlapping loop-local LX staging tile
  2. locally restickify/layout-convert that staging tile into the final KERNEL operand layout
  3. run the matmul tile using the KERNEL operand bytes
```

What worked:

- an archived small synthetic M=4 run was value-correct with a kernel-neighbor carousel:
  - `kernel_neighbor_carousel_M4_212148`
  - `ALLCLOSE True`
  - `MISMATCH 0 / 1024`
- the current backend can emit L3 ring send/recv PCFG nodes for the synthetic path;
- diagnostics have isolated current failures to specific backend lowering/fold/fill stages.

What did not work:

- direct fused ring writes into final KERNEL layout are value-wrong;
- using ring for same-core pieces is unsafe and caused a PCI bus fence in a diagnostic;
- replacing `_lx_neighbor` name-based paths with `_lx_local` avoids one value-corrupting path but exposes DDC/dsc2 metadata/fold gaps.

Why this remains the next path:

It is the only approach that can be both capacity-safe for Granite-scale attention and eventually value-correct: it preserves the staged decomposition but makes it loop/tile scoped.

## 5. Current Status

### PR1 Scatter Path

Scatter / disjoint 1:1 same-layout relayout is the production-shaped first class. It is the path where producer and consumer differ in core ownership, but the logical tensor layout/stick form is compatible.

Status:

- Torch emits DLDSC tensor-distribution-vs-compute metadata.
- Deeptools derives and inserts LX relayout for the same-layout scatter class.
- This is the first production-worthy communication class.
- It does not add new GraphLowering nodes in Torch; it changes DLDSC metadata emitted for existing operations.

### Useful Workloads / Results Already Observed

Granite S=512 checkpoint from earlier artifact:

| Variant | Kernel ms/iter | Wall median ms | Notes |
|---|---:|---:|---|
| Relayout disabled | 14.7258 | 27.6074 | control |
| DLDSC relayout enabled | 13.8213 | 26.5205 | about 1.065x kernel speedup |

This run removed explicit non-weight `ReStickifyOpHBM` rows in the stable relayout artifact. Remaining explicit HBM restickifies were weight-shaped and out of scope.

Flash attention compile/runtime artifact:

| Variant | Runtime result | SDSCs | HBM restickify | LX restickify | Backend plans |
|---|---:|---:|---:|---:|---:|
| relayout off | success | 550 | 32 | 0 | 0 |
| relayout on | success | 550 | 0 | 32 | 32 |

This proves the flash-attention HBM restickify class can be transformed structurally to on-chip `ReStickifyOpLx` plus backend plans. It does not prove final value correctness for the full unpatched flash workload.

### Current Synthetic RHS Operand Broadcast Status

The active blocker is the synthetic M=4 RHS matmul operand broadcast probe on CDX:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506
```

Key dirs:

| Directory | Result | Meaning |
|---|---|---|
| `baseline_no_relayout_M4_044702` | `ALLCLOSE True`, `MISMATCH 0 / 1024` | HBM/no-relayout control is correct |
| `kernel_neighbor_carousel_M4_212148` | `ALLCLOSE True`, `MISMATCH 0 / 1024` | archived passing carousel run |
| `kernel_neighbor_carousel_repro_allow_M4_053418` | `ALLCLOSE False`, `MISMATCH 949 / 1024`, `ROWMAP_OUT0 [0.0, 0.0, 0.0, 0.75]` | current reproduction emits traffic but is value-wrong |
| `kernel_neighbor_carousel_noallow_M4_060533` | compile failure | guard rejects mixed HBM input-neighbor coexistence without diagnostic env |
| `kernel_neighbor_local_suffix_M4_061049` | DXP failure | `_lx_local` suffix exposes dsc2 redistribution assert |
| `kernel_neighbor_lxlocal_foldskip_M4_062008` | DXP failure | `distributeElemArrToTemporalLoops` not enough elements, owner `allocate_lds1_lx` |
| `kernel_neighbor_lxlocal_foldskip3_M4_062949` | DXP failure | passed first guard, later `std::out_of_range map::at` |
| `kernel_neighbor_lxlocal_ddcskip_M4_063420` | DXP failure | current latest: `fillDataInfo` map lookup on transfer with empty coordinate maps |

Latest error signature:

```text
[lx_neighbor_fold_guard] skip redistribution owner=allocate_lds1_lx targetLds=1 dim=out
[lx_neighbor_ddc_fold_guard] skip transfer fold node=transfer_lds1_src:lxlu_dst:ptrow0
...
[fillDataInfo diagnostic] exception=map::at
  processor=transfer_lds1_src:lxlu_dst:ptrow0
  allocation=allocate_lds1_lx
  loop=loop_ds2_ds3_in
  myLdsIdx=1
  constantId=-1
  dataConnect=l3_lx_kernel
  locUnit=42
  locStorage=1
  isProducer=0
  coordDims=0
  coordCoreMap=0
terminate called after throwing an instance of 'std::out_of_range'
  what():  map::at
```

Interpretation:

The current patch got past the first dsc2 redistribution failure by skipping folded LX-neighbor redistribution for the artificial allocation. The next failure is not the same problem. DDC/fillDataInfo is trying to process a generated `transfer_lds1_src:lxlu_dst:ptrow*` node as if it had enough coordinate/data metadata, but the associated coordinate maps are empty. The next likely fix is to make the artificial local/fold transfer carry the coordinate metadata DDC expects, or to route it through a cleaner schedule-local staging representation instead of pretending it is an ordinary tensor transfer.

## 6. Current CDX Technical Thread In Detail

### Synthetic Probe

Script:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506/stable_matmul_operand_broadcast_64diag.py
```

Runner:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506/run_one_64diag.sh
```

The probe uses row-patterned data so wrong row/column placement is visible in `ROWMAP_OUT0`.

### Archived Passing Carousel

Directory:

```text
kernel_neighbor_carousel_M4_212148
```

Result:

```text
ALLCLOSE True
MISMATCH 0 / 1024
```

Important log facts from the archived passing path:

- transfer attached as `transfer_lds1_src:no_component_dst:lx_lx_local`;
- `lx_neighbor_pcfg_dsc2` emitted ring transfer nodes;
- phase/group carousel emitted recv/send groups;
- the run did not use `DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC`.

### Current Wrong Reproduction

Directory:

```text
kernel_neighbor_carousel_repro_allow_M4_053418
```

Result:

```text
ALLCLOSE False
MAX_DIFF 0.7490234375
MISMATCH 949 / 1024
ROWMAP_OUT0 [0.0, 0.0, 0.0, 0.75]
```

It uses a carrier named:

```text
transfer_lds1_src:no_component_dst:no_component_lx_neighbor
```

It emits ring send/recv PCFG nodes but produces wrong final values. This suggests the ring route alone is not enough and/or the current DDC interpretation of `_lx_neighbor` changes address/layout behavior.

### `_lx_local` Suffix Experiments

Changing the artificial transfer suffix to `_lx_local` avoids the `_lx_neighbor` name-specialized DDC path, but revealed backend metadata/fold assumptions:

1. `dsc2` redistribution assert:

```text
[distributeElemArrToTemporalLoops] Not enough elements to distribute.
Owner=allocate_lds1_lx targetLds=1 dim=out
```

2. after adding a fold guard for the artificial LX allocation:

```text
std::out_of_range: map::at
```

3. after adding a DDC fold skip for generated `transfer_lds1_src:lxlu_dst:ptrow*` nodes:

```text
[fillDataInfo diagnostic] exception=map::at
processor=transfer_lds1_src:lxlu_dst:ptrow0
allocation=allocate_lds1_lx
coordDims=0 coordCoreMap=0
```

This is the current CDX handoff point.

### Current Patched Deeptools Files

Most relevant files:

```text
dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp
dsc/dsc2.cpp
dsc/dsc2.h
dsc/dsc2Pcfg.cpp
ddc/ddc_fold.cpp
```

Other dirty files exist and should not be reverted without inspection:

```text
dcc/src/Conversion/PCFGToDataflowIR/PCFGToDFManager.cpp
dcc/src/Conversion/PCFGToDataflowIR/PCFGToDataflowIR.cpp
dcc/src/Conversion/SentientToProgIR/Utils.cpp
dcg/dcg_fe/pcfg_gen/dlOps.cpp
dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp
dcg/dcg_manager/dcg_manager.cpp
dxp/SdscRelayoutInsertion.cpp
```

Relevant functions:

```text
L3DlOpsScheduler.cpp:
  populateMatmulOperandBroadcastStickRingTransfers()
  populateMatmulOperandBroadcastRingTransfersAfterAllocation()
  createAllocationAndTransfer()

dsc2Pcfg.cpp:
  buildLxNeighborRingTransfers()
  translateDscDataTransfer()

ddc_fold.cpp:
  isFoldedLxNeighborAllocation()
  buildSpatialFold()

dsc2.cpp:
  distributeElemArrToTemporalLoops()
  fillDataInfo()
```

## 7. Repro And Runbooks

### List Pods

```bash
kubectl get pods -n a6-quantization | rg 'adnan|NAME'
```

### Check Branch State

```bash
kubectl exec -n a6-quantization adnan-cdx-spyre-dev-pf -- bash -lc '
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

### Build DXP On CDX

```bash
kubectl exec -n a6-quantization adnan-cdx-spyre-dev-pf -- bash -lc '
set -euo pipefail
/home/adnan-cdx/dt-inductor-mixed/.venv/bin/cmake \
  --build /home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/build-deeptools-comms-clean-fast \
  --target dxp_standalone \
  -j8
'
```

### Run The Synthetic M=4 Probe

```bash
kubectl exec -n a6-quantization adnan-cdx-spyre-dev-pf -- bash -lc '
set -euo pipefail
RUN=/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506

export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
export DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1

unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_DIRECT_ALLGATHER_DIAGNOSTIC || true
unset DEEPTOOLS_LX_NEIGHBOR_MAX_RUN || true
unset DEEPTOOLS_LX_NEIGHBOR_VIEW_PROBE || true
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_AVOID_SOURCE_ALIAS_DIAGNOSTIC || true
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_EMIT_LOCAL_RESTICKIFY_STAGE || true
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_REPLICATE_DEST_COORDS || true
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_CONSUMER_LX_VIEW_STRIDE || true
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_FORCE_ROW_MAJOR || true
unset DEEPTOOLS_LX_NEIGHBOR_ATTACH_FINAL_TRANSFER_DIAGNOSTIC || true
unset DEEPTOOLS_LX_NEIGHBOR_ATTACH_ALL_FINAL_TRANSFERS_DIAGNOSTIC || true
unset DEEPTOOLS_LX_NEIGHBOR_ALLOW_SELF_RING_DIAGNOSTIC || true
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_FORCE_OUT_MAJOR || true

CASE=handoff_probe_M4_$(date +%H%M%S)
bash "$RUN/run_one_64diag.sh" "$CASE" mul_one 1 4 64 256 || true
D="$RUN/$CASE"

grep -E "ALLCLOSE|MAX_DIFF|MISMATCH|ROWMAP_OUT0|lx_neighbor_fold_guard|lx_neighbor_ddc_fold_guard|fillDataInfo diagnostic|matmul_operand_stick_transfer|lx_neighbor_pcfg_dsc2|attaching to|DtException|Remaining elemArr|Owner=|map::at" "$D/run.log" | sed -n "1,260p"
printf "RUN_DIR=%s\n" "$D"
'
```

### Replay A Preserved Bundle Through DXP

Use this when a run created a failing `cache/inductor-spyre/<bundle>` directory:

```bash
kubectl exec -n a6-quantization adnan-cdx-spyre-dev-pf -- bash -lc '
set -euo pipefail
ROOT=/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
export PATH="$ROOT/tools/dxp-split-wrapper:$PATH"
export DXP_BACKEND_LX_FRAC_AVAIL=1
BUNDLE=/path/to/cache/inductor-spyre/sdsc_fused_mm_mul_0_xxxxx
dxp_standalone --bundle -d "$BUNDLE"
'
```

For gdb:

```bash
kubectl exec -it -n a6-quantization adnan-cdx-spyre-dev-pf -- bash -lc '
set -euo pipefail
ROOT=/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
export PATH="$ROOT/tools/dxp-split-wrapper:$PATH"
export DXP_BACKEND_LX_FRAC_AVAIL=1
BUNDLE=/path/to/cache/inductor-spyre/sdsc_fused_mm_mul_0_xxxxx
gdb --args dxp_standalone --bundle -d "$BUNDLE"
'
```

### Split DXP LX Env Wrapper

The wrapper at:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/tools/dxp-split-wrapper/dxp_standalone
```

maps:

```bash
DXP_BACKEND_LX_FRAC_AVAIL -> DXP_LX_FRAC_AVAIL
```

This matters because the same knob has opposite practical meaning in Torch and backend use:

- Torch with `DXP_LX_FRAC_AVAIL=0` means frontend/Torch can plan with full LX.
- DXP with `DXP_LX_FRAC_AVAIL=0` means no backend chunk space, causing failures such as `initial chunk parameters must fit in LX`.

For full-Torch-LX but backend DXP space, use:

```bash
export DXP_LX_FRAC_AVAIL=0
export DXP_BACKEND_LX_FRAC_AVAIL=1
export PATH=/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/tools/dxp-split-wrapper:$PATH
```

For quick sweeps, use only `0.2` and `1.0` unless a specific allocator bug requires finer granularity.

### Flash Attention And Granite

Use the runbook repo for environment gotchas:

```text
https://github.ibm.com/Adnan-Hoque1/spyre-granite-e2e-bench
```

Relevant archived artifact dirs on the Torch artifact branch:

```text
docs/results/granite_e2e/comms_collectives_20260704_flash_attention_runtime
docs/results/granite_e2e/comms_collectives_20260704_flash_attention_value_status
docs/results/granite_e2e/comms_collectives_20260704_fresh_granite_s512
docs/results/granite_e2e/comms_collectives_20260704_granite_spill_classification
```

When unblocked, rerun:

1. synthetic M=4/M16/M64 RHS operand broadcast;
2. flash attention value-correct run;
3. Granite block S=512 prefill with trace-derived `kernel_ms_per_iter`.

Do not publish speedups without:

- correctness status;
- fallback status;
- Torch SHA;
- Deeptools SHA;
- env flags;
- pinned DXP wrapper/runtime path;
- archived Kineto trace-derived kernel time.

## 8. Communication Taxonomy And Remaining Work

| Class | Meaning | Current status | Frontend needs | Backend needs |
|---|---|---|---|---|
| Scatter | each source shard moves to one unique destination shard, same logical layout | production-shaped first class | dense DLDSC coordinate maps; classify as same-layout scatter | derive LX moves, allocate safe destination, schedule before consumer |
| Broadcast / multicast | one producer shard feeds multiple consumer cores | partial for same-layout synthetic cases | classify fanout and expose source/consumer maps | avoid duplicate/full materialization when possible; schedule fanout efficiently |
| Gather | multiple producer shards feed one consumer core | partial synthetic evidence | classify fanin and dense split-1 dims | safe destination allocation and non-overlap; local copies for same-core |
| All-gather | every consumer gets many/all producer shards | partial synthetic and attention RHS planning evidence | classify many-to-many, group size, layout metadata | ring gather, local same-core handling, capacity-aware staging |
| Matmul operand broadcast | all-gather activation shards then convert to PT KERNEL operand layout | active blocker | emit source tensor layout, target KERNEL layout, consumer compute split, communication class | loop-scoped gather into staging plus local restickify into KERNEL |
| Layout/restickify LX | same values but stick/layout form changes | `ReStickifyOpLx` exists; coarse path not enough | expose source and target layout/stick metadata | schedule-local restickify; avoid full RHS materialization |
| Reduce | many source values are arithmetically combined | not covered by relayout | specify reduction axes/op/dtype/identity | arithmetic collective lowering and accumulation |
| All-reduce | reduce then redistribute | not covered | same as reduce plus output distribution | reduce plus broadcast/scatter of result |

The current scatter path is not enough to remove all Granite HBM spills. The next major class is `all_gather_replicate + layout_conversion` for matmul KERNEL operands.

## 9. Guardrails

- Do not revert dirty Deeptools edits in the active CDX branch. They are exploratory but contain the current failure isolation.
- Do not use stale clones. Start from the path map above or explicitly document why a different clone is used.
- Do not claim speedup from wall time alone.
- Archive raw run dirs, env, stdout/stderr tails, SDSCs, backend plan JSON, and trace summaries.
- Keep docs and large artifacts on `ah/comms-collectives`; do not put them on production PR branches.
- Do not open PRs from this thread unless Adnan explicitly asks.
- Avoid public PR numbers in internal docs; use descriptive names such as "PR1 scatter" or "production scatter branch".
- Weights/offline preloading are out of scope for this communication lane.
- Same-core transfers must not be represented as self-ring traffic.
- For DLDSC coordinate maps, include split-1 dimensions explicitly as slice `0` when they participate in the relayout dimension set.

## 10. Concrete Next Steps

1. On CDX, preserve the current dirty Deeptools branch state by committing to Adnan's Deeptools fork or archiving a patch before further surgery.
2. Fix the current `fillDataInfo` failure for `_lx_local` artificial transfers:
   - inspect why `coordDims=0` and `coordCoreMap=0` for `transfer_lds1_src:lxlu_dst:ptrow0`;
   - either populate the missing coordinate metadata or route the node through a cleaner schedule-local staging representation;
   - avoid adding broader name-based skips unless there is a precise invariant.
3. Rebuild DXP and rerun the synthetic M=4 probe.
4. If M=4 compiles, check value correctness:
   - `ALLCLOSE True`;
   - `MISMATCH 0 / 1024`;
   - `ROWMAP_OUT0` matches the reference row pattern.
5. Scale synthetic checks to M16/M64 and source split variants.
6. If direct loop-scoped KERNEL writes remain value-wrong, implement the two-stage loop-scoped plan:
   - ring gather into source-layout staging;
   - local copy for same-core chunks;
   - local restickify/layout conversion into the final KERNEL operand allocation;
   - matmul consumes final KERNEL tile.
7. Replay flash attention with relayout on and verify value correctness before performance.
8. Rerun Granite S=512 prefill and compare against the disabled baseline using trace-derived kernel time.
9. Classify remaining Granite HBM spills after this class is fixed:
   - explicit weight restickifies: out of scope;
   - fused FFN/SwiGLU activation boundaries: likely residency/fused-region/WSR work;
   - reductions: separate arithmetic collective lane.

## 11. Existing Artifact Docs To Read Next

Start with these files in the same repo/branch:

```text
docs/results/granite_e2e/comms_collectives_20260705/collective_class_status_20260705.md
docs/results/granite_e2e/comms_collectives_20260705/matmul_operand_broadcast_status_20260705.md
docs/results/granite_e2e/comms_collectives_20260705/matmul_operand_broadcast_loop_scoped_checkpoint_20260705.md
docs/results/granite_e2e/comms_collectives_20260705/two_stage_matmul_operand_broadcast_plan_20260705.md
docs/results/granite_e2e/comms_collectives_20260704_current_state.md
```

Those docs contain narrower snapshots. This file is the high-level handoff that ties them together.
