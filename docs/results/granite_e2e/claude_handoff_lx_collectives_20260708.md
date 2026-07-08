# Claude Handoff: DLDSC LX Relayout And Granite Collectives

Date: 2026-07-08

This is the "read this first" packet for the Granite LX relayout / on-chip
communication work. It collects the goal, architectural direction, relevant
PRs, branches, artifacts, runbooks, and open gaps so another agent can reason
from first principles without replaying the whole chat history.

## One-Sentence Summary

We are trying to remove non-weight HBM spills in Granite by representing
producer/consumer tensor-distribution mismatches in DLDSC metadata, letting the
backend synthesize bounded LX-to-LX movement, and expanding the supported
communication classes from scatter to all-gather/restickify, matmul operand
broadcast, partial-view gather, and eventually reduction-aware collectives.

## North Star

The desired architecture is:

1. Torch/Inductor owns work-division selection.
2. Torch emits the logical DLDSC coordinate contract:
   producer tensor distribution, consumer compute distribution, and classified
   communication edge.
3. Deeptools owns physical realization:
   LX address allocation, legal descriptor generation, ring/local movement, and
   schedule placement.
4. The planner/cost model eventually becomes reshard-aware:
   work division should include the cost of the communication edge it creates.
5. Scheduling/overlap is a follow-up layer:
   first prove bounded value-correct movement; later pipeline movement with
   compute and WSR tiling.

Important scoping rule: this lane should not duplicate WSR. If a full Granite
activation cannot be safely materialized in LX as one resident tensor, this work
should fail closed or fall back, while WSR/tiling makes the communication
tile-scoped later.

## Main Public Tracking

Epic:

```text
https://github.com/torch-spyre/torch-spyre/issues/3049
```

PR1 Torch, scatter metadata:

```text
https://github.com/torch-spyre/torch-spyre/pull/2939
```

PR1 Deeptools, scatter realization:

```text
https://github.ibm.com/ai-chip-toolchain/deeptools/pull/4408
```

Older coordinate-remap Torch PR, useful historical contrast:

```text
https://github.com/torch-spyre/torch-spyre/pull/2789
```

Old Deeptools backend-relayout PR discussed with Charlie/Swagath:

```text
https://github.ibm.com/ai-chip-toolchain/deeptools/pull/4255
```

Deeptools branch Olivier pointed at during the dldsc discussion:

```text
https://github.ibm.com/ai-chip-toolchain/deeptools/tree/an/dxp_relayout_v2
```

SwiGLU SiLU opfunc issue, relevant because it reduces the value of our doing
separate SiLU fusion work:

```text
https://github.com/torch-spyre/torch-spyre/issues/2763
```

## Important Repositories

Torch upstream:

```text
https://github.com/torch-spyre/torch-spyre
```

Adnan Torch fork:

```text
https://github.com/AdnanHoque/torch-spyre
```

Deeptools upstream:

```text
https://github.ibm.com/ai-chip-toolchain/deeptools
```

Adnan Deeptools fork:

```text
https://github.ibm.com/Adnan-Hoque1/deeptools
```

Granite benchmark/runbook repo:

```text
https://github.ibm.com/Adnan-Hoque1/spyre-granite-e2e-bench
```

Flash attention scripts:

```text
https://github.ibm.com/aviros/test-spyre-scripts/blob/main/test_flash.py
https://github.ibm.com/aviros/test-spyre-scripts/blob/05deb9702654f73781b457ed052a3ff69316670f/test_flash_4_head.py
```

Jamie perf-suite branch for fused/unfused SwiGLU:

```text
https://github.ibm.com/ai-sw-acceleration/spyre-perf-suite/tree/jamie/dev
```

## Branch Map

Artifact/lab-notebook branch:

```text
Torch fork: AdnanHoque/torch-spyre:ah/comms-collectives
Local worktree: /Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives
Local SHA inspected for this doc: 3444ad324de6946deca91b32a9bec94fa70d2de5
```

PR1 Torch branch:

```text
AdnanHoque/torch-spyre:pr-lx-relayout-scatter
Local worktree: /Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/pr2939-review-fixes
Local SHA inspected: 5f29f49b3b4868eb8a51737a5c56fd47c448c97a
```

PR2 Torch extraction branch:

```text
AdnanHoque/torch-spyre:pr-lx-relayout-allgather-restickify
Local worktree: /Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/extract_prs_20260707/torch-pr2
Local SHA inspected: c539ba8555bb3cea5d00593217baa954e69c137d
```

PR2 Deeptools extraction branch:

```text
Adnan-Hoque1/deeptools:adnan/lx-relayout-allgather-restickify
Local worktree: /Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/extract_prs_20260707/deeptools
Local SHA inspected: e93de8705039d17b16d766735a63e27d64564cd3
```

Planned stacked branches:

```text
Torch PR3: pr-lx-relayout-broadcast-multicast
Deeptools PR3: adnan/lx-relayout-broadcast-multicast

Torch PR4: pr-lx-relayout-partial-view-gather
Deeptools PR4: adnan/lx-relayout-partial-view-gather
```

Broad prototype source branches:

```text
Torch: AdnanHoque/torch-spyre:ah/comms-collectives
Torch: AdnanHoque/torch-spyre:gather-restickify
Torch: AdnanHoque/torch-spyre:ah/comms-collectives-dldsc-agent
Deeptools: Adnan-Hoque1/deeptools:ah/comms-collectives
```

Do not PR the broad prototype branches as-is. They contain useful experiments,
debug hooks, and historical artifacts, but they are not clean review slices.

## First Principles Model

Granite contains producer/consumer edges where the producer and consumer prefer
different per-core views of the same logical tensor.

Examples:

- Matmul may prefer a 2-D split, such as `{mb:4, out:8}`.
- Pointwise or follow-on ops may prefer a pure-M split, such as `{mb:32}`.
- Attention value-side matmul operands may need many producer shards replicated
  into each consumer's KERNEL/RHS reader.

Without an on-chip relayout, the compiler often resolves this mismatch by
writing the intermediate to HBM and reading it back with a new layout. That is
correct but expensive.

The intended fix is not "always keep everything in LX." The intended fix is:

1. Identify edges where the producer tensor is LX-resident and the consumer
   needs the same logical values under a different core distribution.
2. Classify the communication:
   scatter, all-gather, broadcast, partial-view gather, reduction, etc.
3. Emit enough DLDSC metadata for Deeptools to understand the logical handoff.
4. Let Deeptools materialize the needed bounded movement legally in LX.
5. Fall back when the movement is too large, ambiguous, reduction-like, or not
   yet supported.

## Communication Taxonomy

Current working taxonomy:

| Class | Meaning | Current Status |
|---|---|---|
| scatter / permutation | `N -> N`; one producer shard moves to one consumer shard; no duplication, no arithmetic | PR1 |
| broadcast | `1 -> many`; one source piece copied to every consumer | planned PR3 / backend cardinality probes only |
| multicast | `1 -> subset`; one source copied to a subset | planned PR3 / backend cardinality probes only |
| gather | `many -> 1`; distinct pieces assembled on one core, no arithmetic | planned PR4 if partial-view/source-offset semantics are needed |
| all-gather / replicate | `many -> many`; each consumer receives multiple or all source pieces | PR2 and kernel-neighbor prototypes |
| layout-all-gather restickify | all-gather plus local layout/stick transformation | PR2/prototype flash path |
| matmul operand broadcast | all-gather/replicate specifically for a matmul KERNEL/RHS operand | prototype evidence; PR2-adjacent, not PR3 despite the word "broadcast" |
| reduce / all-reduce | many pieces combined arithmetically | not implemented; needs reduction-aware primitive |
| WSR / streaming | tile-scope the region so LX pressure is bounded | separate WSR work, not this PR set |

## PR1: Scatter Foundation

PR1 adds the first production-shaped DLDSC LX relayout path.

Torch side:

- Branch: `pr-lx-relayout-scatter`
- PR: `https://github.com/torch-spyre/torch-spyre/pull/2939`
- Adds LX relayout metadata for scatter/permutation mismatches.
- Extends the LX planner rather than adding graph nodes.
- Keeps feature gated by `SPYRE_LX_PLANNER_RELAYOUT=1`.
- Reserves destination LX space because Deeptools synthesizes the post-relayout
  allocation before the consumer op.

Deeptools side:

- PR: `https://github.ibm.com/ai-chip-toolchain/deeptools/pull/4408`
- Reads DLDSC tensor distribution vs consumer compute incompatibility.
- Synthesizes LX relayout internally.
- Uses existing STCDPOpLx-style movement machinery under the backend path.

PR1 artifacts say this class removed five HBM input/output round-trip candidates
in the Granite experiment:

```text
HBM input roundtrip candidate: 5 -> 0
HBM output spill: 5 -> 0
scatter: 0 -> 5
```

Source artifact:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/tmp_dldsc_artifacts/dldsc_relayout_1p2_gap_analysis_20260629.md
```

Specific Granite edge inventory later recorded:

```text
buf13 -> buf46 inside old 9_ReStickifyOpHBM: scatter, covered by PR1
buf9 -> buf14 into old 10_batchmatmul: scatter, covered by PR1
```

Source artifact:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/dldsc_collectives_backend_gap_20260702/granite_edge_inventory_clc_20260703.tsv
```

PR1 does not cover all-gather, broadcast, partial-view gather, or reduction.

## PR2: Bounded All-Gather / Restickify Extraction

PR2 is intended to be the next small vertical slice after PR1. It should not be
a mega-PR.

Torch PR2:

```text
Branch: AdnanHoque/torch-spyre:pr-lx-relayout-allgather-restickify
Local SHA inspected: c539ba8555bb3cea5d00593217baa954e69c137d
Diff shape vs upstream/main: 14 files, 1611 insertions, 24 deletions
```

Main touched files:

```text
tests/inductor/test_lx_relayout_dldsc.py
torch_spyre/_inductor/codegen/bundle.py
torch_spyre/_inductor/codegen/superdsc.py
torch_spyre/_inductor/lx_relayout.py
torch_spyre/_inductor/lx_relayout_contracts.py
torch_spyre/_inductor/scratchpad/allocator.py
torch_spyre/_inductor/spyre_kernel.py
```

Deeptools PR2:

```text
Branch: Adnan-Hoque1/deeptools:adnan/lx-relayout-allgather-restickify
Local SHA inspected: e93de8705039d17b16d766735a63e27d64564cd3
Diff shape vs origin/master: 24 files, 1932 insertions, 113 deletions
```

Main touched files:

```text
dxp/SdscRelayoutInsertion.cpp
dcg/dcg_fe/pcfg_gen/dlOpsNew.cpp
dsc/superdsc.cpp
dsc/dsc2Pcfg.cpp
ddc/ddc_fold.cpp
util/LayoutAllgatherRestickify.cpp
util/LayoutAllgatherRestickify.h
util/test/LayoutAllgatherRestickify_unit_test.cpp
```

PR2 should include:

- `all_gather_replicate`
- `matmul_operand_broadcast` metadata when it is the bounded all-gather class
- bounded `gather_then_restickify`
- `ReStickifyOpLx` support
- capacity/chunk-cap fail-closed behavior

PR2 should exclude:

- generic broadcast/multicast
- partial-view gather/source-offset gather
- full Granite streaming
- WSR
- debug hook sprawl

Important naming note: `matmul_operand_broadcast` belongs with PR2, not PR3.
Despite "broadcast" in the name, the Granite/flash case is many producer shards
replicated/gathered into matmul RHS/KERNEL consumers. It is not a simple
`1 -> many` broadcast.

## Current Granite And Flash Edge Inventory

Confirmed edge inventory:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/dldsc_collectives_backend_gap_20260702/granite_edge_inventory_clc_20260703.tsv
```

Contents:

```text
buf13 -> buf46 inside old 9_ReStickifyOpHBM
  class: scatter
  status: covered by PR1 scatter

buf9 -> buf14 into old 10_batchmatmul
  class: scatter
  status: covered by PR1 scatter

mul/buf46 -> batchmatmul/buf14
  class: all-gather + layout-restickify/form-changing
  status: needs staged ReStickifyOpLx/form-changing path

clone/buf21/Tensor1 -> batchmatmul/buf22
  class: all-gather/broadcast, all_gather_replicate
  status: needs matmul transfer loop / kernel-neighbor style collective

weight/prelayout restickifies for buf45, buf47, buf48, buf49
  class: weight/prelayout
  status: excluded from this work; should be handled by offline weight preload/prelayout
```

## Flash Attention Evidence

Flash runtime checkpoint:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/dldsc_flash_runtime_checkpoint_20260702.md
```

Key result:

```text
baseline: 32 ReStickifyOpHBM rows
optimized: 0 ReStickifyOpHBM rows, 32 ReStickifyOpLx rows, 32 backend plans
representative handoff: mul -> ReStickifyOpHBM -> batchmatmul becomes mul -> ReStickifyOpLx -> batchmatmul
communication_class: all_gather
communication_pattern: layout_allgather_restickify
producer_op: mul
consumer_op: batchmatmul
```

Flash contract summary:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/flash_contract_20260702/flash_contract_summary.json
```

Representative named classification:

```text
source_name: buf27
producer_name: buf27
consumer_name: buf7
kind: layout_allgather_restickify
```

Important caveat: baseline flash has an independent correctness issue around
zero-stride/broadcasted unsqueeze views. Jamie's current read is that
TensorArg/SDSC generation drops stride_map/zero-stride information and
recomputes dense strides, so flash numeric correctness is not the right gate
for this communication work until baseline correctness is fixed. Our gate is
whether the on-chip communication introduces additional correctness issues.

## Matmul Operand Broadcast / Kernel-Neighbor Candidate

This is the high-value attention value-side handoff. It is not PR1 scatter.

Artifact:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/dldsc_collectives_backend_gap_20260702/flash_matmul_operand_broadcast_20260703/README.md
```

The isolated flash edge:

```text
sdsc_105.json / 105_batchmatmul consumes Tensor1 as a KERNEL operand
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
realization_strategy = loop_scoped_input_fetch
logical_transfer_count = 1024
source_core_count = 32
consumer_core_count = 32
```

Several early integrated STCDPOpLx versions compiled but were value-wrong. The
standalone 4-core STCDP broadcast with destination chunk offsets proved that the
ring transfer itself can express the needed 4-source to 4-destination RHS
broadcast when each source chunk lands in a distinct destination slot.

Later kernel-neighbor candidate artifacts showed a Granite S512 speedup:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/extract_prs_20260707/torch-artifacts/docs/results/granite_e2e/comms_collectives_20260706/kernel_neighbor_candidate_20260706/runs/granite_s512_profiler_py212_compare/README.md
```

Key result:

```text
shape: B=1, S=512, E=4096, causal prefill
disabled kernel ms/iter: 12.565
enabled kernel ms/iter: 11.875
speedup: 1.058x
attention handoff kernel total: 5.862 ms -> 2.822 ms
attention handoff speedup: 2.077x
```

SDSC comparison:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/extract_prs_20260707/torch-artifacts/docs/results/granite_e2e/comms_collectives_20260706/kernel_neighbor_candidate_20260706/runs/kernel_neighbor_s512_wall_compare/wall_sdsc_comparison.md
```

Key SDSC result:

```text
disabled: 5 ReStickifyOpHBM rows, 0 ReStickifyOpLx rows, 0 backend plans
enabled: 4 ReStickifyOpHBM rows, 1 ReStickifyOpLx row, 2 backend plans
solved edge: attention value-side matmul operand handoff inside first attention kernel
remaining ReStickifyOpHBM rows: weight/prelayout rows, out of scope
```

Do not overclaim this as final PR2 unless the extraction branch contains the
same behavior and passes the same device replay. Treat it as the strongest
prototype evidence for the next PR slice.

## Granite S512 Current Evidence

Current state note:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/extract_prs_20260707/torch-artifacts/docs/results/granite_e2e/comms_collectives_20260704_current_state.md
```

Key result from that checkpoint:

```text
relayout disabled control: 14.7258 kernel ms/iter, 27.6074 wall ms
DLDSC relayout enabled: 13.8213 kernel ms/iter, 26.5205 wall ms
kernel speedup: 1.065x
wall speedup: 1.039x
```

The note says explicit non-weight `ReStickifyOpHBM` rows are gone in the stable
relayout run. However, FFN/SwiGLU still has HBM-backed activation tensors at
fused-region boundaries:

```text
front projection batchmatmul output allocate-Tensor2_hbm
silu input allocate-Tensor0_hbm, output allocate-Tensor1_lx
mul input1 allocate-Tensor1_hbm, output allocate-Tensor2_hbm
down projection consumes HBM activation input in the next fused region
```

This is not the same as the explicit flash `ReStickifyOpHBM` row. It is a
fused-region residency / pool-boundary / WSR-shaped problem.

Granite S512 archive:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/extract_prs_20260707/torch-artifacts/docs/results/granite_e2e/granite_s512_relayout_clc_20260705.md
```

That archive records a disabled baseline with five `ReStickifyOpHBM` rows:

```text
one activation attention RHS layout restickify
four weight/prelayout restickifies
```

Enabled run turns the attention activation row into `ReStickifyOpLx`, but that
specific run aborted later with an LX capacity/chunk fit issue. Later
kernel-neighbor artifacts got a profiled passing comparison.

## Prior Approaches Considered

### Explicit Coordinate Remap

Branches:

```text
Torch: swiglu-ws-co-remap
Deeptools: coordinate-remap patch branches
```

This inserted explicit movement objects from Torch and got useful SwiGLU
correctness/perf evidence. It proved the optimization mattered, but the design
was not ideal for production because it leaned on explicit data-op/mixed-SDSC
style plumbing while the interface direction moved toward DLDSC-only bundles and
backend relayout insertion.

Historical PR/reference:

```text
https://github.com/torch-spyre/torch-spyre/pull/2789
```

### Ranged STCDPOpLx Carrier

This re-used existing STCDPOpLx movement more directly and was better than
inventing a completely new public op. Still, the cleanest contract became:
Torch emits logical DLDSC coordinates/classification, Deeptools realizes the
physical STCDPOpLx/ReStickifyOpLx machinery.

### Zero-Deeptools-Change Path

Explored and rejected for production. Torch can emit more DLDSC metadata, but
without backend changes Deeptools does not know when/how to reserve space,
schedule inserted movement, and bind the consumer operand to the post-relayout
view.

### Backend-Only Automatic Relayout

Charlie/Swagath pointed to backend relayout insertion work. The current
compromise is close to that direction: Torch emits enough DLDSC coordinate
metadata and class information; Deeptools owns physical synthesis. The key
frontend role is still important because work division and reshard cost are
coupled.

### Kernel-Neighbor / Loop-Scoped Input Fetch

This became relevant for matmul operand broadcast. Fully materializing a large
replicated RHS tensor in LX is dangerous or too large. A loop-scoped
kernel-neighbor movement can bring operand chunks where the matmul reader needs
them without pretending the whole replicated operand is resident forever. This
is promising but needs careful extraction.

## Runbooks And Environment Tricks

Granite benchmark/runbook repo:

```text
https://github.ibm.com/Adnan-Hoque1/spyre-granite-e2e-bench
```

Useful pod names:

```text
adnan-spyre-dev-pf
adnan-cdx-spyre-dev-pf
adnan-clc-spyre-dev-pf
namespace: a6-quantization
```

Generic pod execution:

```bash
oc exec -n a6-quantization <pod> -- bash -lc '<command>'
```

Hot reset utility:

```text
/opt/ibm/spyre/senlib/bin/aiu_dd2_hot_reset
```

Split LX wrapper trick:

Torch/frontend often wants:

```bash
export DXP_LX_FRAC_AVAIL=0
```

But backend DXP needs workspace for inserted movement:

```bash
export DXP_BACKEND_LX_FRAC_AVAIL=1
```

The wrapper maps:

```bash
if [[ -n "${DXP_BACKEND_LX_FRAC_AVAIL:-}" ]]; then
  export DXP_LX_FRAC_AVAIL="$DXP_BACKEND_LX_FRAC_AVAIL"
fi
```

This means Torch sees full LX planning while the DXP subprocess sees backend LX
space available. Without this split, the same bundle can fail with:

```text
initial chunk parameters must fit in LX
```

Public feature flag goal:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

Older prototype artifacts also used subflags:

```bash
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
```

Do not present those as the final desired interface. The intended user-facing
gate is one flag.

## Patch Bundles Given To Antoni

Archived portable patches were created for Antoni's frankenbranch flow:

```text
Torch patch:
https://github.com/AdnanHoque/torch-spyre/blob/b0e39a880d5d4fa5f68c5c3301feaa2ce447ab06/docs/results/granite_e2e/comms_collectives_20260706/torch_patch_rebased_main_20260706/torch_spyre_gather_restickify_rebased_on_main.patch

Deeptools patch:
https://github.com/AdnanHoque/torch-spyre/blob/b0e39a880d5d4fa5f68c5c3301feaa2ce447ab06/docs/results/granite_e2e/comms_collectives_20260706/deeptools_patch_rebased_master_20260706/deeptools_ah_comms_collectives_rebased_on_master.patch
```

Those patches should be treated as prototype sharing artifacts, not PR-clean
source of truth.

## Important Artifacts Index

PR1 / scatter:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/tmp_dldsc_artifacts/dldsc_relayout_1p2_gap_analysis_20260629.md
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/dldsc_collectives_backend_gap_20260702/granite_edge_inventory_clc_20260703.tsv
```

Flash layout all-gather:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/dldsc_flash_runtime_checkpoint_20260702.md
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/flash_contract_20260702/flash_contract_summary.json
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/flash_collectives_20260702/README.md
```

Matmul operand broadcast:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/dldsc_matmul_operand_broadcast_checkpoint_20260702.md
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/dldsc_collectives_backend_gap_20260702/flash_matmul_operand_broadcast_20260703/README.md
```

Granite S512 / kernel-neighbor:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/extract_prs_20260707/torch-artifacts/docs/results/granite_e2e/comms_collectives_20260706/kernel_neighbor_candidate_20260706/runs/granite_s512_profiler_py212_compare/README.md
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/extract_prs_20260707/torch-artifacts/docs/results/granite_e2e/comms_collectives_20260706/kernel_neighbor_candidate_20260706/runs/kernel_neighbor_s512_wall_compare/wall_sdsc_comparison.md
```

Current state snapshots:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/extract_prs_20260707/torch-artifacts/docs/results/granite_e2e/comms_collectives_20260704_current_state.md
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/extract_prs_20260707/torch-artifacts/docs/results/granite_e2e/granite_s512_relayout_clc_20260705.md
```

Partial-view gather:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/extract_prs_20260707/torch-artifacts/docs/results/granite_e2e/comms_collectives_20260706/partial_view_gather_bounded_relayout_20260707/README.md
```

Device/value-correctness follow-up:

```text
/Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/artifact-comms-collectives/docs/results/granite_e2e/comms_collectives_20260704_flash_attention_runtime/value_correctness_followup/
```

## What Worked

1. PR1 scatter is production-shaped and useful.
2. Deeptools can infer and synthesize DLDSC-based scatter relayout from tensor
   distribution vs compute distribution metadata.
3. Flash layout-all-gather/restickify can structurally eliminate the 32
   activation `ReStickifyOpHBM` rows and replace them with `ReStickifyOpLx`.
4. Matmul operand broadcast can be represented logically as
   `matmul_operand_broadcast / all_gather_replicate`.
5. Standalone STCDPOpLx experiments show the ring can express the core movement
   when destination chunk offsets are correct.
6. Kernel-neighbor prototype shows a real Granite S512 kernel-time improvement
   on the attention value-side handoff.

## What Did Not Work Or Is Not PR-Ready

1. Naive direct materialization of full replicated RHS operands is too large or
   too brittle.
2. Simply preserving producer coordinates or replacing them with consumer
   coordinates was rejected by DDC in matmul operand broadcast experiments.
3. Adding a second KERNEL labeled DS to a matmul is not accepted by current
   Deeptools assumptions.
4. Older data-dsc / mixed-SDSC explicit movement is not aligned with the current
   interface direction.
5. Full flash numeric correctness is not a clean gate because baseline flash
   has a separate zero-stride/broadcast-view correctness bug.
6. FFN/SwiGLU activation HBM-backed boundaries are not fully solved by explicit
   relayout rows; this appears more WSR/fused-region-residency shaped.

## Next PR Strategy

Recommended staging:

1. Land PR1 scatter:
   Torch `#2939`, Deeptools `#4408`.
2. Extract PR2 as a small vertical all-gather/restickify slice:
   bounded only, no WSR, no partial-view gather, no generic broadcast/multicast.
3. Keep kernel-neighbor/matmul operand broadcast evidence as either:
   - part of PR2 if the branch is truly lean and tested, or
   - a separate PR2b if it expands the backend surface too much.
4. PR3: generic bounded broadcast/multicast.
5. PR4: bounded partial-view gather with explicit source-offset metadata.
6. Later: reduction/all-reduce classes and WSR integration.

Do not put all remaining collectives into one PR. Deeptools review cost is high.
Each Deeptools PR should be small, vertical, and backed by one bounded fixture.

## Open Technical Questions For Claude To Review

1. Is PR2 extraction currently too broad?
   It is still about 1.6k LOC Torch and 1.9k LOC Deeptools. Review whether
   `matmul_operand_broadcast` and `layout_allgather_restickify` should be one
   PR or split.

2. Is kernel-neighbor the right production shape for matmul operand broadcast?
   It seems more scalable than full replicated LX materialization, but it has
   more backend scheduling implications.

3. Where exactly should the cost model learn communication costs?
   The frontend chooses work division, so it needs scatter/all-gather/broadcast
   cost estimates before picking divisions.

4. How should WSR interact with relayout?
   Relayout should classify and realize bounded communication; WSR should make
   large regions tile-scoped.

5. What is the minimal Deeptools contract for PR2?
   The target is DLDSC metadata plus backend synthesis, not public data-op rows
   in Torch.

6. Can the flash baseline zero-stride correctness bug be isolated from our
   relayout correctness gates?
   If not, create a smaller patterned relayout harness that validates bytes
   immediately after movement and before flash math.

## Suggested Reading Order

1. Read this file.
2. Read the Epic:
   `https://github.com/torch-spyre/torch-spyre/issues/3049`
3. Read PR1:
   `https://github.com/torch-spyre/torch-spyre/pull/2939`
   and `https://github.ibm.com/ai-chip-toolchain/deeptools/pull/4408`
4. Read the Granite edge inventory TSV.
5. Read the flash runtime checkpoint.
6. Read the kernel-neighbor Granite S512 profiler comparison.
7. Only then inspect broad prototype patches.

## One Caution

There are several branches and docs that use overlapping terms:

```text
layout_allgather_restickify
gather_then_restickify
matmul_operand_broadcast
loop_scoped_input_fetch
kernel-neighbor
partial_view_gather
```

Do not assume they are aliases. They are related communication/materialization
strategies at different layers:

- `layout_allgather_restickify`: all-gather plus local restickify/layout change.
- `gather_then_restickify`: backend materialization strategy for bounded
  all-gather/restickify.
- `matmul_operand_broadcast`: logical matmul RHS/KERNEL operand all-gather.
- `loop_scoped_input_fetch` / `kernel-neighbor`: avoid full resident materialization
  by moving operand chunks near the matmul transfer loop.
- `partial_view_gather`: gather from a producer view with nonzero source offset.

Keeping those terms separate is the difference between a clean PR stack and a
mega-branch.
