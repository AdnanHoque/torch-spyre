# DLDSC Flash Runtime Checkpoint - 2026-07-02

This checkpoint records the first passing end-to-end runtime/compiler gate for
the DLDSC layout-allgather flash attention path after pinning the matching
Torch and Deeptools branches.

## Source Run

```text
pod: adnan-clc-spyre-dev-pf
run: /home/adnan/codex-isolated/dldsc_flash_new_deeptools_20260702_101806/runs/flash_validation_deeptools_00a37826_ldfix_20260702_102707
command: /home/adnan/codex-isolated/dldsc_flash_new_deeptools_20260702_101806/run_flash_new_deeptools.sh flash_validation_deeptools_00a37826_ldfix
expanded command: <run>/command.txt
```

Inputs:

```text
test-spyre-scripts: git@github.ibm.com:aviros/test-spyre-scripts.git
test_flash.py SHA: afda166e58b23519d0b4ca871350b011b56d91a3
Torch branch: AdnanHoque/torch-spyre ah/comms-collectives
Torch SHA: 95b818680cfffd94baeb474420f4436467474feb
Deeptools branch: Adnan-Hoque1/deeptools ah/comms-collectives
Deeptools SHA: 00a37826a8c8e1b32f97c7d6edbc2527f1359076
```

Key environment:

```text
DEEPTOOLS_PATH=/home/adnan/codex-isolated/dldsc_flash_new_deeptools_20260702_101806/deeptools
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=1
SPYRE_LX_PLANNER_RELAYOUT=1
LX_BOUNDARY_CLONES=1
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
```

The split LX environment is required: Torch sees full frontend LX planning with
`DXP_LX_FRAC_AVAIL=0`, while the DXP wrapper maps
`DXP_BACKEND_LX_FRAC_AVAIL=1` to backend `DXP_LX_FRAC_AVAIL=1` so Deeptools has
workspace for synthesized movement.

## Result

```text
return code: 0
wall: 243s
SDSC JSON files: 550
bundle.mlir files: 3
layout_allgather_restickify backend plan files: 32
kernel_runner RUN entries: 3
stdout marker: SUCCESS
```

This was a no-H2D compile/runtime probe, not a profiler timing run. The wall
time is not a performance number.

## SDSC Evidence

Structured SDSC op counts from the passing run:

| opFuncName | count |
|---|---:|
| mul | 128 |
| add | 96 |
| sub | 64 |
| batchmatmul | 64 |
| exp | 64 |
| max | 34 |
| sum | 32 |
| ReStickifyOpLx | 32 |
| maximum | 32 |
| identity | 3 |
| realdiv | 1 |
| ReStickifyOpHBM | 0 |

The representative baseline flash SDSC analysis had 32 activation
`ReStickifyOpHBM` rows on `mul -> ReStickifyOpHBM -> batchmatmul` handoffs. In
this DLDSC run those activation restickifies are emitted as `ReStickifyOpLx`,
and the downstream `batchmatmul` rows carry the logical relayout contract.

All 32 layout-allgather backend plan files contain:

```text
artifact_kind=layout_allgather_restickify_backend_plan
communication_class=all_gather
movement_pattern=layout_allgather_restickify
logical_transfer_count=256
group_count=4
consumer_cores_per_group=8
```

The `batchmatmul` SDSCs carry `lxRelayoutClassifications_` entries with:

```text
communication_class: all_gather
communication_pattern: layout_allgather_restickify
producer_work_slice_dims: {0: 4, 2: 8}
consumer_work_slice_dims: {0: 4}
max_fanout: 8
max_fanin: 8
transfer_count: 256
restickify_op: ReStickifyOpLx
producer_op: mul
consumer_op: batchmatmul
```

## Interpretation

This run proves the emitted DLDSC/SuperDSC bundle can compile and execute
through the patched Deeptools path for the flash layout-allgather restickify
case. It removes the activation HBM restickify handoff for this emitted flash
bundle by replacing `ReStickifyOpHBM` with `ReStickifyOpLx` and preserving the
logical all-gather contract for backend synthesis.

This does not yet prove a speedup. The next gate is a profiler run using the
kernel-profiler runbook, with the same Torch SHA, Deeptools SHA, and split-LX
environment pinned.

## Remaining Frontend Gap

The newer `ah/comms-collectives-dldsc-agent` branch at
`75040ee6d9f48518d0c194b72d1075035bb37b7b` did not reproduce this passing
path. It failed before SDSC emission on a pointwise stick incompatibility:

```text
buf10 (Pointwise): no mechanism to resolve stick incompatibility
buf4 STL 0 -> Out STL 0: No mechanism to gather elements from multiple sticks into single stick
```

That failure is a separate frontend planning gap: pointwise gather/broadcast of
multiple producer sticks into a consumer stick. It should be treated as the next
DLDSC communication class to support, not as evidence that the passing
layout-allgather restickify path is invalid.
