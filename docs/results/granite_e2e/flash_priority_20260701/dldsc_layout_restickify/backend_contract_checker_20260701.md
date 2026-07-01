# Flash DLDSC Layout-All-Gather Restickify Backend Gate

## Scope

This checkpoint records the backend-side contract gate for the latest flash attention spill class. All work was done pod-local on `adnan-cdx-spyre-dev-pf` using the Deeptools workspace under:

`/home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300/deeptools`

The flash edge being modeled is the already-identified latest flash handoff:

`mul -> ReStickifyOpHBM -> batchmatmul KERNEL`

This is not scatter. The producer and consumer disagree on layout/stick view and every consumer core in a batch-local group needs multiple producer chunks. The communication class is therefore:

`layout_allgather_restickify` / `all_gather`

## What Changed In The Backend Prototype

A small fail-closed Deeptools utility checker was added in the CDX workspace:

- `util/LayoutAllgatherRestickify.h`
- `util/LayoutAllgatherRestickify.cpp`
- `util/test/LayoutAllgatherRestickify_unit_test.cpp`
- `util/CMakeLists.txt`

The checker accepts a serialized contract only when it contains all of the required logical metadata:

- `kind == layout_allgather_restickify`
- `communication_class == all_gather`
- producer, restickify, and consumer op names
- producer, restickify, and consumer work-slice dimensions
- producer layout, restickify kernel layout, and consumer kernel layout
- explicit dimension rename from restickify view to consumer BMM view
- `requires_staged_realization == true`

It intentionally rejects scatter-shaped metadata and partial contracts. That matters because treating this edge as scatter would silently drop the layout transform and replication semantics.

## Validation

Build command on CDX:

```bash
cd /home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300/deeptools/build-stage-local-safe
ninja util_unit_test
```

Focused test command:

```bash
./util/util_unit_test --gtest_filter=LayoutAllgatherRestickify.*
```

Result:

```text
[==========] Running 4 tests from 1 test suite.
[----------] 4 tests from LayoutAllgatherRestickify
[ RUN      ] LayoutAllgatherRestickify.acceptsCompleteFlashContract
[       OK ] LayoutAllgatherRestickify.acceptsCompleteFlashContract (0 ms)
[ RUN      ] LayoutAllgatherRestickify.rejectsScatterMetadata
[       OK ] LayoutAllgatherRestickify.rejectsScatterMetadata (0 ms)
[ RUN      ] LayoutAllgatherRestickify.rejectsMissingLayoutRename
[       OK ] LayoutAllgatherRestickify.rejectsMissingLayoutRename (0 ms)
[ RUN      ] LayoutAllgatherRestickify.rejectsZeroWidthSplit
[       OK ] LayoutAllgatherRestickify.rejectsZeroWidthSplit (0 ms)
[  PASSED  ] 4 tests.
```

## Artifact

Patch snapshot:

`deeptools_layout_allgather_restickify_checker.patch`

Patch size/checksum from CDX:

```text
336 lines
sha256 0a36a1857170917eec9ea59b1a65b73df81a72ad5f771625ba42f18145a2f5e2
```

## Current Gap

This is a backend gate/checker, not a complete lowering. The next backend step is to synthesize the movement from the validated contract:

1. restickify producer LX chunks into the consumer KERNEL logical view, staying on chip;
2. group transfer by batch-local destination groups;
3. replicate/gather all needed producer chunks into each consumer core’s KERNEL operand;
4. preserve consumer operand lifetime so `batchmatmul` reads LX rather than forcing HBM.

The explicit-remap path already showed that grouped byte-range movement can route through DXP after the DCC stitcher fix, but the latest flash edge still needs this higher-level layout/restickify/all-gather contract before it is safe to mutate real flash SDSCs.
