# CDX Preservation And Staged Next Slice - 2026-07-06

This note records the state after moving between pods during the Granite
collectives work. The key point is that the CDX progress was not lost. It was
preserved on user-owned fork branches and the current clean DEV branches are
ahead of those preserved CDX prototypes.

## Branch State

Torch artifact branch:

```text
repo: AdnanHoque/torch-spyre
clean branch: ah/comms-collectives
clean HEAD on DEV: a7d33b1
preserved CDX branch: ah/comms-collectives-cdx-prototype-20260706
preserved CDX HEAD: bf61f811
relationship: preserved CDX branch is an ancestor of clean branch
```

Deeptools prototype branch:

```text
repo: Adnan-Hoque1/deeptools
clean branch: ah/comms-collectives
clean HEAD on DEV: eb68de6f7
preserved CDX branch: ah/comms-collectives-cdx-prototype-20260706
preserved CDX HEAD: fa30750e1
relationship: preserved CDX branch is an ancestor of clean branch
```

The clean branches therefore remain the source of truth. The CDX branches are
historical evidence and a fallback source for specific prototype snippets, not
branches to merge wholesale.

## What CDX Contributed

The CDX prototype explored grouped matmul operand movement and taught the most
important correctness boundary:

```text
direct producer-layout LX -> final KERNEL operand movement is value-unsafe
when the edge also needs layout conversion
```

The clean branch now preserves that lesson by fail-closing the direct
KERNEL-neighbor shortcut unless an explicit diagnostic escape hatch is set.

## Current Technical Direction

The correct decomposition for Granite/attention matmul operand collectives is:

```text
producer LX/source-layout shards
  -> STCDPOpLx gather/all-gather into source-layout LX staging
  -> local ReStickifyOpLx into the consumer KERNEL operand layout
  -> batchmatmul consumes the converted operand
```

Torch already emits the logical DLDSC contract for this edge:

```text
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
materialization_pattern = all_gather_replicate_with_layout_conversion
requires_layout_conversion = true
realization_strategy = gather_then_restickify
```

The Deeptools utility layer already accepts this staged strategy and expands
logical core-pair movement. The remaining implementation gap is physical DXP/DCG
lowering: allocate the staging buffer, schedule the gather, schedule the local
`ReStickifyOpLx`, and bind the converted operand to the consumer matmul without
falling back through HBM.

## Focused Gate Commands

Known-good DEV command after repairing the shared LLVM checkout:

```bash
cd /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/deeptools
export LD_LIBRARY_PATH=$PWD/build-deeptools/common:$PWD/build-deeptools/dxp:$PWD/build-deeptools/dsc:$PWD/build-deeptools/dcg:$PWD/build-deeptools/dcg/dcg_fe:$PWD/build-deeptools/dcg/dcg_fe/scheduler:$PWD/build-deeptools/ddc:$PWD/build-deeptools/dcc/lib:$PWD/build-deeptools/util:$PWD/build-deeptools/external/g3log:$PWD/build-deeptools/external/json11:${LD_LIBRARY_PATH:-}
./build-deeptools/util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
./build-deeptools/dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"
```

Last observed result before this note:

```text
LayoutAllgatherRestickify.*: 26/26 passed
DxpTestFixture.CoreWorkDivIncomptLxRelayout*: 2/2 passed
```

## LLVM Checkout Repair Note

Some isolated Deeptools configure attempts damaged the shared LLVM checkout by
leaving missing or partial files under `/home/adnan/dt-inductor/llvm-project`.
The repair that worked on DEV was to restore the expected source revision from
the local git object database:

```bash
cd /home/adnan/dt-inductor/llvm-project
TARGET=$(grep LLVM_REVISION /home/adnan/dt-inductor/build/llvm/include/llvm/Support/VCSRevision.h | cut -d\" -f2)
git archive "$TARGET" llvm/include/llvm/ADT llvm/include/llvm/Support mlir/include/mlir | tar -x -C /home/adnan/dt-inductor/llvm-project
```

Avoid fresh external LLVM configure/builds in arbitrary isolated worktrees unless
the LLVM source path is pinned and known-good.

## Next Slice
