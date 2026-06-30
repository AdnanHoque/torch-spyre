# Parallel Agent Results - Granite Non-Weight HBM Spills

Date: 2026-06-30

This directory records two isolated implementation lanes for removing the
remaining non-weight HBM round trips in the Granite prefill block.

## Executive Summary

PR1 scatter relayout already handles the resident owner-core permutation class.
The guarded Granite inventory shows:

| communication class | count | status |
|---|---:|---|
| `scatter` | 14 | handled |
| `layout_restickify_weight` | 4 | out of scope; offline weight prelayout owns these |
| `layout_restickify_activation` | 1 | remaining gap |
| `matmul_operand_broadcast` / `all_gather_replicate` | 1 | remaining gap |

Two agents tested the two architecture directions:

| lane | pod | result |
|---|---|---|
| dl-dsc backend-derived relayout | `adnan-spyre-dev-pf` | safe, but current coordinate contract is insufficient for true layout/stick transforms |
| explicit data-op / `STCDPOpLx` relayout | `adnan-cdx-spyre-dev-pf` | staged all-gather for `sdsc_18` DXP-replays; `sdsc_10` still needs grouped/ranged sub-stick movement |

The useful conclusion is that removing every non-weight spill is not just “more
scatter.”  The two remaining in-scope rows require two additional concepts:

1. `layout_restickify_activation`: an on-chip layout/stick transform contract.
2. `matmul_operand_broadcast`: staged or loop-scoped all-gather/replicate
   movement for matmul operands.

## dl-dsc Lane

Artifact bundle:

```text
dldsc_agent_artifacts_20260630.tgz
```

Isolated pod workdir:

```text
/home/adnan/codex-isolated/20260630_190544
```

Branches:

| repo | branch | commit |
|---|---|---|
| Torch | `ah/comms-collectives-dldsc-agent` | `75040ee6d9f48518d0c194b72d1075035bb37b7b` |
| Deeptools | `ah/comms-collectives-dldsc-agent` | `b0d94ac421cdde2d0472e0d2a89df962d4e0751e` |

The dl-dsc lane attempted a computed-activation-only `ReStickifyOpLx` guard.
It intentionally did not touch weight restickifies.

Result:

| metric | value |
|---|---:|
| median wall ms | `31.861305236816406` |
| kernel ms / iter | `12.048442999999999` |
| memory ms / iter | `0.3565086666666667` |
| `ReStickifyOpHBM` rows | `5` |
| `ReStickifyOpLx` rows | `0` |
| `scatter` realized | `14` |
| `layout_restickify_activation` realized | `0` |
| `matmul_operand_broadcast` realized | `0` |

The marker reached generated code, but the target restickify output allocation
was still `pool`, not `lx`.  The guard therefore correctly left the row as
`ReStickifyOpHBM`.

Design finding:

`allocateCoordinates_.coreIdToWkSlice_` can express producer/consumer ownership
mismatch.  It cannot by itself express the physical stick/layout transform in
the computed activation restickify.  That path needs a richer contract carrying
source layout, destination layout, operand identity, and computed-vs-weight
scope.

## Explicit STCDPOpLx Lane

Artifact bundle:

```text
stcdp_agent_artifacts_20260630.tgz
```

Isolated pod workdir:

```text
/home/adnan-cdx/codex-isolated/comms_collectives_stcdp_agent_20260630_190747
```

Branches:

| repo | branch | commit |
|---|---|---|
| Torch | `ah/comms-collectives-stcdp-agent` | `e419576527726bdcff7b28f5a0b303eeb31ac6b9` |
| Deeptools | `ah/comms-collectives-stcdp-agent` | `693eef6c4a69154dc09e745deb222a9131c5c047` |

Validation:

| check | result |
|---|---|
| Torch focused test `tests/inductor/test_lx_relayout_dldsc.py` | `12 passed in 9.71s` |
| patched `dxp_standalone` build | passed |
| focused `sdsc_18` replay | passed, `RC=0`, `wall_sec=5` |
| staged descriptors emitted for `sdsc_18` | 8 `18_batchmatmul-dataop-*.json` descriptors |
| Granite causal prefill profile | failed during DXP compile before trace output |

The explicit lane successfully validated staged `STCDPOpLx` lowering for the
attention `sdsc_18` `matmul_operand_broadcast` / `all_gather_replicate` case.
It uses 8 staged data-op descriptors rather than one monolithic full-replication
table.

Remaining blocker:

Fresh attention replay fails for `sdsc_10`:

```text
stcdpOp.cpp:4374:
op->inpSP_.at(inpSPIdx).dimToSize_.at(dimNameOuter) >= stickDim
```

Interpretation:

`sdsc_10` wants a KERNEL stick with `out=64`, but the producer restickified
activation is split as `out=256 / 32 = 8` per producer core.  A descriptor that
enumerates one producer-core fragment at a time is therefore sub-stick.  This
class needs grouped or ranged movement descriptors that combine adjacent
producer slices into stick-sized pieces before lowering.

## Current Recommendation

Keep PR1 scoped to resident scatter.  For the next research branch, implement
the two missing non-weight classes explicitly:

1. Add a real `layout_restickify_activation` contract.
2. Add grouped/ranged staged operand movement for `matmul_operand_broadcast`.

The dl-dsc direction remains the cleaner production contract for ownership
metadata, but the explicit STCDP lane is currently more useful for discovering
the required physical grouping constraints.  The next backend design should
preserve the dl-dsc contract while borrowing the explicit lane's staged/grouped
movement lessons.

## Files In This Directory

| file | contents |
|---|---|
| `dldsc_agent_artifacts_20260630.tgz` | dl-dsc lane logs, patches, and run evidence |
| `stcdp_agent_artifacts_20260630.tgz` | explicit STCDP lane patches, logs, focused replay bundles, and validation output |
