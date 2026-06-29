# dldsc LX Relayout Granite 1.2x Gap Analysis - 2026-06-29

## Current State

The dldsc LX relayout path is speed-positive on Granite causal prefill, but it does not yet reproduce the earlier approximately `1.2x` result against the current baseline.

| Variant | Kernel ms/iter | Median wall ms | Kernel speedup |
|---|---:|---:|---:|
| Baseline, relayout off | 12.4741 | 19.1460 | 1.000x |
| dldsc relayout, boundary clones, full Torch LX | 10.9780 | 17.7715 | 1.136x |

A `1.2x` result against the current baseline requires about `10.395 ms/iter`, leaving roughly `0.58 ms/iter` still missing from the best current dldsc run.

## What Works

The working class is `scatter`.

A producer owns final tensor slices in LX. A consumer wants the same tensor in a different per-core ownership. Torch records the producer residency in dl-dsc allocation coordinates. Deeptools sees that the LX allocation coordinates differ from the consumer compute split, inserts an internal `LxRelayout` SuperDSC, and realizes it with `STCDPOpLx` before the consumer compute.

This removes five Granite block HBM input/output round trips in the profiled run:

| Communication class | Baseline off | dldsc full Torch LX |
|---|---:|---:|
| HBM input roundtrip candidate | 5 | 0 |
| HBM output spill | 5 | 0 |
| scatter | 0 | 5 |
| missing matmul operand collective | 1 | 1 |

The current dldsc branch therefore already proves that the dl-dsc coordinate contract can carry the basic on-chip relayout handoff.

## Deeptools Mechanism Observed

Current Deeptools master relayout insertion is a resident materialization algorithm:

1. For each LX-pinned input, compare the input allocation `coreIdToWkSlice_` against the consumer `SuperDsc::coreIdToWkSlice_`.
2. If they differ, create a new `*-Relayout` SuperDSC.
3. If the post-relayout tensor form fits in LX, create a `STCDPOpLx` data op internally.
4. Materialize the post-relayout resident LX view before the original consumer.
5. If the post-relayout form does not fit, fall back to an HBM identity relayout.

The local Deeptools worktree used for the successful runs is not pristine master. It includes local fixes around:

- per-piece LX capacity checks instead of checking the full tensor form per core;
- using the relayout tensor's own `primaryDsInfo_` instead of always using output metadata;
- ignoring consumer work-slice dimensions that are not tensor layout dimensions when building relayout pieces;
- a coordinate-consistency fix in `ddc_fold.cpp` for empty allocation maps defaulting to the SDSC compute map.

Those fixes are still in the resident-materialization family. They do not add the missing communication class below.

## Remaining Gap: Attention Value Operand

The remaining high-value edge is the attention PV/value matmul operand:

| Edge | Planner kind | Communication pattern | Current result |
|---|---|---|---|
| `buf21 -> buf22` | `matmul_operand_broadcast` | `all_gather_replicate` | HBM-backed / unsupported by resident relayout |

Reduced repro:

`docs/source/compiler/artifacts/dldsc_granite_prefill_20260629/buf21_dxp_repro`

Key diagnostic from `Dxp::insertRelayoutSdsc`:

```text
LXREL_DIAG consider sdsc=16_batchmatmul lds=Tensor1 pinned=1 allocCoreMap=32 sdscCoreMap=32
LXREL_DIAG size sdsc=16_batchmatmul lds=Tensor1 out_form_size=4.1943e+06 out_piece_size=4.1943e+06
LXREL_DIAG no_lx_space sdsc=16_batchmatmul lds=Tensor1 core=0
LXREL_DIAG choose_hbm sdsc=16_batchmatmul lds=Tensor1
```

Why this happens:

- Producer residency for Tensor1 is sharded across the value/output dimension: `out=32`.
- Consumer AV matmul compute is split across `mb=32`.
- For Tensor1's layout dimensions, the consumer split does not partition the operand.
- The resident relayout algorithm therefore asks every consumer core to hold the full value operand piece, about `4 MiB/core`.
- That does not fit and is not the intended algorithm.

This is not another `scatter` case. It is a matmul operand collective/broadcast class.

## Cheap Alternatives Checked

### Clear Residency

`buf21_clear_residency_repro` clears Tensor1 custom residency. It can compile, but it removes the actual on-chip handoff semantics. This is a diagnostic only, not a valid optimization.

### Replicated Residency Coordinates

`buf21_replicated_residency_repro` marks Tensor1 as redundantly resident on all consumer cores. This still fails DDC coordinate consistency because the matmul transfer has temporal `in/out/x` loops that the static allocation coordinates do not represent. It also lies about physical data movement; no gather/replicate movement has occurred.

### Force AV Matmul Batch Split

A diagnostic changed the AV matmul split from `mb=32` to `x=32`. It compiled, but regressed performance:

| Variant | Kernel ms/iter | Median wall ms |
|---|---:|---:|
| Best default dldsc relayout | 10.9780 | 17.7715 |
| Forced value-BMM batch split | 12.1928 | 19.1263 |

This rules out a simple work-division override as the route to `1.2x`.

## Existing Deeptools Collective Code

Deeptools has `dsm/coll` support for graph-level collectives such as `AllGather`. That code is for iSenGraph collective nodes and decomposes them into RDMA/HDMA/send/recv style graph operations, primarily inter-card or graph-collective lowering.

It is not currently wired into `Dxp::insertRelayoutSdsc`, which operates on SDSC allocation/compute incompatibilities and synthesizes internal `STCDPOpLx` relayouts. The collective algorithms may be useful for terminology/costing, but they are not a drop-in fix for this intra-AIU LX matmul operand path.

## Required Next Communication Class

To close the remaining Granite prefill gap, the planner/backend contract needs a class beyond resident scatter:

`matmul_operand_broadcast` / `all_gather_replicate`

Expected semantics:

1. Producer owns disjoint operand shards in LX.
2. Consumer matmul needs those shards as a non-primary operand while its compute is split along another dimension.
3. Backend lowers the movement as loop-scoped/staged operand movement into the matmul transfer schedule.
4. It does not materialize a full resident post-relayout operand on every consumer core.
5. WSR can decide how much of the operand is staged at once, but the relayout planner still needs to name and cost this communication class.

## Architectural Readout

The dldsc coordinate contract is aligned with the North Star for `scatter`: frontend chooses work division and emits tensor/compute coordinates; backend synthesizes physical LX movement.

For Granite attention, the current backend synthesis is not rich enough. Coordinates can describe the mismatch, but the backend only realizes the resident materialization strategy. The missing piece is a backend lowering class that treats the mismatch as a matmul operand collective instead of a resident scatter.

So the current dldsc path is production-shaped for PR1, but reproducing the full old `1.2x` Granite block speedup requires one additional communication class, not another capacity sweep or broader scatter guard.
