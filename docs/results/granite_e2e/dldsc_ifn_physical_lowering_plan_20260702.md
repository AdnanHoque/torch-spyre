# DLDSC Matmul Operand Broadcast Physical-Lowering Plan - 2026-07-02

## Context

The current DLDSC path has two working layers:

1. Torch emits a logical contract for the Granite attention value operand:
   `matmul_operand_broadcast` / `all_gather_replicate`.
2. Deeptools parses the contract and synthesizes a deterministic logical plan:
   32 producer shards x 32 consumer replicas = 1024 logical transfers.

The current Deeptools branch intentionally fails closed after writing the backend
plan artifact. This is correct for the checkpoint because falling through to the
generic resident relayout path would allocate the full value operand per consumer
core and recreate the original capacity failure.

## Why This Is Not Scatter

PR1 scatter covers one-to-one coordinate movement where the destination resident
slice can be allocated per consumer core. The Granite value operand is different:

- producer owns operand shards along an operand/tensor dimension;
- consumer matmul is split along `mb`;
- `mb` is not a value-operand tensor dimension;
- a resident post-relayout allocation becomes full-operand replication.

The correct lowering is loop-scoped movement into the matmul transfer loop, not a
resident LX relayout allocation.

## Existing Deeptools Hook To Reuse

Existing code already has an input-neighbor-fetch route:

- `DcgManager::runDcgForDataOpsDlOps`
- `DcgFE::generatePcfgIRForDataOpInpFetch`
- `DcgFE::createPcfgForInputFetchNeighbor`
- data-op descriptor shape based on `STCDPOpLx`

The schedule marker is already encoded in `DscScheduleStep`:

```cpp
DscScheduleStep(int datadsc_idx, int dldsc_idx, bool before_sync, bool after_sync)
```

A step with both `datadsc_idx >= 0` and `dldsc_idx >= 0` is treated as input
neighbor fetch.

## Current DXP Gap

`Dxp::runCodegen` currently routes pure data-op SDSCs differently from DL SDSCs:

```cpp
if (sdsc->dscs_.size() == 0 && sdsc->dataOpdscs_.size() > 0) {
  dcg.runDcg(*sdsc);
} else {
  dcg.runDcgForDlOpsStandalone(*sdsc);
}
```

That means a mixed SDSC containing both a DL matmul and a generated IFN data-op
will not naturally reach `runDcgForDataOpsDlOps` unless DXP adds an explicit
mixed route.

## Stage-Sweep Evidence

Historical stage sweep:

`/home/adnan/codex-isolated/comms_collectives_20260629/runs/dxp_nonpaired_stage_sweep_clc_20260630_135558`

All stage factors compiled:

- stage 1: 32 one-producer dataops
- stage 2: 16 dataops
- stage 4: 8 dataops
- stage 8: 4 dataops
- stage 16: 2 dataops
- stage 32: 1 dataop

The stage-32 standalone data-op has:

- op: `STCDPOpLx`
- input lds: `Tensor1-LxInputNeighborFetch-inp`
- output lds: `Tensor1`
- layout: `out,in,x`
- stick dim: `out`
- producer pieces: 32 shards of `out:4,in:512,x:32`
- `prodConsList`: each producer shard fans out to all 32 consumer cores

This proves the movement descriptor shape is viable. It does not by itself prove
mixed IFN + DL matmul schedule fusion in DXP.

## Next Backend Patch Shape

1. In DXP, when `matmul_operand_broadcast` metadata matches an LX-pinned input,
   synthesize an IFN-style `STCDPOpLx` data-op from the consumer SDSC input lds
   and allocation coordinates.
2. Attach the data-op to the consumer SDSC instead of creating a resident relayout
   SuperDSC.
3. Rewrite the consumer core schedule so each participating core executes one
   mixed step, e.g. `DscScheduleStep(0, 0, false, false)`.
4. Update `Dxp::runCodegen` to route mixed DL+data-op SDSCs through
   `runDcgForDataOpsDlOps`.
5. Validate on the reduced buf21 artifact before full Granite.

## Acceptance For This Primitive

- Reduced buf21 passes DXP without HBM fallback.
- Full Granite attention value operand no longer falls back to resident full
  materialization.
- SDSC artifacts show `matmul_operand_broadcast` / `all_gather_replicate` and no
  replacement `ReStickifyOpHBM` for this activation edge.
- Hardware smoke reaches runtime before performance is measured.
