# DLDSC Mixed IFN Route Checkpoint - 2026-07-02

## Summary

A narrow Deeptools infrastructure patch now routes scheduled mixed DataOp+DL
SDSCs through the existing input-neighbor-fetch-aware DCG path.

This does not complete Granite value-operand physical lowering. It removes one
backend routing gap and exposes the next concrete blocker: the existing
InputFetchNeighbor helper assumes `DsTypes::INPUT`, while the Granite value
operand is the non-primary/RHS `batchmatmul` operand.

## Deeptools Branch

- repo: `Adnan-Hoque1/deeptools`
- branch: `ah/comms-collectives`
- new commit: `3c7b754f0`
- commit message: `[DXP] Route scheduled mixed input-fetch SDSCs`
- note: DCO signed. Not PGP signed on CDX because the pod does not have the GPG
  secret key.

## What Changed

1. `Dxp::runCodegen` now detects SDSCs that have both:
   - a DL DSC,
   - a data-op DSC,
   - a core schedule step with both `datadsc_idx >= 0` and `dldsc_idx >= 0`.

   Those SDSCs are routed through:

   ```cpp
   dcg.runDcgForDataOpsDlOps(*sdsc)
   ```

   instead of the pure DL standalone path.

2. `DcgFE::generatePcfgIRForDataOpInpFetch` now uses the scheduled
   `datadscIdx` instead of always using `dataOpdscs_[0]`.

## Why This Matters

`DscScheduleStep` already encodes the mixed IFN case:

```cpp
DscScheduleStep(datadsc_idx, dldsc_idx, before_sync, after_sync)
```

When both `datadsc_idx` and `dldsc_idx` are nonnegative,
`DcgManager::runDcgForDataOpsDlOps` treats the schedule step as an input fetch.
The DXP frontend was not routing mixed SDSCs into that path. This patch makes the
route available without changing pure DL or pure data-op SDSCs.

## Validation

On `adnan-cdx-spyre-dev-pf`:

```bash
cmake --build build-dxp-focused --target dxp_standalone -j 16
build-focused/util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*" --gtest_brief=1
```

Result:

- `dxp_standalone` built successfully.
- `LayoutAllgatherRestickify.*`: 19 tests passed.

## Remaining Blocker

The matmul operand broadcast path still cannot complete physical lowering because
InputFetchNeighbor is hard-wired to the primary input/output roles:

- `inputNeighFetchOp.cpp` checks `primaryDsInfo_[DsTypes::INPUT]`.
- `inputNeighFetchOp.cpp` builds IFN lds info from `INPUT` and `OUTPUT`.
- `inputNeighFetchOp.cpp` filters DB loop order through `primaryDsInfo_[INPUT]`.

The Granite value operand is the RHS/non-primary `batchmatmul` operand. The next
backend patch must make IFN operand-aware, keyed by the DLDSC operand/read index
or the matched `LabeledDsInfo`, rather than assuming `DsTypes::INPUT`.

## Next Patch Shape

1. Generate or attach an IFN-style `STCDPOpLx` data-op for the matched RHS/value
   operand.
2. Schedule it with the consumer matmul as a mixed step, e.g.
   `DscScheduleStep(0, 0, false, false)`.
3. Extend InputFetchNeighbor helpers to operate on the matched input `LabeledDsInfo`
   instead of only `DsTypes::INPUT`.
4. Validate first on reduced `buf21`, then on full Granite.
