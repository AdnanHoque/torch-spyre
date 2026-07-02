# Flash DLDSC STCDP Fission And Generic Relayout Probes - 2026-07-02

## Summary

This checkpoint tests two plausible backend explanations for the value-wrong DLDSC flash attention run:

1. `STCDPOpLx` fission mis-detects relayout pieces because it compares input pieces against `pieces.at(op)` instead of `outPieces.at(op)`.
2. The special mixed-dataop `layout_allgather_restickify` helper may be the wrong carrier; perhaps the existing standalone `LxRelayout` SDSC path can handle the edge if Torch provides consumer-labelled coordinates.

Neither probe recovered correctness. Both runs still fail at approximately 99.2% mismatched elements. This strongly suggests the remaining gap is a real backend materialization gap for renamed/grouped layout-allgather, not a simple fission typo or generic relayout routing issue.

## Workspace

Pod:

```text
adnan-cdx-spyre-dev-pf
```

Root:

```text
/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525
```

Torch tree:

```text
/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/torch-spyre
```

Deeptools tree:

```text
/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/deeptools
```

Important caveat: this CDX workspace includes local experimental Torch-side consumer-coordinate probe edits. Treat these runs as backend probes, not a production branch state.

## Representative Edge

The failing repeated edge remains:

```text
sdsc_188 mul
  -> sdsc_189 ReStickifyOpLx
  -> sdsc_190 batchmatmul Tensor1 KERNEL input
```

Communication class:

```text
layout_allgather_restickify
```

Rename contract:

```text
restickify.x   -> batchmatmul.out
restickify.out -> batchmatmul.in
restickify.mb  -> batchmatmul.x
```

Expected grouped fanout example:

```text
source core 4 = group 0, producer chunk 1
expected destinations = 0,4,8,12,16,20,24,28
```

## Probe 1: STCDPOpLx Fission Indexing Fix

Patch:

```cpp
// dsm/sen_data_ops.cpp, Dsm::doStcdpDataDscFission
- auto& outPiece = pieces.at(op);
+ auto& outPiece = outPieces.at(op);
```

Build:

```text
cd build-deeptools
ninja dsm/libdsm.so -j 8
```

Run:

```text
runs/test_flash_fissionfix_20260702_175617
```

Result:

```text
Mismatched elements: 16646923 / 16777216 (99.2%)
Greatest absolute difference: inf at index (0, 0, 0, 0)
```

Conclusion: the indexing bug is likely real cleanup, but it is not sufficient for flash correctness.

## Probe 2: Force Existing Standalone LxRelayout Path

Patch:

```text
Bypass attachLayoutAllgatherRestickifyInputFetch() and let the classified layout_allgather_restickify edge fall through to the existing standalone LxRelayout SDSC insertion path.
```

Build:

```text
cd build-deeptools
ninja dxp_standalone -j 8
```

Run:

```text
runs/test_flash_genericrelayout_fissionfix_20260702_180256
```

Result:

```text
Mismatched elements: 16646958 / 16777216 (99.2%)
Greatest absolute difference: inf at index (0, 0, 0, 0)
```

Conclusion: generic standalone `LxRelayout` is not sufficient for the renamed layout-allgather/restickify class. It does not by itself encode the producer chunk axis into the consumer operand placement correctly.

## Current Interpretation

DLDSC is still the right contract direction. The frontend is successfully expressing the relevant edge as a logical communication class with dimension rename. The backend can parse the metadata and emit plan artifacts. The missing piece is the physical materialization of this class.

What did not work:

```text
mixed dataOpdscs_ helper with grouped fanout
expanded destination allocation
local-range address offsets
STCDPOpLx fission indexing cleanup alone
generic standalone LxRelayout fallback
```

What remains needed:

```text
A dedicated backend materializer for layout_allgather_restickify that:
  - consumes the DLDSC coordinate contract;
  - applies dimension_rename explicitly;
  - creates grouped all-gather source/destination pieces;
  - allocates destination LX for the consumer operand;
  - preserves producer chunk offsets in consumer coordinates;
  - schedules movement before the consuming batchmatmul;
  - lowers through a carrier Deeptools already supports reliably.
```

## Probe Diff

The exact Deeptools experiment diff is archived next to this file:

```text
deeptools_fission_generic_relayout_probe.diff
```
