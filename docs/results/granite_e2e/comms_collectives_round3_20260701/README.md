# Comms Collectives Round 3 - 2026-07-01

This checkpoint extends the Granite/attention communication-class exploration
across the three AIU pods.  It records one attention blocker and two concrete
backend-carrier advances.

## Summary

| lane | pod | result | next gap |
|---|---|---|---|
| attention scatter validation | `adnan-spyre-dev-pf` | isolated the first DXP failure to a baseline `ReStickifyOpHBM` dim-map issue, not scatter relayout metadata | support or avoid this restickify coordinate pattern |
| compact DLDSC/STCDP | `adnan-cdx-spyre-dev-pf` | added and unit-tested LE128 local transfer metadata with independent source and destination byte offsets | emit local LX assemble/extract nodes before stick-addressed ring lowering |
| explicit byte-range remap | `adnan-clc-spyre-dev-pf` | compressed `sdsc_10 Tensor1` from 2,097,152 modeled moves to 128 grouped rows; semantic and DT-table checks pass | DCC stitching rejects grouped dataops with `unit already set for associated schedule step` |

## Attention DXP Dim-Map Blocker

Workspace:

```text
/home/adnan/codex-isolated/dxp_flash_dimmap_20260701_023824
```

Artifact:

```text
attention_dimmap/dxp_flash_dimmap_20260701_023824.tar.gz
```

The first DXP failure in current `test_flash.py` is:

```text
scatter_full_bundle/sdsc_6.json
op = 6_ReStickifyOpHBM
```

Generated op shape:

```text
op='ReStickifyOpHBM'
iteration_space={c0: (32, 1), c1: (4096, 32), c2: (128, 1)}
input:  device_size=[32, 4096, 2, 64],
        coords=[c0, c1, floor(c2/64), Mod(c2, 64)],
        allocation={'lx': 16384}
output: device_size=[32, 128, 64, 64],
        coords=[c0, c2, floor(c1/64), Mod(c1, 64)],
        allocation={'pool': 50331648}
```

DXP fails with:

```text
DtException: Could not find any suitable dimension mapping
```

This is not a scatter-relayout failure:

- `sdsc_6` has no `lx_relayout_classifications`.
- Baseline `sdsc_6` fails with the same DXP exception.
- Scatter-only `sdsc_83` replays successfully.

Interpretation:

The singleton-stick restickify patch gets compilation past an earlier optimizer
blocker, but then exposes an existing unsupported `ReStickifyOpHBM`
shape/layout in DXP/DDL lowering.  Attention scatter validation should not be
judged until this baseline restickify coordinate pattern is supported or
avoided.

## Compact DLDSC/STCDP Local LE128 Metadata

Workspace:

```text
/home/adnan-cdx/codex-isolated/dldsc_collectives_stage_local_20260701_015300
```

Artifact:

```text
dldsc_le128/le128_substick_metadata_20260701_024920.tgz
```

Implemented in the isolated Deeptools tree:

- `dsc/pcfg.h`, `dsc/pcfg.cpp`
  - `LE128BTransferInfo` carries `srcByteOffset`, `dstByteOffset`,
    `addSrcBaseAddr`, and `addDstBaseAddr`.
  - Legacy `Offset` / `addBaseAddr` JSON compatibility is preserved.
- `dcg/dcg_be/dcgbeCodegen.cpp`
  - LX LE128 lowering uses source-side offsets for `LXLU` and destination-side
    offsets for `LXSU`.
- `senulator/pcfgtransfer.cpp`
  - replay selects source or destination offset based on transfer direction.
- `dcg/test/dcg_unit_test.cpp`
  - added `stcdpLibtest.lxLe128LocalSubStickAssembleRoundTrip`.

Validation:

```bash
cmake --build build-stage-local-safe --target dcg_unit_test -j 8
./build-stage-local-safe/dcg/dcg_unit_test \
  --gtest_filter=stcdpLibtest.lxLe128LocalSubStickAssembleRoundTrip \
  --gtest_color=no
cmake --build build-stage-local-safe --target senulator -j 8
git diff --check
```

Result:

```text
new unit test passed
senulator target built
git diff --check clean
```

Interpretation:

The compact path now has the local metadata vocabulary needed for sub-stick
assemble/extract.  The remaining connection point is still STCDP ring-DT:
`SubStickRangeInfo` is collected, but direct ring lowering is intentionally
blocked because ring DT is stick-addressed.  The next implementation step is to
generate local LE128 assemble/extract nodes around whole-stick ring movement.

## Explicit Grouped Byte-Range Remap

Workspace:

```text
/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212
```

Artifact:

```text
explicit_grouped/explicit_range_grouped_sdsc10_20260701_024349.tgz
```

Prototype schema:

```text
grouped-strided-byte-range-v1
```

Target edge:

```text
attention sdsc_10 Tensor1
```

Result:

| field | value |
|---|---:|
| grouped JSON rows | 128 |
| expanded modeled moves | 2,097,152 |
| modeled bytes | 67,108,864 |
| bytes per destination core | 2,097,152 contiguous |
| semantic checker | pass |
| DXP diagnostic DT rows checked | 128 |
| `numTransactions` per DT row | 16,384 |

The old explicit-remap blockers were removed:

- no longer capped at `<=16` ranges;
- no longer limited to `count == 1`;
- grouped ranges parse and lower into diagnostic DT rows.

The remaining backend failure is later:

```text
DtException: unit already set for associated schedule step
source: dcc/src/Stitcher/ModuleStitcher.cpp:279
```

Interpretation:

Grouped explicit ranges avoid the frontend IR blow-up for this attention edge:
2,097,152 individual moves compress to 128 grouped rows.  This makes the
explicit path viable as a research carrier only if Deeptools can schedule and
stitch grouped dataops without one-unit-per-schedule-step conflicts.

## Current Architecture Read

The results still point toward two viable implementation directions:

1. Compact DLDSC contract:
   - Torch marks the tensor-vs-compute coordinate mismatch and communication
     class.
   - Deeptools synthesizes legal whole-stick ring movement plus local
     sub-stick assemble/extract.
   - Best fit for production if the backend is expected to own physical
     movement and scheduling.

2. Explicit grouped remap contract:
   - Torch owns physical movement planning but emits grouped/count-stride
     descriptors instead of expanded rows.
   - Backend executes the grouped transfers.
   - Useful for diagnostics and for cases where frontend scheduling agency is
     required, but it needs stronger grouped dataop scheduling support.

Both paths agree on the same next communication primitive:

```text
whole-stick remote movement + local partial-stick assemble/extract
```

For reduce/all-reduce, byte movement is not enough; the follow-up primitive must
also support local partial-stick reduction/accumulation.
