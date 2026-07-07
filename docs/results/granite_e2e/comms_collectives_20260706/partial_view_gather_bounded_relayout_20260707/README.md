# Bounded Partial-View Gather Relayout - 2026-07-07

This checkpoint records the first bounded physical realization for the
`partial_view_gather` communication class.

## What Changed

Torch now emits enough DLDSC metadata for a partial LX producer view:

- source tensor name;
- target tensor name;
- constant source offset in elements;
- source and target LX layout/distribution metadata when available.

Deeptools now accepts the named `partial_view_gather` class for bounded LX
relayouts and adjusts the source LX base address by `source_offset_elems *
wordLength` before using the existing STCDPOpLx relayout mechanics.

This is intentionally still a bounded-tile substrate feature. It does not add a
full-tensor streaming system and does not change the WSR boundary.

## Code Heads

Torch:

```text
repo: AdnanHoque/torch-spyre
branch: gather-restickify
commit: bced14b4 inductor: enrich partial-view gather contracts
```

Deeptools:

```text
repo: Adnan-Hoque1/deeptools
branch: ah/comms-collectives
commit: 9cd9c79c3 [DXP] test partial-view gather offset validation
previous source-offset assertion commit: 53ee16264 [DXP] assert partial-view gather source offset
previous realization commit: 2fa9220a6 [DXP] realize bounded partial-view gather relayout
```

## Archived Patches

- `patches/torch_partial_view_gather_contract_enrichment_bced14b4.patch`
- `patches/deeptools_bounded_partial_view_gather_2fa9220a6.patch`
- `patches/deeptools_partial_view_gather_source_offset_assert_53ee16264.patch`
- `patches/deeptools_partial_view_gather_offset_validation_tests_9cd9c79c3.patch`

## Validation

Ran on `adnan-cdx-spyre-dev-pf`.

DXP focused regression:

```bash
./build-deeptools/dxp/dxp_unit_test \
  --gtest_filter="DxpTestFixture.MatmulOperandBroadcastPattern*:DxpTestFixture.MatmulOperandBroadcastChunkCapFailsClosed:DxpTestFixture.CoreWorkDivIncomptLxRelayout*:DxpTestFixture.PartialViewGatherBoundedOffsetRelayoutCompiles"
```

Result:

```text
8 tests passed
```

All-gather/restickify utility regression:

```bash
./build-deeptools/util/util_unit_test \
  --gtest_filter="LayoutAllgatherRestickify.*"
```

Result:

```text
32 tests passed
```


The 8-test DXP focused regression includes positive bounded realization plus
fail-closed coverage for missing and invalid `source_offset_elems`.

Logs:

- `logs/dxp_focused_regression_20260707.log`
- `logs/layout_allgather_restickify_20260707.log`
- `logs/dxp_focused_regression_after_offset_assert_20260707.log`
- `logs/layout_allgather_restickify_after_offset_assert_20260707.log`
- `logs/dxp_focused_regression_after_offset_validation_20260707.log`
- `logs/layout_allgather_restickify_after_offset_validation_20260707.log`

## What This Proves

This proves the descriptor/DXP side can realize a bounded partial-view gather
without dropping the producer subview offset. The latest DXP unit test also
asserts that the generated relayout SDSC contains the offset-adjusted LX source
address `156672` for the fixture source base `131072` plus
`12800 * 2` bytes.

Concretely, the backend no longer treats a producer view such as `buf33 +
12800` as if it began at the base of `buf33`. The source base address used for
the LX copy is offset-aware.

## What This Does Not Yet Prove

This is not yet an AIU value-correctness claim for every Granite partial-view
edge. The next step is to run a bounded synthetic value test, then a flash or
Granite structural/value smoke that exercises this exact class.

It also does not make large full activations fit. If the required tile is too
large for LX or would generate an unsafe number of descriptors, the compiler
should preserve HBM fallback or fail closed until WSR provides smaller live
tiles.

