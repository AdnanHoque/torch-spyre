# Stage 121: Torch-Spyre Descriptor And Packaging Probe

## Summary

This stage moved the Stage 120 LX-to-LX `InputFetchNeighbor` proof one step
closer to Torch-Spyre integration. Torch-Spyre can now emit a default-off
sidecar descriptor for an eligible producer -> restickify -> consumer edge, and
the standalone InputFetchNeighbor probe can consume that descriptor instead of
guessing from adjacent filenames.

This is not a runtime-successful fused bundle yet. It is a cleaner compiler
handoff plus a descriptor-driven Deeptools proof.

## What Changed

- Added `SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1`.
- Bundle generation writes `restickify_lx_neighbor_edges.json` beside
  `bundle.mlir` and `sdsc_*.json`.
- The descriptor records the producer SDSC, restickify SDSC, consumer SDSC,
  source kind/name, Stage 3B core mapping override, and locality certificate.
- `tools/restickify_input_fetch_neighbor_probe.py` can now use
  `--use-lx-neighbor-descriptor` to stage the exact edge emitted by
  Torch-Spyre.
- The senprog wrapper now exits immediately after writing `senprog.txt`; this
  avoids an installed-Deeptools teardown/export crash after the program is
  already emitted.

## Validation Case

Case:

```text
computed_transpose_adds_then_matmul, size=2048
```

The generated first bundle contains:

```text
sdsc_0_add.json
sdsc_1_ReStickifyOpHBM.json
sdsc_2_add.json
```

The emitted descriptor selects edge `0:1:2`:

```text
producer    sdsc_0_add.json
restickify  sdsc_1_ReStickifyOpHBM.json
consumer    sdsc_2_add.json
source      buf0, in_graph_computed
certificate certified_byte_hops=0, certified_bytes_moved=8388608
```

Descriptor-driven InputFetchNeighbor staging succeeded. The generated senprog
for a reverse-core ownership probe had:

```text
hbm                     0
Program for unit l3lu   32
Program for unit l3su   32
L3_LDU                  64
L3_STU                  64
EAR                     64
```

Deeptools program verification reported `Passed`. This confirms that the
Torch-Spyre-emitted descriptor can drive the same HBM-free LX-to-LX movement
path as the prior standalone proof.

Artifacts:

```text
artifacts/stage121_torch_spyre_integration/restickify_lx_neighbor_edges.json
artifacts/stage121_torch_spyre_integration/input_fetch_neighbor_summary.json
artifacts/stage121_torch_spyre_integration/senprog.txt
```

## New Blockers

Two runtime-packaging blockers are now explicit:

1. `InputFetchNeighbor` currently requires every labeled data structure in the
   main/consumer SDSC to be LX/ring/SFP-ring pinned. A targeted probe that only
   pinned the producer output and consumer input failed in Deeptools at
   `inputNeighFetchOp.cpp:30`.

2. A direct attempt to compile a consumer compute SDSC with an internal LX input
   and normal HBM output failed in the normal DXP path because the L3 scheduler
   still expects a valid HBM allocate node for that input
   (`L3DlOpsScheduler.cpp:5864`).

These failures explain why the Stage 120 all-pinned shell is enough to generate
the transfer program but is not yet a value-correct runtime bundle.

## Next Patch Point

The next integration should separate the data-movement shell from the compute
SDSCs:

1. Compile the producer compute SDSC normally.
2. Generate a dedicated all-pinned InputFetchNeighbor shell from the descriptor.
3. Compile a consumer compute SDSC whose selected input is an internal LX input
   while outputs remain normal.
4. Package the three programs in one DXP/Flex artifact.

If current Deeptools cannot compile step 3 through the public SDSC path, the
smallest Deeptools-side change is likely to relax/generalize the consumer
internal-LX input contract rather than changing restickify placement in
Torch-Spyre.
