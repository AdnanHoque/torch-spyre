# Deeptools patch rebased on master

This directory archives the portable Deeptools patch for the experimental Granite/flash LX communication work.

Source branch: Adnan-Hoque1/deeptools ah/comms-collectives
Head: 9092d48e0ae6af1d7cc66e4bd6128f2196e7f495
Base: origin/master 0a9da5eb19d08712383312bb7dec18fbd7caf711
Generated: 2026-07-06

## Gate

Normal Torch-launched runs should use one flag:

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
```

For direct manual SDSC replay that bypasses Torch, also provide backend LX workspace:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1 DXP_LX_FRAC_AVAIL=1 dxp_standalone -d path/to/sdsc_bundle
```

## Included changes

- Gate the experimental Deeptools relayout paths behind `SPYRE_LX_PLANNER_RELAYOUT`.
- Keep the older `DEEPTOOLS_ENABLE_*` flags as diagnostic aliases.
- Add coordinate-overlap grouping for matmul operand relayouts where producer and consumer splits differ, e.g. producer 32-way to consumer 8-way replicated groups.

## Focused validation

- `LayoutAllgatherRestickify.*`: 32 passed
- `DxpTestFixture.CoreWorkDivIncomptLxRelayout*`: 2 passed

## Current known gap

The full Granite S512 all-gather/replicate matmul operand replay now computes the correct logical transfer map, but the current physical carrier still fails DCC IBUFF at 134/128. See `../granite_s512_coord_overlap_ibuff_20260706/` for the replay artifact and tail.
