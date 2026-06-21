# Deeptools LX Coordinate Remap Patch Record

This directory records the Deeptools-side changes needed by Torch's
`LXCoordinateRemapOp` carrier for AIU LX-to-LX relayout.

The Torch branch can be pushed, but we do not currently have a Deeptools fork
to push to.  The patch files here are therefore the source-of-truth handoff for
the backend changes.

## Files

- `deeptools_lx_coordinate_remap_ranged_83f9320.patch`
  - Lean candidate patch copied from the pod artifact
    `/tmp/deeptools-83f9320-ranged-lx-coordinate-remaps.patch`.
  - Patch commit header: `83f9320cd6924833950c1aa362dfdb9abe0c29d7`.
  - Subject: `Lower ranged LX coordinate remaps`.
  - Touched files:
    - `dcg/dcg_fe/pcfg_gen/pcfg_gen.cpp`
    - `dsc/dataOpDsc.cpp`
    - `dsc/dataOpDsc.h`
  - This is the candidate to port onto current Deeptools main.

- `deeptools_lx_coordinate_remap_e2e_wip_20260619_214732.diff`
  - Historical WIP diff copied from the pod artifact
    `/tmp/deeptools-coordinate-remap-e2e-wip-20260619-214732.diff`.
  - Keep this as context for DCC/LBR/static-pinning issues that came up during
    end-to-end bringup.
  - Do not apply this wholesale without re-auditing the broader scheduler and
    DCC changes.

- `sample_mixed_lx_coordinate_remap_sdsc.json`
  - Reduced mixed-SDSC contract sample with one `LXCoordinateRemapOp` data-op,
    one consumer DL row, and a schedule that places remap before compute.
  - Full emitted examples are archived under
    `docs/source/compiler/lx_coordinate_remap_benchmarks/2026-06-20/`.

- `manifest.json`
  - Checksums, provenance, and per-file intent.

## Expected Backend Behavior

Torch emits a mixed SuperDSC containing both `datadscs_` and DL `dscs_` rows.
The data-op row carries exact source and destination LX byte ranges. Deeptools
must import the op, lower each range to L3 ring send/receive movement, and
respect `coreIdToDscSchedule` so the remap rows execute before the consumer DL
row.

The v1 contract is intentionally narrow:

- whole-stick LX movement;
- byte-addressed source and destination ranges;
- explicit source and destination core IDs;
- range-encoded movement lists;
- no reductions, fan-out materialization, or layout-changing PT/LX restickify.

## Apply Check

From a clean Deeptools checkout:

```bash
git apply --check /path/to/deeptools_lx_coordinate_remap_ranged_83f9320.patch
```

If that fails on current main, regenerate a canonical patch from the latest
working Deeptools worktree and replace the lean candidate here. The patch record
must always include the exact base SHA, patch SHA if one exists, and a passing
`git apply --check` result.

## Known Caveat

During audit, one older artifact referenced a
`generatePcfgIRForLXCoordinateRemapOp` helper while the concrete ranged lowering
appears in generic data-op PCFG generation. Before upstreaming Deeptools, verify
there is one compiled lowering path and no stale call to a missing helper.
