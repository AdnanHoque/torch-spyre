# Stage 076: DCC Same-Step Stitch Region Probe

Date: 2026-05-28

## Purpose

Stage075 narrowed the remaining unsafe shape to a paired schedule row:

```text
[loader STCDPOpHBM, current DL compute]
```

The DCC stitcher represents mixed dataDSC plus DLDSC schedules by cloning
non-DL program units into uniformized regions around the DLDSC program unit.
For a same-step dataDSC plus DLDSC slot, the current stitcher places the
non-DL unit in the `after_dldsc` region.

Stage076 tests whether the corruption is caused by that after-region placement.

## Lower-Stack Probe

Added a default-off DCC diagnostic switch on the pod:

```text
DT_DCC_STITCH_SAME_STEP_DATADSC_BEFORE_DLDSC=1
```

When enabled, `dcc/src/Stitcher/ModuleStitcher.cpp` keeps same-step non-DL
program units in the `before_dldsc` uniform region instead of moving them to
`after_dldsc`.  The generated SDSC schedule is unchanged; only the stitch
region chosen for a same-step non-DL unit changes.

The lower stack was rebuilt with:

```text
make -j16 dcc dxp
```

## Device Results

Full-tile loader fanout, no after-sync, before-DL stitch:

```text
variant = onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_no_after_sync_probe
env = DT_DCC_STITCH_SAME_STEP_DATADSC_BEFORE_DLDSC=1

L=128:
cache = /tmp/sdpa-stage169-stitch-before-fulltile-overlap-no-after-sync-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_no_after_sync_probe-B1-H8-L128-D64-C0-778333-434428
status = failed
Mismatched elements: 15988 / 65536 (24.4%)
Greatest absolute difference: 0.75 at index (0, 7, 94, 53)

L=256:
cache = /tmp/sdpa-stage169-stitch-before-fulltile-overlap-no-after-sync-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_no_after_sync_probe-B1-H8-L256-D64-C0-778333-948557
status = failed
Mismatched elements: 5205 / 131072 (4.0%)
Greatest absolute difference: 0.74658203125 at index (0, 1, 242, 11)
```

Full-tile loader fanout, after-sync enabled, before-DL stitch:

```text
variant = onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_probe
env = DT_DCC_STITCH_SAME_STEP_DATADSC_BEFORE_DLDSC=1
cache = /tmp/sdpa-stage170-stitch-before-fulltile-overlap-after-sync-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_probe-B1-H8-L256-D64-C0-778824-835469
status = failed
Mismatched elements: 5205 / 131072 (4.0%)
Greatest absolute difference: 0.74658203125 at index (0, 1, 242, 11)
```

Direct loader copyback without fanout, before-DL stitch:

```text
variant = onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_tail_probe
env =
  DT_DCC_STITCH_SAME_STEP_DATADSC_BEFORE_DLDSC=1
  SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_TAIL_CURRENT=0
cache = /tmp/sdpa-stage171-stitch-before-direct-copyback-overlap-onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_direct_copyback_tail_probe-B1-H8-L256-D64-C0-779131-393199
status = failed
Mismatched elements: 5205 / 131072 (4.0%)
Greatest absolute difference: 0.74658203125 at index (0, 1, 242, 11)
```

## Interpretation

The before-DL stitch switch is active: the generated SDSC schedules stay the
same, while the emitted SMC order changes.  However, moving same-step dataDSC
units to the `before_dldsc` region does not make the overlap path correct.

The identical L=256 result for full fanout and direct copyback means this
failure is upstream of the fanout row.  It is not an STCDP-LX fanout issue.
It also does not depend on disabling the paired-row after-sync bit.

This rules out a simple lower-stack stitch-region flip as the missing
warp-specialized contract.  The remaining path still needs one of:

- a genuinely independent loader lane/core/corelet contract that does not
  perturb the current DL program unit,
- a lower-stack schedule representation that can express concurrent loader
  work without cloning it into the DL unit's before/after uniform regions, or
- a new prefetch primitive with explicit resource semantics for loader HBM to
  LX movement beside attention compute.

