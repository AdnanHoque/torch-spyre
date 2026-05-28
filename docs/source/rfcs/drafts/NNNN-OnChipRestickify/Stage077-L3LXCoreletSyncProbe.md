# Stage077 - L3/LX Corelet Sync Probe

## Question

Can the failing same-row K/V HBM prefetch overlap be fixed by preserving the
loader DataOp's corelet-1 intent in the generated L3/LX synchronization?

## Baseline

Stage175 re-ran the full-tile overlap probe with L3 corelet preservation
disabled and IR dumps enabled:

```text
variant: onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_probe
shape: B1 H8 L128 D64
status: failed
mismatched: 482 / 65536
max abs: 0.450439453125 at (0, 0, 2, 11)
max rel: 75.625 at (0, 1, 2, 32)
```

The DataOp still carried `coreletId : 1`, but the emitted current-prefetch IR
only synchronized against `lxlu0`.

## Lower-Stack Probe

Added a default-off lower-stack diagnostic gated by:

```text
DT_DSC_L3_LX_SYNC_BOTH_CORELETS=1
```

The first broad version widened all generic LXLU/LXSU syncs and caused an
early DXP abort in a normal batchmatmul SDSC. The probe was narrowed to:

- only SDSCs with a DataOp whose op has explicit `coreletId == 1`
- only the scheduler-created L3LU/LXLU sync node names
- only the LXLU side of those syncs

Files touched on the pod lower-stack tree:

- `dsc/designSpaceConfig.h`
- `dsc/dsc2.cpp`
- `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp`

DCC/DXP rebuilt successfully.

## Result

Stage178 used the narrowed DataOp-corelet-aware probe:

```text
variant: onchip_hbm_kv_layout_xform_kv_hbm_prefetch_loader_fanout_fulltile_overlap_probe
shape: B1 H8 L128 D64
env:
  SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_PREFETCH_CORELET1=1
  DT_DSC_L3_LX_SYNC_BOTH_CORELETS=1
  CODEGEN_DUMP_IRS=1
status: failed
mismatched: 482 / 65536
max abs: 0.450439453125 at (0, 0, 2, 11)
max rel: 75.625 at (0, 1, 2, 32)
```

The IR changed as intended:

```text
dataflow.get_unit {core = 0, corelet = 1, name = "lxlu1", type = "lxlu"}
dataflow.sync_send ... "c0-l3lu-sync-send-lxlu0-lxlu1"
sentient.sync ... units = [lxlu1, lxlu0]
```

But the final numeric mismatch was identical to the baseline.

## Interpretation

This rules out the narrow theory that the bug is only missing `lxlu1` sync.
The generated `senprog.txt` shows that corelet-1 LXLU programs are sync-only.
The real LXLU load/store program remains on corelet 0, and the L3LU HBM->LX
load/store path still writes the same LX-side data path used by the current
consumer.

So the remaining same-row overlap failure is more likely one of:

- LX address/buffer aliasing between the prefetch tile and current tile
- STCDP/DataOp corelet intent not being applied to the actual data movement
  resource, only to metadata/sync
- shared L3LU/LX memory-path ordering that cannot be fixed by adding sync
  participants alone

## Next Probe

The next useful lower-stack probe is not more sync widening. It should either:

- force the prefetch HBM->LX destination address range away from the current
  consumer range and verify the emitted LBR/LAR constants, or
- push `coreletId == 1` into the actual STCDP/LX data movement path instead of
  only adding a sync-only LXLU1 program.

