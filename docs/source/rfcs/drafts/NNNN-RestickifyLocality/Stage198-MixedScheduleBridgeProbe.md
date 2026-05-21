# Stage 198: Mixed Schedule Bridge Probe

## Summary

This stage moved one step past the failed Stage 197 "replace the restickify
SDSC with a top-level `datadscs_` bridge" attempt.

The new probe emits a single mixed SuperDsc containing:

- the real Torch-Spyre consumer DL DSC under `dscs_`;
- the PT-aware LX bridge data ops under `datadscs_`;
- an explicit `coreIdToDscSchedule`.

This is still default-off diagnostic work. It does not change production
lowering or launch hardware.

## Tool Added

```text
tools/restickify_mixed_schedule_probe.py
```

The tool starts from a generated Torch-Spyre bundle, finds the adjacent
`producer -> ReStickifyOpHBM -> consumer` region, and writes candidate mixed
schedule variants.

The most important variant is:

```text
bridge_then_dl:
  [0, -1, 0, 1]   # ReStickifyOpWithPTLx data op
  [1, -1, 1, 1]   # STCDPOpLx data op
  [-1, 0, 1, 0]   # consumer DL op
```

In words: perform the PT-aware LX bridge first, then run the consumer DL op
against the LX-resident input.

## Probe Command

The seed bundle was generated without hardware launch:

```sh
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --skip-kernel-launch \
  --copy-kernel-code \
  --output-dir /tmp/stage198-generate \
  --fail-on-error
```

Then the mixed schedules were generated and passed through Deeptools:

```sh
python tools/restickify_mixed_schedule_probe.py \
  --bundle-dir /tmp/stage198-generate/kernel_code/computed_transpose_adds_then_matmul_tuple_2048/0001_sdsc_fused_add_t_0 \
  --output-dir /tmp/stage198-mixed-schedule \
  --run-deeptools
```

## Results

| Variant | DCC | DXP | HBM in DCC IR | Work ops | Interpretation |
|---|---:|---:|---|---:|---|
| `bridge_then_dl` | 0 | -6 | no | 822 | DCC accepts the mixed DL+data-op schedule and emits real work. |
| `bridge_then_paired_dl` | -6 | -6 | no | 0 | The paired input-neighbor spelling does not match this two-step PT bridge. |
| `paired_only` | -6 | -6 | no | 0 | Pure paired input-neighbor is not enough for the PT-aware restickify case. |
| `paired_then_dl` | -6 | -6 | no | 0 | Also rejected by DCC. |

For the successful `bridge_then_dl` variant, the DCC output unit counts were:

```text
l0lu0=32, l0su0=32,
l3lu=993, l3su=932,
lxlu0=32, lxsu0=32,
pe0=32,
ptrow0_0..ptrow7_0=32 each,
pt_slice_mask_arf_write=8,
sfp0=32
```

The `L3LU/L3SU` units are expected for ring-facing transfer work. The important
property is that this diagnostic DCC IR did not mention HBM while still
including the PT/SFP/LX units needed by the value-correct Stage 195 bridge.

DXP still rejects every mixed variant before codegen:

```text
DtException: Datadsc not allowed, use dldsc, file /project_src/deeptools/dxp/SdscTree.cpp line 152
```

## Interpretation

This narrows the normal-lowering path:

1. A simple DLDSc JSON rewrite is not enough. Stage 38 already showed it emits
   empty or non-useful L3 programs.
2. A top-level data-op SDSC is not enough. Stage 197 showed DXP rejects it in
   normal bundle import.
3. A mixed `datadscs_ + dscs_ + coreIdToDscSchedule` SuperDsc is the closest
   production-shaped artifact found so far. DCC lowers the best schedule shape.
4. The remaining blocker is the DXP bundle path, which currently rejects
   imported `datadscs_` and routes imported bundle SDSCs through DL-only
   codegen instead of the existing `runDcgForDataOpsDlOps` path.

## Next Engineering Step

The smallest Deeptools-side experiment is:

- allow bundle-imported SuperDscs with both `dscs_` and `datadscs_`;
- in `Dxp::runCodegen`, if `coreIdToDscSchedule` is populated, call
  `DcgManager::runDcgForDataOpsDlOps` instead of
  `runDcgForDlOpsStandalone`;
- keep the existing pure-DL and pure-data-op behavior unchanged.

On the Torch-Spyre side, the corresponding production-shaped lowering is:

- generate `bridge_then_dl` as one mixed SuperDsc for certified eligible
  in-graph restickify edges;
- remove the separate `ReStickifyOpHBM` bundle frame;
- make the consumer input LX-resident at the bridge output address;
- keep all of this behind default-off prototype flags until DXP can compile and
  launch the mixed artifact.

## Artifacts

Local copied artifacts:

```text
artifacts/stage198_mixed_schedule/summary.json
artifacts/stage198_mixed_schedule/bridge_then_dl/
```

Pod artifacts:

```text
/tmp/stage198-generate
/tmp/stage198-mixed-schedule
```
