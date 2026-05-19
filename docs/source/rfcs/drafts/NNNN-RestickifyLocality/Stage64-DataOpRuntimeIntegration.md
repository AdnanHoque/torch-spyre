# Stage 64: Data-Op Runtime Integration Probe

## Summary

Stage 64 tested whether the scheduled two-step data-op restickify contract from
Stage 63 can move beyond JSON/PCFG inspection into the installed Deeptools
lowering stack.

The result is a useful forward step:

- `DataOpStandalone` accepts the scheduled two-step SDSC.
- It recognizes the input as a scheduled DL+Data op path.
- It emits stitched dataflow MLIR and a merged SDSC.
- `dcc_standalone --kEmitProgIR` lowers the stitched MLIR to ProgIR.
- The stitched MLIR and ProgIR have no textual `HBM` or `L3` traffic terms and
  do contain `LXLU`/`LXSU` activity.

But this still does not prove hardware runtime execution:

- `dcg_standalone -s` and `DataOpStandalone --ddsc-s` both abort in direct
  senprog generation with `Codegen for Folded Super-DSC is not supported`.
- `dxp_standalone` is still not a viable raw-dataOp route because generic DXP
  SDSC import rejects `dataOpdscs_` with `Datadsc not allowed, use dldsc`.

So the data-op route is now proven through stitched Dataflow IR and ProgIR, but
not yet through an executable Torch-Spyre/Flex runtime bundle.

## Input Artifact

The probe regenerated the Stage 63 scheduled two-step data-op SDSC in the pod:

```sh
SPYRE_RESTICKIFY_LX_DATAOP=1 \
python3 tools/restickify_lx_dataop_probe.py \
  --size 2048 \
  --num-cores 32 \
  --two-step-lx-restickify \
  --mode stage3b \
  --output-dir /tmp/stage64-two-step-scheduled-tool
```

Input:

```text
/tmp/stage64-two-step-scheduled-tool/sdsc_stage3b_TwoStepReStickifyLxStcdp_2048.json
```

## Deeptools Paths Checked

### DataOpStandalone

Command:

```sh
DataOpStandalone \
  --ddsc-init-sdsc=/tmp/stage64-two-step-scheduled-tool/sdsc_stage3b_TwoStepReStickifyLxStcdp_2048.json \
  --ddsc-out-dir=/tmp/stage64-dataop-standalone/out \
  --ddsc-pcfg-verbose=1
```

Result: pass.

Generated files:

| File | Size |
|---|---:|
| `dataOp_pcfgtodf.mlir` | 231597 bytes |
| `dataOp_out.mlir` | 253127 bytes |
| `sdsc_pre.json` | 972851 bytes |
| `sdsc.json` | 1847122 bytes |
| `pcfg.json` | 80 bytes |

Important stdout evidence:

```text
Running DCG for DL+Data Op: Node-name:0_TwoStepReStickifyLxStcdp_stage3b_dataop
Computing Re-StickifyOp transfer function..
Creating pcfg for coreID:0 : LX : PE0 ...
...
Creating pcfg for coreID:31 : LX : PE0 ...
Computing transfer function metaData..
0 --> [ 0 ]
...
31 --> [ 31 ]
maxConsumers: 1
=== Number of modules :2  ===
0: [0,1,]
...
31: [0,1,]
```

This is the first probe where the scheduled two-step restickify contract is
accepted as a real scheduled multi-module data-op payload, not just as a JSON
shape.

### Stitched MLIR Traffic Shape

`dataOp_pcfgtodf.mlir`:

| Term | Count |
|---|---:|
| `HBM` | 0 |
| `L3` | 0 |
| `LXLU` | 512 |
| `LXSU` | 448 |
| `LX` | 1792 |

`dataOp_out.mlir`:

| Term | Count |
|---|---:|
| `HBM` | 0 |
| `L3` | 0 |
| `LXLU` | 512 |
| `LXSU` | 448 |
| `LX` | 1792 |

This matches the intended LX-resident movement shape. It is stronger than the
earlier DDL bridge because the explicit producer-side LX read path is present in
the stitched module.

### DCC ProgIR

Command:

```sh
dcc_standalone \
  --kEmitProgIR \
  -o /tmp/stage64-dcc-progir2/progir.mlir \
  /tmp/stage64-dataop-standalone/out/dataOp_out.mlir
```

Result: pass.

Generated file:

```text
/tmp/stage64-dcc-progir2/progir.mlir
```

Size:

```text
1172534 bytes
```

Traffic-shape scan:

| Term | Count |
|---|---:|
| `HBM` | 0 |
| `L3` | 0 |
| `LXLU` | 288 |
| `LXSU` | 2304 |
| `sentient.load_and_send` | 64 |
| `sentient.receive_and_store` | 2080 |
| `sentient.sync` | 128 |

This proves the stitched data-op MLIR can be lowered to ProgIR without producing
an HBM/L3-shaped program.

## Blocked Paths

### dcg_standalone without senprog

Command:

```sh
dcg_standalone \
  -initSdsc /tmp/stage64-two-step-scheduled-tool/sdsc_stage3b_TwoStepReStickifyLxStcdp_2048.json \
  -d /tmp/stage64-dcg-nos/out
```

Result: pass, but only PCFG/SDSC artifacts are useful.

Generated files:

| File | Size |
|---|---:|
| `out/pcfg.json` | 830685 bytes |
| `out/sdsc.json` | 125905 bytes |
| `dataDSC/senprog.txt` | 0 bytes |
| `dataDSC/senprog.txt_ir` | 0 bytes |
| `dataDSC/smc.txt` | 0 bytes |

### Direct senprog generation

Commands:

```sh
dcg_standalone \
  -initSdsc /tmp/stage64-two-step-scheduled-tool/sdsc_stage3b_TwoStepReStickifyLxStcdp_2048.json \
  -d /tmp/stage64-dcg-s/out \
  -s
```

```sh
DataOpStandalone \
  --ddsc-init-sdsc=/tmp/stage64-two-step-scheduled-tool/sdsc_stage3b_TwoStepReStickifyLxStcdp_2048.json \
  --ddsc-out-dir=/tmp/stage64-dataop-s/out \
  --ddsc-s
```

Result: both abort.

Error:

```text
DtException: Codegen for Folded Super-DSC is not supported,
file /project_src/deeptools/dcg/dcg_manager/dcg_manager.cpp line 423
```

Interpretation: the legacy DCG senprog path is not the right execution path for
this scheduled multi-dataop folded SDSC. The DCC path can lower it to ProgIR;
the missing step is packaging/executing that DCC-produced program through the
runtime-facing path.

### DXP

The generic DXP import path still rejects raw data-op SDSCs:

```cpp
DT_CHECK_MSG(mySdsc->dataOpdscs_.empty(), "Datadsc not allowed, use dldsc");
```

Source:

```text
upstream-src/deeptools/dxp/SdscTree.cpp
```

So DXP bundle execution probably requires either:

1. a DLDSc representation, or
2. a higher-level Deeprt/export path that knows how to carry data-op PCFG/ProgIR
   payloads instead of asking DXP to import raw `dataOpdscs_`.

## Validation

The focused data-op tests still pass in the pod:

```text
python3 -m py_compile tools/restickify_lx_dataop_probe.py \
  torch_spyre/_inductor/codegen/restickify_lx_dataop.py

python3 -m pytest tests/inductor/test_restickify_lx_dataop.py -q
.....                                                                    [100%]
5 passed in 7.15s
```

## Conclusion

Stage 64 moves the LX-to-LX restickify prototype one step closer to reality:

- Stage 63 proved the scheduled two-step data-op JSON contract.
- Stage 64 proves that contract is accepted by Deeptools' data-op lowering path
  and can reach stitched Dataflow IR plus ProgIR without HBM/L3 traffic terms.

It does not yet prove runtime execution or correctness on hardware.

The next blocker is no longer "can Deeptools express the data movement?" It can.
The blocker is "how does Torch-Spyre hand this DCC-produced data-op program to
the runtime in a supported executable form?"

## Recommended Next Step

Follow the Deeprt export path rather than the DXP raw SDSC path.

Relevant source facts:

- `deeprt_scheduler_codegen_pipeline.cpp` has a pure data-op scheduler path.
- `deeprt.cpp` selects DCG/DCC behavior for pure data-op and DL+data-op cases.
- `export.cpp` exports programs for data-op SDSCs when program content exists.
- `dxp/SdscTree.cpp` rejects raw `dataOpdscs_`.

The next probe should construct the smallest SenGraph or Torch-Spyre compile
fixture that routes this scheduled data-op SDSC through Deeprt/export, not
through direct DXP import. If that is too large, the intermediate fallback is to
find the runtime container format that consumes the `dcc_standalone
--kEmitProgIR` output and wrap the generated ProgIR there.
