# Stage 199: Mixed Schedule Torch-Spyre Lowering

## Summary

This stage moved the Stage198 mixed-schedule probe into Torch-Spyre bundle
generation behind a new default-off flag:

```text
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
```

For eligible adjacent:

```text
producer -> ReStickifyOpHBM -> consumer
```

the prototype now emits:

```text
producer SDSC
mixed bridge+consumer SDSC
```

instead of:

```text
producer SDSC
ReStickifyOpHBM SDSC
consumer SDSC
```

The mixed SDSC contains:

- the `ReStickifyOpWithPTLx` data op;
- the `STCDPOpLx` data op;
- the original consumer DL DSC;
- the Stage198 `bridge_then_dl` `coreIdToDscSchedule`.

This is still not a runnable production path until Deeptools/DXP accepts mixed
SuperDscs in normal bundle import.

## Code Changes

- Added config flag:

  ```text
  SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
  ```

- Added `patch_restickify_ptlx_mixed_schedules(...)`.
- Updated bundle emission so the consumed standalone consumer SDSC is omitted
  when it has been folded into the mixed SDSC.
- Added focused unit coverage for the mixed schedule shape.

## Validation

Focused unit/static validation in the pod:

```text
python3 -m py_compile \
  torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py \
  torch_spyre/_inductor/codegen/bundle.py \
  tools/restickify_mixed_schedule_probe.py

python3 -m pytest tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
6 passed
```

## 2048 Codegen Probe

Command shape:

```sh
SPYRE_RESTICKIFY_RING_TELEMETRY=1 \
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 \
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1 \
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_BRIDGE_AUDIT_JSONL=/tmp/stage199-mixed-lowering/audit.jsonl \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --skip-kernel-launch \
  --copy-kernel-code \
  --output-dir /tmp/stage199-mixed-lowering \
  --fail-on-error
```

The Torch-Spyre patch fired:

```json
{
  "status": "patched",
  "kind": "ptlx-mixed-schedule",
  "replacement_sdsc": "1_MixedReStickifyOpWithPTLxConsumer",
  "consumer_index_omitted": 2,
  "mixed_schedule": [[0, -1, 0, 1], [1, -1, 1, 1], [-1, 0, 1, 0]],
  "size": 2048,
  "num_cores": 32,
  "direction": "kernel-to-output"
}
```

The generated bundle now contains only two SDSC executions:

```mlir
module {
  func.func @sdsc_bundle() {
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_0_add.json"}
    sdscbundle.sdsc_execute () {sdsc_filename="sdsc_1_MixedReStickifyOpWithPTLxConsumer.json"}
    return
  }
}
```

The mixed SDSC has:

```text
dscs_ = 1
datadscs_ = 2
core0 schedule = [[0, -1, 0, 1], [1, -1, 1, 1], [-1, 0, 1, 0]]
data ops = ReStickifyOpWithPTLx, STCDPOpLx
```

## Deeptools Result

As expected, normal DXP bundle compilation still fails at import:

```text
DtException: Datadsc not allowed, use dldsc, file /project_src/deeptools/dxp/SdscTree.cpp line 152
```

But standalone DCC accepts the Torch-Spyre-generated mixed SDSC:

```text
DCC_RC=0
has_hbm=False
```

Unit counts from the emitted DCC IR:

```text
L3LU=993, L3SU=932,
LXLU=32, LXSU=32,
PE=32,
PT rows present,
SFP=32
```

This reproduces the Stage198 result, now from the normal Torch-Spyre lowering
hook rather than from an offline handcrafted probe.

## Conclusion

The Torch-Spyre side can now emit the mixed-schedule shape we want. The next
blocker is no longer "how should Torch-Spyre represent this edge?" It is:

```text
make DXP bundle import/codegen accept mixed SuperDscs that contain both
dscs_ and datadscs_ with a populated coreIdToDscSchedule.
```

The smallest Deeptools experiment remains:

1. allow imported bundle SDSCs with both `dscs_` and `datadscs_`;
2. in `Dxp::runCodegen`, route scheduled mixed SDSCs through
   `DcgManager::runDcgForDataOpsDlOps`;
3. keep pure-DL and pure-data-op behavior unchanged.

## Artifacts

Local copied artifacts:

```text
artifacts/stage199_mixed_lowering/
```

Pod artifacts:

```text
/tmp/stage199-mixed-lowering
/tmp/stage199-dcc-check
/tmp/torchinductor_1000800000/tmp7snvp9fy
```
