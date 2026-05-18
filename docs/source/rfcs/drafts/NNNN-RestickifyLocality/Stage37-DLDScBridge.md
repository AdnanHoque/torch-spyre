# Stage 37: DLDSc Bridge Spike

## Summary

Stage 36 ended at a concrete production-path blocker: the normal DXP bundle path
rejects a standalone `ReStickifyOpLx` data-op SDSC with:

```text
DtException: Datadsc not allowed, use dldsc
file /project_src/deeptools/dxp/dxp.cpp line 489
```

Stage 37 tested the smallest possible bridge from the working data-op prototype
to the DLDSc-shaped artifact DXP expects.

The result is mixed but useful:

- Deeptools can lower the synthetic `ReStickifyOpLx` data-op SDSC to
  Dataflow IR, Sentient IR, and ProgIR with `dcc_standalone`.
- The DXP bundle path can accept a `ReStickifyOpLx` op when it is represented as
  a normal `dscs_` unit and referenced by `bundle.mlir`.
- A trivial JSON rewrite is not enough to make that `dscs_` unit LX-local. The
  inherited DLDSc schedule still uses HBM, and changing allocation nodes from
  HBM to LX is rejected by the scheduler.

So the next bridge is not a one-line key rename from `datadscs_` to `dscs_`.
It requires generating a real DLDSc/DDL-backed restickify schedule, or changing
the DXP bundle path to accept the DCC-produced data-op module.

## Installed Deeptools Facts

The installed Deeptools headers expose both normal DLDSc scheduling and data-op
DSCs:

```text
/opt/ibm/spyre/deeptools/include/dsc/superdsc.h
  int datadsc_idx = -1;
  int dldsc_idx = -1;
  std::vector<DesignSpaceConfig> dscs_;
  std::vector<DataOpDsc> dataOpdscs_;

/opt/ibm/spyre/deeptools/include/dsc/designSpaceConfig.h
  bool dcc_dump_input_dldsc = false;
  bool dcc_dump_input_datadscs = false;

/opt/ibm/spyre/deeptools/include/dsc/dataOpDsc.h
  struct ReStickifyOpLx : APEOpLX {};
  struct ReStickifyOpHBM : APEOpHBM {};
```

The installed system config also lists both restickify op names:

```text
/opt/ibm/spyre/deeptools/share/dsc/HardwareArchMapping/sysConfigs2.0/sentient_dd2_sysconfig.json
  ReStickifyOpLx
  ReStickifyOpHBM
```

And the installed DDL template has explicit LX restickify bindings:

```text
/opt/ibm/spyre/deeptools/share/ddc/ddl_templates/restickify_sen1p5.ddl
  ddl.operation_bind(...) {opFuncName="ReStickifyOpHBM"}
  ddl.operation_bind(...) {opFuncName="ReStickifyOpLx"}
  ddl.get_external_data_transfer_allocation(...) {memory="lx", data_connect="lxlu_input"}
  ddl.get_external_data_transfer_allocation(...) {memory="lx", data_connect="lxsu_input"}
```

This says `ReStickifyOpLx` is a known Deeptools concept. The missing piece is
how Torch-Spyre should hand that concept to DXP in the normal bundle path.

## Probe 1: Direct DDL/DDC On Current ReStickify SDSC

Using a real generated `ReStickifyOpHBM` SDSC:

```text
/tmp/torchinductor_1000800000/tmp7tbibx5t/inductor-spyre/sdsc_fused_add_t_0_tsglekht/sdsc_0_ReStickifyOpHBM.json
```

Direct DDL mapping did not work:

```sh
ddl_standalone \
  -s input_sdsc.json \
  -d /opt/ibm/spyre/deeptools/share/ddc/ddl_templates/restickify_sen1p5.ddl
```

Result:

```text
[DDC] DDL found but not suitable for op ReStickifyOpHBM
Ddl mapping failed
```

`ddc_standalone -s input_sdsc.json` and
`dsc_standalone -f input_sdsc.json --upgradeDSC` both aborted with:

```text
std::out_of_range
what(): map::at
```

Interpretation: the existing Torch-Spyre restickify SDSC is already a lowered
`dscs_` form. It is not the high-level DDL input shape the restickify DDL
template expects.

## Probe 2: Data-Op SDSC Through DCC

The Stage 36 synthetic artifact is:

```text
/tmp/restickify-stage36-synthetic-bundle/sdsc_0_ReStickifyOpLx_dataop.json
```

It contains a populated `datadscs_` list and no normal `dscs_` work:

```text
datadscs_: [0_ReStickifyOpLx_dataop]
dscs_: []
coreIdToDscSchedule: {}
```

The DXP bundle path rejects this directly, but DCC can lower it:

```sh
dcc_standalone --input-mode=sdsc --kEmitDataflowIR input.json
dcc_standalone --input-mode=sdsc --kEmitSentientIR input.json
dcc_standalone --input-mode=sdsc --kEmitProgIR input.json
```

The Dataflow IR contains LX-local units and explicit send/receive movement:

```text
dataflow.get_unit {name = "lx", ... type = "lx"}
dataflow.get_logical_memory_view ... memref<64xf16>
agen.vector_load ...
dataflow.send ...
dataflow.receive ...
agen.vector_store ...
```

The ProgIR path also succeeds and names LX load/store units:

```text
dataflow.get_unit {name = "lxlu0", type = "lxlu"}
dataflow.get_unit {name = "lxsu0", type = "lxsu"}
dataflow.get_unit {name = "pe0", type = "pe"}
sentient.load_and_send ...
sentient.receive_and_store ...
sentient.vector_binary ... opA = lx
```

Interpretation: the `ReStickifyOpLx` data-op contract is not bogus. Deeptools
can translate it into executable-looking lower IR. The blocker is that DXP's
bundle frontend does not accept raw `datadscs_` as a bundle unit.

## Probe 3: JSON Key Compatibility

Three small rewrites were tested against the synthetic data-op bundle:

- rename `datadscs_` to `dataOpdscs_`
- add an explicit data-op schedule entry `[[0, -1, 0, 0]]`
- do both

Results:

```text
rename only:
  DtException: No dsc in sdsc input

schedule only:
  std::out_of_range: map::at

rename and schedule:
  std::out_of_range: map::at
```

Interpretation: DXP is not simply looking for the newer serialized field name.
For bundle execution it wants a normal `dscs_`/`dldsc_idx` path.

## Probe 4: Minimal `dscs_` ReStickifyOpLx

A normal generated `ReStickifyOpHBM` SDSC has the shape DXP accepts:

```text
dscs_: [{ "ReStickifyOpHBM": ... }]
coreIdToDscSchedule:
  "0": [[-1, 0, 0, 0]]
```

The `[-1, 0, 0, 0]` schedule points at `dldsc_idx=0`, not a data-op index.

As a smoke test, the SDSC was rewritten to:

- rename the DLDSc key from `ReStickifyOpHBM` to `ReStickifyOpLx`
- change `computeOp_[0].opFuncName` to `ReStickifyOpLx`
- keep the existing allocation schedule intact
- add a matching `bundle.mlir`

DXP accepted the bundle and emitted normal bundle artifacts:

```text
execute_dsg.txt
loadmodel_to_device_dsg.txt
loadprogram_to_device_dsg.txt
segment_size.json
```

But the DCC Dataflow IR still contained HBM/L3 units:

```text
name = "hbm"
name = "l3ibr"
name = "l3lu"
name = "l3su"
name = "lx"
```

Interpretation: a `ReStickifyOpLx` name can travel through the DLDSc bundle
path, but the inherited DLDSc schedule is still the HBM restickify schedule.
That is not the locality-preserving implementation we want.

## Probe 5: Naive LX Allocation Rewrite

The next attempt changed the normal DLDSc allocation nodes from HBM to LX:

```text
allocate-Tensor0_hbm -> allocate-Tensor0_lx
allocate-Tensor1_hbm -> allocate-Tensor1_lx
component_: hbm -> lx
```

DCC still printed HBM/L3 units, and DXP rejected the bundle:

```text
DtException: Expect a valid HBM allocate node.
file /project_src/deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 5864
```

Interpretation: the current lowered DLDSc schedule is structurally an HBM/L3
restickify schedule. Making it LX-local requires a different schedule, not a
post-hoc allocation rename.

## Conclusion

Stage 37 narrows the integration problem:

1. `ReStickifyOpLx` as a Deeptools data-op can lower through DCC to
   Dataflow/Sentient/ProgIR.
2. DXP bundles reject raw `datadscs_`.
3. DXP bundles accept `ReStickifyOpLx` only when it is represented as a normal
   `dscs_`/DLDSc unit.
4. The existing Torch-Spyre-generated DLDSc restickify schedule is HBM-shaped.
   It cannot be made LX-local by renaming the op or allocation components.

So the bridge we need is one of:

- Generate the correct DDL/DLDSc input shape for `restickify_sen1p5.ddl`, so
  Deeptools produces an LX-local DLDSc schedule.
- Add a Deeptools/Torch-Spyre bundle integration path that lets DXP consume the
  DCC-lowered data-op module for `ReStickifyOpLx`.
- Extend the existing SuperDSC/DLDSc generator with a first-class
  `ReStickifyOpLx` schedule instead of reusing the HBM restickify schedule.

The smallest likely production path is the third option if Torch-Spyre owns the
restickify SDSC generator, but the DDL template is valuable evidence for what
the schedule must express: LX input allocation, LX output allocation, and
LX/PE/PT-local movement rather than L3/HBM movement.

## Next Step

The next stage should stop treating `datadscs_` as the final artifact. Instead,
build a minimal DLDSc/LX schedule generator for a single-core synthetic
`ReStickifyOpLx` and compare its DCC output to the known-good data-op DCC output.

Acceptance for that next stage:

- `dcc_standalone --input-mode=sdsc --kEmitProgIR` shows `lxlu0`, `lxsu0`, and
  no `hbm`/`l3lu`/`l3su` units for the synthetic restickify.
- `dxp_standalone --bundle` accepts the synthetic bundle.
- Only after that should we reconnect it to the real Stage 3B
  `adds_then_matmul` case.

