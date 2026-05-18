# Stage 39: Restickify DDL Contract Discovery

## Summary

Stage 38 showed that simple post-lowering JSON rewrites are not enough to get a
real LX-local restickify through the normal DXP bundle path. Stage 39 moved one
level earlier in Deeptools and checked the DDC/DDL contract.

The result is useful: Deeptools already has a restickify DDL path that expands a
small restickify SDSC into a non-empty LX/SFP/PT schedule. The raw input lowers
to empty L3 program units, but the DDC-produced `input.out.json` lowers through
DCC to real work.

The remaining blocker is DXP bundle integration. DXP still aborts on the
DDC-produced fixture with a corelet-split assertion:

```text
DtException: data_stage_params.size() == 1, file /project_src/deeptools/dsm/SdscCoreletSplit.cpp line 70
```

So this stage does not prove an end-to-end executable LX restickify yet. It does
prove that the right contract is DDC/DDL-generated DLDSc, not direct mutation of
Torch-Spyre's already-lowered `ReStickifyOpHBM` JSON.

## Code Change

Added a diagnostic tool:

```text
tools/restickify_ddl_contract_probe.py
```

The tool:

- normalizes a Deeptools SDSC fixture by stripping `//` test-run comments
- summarizes the original SDSC schedule
- optionally runs `ddc_standalone -s <input> -d`
- summarizes the DDC-produced `<input>.out.json`
- runs DCC on both the input and DDC output
- tries the DXP bundle path on the DDC output and records the result

This keeps the DDC/DDL finding reproducible instead of relying on one-off shell
commands in `/tmp`.

## Deeptools Source Findings

The direct data-op route still cannot be passed to DXP as a normal bundle:

```text
dxp/SdscTree.cpp:152
DT_CHECK_MSG(mySdsc->dataOpdscs_.empty(), "Datadsc not allowed, use dldsc");
```

The DDC path is the one that selects the restickify DDL template:

```text
ddc/ddcv1.cpp
prepDsc();
attachToPrefilledSchedule();
DdlConvertInterface ddlConvTnterface(...);
bool parseDdl = ddlConvTnterface.selectAndParseDdlTemplate();
```

The template map binds both restickify names to `restickify_sen1p5.ddl` for the
SEN1P5 ISA path:

```text
ddc/ddl/ddl_conversion.h
OpFuncs::ReStickifyOpLx  -> restickify_sen1p5.ddl
OpFuncs::ReStickifyOpHBM -> restickify_sen1p5.ddl
```

There is also explicit restickify handling in DDC allocation/spread logic:

```text
ddc/ddcv1.cpp
if ((computeOp.opFuncName == OpFuncs::ReStickifyOpLx ||
     computeOp.opFuncName == OpFuncs::ReStickifyOpHBM) &&
    dscGlobal.sysDef.coreArch == SEN1P5_ISA)
  is_restickify_sen1p5 = true;
```

## Fixture

The key fixture is in Deeptools:

```text
ddc/ddl_templates/test/sdsc_restickify.json
```

Its test line uses DDC directly:

```text
SENARCH=rcudd1a ddc_standalone -s <fixture>.json -d
```

The fixture is a compact pre-DDC restickify SDSC:

- op: `ReStickifyOpHBM`
- input allocation: `lx`
- output allocation: `lx`
- schedule nodes: `10`
- data-stage params: `2`
- labeled datasets: `2`

The important point is that this fixture is not the same shape as the
Torch-Spyre-generated post-lowered `ReStickifyOpHBM` SDSCs. It is small enough
for DDC to attach the DDL template and synthesize the real schedule.

## Probe Command

On the pod:

```sh
python3 /tmp/restickify_ddl_contract_probe.py \
  --sdsc /tmp/stage39-ddc-fixture/sdsc_restickify.json \
  --output-dir /tmp/stage39-ddl-contract-probe \
  --deeptools-bin /opt/ibm/spyre/deeptools/bin \
  --senarch rcudd1a \
  --run-deeptools
```

The summary artifact was copied locally to:

```text
artifacts/stage39_ddl_contract_probe/summary.json
```

## Results

| Artifact | DCC | DXP | Units | Work ops | Interpretation |
|---|---:|---:|---|---:|---|
| raw fixture input | 0 | n/a | `l3lu:18`, `l3su:18` | 0 | The pre-DDC fixture alone is not an executable schedule. |
| DDC output | 0 | -6 | `lxlu/lxsu/sfp/pt/l0/l3` units | 38 | DDC generated real schedule work, but DXP bundle splitting aborts. |

DDC grew the schedule from a small structural input into a full schedule:

| Field | Input | DDC output |
|---|---:|---:|
| schedule nodes | 10 | 74 |
| allocate nodes | 2 | 9 |
| transfer nodes | 2 | 21 |
| compute nodes | 0 | 32 |
| labeled datasets | 2 | 5 |
| primary datasets | 2 | 3 |
| data-stage params | 2 | 6 |

The DDC output contains LX-local data-connect names:

```text
lxlu_input
lxsu_input
sfp_input
sfp_internal
sfp_output
pt_fifo
compute_out
```

And DCC lowers it into real movement/compute ops:

```text
sentient.load_and_send      count 9
sentient.receive_and_store  count 2
sentient.vector_binary      count 27
```

The DCC output still declares `l3lu/l3su` program units, so this is not yet a
clean "no L3 unit exists" proof. The meaningful change versus Stage 38 is that
the DDC output has non-empty LX/SFP/PT work, while the direct DLDSc rewrites had
only empty `l3lu/l3su` units.

## Interpretation

This narrows the engineering path:

1. Torch-Spyre should not try to produce LX restickify by renaming an already
   lowered `ReStickifyOpHBM` DLDSc.
2. The first-class schedule appears to be generated by DDC from a pre-DDC SDSC
   that matches `restickify_sen1p5.ddl`.
3. A production-quality LX restickify path likely needs either:
   - Torch-Spyre to emit the pre-DDC restickify contract and run DDC for this op,
   - a DXP-compatible bridge for the DDC-produced schedule,
   - or a direct Torch-Spyre DLDSc generator that reproduces DDC's schedule
     fields and data connects.

The next blocker is not locality theory. It is the DXP integration failure on
the DDC-produced fixture:

```text
data_stage_params.size() == 1
```

That assertion is where I would focus the next small spike. If DXP can accept
the DDC output, we can move from "DCC proves the schedule exists" to "DXP can
bundle and run it."

## Validation

Local:

```text
python3 -m py_compile tools/restickify_ddl_contract_probe.py
```

Pod:

```text
python3 /tmp/restickify_ddl_contract_probe.py ... --run-deeptools
```

The pod run produced:

```text
ddc_rc=0
dcc input rc=0, work_op_count=0
dcc DDC-output rc=0, work_op_count=38
dxp DDC-output rc=-6
```

