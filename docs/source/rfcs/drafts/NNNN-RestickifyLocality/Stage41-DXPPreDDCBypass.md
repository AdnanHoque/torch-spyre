# Stage 41: DXP Pre-DDC Bypass Prototype

## Summary

Stage 40 identified the first DXP blocker for the Deeptools restickify DDL
fixture: DXP runs generic corelet splitting before DDC, and that splitter
assumes a single `dataStageParam_`.

Stage 41 tested the actual patch direction. The first attempt, skipping only
`Dsm::doCoreletSplitSdsc`, moved the failure forward but did not compile the
bundle:

```text
[restickify-probe] skipped Dsm::doCoreletSplitSdsc via LD_PRELOAD
DtException: Invalid scheduleTree., file L3DlOpsScheduler.cpp line 7141
```

That showed a second pre-DDC blocker: DXP also runs the generic L3 scheduler
before DDC. The restickify fixture is a pre-DDC DDL input, so that scheduler
sees an incomplete schedule tree. If we bypass both generic pre-DDC passes,
DXP reaches DDC, DDC expands the restickify template, DCC runs, and DXP exits
successfully.

## Code Change

Added a reproducible validation probe:

```text
tools/restickify_dxp_preddc_shim_probe.py
```

The tool compiles a small `LD_PRELOAD` shim that no-ops:

```text
Dsm::doCoreletSplitSdsc(SuperDsc*)
L3DlOpsScheduler::run(SuperDsc&)
```

Then it runs `dxp_standalone` on the same Deeptools `sdsc_restickify.json`
fixture and summarizes the generated debug SDSCs and senprog.

This is not a production mechanism. It is a low-latency proof that the proposed
Deeptools patch direction is viable.

## Probe Command

On the pod:

```sh
python3 /tmp/restickify_dxp_preddc_shim_probe.py \
  --sdsc /tmp/stage39-ddc-fixture/sdsc_restickify.json \
  --output-dir /tmp/stage41-dxp-preddc-shim-probe \
  --deeptools-bin /opt/ibm/spyre/deeptools/bin \
  --senarch rcudd1a
```

The summary artifact was copied locally to:

```text
artifacts/stage41_dxp_preddc_shim_probe/summary.json
```

## Results

DXP succeeded with both pre-DDC passes bypassed:

```text
dxp_rc=0
```

The debug SDSC snapshots show the intended pipeline:

| Snapshot | Schedule nodes | Data-stage params | Labeled DS count | Meaning |
|---|---:|---:|---:|---|
| pre-DDC `sdsc.out.json` | 10 | 2 | 2 | Compact DDL input fixture. |
| post-DDC `sdsc.out.out.json` | 74 | 6 | 5 | DDC-expanded restickify schedule. |
| post-DIP `sdsc.out.out.out.json` | 74 | 6 | 5 | Schedule survived DIP. |

The generated `senprog.txt` is the most important sanity check:

```text
bytes: 261378
LXLU: 18
LXSU: 18
SFP: 504
PT: 1962
L3LU: 0
L3SU: 0
HBM: 0
```

So with the generic pre-DDC passes bypassed, the restickify fixture compiles to
a program that names LX/SFP/PT units and does not name L3/HBM units in the
generated senprog.

## Deeptools Prototype Patch

I also made a local Deeptools source prototype in:

```text
/tmp/deeptools-stage39
branch: AdnanHoque/prototype-restickify-skip-corelet-split
commit: 243bbd7 Prototype restickify DXP pre-DDC bypass
```

The internal Deeptools repo rejected branch push with:

```text
ERROR: Write access to repository not granted.
```

The local source patch has two parts.

First, skip generic pre-DDC corelet splitting for restickify multi-data-stage
DDL inputs:

```cpp
static bool isRestickifyMultiDataStageDsc(const DesignSpaceConfig& dsc) {
  if (dsc.computeOp_.size() != 1) return false;
  const auto& op_func = dsc.computeOp_.front().opFuncName;
  return dsc.dataStageParam_.size() != 1 &&
         (op_func == OpFuncs::ReStickifyOpLx ||
          op_func == OpFuncs::ReStickifyOpHBM);
}

...

if (isRestickifyMultiDataStageDsc(dsc)) continue;
```

Second, skip the generic L3 scheduler before DDC for the same input family:

```cpp
static bool isRestickifyMultiDataStageSdsc(const SuperDsc& sdsc) {
  if (sdsc.dscs_.empty()) return false;
  return std::all_of(sdsc.dscs_.begin(), sdsc.dscs_.end(), [](const auto& dsc) {
    if (dsc.computeOp_.size() != 1) return false;
    const auto& op_func = dsc.computeOp_.front().opFuncName;
    return dsc.dataStageParam_.size() != 1 &&
           (op_func == OpFuncs::ReStickifyOpLx ||
            op_func == OpFuncs::ReStickifyOpHBM);
  });
}

void Dxp::runDdc(SuperDsc* sdsc) {
  if (!isRestickifyMultiDataStageSdsc(*sdsc)) {
    L3DlOpsScheduler l3_scheduler(dscGlobal, memTrackers.get(), {executionStep},
                                  verbose);
    l3_scheduler.run(*sdsc);
  }

  ...
}
```

## Interpretation

This is the strongest evidence so far that Deeptools already has a true
LX-local restickify lowering path:

1. `ddc_standalone` can expand the restickify DDL fixture.
2. DCC can lower the expanded schedule to LX/SFP/PT work.
3. DXP can compile the fixture end-to-end if its generic pre-DDC passes are
   bypassed for this special DDL-template input.
4. The resulting senprog contains `LXLU/LXSU/SFP/PT` and no `L3/HBM` tokens.

This still needs a real Deeptools build and regression pass. The preload shim
proves the control-flow hypothesis; it is not itself a production patch.

## Next Step

The next engineering step is a Deeptools branch, built in an environment with
Deeptools source/build access, containing the two targeted bypasses above.

Then rerun:

```text
dxp_standalone --bundle -d <fixture bundle>
```

without `LD_PRELOAD`. Success criteria:

- DXP exits `0`
- generated senprog still has `LXLU/LXSU/SFP/PT`
- generated senprog still has no `L3LU/L3SU/HBM`
- normal non-restickify DXP tests continue to use the original corelet split and
  L3 scheduling paths

Only after that should Torch-Spyre learn to emit or route to this
restickify-DDL input form.

## Validation

Local:

```text
python3 -m py_compile tools/restickify_dxp_preddc_shim_probe.py
```

Pod:

```text
python3 /tmp/restickify_dxp_preddc_shim_probe.py ... 
```

