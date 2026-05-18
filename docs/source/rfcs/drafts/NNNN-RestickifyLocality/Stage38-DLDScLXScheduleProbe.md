# Stage 38: DLDSc LX Schedule Probe

## Summary

Stage 37 proved two separate facts:

- `ReStickifyOpLx` data-op SDSCs lower through DCC to LX-local looking IR.
- DXP bundles reject raw `datadscs_` and accept only the normal `dscs_` /
  DLDSc-shaped path.

Stage 38 tested whether we can bridge the gap by mutating a normal
`ReStickifyOpHBM` DLDSc SDSC into a true LX-local `ReStickifyOpLx` DLDSc SDSC.

The answer is no for the simple JSON-rewrite path. We can make DXP accept a
`ReStickifyOpLx`-named DLDSc, but the lowered IR is either empty or still routes
through L3-style units. The only artifact that currently lowers to real
LX-local work is still the data-op SDSC.

## Code Change

Added a diagnostic tool:

```text
tools/restickify_dldsc_bridge_probe.py
```

The tool takes a seed `ReStickifyOpHBM` SDSC, writes a matrix of
`ReStickifyOpLx` DLDSc-shaped variants, optionally runs DCC/DXP, and summarizes:

- DCC return code
- DXP return code
- whether DXP accepted the bundle
- lowered unit names
- whether HBM/L3 units are still present
- count of real movement/compute work ops in the lowered IR

This keeps the Stage 38 result reproducible instead of relying on ad hoc JSON
edits in `/tmp`.

## Probe Command

The best seed was the one generated restickify that already had LX input and HBM
output:

```text
/tmp/torchinductor_1000800000/tmp6tff24ol/inductor-spyre/sdsc_fused_mm_1_1kfnjmpd/sdsc_0_ReStickifyOpHBM.json
```

Command:

```sh
python3 /tmp/restickify_dldsc_bridge_probe.py \
  --seed-sdsc /tmp/torchinductor_1000800000/tmp6tff24ol/inductor-spyre/sdsc_fused_mm_1_1kfnjmpd/sdsc_0_ReStickifyOpHBM.json \
  --dataop-sdsc /tmp/restickify-stage36-synthetic-bundle/sdsc_0_ReStickifyOpLx_dataop.json \
  --output-dir /tmp/stage38-dldsc-bridge-probe \
  --deeptools-bin /opt/ibm/spyre/deeptools/bin \
  --run-deeptools
```

## Results

| Variant | DCC | DXP | Units | Work ops | Interpretation |
|---|---:|---:|---|---:|---|
| `dldsc_rename_only` | 0 | 0 | `l3lu:32`, `l3su:32` | 0 | DXP accepts the renamed op, but no useful restickify work is emitted. |
| `dldsc_output_lx` | 0 | -6 | `l3lu:32`, `l3su:32` | 0 | Output allocation changed to LX, but DXP rejects it. |
| `dldsc_output_lx_memorg_lx` | 0 | 0 | `l3lu:32`, `l3su:32` | 0 | DXP accepts, but lowered IR is empty L3 program units. |
| `dldsc_all_lx_memorg_lx` | 0 | 0 | `l3lu:32`, `l3su:32` | 0 | DXP accepts, but lowered IR is still empty L3 program units. |
| `dataop_lx_reference` | 0 | n/a | `lxlu0:1`, `lxsu0:1`, `pe0:1` | 5 | Data-op path lowers to real LX-local movement. |

The accepted DLDSc variants are not a valid locality solution. The ProgIR for
the accepted LX-tagged DLDSc contains only empty `l3lu/l3su` program units:

```text
dataflow.get_unit {core = 0, name = "l3lu", type = "l3lu"}
dataflow.get_unit {core = 0, name = "l3su", type = "l3su"}
...
dataflow.program_unit iter_arg : %arg0 -> (...) : {
}
dataflow.program_unit iter_arg : %arg0 -> (...) : {
}
```

The data-op reference remains the only successful LX-local lowering:

```text
dataflow.get_unit {name = "lxlu0", type = "lxlu"}
dataflow.get_unit {name = "lxsu0", type = "lxsu"}
dataflow.get_unit {name = "pe0", type = "pe"}
sentient.load_and_send
sentient.receive_and_store
sentient.vector_binary
```

## Conclusion

The first-class LX-local restickify bridge is not achievable by mutating the
post-lowered DLDSc JSON that Torch-Spyre currently emits for
`ReStickifyOpHBM`.

The important split is:

- **Data-op path:** knows how to express real LX-local restickify movement, but
  DXP bundle ingestion rejects raw `datadscs_`.
- **DLDSc path:** DXP accepts it, but the existing generated restickify schedule
  is HBM/L3-shaped. Renaming it to `ReStickifyOpLx` or forcing LX allocations
  does not synthesize the missing LX load/store/PE schedule.

This means the next productive engineering direction is not more JSON mutation.
It is one of:

1. Feed the restickify DDL template the right pre-lowered input so Deeptools
   generates the LX-local DLDSc schedule itself.
2. Add a DXP bundle integration path for the DCC-lowered data-op module.
3. Implement a real Torch-Spyre DLDSc schedule generator for `ReStickifyOpLx`
   using the data-op/DDL lowering as the reference contract.

Option 1 is the cleanest next experiment because the installed
`restickify_sen1p5.ddl` already names `ReStickifyOpLx`, `lxlu_input`, and
`lxsu_input`. The blocker is that our current generated SDSC is already too
low-level for DDL matching:

```text
[DDC] DDL found but not suitable for op ReStickifyOpHBM
Ddl mapping failed
```

So the next step should be to find the pre-DLDSc representation that DDL expects
for restickify, not to keep editing the final DLDSc schedule.

## Validation

Local:

```text
python3 -m py_compile tools/restickify_dldsc_bridge_probe.py
```

Pod:

```text
python3 /tmp/restickify_dldsc_bridge_probe.py ... --run-deeptools
```

Output summary was written to:

```text
/tmp/stage38-dldsc-bridge-probe/summary.json
```

