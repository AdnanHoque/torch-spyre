# Stage 201: True-DL Restickify Probe

## Goal

Test whether we can avoid the mixed `datadscs_` route by representing
LX-to-LX restickification as a normal DLDSc / DL op in the bundle path.

This stage asks a narrow question:

```text
Can an existing top-level Deeptools DL opfunc generate real LX/PT/SFP
restickify movement when placed in the normal Torch-Spyre `dscs_` bundle path?
```

## Why This Matters

Stage200 proved that a mixed schedule can carry the non-HBM bridge:

```text
DL consumer in dscs_
PT/LX restickify bridge in datadscs_
coreIdToDscSchedule runs data ops before the consumer
```

But DXP's normal bundle path is not comfortable with raw `datadscs_`.
The cleanest production path, if it existed, would be a true DL restickify
carrier: no mixed schedule, no embedded PCFG import, and no runtime splice.

## Probe

Added:

```text
tools/restickify_true_dl_probe.py
```

The tool starts from a real stock `ReStickifyOpHBM` SDSC emitted by
Torch-Spyre and rewrites only:

- the top-level DSC key;
- `computeOp_[0].opFuncName`;
- optional output/all allocation metadata to `lx`.

It then runs:

```text
dcc_standalone --input-mode=sdsc --kEmitProgIR
dxp_standalone --bundle -d <variant-dir>
```

Candidate opfuncs:

- `ReStickifyOpHBM` control
- `ReStickifyOpLx`
- `ReStickifyOpWithPTLx`
- `interslicetranspose_fp16`

Memory variants:

- original stock metadata
- output marked LX
- all inputs/outputs marked LX

## Results

### ReStickify DL Names

For both a recent high-signal 2048 seed and a fresh 512 seed:

| Opfunc | Memory marking | DCC | DXP | Lowered units | Work ops | Result |
|---|---|---:|---:|---|---:|---|
| `ReStickifyOpHBM` | original/output-LX/all-LX | 0 | 0 | `l3lu:32,l3su:32` | 0 | Empty DL shell |
| `ReStickifyOpLx` | original/output-LX/all-LX | 0 | 0 | `l3lu:32,l3su:32` | 0 | Empty DL shell |
| `ReStickifyOpWithPTLx` | original/output-LX/all-LX | 0 | 0 | `l3lu:32,l3su:32` | 0 | Empty DL shell |

Interpretation:

The `ReStickifyOpLx` and `ReStickifyOpWithPTLx` names are accepted by the
DLDSc bundle machinery, but as DL opfunc rewrites they do not generate
restickify movement.  There are no `lxlu`, `lxsu`, `pt`, or `sfp` units and no
counted movement instructions.

This matches the Deeptools source inspection: the real `ReStickifyOpLx` and
`ReStickifyOpWithPTLx` implementations live in the data-op PCFG generation
path, not as stock DL compute schedules.

### `interslicetranspose_fp16`

Without an overlay, DXP fails because the installed image has no:

```text
/opt/ibm/spyre/deeptools/share/ddc/ddl_templates/inter_slice_transpose.ddl
```

With `tools/restickify_interslice_2d_template.ddl` copied into a temporary
`DEEPTOOLS_PATH` overlay, DXP finds the template but still fails from a stock
restickify DLDSc seed:

```text
DtException: Cannot allocate even the smallest size
file /project_src/deeptools/ddc/ddcv1.cpp line 1286
```

This happened for both:

- the 2048 stock restickify seed;
- a fresh 512 stock restickify seed.

Interpretation:

`interslicetranspose_fp16` can be made value-correct in the earlier custom DDL
bridge path, but a stock post-lowered `ReStickifyOpHBM` DLDSc is not the right
input contract for that template.  Renaming the opfunc after Torch-Spyre has
already produced the final DLDSc shape does not reconstruct the high-level
DDL/datastage contract.

## Artifacts

Local summaries:

```text
artifacts/stage201_true_dl_probe/summary_2048_stock.json
artifacts/stage201_true_dl_probe/summary_2048_interslice_overlay.json
artifacts/stage201_true_dl_probe/summary_512_interslice_overlay.json
```

Pod summaries:

```text
/tmp/stage201-true-dl-probe/summary.json
/tmp/stage201-true-dl-interslice-overlay/summary.json
/tmp/stage201-true-dl-probe-512/summary.json
```

## Conclusion

The true-DL-by-rewrite route is negative.

Existing top-level DL opfunc names do not give us a production-ready
LX-to-LX restickify bridge:

- `ReStickifyOpLx` and `ReStickifyOpWithPTLx` lower as empty L3 DL shells when
  used this way.
- `interslicetranspose_fp16` needs a real high-level DDL template contract, not
  a post-lowered stock restickify DLDSc with the op name changed.

This narrows the production directions:

1. **Supported mixed DL+data-op schedule:** teach the normal bundle path to
   accept the data-op bridge contract that already generates LX/PT/SFP work.
2. **Real high-level DDL lowering:** generate a correct pre-DLDSc
   restickify/interslice descriptor from Torch-Spyre so DDC creates the bridge
   schedule, rather than mutating the final DLDSc.
3. **Deeptools contract addition:** add or expose a first-class DL carrier that
   internally invokes the proven `ReStickifyOpLx` / `ReStickifyOpWithPTLx`
   PCFG generation path.

So the next best step is not more DL-name rewriting.  It is to pursue either an
official mixed-schedule bundle contract or a real pre-DLDSc DDL descriptor.

## Validation

Local:

```text
python3 -m py_compile tools/restickify_true_dl_probe.py
```

Pod compile-only:

```text
python3 tools/restickify_true_dl_probe.py \
  --seed-sdsc <stock ReStickifyOpHBM SDSC> \
  --output-dir /tmp/stage201-true-dl-probe \
  --run-deeptools
```

No hardware launch was used in this stage.
