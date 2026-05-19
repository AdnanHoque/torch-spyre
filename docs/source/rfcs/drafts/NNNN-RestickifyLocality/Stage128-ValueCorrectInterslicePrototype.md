# Stage 128: Value-Correct Inter-Slice Prototype

## Goal

Take the Stage127 diagnosis one step further: prove that the LX-to-LX internal
edge can be made value-correct when lowered as an inter-slice transpose, rather
than as the generic `restickify.ddl` template.

## Prototype Setup

This stage used the existing default-off Torch-Spyre DDL bridge path with:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1
SPYRE_RESTICKIFY_DDL_BRIDGE_SOURCE_ADDRESS=compact-lxlu
SPYRE_RESTICKIFY_DDL_BRIDGE_BOUNDARY_PATCH=1
SPYRE_ALIGN_CORE_DIVISION_CONTINUITY=1
SPYRE_ALIGN_CORE_MAPPING_CONTINUITY=1
SPYRE_RESTICKIFY_DDL_BRIDGE_OPFUNC=interslicetranspose_fp16
```

The installed Deeptools image only ships `restickify.ddl`, so the run used a
temporary `DEEPTOOLS_PATH=/tmp/deeptools-share` overlay.  The working prototype
template is captured in `tools/restickify_interslice_2d_template.ddl`.

The important template difference versus the stock source
`inter_slice_transpose.ddl`:

- use an empty slice layout,
- map only `%asdin` and `%asdout` as the two global/stick dimensions,
- keep the bottom datastage so the PT/L0 allocation is small enough.

## Result

Fixture:

```python
def fn(a, b, c, d):
    u = a + (b + c).t()
    return u, u @ d
```

Inputs:

- `b[i, j] = i`
- `a = c = d = 0`
- expected `u[row, col] = col`

Hardware results:

| Size | Result |
|---:|---|
| 512 | `total_bad 0 / 262144`, `max_abs 9.155e-05` |
| 1024 | `total_bad 0 / 1048576`, `max_abs 9.155e-05` |
| 1536 | DXP allocation failure: `Cannot allocate even the smallest size` |
| 2048 | DXP allocation failure: `Cannot allocate even the smallest size` |

This is the first value-correct LX-to-LX restickify-style internal edge in the
prototype line.  It avoids the stock `ReStickifyOpHBM` path for this edge and
uses a consumer LX boundary instead of a generated HBM reload.

## Interpretation

We have separated the problem into two layers:

1. **Semantic layer:** value-correct internal-edge transpose is possible with an
   inter-slice style DDL contract.  This is proven for 512 and 1024.
2. **Capacity/scheduling layer:** the current temporary template still allocates
   too much for 1536 and 2048.  The next blocker is datastage sizing, not the
   logical coordinate transform.

## Next Step

Package the template overlay into the Torch-Spyre prototype path rather than
manually constructing `/tmp/deeptools-share`, then tune the datastage/chunking
contract until the 2048 fixture compiles.

Acceptance for the next stage:

- 2048 tuple fixture compiles,
- `u` is value-correct,
- no stock `ReStickifyOpHBM` kernel is emitted for the internal edge,
- the consumer reads the bridge output from LX.
