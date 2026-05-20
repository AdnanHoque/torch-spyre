# Stage 129: DXP Knob Restickify Sweep

## Goal

Test whether `DXP_LX_FRAC_AVAIL` changes the current high-size blocker for the
value-correct LX-to-LX inter-slice restickify prototype.

The fixture is the same deterministic boundary-unit graph used in Stage 128:

```python
def fn(a, b, c, d):
    u = a + (b + c).t()
    return u, u @ d
```

Inputs are chosen so `u[row, col] = col`. This makes value failures easy to
spot after the bridge.

## Setup

The run used the temporary inter-slice DDL overlay:

```text
DEEPTOOLS_PATH=/tmp/deeptools-share
SPYRE_RESTICKIFY_DDL_BRIDGE_OPFUNC=interslicetranspose_fp16
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1
SPYRE_RESTICKIFY_DDL_BRIDGE_SOURCE_ADDRESS=compact-lxlu
SPYRE_RESTICKIFY_DDL_BRIDGE_BOUNDARY_PATCH=1
SPYRE_RESTICKIFY_DDL_SHIM_SKIP_CORELET_NAMES=0_add,1_interslicetranspose_fp16_ddl_bridge,2_add
SPYRE_RESTICKIFY_DDL_SHIM_SKIP_L3_NAMES=1_interslicetranspose_fp16_ddl_bridge
```

Artifacts are under:

```text
artifacts/stage129_dxp_sweep/
```

## Results

| `DXP_LX_FRAC_AVAIL` | Size | Result |
|---:|---:|---|
| `0` | 512 | fails before bridge: `initial chunk parameters must fit in LX for SuperDSC: 0_add` |
| `0` | 1024 | fails before bridge: `initial chunk parameters must fit in LX for SuperDSC: 0_add` |
| `0.05` | 512 | value-correct, `bad=0`, `max_abs=9.155e-05` |
| `0.05` | 1024 | value-correct, `bad=0`, `max_abs=9.155e-05` |
| `0.05` | 1536 | fails in DDC: `Cannot allocate even the smallest size` |
| `0.05` | 2048 | fails in DDC: `Cannot allocate even the smallest size` |
| `0.2` | 512 | value-correct, `bad=0`, `max_abs=9.155e-05` |
| `0.2` | 1024 | value-correct, `bad=0`, `max_abs=9.155e-05` |
| `0.2` | 1536 | fails in DDC: `Cannot allocate even the smallest size` |
| `0.2` | 2048 | fails in DDC: `Cannot allocate even the smallest size` |
| `0.5` | 512 | value-correct, `bad=0`, `max_abs=9.155e-05` |
| `0.5` | 1024 | value-correct, `bad=0`, `max_abs=9.155e-05` |
| `0.5` | 1536 | fails in DDC: `Cannot allocate even the smallest size` |
| `0.5` | 2048 | fails in DDC: `Cannot allocate even the smallest size` |
| `1` | 512 | value-correct, `bad=0`, `max_abs=9.155e-05` |
| `1` | 1024 | value-correct, `bad=0`, `max_abs=9.155e-05` |
| `1` | 1536 | fails in DDC: `Cannot allocate even the smallest size` |
| `1` | 2048 | fails in DDC: `Cannot allocate even the smallest size` |

## Interpretation

`DXP_LX_FRAC_AVAIL` affects the ordinary frontend/backend LX reservation
boundary. At `0`, even the small graph fails before the bridge at `0_add`, which
means the knob can make normal op scheduling too aggressive for this fixture.

For `0.05` through `1`, the 512 and 1024 cases remain value-correct, so the
knob does not break the working LX-to-LX prototype.

The important negative result is that the 1536 and 2048 cases fail with the same
DDC allocation error for every nonzero setting. Therefore the current high-size
blocker is not solved by exposing more or less LX through `DXP_LX_FRAC_AVAIL`.
It is still the DDL/datastage contract for the inter-slice bridge.

## Next Step

Continue at the template/contract layer:

- either express a stock-restickify-like mixed-layout intermediate for the
  inter-slice bridge,
- or split the bridge into a Deeptools-native two-step movement/restickify shape
  that keeps the consumer LX boundary but reduces per-stage allocation.

The DXP knob is still useful as a diagnostic control, but it should not be the
main path to the 2048 prototype.
