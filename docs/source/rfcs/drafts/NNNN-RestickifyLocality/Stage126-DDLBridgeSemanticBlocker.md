# Stage 126: DDL Bridge Semantic Blocker

## Goal

The active goal is still a value-correct LX-to-LX restickify inside a
Torch-Spyre graph:

```text
producer add -> LX-local restickify bridge -> consumer add
```

Stage 125 proved that the consumer boundary can be patched so it reads the
bridge output from LX instead of reloading that tensor from HBM. Stage 126 asks
the next question: does the compact DDL bridge compute the same logical
restickification as `ReStickifyOpHBM`?

## New Probe Knob

I added one default-off diagnostic knob:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_LOOP_ORDER={reversed-input,input,output,reversed-output}
```

The default is the previous behavior, `reversed-input`. This does not change
normal Torch-Spyre behavior. It only lets us test whether the compact DDL input
was feeding the restickify template the wrong loop order.

## Deterministic Pattern Probe

The useful fixture was:

```python
def fn(a, b, c, d):
    u = a + (b + c).t()
    return u, u @ d
```

with:

```python
b[i, j] = i
a = c = d = 0
```

The expected intermediate is:

```text
u[row, col] = col
```

That makes layout mistakes visible without needing to reason through random
matmul output.

## Results

With the DDL bridge, boundary patch, and add corelet-split skip:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1
SPYRE_RESTICKIFY_DDL_BRIDGE_BOUNDARY_PATCH=1
SPYRE_RESTICKIFY_DDL_BRIDGE_SOURCE_ADDRESS=compact-lxlu
SPYRE_RESTICKIFY_DDL_SHIM_SKIP_CORELET_NAMES=0_add,2_add
```

the bridge is not value-correct:

| Mode | Mismatches / 4,194,304 | Max Abs Error | Observation |
|---|---:|---:|---|
| `reversed-input` | `3,792,257` | `2046` | Transposes within 64-wide tiles but does not swap tile ownership globally. |
| `input` | `3,792,257` | `2046` | Same as default. |
| `output` | `3,792,257` | `2046` | Same as default. |
| `reversed-output` | `3,792,257` | `2046` | Same as default. |

Representative default output:

```text
actual row 0: 0, 1, 2, ..., 63, 0, 1, ...
expected row 0: 0, 1, 2, ..., 63, 64, 65, ...
```

This rules out simple loop-order selection as the fix.

Enabling the broader producer-consumer continuity prototype improves the result:

```sh
SPYRE_ALIGN_CORE_DIVISION_CONTINUITY=1
SPYRE_ALIGN_CORE_MAPPING_CONTINUITY=1
```

| Mode | Mismatches / 4,194,304 | Max Abs Error | Observation |
|---|---:|---:|---|
| DDL bridge + continuity + add corelet-split skip | `362,400` | `63` | Cross-tile ownership is mostly fixed; remaining error is tile/corelet shaped. |
| DDL bridge + continuity, normal add corelet split | `2,240,992` | `2047` | Bridge stays one-corelet while adjacent adds are split into two corelets. |
| `ReStickifyOpLx` opfunc + continuity + skip | `434,784` | `63` | Different wrong tile pattern; not a fix. |

So continuity is relevant, but it does not fully solve the bridge semantics.

## Interpretation

The current compact DDL bridge is an **intra-tile restickifier**, not a complete
logical tensor restickifier for this transpose-shaped edge. It can move data
through LX/SFP/PT without HBM, but it does not by itself perform the full
global tile exchange that `ReStickifyOpHBM` provides through the normal
materialized tensor path.

This also explains why the earlier Stage 3B byte-hop model was too optimistic
for this end-to-end replacement goal. The model can say ownership is aligned at
the logical slice level, but the DDL bridge still needs the layout's tile-level
producer/consumer relationship to be represented in the schedule. Without that
relationship, the program can be HBM-free and still wrong.

## Current Blocker

We should stop treating the compact DDL bridge as a drop-in replacement for
`ReStickifyOpHBM`.

The next working prototype needs a first-class tile movement contract:

```text
producer-owned LX tiles
  -> tile ownership exchange when needed
  -> local stick/tile restickification
  -> consumer-owned LX tiles
```

There are two plausible implementation directions:

1. Extend the producer/restickify/consumer internal-edge descriptor so it can
   describe tile ownership, not only per-core base addresses.
2. Compose two Deeptools capabilities: an `InputFetchNeighbor`/STCDP-style
   LX tile movement stage plus the local DDL restickify stage, then fuse that
   compound movement into the same runtime bundle.

## Validation

Pod unit tests after adding the loop-order knob:

```text
python3 -m pytest tests/inductor/test_restickify_ddl_bridge.py -q
17 passed in 0.07s
```

Important negative result:

```text
SPYRE_RESTICKIFY_DDL_BRIDGE_LOOP_ORDER=...
```

does not repair the deterministic pattern for any of the four tested loop
orders. The current blocker is semantic, not a simple loop-order typo.
