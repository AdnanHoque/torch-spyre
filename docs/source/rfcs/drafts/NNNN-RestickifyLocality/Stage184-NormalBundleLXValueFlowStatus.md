# Stage 184: Normal Bundle LX Value-Flow Status

## Goal

Move the LX-to-LX restickify prototype from split/standalone probes into the
normal Torch-Spyre producer/restickify/consumer bundle.

The target fixture remains:

```python
def fn(a, b, c, d):
    u = a + (b + c).t()
    return u, u @ d
```

## What Now Works

The same-artifact splice path can replace the generated
`sdsc_1_ReStickifyOpHBM` frame with an HBM-free LX bridge frame while keeping
the original fused bundle shape.

The safe path uses:

```text
SPYRE_RESTICKIFY_LX_BRIDGE_PATCH_CONSUMER_SDSC=1
SPYRE_RESTICKIFY_LX_DATAOP_DIRECTION=restickify-stcdp-restickify
```

and explicitly does not use the unsafe consumer-frame replacement path.

The generated bridge frame has no HBM tokens. A representative 2048 run emitted:

```text
HBM=0, L3LU=96, L3SU=96, LXLU=64, LXSU=64, PT=0, SFP=0
```

The live bundle can also be patched so:

- producer output LDS is allocated in LX at the bridge source address;
- consumer input LDS is allocated in LX at the bridge destination address;
- the full fused bundle recompiles and launches without a hardware scheduler
  error.

## Current Blocker

The bundle is not yet value-correct.

The diagnostic `actual_u - a` has nearly the same distribution as the expected
bridge payload `(b + c).t()`, but near-zero correlation with the logical tensor.
That means data is moving through the LX bridge, but the payload is delivered in
the wrong logical/tiled order.

So the blocker is no longer:

```text
can we launch an HBM-free LX bridge?
```

It is:

```text
can the producer, bridge, and consumer agree on the same logical source view and
destination view inside one normal Torch-Spyre bundle?
```

## Negative Attempts

Tested and still not sufficient:

- `output-to-kernel`
- `kernel-to-output`
- `restickify-stcdp-restickify`
- single-op `output-to-kernel`
- producer output LX endpoint patch only
- consumer input LX endpoint patch only
- producer source-view patch using a custom role
- producer source-view patch using `KERNEL`
- producer source-view patch using `OUTPUT`

The custom producer source role fails DXP recompile with `map::at`.  The
`KERNEL` role reaches DDL conversion but fails with a slice-size constraint.
The `OUTPUT` role launches but remains value-incorrect.

## Interpretation

Endpoint aliasing is necessary but not sufficient.

The HBM path is value-correct because HBM acts as a global materialization
point.  An LX path must explicitly materialize the same logical view at the
consumer LX endpoint.  A pure LX address patch only proves transport and
lifetime, not restickify semantics.

## Next Step

The next implementation should add a first-class value-flow verifier and then
feed its result into bridge generation:

1. Compare producer physical output coordinates, restickify source-view
   coordinates, bridge output coordinates, and consumer input coordinates.
2. Refuse to launch an LX bridge if those coordinate systems do not match.
3. Generate a descriptor-driven bridge that materializes the exact
   `producer_output -> restickify_source_view -> consumer_input` transform.

The most promising route is a chunked inter-slice/materialization bridge: the
older inter-slice DDL route proved value correctness for 512 and 1024, but 1536
and 2048 still hit DDC allocation limits.  Chunking that bridge is the likely
next way to make the high-signal 2048 case value-correct without falling back to
`ReStickifyOpHBM`.
