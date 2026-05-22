# Stage 293: Coordinate-Aware Bridge Selection

## Goal

Tighten the streaming LX-neighbor bridge candidate so it does not label a
coordinate-changing restickify edge as a plain same-layout `STCDPOpLx` ownership
remap.

The concrete failing shape was:

```python
def fn(a, b, c, d):
    u = a + (b + c).t()
    return u, u @ d
```

at `size=512`.

## What We Found

The forced legacy streaming PT-LX path emitted a no-HBM bundle at 512, but the
hardware tuple check failed:

```text
Mismatched elements: 131004 / 262144 (50.0%)
```

A patterned input showed a deterministic tile/coordinate scramble, not random
device behavior.  The edge's restickify direction is:

```text
output-to-kernel
```

but the legacy 2D streaming helper was being forced through a shape that did not
prove that coordinate transform.

Separately, the LX-neighbor sidecar had a selection bug: it picked
`same-layout-lx-ownership-remap` when the producer and destination primary
layout/stick metadata matched, even if the symbolic tensor coordinates did not.
For the 512 transpose edge, the source-view contract says:

```text
producer physical output:        floor(c1/64), c0, Mod(c1,64)
restickify logical source view:  floor(d0/64), d1, Mod(d0,64)
restickify destination view:     floor(d1/64), d0, Mod(d1,64)
consumer input view:             floor(c1/64), c0, Mod(c1,64)
```

So this is not a pure same-layout ownership remap.  It needs a real PT-LX
layout/coordinate transform.

## Code Change

`lx_neighbor_descriptor.py` now requires both conditions before selecting the
`same-layout-lx-ownership-remap` bridge:

- producer and destination primary layout/stick metadata match;
- symbolic coordinate relations are identity for both
  producer-to-restickify-source and restickify-destination-to-consumer.

If either coordinate relation is not identity, the sidecar is emitted as:

```text
bridge_kind: direct-ptlx-layout-transform
```

The descriptor also threads the inferred restickify direction into direct PT-LX
sidecars.  For the 512 transpose edge, the emitted sidecar now reports:

```json
{
  "bridge_kind": "direct-ptlx-layout-transform",
  "direction": "output-to-kernel",
  "bridge_metadata": {
    "coalescing": "direct-64x64-tiles",
    "direction": "output-to-kernel",
    "semantic_transform_certified": false,
    "fallback": "ReStickifyOpHBM"
  }
}
```

The stock HBM fallback remains the executable path.

## Validation

Unit validation in the pod:

```text
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py -q
```

Result:

```text
11 passed in 7.74s
```

Real 512 descriptor probe:

```text
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED=1
SPYRE_RESTICKIFY_LX_NEIGHBOR_STREAMING_BRIDGE=1
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 512 \
  --skip-correctness \
  --skip-kernel-launch \
  --copy-kernel-code \
  --output-dir /tmp/stage293-descriptor-512 \
  --fail-on-error
```

Result:

```text
ok size=512 case=computed_transpose_adds_then_matmul_tuple
```

The sidecar now correctly emits a direct PT-LX candidate instead of an
STCDP-only remap.

Exporting that direct sidecar through the DeeRT data-op exporter still fails:

```text
DtException:
PieceInfo::getTotalValid(op->inpLds->validGap_.at(stickDimOut))
<= PieceInfo::getTotalValid(op->outLds->validGap_.at(stickDimOut))
restickifyOp.cpp line 1926
```

That is the next blocker: the sidecar is now classified correctly, but the
direct 64x64 PT-LX piece contract is not yet accepted by Deeptools for this
fan-in shape.

## Interpretation

This stage does not complete the production PT-LX path for non-2048 sizes.  It
does remove an unsafe false positive:

- before: a coordinate-changing edge could be represented as a same-layout
  `STCDPOpLx` remap and appear semantically certified;
- after: the same edge is classified as a direct PT-LX transform with
  `semantic_transform_certified=false`, preserving the stock `ReStickifyOpHBM`
  fallback.

The next production-shaped step is to make the direct `output-to-kernel`
64x64-tile PT-LX piece contract acceptable to Deeptools, or to lower the fan-in
gather through an InputFetchNeighbor-compatible same-stick phase before the
local PT-LX transform.
