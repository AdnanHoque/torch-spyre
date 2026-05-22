# Stage 271: Direction-Aware Direct Tile Findings

## Summary

This stage tightened the tiled PT-LX prototype after the `512`
`computed_transpose_adds_then_matmul` probe showed wrong values with the
same-layout LX remap bridge.

The key finding is that this case is not a same-layout ownership remap.  The
stock SDSC is a real restickify:

- source view: `out, mb` with stick `mb`
- destination view: `mb, out` with stick `out`

The previous same-layout selector was too permissive because it compared only
the adjacent producer output and restickify destination.  It now requires the
producer output, restickify source, and restickify destination layouts/sticks to
all match before selecting the pure `STCDPOpLx` remap path.

## What Changed

- Split same-layout LX remaps into explicit source/destination logical
  intersections instead of many-source-to-one-destination pieces.
- Added direction-aware direct tiled PT-LX descriptor generation.
- Added a validation-only force flag:
  `SPYRE_RESTICKIFY_PTLX_FORCE_DIRECT_TILE_E2E=1`.
- Passed the restickify logical direction into the direct tiled descriptor so an
  `output-to-kernel` restickify emits a consumer-shaped `mb,out` / `out`
  output descriptor.

## Validation

No-launch descriptor probe:

```sh
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1 \
SPYRE_RESTICKIFY_PTLX_DIRECT_TILE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul \
  --size 512 \
  --skip-correctness \
  --skip-kernel-launch
```

The generated direct-tile bridge now matches the consumer descriptor:

- bridge layout: `mb,out`
- bridge stick: `out`
- consumer layout: `mb,out`
- consumer stick: `out`
- endpoint contract: valid
- value-preservation count: valid

Forced validation run:

```sh
SPYRE_RESTICKIFY_PTLX_FORCE_DIRECT_TILE_E2E=1 ...
```

This did not reach value checking.  Deeptools rejected the generated gather:

```text
myOp->outLds->stickDimOrder_ == myOp->inpLds->stickDimOrder_
```

## Conclusion

The production-shaped path cannot use plain `STCDPOpLx` for a gather that also
changes the stick dimension.  The next implementation step needs one of these
forms:

- an `InputFetchNeighbor`/native remote-LX gather that materializes the
  restickify source view directly;
- a `ReStickifyOpWithPTLx` tile descriptor that consumes the producer fragments
  directly without a stick-changing `STCDPOpLx` pre-gather;
- or a Deeptools-native fused data-op contract for remote-LX restickification.

The stock HBM fallback remains intact.  The direct tile force flag is
validation-only and must not be treated as production eligibility.
