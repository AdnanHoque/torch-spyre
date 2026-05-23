# Stage 316: Valid-Gap Endpoint Adapter Probe

## Summary

This stage adds a compile-only endpoint adapter candidate for the native PT-LX
tile path. The previous direct adapter tried to consume the native
`j_, i_, out_, mb_` tile workspace and write the consumer `out_, mb_` endpoint
directly. Deeptools rejected that shape in `ReStickifyOpWithPTLx`.

The new diagnostic adapter keeps the stock HBM fallback and emits a single
`ReStickifyOpWithPTLx` using the valid-gap consumer descriptor family:

- input layout: `out_, mb_, in_`
- input stick: `out_`
- output layout: `mb_, in_`
- output stick: `in_`
- source base: native PT-LX restickify output workspace
- destination base: consumer LX endpoint

The adapter is intentionally marked `semantic_transform_certified=False`.
It proves that the endpoint shape can pass DCG, not that the native workspace
can be reinterpreted this way without an explicit value transform.

## Commands

```sh
SPYRE_RESTICKIFY_LX_DATAOP=1 python tools/restickify_lx_dataop_probe.py \
  --streaming-ptlx-tile \
  --validgap-endpoint-adapter-tile \
  --size 512 \
  --tile-index 0 \
  --output-dir /tmp/stage316-validgap-endpoint-adapter-tile \
  --run-dcg \
  --dcg-standalone "$(command -v dcg_standalone)"

SPYRE_RESTICKIFY_LX_DATAOP=1 python tools/restickify_lx_dataop_probe.py \
  --streaming-ptlx-tile \
  --validgap-endpoint-adapter-tile \
  --size 2048 \
  --tile-index 0 \
  --output-dir /tmp/stage316-validgap-endpoint-adapter-tile-2048 \
  --run-dcg \
  --dcg-standalone "$(command -v dcg_standalone)"
```

## Results

| Size | Tile | Mode | DCG |
|---:|---:|---|---:|
| 512 | 0 | `validgap_ptlx_endpoint_adapter_tile` | 0 |
| 2048 | 0 | `validgap_ptlx_endpoint_adapter_tile` | 0 |

The 2048 DCG log reached PCFG generation:

```text
Computing Re-StickifyOpWithPT (Special re-stickify) transfer function..
Creating PCFG for DataDsc..
Writing DataDsc to .../sdsc.json..
Writing PCFG to .../pcfg.json..
```

## Interpretation

This moves the blocker forward. The direct native endpoint adapter is still not
accepted by Deeptools, but a first-class valid-gap endpoint adapter does compile
for both a small shape and the high-signal 2048 shape.

What this does not prove:

- It does not prove value correctness.
- It does not prove that native `j_, i_, out_, mb_` data can be safely viewed as
  the valid-gap `out_, mb_, in_` descriptor.
- It does not remove the production fallback to `ReStickifyOpHBM`.

Next step: connect this accepted endpoint adapter to the native PT-LX sidecar in
a controlled bundle and value-check one tile before widening beyond compile-only
validation.

## Validation

```sh
python -m pytest \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  tests/inductor/test_restickify_lx_dataop.py -q
```

Result: `61 passed`.
