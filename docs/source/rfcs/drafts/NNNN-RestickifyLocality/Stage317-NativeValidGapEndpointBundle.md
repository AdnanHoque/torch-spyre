# Stage 317: Native PT-LX + Valid-Gap Endpoint Bundle

## Summary

Stage 316 proved that the valid-gap consumer endpoint adapter compiles as a
standalone data-op. Stage 317 connects that adapter to the native PT-LX tile
sidecar in one controlled compile-only artifact:

```text
STCDPOpLx gather
ReStickifyOpWithPTLx native local tile transform
ReStickifyOpWithPTLx valid-gap consumer endpoint adapter
```

The native helper's old final scatter is intentionally omitted. In this probe,
the valid-gap endpoint adapter is the consumer-facing write.

## Implementation

The new helper is:

```text
generate_native_ptlx_validgap_endpoint_tile_bridge_sdsc(...)
```

It builds the first two data-ops from the native PT-LX tile helper, then appends
the valid-gap endpoint adapter from Stage 316.

The first combined attempt failed because the generic combiner scheduled every
data-op on every core up to `numCoresUsed_`. The tile data-ops only use the
stage-local cores, so DCG rejected the adapter with:

```text
DtException: coreIDX >= 0
```

The fix was to replace the broad sequential schedule with a sparse schedule
derived from each data-op's `coreIdsUsed_`.

## Commands

```sh
SPYRE_RESTICKIFY_LX_DATAOP=1 python tools/restickify_lx_dataop_probe.py \
  --streaming-ptlx-tile \
  --native-validgap-endpoint-tile \
  --size 512 \
  --tile-index 0 \
  --output-dir /tmp/stage317-native-validgap-endpoint-tile-512-sparse-schedule \
  --run-dcg \
  --dcg-standalone "$(command -v dcg_standalone)"

SPYRE_RESTICKIFY_LX_DATAOP=1 python tools/restickify_lx_dataop_probe.py \
  --streaming-ptlx-tile \
  --native-validgap-endpoint-tile \
  --size 2048 \
  --tile-index 0 \
  --output-dir /tmp/stage317-native-validgap-endpoint-tile-2048-sparse-schedule \
  --run-dcg \
  --dcg-standalone "$(command -v dcg_standalone)"
```

## Results

| Size | Tile | Mode | DCG |
|---:|---:|---|---:|
| 512 | 0 | `native_ptlx_validgap_endpoint_tile` | 0 |
| 2048 | 0 | `native_ptlx_validgap_endpoint_tile` | 0 |

The 2048 log shows all three data-ops reached PCFG generation:

```text
Computing transfer function metaData..
Creating PCFG for DataDsc..
Computing Re-StickifyOpWithPT (Special re-stickify) transfer function..
Creating PCFG for DataDsc..
Computing Re-StickifyOpWithPT (Special re-stickify) transfer function..
Creating PCFG for DataDsc..
Writing DataDsc to .../sdsc.json..
Writing PCFG to .../pcfg.json..
```

## Interpretation

This is stronger than Stage 316:

- native PT-LX local transform and valid-gap endpoint adapter can coexist in one
  DCG artifact;
- the combined artifact no longer depends on the stock HBM restickify path for
  code generation;
- the schedule must be sparse and derived from the actual data-op core sets.

This is still not production proof:

- no hardware value run has been performed for this combined artifact;
- the valid-gap endpoint adapter still reinterprets the native workspace and is
  marked `semantic_transform_certified=False`;
- stock `ReStickifyOpHBM` remains the required fallback.

Next step: create the smallest runtime/value harness for this exact three-dataop
bundle and compare the output tile against the stock HBM restickify result.

## Validation

```sh
python -m pytest \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  tests/inductor/test_restickify_lx_dataop.py -q
```

Result: `62 passed`.
