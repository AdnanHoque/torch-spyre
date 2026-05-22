# Stage 225: Full 512 Streaming PT-LX Bridge Compile

## Summary

Stage 224 compiled every 512 tile as a separate static payload. This stage
packages all 64 tiles into one compile-only bridge artifact and asks
`dcg_standalone` to compile the full bridge.

Result: the full 512 bridge compiles.

This is the closest prototype so far to the production-shaped fix:

- one artifact represents the whole 512 tensor bridge;
- the bridge contains gather, PT-LX restickify, and scatter stages for every
  64x64 tile;
- DCG accepts the artifact;
- generated DCG output has no `ReStickifyOpHBM` and no `HBM` token;
- hardware launch and value correctness are still not attempted.

## What Changed

Added `generate_streaming_ptlx_full_bridge_sdsc(...)`, which combines every
materialized streaming tile into one SuperDSC-shaped static payload.

`tools/restickify_lx_dataop_probe.py` now supports:

- `--full-streaming-bridge`

used together with:

- `--streaming-ptlx-tile`
- `--all-streaming-tiles`

The full bridge preserves the sparse per-core schedules from the individual
tile payloads. Different tiles use different bridge cores as destination
ownership changes, which is expected.

## Validation

Validated in the Spyre pod at `/tmp/torch-spyre-bench216`:

```bash
export SPYRE_RESTICKIFY_LX_DATAOP=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

python -m py_compile \
  tools/restickify_lx_dataop_probe.py \
  torch_spyre/_inductor/codegen/restickify_lx_dataop.py \
  tests/inductor/test_restickify_lx_dataop.py

python -m pytest tests/inductor/test_restickify_lx_dataop.py -q

python tools/restickify_lx_dataop_probe.py \
  --streaming-ptlx-tile \
  --all-streaming-tiles \
  --full-streaming-bridge \
  --size 512 \
  --output-dir /tmp/stage225-streaming-full-512 \
  --run-dcg \
  --dcg-standalone /home/adnan-cdx/dt-inductor-mixed/sentient/deeptools/bin/dcg_standalone
```

Results:

- `tests/inductor/test_restickify_lx_dataop.py`: `20 passed`
- full bridge `dcg_rc=0`
- generated input payload:
  `/tmp/stage225-streaming-full-512/sdsc_streaming_ptlx_full_512.json`
- generated DCG `sdsc.json`:
  `/tmp/stage225-streaming-full-512/dcg/sdsc_streaming_ptlx_full_512/sdsc.json`
- generated DCG `pcfg.json`:
  `/tmp/stage225-streaming-full-512/dcg/sdsc_streaming_ptlx_full_512/pcfg.json`

Token inspection:

| Artifact | ReStickifyOpHBM | ReStickifyOpWithPTLx | STCDPOpLx | HBM | LX | PT |
|---|---:|---:|---:|---:|---:|---:|
| input static payload | 1 | 192 | 384 | 1 | 3 | 195 |
| generated `sdsc.json` | 0 | 129 | 257 | 0 | 257 | 130 |
| generated `pcfg.json` | 0 | 0 | 0 | 0 | 1 | 1 |

The `ReStickifyOpHBM` and `HBM` mentions in the input payload are from the
recorded fallback string, not an emitted data-op. DCG output removes those
mentions.

## Interpretation

This resolves a major prototype gap. We are no longer proving only isolated
tiles: DCG accepts a complete streaming bridge for the 512 case, packaged as
one static data-op artifact.

The remaining production gaps are now clearer:

- integrate this payload generation into normal Torch-Spyre lowering for an
  actual producer/restickify/consumer boundary;
- bind producer and consumer LX endpoint addresses from scratchpad planning,
  not probe defaults;
- preserve value-flow and lifetime metadata across the normal bundle;
- run a small hardware/value-correct graph only after static integration checks
  show no `ReStickifyOpHBM` on the patched boundary.

## Next Step

Try the same full-bridge compile for 1024. If that succeeds, the next real
implementation step is normal-lowering integration: produce this full streaming
bridge from the compiler's restickify boundary plan instead of from the
standalone probe harness.
