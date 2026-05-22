# Stage 223: Compile-Only Streaming PT-LX Tile

## Summary

This stage feeds the static streaming PT-LX tile payload from Stage 222 into
`dcg_standalone` without launching hardware. The result is the first
compile-accepted production-shaped tile bridge:

- gather producer fragments with `STCDPOpLx`;
- run local `ReStickifyOpWithPTLx` on the bridge core;
- write the consumer-layout tile with `STCDPOpLx`;
- keep the stock `ReStickifyOpHBM` path as fallback outside this static tile
  probe.

## Fixes Needed For DCG Acceptance

The first compile attempt failed because the PT-LX restickify input was still
fragmented into 16-row source pieces. `ReStickifyOpWithPTLx` requires the local
input/output pieces to be at least one output stick, so the gather output now
coalesces source fragments into one full 64x64 tile on the bridge core.

The second compile attempt failed because the restickify data-op claimed all 32
cores even though only the bridge core had restickify input/output pieces. The
lowering now emits sparse per-stage core participation:

- gather stage: source cores plus bridge core;
- restickify stage: bridge core only;
- scatter stage: destination cores plus bridge core.

The top-level `coreIdToDscSchedule` is now sparse as well. Cores that only
participate in gather do not schedule the restickify or scatter stages, and idle
cores have an empty schedule.

## Validation

Validated in the Spyre pod at `/tmp/torch-spyre-bench216`:

```bash
export SPYRE_RESTICKIFY_LX_DATAOP=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

python -m py_compile \
  tools/restickify_lx_dataop_probe.py \
  torch_spyre/_inductor/codegen/restickify_lx_dataop.py \
  torch_spyre/_inductor/codegen/restickify_ptlx_streaming.py \
  tests/inductor/test_restickify_lx_dataop.py

python -m pytest tests/inductor/test_restickify_lx_dataop.py -q

python tools/restickify_lx_dataop_probe.py \
  --streaming-ptlx-tile \
  --size 512 \
  --output-dir /tmp/stage223-streaming-tile-dcg-final \
  --run-dcg \
  --dcg-standalone /home/adnan-cdx/dt-inductor-mixed/sentient/deeptools/bin/dcg_standalone
```

Results:

- `tests/inductor/test_restickify_lx_dataop.py`: `19 passed`
- `dcg_standalone -initSdsc`: `dcg_rc=0`
- generated input payload:
  `/tmp/stage223-streaming-tile-dcg-final/sdsc_streaming_ptlx_tile_512.json`
- generated DCG output:
  `/tmp/stage223-streaming-tile-dcg-final/dcg/sdsc_streaming_ptlx_tile_512/sdsc.json`
- generated PCFG:
  `/tmp/stage223-streaming-tile-dcg-final/dcg/sdsc_streaming_ptlx_tile_512/pcfg.json`

Token inspection:

| Artifact | ReStickifyOpHBM | ReStickifyOpWithPTLx | STCDPOpLx | HBM | LX | PT |
|---|---:|---:|---:|---:|---:|---:|
| input static payload | 1 | 3 | 6 | 1 | 2 | 5 |
| generated `sdsc.json` | 0 | 3 | 5 | 0 | 5 | 4 |
| generated `pcfg.json` | 0 | 0 | 0 | 0 | 1 | 1 |

The `ReStickifyOpHBM` and `HBM` mentions in the input static payload are from
the fallback note, not from an emitted data-op. The generated DCG artifacts have
no `ReStickifyOpHBM` and no `HBM` token.

## Current Limitation

The `-s` senprog path still fails on this imported payload with Deeptools'
folded-SuperDSC restriction:

```text
DtException: Codegen for Folded Super-DSC is not supported
```

Removing the fold object entirely causes JSON import to fail, so senprog export
needs the older no-op-fold stripping workaround or a dedicated data-op export
path. This is separate from compile-only DCG acceptance, which now succeeds.

## Next Step

The next production-shaped step is to generate all tiles for the 512 case and
package them as a complete streaming bridge, still compile-only first. The next
acceptance target is:

- every materialized tile payload compiles with `dcg_rc=0`;
- no generated tile payload contains `ReStickifyOpHBM`;
- the full-tile package preserves sparse per-stage schedules;
- hardware launch remains disabled until the multi-tile compile artifact is
  structurally clean.
