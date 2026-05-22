# Stage 224: Multi-Tile Streaming PT-LX Compile Sweep

## Summary

Stage 223 proved that one 512x512 streaming PT-LX tile can be represented as
three data-op stages and accepted by `dcg_standalone`. This stage asks whether
that recipe generalizes across every tile for several non-2048 sizes.

The answer is yes at compile-only granularity:

| Size | Tiles | DCG Pass | DCG Fail | Fan-In/Fan-Out Pattern |
|---:|---:|---:|---:|---|
| 512 | 64 | 64 | 0 | `(4, 1)` for all tiles |
| 1024 | 256 | 256 | 0 | `(2, 1)` for all tiles |
| 1536 | 576 | 576 | 0 | `(2, 1)` for all tiles |

No hardware launch was attempted.

## What Changed

`tools/restickify_lx_dataop_probe.py` now supports:

- `--streaming-ptlx-tile`
- `--all-streaming-tiles`
- `--tile-index`

With `--all-streaming-tiles`, the probe materializes every sampled 64x64 tile,
writes a separate static tile payload for each one, and optionally runs
`dcg_standalone -initSdsc` on each payload.

This still emits separate per-tile compile probes, not one fused runtime
artifact.

## Validation Commands

All commands ran in the Spyre pod from `/tmp/torch-spyre-bench216`:

```bash
export SPYRE_RESTICKIFY_LX_DATAOP=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

python -m py_compile tools/restickify_lx_dataop_probe.py

python tools/restickify_lx_dataop_probe.py \
  --streaming-ptlx-tile \
  --all-streaming-tiles \
  --size 512 \
  --output-dir /tmp/stage224-streaming-tiles-512 \
  --run-dcg \
  --dcg-standalone /home/adnan-cdx/dt-inductor-mixed/sentient/deeptools/bin/dcg_standalone

python tools/restickify_lx_dataop_probe.py \
  --streaming-ptlx-tile \
  --all-streaming-tiles \
  --size 1024 \
  --output-dir /tmp/stage224-streaming-tiles-1024 \
  --run-dcg \
  --dcg-standalone /home/adnan-cdx/dt-inductor-mixed/sentient/deeptools/bin/dcg_standalone

python tools/restickify_lx_dataop_probe.py \
  --streaming-ptlx-tile \
  --all-streaming-tiles \
  --size 1536 \
  --output-dir /tmp/stage224-streaming-tiles-1536 \
  --run-dcg \
  --dcg-standalone /home/adnan-cdx/dt-inductor-mixed/sentient/deeptools/bin/dcg_standalone
```

Summaries:

- `/tmp/stage224-streaming-tiles-512/summary.json`: `64/64` tiles compiled.
- `/tmp/stage224-streaming-tiles-1024/summary.json`: `256/256` tiles compiled.
- `/tmp/stage224-streaming-tiles-1536/summary.json`: `576/576` tiles compiled.

## Interpretation

This is a meaningful upgrade from the earlier full-tensor PT-LX bridge:

- The old full bridge only cleanly handled the special 2048 case.
- Smaller sizes failed because the full bridge produced sub-stick pieces.
- The streaming bridge avoids that by gathering fragments into one full 64x64
  bridge tile before `ReStickifyOpWithPTLx`.
- Sparse per-core schedules are required. Source-only cores run gather only;
  the bridge core runs gather/restickify/scatter; idle cores run nothing.

The current proof is still compile-only and per-tile. It does not yet prove a
single packaged multi-tile runtime artifact or value correctness.

## Next Step

Build one complete 512 streaming bridge artifact that contains all 64 tile
stages in one package, still compile-only first. The next acceptance target is:

- one generated artifact represents the entire 512 tensor bridge;
- `dcg_standalone -initSdsc` accepts it;
- generated DCG output has no `ReStickifyOpHBM`;
- only after compile-only success should we attempt a small hardware/value
  correctness run.
