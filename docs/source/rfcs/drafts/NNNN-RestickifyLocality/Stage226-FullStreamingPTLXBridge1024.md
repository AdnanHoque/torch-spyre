# Stage 226: Full 1024 Streaming PT-LX Bridge Compile

## Summary

After the full 512 bridge compiled in Stage 225, this stage repeats the same
compile-only experiment for 1024.

Result: the full 1024 bridge compiles.

This matters because the original full-tensor PT-LX prototype only had a clean
special case at 2048. The tiled streaming path now has compile-only evidence for
multiple non-2048 sizes:

- 512 full bridge: compiles as one artifact;
- 1024 full bridge: compiles as one artifact;
- 1536 per-tile sweep: every tile compiles individually.

No hardware launch was attempted.

## Validation Command

Executed in the Spyre pod from `/tmp/torch-spyre-bench216`:

```bash
export SPYRE_RESTICKIFY_LX_DATAOP=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

python tools/restickify_lx_dataop_probe.py \
  --streaming-ptlx-tile \
  --all-streaming-tiles \
  --full-streaming-bridge \
  --size 1024 \
  --output-dir /tmp/stage226-streaming-full-1024 \
  --run-dcg \
  --dcg-standalone /home/adnan-cdx/dt-inductor-mixed/sentient/deeptools/bin/dcg_standalone
```

Result summary:

```text
mode: streaming_ptlx_full_bridge
size: 1024
sample_tiles: 256
tile_count: 256
dcg_rc: 0
```

Generated artifacts:

- input payload:
  `/tmp/stage226-streaming-full-1024/sdsc_streaming_ptlx_full_1024.json`
- generated DCG `sdsc.json`:
  `/tmp/stage226-streaming-full-1024/dcg/sdsc_streaming_ptlx_full_1024/sdsc.json`
- generated DCG `pcfg.json`:
  `/tmp/stage226-streaming-full-1024/dcg/sdsc_streaming_ptlx_full_1024/pcfg.json`

Token inspection:

| Artifact | ReStickifyOpHBM | ReStickifyOpWithPTLx | STCDPOpLx | HBM | LX | PT |
|---|---:|---:|---:|---:|---:|---:|
| input static payload | 1 | 768 | 1536 | 1 | 3 | 771 |
| generated `sdsc.json` | 0 | 513 | 1025 | 0 | 1025 | 514 |
| generated `pcfg.json` | 0 | 0 | 0 | 0 | 1 | 1 |

As in Stage 225, the input payload's `ReStickifyOpHBM`/`HBM` mentions are from
the recorded fallback string. The generated DCG artifacts have no HBM restickify
op and no HBM token.

## Interpretation

This strengthens the case that the tiled PT-LX shape is not a one-off for 512.
The compiler can generate a full static bridge with hundreds of data-op stages,
and DCG can compile it.

The next useful work is not more standalone compile sweeps. It is normal
Torch-Spyre integration:

- derive the streaming descriptor from a real restickify boundary;
- bind producer/consumer LX endpoint ranges from the normal scratchpad plan;
- replace only the eligible restickify boundary with the full streaming bridge;
- statically confirm no `ReStickifyOpHBM` remains for that boundary;
- then attempt a tiny value-correct hardware run.
