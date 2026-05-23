# Stage 324: Chunked Bridge Frame Splice

## Summary

Stage 324 consumes the Stage 323 export manifests and materializes each selected
chunk sequence as one concatenated PT-LX bridge frame. It then uses the existing
same-artifact splice probe to replace the stock `ReStickifyOpHBM` program frame
inside the normal `matmul_then_add` runtime bundle.

This is still compile/package-only. Hardware was not launched in this stage.

The new helper is:

```sh
python tools/restickify_chunked_bridge_frame.py \
  --manifest <stage323-manifest.json> \
  --output-dir <bridge-frame-dir> \
  --require-no-hbm
```

The helper writes:

- `init_binary.bin`
- `init.txt`
- `init_binary_sentinel_cleared.bin`
- `init_sentinel_cleared.txt`
- concatenated `senprog.txt`
- `summary.json`

For same-artifact splicing, the sentinel-cleared binary is the intended input:
the stock `matmul_then_add` bundle order is:

```text
sdsc_0_batchmatmul.json
sdsc_1_ReStickifyOpHBM.json
sdsc_2_add.json
```

So the replacement bridge starts at frame position 1, not at the beginning of
the full program.

## Commands

Materialize bridge frames:

```sh
for size in 512 1024 2048; do
  out=/tmp/stage324-chunked-bridge-frame-${size}
  rm -rf "$out"
  python3 tools/restickify_chunked_bridge_frame.py \
    --manifest /tmp/stage323-chunked-sidecar-manifest-${size}-r5-timeout/manifest.json \
    --output-dir "$out" \
    --require-no-hbm
done
```

Splice into normal bundles:

```sh
for size in 512 1024 2048; do
  base=/tmp/stage322-native-validgap-auto-rowchunk-files-real-sidecar-${size}/kernel_code/matmul_then_add_${size}/0001_sdsc_fused_addmm_t_0
  out=/tmp/stage324-spliced-chunked-bridge-${size}
  rm -rf "$out"
  python3 tools/restickify_lx_bridge_same_artifact_splice.py \
    --code-dir "$base" \
    --bridge-frame-dir /tmp/stage324-chunked-bridge-frame-${size} \
    --output-dir "$out" \
    --summary "$out.summary.json" \
    --require-hbm-free
done
```

## Results

| Size | Chunks | Bridge Bytes | Bridge Flits | Selected `HBM` Tokens | Selected `LXLU` Tokens | Selected `LXSU` Tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 80,128 | 626 | 0 | 128 | 128 |
| 1024 | 16 | 204,928 | 1,601 | 0 | 512 | 512 |
| 2048 | 32 | 331,776 | 2,592 | 0 | 2,048 | 2,048 |

Splice results:

| Size | Original Bundle Bytes | Stock Restickify Frame Bytes | Bridge Frame Bytes | Patched Bundle Bytes | Restickify Position |
|---:|---:|---:|---:|---:|---:|
| 512 | 22,400 | 7,040 | 80,128 | 95,488 | 1 |
| 1024 | 22,656 | 6,784 | 204,928 | 220,800 | 1 |
| 2048 | 24,192 | 7,040 | 331,776 | 348,928 | 1 |

The splice probe updated:

- `loadprogram_to_device/*/init.txt`
- `loadprogram_to_device_dsg.txt`
- `segment_size.json`

It did not update `spyreCodeDir`, because these captured probe bundles did not
have a `spyreCodeDir` directory.

## Interpretation

This stage proves that the current chunked PT-LX bridge can be packaged into the
same byte-stream location that previously held `ReStickifyOpHBM`.

It does not yet prove runtime validity. The important remaining risks are:

- the runtime `bundle.mlir` and SDSC metadata still describe
  `ReStickifyOpHBM`, while the actual program frame is now the PT-LX chunk
  sequence;
- concatenating independently exported DeeRT chunk frames may not be a valid
  single program sequence for Flex/SpyreCode execution;
- value correctness still depends on the producer LX output, bridge output, and
  consumer LX input agreeing on the same internal data-location contract.

So the next validation must be a cautious hardware smoke on the smallest
spliced case first, with a known-good stock smoke before and after. If the
512 spliced bundle does not retire cleanly, the failure is likely in program
frame sequencing or stale bundle metadata, not in DCC lowering of individual
chunks.

## Artifacts

Pod artifacts:

```text
/tmp/stage324-chunked-bridge-frame-512/summary.json
/tmp/stage324-chunked-bridge-frame-1024/summary.json
/tmp/stage324-chunked-bridge-frame-2048/summary.json
/tmp/stage324-spliced-chunked-bridge-512.summary.json
/tmp/stage324-spliced-chunked-bridge-1024.summary.json
/tmp/stage324-spliced-chunked-bridge-2048.summary.json
```

Local copies for analysis:

```text
artifacts/stage324_chunked_bridge_splice/bridge_frame_512.json
artifacts/stage324_chunked_bridge_splice/bridge_frame_1024.json
artifacts/stage324_chunked_bridge_splice/bridge_frame_2048.json
artifacts/stage324_chunked_bridge_splice/splice_512.json
artifacts/stage324_chunked_bridge_splice/splice_1024.json
artifacts/stage324_chunked_bridge_splice/splice_2048.json
```
