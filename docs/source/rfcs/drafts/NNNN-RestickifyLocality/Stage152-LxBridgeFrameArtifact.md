# Stage 152: LX Bridge Frame Artifact

## Summary

This stage produced a reusable compile-only LX-to-LX bridge frame for the
high-signal restickify case:

```text
computed_transpose_adds_then_matmul_tuple, size=2048
```

No hardware kernels were launched.  The goal was to package the already proven
schema-v4 materialization path into a frame artifact that can later be spliced
into the original fused Torch-Spyre runtime bundle.

The bridge frame represents:

```text
producer LX -> ReStickifyOpLx/STCDPOpLx data-op -> consumer LX
```

This is deliberately not yet production lowering.  It is the next artifact
needed after Stage 151 showed that split-launching producer, data-op, and
consumer as separate runtime bundles is not a valid validation model.

## Code Change

Added:

```text
tools/restickify_lx_bridge_frame.py
```

The tool:

1. reads a generated Torch-Spyre code directory with
   `restickify_lx_neighbor_edges.json`;
2. invokes `restickify_address_preserving_dataop_probe.py` to build a patched
   address-preserving data-op SDSC;
3. requires the schema-v4 `lx_materialization_contract`;
4. exports the data-op SDSC through the DeeRT data-op exporter;
5. copies `init.txt`, `senprog.txt`, and the patched SDSC into a standalone
   frame directory;
6. materializes `init_binary.bin` and a sentinel-cleared frame variant;
7. writes `summary.json` with the frame size, token counts, and contract fields.

## Validation

First generate the compile-only Torch-Spyre code directory:

```sh
SPYRE_RESTICKIFY_RING_TELEMETRY=1 \
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 \
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_SPLIT_PREPARE_ONLY=1 \
SPYRE_RESTICKIFY_LX_SPLIT_STOP_AFTER_BRIDGE=1 \
SPYRE_RESTICKIFY_LX_SPLIT_REQUIRE_MATERIALIZATION_CONTRACT=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --copy-kernel-code \
  --kernel-launch-log \
  --lx-split-dataop-prototype \
  --validate-tuple-prefix 1 \
  --output-dir /tmp/stage152-frame-prepare-2048 \
  --fail-on-error
```

Result:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple
restickifies=1 bytes=8388608 byte_hops=0 device_events=0
```

Then generate the bridge frame:

```sh
python tools/restickify_lx_bridge_frame.py \
  --code-dir /tmp/stage152-frame-prepare-2048/kernel_code/computed_transpose_adds_then_matmul_tuple_2048/0001_sdsc_fused_add_t_0 \
  --output-dir /tmp/stage152-lx-bridge-frame-2048-ok \
  --mode stage3b \
  --fail-on-hbm \
  --fail-on-missing-senprog
```

Result:

```json
{
  "status": "ok",
  "contract_source": "schema-v4-lx-materialization-contract",
  "frame_bytes": 17664,
  "frame_flits_128b": 138,
  "hbm_free": true,
  "tokens": {
    "HBM": 0,
    "L3LU": 96,
    "L3SU": 96,
    "LXLU": 64,
    "LXSU": 64,
    "PT": 0,
    "SFP": 0
  }
}
```

The output directory contains:

```text
init.txt
init_binary.bin
init_binary_sentinel_cleared.bin
init_sentinel_cleared.txt
sdsc_lx_bridge_dataop.json
senprog.txt
summary.json
deeprt_export/*
address_preserving_dataop/*
```

## Interpretation

This stage proves we can generate the bridge frame we need for same-artifact
packaging:

- the frame is built from a schema-v4 LX materialization descriptor;
- all 32 producer endpoint pieces and 32 consumer endpoint pieces were patched;
- the generated frame is 128-byte aligned;
- the bridge program contains no textual HBM instructions;
- the program does contain L3 ring-facing load/store tokens and local LX endpoint
  load/store tokens.

The `L3LU/L3SU` tokens are expected for the DeeRT data-op path.  They indicate
ring-facing movement through the L3/LX data-transfer machinery, while the
important negative check for this stage is `HBM=0`.

## Next Step

Use this frame artifact for same-artifact replacement:

1. rerun DXP debug on the original fused code directory to recover per-SDSC frame
   sizes;
2. replace the `ReStickifyOpHBM` frame inside the original
   `loadprogram_to_device/*/init.txt` with `init_binary_sentinel_cleared.bin`;
3. update `segment_size.json` and `spyreCodeDir` size metadata;
4. do a compile/package-only check first;
5. only after a clean package exists, attempt one hardware validation of the
   original fused bundle ordering.

That keeps the producer and consumer in the normal fused runtime context, which
is the lesson from Stage 151.
