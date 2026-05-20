# Stage 140: Consumer LX Metadata Compile Sweep

## Summary

After the Stage 139 bus fence, I avoided all hardware launches and moved the
next step offline. This stage adds a compile-only sweep for the consumer-side
LX input contract. The tool generates several single-SDSC consumer metadata
variants and runs `dxp_standalone --bundle`; it does not call `launch_kernel`.

The goal is to reduce the next reset-gated runtime attempt to a small set of
compile-clean candidates.

## Command

The sweep used the 2048 high-signal consumer SDSC from the descriptor-driven
split-runtime fixture:

```sh
python tools/restickify_consumer_lx_metadata_sweep.py \
  --consumer-sdsc /tmp/stage136-split-dataop-descriptor-2048/kernel_code/computed_transpose_adds_then_matmul_tuple_2048/0001_sdsc_fused_add_t_0/sdsc_2_add.json \
  --output-dir /tmp/stage140-consumer-lx-metadata-sweep \
  --target-lds-idx 1 \
  --lx-base 8192
```

Local summary copies were saved under:

```text
artifacts/stage140_consumer_lx_metadata_sweep/
```

## Results

| Variant | Compiled | Return | Init files | Notes |
|---|---:|---:|---:|---|
| `original_hbm` | yes | 0 | 1 | Baseline compile only. This is not a safe standalone launch proof after Stage 139. |
| `lx_only_output_corestate` | yes | 0 | 1 | LX-only allocation, keeps `dsType_=OUTPUT`, injects `coreStateInit_`. |
| `lx_only_input_corestate_primary` | yes | 0 | 1 | LX-only allocation, changes target to `INPUT`, copies primary input role, injects `coreStateInit_`. |
| `lx_hbm_present_output_corestate` | no | -6 | 0 | Fails in `L3DlOpsScheduler.cpp`: `Expect a valid HBM allocate node.` |
| `lx_only_output_no_corestate` | yes | 0 | 1 | LX-only allocation, keeps `dsType_=OUTPUT`, no injected `coreStateInit_`. |
| `lx_hbm_present_input_primary` | no | -6 | 0 | Same HBM allocate-node failure. |
| `lx_only_input_no_corestate_primary` | yes | 0 | 1 | LX-only allocation, changes target to `INPUT`, copies primary input role, no injected `coreStateInit_`. |

## Interpretation

The sweep did not find a DXP compile-time rejection for most LX-only consumer
contracts. That is useful, but it also means compile success alone is not a
sufficient safety signal: Stage 138 already showed that a compile-clean patched
consumer can still poison the stream at launch.

Two observations are actionable:

- Keeping both HBM and LX present while moving the allocation to LX is not a
  viable metadata shape for this path. Deeptools aborts while looking for a
  valid HBM allocation node.
- The next runtime candidates should be the no-`coreStateInit_` variants first.
  The Stage 138 failing consumer shape injected `coreStateInit_`, while
  `lx_only_output_no_corestate` and `lx_only_input_no_corestate_primary` still
  compile cleanly and have less synthetic metadata.

## Next Safe Gate

Do not launch more hardware kernels until the card has been reset and a known
good Torch-Spyre bundle has passed as a health check.

After reset:

1. Run a known-good unmodified Torch-Spyre generated bundle to verify the device
   is healthy.
2. Launch `original_hbm` only as a fixture sanity check if the known-good bundle
   passes.
3. Launch `lx_only_output_no_corestate` in a fresh process.
4. Launch `lx_only_input_no_corestate_primary` in a fresh process.
5. Stop immediately on any stream poison, RAS event, or hardware error.

Success for the next stage is not final performance. It is simply: the consumer
reads the LX-resident bridge tensor without a generated HBM reload and retires
cleanly on hardware.
