# Stage 321: Chunked Native Valid-Gap Endpoint Sidecars

## Summary

Stage 321 changes the diagnostic native valid-gap endpoint sidecar from one
large SDSC into bounded tile chunks. This is a direct response to the Stage320
IBUFF failure:

```text
448 tiles/dataops shape: 49 tiles, 196 dataops -> DCC IBUFF current 146 > 128
512 tiles/dataops shape: 64 tiles, 256 dataops -> DCC IBUFF current 169 > 128
```

The chunked path is still default-off with the rest of the PT-LX sidecar stack.
When enabled, it writes separate sidecar JSON files:

```text
restickify_lx_neighbor_streaming_bridge_edge_<idx>_chunk<N>.json
```

The default chunk size is `16` tiles and can be controlled with:

```text
SPYRE_RESTICKIFY_PTLX_NATIVE_VALIDGAP_ENDPOINT_TILE_CHUNK_SIZE=<N>
```

A value `<= 0` keeps the older single-SDSC full bridge.

## Validation

```sh
python -m py_compile \
  torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/restickify_lx_dataop.py \
  torch_spyre/_inductor/codegen/lx_neighbor_descriptor.py \
  torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py \
  tests/inductor/test_restickify_lx_dataop.py
git diff --check
python -m pytest \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  tests/inductor/test_restickify_lx_dataop.py -q
```

Result: `64 passed`.

## 512 Real Sidecar Result

For `matmul_then_add`, size `512`, the sidecar now emits 4 chunk files with 16
tiles each:

| Chunk | Tiles | DataOps | DeeRT Export | HBM Tokens | Notes |
|---:|---:|---:|---:|---:|---|
| 0 | 16 | 64 | rc=139 | 0 | emits `senprog.txt`, then exporter segfaults during post-export metadata |
| 1 | 16 | 64 | rc=0 | 0 | clean export |
| 2 | 16 | 64 | rc=0 | 0 | clean export |
| 3 | 16 | 64 | rc=0 | 0 | clean export |

The key improvement is that the 512 path no longer hits the DCC IBUFF limit
when exported as separate chunks. The remaining issue is chunk 0's exporter
segfault after it has already emitted `sdsc.json`, `senprog.txt`, `smc.txt`,
and `init.txt`.

## Smaller Chunk Probe

A 4-tile chunk size produced 16 files. It did not improve reliability:

- some chunks returned rc=0 with no HBM tokens;
- some chunks failed DCC with `dsc_schedule.size() > 0`;
- some chunks emitted no-HBM `senprog.txt` and then segfaulted.

This means the next blocker is not only chunk size. It is likely tied to how
some tile groups map active core schedules, especially chunks that touch only a
subset of destination cores while the SDSC still advertises broader core
coverage.

## Interpretation

Chunking is the right direction for the IBUFF problem, but the chunk boundary
must be schedule-aware:

- each chunk should include a compact `numCoresUsed_` and remapped local core
  IDs, or
- each chunk should retain a dense schedule for every advertised core, or
- chunks should be grouped by destination/core bands that Deeptools can lower
  without empty dataflow schedules.

The stock `ReStickifyOpHBM` fallback remains the only runnable path. The PT-LX
sidecars are still diagnostic evidence.
