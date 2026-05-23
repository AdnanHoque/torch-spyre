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

The default chunk mode is schedule-aware row chunking and can be controlled with:

```text
SPYRE_RESTICKIFY_PTLX_NATIVE_VALIDGAP_ENDPOINT_TILE_CHUNK_SIZE=<N>
```

Values:

- `0`: automatic row chunks, one complete 64x64 tile row per sidecar file.
- positive: fixed tile count per chunk.
- negative: keep the older single-SDSC full bridge.

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

Result: `65 passed`.

## 512 Real Sidecar Result

For `matmul_then_add`, size `512`, fixed 16-tile chunks emit 4 chunk files:

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

## Row-Chunk Probe

Automatic row chunking emits 8 files with 8 tiles and 32 dataops each. Every
chunk has a dense non-empty schedule for all advertised cores, so the earlier
`dsc_schedule.size() > 0` failure disappears.

| Chunks | Tiles/Chunk | DataOps/Chunk | DeeRT Export | HBM Tokens |
|---:|---:|---:|---|---:|
| 8 | 8 | 32 | 5 clean, 3 post-export segfaults | 0 for all emitted programs |

The remaining segfaults occur after `sdsc.json`, `senprog.txt`, `smc.txt`, and
`init.txt` are emitted. They look like a DeeRT export/metadata finalization
issue rather than a DCC lowering issue. The generated `senprog.txt` files still
show no `HBM` tokens.

The post-export segfault is nondeterministic. A retry loop over the 8 row chunks
with up to 3 attempts per chunk produced at least one clean rc=0 export for
every chunk:

```text
summary ok=8 fail=0
HBM=0 for every successful chunk
```

## Larger Shape Smoke

The same default row-chunk path was checked on larger `matmul_then_add` sizes
without launching hardware:

| Size | Chunk Files | Tiles/Chunk | DataOps/Chunk | Schedule Holes | Export Sample |
|---:|---:|---:|---:|---:|---|
| 1024 | 16 | 16 | 64 | 0 | chunks 0, 1, 8, 15 rc=0, `HBM=0` |
| 2048 | 32 | 32 | 128 | 0 | chunks 0, 1, 16, 31 rc=0, `HBM=0` |

The 2048 sample is important because this is the high-signal size where the
stock HBM restickify path was most interesting in earlier measurements. The
row-chunk sidecar now generates bounded, dense-schedule, no-HBM programs for
representative chunks at that size.

## Interpretation

Chunking is the right direction for the IBUFF problem, and the chunk boundary
does need to be schedule-aware. Row chunks are better than arbitrary fixed
chunks because they avoid empty per-core schedules.

The next blocker is not DCC lowering for the row chunks. It is making the
export/runtime packaging path deterministic enough to launch the generated
chunks as one bridge sequence, or bypassing the flaky sidecar export finalizer
and consuming the emitted `senprog.txt`/`init.txt` artifacts directly.

The stock `ReStickifyOpHBM` fallback remains the only runnable path. The PT-LX
sidecars are still diagnostic evidence.
