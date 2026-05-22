# Stage 246: Local-Tile PT-LX Contract

## Summary

This stage narrows the streaming PT-LX restickify work to the smallest native
Deeptools transform contract we can certify: one local LX tile on one bridge
core, using `ReStickifyOpWithPTLx` to switch the stick dimension from `out_` to
`j_`.

This is not the full production bridge. It is the middle phase that must be
surrounded by same-stick gather and scatter movement when producer and consumer
cores differ.

## Result

Added `generate_ptlx_local_tile_restickify_sdsc()`, which emits a single
data-op SuperDSC with:

- `op = ReStickifyOpWithPTLx`
- dimensions `j_, i_, out_, mb_`
- input stick dimension `out_`
- output stick dimension `j_`
- LX-only input/output placements
- one bridge core in `coreIdToDscSchedule`
- explicit `streamingPTLXLocalTile_` metadata marking the local semantic
  transform contract as certified

The generated payload compiles through Deeptools:

```sh
/opt/ibm/spyre/deeptools/bin/dcg_standalone \
  -initSdsc /tmp/stage246-local-tile/local_tile.json \
  -d /tmp/stage246-local-tile/dcg
```

Observed output:

```text
Computing Re-StickifyOpWithPT (Special re-stickify) transfer function..
Creating PCFG for DataDsc..
Writing DataDsc to /tmp/stage246-local-tile/dcg/sdsc.json..
Writing PCFG to /tmp/stage246-local-tile/dcg/pcfg.json..
```

## Interpretation

This confirms that Deeptools accepts a native local PT-LX restickify transform
shape. The earlier streaming bridge failed semantic certification because it
used same-stick `STCDPOpLx` gather/scatter descriptors as if they could also
certify the transpose/remap. This stage separates the concerns:

- `STCDPOpLx` / neighbor movement: same-stick data movement
- `ReStickifyOpWithPTLx`: local stick-dimension transform
- Torch-Spyre integration: still responsible for generating a correct
  producer -> gather -> local PT transform -> scatter -> consumer value flow

## Validation

```sh
python -m py_compile \
  torch_spyre/_inductor/codegen/restickify_lx_dataop.py \
  tests/inductor/test_restickify_lx_dataop.py

python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
```

Pod result:

```text
27 passed in 0.47s
```

## Remaining Work

This stage does not yet enable PT-LX restickify for a wide array of sizes. The
remaining production-shaped work is:

1. Generate legal same-stick gather/scatter descriptors around this local
   transform.
2. Tile large tensors so each bridge core uses bounded LX workspace.
3. Preserve the normal producer/restickify/consumer bundle contract without
   falling back to `ReStickifyOpHBM`.
4. Prove value correctness on hardware across several sizes before collecting
   performance numbers.
