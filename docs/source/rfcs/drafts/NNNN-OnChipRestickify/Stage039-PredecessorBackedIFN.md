# Stage 039: Predecessor-Backed IFN Pair Probe

## Summary

Stage038 proved that a single-SDSC InputFetchNeighbor-shaped flash overlap
artifact can codegen but is not runtime-safe: it times out without a real
predecessor producer.  Stage039 adds an explicit two-SDSC contract for the next
probe:

- a predecessor sidecar with the producer output LX-pinned;
- a consumer sidecar with input `lds0` LX-pinned;
- a Torch-authored `STCDPOpLx` copy from the predecessor LX output region to
  the consumer LX input region;
- an explicit copy-then-compute consumer schedule:
  `[[0, -1, 0, 1], [-1, 0, 1, 0]]`.

The first DXP predecessor-backed IFN lowering experiment used bundle attrs on
the consumer:
`ifn_enable`, `ifn_predecessor="prev_sibling"`, and
`ifn_predecessor_sdsc_filename=...`.  That lowered, but did not produce correct
data.  The working Stage039 path intentionally does not emit those attrs; bundle
order plus the explicit STCDP copy carries the predecessor value.

The Torch gate is default-off:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PAIR_TILE=-1
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE=-1
```

When enabled, the builder only emits the pair for a strict latest-producer,
single-consumer, same-physical-layout edge.  The real SDPA `L=128` probe stayed
fail-closed: tile 2 has the apparent producer relation, but needs a layout
transform rather than a same-physical copy:

```text
producer=['mb_', 'x_', 'out_']/out_
consumer=['x_', 'mb_', 'in_']/in_
```

## Deeptools Pod Patch

The pod Deeptools patch stack now includes an experimental bundle predecessor
path:

- `dxp/dxp.cpp` resolves IFN predecessor attrs from `SdscNode::bundleOp`.
- `Dxp::runDxpOnSdsc` and `Dxp::runCodegen` thread an optional `SuperDsc*`.
- `DcgManager::runDcgForDataOpsDlOps` passes that pointer into
  `generatePcfgIRForDataOpInpFetch`.
- `generatePcfgIRForDataOpInpFetch` fills the scheduled placeholder dataop when
  a predecessor-backed bundle path supplies `datadscIdx`.
- The two-SDSC IFN verifier was relaxed like the single-SDSC verifier: HBM
  non-neighbor tensors and internal predecessor outputs are no longer rejected
  solely for not being LX-pinned.

`dxp_standalone` rebuild passed after each patch.

## Results

Real SDPA fail-closed control:

```text
L=128 ifn_pair_tile1 status=ok
median=0.273274ms max_err=0.00341797
```

No IFN pair sidecars were emitted for that run.  Cache-side rejection scan:

```text
tile 0 input0:no_latest_producer
tile 1 input0:no_latest_producer
tile 2 input0:layout_transform_required:producer=['mb_', 'x_', 'out_']/out_:consumer=['x_', 'mb_', 'in_']/in_
```

That rejection now has a separate default-off Torch probe:
`SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE=2`, exposed in
`tools/onchip_sdpa_sweep.py` as `layout_xform_pair_tile2`.  It emits the same
two ordered sidecars, but the consumer copy data-op describes the source as the
producer physical order with the stick dim aliased from `out_` to `in_`:

```text
src layout=['mb_', 'x_', 'in_'] stick=in_ LX=16384
dst layout=['x_', 'mb_', 'in_'] stick=in_ LX=8192
schedule=[[0, -1, 0, 1], [-1, 0, 1, 0]]
```

This is intentionally not folded into the same-physical IFN-pair gate.  It is a
device-validation probe for whether ordinary `STCDPOpLx` can reindex a
producer-described LX payload into the consumer's dim names and work slices.  If
it fails, the next contract has to use a certified `ReStickifyOpWithPTLx` path
or Deeptools dim-relation support.

The first real SDPA `L=128` run showed that the producer is the preceding
`ReStickifyOpHBM`, not a single-split BMM output.  Its work division is
multi-split (`mb:2, x:2`) and its physical extents map positionally to the
consumer:

```text
producer layout=['mb_', 'x_', 'out_'] N={mb_:2, x_:128, out_:64}
consumer layout=['x_', 'mb_', 'in_'] N={x_:2, mb_:128, in_:64}
dim map: mb_->x_, x_->mb_, out_->in_
source pieces: 4 producer work slices
destination pieces: 32 consumer work slices
```

After teaching the probe to build source `PieceInfo` from the producer's
`coreIdToWkSlice_`, the real sweep produced and executed the pair:

```text
L=128 layout_xform_pair_tile2 status=ok
median=0.260946ms mean=0.260946ms max_err=0.00341797 mixed=5
cache=/tmp/sdpa-stage039-layout-xform-pieces-layout_xform_pair_tile2-B1-H2-L128-D64-593274-387548
```

The cache contains:

```text
sdsc_mixed_flash_layout_xform_pair_tile_2_predecessor.json
sdsc_mixed_flash_layout_xform_pair_tile_2_consumer.json
bundle.mlir lines:
  sdsc_mixed_flash_layout_xform_pair_tile_2_predecessor.json
  sdsc_mixed_flash_layout_xform_pair_tile_2_consumer.json
senprog summary:
  consumer HBM=0 LX_LDSTU=36 PT=4352 SFP=96
```

The same gate also passed an `L=256` smoke run:

```text
L=256 layout_xform_pair_tile2 status=ok
median=0.428628ms mean=0.428628ms max_err=0.00292969 mixed=9
cache=/tmp/sdpa-stage039-layout-xform-pieces-l256-layout_xform_pair_tile2-B1-H2-L256-D64-593581-25170
```

Synthetic chained matmul probe:

```python
def chain(a, b, c):
    return torch.matmul(torch.matmul(a, b), c)
```

With `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PAIR_TILE=1`, the first DXP
predecessor-backed branch generated:

```text
sdsc_mixed_flash_ifn_pair_tile_1_consumer-LxInputNeighborFetch
transfer_lds0_src:no_component_dst:no_component_lx_neighbor
sync_soft_send_l3lu_to_lxlu
sync_soft_receive_lxlu_from_l3lu
```

The compile/codegen/DCC path completed, then runtime timed out at
`RuntimeStream::synchronize`.  Adding the missing consumer L3 PCFG made the
program return, but the result was incorrect because the generated IFN transfer
did not materialize a correct LX-to-LX copy for the same-core predecessor edge.

The working cache used the same two sidecars but compiled the consumer as an
ordinary two-step mixed SDSC:

```text
sdsc_mixed_flash_ifn_pair_tile_1_predecessor
sdsc_mixed_flash_ifn_pair_tile_1_consumer
  step 0: STCDPOpLx copy producer LX 16384 -> consumer LX 8192
  step 1: batchmatmul consumes lds0 from LX 8192
```

Synthetic chained-matmul correctness matched the non-IFN baseline:

```text
max_err=1.5 mean_err=0.23056954145431519 allclose_loose=True
```

## Next

The next useful split is:

- upstream the explicit LX-copy sidecar path as the default-off safe probe;
- keep the DXP predecessor-generated IFN path as a separate Deeptools follow-up;
- extend the layout-xform sweep across more lengths, block sizes, and batch/head
  shapes;
- decide whether the now-proven layout-transforming pair can graduate from
  explicit probe gate to a production-candidate on-chip SDPA path.
