# Stage 039: Predecessor-Backed IFN Pair Probe

## Summary

Stage038 proved that a single-SDSC InputFetchNeighbor-shaped flash overlap
artifact can codegen but is not runtime-safe: it times out without a real
predecessor producer.  Stage039 adds an explicit two-SDSC contract for the next
probe:

- a predecessor sidecar with the producer output LX-pinned;
- a consumer sidecar with input `lds0` LX-pinned and the IFN marker transfer;
- bundle attrs on the consumer:
  `ifn_enable`, `ifn_predecessor="prev_sibling"`, and
  `ifn_predecessor_sdsc_filename=...`.

The Torch gate is default-off:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PAIR_TILE=-1
```

When enabled, the builder only emits the pair for a strict latest-producer,
single-consumer, same-physical-layout edge.  The real SDPA `L=128` probe stayed
fail-closed: tile 2 has the apparent producer relation, but the physical layouts
are ordered differently:

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
tile 2 input0:physical_layout_mismatch:producer=['mb_', 'x_', 'out_']/out_:consumer=['x_', 'mb_', 'in_']/in_
```

Synthetic chained matmul probe:

```python
def chain(a, b, c):
    return torch.matmul(torch.matmul(a, b), c)
```

With `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PAIR_TILE=1`, DXP generated the
predecessor-backed branch:

```text
sdsc_mixed_flash_ifn_pair_tile_1_consumer-LxInputNeighborFetch
transfer_lds0_src:no_component_dst:no_component_lx_neighbor
sync_soft_send_l3lu_to_lxlu
sync_soft_receive_lxlu_from_l3lu
```

The compile/codegen/DCC path completed, then runtime timed out at
`RuntimeStream::synchronize`.  This is progress relative to Stage038: the
timeout now comes after a real predecessor-backed IFN data-op is generated, not
from the single-SDSC placeholder path.

## Next

The next useful split is:

- inspect the generated chained-matmul IFN PCFG/senprog for unmatched soft sync
  or missing producer/consumer ordering;
- decide whether IFN needs bundle-level runtime dependency metadata beyond
  lowering the consumer data-op;
- keep SDPA fail-closed until a same-physical producer edge exists or a
  deliberate layout-transforming IFN path is designed.
