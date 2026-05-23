# Stage 320: Native Valid-Gap Endpoint Scatter

## Summary

Stage 320 fixes the real-consumer endpoint shape found after Stage319. The
previous native valid-gap endpoint bridge wrote the adapter directly into the
consumer fragments. That worked for standalone 64-wide endpoint fragments, but
the real `matmul_then_add` consumer owns smaller fragments. Deeptools rejected
that shape because each output piece on stick dim `in_` was narrower than the
64-wide stick size.

The new diagnostic bridge is:

```text
STCDPOpLx gather
ReStickifyOpWithPTLx native local tile transform
ReStickifyOpWithPTLx valid-gap endpoint adapter into a full 64x64 tile workspace
STCDPOpLx same-stick scatter into the actual consumer fragments
```

This keeps the stock `ReStickifyOpHBM` fallback. The PT-LX path is still a
sidecar only and remains default-off behind:

```text
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED=1
SPYRE_RESTICKIFY_LX_NEIGHBOR_STREAMING_BRIDGE=1
SPYRE_RESTICKIFY_PTLX_NATIVE_VALIDGAP_ENDPOINT_TILE_E2E=1
```

## Validation

Static/unit validation:

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

Result: `63 passed`.

## Real Sidecar Export Sweep

All rows are `matmul_then_add` with normal Torch-Spyre sidecar generation and
kernel launch skipped. The emitted sidecar coalescing is
`native-validgap-endpoint-scatter-64x64-tiles`.

| Size | Tiles | DataOps | DeeRT Export | HBM Tokens | Notes |
|---:|---:|---:|---:|---:|---|
| 128 | 4 | 16 | rc=0 | 0 | `LXLU=4`, `LXSU=4`, `SFP=58`, `PT=272` |
| 256 | 16 | 64 | rc=0 | 0 | `LXLU=8`, `LXSU=8`, `SFP=116`, `PT=544` |
| 384 | 36 | 144 | rc=0 | 0 | `LXLU=12`, `LXSU=12`, `SFP=174`, `PT=816` |
| 448 | 49 | 196 | rc=134 | n/a | DCC IBUFF limit: max 128, current 146 |
| 512 | 64 | 256 | rc=134 | n/a | DCC IBUFF limit: max 128, current 169 |

The earlier Deeptools shape assertion is gone. The current blocker is code
size/IBUFF pressure from emitting every 64x64 tile as a separate four-dataop
sequence in one SDSC.

## Interpretation

This is meaningful progress but not production-ready:

- real Torch-Spyre sidecars can now export through DeeRT for sizes up to 384
  without HBM tokens;
- the shape contract now handles real consumer fragments by adding the final
  scatter stage;
- 448 and 512 show the next production-shaped issue: the bridge must be tiled
  into smaller executable chunks, folded more compactly, or lowered with a less
  unrolled Deeptools contract.

Next step: reduce IBUFF pressure by splitting the sidecar into per-row or
bounded-tile SDSCs, or by changing the full bridge schedule so DCC does not
materialize all tile transfers in one lxsu program.
