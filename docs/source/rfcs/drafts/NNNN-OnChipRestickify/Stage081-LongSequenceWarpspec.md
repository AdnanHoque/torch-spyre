# Stage081 - Long-Sequence Warpspec Coverage

## Question

Does the serialized loader-core K/V HBM prefetch path remain selectable beyond
the initial L128/L256 promotion island?

## Result

The `onchip_warpspec_kv_hbm_prefetch_loader_core31` variant was swept for
non-causal `B1 H2 D64 block64` at longer sequence lengths:

```text
B1 H2 L384 D64:  ok, median 0.715492 ms, max abs 0.00341797, mixed 30
B1 H2 L512 D64:  ok, median 0.835817 ms, max abs 0.00195312, mixed 40
B1 H2 L768 D64:  ok, median 1.264168 ms, max abs 0.00183105, mixed 60
B1 H2 L1024 D64: ok, median 1.684362 ms, max abs 0.00158691, mixed 79
```

Each row emitted the expected loader-specialized artifact:

```text
kv_repack_hbm_prefetch_hoist_role == "current_prefetch"
kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id == 31
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout == true
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces == true
kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch == true
opFuncsUsed contains STCDPOpHBM
```

The warpspec promotion gate now treats `b1h2d64_block64_loader_core31` as a
long-sequence case:

```text
lengths = 128,256,384,512,768,1024
```

The expanded gate validated against the accumulated hardware rows:

```text
PROMOTION_GATE_PASSED gate=onchip_warpspec cases=7 rows=18
```

## Interpretation

The first loader-specialized path is not only a short-sequence artifact. For
the baseline non-causal B1/H2/D64 shape, the same schedule and metadata
invariant hold through L1024. The one-core local serialization cost is still
present, but the artifact selection and correctness behavior scale with tile
count.
