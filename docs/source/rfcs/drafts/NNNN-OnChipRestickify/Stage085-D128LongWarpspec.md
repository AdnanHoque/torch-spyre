# Stage085 - D128 Long-Sequence Warpspec Coverage

## Question

Does the serialized loader-core K/V HBM prefetch path keep working for the
`B1 H2 D128 block64` shape beyond the original L128/L256 island?

## Result

Yes. The `onchip_warpspec_kv_hbm_prefetch_loader_core31` variant now has
hardware coverage through L1024 for non-causal `B1 H2 D128 block64`.

```text
B1 H2 L384 D128:  ok, median 0.833297 ms, max abs 0.00341797, mixed 29
B1 H2 L512 D128:  ok, median 1.074050 ms, max abs 0.00280762, mixed 39
B1 H2 L768 D128:  ok, median 1.481457 ms, max abs 0.00244141, mixed 60
B1 H2 L1024 D128: ok, median 1.938134 ms, max abs 0.00158691, mixed 80
```

The L768/L1024 rows both emitted the expected loader-specialized artifact:

```text
opFuncsUsed contains STCDPOpHBM
kv_repack_hbm_prefetch_hoist_role == "current_prefetch"
kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id == 31
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout == true
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces == true
kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch == true
kv_repack_additional_consumers == ["22_maxnonstick:input1"]
```

## Promotion Gate

The warpspec promotion gate now treats `b1h2d128_block64_loader_core31` as a
long-sequence case:

```text
lengths = 128,256,384,512,768,1024
min_mixed_by_length = {128:10, 256:20, 384:29, 512:39, 768:60, 1024:80}
```

The expanded gate validated against the accumulated hardware rows:

```text
PROMOTION_GATE_PASSED gate=onchip_warpspec cases=8 rows=25
```

## Interpretation

The loader-specialized schedule is not tied to D64. With the multi-split
producer support and the extra-consumer-preserving prefetch hoist, the D128
shape shows the same serialized loader-core metadata invariant through L1024.
