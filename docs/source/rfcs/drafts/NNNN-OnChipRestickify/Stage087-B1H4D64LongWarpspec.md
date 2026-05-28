# Stage087 - B1H4D64 Long Warpspec Coverage

## Question

Does the serialized loader-core K/V HBM prefetch path keep working when the
`B1 H2 D64 block64` long-sequence case is broadened to four heads?

## Result

Yes, on the decoupled loader-specialized variant. The `B1 H4 D64 block64`
warpspec path is correct and loader-specialized through L1024:

```text
B1 H4 L384 D64: ok, median 0.882721 ms, max abs 0.00244141, mixed 30
B1 H4 L512 D64: ok, median 1.069644 ms, max abs 0.00488281, mixed 40
B1 H4 L768 D64: ok, median 1.742084 ms, max abs 0.00268555, mixed 59
B1 H4 L1024 D64: ok, median 2.378883 ms, max abs 0.00219727, mixed 78
```

Both rows emitted the expected loader-specialized artifact:

```text
opFuncsUsed contains STCDPOpHBM
kv_repack_hbm_prefetch_hoist_role == "current_prefetch"
kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id == 31
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout == true
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces == true
kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch == true
kv_repack_additional_consumers == ["22_maxnonstick:input1"]
```

## Layout-Coupled Failure

The earlier layout-coupled variant failed L768/L1024, and
`onchip_hbm_kv_layout_xform` failed with the same mismatch summaries:

```text
B1 H4 L768 D64 block64:
  mismatched: 58 / 196608
  max abs: 0.21875 at (0, 0, 397, 34)

B1 H4 L1024 D64 block64:
  mismatched: 74 / 262144
  max abs: 0.311279296875 at (0, 2, 589, 3)
```

The recovered no-layout-pair rows show that those failures were not caused by
the loader-core prefetch schedule itself.

```text
tile0:hoist:no_future_kv_candidate
auto:no_candidate_tiles
```

## Promotion Gate

The decoupled warpspec promotion gate covers the recovered long rows:

```text
b1h4d64_block64_long_decoupled_loader_core31
lengths = 768,1024
min_mixed_by_length = {768:59, 1024:78}
```

## Interpretation

The loader-core schedule broadens from two heads to four heads for the full
tested long-sequence island. Layout-transform pair promotion remains a separate
correctness problem; the AIU warpspec-like K/V prefetch path does not need to be
coupled to it.
