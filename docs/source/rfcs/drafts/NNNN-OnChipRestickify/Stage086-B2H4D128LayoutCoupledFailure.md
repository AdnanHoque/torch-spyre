# Stage086 - B2H4D128 Layout-Coupled Failure

## Question

Why did `B2 H4 D128 block64` fail beyond the original L128/L256 gate island?

## Initial Result

The layout-coupled warpspec variant failed L384/L512 with the same mismatch
summaries as `onchip_hbm_kv_layout_xform`:

```text
B2 H4 L384 D128 block64:
  mismatched: 1110 / 393216
  max abs: 0.6083984375 at (1, 3, 9, 73)

B2 H4 L512 D128 block64:
  mismatched: 1109 / 524288
  max abs: 0.669921875 at (0, 3, 157, 1)
```

This looked baseline-limited until a layer probe separated the layout-transform
pair from the loader-core K/V prefetch path.

## Recovery

Disabling the layout-transform pair while keeping the serialized loader-core
K/V HBM prefetch recovers the whole tested long island:

```text
B2 H4 L384 D128:  ok, median 1.297006 ms, max abs 0.00317383, mixed 22
B2 H4 L512 D128:  ok, median 1.685478 ms, max abs 0.00463867, mixed 31
B2 H4 L768 D128:  ok, median 3.344942 ms, max abs 0.00390625, mixed 47
B2 H4 L1024 D128: ok, median 5.046582 ms, max abs 0.00256348, mixed 63
```

Each recovered row emitted the expected loader-specialized artifact:

```text
opFuncsUsed contains STCDPOpHBM
kv_repack_hbm_prefetch_hoist_role == "current_prefetch"
kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id == 31
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout == true
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces == true
kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch == true
```

## Interpretation

The failure was coupled to the layout-transform pair, not to the loader-core
prefetch schedule. The warpspec sweep variant should certify the loader-core
K/V prefetch invariant directly and leave layout-transform pair promotion as a
separate concern.
