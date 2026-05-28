# Stage083 - Extra HBM Consumer Hoist

## Question

Can the K/V HBM prefetch hoist accept a future K/V producer whose HBM output is
also read by a later softmax reduction SDSC?

## Root Cause

For `B1 H2 L384 D64 block128`, the loader-specialized path previously missed
because the target K/V producer was not single-consumer:

```text
input1:not_single_consumer:4_batchmatmul:input1,22_maxnonstick:input1
```

The same pattern appears on later tiles with `sumnonstick`. The old edge
resolver used the single-consumer invariant from the broadcast pair path.

## Implementation

The K/V broadcast edge resolver now has a default-off
`allow_additional_consumers` flag. The prefetch hoist path enables it; the
broadcast pair/plan path keeps the stricter single-consumer guard.

This is safe for the hoist because the generated hoisted producer is inserted
before the current tile and still writes the original HBM address. The target
future batchmatmul consumes the prefetched LX value, while later consumers can
continue to read the preserved HBM value.

The emitted metadata records preserved extra consumers:

```text
kv_repack_additional_consumers = ["22_maxnonstick:input1"]
```

## Results

Focused logic test:

```text
test_flash_kv_repack_hbm_prefetch_hoist_allows_preserved_extra_hbm_consumer
```

Hardware after the relaxation:

```text
B1 H2 L384 D64 block128: ok, median 0.636375 ms, max abs 0.00341797, mixed 15
```

The selected artifact had the expected serialized loader-core metadata:

```text
opFuncsUsed contains STCDPOpHBM
kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id == 31
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout == true
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces == true
kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch == true
kv_repack_additional_consumers == ["22_maxnonstick:input1"]
```

## Remaining Gap

The same relaxation does not recover `B1 H2 D64 block128` at L768/L1024. Those
rows still fail after the requested hoist is not realizable, so the long
block128 issue is distinct from the extra-consumer guard.
