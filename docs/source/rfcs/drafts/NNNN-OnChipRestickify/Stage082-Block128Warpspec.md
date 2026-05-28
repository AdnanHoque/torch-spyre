# Stage082 - Block128 Warpspec Coverage

## Question

Does the serialized loader-core K/V HBM prefetch path survive a larger SDPA
block size, or is it tied to block64 tiling?

## Results

The `onchip_warpspec_kv_hbm_prefetch_loader_core31` variant was swept for
non-causal `B1 H2 D64 block128`.

Proven loader-specialized rows:

```text
B1 H2 L256 D64 block128: ok, median 0.468861 ms, max abs 0.00390625, mixed 10
B1 H2 L384 D64 block128: ok, median 0.636375 ms, max abs 0.00341797, mixed 15
B1 H2 L512 D64 block128: ok, median 0.769282 ms, max abs 0.00213623, mixed 20
```

Both rows emitted:

```text
kv_repack_hbm_prefetch_hoist_role == "current_prefetch"
kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id == 31
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout == true
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces == true
kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch == true
opFuncsUsed contains STCDPOpHBM
```

Negative/partial coverage:

```text
B1 H2 L128 D64 block128: ok, but no loader-prefetch artifact
B1 H2 L768 D64 block128: failed numerical check after hoist was not realizable
B1 H2 L1024 D64 block128: failed numerical check after hoist was not realizable
```

For the failing long rows, the bundle warnings included:

```text
Requested K/V HBM prefetch hoist probe was not realizable
```

After the extra-consumer relaxation, the main long block128 SDPA bundle does
select the loader-prefetch artifact, but the numerical failure remains. The
non-warpspec `onchip_hbm_kv_layout_xform` baseline fails the same rows with the
same mismatch summaries:

```text
L768:  mismatched 63 / 98304, max abs 0.25390625 at (0, 0, 463, 51)
L1024: mismatched 2 / 131072, max abs 0.120361328125 at (0, 1, 728, 8)
```

The warpspec promotion gate therefore includes only the proven block128 island:

```text
b1h2d64_block128_loader_core31: lengths = 256,384,512
```

With this case added, the current warpspec promotion matrix validates as:

```text
PROMOTION_GATE_PASSED gate=onchip_warpspec cases=8 rows=21
```

## Interpretation

The loader-specialized mechanism is not fundamentally block64-only. The L384
miss was a candidate-selection issue: the hoisted producer's HBM output also
fed a later softmax reduction consumer. The prefetch hoist can preserve that
consumer because it still writes the same HBM address before omitting the
original future producer. The long-sequence fallback/numerical failure at
L768/L1024 remains open.
