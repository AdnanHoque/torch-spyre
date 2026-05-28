# Stage079 - Warpspec Promotion Gate

## Question

Can the serialized loader-core K/V HBM prefetch path be promoted from a
standalone probe into a repeatable promotion gate?

## Implementation

Added a new promotion gate:

```text
tools/onchip_sdpa_promotion_gate.py --gate onchip_warpspec
```

The gate defaults to:

```text
onchip_warpspec_kv_hbm_prefetch_loader_core31
```

Unlike the existing `onchip_layout_xform` gate, this gate allows K/V repack
artifacts and requires the specific serialized loader-core prefetch metadata:

```text
kv_repack_hbm_prefetch_hoist_role == "current_prefetch"
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout == true
kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id == 31
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces == true
kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch == true
opFuncsUsed contains STCDPOpHBM
```

The first gate matrix covers the D64 single-batch shapes where the artifact has
been verified to select:

```text
B1 H2 L128,L256 D64 block64
B1 H4 L128,L256 D64 block64
B1 H8 L128,L256 D64 block64
```

## Results

Focused pod tests:

```text
326 passed in 2.11s
```

Promotion gate:

```text
PROMOTION_GATE_PASSED gate=onchip_warpspec cases=3 rows=6
```

Per-row hardware results:

```text
B1 H2 L128 D64: ok, median 0.395549 ms, max abs 0.00341797, mixed 10
B1 H2 L256 D64: ok, median 0.473572 ms, max abs 0.00292969, mixed 20
B1 H4 L128 D64: ok, median 0.435244 ms, max abs 0.00585938, mixed 10
B1 H4 L256 D64: ok, median 0.654960 ms, max abs 0.00317383, mixed 20
B1 H8 L128 D64: ok, median 0.530040 ms, max abs 0.00439453, mixed 10
B1 H8 L256 D64: ok, median 0.780867 ms, max abs 0.00439453, mixed 20
```

## Negative Coverage

The same sweep alias is numerically correct for the following shapes, but did
not emit the serialized loader-core K/V prefetch artifact:

```text
B2 H2 L128,L256 D64
B1 H2 L128,L256 D128
```

Those shapes should not be counted as warpspec coverage yet. The likely next
work is to relax or extend the K/V prefetch hoist candidate detection so the
loader-specialized artifact can select across batch > 1 and D128 layouts.

## Interpretation

The warp-specialized path now has a promotion target, not just a one-off probe.
The current supported island is:

```text
batch = 1
dim = 64
heads in {2, 4, 8}
length in {128, 256}
block_size = 64
loader_core = 31
```

The remaining architectural gap is generalization. Correctness is now proven
for this island; the next milestone is making the artifact select for broader
batch and head-dim layouts without silently falling back to the older on-chip
layout-xform path.
