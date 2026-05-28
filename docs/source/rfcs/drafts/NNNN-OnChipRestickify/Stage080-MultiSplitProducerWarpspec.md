# Stage080 - Multi-Split Producer Warpspec Coverage

## Question

Why did the loader-core K/V HBM prefetch variant select for B1/D64 shapes but
not for B2/D64 or B1/D128, even though those shapes were numerically correct?

## Root Cause

The K/V prefetch hoist candidate detector assumed the low-core K/V producer had
exactly one split dimension:

```text
producer numWkSlicesPerDim_ -> single split dim only
```

That was true for the original B1/D64 island. The broader shapes use explicit
multi-dimensional producer work slices:

```text
B2 H2 D64:   producer split over mb and x
B1 H2 D128:  producer split over mb and x
```

The lower-level source-piece mapping already knew how to consume
`coreIdToWkSlice_` for multi-dimensional producer pieces. The gate failed
earlier, during candidate detection, with:

```text
input1:missing_layout_stick_or_split
```

## Implementation

The K/V repack broadcast edge resolver now accepts explicit producer split
dimensions as a list. It validates that:

```text
all producer split dims map into consumer iteration dims
each producer split factor divides the mapped consumer dimension
the product of producer split factors equals producer_num_cores
producer_num_cores remains lower than consumer_num_cores
```

Single-split metadata is kept as a string for compatibility. Multi-split
metadata is recorded as a list, for example:

```text
kv_repack_producer_split = ["mb_", "x_"]
kv_repack_mapped_split = ["x_", "in_"]
```

The logic test coverage now includes a four-core producer split over `mb` and
`x`, mapped into a 32-core consumer.

## Results

Focused local logic tests:

```text
163/163 passed
```

Focused pod tests after syncing the selector:

```text
327 passed in 2.81s
```

Exploratory hardware sweeps with
`onchip_warpspec_kv_hbm_prefetch_loader_core31`:

```text
B2 H2 L128 D64:  ok, median 0.447694 ms, max abs 0.00390625, mixed 8
B2 H2 L256 D64:  ok, median 0.856752 ms, max abs 0.00231934, mixed 16
B1 H2 L128 D128: ok, median 0.400670 ms, max abs 0.00512695, mixed 10
B1 H2 L256 D128: ok, median 0.610014 ms, max abs 0.00341797, mixed 20
```

All four rows emitted a serialized loader-core K/V prefetch artifact with:

```text
kv_repack_hbm_prefetch_hoist_role == "current_prefetch"
kv_repack_hbm_prefetch_hoist_prefetch_loader_core_id == 31
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout == true
kv_repack_hbm_prefetch_hoist_prefetch_loader_fanout_full_tile_pieces == true
kv_repack_hbm_prefetch_hoist_serialize_loader_core_prefetch == true
opFuncsUsed contains STCDPOpHBM
```

The warpspec promotion gate now includes these additional cases:

```text
B2 H2 L128,L256 D64 block64
B1 H2 L128,L256 D64 block64 causal
B1 H2 L128,L256 D128 block64
B2 H4 L128,L256 D128 block64
```

The expanded promotion gate then passed the full seven-case matrix:

```text
PROMOTION_GATE_PASSED gate=onchip_warpspec cases=7 rows=14
```

Per-row gate results:

```text
B1 H2 L128 D64:  ok, median 0.387292 ms, max abs 0.00341797, mixed 10
B1 H2 L256 D64:  ok, median 0.472078 ms, max abs 0.00292969, mixed 20
B1 H2 L128 D64 causal: ok, median 0.683108 ms, max abs 0.00585938, mixed 8
B1 H2 L256 D64 causal: ok, median 0.901131 ms, max abs 0.00488281, mixed 16
B2 H2 L128 D64:  ok, median 0.462752 ms, max abs 0.00585938, mixed 8
B2 H2 L256 D64:  ok, median 0.718888 ms, max abs 0.00317383, mixed 16
B1 H2 L128 D128: ok, median 0.446659 ms, max abs 0.00439453, mixed 10
B1 H2 L256 D128: ok, median 0.664996 ms, max abs 0.00585938, mixed 20
B2 H4 L128 D128: ok, median 0.631139 ms, max abs 0.00683594, mixed 8
B2 H4 L256 D128: ok, median 1.040170 ms, max abs 0.00634766, mixed 16
B1 H4 L128 D64:  ok, median 0.424590 ms, max abs 0.00585938, mixed 10
B1 H4 L256 D64:  ok, median 0.677655 ms, max abs 0.00317383, mixed 20
B1 H8 L128 D64:  ok, median 0.561142 ms, max abs 0.00439453, mixed 10
B1 H8 L256 D64:  ok, median 0.709759 ms, max abs 0.00439453, mixed 20
```

## Interpretation

The Stage079 negative coverage was not a hardware correctness failure. It was a
candidate-selection limitation: the loader-specialized artifact could not see
valid low-core K/V producers when their work was split across multiple logical
dimensions.

The current supported gate island is now:

```text
loader_core = 31
block_size = 64
length in {128, 256}

B1 H2 D64
B1 H2 D64 causal
B2 H2 D64
B1 H2 D128
B2 H4 D128
B1 H4 D64
B1 H8 D64
```

This is still an AIU-specific analogue of warp specialization rather than a
literal GPU warp-specialized kernel. The correctness invariant remains the same:
the loader core must not overlap its HBM prefetch data movement with its own
current attention compute slice, while the other cores may continue compute.
