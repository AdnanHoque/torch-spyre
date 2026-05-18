# Stage 30: Mapping-Only Continuity And Hardware Bandwidth Notes

## Question

After Stage 29, the safe core-continuity rule was clear: do not trade away
parallelism for locality. This stage tested a narrower idea:

- Can we improve locality by changing only physical core mapping, while leaving
  work split factors unchanged?
- Does the Spyre knowledge base validate the hardware model we are using for
  ring-byte-hop reasoning?

## Implementation

Added a default-off flag:

```sh
SPYRE_ALIGN_CORE_MAPPING_CONTINUITY=1
```

This runs the existing certification path without enabling work-division
steering. It can only attach a mapping override when producer and consumer split
factors already match after symbol correspondence. It does not alter the number
of cores, split dimensions, tensor semantics, restickify count, or restickify
placement.

Restickify mapping-only was also tested separately with:

```sh
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=0
```

## Probe Set

Cases:

- `pointwise_transpose_add`
- `adds_then_matmul`
- `transpose_chain`
- `fanout_diamond`

Sizes:

- `1024`
- `1536`
- `2048`
- `3072`

Each mode ran with telemetry and timing. The runs completed with zero probe
errors.

Artifacts:

- `artifacts/stage30_mapping_only/raw`
- `artifacts/stage30_restickify_mapping_only/raw`

## Results

### Ordinary Producer-Consumer Mapping Only

| Mode | Rows | Aligned Edges | Positive Continuity-Hop Rows | Total Continuity Byte-Hops |
|---|---:|---:|---:|---:|
| Baseline | 52 | 0 | 24 | 1,601,961,984 |
| Mapping-only | 52 | 0 | 24 | 1,601,961,984 |

The pass found no eligible mapping-only continuity edges. In the tested graphs,
either the existing split was already local, the producer-consumer split factors
did not match, or the symbol correspondence was incomplete.

### Restickify Mapping Only

| Mode | Probe Rows | Restickifies | Bytes Moved | Ring Byte-Hops |
|---|---:|---:|---:|---:|
| Baseline | 16 | 24 | 204,472,320 | 266,993,664 |
| Restickify mapping-only | 16 | 24 | 204,472,320 | 266,993,664 |

The known high-signal `adds_then_matmul` cases did not improve when only the
core mapping flag was enabled. The important 2048 case stayed at:

| Case | Size | Baseline Byte-Hops | Mapping-Only Byte-Hops |
|---|---:|---:|---:|
| `adds_then_matmul` | 2048 | 67,108,864 | 67,108,864 |

This confirms that the Stage 3B win is not a pure physical-ID relabeling win.
It requires choosing the restickify work split on the corresponding logical
dimension first; only then can a physical mapping override make the ownership
local.

## Knowledge Base Validation

The Spyre knowledge base validates the main architectural assumptions behind
our byte-hop model:

- `wiki/entities/aiu.md`: AIU has 32 cores on a bidirectional ring. The ring is
  the sole path for off-chip-memory to scratchpad transfers, cross-core LX-LX
  transfers, and GTR multicast.
- `wiki/concepts/core-functional-units.md`: L3LU and L3SU are the ring-facing
  interfaces, and there is no direct off-chip-memory to core path that bypasses
  L3.
- `wiki/foundations/hardware/hardware-generations.md`: AIU 1.0 is a single-chip
  75W card with 128 GB LPDDR and about 170 GBps off-chip memory bandwidth. AIU
  1.5 moves to HBM3e with much higher bandwidth.
- `wiki/foundations/hardware/observability.md`: profiling observability is
  expected to flow through the profiling-toolkit/libaiupti path, but the
  knowledge base does not document a currently exposed per-restickify RIU
  traffic counter in the PyTorch trace path.

One note: `wiki/entities/aiu.md` also mentions "128 bytes/cycle (4 TB/s) per
direction". For the current AIU 1.0 measurements, the canonical numbers we
should use are the sysconfig-style numbers: 128 B/cycle/direction at 1.3 GHz,
or about 166 GB/s per direction. Treat the larger value as a different
aggregate interpretation unless hardware owners clarify it.

## Peak Bandwidth Reference

For the AIU 1.0 card used in these experiments:

| Fabric / Path | Peak |
|---|---:|
| RIU data ring, one direction | `128 B/cyc * 1.3 GHz = 166.4 GB/s` |
| RIU data ring, bidirectional aggregate | `332.8 GB/s` |
| Off-chip memory bus | about `166-170 GB/s` |
| RIU request ring | `1 B/cyc * 1.3 GHz = 1.3 GB/s` |
| SFPDataIU ring, per corelet direction | `32 B/cyc * 1.1 GHz = 35.2 GB/s` |
| SFPDataIU, both corelet rings | `70.4 GB/s` |
| LX port, per core | `128 B/cyc * 1.1 GHz = 140.8 GB/s` |
| LX ports, 32-core theoretical local aggregate | about `4.5 TB/s` |

The 32-core LX aggregate is not a global traffic number. It only applies when
each core is moving local data through its own LX port without being bottlenecked
by RIU, L3, or off-chip memory.

With multicast, the physical injection bandwidth does not multiply. A producer
still injects onto the RIU at the available physical ring rate. The win is
effective delivered bandwidth: one transmitted stream can satisfy many
consumers. If one producer stream at 166 GB/s reaches 31 other cores, the
logical delivered bandwidth can be viewed as roughly `31 * 166 = 5.1 TB/s`,
while physical ring injection remains bounded by the RIU path.

## Conclusion

Mapping-only continuity is too narrow to be useful on the tested graphs. The
compiler generally needs work-division steering to create matching split
factors before a physical core mapping override can eliminate modeled byte-hops.

The safe path remains:

1. Preserve core count and parallelism.
2. Only steer split dimensions when the alternative split uses the same number
   of cores.
3. Attach a certified physical mapping override only after the split factors
   match and modeled byte-hops become zero.

For future ring-aware projects, the KB strengthens the case for multicast-aware
constant/weight fanout: GTR multicast is a real hardware mechanism, and it can
turn N unicast-style deliveries into one physical stream with many logical
receivers.
