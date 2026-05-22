# Stage 257: LX Neighbor Streaming Descriptor

## Goal

Convert the Stage256 inter-slice blocker into a production-shaped compiler
contract.

Stage256 showed that the compact PT-LX bridge can compile and launch, but it
does not produce correct values when the producer and restickify/consumer own
different physical LX regions.  The high-signal 2048 case is:

```text
producer output ownership:    mb:32,out:1
restickify output ownership:  mb:1,out:32
```

This stage does not yet emit a runtime data-op bridge.  Instead, it makes the
compiler sidecar describe the exact 64x64 remote-LX materialization that a
future lowering needs to generate.

## Code Changes

`restickify_lx_neighbor_edges.json` now includes a `streaming_ptlx` section in
the materialization contract when the producer and restickify endpoint metadata
are known.

The descriptor records:

- producer work-slice ownership;
- destination/restickify work-slice ownership;
- sampled source and destination core mappings;
- whether remote LX gather or scatter is required;
- a bounded 64x64 tile contract;
- tile counts, local/moving counts, max fan-in/fan-out, and modeled byte-hops;
- fallback path, which remains `ReStickifyOpHBM`.

The destination side intentionally uses the restickify output ownership, not the
following consumer payload.  The restickify output is the boundary we must
materialize before the consumer can safely read the tensor.

## Unit Validation

In the pod:

```text
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  tests/inductor/test_restickify_tile_ownership_probe.py \
  tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
58 passed in 8.70s
```

The focused descriptor test also covers both normal 2D logical tensor sizes and
the tiled 3D restickify representation:

```text
[tile_count, cols, tile_size]
```

For the 2048 case, `[32, 2048, 64]` is interpreted as a 2048x2048 logical
tensor.

## Real Compiler Descriptor Probes

Probe command shape:

```text
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size <size> \
  --skip-correctness \
  --skip-kernel-launch \
  --copy-kernel-code \
  --output-dir /tmp/stage257-lx-neighbor-descriptor-<size> \
  --fail-on-error
```

### Size 2048

```text
restickifies: 1
bytes:        8,388,608
source:       in-graph computed
```

Descriptor result:

```text
producer slices:      mb:32,out:1
destination slices:   mb:1,out:32
tile size:            64x64 fp16
total tiles:          1024
local tiles:          32
moving tiles:         992
max fan-in:           1
max fan-out:          1
modeled byte-hops:    67,108,864
bounded workspace:    24,576 bytes
remote LX gather:     yes
remote LX scatter:    no
```

This is the clean 2048 contract we wanted: each bridge tile has exactly one
source core and one destination core.  Most tiles move because the producer
owns rows while the restickify output owns columns.

### Size 512

```text
restickifies: 1
bytes:        524,288
source:       in-graph computed
```

Descriptor result:

```text
producer slices:      mb:32,out:1
destination slices:   mb:4,out:8
tile size:            64x64 fp16
total tiles:          64
local tiles:          0
moving tiles:         64
max fan-in:           4
max fan-out:          1
modeled byte-hops:    4,194,304
bounded workspace:    24,576 bytes
remote LX gather:     yes
remote LX scatter:    no
```

This is also a valid descriptor, but it is less clean than 2048.  A single
64x64 destination tile can span four producer row slices, so the bridge needs a
four-fragment gather.  That is why 512 is a harder production-lowering case
than the apparent smaller size suggests.

## Interpretation

We have moved from "PT-LX works for one hand-spliced case" to "the compiler can
derive the actual remote-LX materialization plan for real generated SDSCs."

What is done:

- identify the in-graph producer-to-restickify edge;
- recover source and destination LX ownership;
- compute a bounded 64x64 tile movement plan;
- distinguish local tiles from RIU-moving tiles;
- handle both 2048 and 512 ownership shapes;
- keep the HBM restickify as the fallback.

What is not done yet:

- emit the `InputFetchNeighbor`/`STCDPOpLx` bridge from this descriptor;
- patch the consumer boundary to read the materialized LX value;
- prove value correctness through the normal Torch-Spyre bundle;
- enable the path for a wide size range.

## Distance To Wide Size Enablement

The path is not ready to broadly enable yet.  The hardest compiler decision is
now visible, though: small sizes may require multi-fragment gathers per 64x64
tile, while 2048 has a simpler one-source-tile contract.

A plausible enablement sequence is:

1. support the 2048-style single-source/single-destination 64x64 contract first;
2. add value-correct hardware validation for the generated bridge;
3. extend to multi-fragment gathers like 512;
4. add legality gates for workspace, fan-in, fan-out, tile divisibility, and
   consumer compatibility;
5. only then consider replacing `ReStickifyOpHBM` by default.

## Artifacts

```text
artifacts/stage257_lx_neighbor_streaming_descriptor/
```

