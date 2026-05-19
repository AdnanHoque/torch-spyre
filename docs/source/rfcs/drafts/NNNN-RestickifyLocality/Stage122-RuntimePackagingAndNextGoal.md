# Stage 122: Runtime Packaging And Torch-Spyre LX Goal

## Objective

Move the LX-to-LX restickify work from standalone compiler proofs toward a
Torch-Spyre runtime prototype:

```text
producer add -> LX-to-LX restickify / neighbor movement -> consumer add
```

The target success criterion is stricter than previous stages: a Torch-Spyre
generated fused bundle should retire on hardware without `ReStickifyOpHBM`,
without stream hardware error, and with value correctness.

## What Now Works

Torch-Spyre can emit a sidecar descriptor for an eligible
producer/restickify/consumer edge, and the runtime probe can consume that
descriptor to generate an `InputFetchNeighbor` program frame.

The new runtime splice path:

1. finds `restickify_lx_neighbor_edges.json` beside a generated bundle;
2. stages the descriptor through the standalone `InputFetchNeighbor` adapter;
3. uses `dcg_inpfetch_senprog_probe.cpp` to materialize `init.txt` and
   `init_binary.bin`;
4. replaces the original restickify frame in
   `loadprogram_to_device/<kernel>-SenProgSend/init.txt`;
5. updates the runtime artifact metadata.

Validation command:

```sh
python3 tools/restickify_lx_neighbor_runtime_probe.py \
  --case computed_transpose_adds_then_matmul \
  --size 2048 \
  --consumer-core-map reverse \
  --skip-correctness \
  --output-dir /tmp/stage122-runtime-reverse2
```

Result:

```text
status              ok
patch_count         1
original frame      18,816 bytes
patched frame       16,640 bytes
original restickify 7,296 bytes
InputFetchNeighbor  5,120 bytes
hardware stream     retired without stream error
correctness         skipped
```

This proves that a descriptor-generated HBM-free movement frame can be packaged
into the same runtime artifact shape as a normal Torch-Spyre fused bundle.

## What Does Not Work Yet

### InputFetchNeighbor Is Not A Restickify

The identity-core runtime probe retired but failed correctness:

```text
mismatched elements 3,475,977 / 4,194,304
max abs diff        3.265625
```

That is expected in hindsight. `InputFetchNeighbor` can move an already
compatible LX layout between cores, but it does not perform the stick-layout
conversion that restickification requires.

### Split Data-Op Launch Is Still Not A Valid Boundary

The split prototype now retags the consumer's LX boundary input as `INPUT`,
matching the fused-boundary experiment. The consumer launch reaches
`after_consumer`, but the stream reports a hardware error before the following
matmul can launch.

This keeps the earlier conclusion intact: separate kernel launches are not a
safe way to pass an ephemeral LX allocation as a tensor boundary. The LX path
must be packaged as a fused/internal edge, or lowered through a first-class
runtime contract that explicitly preserves the producer/consumer LX buffers.

### DDL Bridge Retires But Is Not Value Correct

For `computed_transpose_adds_then_matmul`, replacing the in-bundle
`ReStickifyOpHBM` with the compact DDL bridge can retire on hardware after
retagging the consumer LX input. It still fails value correctness:

```text
mismatched elements about 83% to 85%
```

Sweeping compact output bases (`0`, `8192`, `16384`, `32768`) did not fix the
error. Matching only the corelet-0 producer/consumer scheduled addresses also
did not fix it.

A full endpoint-map stitch, where the bridge input used the producer scheduled
`sfp -> lxsu` map and the bridge output used the consumer scheduled
`lxlu -> sfp` map, failed earlier in DXP with:

```text
Different cardinality between json and caller
```

That suggests the hand-patched DDL bridge still lacks a first-class internal
data-location contract. Allocation-node patching alone is not enough.

## Important Discovery

The strongest Stage 3B telemetry case remains `adds_then_matmul` at `2048`:

| Mode | Restickifies | Exact In-Graph Byte-Hops |
|---|---:|---:|
| baseline | 2 | `67,108,864` |
| Stage 3B | 2 | `0` |

However, the physical runtime bundles are split:

```text
bundle 1: graph-input restickify + adds
bundle 2: in-graph restickify + matmul
```

The high-signal Stage 3B restickify is in the matmul bundle, while its producer
is in the previous bundle. That means the current generated runtime artifact no
longer has the producer LX allocation available as a same-bundle internal edge.

This is the main reason the next prototype cannot simply replace a
`ReStickifyOpHBM` SDSC with a local bridge in-place. For the high-signal case,
we either need:

1. a fused runtime artifact spanning producer -> restickify -> consumer;
2. a real cross-bundle LX-resident handoff contract; or
3. a layout-aware lowering that prevents the HBM boundary from being introduced
   before the matmul bundle.

## Next Implementation Target

The next stage should stop treating LX addresses as JSON patches and instead
model an internal edge explicitly:

```text
producer output data-location:
  memory_space = lx
  layout       = producer output stick layout
  core_set     = producer core ownership
  lifetime     = dominates restickify/consumer

restickify bridge:
  input  aliases producer output location
  output aliases consumer input location

consumer input:
  reads the bridge output location as an internal INPUT
```

The Knowledge Base schedule-IR notes describe exactly this kind of object:
`memory_space`, `layout_order`, `core_set`, `transfer`, allocation lifetime, and
implicit producer-consumer synchronization. The current prototype is missing
that object and is trying to infer it after DXP has already made private
scheduling choices.

## Practical Next Step

The branch now has a conservative bundle-level guard before any DDL bridge
replacement:

- only emit the bridge when producer, restickify, and consumer are adjacent in
  the same bundle;
- require a locality certificate;
- require a materialized internal-edge descriptor;
- otherwise leave `ReStickifyOpHBM` unchanged.

Validation on `adds_then_matmul`, size `2048`, with DDL bridge enabled and
Stage 3B flags enabled now succeeds by skipping the impossible cross-bundle
replacement:

```text
status            ok
restickify_count  2
DDL audit rows    skipped, reason=mixed-kernel-bundle
bundle 1 files    sdsc_0_ReStickifyOpHBM.json, sdsc_1_add.json, sdsc_2_add.json
bundle 2 files    sdsc_0_ReStickifyOpHBM.json, sdsc_1_batchmatmul.json
```

Then build a minimal same-bundle fixture around that descriptor. The first value
correct case can be zero-hop; after that works, extend the same mechanism to the
high-signal Stage 3B case, which currently crosses a bundle boundary.
