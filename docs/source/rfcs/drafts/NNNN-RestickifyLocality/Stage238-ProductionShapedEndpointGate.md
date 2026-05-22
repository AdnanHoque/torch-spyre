# Stage 238: Production-Shaped PT-LX Endpoint Gate

## Summary

This stage tightened the PT-LX restickify prototype around the production-shaped
contract we can defend today:

- Keep stock `ReStickifyOpHBM` as the fallback.
- Only force LX endpoints for an in-graph restickify edge when the compiler has
  a direct internal value-flow contract.
- Treat graph-output diagnostics, graph-input/weight sources, and PT/matmul
  consumers as fallback-only until a deeper Deeptools PT input contract exists.

The important change is not broad enablement. It is preventing the prototype
from partially allocating LX endpoints when the later bridge patch will skip.
That keeps fallback SDSCs clean and avoids leaving stock HBM restickify with
prototype-only LX metadata.

## Endpoint Rule

For uncertified streaming PT-LX candidates, scratchpad planning now requires:

```text
source kind = in_graph_computed
producer/restickify core counts match
shape is 64x64-tile compatible
restickify output has exactly one direct internal consumer
consumer is not a PT/matmul-like op
endpoint bytes fit the conservative LX budget
```

If these checks fail, no endpoint is forced and the generated program keeps the
ordinary HBM restickify path.

## Probe Results

`computed_transpose_adds_then_matmul_tuple`, size `512`:

```text
skip reason: not-stick-sized-and-not-streaming-candidate:consumer-count:0
```

This tuple probe returns the joined tensor as a graph output, so the restickify
output is not a clean internal bridge endpoint during scratchpad planning.
Keeping fallback is the right behavior.

`computed_transpose_join`, size `512`:

```text
skip reason: source-not-in-graph-computed
streaming candidate: available
total_transfer_bytes: 1,048,576
total_byte_hops: 1,376,256
```

This is a useful measurement shape, but not an eligible production PT-LX edge
because the source is not classified as an in-graph computed producer.

`adds_then_matmul`, size `2048`:

```text
skip reason: output-to-kernel-pt-consumer-mixed-schedule-unsafe
status: ok
restickify_count: 2
bytes: 16,777,216
```

The audit still reports the streaming opportunity:

```text
total_transfer_bytes: 16,777,216
total_byte_hops: 67,108,864
tile_size: 64
total_tiles: 1024
bounded_workspace_bytes: 24,576
```

But the generated hardware path stays on `ReStickifyOpHBM` because Stage 237
showed the mixed data-op -> PT consumer bundle can compile and launch but does
not retire reliably.

## Validation

Static/focused:

```text
python -m py_compile torch_spyre/_inductor/scratchpad.py
python -m pytest \
  tests/inductor/test_scratchpad_patterns.py::TestExamplePattern::test_certified_ptlx_restickify_edge_forces_only_endpoint_buffers \
  tests/inductor/test_restickify_lx_dataop.py::test_streaming_ptlx_patch_replaces_small_shape_hbm_restickify_boundary \
  tests/inductor/test_restickify_lx_dataop.py::test_ptlx_bridge_accepts_output_to_kernel_direction -q

3 passed
```

Broader PT-LX/restickify focused suite:

```text
python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_tile_ownership_probe.py \
  tests/inductor/test_restickify_mapping_alignment.py -q

62 passed
```

Hardware-facing guarded smoke:

```text
adds_then_matmul, size=2048, PT-LX flags enabled, skip correctness
status = ok
```

## Next Step

The next production-shaped increment is not to remove the HBM path. It is to
make one of these contracts real:

1. A scheduler-visible data-op-to-PT input contract so the PT consumer can read
   the bridge output directly.
2. Earlier producer output planning so the producer writes the tensor in the
   consumer-required layout and no restickify is emitted for the edge.

Until then, PT/matmul consumers must remain fail-closed on stock
`ReStickifyOpHBM`.
