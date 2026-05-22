# Stage 239: Internal Non-PT Consumer Probe

## Summary

This stage looked for a small non-PT pointwise graph where the production-shaped
streaming PT-LX gate could replace an in-graph `ReStickifyOpHBM` at size `512`.
The target was a direct internal edge:

```text
computed producer -> restickify -> pointwise consumer
```

This is intentionally separate from the PT/matmul path, which still fails closed
because Stage 237 showed the mixed bridge can compile and launch but not retire
reliably.

## Code Changes

- Fixed scratchpad bookkeeping to use names from `get_read_writes().reads`
  instead of `get_read_names()` when building `buf_users`.
  - `get_read_names()` can be stale after `insert_restickify` patches a
    consumer's `inner_fn`.
  - PT-LX endpoint planning needs the recomputed read/write dependencies because
    that is where the restickify output -> consumer edge is visible.
- Added a focused unit test for that stale-read-name case.
- Added small probe variants around computed `contiguous()` and `clone()` joins
  so we can see whether operand order changes which side gets restickified.

## Probe Results

Environment:

```text
LX_PLANNING=1
SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS=20
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
SPYRE_RESTICKIFY_PTLX_CROSS_BUNDLE_E2E=0
```

All probes below used `size=512` and `--skip-kernel-launch`.

| Case | Result | PT-LX audit |
|---|---|---|
| `computed_transpose_join` | compile ok | skipped: `source-not-in-graph-computed` |
| `computed_contiguous_then_add` | compile ok | skipped: `source-not-in-graph-computed` |
| `computed_contiguous_add_reversed` | compile ok | skipped: `source-not-in-graph-computed` |
| `computed_clone_then_add` | compile ok | skipped: `source-not-in-graph-computed` |
| `computed_clone_add_reversed` | compile ok | skipped: `source-not-in-graph-computed` |
| `computed_self_transpose_join` | DDL slice-size failure | skipped: `producer-endpoint-not-allocator-backed:prototype-default` |
| `computed_self_transpose_join3` | DDL slice-size failure | skipped: `producer-endpoint-not-allocator-backed:prototype-default` |

The computed `contiguous()`/`clone()` variants are useful negative controls: the
layout optimizer tends to restickify the graph input side of these joins, not the
computed producer side, so the production PT-LX gate correctly keeps the stock
path.

The self-transpose variants are closer to an in-graph source, but they expose a
different limitation: the same producer is consumed in original and transposed
forms inside one pointwise op. The existing `insert_restickify` rewrite maps by
buffer name, which is too coarse for "same buffer, different logical view"
patterns. That case is not a clean direct bridge endpoint yet.

## Validation

Focused scratchpad tests:

```text
python -m pytest \
  tests/inductor/test_scratchpad_patterns.py::TestExamplePattern::test_buf_analysis_uses_recomputed_read_writes_after_restickify \
  tests/inductor/test_scratchpad_patterns.py::TestExamplePattern::test_streaming_ptlx_endpoint_requires_internal_non_pt_consumer \
  tests/inductor/test_scratchpad_patterns.py::TestExamplePattern::test_certified_ptlx_restickify_edge_forces_only_endpoint_buffers -q

3 passed
```

PT-LX/restickify focused suite:

```text
python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_tile_ownership_probe.py \
  tests/inductor/test_restickify_mapping_alignment.py -q

62 passed
```

## Next Step

The next useful production-shaped increment is one of:

1. Extend `insert_restickify` to distinguish individual logical uses of the same
   buffer, so a graph like `u + u.t()` can restickify only the transposed use
   instead of rewriting every load of `u`.
2. Continue the PT/matmul route by replacing JSON-level patching with a real
   scheduler-visible data-op-to-PT input contract.

The second path is still required for the high-value `adds_then_matmul` family;
the first path gives us a smaller non-PT internal edge to validate the streaming
PT-LX bridge without involving PT.
