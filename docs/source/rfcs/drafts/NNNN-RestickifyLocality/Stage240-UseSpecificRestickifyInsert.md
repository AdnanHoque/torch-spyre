# Stage 240: Use-Specific Restickify Insert Probe

## Summary

This stage explored the smaller non-PT blocker identified in Stage 239:
`insert_restickify` rewrites consumer loads by buffer name.  For a consumer like
`u + u.t()`, the same producer buffer appears through two logical views.  A
name-only rewrite is too coarse because it redirects both uses to the
restickified buffer.

The new prototype carries the `MemoryDep.index` for the incompatible edge
through the restickify plan and can, behind a default-off flag, rewrite only the
matching logical use.

## Flag

```text
SPYRE_RESTICKIFY_USE_SPECIFIC_INSERT=1
```

Default behavior is unchanged.

## Code Changes

- `finalize_layouts` now records `dep_index` in each restickify-plan entry.
- `NameSwapHandler` can hold multiple rewrite rules per buffer name.
- With the flag off, rewrites keep the old name-wide behavior.
- With the flag on, a repeated-buffer consumer uses strict index matching so
  only the incompatible logical use is redirected.

## Probe

Case:

```python
def fn(a, b):
    u = a + b
    return u + u.t()
```

Size: `512`

Default mode:

```text
SPYRE_RESTICKIFY_USE_SPECIFIC_INSERT unset
result: DDL slice-size failure
audit: producer-endpoint-not-allocator-backed:prototype-default
streaming candidate: available
```

Use-specific mode:

```text
SPYRE_RESTICKIFY_USE_SPECIFIC_INSERT=1
result: DCG scheduler failure
DtException: There must be at least one valid candidate.
```

The generated OpSpecs no longer contain the malformed "both inputs are the
restickified buffer" shape.  Instead, they expose the next backend contract
problem: the consumer wants to read one LX-resident producer output through two
logical layouts.  That is closer to the desired internal value-flow contract,
but it is still not a valid scheduled program.

## Validation

Default regression:

```text
python -m pytest tests/inductor/test_restickify.py -q
97 passed
```

Focused scratchpad/PT-LX tests:

```text
python -m pytest \
  tests/inductor/test_scratchpad_patterns.py::TestExamplePattern::test_buf_analysis_uses_recomputed_read_writes_after_restickify \
  tests/inductor/test_scratchpad_patterns.py::TestExamplePattern::test_streaming_ptlx_endpoint_requires_internal_non_pt_consumer \
  tests/inductor/test_scratchpad_patterns.py::TestExamplePattern::test_certified_ptlx_restickify_edge_forces_only_endpoint_buffers -q
3 passed
```

## Interpretation

This does not complete the non-PT streaming PT-LX path, but it narrows the
failure:

- The compiler can now represent the idea of restickifying one logical use of a
  repeated buffer without changing every load of that buffer.
- The backend still needs either a real bridge data-op between the two logical
  views or a schedule contract that accepts the same LX allocation read through
  both layouts.

## Next Step

Use this flag only as a diagnostic lane.  The next production-shaped step is to
combine the use-specific consumer rewrite with an explicit streaming bridge
SDSC, rather than letting the consumer directly reinterpret the producer LX
allocation in two layouts.
