# Stage 308: Production Blockers For Wide PT-LX Restickify

## Summary

This stage made the streaming PT-LX value-flow contract report why a candidate
is or is not production-safe.  The branch already separated diagnostic success
from production proof; this follow-up makes the missing proof obligation
machine-readable.

## What Changed

`_streaming_value_flow_contract(...)` now reports:

```text
diagnostic_valid
production_valid
production_requirements
production_blocker
production_blocker_stage
production_required_primitive
production_required_lowering
```

For same-layout LX ownership remaps, `production_valid=true` and the blocker is
`None`.  For actual PT-LX layout transforms, the contract now explains that the
missing primitive is a remote-fragment-aware coordinate remap:

```text
STCDPOpLx/InputFetchNeighbor gather
  -> local PT/interslice tile transform
  -> STCDPOpLx/InputFetchNeighbor scatter or consumer LX write
```

Diagnostic force flags still make `diagnostic_valid=true`, but they now report:

```text
production_blocker = diagnostic-force-is-not-a-production-certificate
```

That keeps force-mode hardware experiments from being mistaken for a compiler
proof.

The streaming mixed-schedule patchers now use `production_valid`, not
`diagnostic_valid`, as the replacement gate.  A candidate that only passes
because of a force flag remains audit evidence and falls back to
`ReStickifyOpHBM`.

## Current Distance To Wide-Size Enablement

The wide-size PT-LX path is not blocked by endpoint plumbing anymore.  It is
blocked by semantic certification of the data transform:

- direct 64x64 PT-LX tiles need a proven remote-fragment coordinate map;
- native 64x64 PT-LX tiles need a consumer-fragment coordinate map;
- validGap consumer tiles need hardware value proof;
- plain STCDP gather/scatter does not by itself certify a stick-layout
  transform.

The production-shaped path should therefore build the data-shuffler chain:

```text
producer real LX fragments
  -> remote gather
  -> bounded local PT/interslice tile transform
  -> scatter/materialize consumer LX fragments
```

The stock `ReStickifyOpHBM` fallback remains unchanged unless a candidate
reports `production_valid=true`.

## Validation

Pod validation:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m py_compile \
  torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py \
  tests/inductor/test_restickify_lx_dataop.py

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  -q
```

Result:

```text
58 passed in 8.59s
58 passed in 4.12s after tightening the replacement gate
```

## Next Step

Generate the first real remote-fragment descriptor from the producer and
consumer SDSCs.  It should enumerate 64x64 tiles, list the producer LX fragments
needed by each tile, allocate bounded per-core workspace, and lower to the
gather -> local transform -> scatter chain above.  Until that descriptor can be
checked for value correctness, it must remain audit-only and fall back to
`ReStickifyOpHBM`.
