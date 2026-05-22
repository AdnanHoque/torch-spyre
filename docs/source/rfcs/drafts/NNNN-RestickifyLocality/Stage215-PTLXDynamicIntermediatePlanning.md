# Stage 215: PT-LX Dynamic Intermediate Planning

## Summary

This stage removes one hardcoded assumption from the PT-LX mixed restickify prototype:
the bridge no longer always places its temporary/intermediate LX buffer at a fixed
`1 MiB` per-core address.  Instead, the mixed-schedule lowering now plans a free
per-core LX range using the producer endpoint, consumer endpoint, and the bridge
intermediate piece size.

This is a step toward supporting more shapes.  It does not yet solve every split
shape: sizes whose producer pieces are smaller than a stick still need an extra
regroup/gather stage, and very large shapes may need streaming or in-place
tiling because producer + intermediate + consumer pieces do not all fit in a
single core's 2 MiB LX.

## Code Changes

- `generate_ptlx_restickify_bridge_sdsc()` now accepts
  `intermediate_start_address`.
- PT-LX mixed lowering computes bridge storage before emitting the mixed SDSC.
- Audit JSON now includes `bridge_storage` with producer, intermediate, consumer,
  LX limit, alignment, and skip reason.
- A benchmark-only force mode was added:
  `SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1`.
  It only accepts explicit producer and consumer endpoint bases from env vars.

## Validation

Focused tests:

```sh
python -m py_compile \
  torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/restickify_lx_dataop.py \
  torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py

python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
18 passed
```

Hardware smoke, normal fail-closed PT-LX mode:

| size | status | note |
|---:|---|---|
| 1536 | skipped | no allocator-backed endpoint in normal mode |
| 2048 | patched | dynamic intermediate planned at `524288..786432` |
| 4096 | skipped | no allocator-backed endpoint in normal mode |

Forced endpoint probes:

| size | forced endpoint result | note |
|---:|---|---|
| 1536 | skipped | `ptlx-piece-smaller-than-stick:producer-input:mb:split=32:max=24` |
| 2048 | patched | producer `0..262144`, intermediate `262144..524288`, consumer `524288..786432` |
| 4096 | skipped | `missing-intermediate-lx-space` |

## Interpretation

The 2048 case still works and still runs at about `1.02 ms` in the forced PT-LX
path.  The planner now explains why nearby shapes do not patch:

- `1536` is not an address-planning problem.  The producer split creates
  `48`-row pieces, smaller than the 64-element stick requirement.  Supporting it
  needs a pre-restickify regroup/fetch stage.
- `4096` is not a core-locality problem.  Each per-core piece is about `1 MiB`,
  so producer + intermediate + consumer would require about `3 MiB` per core in
  the current all-at-once two-step bridge.  Supporting it needs streaming,
  in-place planning, or a tiled bridge that does not require all three ranges to
  be live at once.

## Next Step

The next bridge-generalization stage should add one of:

- a regroup bridge stage for sub-stick producer splits, targeting the `1536`
  class, or
- a streaming/tiled intermediate plan for large pieces, targeting the `4096`
  class.

