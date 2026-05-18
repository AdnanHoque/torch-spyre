# Stage 45: DDL Bridge Source Gate

## Summary

Stage 44 showed that the selective DDL bridge could compile inside a mixed
bundle, but the 2048 correctness run failed. Stage 45 identifies the first
correctness bug: the bridge was being applied to a graph-input restickify.

That is invalid for the current DDL bridge contract. The DDL bridge models
LX-local movement, so the input must already be produced inside the compiled
graph and resident on the producer side of the fused dataflow. A graph input or
weight starts outside the compiled graph; treating it as an LX-local source can
read the wrong data.

## Code Change

`insert_restickify` now annotates each inserted restickify buffer with:

```text
restickify_source_name
restickify_source_kind
```

The source kind is currently:

- `in_graph_computed` for a `ComputedBuffer` source,
- `graph_input_or_weight` for an `InputBuffer` source,
- `unknown` or the underlying buffer class name otherwise.

`SpyreKernel` carries those fields into `OpSpec.op_info`, and the default-off
DDL bridge now requires:

```text
restickify_source_kind == "in_graph_computed"
```

The DDL audit JSONL also records `source_name` and `source_kind`.

## Correctness Probe

Command:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1 \
SPYRE_RESTICKIFY_DDL_BRIDGE_AUDIT_JSONL=/tmp/stage45-ddl-source-gate-correctness-2048/audit.jsonl \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 2048 \
  --ring-telemetry \
  --output-dir /tmp/stage45-ddl-source-gate-correctness-2048 \
  --fail-on-error
```

Result:

```text
ok size=2048 case=adds_then_matmul restickifies=2 bytes=16777216 byte_hops=67108864
```

The DDL bridge emitted no replacement SDSC. The audit rows explain why:

```json
{"source_name":"arg1_1","source_kind":"graph_input_or_weight","reason":"source-not-in-graph-computed","status":"skipped"}
{"source_name":"buf1","source_kind":"in_graph_computed","reason":"output-stick-is-not-split-dim","status":"skipped"}
```

This recovers correctness by refusing the unsafe graph-input bridge.

## Interpretation

This stage is not a performance win. It is a correctness boundary.

What we learned:

- The compile-success DDL direction from Stage 44 was not the high-signal
  producer-to-restickify edge. It was a graph-input restickify.
- Graph-input and weight restickifies are not valid LX-local bridge candidates
  unless we also solve input/weight placement, prepacking, or load-side layout
  management.
- The real high-signal in-graph restickify is still the mirrored direction and
  remains blocked by the DDL/DCC register-boundary failure.

So the remaining core-to-core proof target is narrower and cleaner:

```text
source_kind=in_graph_computed
producer split:   d1:32
restickify split: d0:32
current blocker:  output-stick-is-not-split-dim / LXLU register-boundary failure
```

## Next Step

The next task should focus only on the mirrored in-graph DDL contract:

1. Build a tiny DDL fixture with an in-graph-style source and the mirrored
   layout direction.
2. Reduce dimensions until DCC/DXP passes, then scale up until the LXLU
   boundary failure appears.
3. Compare the post-DDC internal `INTERNAL` layout and autoshuffle schedule for
   the passing direction versus the mirrored direction.
4. Adjust internal tiling/chunking or layout ordering, not the public flag
   surface, until the mirrored direction compiles and passes correctness.

Only that path can prove the Stage 3B-style in-graph LX-to-LX restickify claim.

## Validation

Pod:

```text
python -m py_compile \
  torch_spyre/_inductor/insert_restickify.py \
  torch_spyre/_inductor/spyre_kernel.py \
  torch_spyre/_inductor/codegen/restickify_ddl_bridge.py \
  torch_spyre/_inductor/codegen/superdsc.py \
  tests/inductor/test_restickify_ddl_bridge.py

python -m pytest \
  tests/inductor/test_restickify_ddl_bridge.py \
  tests/inductor/test_restickify_mapping_alignment.py \
  -q
```

Result:

```text
24 passed
```

Default-off focused regression:

```text
python -m pytest tests/inductor/test_restickify.py \
  -k "opt_adds_then_matmul_x or opt_matmul_then_adds or opt_chain_transposed_intermediate" \
  -q
```

Result:

```text
3 passed, 94 deselected
```
