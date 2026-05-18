# Stage 60: Second-Family Restickify Telemetry

## Summary

Stage 60 tested whether the Stage 3B signal generalizes from
`adds_then_matmul` into three nearby families:

```text
matmul_then_add
transpose_chain
mlp_gated_projection_join
```

The run used stock Deeptools templates and sizes:

```text
512, 1024, 2048
```

For the MLP proxy:

```sh
SPYRE_PROBE_HIDDEN=2048
SPYRE_PROBE_INTERMEDIATE=2048
```

## Result

All cases compiled and ran in both baseline and Stage 3B modes. Restickifies were
present, but baseline modeled in-graph byte-hops were zero for every row:

| case | size | restickifies | bytes moved | baseline byte-hops | Stage 3B byte-hops | source/skip pattern |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `matmul_then_add` | 512 | 1 | 524,288 | 0 | 0 | `in_graph_computed`, `incomplete-symbol-map` |
| `matmul_then_add` | 1024 | 1 | 2,097,152 | 0 | 0 | `in_graph_computed`, `incomplete-symbol-map` |
| `matmul_then_add` | 2048 | 1 | 8,388,608 | 0 | 0 | `in_graph_computed`, `incomplete-symbol-map` |
| `transpose_chain` | 512 | 1 | 524,288 | 0 | 0 | `graph_input_or_weight` |
| `transpose_chain` | 1024 | 1 | 2,097,152 | 0 | 0 | `graph_input_or_weight` |
| `transpose_chain` | 2048 | 1 | 8,388,608 | 0 | 0 | `graph_input_or_weight` |
| `mlp_gated_projection_join` | 512 | 2 | 4,194,304 | 0 | 0 | `graph_input_or_weight` |
| `mlp_gated_projection_join` | 1024 | 2 | 8,388,608 | 0 | 0 | `graph_input_or_weight` |
| `mlp_gated_projection_join` | 2048 | 2 | 16,777,216 | 0 | 0 | `graph_input_or_weight` |

## Interpretation

This is useful negative evidence.

`matmul_then_add` has an in-graph computed source, but the current telemetry and
Stage 3B matching logic cannot construct a complete symbol map:

```text
producer splits:    d0:32
restickify splits:  d0:32       at 2048
skip reason:        incomplete-symbol-map
```

Because the exact byte-hop estimator skips the row, Stage 3B does not have a
certified optimization target here.

`transpose_chain` and `mlp_gated_projection_join` mostly expose graph-input or
weight layout materializations. Those restickifies can move many bytes, but they
are not producer-to-restickify ownership-continuity opportunities because the
producer is outside the compiled graph.

This reinforces the split between two optimization families:

1. Stage 3B: narrow in-graph producer-to-restickify ownership continuity.
2. Future work: graph-input, weight, and persistent-state layout handling.

## Consequence For Stage 3B

The Stage 3B PR should stay narrow and honest:

- it has one strong positive family, `adds_then_matmul`;
- the positive case is repeatable with stock templates;
- nearby families do emit restickifies, but they are not currently eligible;
- broader model-block wins likely require input/weight/state layout control, not
  just restickify work-distribution alignment.

## Next Step

There are two useful directions from here:

1. Run a wider source-attribution survey across the restickify test family to
   quantify how often rows are `in_graph_computed`, `graph_input_or_weight`, or
   `incomplete-symbol-map`.
2. Prototype the next optimization family: graph-input/weight layout placement,
   prepacking, or persistent-state layout management.

For Stage 3B specifically, the next engineering step is to turn the current
prototype into a small, reviewable, default-off patch with a locality certificate
and focused tests, rather than broadening the scope.
