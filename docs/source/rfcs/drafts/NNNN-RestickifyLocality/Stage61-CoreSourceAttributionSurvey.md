# Stage 61: Core Restickify Source-Attribution Survey

## Summary

Stage 61 ran a stock-template source-attribution survey over the core
restickify probe families at sizes `512` and `2048`.

Cases:

```text
pointwise_transpose_add
pointwise_three_mixed
matmul_lhs_wrong_stick
matmul_rhs_wrong_stick
adds_then_matmul
matmul_then_add
transpose_chain
fanout_diamond
transpose_4d_chain
linear_weight_transposed
```

This was a baseline-only attribution run. The goal was to measure where
restickifies come from, not to time kernels.

## Aggregate Result

Across 20 compiled probe rows:

```text
restickify rows: 24
total bytes moved by restickify rows: 99,352,576
exact modeled byte-hops: 68,485,120
```

Source categories:

| source kind | rows | interpretation |
| --- | ---: | --- |
| `graph_input_or_weight` | 20 | outside Stage 3B scope |
| `in_graph_computed` | 4 | possible Stage 3B candidates |

Skip categories:

| skip reason | rows | interpretation |
| --- | ---: | --- |
| `graph-input-or-missing-producer` | 20 | producer is outside compiled graph |
| `incomplete-symbol-map` | 2 | producer is in graph, but exact correspondence was not proven |
| none | 2 | exact byte-hop estimate available |

The two exact nonzero rows are both `adds_then_matmul`:

| case | size | bytes moved | byte-hops | avg/max hops | producer split | restickify split |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `adds_then_matmul` | 512 | 524,288 | 1,376,256 | 2.625 / 7 | `d1:32` | `d0:8,d1:4` |
| `adds_then_matmul` | 2048 | 8,388,608 | 67,108,864 | 8.000 / 16 | `d1:32` | `d0:32` |

The two in-graph but skipped rows are `matmul_then_add`:

| case | size | bytes moved | producer split | restickify split | skip |
| --- | ---: | ---: | --- | --- | --- |
| `matmul_then_add` | 512 | 524,288 | `d0:32` | `d0:8,d1:4` | `incomplete-symbol-map` |
| `matmul_then_add` | 2048 | 8,388,608 | `d0:32` | `d0:32` | `incomplete-symbol-map` |

All other rows are graph-input/weight restickifies.

## Per-Case Summary

| case | size | restickifies | bytes moved | exact byte-hops | source/skip summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `pointwise_transpose_add` | 512 | 1 | 524,288 | 0 | graph-input |
| `pointwise_three_mixed` | 512 | 1 | 524,288 | 0 | graph-input |
| `matmul_lhs_wrong_stick` | 512 | 1 | 524,288 | 0 | graph-input |
| `matmul_rhs_wrong_stick` | 512 | 1 | 524,288 | 0 | graph-input |
| `adds_then_matmul` | 512 | 2 | 1,048,576 | 1,376,256 | one graph-input, one exact in-graph |
| `matmul_then_add` | 512 | 1 | 524,288 | 0 | in-graph, incomplete symbol map |
| `transpose_chain` | 512 | 1 | 524,288 | 0 | graph-input |
| `fanout_diamond` | 512 | 2 | 1,048,576 | 0 | graph-input |
| `transpose_4d_chain` | 512 | 1 | 262,144 | 0 | graph-input |
| `linear_weight_transposed` | 512 | 1 | 524,288 | 0 | graph-input |
| `pointwise_transpose_add` | 2048 | 1 | 8,388,608 | 0 | graph-input |
| `pointwise_three_mixed` | 2048 | 1 | 8,388,608 | 0 | graph-input |
| `matmul_lhs_wrong_stick` | 2048 | 1 | 8,388,608 | 0 | graph-input |
| `matmul_rhs_wrong_stick` | 2048 | 1 | 8,388,608 | 0 | graph-input |
| `adds_then_matmul` | 2048 | 2 | 16,777,216 | 67,108,864 | one graph-input, one exact in-graph |
| `matmul_then_add` | 2048 | 1 | 8,388,608 | 0 | in-graph, incomplete symbol map |
| `transpose_chain` | 2048 | 1 | 8,388,608 | 0 | graph-input |
| `fanout_diamond` | 2048 | 2 | 16,777,216 | 0 | graph-input |
| `transpose_4d_chain` | 2048 | 1 | 1,048,576 | 0 | graph-input |
| `linear_weight_transposed` | 2048 | 1 | 8,388,608 | 0 | graph-input |

## Interpretation

This survey explains the Stage 3B opportunity size:

- Stage 3B is aimed at in-graph producer-to-restickify ownership continuity.
- In this core survey, most restickifies are not in-graph producer edges.
- The only exact nonzero in-graph byte-hop rows are the known
  `adds_then_matmul` cases.
- `matmul_then_add` might become analyzable with stronger symbol/view
  correspondence, but it is not currently a certified Stage 3B target.

The larger optimization family remains graph-input, weight, and persistent-state
layout handling. Those rows dominate count and bytes in this survey.

## Recommendation

For Stage 3B:

1. keep the patch small and default-off;
2. present it as a locality-certified ownership-continuity prototype;
3. include `adds_then_matmul` as the primary positive test;
4. include graph-input and incomplete-symbol-map cases as negative tests.

For next feature work:

1. investigate graph-input/weight layout selection or prepacking;
2. decide whether `matmul_then_add` is worth improving in the symbol-mapping
   estimator;
3. use the source-attribution table to prioritize restickify scenarios by bytes,
   not just by count.
