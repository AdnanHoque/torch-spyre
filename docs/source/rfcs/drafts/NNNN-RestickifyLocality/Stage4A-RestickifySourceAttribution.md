# Stage 4A Restickify Source Attribution

This note records the first restickify source-attribution pass for the
Restickify Locality RFC. The goal was to split the previous coarse
`graph-input-or-missing-producer` skip bucket into useful source metadata.

## Implementation

Telemetry rows now include source and consumer metadata:

- `source_name`
- `source_kind`
- `consumer`
- `consumer_kind`
- `source_stride_map`
- `target_stride_map`

The source categories are:

- `in_graph_computed`
- `graph_input_or_weight`
- `constant_or_extern`
- `mutation_target`
- `unknown`

Exact byte-hop estimation is unchanged. In-graph producer rows still compute
byte-hops the same way as Stage 3A/3B. Skipped rows now retain bytes moved plus
source category and stride-map context.

## Validation

| Check | Result |
|---|---:|
| `python3 -m py_compile` on modified files | passed |
| `python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q` | 13 passed |
| `python -m pytest tests/inductor/test_restickify.py -q` | 97 passed |

The scenario probe CSV/JSON summaries also now include:

- `ring_source_kinds`
- `ring_exact_rows`
- `ring_skipped_rows`

## Synthetic Attribution Survey

The synthetic survey ran all probe cases at size `128` with forward-looking
cases enabled.

| Metric | Value |
|---|---:|
| Scenario rows | 29 |
| Scenario errors | 1 |
| Restickify telemetry rows | 30 |
| Graph-input/weight rows | 26 |
| In-graph computed rows | 4 |
| Exact rows | 3 |
| Skipped rows | 27 |
| Total bytes from graph-input/weight rows | 2,228,224 |
| Total bytes from in-graph computed rows | 327,680 |
| Exact in-graph byte-hops | 2,383,872 |

Source breakdown:

| Source kind | Rows | Bytes moved | Byte-hops |
|---|---:|---:|---:|
| `graph_input_or_weight` | 26 | 2,228,224 | 0 |
| `in_graph_computed` | 4 | 327,680 | 2,383,872 |

The top exact byte-hop rows were:

| Case | Source kind | Consumer kind | Bytes | Byte-hops |
|---|---|---|---:|---:|
| `prefill_projection_join` | `in_graph_computed` | `reduction:batchmatmul` | 131,072 | 1,048,576 |
| `decode_projection_join` | `in_graph_computed` | `reduction:batchmatmul` | 131,072 | 1,048,576 |
| `adds_then_matmul` | `in_graph_computed` | `reduction:batchmatmul` | 32,768 | 286,720 |

Most block-like rows were graph-input/weight sourced. Example:

```text
source_kind=graph_input_or_weight
consumer_kind=computed or reduction:batchmatmul
skip_reason=graph-input-or-missing-producer
```

## Granite 4 H-Tiny Model-Op Probe

The passing Granite 4 H-Tiny model-op subset was rerun with attribution enabled:

| Test subset | Result | Telemetry bytes |
|---|---:|---:|
| 6 passing model-op tests | 6 passed | 0 |

The current passing model-op subset still does not emit restickify telemetry.
That means it is not useful for source attribution yet. The unsupported Granite
nodes remain outside this measurement because they fail before restickify
telemetry can observe rows.

## Interpretation

This is the most important direction shift so far:

- Stage 3B has a real, measurable win for eligible in-graph producer rows.
- The broader synthetic and fused-block evidence shows many restickifies are
  graph-input/weight sourced instead.
- Those rows are outside Stage 3B's optimization scope.
- The next optimization family should focus on input layout selection, weight
  prepacking, persistent-state layout management, or avoiding repeated
  graph-input restickification.

Stage 3B should remain default-off and evidence-backed. Source attribution
should become the default diagnostic lens for deciding which restickify family
to optimize next.
