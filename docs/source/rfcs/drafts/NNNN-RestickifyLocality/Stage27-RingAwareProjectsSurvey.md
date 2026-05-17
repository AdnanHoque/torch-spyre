# Stage 27: Ring-Aware Projects Telemetry Survey

## Summary

This stage ran the first survey for the two new ring-aware project directions:

1. **core-division continuity** across producer-consumer edges
2. **input/weight/constant fanout** as a possible multicast or prepacking target

All compiler behavior was baseline/default-off for the broad survey:

```text
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=0
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=0
SPYRE_RESTICKIFY_RING_TELEMETRY=1
SPYRE_CORE_CONTINUITY_TELEMETRY=1
SPYRE_INPUT_FANOUT_TELEMETRY=1
```

Artifacts were copied locally under:

```text
artifacts/stage27_ring_projects_survey/
```

## Coverage

Baseline survey:

- 30 core probe rows
- 14 model-ish probe rows
- 0 errors
- 62 total restickifies
- 179,732,480 total restickify bytes
- 105,078,784 producer-to-restickify byte-hops

Restickify source kinds:

| Source kind | Rows |
|---|---:|
| `graph_input_or_weight` | 52 |
| `in_graph_computed` | 10 |

This keeps confirming the earlier picture: many restickifies are graph-input or
weight sourced, while the producer-alignment problem exists but is narrower.

## Producer-To-Restickify Hotspots

Top rows from the existing restickify ring telemetry:

| Case | Size | Restickifies | Bytes | Byte-hops |
|---|---:|---:|---:|---:|
| `adds_then_matmul` | 2048 | 2 | 16,777,216 | 67,108,864 |
| `prefill_projection_join` | 2048 | 2 | 4,194,304 | 16,777,216 |
| `decode_projection_join` | 2048 | 2 | 4,194,304 | 16,777,216 |
| `adds_then_matmul` | 512 | 2 | 1,048,576 | 1,376,256 |
| `prefill_projection_join` | 512 | 2 | 1,048,576 | 1,376,256 |
| `decode_projection_join` | 512 | 2 | 1,048,576 | 1,376,256 |
| `adds_then_matmul` | 128 | 2 | 65,536 | 286,720 |

The model-ish projection-join probes now reproduce the same pattern as
`adds_then_matmul`, which is useful: the high-signal edge is not limited to the
old bare synthetic case.

## Core-Continuity Hotspots

The new continuity telemetry sees additional ownership mismatch beyond
producer-to-restickify. Top baseline rows grouped by case and size:

| Case | Size | Exact rows | Skipped rows | Byte-hops | Max hops |
|---|---:|---:|---:|---:|---:|
| `attention_prefill_no_softmax` | 2048 | 1 | 3 | 268,435,456 | 16 |
| `adds_then_matmul` | 2048 | 3 | 1 | 134,217,728 | 16 |
| `fanout_diamond` | 2048 | 6 | 0 | 134,217,728 | 16 |
| `pointwise_transpose_add` | 2048 | 1 | 0 | 67,108,864 | 16 |
| `pointwise_three_mixed` | 2048 | 2 | 0 | 67,108,864 | 16 |
| `matmul_then_add` | 2048 | 1 | 1 | 67,108,864 | 16 |
| `transpose_chain` | 2048 | 2 | 0 | 67,108,864 | 16 |
| `attention_score_join_value_projection` | 2048 | 3 | 3 | 67,108,864 | 16 |
| `prefill_projection_join` | 2048 | 3 | 1 | 33,554,432 | 16 |
| `decode_projection_join` | 2048 | 3 | 1 | 33,554,432 | 16 |

This is the most important new signal. Stage 3B only optimizes the
producer-to-restickify edge. Continuity telemetry also sees restickify-to-consumer
and direct producer-consumer ownership mismatches.

## Stage 3B Targeted Comparison

I reran the producer-to-restickify hotspots with Stage 3B enabled:

```text
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1
```

| Case | Size | Baseline byte-hops | Stage 3B byte-hops | Reduction |
|---|---:|---:|---:|---:|
| `adds_then_matmul` | 512 | 1,376,256 | 655,360 | 52.4% |
| `adds_then_matmul` | 2048 | 67,108,864 | 0 | 100.0% |
| `prefill_projection_join` | 512 | 1,376,256 | 655,360 | 52.4% |
| `prefill_projection_join` | 2048 | 16,777,216 | 2,621,440 | 84.4% |
| `decode_projection_join` | 512 | 1,376,256 | 655,360 | 52.4% |
| `decode_projection_join` | 2048 | 16,777,216 | 2,621,440 | 84.4% |

Stage 3B therefore generalizes to the projection-join probes, but does not
always reduce modeled producer-to-restickify byte-hops to zero.

The continuity telemetry still reports residual byte-hops after Stage 3B:

| Case | Size | Baseline continuity byte-hops | Stage 3B continuity byte-hops |
|---|---:|---:|---:|
| `adds_then_matmul` | 2048 | 134,217,728 | 67,108,864 |
| `prefill_projection_join` | 2048 | 33,554,432 | 19,398,656 |
| `decode_projection_join` | 2048 | 33,554,432 | 19,398,656 |

That residual is exactly why Project A should continue: there are locality
misses outside the narrow producer-to-restickify edge.

## Input/Weight Fanout Findings

Top graph-input/weight restickify byte rows:

| Case | Size | Sources | Restickify bytes | Max consumers/source |
|---|---:|---:|---:|---:|
| `attention_prefill_no_softmax` | 2048 | 4 | 34,603,008 | 1 |
| `fanout_diamond` | 2048 | 4 | 16,777,216 | 1 |
| `attention_score_join_value_projection` | 2048 | 6 | 16,777,216 | 1 |
| `pointwise_transpose_add` | 2048 | 2 | 8,388,608 | 1 |
| `matmul_lhs_wrong_stick` | 2048 | 2 | 8,388,608 | 1 |
| `linear_weight_transposed` | 2048 | 2 | 8,388,608 | 1 |
| `mlp_gated_projection_join` | 2048 | 8 | 4,194,304 | 1 |
| `moe_two_expert_join` | 2048 | 8 | 4,194,304 | 3 |

The survey found multi-consumer graph-input sources mainly in the MoE probe:

| Case | Size | Source | Consumers | Restickify bytes | Target layouts |
|---|---:|---|---:|---:|---:|
| `moe_two_expert_join` | 512 | `arg0_1` | 3 | 0 | 1 |
| `moe_two_expert_join` | 2048 | `arg0_1` | 3 | 0 | 1 |
| `moe_two_expert_join` | 512 | `arg6_1` | 2 | 0 | 2 |
| `moe_two_expert_join` | 2048 | `arg6_1` | 2 | 0 | 2 |

So multicast-aware fanout is plausible, but this survey does not yet show it as
the main restickify-byte reducer. The stronger graph-input/weight opportunity is
still layout selection or prepacking: many sources have one consumer that needs a
different layout, rather than many consumers of the same source layout.

## Conclusion

The next optimizer project should be **Project A: core-division continuity**.

Reason:

- It has exact nonzero byte-hop rows across several families.
- It exposes residual locality cost after Stage 3B.
- It stays in torch-spyre's current control surface: work division and
  `coreIdToWkSlice_`.

Project B should continue as telemetry plus a backend contract probe:

- first prove `coreIdToGTRInfo_` can be emitted and consumed from
  torch-spyre-generated SDSC
- then target same-layout, read-only multi-consumer sources
- separately consider input/weight layout prepacking for one-consumer
  graph-input restickifies

## Next Step

Implement a default-off continuity alignment prototype:

```text
SPYRE_ALIGN_CORE_DIVISION_CONTINUITY=1
```

Start with pointwise/restickify consumers only:

- source is in-graph computed
- exact symbol map exists
- consumer split factors can match producer split factors
- modeled continuity byte-hops do not increase
- optional assert mode requires zero modeled continuity byte-hops

Do not change matmul/reduction distribution yet. The skipped rows show that
matmul/reduction edges need a separate estimator and a compute-vs-locality cost
model.

