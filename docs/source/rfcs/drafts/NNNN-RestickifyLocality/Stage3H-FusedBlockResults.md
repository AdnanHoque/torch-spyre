# Stage 3H Fused-Block Results

This note records the fused MLP/attention/Mamba/MoE block sweep for the
Restickify Locality RFC. The goal was to test whether more realistic block-like
graphs around `tokens ~= hidden ~= 2048` create the same eligible
producer-to-restickify edge seen in the bare projection proxy.

## Probe Cases

Five forward-looking stress probes were added:

- `mlp_post_activation_join`
- `gated_mlp_post_activation_join`
- `attention_score_join_value_projection`
- `mamba_projection_state_gate_join`
- `moe_combine_join_projection`

Each case introduces a layout-changing join inside a fused block-like graph,
then feeds the result into a projection-style consumer.

## Sweep

The sweep used:

- `tokens={512,2048}`
- `hidden={1024,2048}`
- `intermediate=hidden`
- baseline: Stage 3B flags off
- candidate: `SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1` and
  `SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1`
- `LX_PLANNING=0`

All 40 rows compiled successfully: 5 cases x 2 token sizes x 2 hidden sizes x 2
modes.

## Result

Every fused block created restickifies, but every restickify was attributed to a
graph input or weight source. No fused block produced nonzero eligible in-graph
byte-hops.

| Hidden | Tokens | Cases | Restickifies | Source kind | Byte-hops |
|---:|---:|---:|---:|---|---:|
| 1024 | 512 | 5 | 6 | `graph_input_or_weight` | 0 |
| 1024 | 2048 | 5 | 6 | `graph_input_or_weight` | 0 |
| 2048 | 512 | 5 | 6 | `graph_input_or_weight` | 0 |
| 2048 | 2048 | 5 | 6 | `graph_input_or_weight` | 0 |

The attention stress case produced two restickifies per shape; the other block
probes produced one restickify each. Stage 3B did not change byte-hops because
there were no eligible in-graph producer restickifies to align.

## Interpretation

This is a useful negative result:

- More realistic fused block probes did not reproduce the bare projection
  locality win.
- The block-like restickifies were dominated by graph-input/weight layout
  boundaries.
- Stage 3B is still valid, but it is not the right tool for these rows because
  there is no in-graph producer ownership to align.

No timing was run because no case met the telemetry acceptance rule: nonzero
eligible baseline byte-hops with at least 50% Stage 3B reduction.

## Next Step

The project should shift from additional Stage 3B tuning toward source
attribution and graph-input/weight layout handling. The evidence now suggests
that realistic block-like restickifies are more often input/weight boundaries
than producer-to-consumer locality mismatches.
