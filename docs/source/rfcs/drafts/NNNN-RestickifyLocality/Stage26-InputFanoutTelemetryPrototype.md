# Stage 26: Input Fanout Telemetry Prototype

## Summary

This stage starts Project B from Stage 24 with default-off source fanout
telemetry for graph inputs, weights, constants, and other external sources.

The pass is attribution-only. It does not emit GTR metadata, change SDSC JSON,
change restickify placement, or change input/weight layouts.

## Interface

New flags:

```text
SPYRE_INPUT_FANOUT_TELEMETRY=1
SPYRE_INPUT_FANOUT_TELEMETRY_JSONL=/path/to/file.jsonl
```

The pass runs after `work_distribution` and before `scratchpad_planning`.

Each JSONL row contains:

- source name
- source kind, such as `graph_input_or_weight` or `constant_or_extern`
- consumer count
- consumer names
- consumer kind histogram
- restickify consumers
- bytes moved by restickifies sourced from the input
- approximate total consumer bytes
- source stride map
- target stride maps requested by consumers

## Why This Matters

The fused-block telemetry repeatedly showed restickifies dominated by
`graph_input_or_weight`. Stage 3B cannot optimize those rows because there is no
in-graph producer ownership to align with.

Fanout telemetry is the first measurement step for two possible future
optimizations:

1. input/weight layout selection or prepacking
2. GTR/multicast-aware staging for read-only sources with many compatible
   consumers

## Smoke Validation

Focused unit/static validation:

```text
python3 -m py_compile torch_spyre/_inductor/input_fanout_telemetry.py \
  torch_spyre/_inductor/core_continuity_telemetry.py \
  torch_spyre/_inductor/config.py torch_spyre/_inductor/passes.py
python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q
```

Result:

```text
21 passed in 0.12s
```

Combined compiler smoke with both new telemetry streams enabled:

```text
SPYRE_CORE_CONTINUITY_TELEMETRY=1 \
SPYRE_CORE_CONTINUITY_TELEMETRY_JSONL=/tmp/core-continuity-smoke.jsonl \
SPYRE_INPUT_FANOUT_TELEMETRY=1 \
SPYRE_INPUT_FANOUT_TELEMETRY_JSONL=/tmp/input-fanout-smoke.jsonl \
python -m pytest tests/inductor/test_restickify.py \
  -k "opt_matmul_then_adds or opt_chain_transposed_intermediate" -q
```

Result:

```text
2 passed, 95 deselected
core continuity JSONL rows: 4
input fanout JSONL rows: 6
```

Example fanout row:

```json
{
  "source_name": "arg0_1",
  "source_kind": "graph_input_or_weight",
  "consumer_count": 1,
  "consumer_kinds": {"reduction:batchmatmul": 1},
  "consumers": ["buf0"],
  "restickify_consumers": [],
  "restickify_bytes_moved": 0,
  "approximate_consumer_bytes": 4194304,
  "source_stride_map": [64, 128, 1],
  "target_stride_maps": [[64, 128, 1]]
}
```

## Interpretation

The first smoke only proves the telemetry path works. It is too small to decide
whether multicast or prepacking is valuable.

The next useful run is a larger survey over:

- restickify test families
- fused MLP/attention/Mamba/MoE probes
- long-context projection joins

Useful summary cuts:

- top sources by `restickify_bytes_moved`
- sources with `consumer_count > 1`
- sources whose consumers all request the same target stride map
- sources whose consumers request incompatible target stride maps

## Next Step

Before any GTR behavior change, run a backend contract probe for
`coreIdToGTRInfo_`:

1. inspect an SDSC JSON/schedule-tree example accepted by Deeptools
2. hand-add a minimal `coreIdToGTRInfo_` field to a transfer node
3. run `dxp_standalone --bundle`
4. inspect generated Dataflow IR/logs for multicast/GTR lowering

If Deeptools ignores or rejects the field, this becomes a backend enablement
project. If it accepts and lowers it, torch-spyre can add a default-off compiler
prototype for read-only, same-layout fanout sources.

