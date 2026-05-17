# Stage 25: Core Continuity Telemetry Prototype

## Summary

This stage starts Project A from Stage 24 with a default-off telemetry pass for
in-graph producer-consumer core ownership continuity.

The pass does not change code generation, work distribution, restickify
placement, or tensor layouts. It only records where adjacent in-graph ops appear
to assign the same logical tensor regions to different physical cores.

## Interface

New flags:

```text
SPYRE_CORE_CONTINUITY_TELEMETRY=1
SPYRE_CORE_CONTINUITY_TELEMETRY_JSONL=/path/to/file.jsonl
```

The pass runs after `work_distribution`, after the default-off Stage 3B
restickify mapping hook, and before `scratchpad_planning`.

Each JSONL row contains:

- source tensor name
- producer and consumer names
- producer and consumer kinds
- bytes moved
- modeled RIU byte-hops
- average and max hops
- producer and consumer split maps
- symbol correspondence
- skip reason when exact edge modeling is not supported

## Implementation Notes

The first estimator is intentionally strict. It only computes exact byte-hops
when the producer write and consumer read can be mapped as the same logical
tensor region by unique stride-symbol correspondence. Edges with consumer-only
dimensions, reduction dimensions, ambiguous strides, or shape mismatches are
reported with a skip reason instead of guessed.

This keeps the telemetry conservative: a nonzero row should mean "we can model
this ownership mismatch exactly", while a skipped row means "this needs a more
specialized estimator".

## Smoke Validation

Focused unit/static validation:

```text
python3 -m py_compile torch_spyre/_inductor/core_continuity_telemetry.py \
  torch_spyre/_inductor/config.py torch_spyre/_inductor/passes.py
python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q
```

Result:

```text
20 passed in 0.10s
```

Compiler smoke on pod with `/opt/ibm/spyre/deeptools` overrides:

```text
SPYRE_CORE_CONTINUITY_TELEMETRY=1 \
SPYRE_CORE_CONTINUITY_TELEMETRY_JSONL=/tmp/core-continuity-smoke.jsonl \
python -m pytest tests/inductor/test_restickify.py \
  -k "opt_matmul_then_adds or opt_chain_transposed_intermediate" -q
```

Result:

```text
2 passed, 95 deselected
JSONL rows: 4
```

Example row:

```json
{
  "producer": "buf2",
  "producer_kind": "restickify",
  "consumer": "buf1",
  "consumer_kind": "computed",
  "bytes_moved": 32768,
  "byte_hops": 286720,
  "avg_hops": 8.75,
  "max_hops": 16,
  "producer_splits": {"d0": 2, "d1": 2},
  "consumer_splits": {"d1": 32},
  "symbol_map": {"d0": "d0", "d1": "d1"},
  "skip_reason": null
}
```

The same smoke also produced a skipped row:

```text
producer_kind=reduction:batchmatmul
consumer_kind=restickify
skip_reason=incomplete-symbol-map
```

That skip is expected for this first pass: matmul-to-restickify edges include
symbols that are not a simple same-region producer/consumer mapping.

## Early Interpretation

The prototype already finds exact nonzero ownership mismatch rows on
restickify-to-consumer edges. That suggests the continuity telemetry can expose
movement beyond the original producer-to-restickify Stage 3B view.

It is not yet an optimizer. The next useful measurement is a larger survey over
the restickify test family and model-ish probes with both telemetry streams
enabled:

- `SPYRE_RESTICKIFY_RING_TELEMETRY=1`
- `SPYRE_CORE_CONTINUITY_TELEMETRY=1`

Then compare:

- producer-to-restickify byte-hops
- restickify-to-consumer byte-hops
- direct producer-to-consumer byte-hops
- skipped edge reasons

## Next Step

Run a telemetry survey and classify the top rows. Only if exact nonzero
continuity rows show up repeatedly should we add the next behavior flag:

```text
SPYRE_ALIGN_CORE_DIVISION_CONTINUITY=1
```

