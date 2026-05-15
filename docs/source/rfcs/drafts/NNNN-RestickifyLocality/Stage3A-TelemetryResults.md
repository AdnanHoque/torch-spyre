# Stage 3A Telemetry Results

This note records the first telemetry-only Stage 3A run for the Restickify
Locality RFC. It is not a performance claim and does not justify enabling a
ring-aware optimizer by default. The goal was narrower: prove that the compiler
can attribute byte-hop cost to specific restickify edges and identify whether
there are high-impact synthetic scenarios worth optimizing next.

## Implementation Validated

Stage 3A adds default-off compiler telemetry:

- `SPYRE_RESTICKIFY_RING_TELEMETRY=1`
- `SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL=/path/to/file.jsonl`

The telemetry runs after `work_distribution`, after optional Stage 2 mapping
alignment, and before scratchpad planning. It emits one JSONL row per
compiler-inserted restickify with producer, consumers, bytes moved, byte-hops,
average hops, max hops, split maps, symbol map, and skip reason when exact cost
cannot be computed.

The synthetic probe can summarize this telemetry with:

```sh
python tools/restickify_scenario_probe.py \
  --include-forward-looking \
  --size 128 \
  --ring-telemetry \
  --skip-correctness \
  --output-dir /tmp/restickify-stage3a/survey-128
```

## Validation

The following checks passed on the Spyre pod:

| Check | Result |
|---|---:|
| `python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q` | 7 passed |
| `python -m pytest tests/inductor/test_restickify.py -q` with telemetry off | 97 passed |
| `tools/restickify_scenario_probe.py --include-forward-looking --size 128 --ring-telemetry --skip-correctness` | 16 rows, 0 errors |

## Survey Summary

The size-128 taxonomy survey found restickification in the current compiler, but
only one scenario with exact nonzero in-graph byte-hop cost:

| Run | Rows | Restickifies | Bytes moved | Estimated byte-hops | Notes |
|---|---:|---:|---:|---:|---|
| `survey-128` | 16 | 13 | 491,520 | 286,720 | Only `adds_then_matmul` had nonzero exact byte-hops |
| `adds_then_matmul`, 512, baseline | 1 | 2 | 1,048,576 | 1,376,256 | One graph-input skip, one measurable in-graph restickify |
| `adds_then_matmul`, 512, Stage 2 alignment | 1 | 2 | 1,048,576 | 1,376,256 | Unchanged because splits are incompatible |
| `adds_then_matmul`, 2048, baseline | 1 | 2 | 16,777,216 | 67,108,864 | High-impact synthetic signal |
| `adds_then_matmul`, 2048, Stage 2 alignment | 1 | 2 | 16,777,216 | 67,108,864 | Unchanged because splits are incompatible |

## High-Signal Case

The clearest current synthetic case is:

```python
(a + b.t() + c.t()) @ d
```

The graph has two restickifies. One is graph-input sourced and therefore
outside the Stage 2 producer-alignment scope. The other is an in-graph producer
to restickify edge with exact ownership attribution:

| Size | Measurable bytes | Producer splits | Restickify splits | Byte-hops | Avg hops | Max hops |
|---:|---:|---|---|---:|---:|---:|
| 128 | 32,768 | `d1:32` | `d0:2,d1:2` | 286,720 | 8.75 | 16 |
| 512 | 524,288 | `d1:32` | `d0:8,d1:4` | 1,376,256 | 2.625 | 7 |
| 2048 | 8,388,608 | `d1:32` | `d0:32` | 67,108,864 | 8.0 | 16 |

This explains why Stage 2 alignment does not move the number: the producer and
restickify own the same logical tensor through different split factors. A pure
physical remap is intentionally skipped. Reducing this case requires Stage 3
work-distribution or layout steering so the restickify can split along the
producer-corresponding dimension when legal.

## Interpretation

Stage 3A gives us enough evidence for a next compiler experiment, but not enough
evidence for a default-on optimization:

- We do have a high-impact synthetic in-graph restickify: 67,108,864 byte-hops
  at size 2048 for `adds_then_matmul`.
- The current conservative mapping alignment is not sufficient for that case
  because the split shapes differ.
- The survey still shows many restickifies are graph-input or weight sourced,
  which need a different optimization path such as input layout selection,
  prepacking, or persistent layout caching.
- We do not yet have model-slice or end-to-end evidence that eligible
  restickifies are a meaningful fraction of runtime.

## Recommended Next Step

Stage 3B should be a guarded work-distribution experiment, not a placement
rewrite. For restickify ops with a clear producer symbol map, prefer the split
dimension that corresponds to the producer's dominant split when divisibility
and existing work-division constraints allow it. Then rerun the Stage 3A
telemetry commands and require byte-hop reduction before any latency benchmark.
