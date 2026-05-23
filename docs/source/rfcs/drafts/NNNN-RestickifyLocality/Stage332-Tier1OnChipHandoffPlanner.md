# Stage 332: Tier 1 On-Chip Handoff Planner

## Summary

Stage 332 starts Tier 1 as a planner, not a lowering replacement.

The pass finds in-graph producer-to-consumer edges that:

- are not compiler-inserted restickify edges;
- have an exact producer/consumer symbol correspondence;
- are therefore same-stick/logical identity edges;
- have nonzero modeled RIU byte-hops under the current producer and consumer
  work divisions.

The pass emits telemetry and attaches metadata to the consumer op, but it does
not change lowering.  Every valid plan explicitly reports:

```text
realization_status = blocked-missing-foundation-contract
```

This is intentional.  A normal Torch-Spyre bundle still emits one SDSC per
`OpSpec`, and LX does not survive independent `sdsc_execute` boundaries.  Tier 1
needs the Deeptools Foundation contract before a plan can become an on-chip
handoff:

- mixed data-op + DL-op SuperDSC import;
- one schedule containing producer, transport, and consumer;
- a first-class binding from data-op output to the consumer input `labeledDs_`.

## Code Shape

New default-off flags:

```sh
SPYRE_ON_CHIP_HANDOFF_PLANNING=1
SPYRE_ON_CHIP_HANDOFF_PLAN_JSONL=/path/to/plan.jsonl
SPYRE_ON_CHIP_HANDOFF_FOUNDATION_CONTRACT=0
```

New module:

```text
torch_spyre/_inductor/on_chip_handoff.py
```

Pass placement:

```text
work_distribution
align_restickify_core_mappings
align_core_continuity_mappings
restickify_ring_telemetry
core_continuity_telemetry
input_fanout_telemetry
plan_on_chip_handoffs
scratchpad_planning
```

This matches the RFC framing: detection needs committed work division, but
realization must happen later during SDSC/fusion/codegen and is gated on the
Foundation contract.

## Focused Validation

Pod validation used a temporary source tree:

```text
/tmp/torch-spyre-tier1-planner
```

Environment:

```sh
export DTI_PROJECT_ROOT=$HOME/dt-inductor-mixed
source "$DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh"
export PYTHONPATH=/tmp/torch-spyre-tier1-planner:${PYTHONPATH:-}
```

Static/focused tests:

```text
python -m py_compile \
  torch_spyre/_inductor/on_chip_handoff.py \
  torch_spyre/_inductor/passes.py \
  torch_spyre/_inductor/config.py \
  tests/inductor/test_restickify_mapping_alignment.py

python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q
```

Result:

```text
28 passed in 0.11s
```

## Initial Planner Survey

Command shape:

```sh
SPYRE_ON_CHIP_HANDOFF_PLANNING=1
SPYRE_ON_CHIP_HANDOFF_PLAN_JSONL=/tmp/stage-tier1-all512/on_chip_handoff.jsonl
python tools/restickify_scenario_probe.py \
  --include-forward-looking \
  --size 512 \
  --skip-correctness \
  --skip-kernel-launch \
  --output-dir /tmp/stage-tier1-all512 \
  --copy-kernel-code
```

Result:

```text
38 probe rows, 3 expected compile errors in stress cases
164 on-chip handoff edge rows
4 planned Tier 1 candidates
160 skipped rows
```

Skip reasons:

| Reason | Count |
|---|---:|
| `incomplete-symbol-map` | 60 |
| `already-core-local` | 55 |
| `stick-changing-restickify-edge-is-tier2` | 45 |

Planned candidates at size 512:

| Case | Planned edges | Bytes | Byte-hops |
|---|---:|---:|---:|
| `computed_contiguous_then_add` | 2 | 1,048,576 | 5,570,560 |
| `computed_self_transpose_join3` | 1 | 524,288 | 4,194,304 |
| `attention_value` | 1 | 262,144 | 327,680 |

All planned rows reported:

```text
transport_kind = same-stick-lx-to-lx
realization_status = blocked-missing-foundation-contract
```

## Candidate Size Sweep

Command shape:

```sh
SPYRE_ON_CHIP_HANDOFF_PLANNING=1
SPYRE_ON_CHIP_HANDOFF_PLAN_JSONL=/tmp/stage-tier1-candidates/on_chip_handoff.jsonl
python tools/restickify_scenario_probe.py \
  --case computed_contiguous_then_add \
  --case computed_self_transpose_join3 \
  --case attention_value \
  --size 512 \
  --size 1024 \
  --size 2048 \
  --skip-correctness \
  --skip-kernel-launch \
  --output-dir /tmp/stage-tier1-candidates \
  --copy-kernel-code
```

Planned rows:

| Case | Size | Planned edges | Bytes | Byte-hops | Max hops |
|---|---:|---:|---:|---:|---:|
| `attention_value` | 512 | 1 | 262,144 | 327,680 | 3 |
| `attention_value` | 1024 | 1 | 524,288 | 262,144 | 1 |
| `computed_contiguous_then_add` | 512 | 2 | 1,048,576 | 5,570,560 | 16 |
| `computed_contiguous_then_add` | 1024 | 2 | 4,194,304 | 27,918,336 | 16 |
| `computed_contiguous_then_add` | 2048 | 1 | 8,388,608 | 67,108,864 | 16 |
| `computed_self_transpose_join3` | 512 | 1 | 524,288 | 4,194,304 | 16 |
| `computed_self_transpose_join3` | 1024 | 1 | 2,097,152 | 16,777,216 | 16 |

`computed_self_transpose_join3` still trips an existing Deeptools broadcast
compile error, but the planner rows are emitted before that backend failure.

## Interpretation

The planner proves the RFC's key Tier 1 detection claim:

```text
same-stick in-graph edges with nonzero core-to-core movement are currently
invisible to restickify telemetry, but they exist and can be found after work
division.
```

The implementation does not yet prove an on-chip runtime speedup.  That remains
blocked by the Deeptools Foundation contract.  The correct next milestone is a
same-stick synthetic edge that is realized as one mixed SDSC:

```text
producer DL op -> STCDPOpLx/InputFetchNeighbor -> consumer DL op
```

with:

- `HBM=0`;
- value correctness;
- no artifact splice;
- no LD_PRELOAD shim;
- stock HBM path as fallback when the contract is unavailable.
