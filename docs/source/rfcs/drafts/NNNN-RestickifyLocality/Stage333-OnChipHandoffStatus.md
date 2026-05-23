# Stage 333: On-Chip Handoff Status Checkpoint

Date: 2026-05-23

## Purpose

This is the handoff checkpoint for continuing the on-chip restickify and Tier 1
handoff work in a fresh Codex thread.

The current context should be treated as evidence and state, not as a final
design.  The most important conclusion is that we have separated three related
problems:

1. ring-aware work ownership planning;
2. restickify replacement for stick-changing edges;
3. general on-chip handoff for same-stick nonlocal edges.

The third item is now the cleanest production-shaped direction.

## Current Branch

Repository:

```text
torch-spyre-first-principles
```

Branch:

```text
AdnanHoque/tier1-on-chip-handoff-planner
```

Last implementation commit before this checkpoint:

```text
b578592 Add Tier 1 on-chip handoff planner
```

No PR has been opened.  Pushing to Adnan-owned branches is allowed; opening PRs
or merging to main still requires explicit permission.

## Known-Good Environment

Run workflows on the cdx pod:

```text
adnan-cdx-spyre-dev-pf
```

Known-good shell setup:

```sh
export HOME=/home/adnan-cdx
export DTI_PROJECT_ROOT=$HOME/dt-inductor-mixed
source "$DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh"
export PYTHONPATH=/tmp/torch-spyre-tier1-planner:${PYTHONPATH:-}
```

Do not add `/opt/ibm/spyre` runtime overrides in this environment unless that is
the thing being tested.  Mixing those overrides with this env previously caused
`torch_spyre._C.so` to fail with an undefined `flex::AllocationDirective`
symbol.

## What Is Proven

### Stock Restickify Uses The HBM Path

Generated kernel names such as `ReStickifyOpHBM` are the direct compiler
evidence that the current stock restickify path is HBM-backed.  Counter runs on
isolated restickify kernels also showed balanced read/write memory traffic,
which is consistent with an HBM round trip.

This does not prove all possible restickification must use HBM.  It proves the
stock lowering path currently does.

### LX-To-LX Movement Exists

Standalone and mixed experiments proved that LX-to-LX movement is expressible in
Deeptools using data-op style transfers such as `InputFetchNeighbor` /
`STCDPOpLx`.  A small no-HBM bridge launched cleanly with counters like:

```text
HBM=0, LXLU>0, LXSU>0
```

That proves the fabric/runtime can execute LX-load/LX-store traffic without an
HBM round trip.

### Full-Tensor PT-LX Restickify Has One Narrow Correct Case

The full-tensor PT-LX path works for the high-signal 2048 square case:

```text
computed_transpose_adds_then_matmul_tuple, size 2048
```

It avoids `ReStickifyOpHBM`, uses a mixed PT-LX bridge, and is value-correct.

This path is narrow.  It is not production-ready for arbitrary sizes.

### Streaming/Tiled PT-LX Can Launch Without HBM But Is Not Correct Yet

The streaming/tiled PT-LX direction can produce no-HBM launches for smaller
cases.  The current blocker is semantic correctness:

- chunked path launches but the final visible descriptor does not match the
  consumer contract;
- direct-tile path matches the consumer-shaped descriptor better but returns
  wrong values;
- the likely issue is a coordinate transform or final descriptor mismatch
  between producer LX ownership, bridge scatter, and consumer input indexing.

Do not claim speedup from streaming/tiled PT-LX until size 512 is value-correct.

## Stage 3B Status

Stage 3B is the earlier narrow restickify optimization.  It aligns restickify
work distribution and core mapping for eligible in-graph producer-to-restickify
edges.  It does not remove restickify and does not replace the HBM template.

What it proved:

- exact modeled byte-hops can drop to zero for a narrow class of edges;
- at size 2048, the high-signal synthetic case showed a small runtime win;
- restickify count and bytes moved stayed unchanged.

Why it is not the main production path:

- it only applies to eligible in-graph restickify edges;
- many realistic restickifies are graph-input, weight, or persistent-state
  sourced;
- it does not solve HBM-backed stock restickify lowering.

Keep Stage 3B as telemetry/evidence and a secondary locality optimization.

## Tier 1 Planner Status

Tier 1 targets same-stick in-graph producer-to-consumer edges.  These are not
restickify edges.  The producer has already computed the logical tensor, but
the consumer wants a different physical ownership partition.  The desired
realization is:

```text
producer DL op -> STCDPOpLx/InputFetchNeighbor -> consumer DL op
```

inside one mixed SuperDSC.

Implemented in:

```text
torch_spyre/_inductor/on_chip_handoff.py
```

Default-off flags:

```sh
SPYRE_ON_CHIP_HANDOFF_PLANNING=1
SPYRE_ON_CHIP_HANDOFF_PLAN_JSONL=/path/to/plan.jsonl
SPYRE_ON_CHIP_HANDOFF_FOUNDATION_CONTRACT=0
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

The pass only plans and emits JSONL.  It does not change lowering.

Every valid row currently reports:

```text
realization_status = blocked-missing-foundation-contract
```

That is intentional.  Stock Torch-Spyre emits one SDSC per `OpSpec`, and LX does
not persist across independent `sdsc_execute` boundaries.

## Tier 1 Validation

Pod temp source tree:

```text
/tmp/torch-spyre-tier1-planner
```

Validation commands:

```sh
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

Broad planner survey:

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

Candidate sweep:

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

## Hardware Bandwidth Interpretation

Use these as plausibility bounds, not hardware proof:

- RIU data ring: 166 GB/s per direction, 333 GB/s aggregate bidirectional;
- HBM link: roughly 166 GB/s raw;
- LX port: about 140 GB/s per core.

For a modeled byte-hop count, the optimistic lower bound is:

```text
byte_hops / 333 GB/s
```

The conservative one-direction bound is:

```text
byte_hops / 166 GB/s
```

These bounds explain why Stage 3B runtime wins were small even when byte-hops
dropped sharply: restickify or transfer time is only one part of the full kernel
or fused graph runtime, and the modeled byte-hop term is a lower-bound proxy
rather than a direct profiler counter.

## Current Blockers

### For Streaming/Tiled PT-LX Restickify

Need value correctness for size 512.  The current no-HBM bridge can launch, but
the producer/bridge/consumer coordinate contract is wrong.

Next concrete check:

1. compare producer runtime LX output `PieceInfo` against bridge gather pieces;
2. compare bridge scatter/output pieces against consumer input descriptor;
3. identify the first coordinate where expected logical element and produced
   logical element diverge;
4. fix transform or descriptor;
5. only then expand to 1024/2048/2560/3072.

### For Tier 1 General On-Chip Handoff

Need the Deeptools Foundation contract:

- mixed data-op + DL-op SuperDSC import;
- one schedule containing producer, transfer, and consumer;
- real binding from data-op output into consumer `labeledDs_`;
- fail-closed fallback to stock HBM/materialized path.

The Tier 1 planner is already the correct Inductor-side front end.

## Recommended Next Step

Start with a minimal Tier 1 realization prototype, not the full streaming
restickify path.

Target case:

```text
computed_contiguous_then_add, size 512
```

Prototype target:

```text
producer add -> STCDPOpLx/InputFetchNeighbor -> consumer add
```

Acceptance criteria:

- one mixed SuperDSC or equivalent Foundation-contract artifact;
- no hand-spliced runtime artifact;
- no LD_PRELOAD shim;
- no `ReStickifyOpHBM`;
- `HBM=0` in generated/observed counters for the internal handoff;
- value-correct hardware run;
- stock HBM/materialized fallback when the Foundation contract is unavailable.

This is narrower than solving all restickification, but it is the right
production-shaped primitive.  Once same-stick handoff works, stick-changing
restickify can reuse the same mixed scheduling and binding machinery.

## Fresh Thread Prompt

Use this prompt in a new Codex thread:

```text
We are continuing Torch-Spyre on-chip handoff work from
docs/source/rfcs/drafts/NNNN-RestickifyLocality/Stage333-OnChipHandoffStatus.md.

Use branch AdnanHoque/tier1-on-chip-handoff-planner.
Do not open a PR or merge.
Run workflows on pod adnan-cdx-spyre-dev-pf using DTI_PROJECT_ROOT=$HOME/dt-inductor-mixed.

Goal: implement the first production-shaped Tier 1 realization prototype:
producer add -> STCDPOpLx/InputFetchNeighbor -> consumer add inside one mixed
Foundation-contract artifact, with HBM=0 and value correctness.  Start from the
planner in torch_spyre/_inductor/on_chip_handoff.py and the candidate case
computed_contiguous_then_add size 512.
```
