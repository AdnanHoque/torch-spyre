# Stage 7: Restickify Memory-Hierarchy Sweep

This note records a focused sweep over restickify-heavy families modeled after
`tests/inductor/test_restickify.py`. The goal was to compare compiler
restickify telemetry and whole-kernel timing against three simple hardware
models:

1. graph-input or weight restickification as a global-memory/HBM boundary,
2. in-graph restickification with suboptimal LX-to-LX ownership, modeled as
   RIU byte-hop traffic, and
3. in-graph restickification with aligned ownership, modeled as local LX work
   with no cross-core ring traffic.

These are plausibility models, not direct hardware-counter measurements.
Compiler byte-hop telemetry tells us what movement is implied by ownership
geometry; it does not prove which physical fabric carried every byte.

## Setup

The sweep used the pod checkout at `/tmp/torch-spyre-refresh` and forced imports
with `PYTHONPATH=/tmp/torch-spyre-refresh`. Timing used `LX_PLANNING=0`,
`SENCORES=32`, `warmup=5`, and `iters=10`.

The hardware lower bounds use:

| Model | Formula |
|---|---|
| HBM one-way | `bytes_moved / 166 GB/s` |
| HBM round trip | `2 * bytes_moved / 166 GB/s` |
| RIU optimistic | `byte_hops / 333 GB/s` |
| RIU one-direction | `byte_hops / 166 GB/s` |
| balanced local LX | `2 * bytes_moved / (32 * 140 GB/s)` |
| single-core local LX | `2 * bytes_moved / 140 GB/s` |

The local artifacts are saved under:

- `artifacts/restickify_hierarchy_sweep/main/hierarchy_rows.csv`
- `artifacts/restickify_hierarchy_sweep/main/hierarchy_rows.jsonl`
- `artifacts/restickify_hierarchy_sweep/main/hierarchy_pairs.csv`
- `artifacts/restickify_hierarchy_sweep/main/control_deltas.csv`

## Probe Tool

This stage adds `tools/restickify_hierarchy_sweep.py`. It wraps the existing
`restickify_scenario_probe.py` machinery with a test-family taxonomy and
hardware-bound columns.

One implementation detail matters: Torch-Spyre's config module caches env vars
at import time. The tool therefore patches both the environment and the live
`torch_spyre._inductor.config` booleans for Stage 3B mode, mirroring the way
the existing probe already patches ring telemetry config.

Example command:

```sh
python3.12 -u tools/restickify_hierarchy_sweep.py \
  --case adds_then_matmul_x \
  --size 2048 \
  --mode baseline \
  --mode stage3b \
  --time \
  --warmup 5 \
  --iters 10 \
  --skip-correctness \
  --output-dir /tmp/restickify-hierarchy-sweep
```

## Coverage

The main sweep covered 12 cases, 4 sizes (`128`, `512`, `1024`, `2048`), and
two modes:

| Mode | Flags |
|---|---|
| Baseline | `SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=0`, `SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=0` |
| Stage 3B | `SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1`, `SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1` |

The main corrected summary contains 95 successful rows. One Stage 3B
`pointwise_control` row at size `128` timed out and was excluded; it has no
restickify and is not material to the hierarchy classification.

A separate 4D smoke covered `transpose_4d_chain` at sizes `32` and `64`. Size
`64` compiled and produced one `graph_input_or_weight` restickify with zero
exact byte-hops. Size `32` failed with an `InductorError`.

## Totals

| Mode | Rows | Restickifies | Bytes moved | Exact byte-hops |
|---|---:|---:|---:|---:|
| Baseline | 48 | 56 | 154,599,424 | 159,825,920 |
| Stage 3B | 47 | 56 | 154,599,424 | 3,981,312 |

Stage 3B preserved restickify count and bytes moved, while reducing exact
byte-hops by 155,844,608 byte-hops across the comparable rows.

Source categories were unchanged:

| Source category | Count |
|---|---:|
| `graph_input_or_weight` | 44 |
| `in_graph_computed` | 12 |

Skip reasons were also unchanged:

| Skip reason | Count |
|---|---:|
| `graph-input-or-missing-producer` | 44 |
| `incomplete-symbol-map` | 4 |

## In-Graph RIU Signal

The strongest RIU signal remains the `adds_then_matmul` family. These cases
contain one graph-input restickify and one eligible in-graph restickify. Stage
3B does not remove either restickify, but it aligns ownership for the in-graph
one.

| Case | Size | Bytes moved | Baseline byte-hops | Stage 3B byte-hops | Observed delta | RIU 333 GB/s bound | RIU 166 GB/s bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| `adds_then_matmul_x` | 2048 | 16,777,216 | 67,108,864 | 0 | 53.6 us | 201.5 us | 404.3 us |
| `adds_then_matmul_y_long_chain` | 2048 | 16,777,216 | 67,108,864 | 0 | 43.6 us | 201.5 us | 404.3 us |
| `adds_then_matmul_x` | 1024 | 4,194,304 | 11,141,120 | 1,048,576 | -5.1 us | 30.3 us | 60.8 us |
| `adds_then_matmul_y_long_chain` | 1024 | 4,194,304 | 11,141,120 | 1,048,576 | 0.0 us | 30.3 us | 60.8 us |
| `adds_then_matmul_x` | 512 | 1,048,576 | 1,376,256 | 655,360 | 6.4 us | 2.2 us | 4.3 us |
| `adds_then_matmul_y_long_chain` | 512 | 1,048,576 | 1,376,256 | 655,360 | 5.5 us | 2.2 us | 4.3 us |

The 2048 rows are directionally consistent with the RIU locality model: the
same bytes are moved, the modeled cross-core byte-hop term drops to zero, and
the kernel gets faster. The observed latency delta is much smaller than the
serialized RIU lower-bound estimate, which is expected because byte-hop
telemetry is ownership geometry, not a serialized traffic counter; real
execution can overlap movement with compute and other memory operations.

## Graph-Input And Weight Boundary Signal

Graph-input and weight restickifies have bytes moved but zero exact byte-hops:
the compiler cannot attribute them to an in-graph producer core map. Their
extra latency versus matched-layout controls scales much more like a global
memory copy than like balanced local LX traffic.

Baseline examples:

| Case | Size | Bytes moved | Observed extra vs control | HBM one-way | HBM round trip | Balanced local LX |
|---|---:|---:|---:|---:|---:|---:|
| `pointwise_transpose_add` | 512 | 524,288 | 10.9 us | 3.2 us | 6.3 us | 0.234 us |
| `pointwise_transpose_add` | 1024 | 2,097,152 | 41.5 us | 12.6 us | 25.3 us | 0.936 us |
| `pointwise_transpose_add` | 2048 | 8,388,608 | 157.1 us | 50.5 us | 101.1 us | 3.745 us |
| `matmul_lhs_wrong_stick` | 2048 | 8,388,608 | 146.8 us | 50.5 us | 101.1 us | 3.745 us |
| `matmul_rhs_wrong_stick` | 2048 | 8,388,608 | 146.6 us | 50.5 us | 101.1 us | 3.745 us |

These rows do not prove that every graph-input restickify performs a full HBM
round trip, but they strongly argue against the "pure balanced local LX" model:
the measured overhead is tens to hundreds of microseconds, while the balanced
local-LX lower bound is below 4 us even at size 2048.

## What The Three Paths Mean

All three proposed paths are valid restickify traffic hypotheses, but they
apply to different compiler-visible situations:

| Path | Valid when | What we can verify now |
|---|---|---|
| HBM/global boundary | source is a graph input, weight, constant, persistent state, or otherwise produced outside the compiled graph | source kind is `graph_input_or_weight`; bytes scale with HBM-like latency; Stage 3B cannot reduce byte-hops |
| LX-to-LX poor locality | source is `in_graph_computed`, producer ownership and restickify ownership differ | exact byte-hop telemetry is nonzero; Stage 3B can reduce it if mapping is compatible |
| LX-to-LX local/optimal | source is `in_graph_computed`, same physical cores own the same logical regions | exact byte-hop telemetry is zero; work may still include local layout rewriting |

The main limiter is not an explicit "do not use LX-to-LX" rule. The limiter is
whether the value is live and visible inside the compiled schedule. If the
restickify source is outside the compiled graph, or if an intermediate crosses a
kernel/global-memory boundary, the compiler has no producer core ownership to
reuse. In that situation the restickify op operates on the tensor's existing
stored layout, and Stage 3B has nothing to align.

## Conclusion

This sweep gives a more precise framing:

- Graph-input and weight restickifies are common in these tests and look
  bandwidth-like at the HBM/global-memory scale.
- Eligible in-graph restickifies are narrower, but their byte-hop term is real
  and can be reduced without changing restickify count or bytes moved.
- Stage 3B is therefore a valid locality prototype, but it only addresses the
  LX-to-LX poor-locality class.
- The higher-value next optimization family is still input, weight, and
  persistent-state layout handling, because those are the cases the compiler
  currently classifies as `graph_input_or_weight`.

Direct confirmation of the physical fabric still requires profiler or hardware
counters. The next measurement should pair this compiler telemetry with
PrivateUse1/libaiupti traces or RIU/HBM counters once the profiler path can
capture kernel-level AIU activity reliably.

No PR was created and no merge was performed.
