# Stage 6: Restickify Telemetry Survey and Profiler Env

This note records the current-main restickify telemetry survey and the separate
profiler-environment bring-up. The restickify measurement used the disposable
pod checkout at `/tmp/torch-spyre-refresh` with `PYTHONPATH` forced to that
source tree. The profiler work used a separate pod workspace at
`/home/adnan-cdx/dt-inductor-profiler`.

## Survey Setup

The survey compared the same source and build in two modes:

| Mode | Flags |
|---|---|
| Baseline | `SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=0`, `SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=0` |
| Stage 3B | `SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1`, `SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1` |

Telemetry was enabled with `SPYRE_RESTICKIFY_RING_TELEMETRY=1` and
`SENCORES=32`. Timed runs used `LX_PLANNING=0`, `warmup=5`, and either
`iters=50` for core cases or `iters=30` for model-ish cases.

The raw summaries are saved locally under:

- `artifacts/restickify_telemetry_survey/summary/all_rows.csv`
- `artifacts/restickify_telemetry_survey/summary/all_rows.jsonl`
- `artifacts/restickify_telemetry_survey/summary/timed_pairs.json`
- `artifacts/restickify_telemetry_survey/summary/timed_selection.tsv`

## Telemetry Totals

The survey produced 45 baseline telemetry rows and 45 Stage 3B telemetry rows.
Both modes generated the same number of restickifies and moved the same number
of bytes.

| Mode | Rows | Restickifies | Bytes moved | Exact byte-hops | Exact rows | Skipped rows |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 45 | 64 | 313,950,208 | 105,078,784 | 8 | 56 |
| Stage 3B | 45 | 64 | 313,950,208 | 7,495,680 | 8 | 56 |

Stage 3B reduced modeled exact byte-hops by 97,583,104 byte-hops, or about
92.9 percent, without changing restickify placement, count, or bytes moved.

The source categories were identical in both modes:

| Source category | Count |
|---|---:|
| `graph_input_or_weight` | 53 |
| `in_graph_computed` | 11 |

The skip reasons were also identical:

| Skip reason | Count |
|---|---:|
| `graph-input-or-missing-producer` | 53 |
| `incomplete-symbol-map` | 3 |

This is the key interpretation point: many restickifies move real bytes, but
most of this survey's rows are not eligible for Stage 3B because the source is
outside the compiled graph or the exact producer-to-restickify symbol map is not
available.

## High-Signal Rows

Top byte-hop rows in baseline:

| Case | Size | Bytes moved | Baseline byte-hops | Stage 3B byte-hops | Result |
|---|---:|---:|---:|---:|---|
| `adds_then_matmul` | 2048 | 16,777,216 | 67,108,864 | 0 | full alignment |
| `prefill_projection_join` | 2048 | 4,194,304 | 16,777,216 | 2,621,440 | partial alignment |
| `decode_projection_join` | 2048 | 4,194,304 | 16,777,216 | 2,621,440 | partial alignment |
| `adds_then_matmul` | 512 | 1,048,576 | 1,376,256 | 655,360 | partial alignment |
| `prefill_projection_join` | 512 | 1,048,576 | 1,376,256 | 655,360 | partial alignment |
| `decode_projection_join` | 512 | 1,048,576 | 1,376,256 | 655,360 | partial alignment |

The long-context smoke, `prefill_projection_join` at size `65536`, moved
134,217,728 restickify bytes but had zero exact byte-hops in both modes. In
this current probe it is dominated by a graph-input/weight boundary, so Stage
3B has no in-graph producer ownership to align.

## Runtime And RIU Bounds

The RIU estimates below treat byte-hop reduction as if each byte-hop serialized
against either 333 GB/s aggregate bi-ring bandwidth or 166 GB/s one-direction
bandwidth. That is a plausibility bound, not a direct hardware measurement:
byte-hop telemetry is compiler ownership geometry, and real execution can
overlap movement with other work.

| Case | Size | Median baseline ms | Median Stage 3B ms | Speedup | Byte-hop delta | 333 GB/s bound | 166 GB/s bound | Observed delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `adds_then_matmul` | 2048 | 1.557 | 1.479 | 1.053x | 67,108,864 | 201.5 us | 404.3 us | 78.1 us |
| `prefill_projection_join` | 2048 | 0.338 | 0.331 | 1.020x | 14,155,776 | 42.5 us | 85.3 us | 6.5 us |
| `decode_projection_join` | 2048 | 0.342 | 0.335 | 1.022x | 14,155,776 | 42.5 us | 85.3 us | 7.4 us |
| `adds_then_matmul` | 512 | 0.129 | 0.130 | 0.989x | 720,896 | 2.2 us | 4.3 us | -1.5 us |
| `prefill_projection_join` | 512 | 0.134 | 0.128 | 1.046x | 720,896 | 2.2 us | 4.3 us | 5.9 us |
| `decode_projection_join` | 512 | 0.129 | 0.130 | 0.991x | 720,896 | 2.2 us | 4.3 us | -1.1 us |

Zero-byte-hop control rows moved substantial bytes but showed only small
positive or negative timing noise, typically within about +/- 12 us. Examples
include `pointwise_transpose_add`, `matmul_lhs_wrong_stick`,
`matmul_rhs_wrong_stick`, `matmul_then_add`, `transpose_chain`,
`fanout_diamond`, `attention_prefill_no_softmax`, and
`attention_score_join_value_projection` at size `2048`.

## Interpretation

Stage 3B is doing the narrow thing it was designed to do:

- It preserves restickify count, placement, and bytes moved.
- It reduces exact ring byte-hops only when there is an in-graph producer and a
  compatible producer-to-restickify ownership map.
- It does not address graph-input, weight, or persistent-state layout
  boundaries.

The best current demo remains `adds_then_matmul` at size `2048`: it has one
graph-input/weight restickify and one eligible in-graph restickify. Stage 3B
does not remove either restickify, but it changes physical work ownership for
the eligible one so the modeled byte-hops drop from 67,108,864 to zero. The
observed local speedup, about 5.3 percent for that kernel-sized probe, is
directionally consistent with the RIU bound but smaller than the bound, which is
expected because byte-hop is not a direct serialized-traffic counter.

The broader survey reinforces the project framing: ring-aware in-graph
restickify alignment is real but narrow. The next higher-value optimization
family is likely input, weight, and persistent-state layout handling, because
most restickify rows in this survey came from `graph_input_or_weight` sources.

## Profiler Env Status

A separate profiler workspace was created at
`/home/adnan-cdx/dt-inductor-profiler` with venv `.venv-py212`.

The fast PyTorch 2.12 path is viable at the registration/build level:

| Check | Result |
|---|---|
| Installed torch | `2.12.0+cu130` |
| `torch/csrc/profiler/standalone/privateuse1_profiler.h` | present |
| Kineto headers | present |
| torch-spyre PR #1856 build with `USE_SPYRE_PROFILER=1` | passed after adding senlib include/lib paths and using `--no-deps` |
| `ProfilerActivity.PrivateUse1` | present |
| Allocation-only Spyre profiler smoke | exported Chrome trace with `__aiu_profiler__` and memory events |
| `acelyzer` on allocation trace | failed because the trace had no deviceProperties metadata |

The first op-level smoke, `torch.ones(..., device="spyre")`, did not complete on
the PyTorch 2.12 profiler env. It failed inside TorchDynamo/FakeTensor with a
no-dispatch assertion before a full trace could be exported. The allocation-only
smoke is therefore enough to show profiler registration is alive, but not enough
to claim kernel-level AIU activity profiling is ready.

Profiler artifacts are saved locally under:

- `artifacts/profiler_env/profiler-build-nodeps.log`
- `artifacts/profiler_env/profiler-smoke-alloc.trace.json`
- `artifacts/profiler_env/acelyzer-smoke.log`

## Validation

Validation used the same `/tmp/torch-spyre-refresh` source path as the survey.

| Check | Result |
|---|---|
| `python -m py_compile tools/restickify_scenario_probe.py` | passed |
| `python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q` | 17 passed |
| selected `tests/inductor/test_restickify.py` families | 10 passed, 87 deselected |

No PR was created and no merge was performed.
