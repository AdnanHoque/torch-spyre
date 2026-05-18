# Stage 29: Core-Continuity Runtime Validation

## Summary

Stage 28 showed that a default-off core-division continuity prototype can reduce
modeled producer-consumer RIU byte-hops for exact in-graph pointwise edges. This
stage asks the next question: does the modeled byte-hop reduction translate into
runtime improvement?

The short answer is **not yet**. The initial timing sweep found real modeled
byte-hop reductions, but runtime was mostly flat or slower. The sweep also found
one important compiler-policy bug: the prototype sometimes traded away core
parallelism for locality.

That bug is now fixed by requiring the preferred locality dimension to preserve
the best available output-dimension split count. In simple terms: do not use
fewer cores just to save modeled ring hops.

## Experiment

Modes:

- baseline: `SPYRE_ALIGN_CORE_DIVISION_CONTINUITY=0`
- prototype: `SPYRE_ALIGN_CORE_DIVISION_CONTINUITY=1`

Common setup:

```sh
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=0
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=0
SPYRE_CORE_CONTINUITY_TELEMETRY=1
SPYRE_RESTICKIFY_RING_TELEMETRY=1
SENCORES=32
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```

Probe command:

```sh
python tools/restickify_scenario_probe.py \
  --case pointwise_transpose_add \
  --case transpose_chain \
  --case fanout_diamond \
  --case adds_then_matmul \
  --size 1024 \
  --size 1536 \
  --size 2048 \
  --size 3072 \
  --ring-telemetry \
  --skip-correctness \
  --time \
  --warmup 5 \
  --iters 30 \
  --fail-on-error
```

Initial run:

```text
/tmp/restickify-stage29-runtime-1779064134
artifacts/stage29_runtime_validation/
```

The run used three alternating repeats:

- repeat 1: baseline, prototype
- repeat 2: prototype, baseline
- repeat 3: baseline, prototype

All 96 timed rows completed with zero probe errors.

## Initial Runtime Results

Median below is the median of the three per-run medians.

| Case | Size | Base ms | Prototype ms | Speedup | Ring hops | Core-continuity hops |
|---|---:|---:|---:|---:|---:|---:|
| `adds_then_matmul` | 1024 | 0.354 | 0.353 | 1.003x | 11,141,120 -> 11,141,120 | 22,282,240 -> 22,282,240 |
| `adds_then_matmul` | 1536 | 0.683 | 0.735 | 0.929x | 37,748,736 -> 0 | 75,497,472 -> 0 |
| `adds_then_matmul` | 2048 | 1.534 | 1.519 | 1.010x | 67,108,864 -> 0 | 134,217,728 -> 0 |
| `adds_then_matmul` | 3072 | 3.336 | 3.406 | 0.979x | 150,994,944 -> 0 | 301,989,888 -> 0 |
| `fanout_diamond` | 1024 | 0.373 | 0.374 | 0.997x | 0 -> 0 | 22,282,240 -> 22,282,240 |
| `fanout_diamond` | 1536 | 0.730 | 0.856 | 0.853x | 0 -> 0 | 75,497,472 -> 0 |
| `fanout_diamond` | 2048 | 1.389 | 1.397 | 0.994x | 0 -> 0 | 134,217,728 -> 0 |
| `fanout_diamond` | 3072 | 2.952 | 3.055 | 0.966x | 0 -> 0 | 301,989,888 -> 0 |
| `pointwise_transpose_add` | 1024 | 0.145 | 0.144 | 1.007x | 0 -> 0 | 11,141,120 -> 11,141,120 |
| `pointwise_transpose_add` | 1536 | 0.268 | 0.293 | 0.915x | 0 -> 0 | 37,748,736 -> 0 |
| `pointwise_transpose_add` | 2048 | 0.417 | 0.464 | 0.899x | 0 -> 0 | 67,108,864 -> 0 |
| `pointwise_transpose_add` | 3072 | 0.970 | 1.059 | 0.916x | 0 -> 0 | 150,994,944 -> 0 |
| `transpose_chain` | 1024 | 0.215 | 0.214 | 1.005x | 0 -> 0 | 11,141,120 -> 11,141,120 |
| `transpose_chain` | 1536 | 0.404 | 0.429 | 0.942x | 0 -> 0 | 37,748,736 -> 0 |
| `transpose_chain` | 2048 | 0.754 | 0.736 | 1.024x | 0 -> 0 | 67,108,864 -> 0 |
| `transpose_chain` | 3072 | 1.590 | 1.659 | 0.958x | 0 -> 0 | 150,994,944 -> 0 |

## What Went Wrong

The prototype originally asked: "can the consumer split along the producer's
dominant split dimension?" That was too weak.

For sizes like `1536` and `3072`, the aligned dimension only supported a
24-way split, while the default consumer dimension supported a 32-way split.
The prototype saved modeled byte-hops by reducing the number of active cores.
That made runtime worse.

Example:

```text
pointwise_transpose_add size 1536
baseline: producer d0:24, consumer d1:32, byte-hops 37,748,736
initial prototype: producer d0:24, consumer d0:24, byte-hops 0
```

The byte-hop model was doing what it was asked to do, but the optimization was
making the wrong tradeoff.

## Fix

`maybe_prioritize_core_continuity_output_dims` now skips alignment when the
preferred locality dimension would use fewer cores than another output
dimension can use.

This preserves the clean 2048 cases, where the producer-corresponding dimension
can still use 32 cores, and skips the 1536/3072 cases where locality would cost
parallelism.

## Gate Validation

Follow-up run:

```text
/tmp/restickify-stage29-parallelism-gate-1779078415
artifacts/stage29_parallelism_gate/
```

| Case | Size | Base ms | Prototype ms | Ring hops | Core-continuity hops |
|---|---:|---:|---:|---:|---:|
| `adds_then_matmul` | 1536 | 0.721 | 0.728 | 37,748,736 -> 37,748,736 | 75,497,472 -> 75,497,472 |
| `adds_then_matmul` | 2048 | 1.458 | 1.469 | 67,108,864 -> 0 | 134,217,728 -> 0 |
| `adds_then_matmul` | 3072 | 3.171 | 3.202 | 150,994,944 -> 150,994,944 | 301,989,888 -> 301,989,888 |
| `fanout_diamond` | 1536 | 0.705 | 0.714 | 0 -> 0 | 75,497,472 -> 75,497,472 |
| `fanout_diamond` | 2048 | 1.424 | 1.430 | 0 -> 0 | 134,217,728 -> 0 |
| `fanout_diamond` | 3072 | 2.837 | 3.030 | 0 -> 0 | 301,989,888 -> 301,989,888 |
| `pointwise_transpose_add` | 1536 | 0.253 | 0.257 | 0 -> 0 | 37,748,736 -> 37,748,736 |
| `pointwise_transpose_add` | 2048 | 0.455 | 0.464 | 0 -> 0 | 67,108,864 -> 0 |
| `pointwise_transpose_add` | 3072 | 0.857 | 0.846 | 0 -> 0 | 150,994,944 -> 150,994,944 |
| `transpose_chain` | 1536 | 0.387 | 0.391 | 0 -> 0 | 37,748,736 -> 37,748,736 |
| `transpose_chain` | 2048 | 0.705 | 0.722 | 0 -> 0 | 67,108,864 -> 0 |
| `transpose_chain` | 3072 | 1.392 | 1.433 | 0 -> 0 | 150,994,944 -> 150,994,944 |

The gate validation is only one repeat, so it is not a final timing claim. Its
purpose was to verify policy behavior:

- 1536 and 3072 no longer trade 32-core parallelism for 24-core locality.
- 2048 remains eligible and still reaches zero modeled byte-hops.
- Runtime is still flat to slightly negative in this short rerun.

## Validation

After the gate fix:

```text
python -m py_compile torch_spyre/_inductor/work_division.py
python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q
25 passed in 0.15s
```

The default-off full restickify regression had already passed for the same
Stage 28 branch:

```text
python -m pytest tests/inductor/test_restickify.py -q
97 passed in 70.42s
```

The gate change only affects the default-off prototype path guarded by
`SPYRE_ALIGN_CORE_DIVISION_CONTINUITY=1`.

## Conclusion

This stage changes the recommendation.

Stage 28 is useful as compiler infrastructure and telemetry, but the current
optimization should **not** be framed as a performance win. The measurements
show that modeled RIU byte-hop reduction alone is not enough. We must preserve
parallelism first, and even when we do, the 2048 runtime signal is weak.

The next productive direction is either:

- use kernel/fabric counters to see whether these byte-hops are actually on the
  critical path, or
- move to an optimization where the compiler can reduce modeled traffic without
  changing work split shape, such as certified core mapping overrides for cases
  that already have identical split factors.

For now, keep `SPYRE_ALIGN_CORE_DIVISION_CONTINUITY` default-off.
