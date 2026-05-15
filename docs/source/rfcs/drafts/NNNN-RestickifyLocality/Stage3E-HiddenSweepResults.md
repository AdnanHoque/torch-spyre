# Stage 3E Hidden-Size Sweep Results

This note records the follow-up hidden-size sweep for the Restickify Locality
RFC. Stage 3D showed that increasing active tokens makes the projection-join
slice expose more byte-hop opportunity. Stage 3E asks whether increasing hidden
size makes that locality reduction matter more at runtime.

The tested slice is:

```python
(x + y.t() + z.t()) @ w
```

This remains a fused synthetic projection proxy, not an end-to-end model claim.

## Sweep Setup

The sweep used:

- `SPYRE_PROBE_HIDDEN=1024` and `2048`
- active tokens `512` and `2048`
- `warmup=5`, `iters=20`
- baseline: Stage 3B flags off
- candidate: `SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1` and
  `SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1`
- `LX_PLANNING=0`

## Telemetry And Timing

| Hidden | Active tokens | Baseline byte-hops | Stage 3B byte-hops | Reduction | Baseline median ms | Stage 3B median ms | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 512 | 524,288 | 524,288 | 0.0% | 0.228059 | 0.225124 | 1.013x |
| 1024 | 2048 | 33,554,432 | 2,097,152 | 93.8% | 0.661233 | 0.663352 | 0.997x |
| 2048 | 512 | 0 | 0 | 0.0% | 0.477438 | 0.468908 | 1.018x |
| 2048 | 2048 | 67,108,864 | 0 | 100.0% | 1.541940 | 1.491295 | 1.034x |

The split behavior explains the result:

| Hidden | Active tokens | Baseline split | Stage 3B split |
|---:|---:|---|---|
| 1024 | 512 | producer `d1:32`, restickify `d0:2,d1:16` | unchanged |
| 1024 | 2048 | producer `d1:32`, restickify `d0:32` | restickify `d0:2,d1:16` |
| 2048 | 512 | producer `d1:32`, restickify `d1:32` | unchanged |
| 2048 | 2048 | producer `d1:32`, restickify `d0:32` | restickify `d1:32` |

The high-signal point is `hidden=2048`, `active_tokens=2048`: Stage 3B changes
the restickify split from `d0:32` to `d1:32`, exactly matching the producer and
eliminating the measured byte-hop cost.

## Focused Repeat

The `hidden=2048`, `active_tokens=2048` point was repeated three times with
`warmup=5` and `iters=30`.

| Repeat | Baseline median ms | Stage 3B median ms | Speedup | Baseline byte-hops | Stage 3B byte-hops |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.545642 | 1.490371 | 1.037x | 67,108,864 | 0 |
| 2 | 1.569750 | 1.483726 | 1.058x | 67,108,864 | 0 |
| 3 | 1.551877 | 1.461262 | 1.062x | 67,108,864 | 0 |

Summary:

- mean speedup: `1.052x`
- median speedup: `1.058x`
- min/max speedup: `1.037x` / `1.062x`

## Interpretation

This is the strongest result so far:

- the telemetry signal is stable and exact
- Stage 3B eliminates all measured byte-hops for the high-signal shape
- repeated runtime shows a consistent local-kernel speedup around 4-6%

The result also clarifies why shape sweeping matters:

- `hidden=2048`, `tokens=512` already has zero baseline byte-hops because the
  default work distribution chooses `d1:32`
- `hidden=1024`, `tokens=2048` gets a large byte-hop reduction but little timing
  movement, suggesting the restickify locality cost is not always on the
  critical path
- `hidden=2048`, `tokens=2048` is the first shape where byte-hop elimination and
  stable runtime improvement line up

This supports continuing the project, but still as a guarded/locality feature:
the optimization is clearly shape-dependent and should remain telemetry-driven
until fused model slices or real workloads show enough eligible restickify share.

## Recommended Next Step

Build more faithful fused slices around the same high-signal aspect ratio:

- MLP block: projection, activation/gate, projection, residual
- attention block without unsupported softmax first, then with supported
  approximations if needed
- Mamba/MoE slices that force in-graph producer-to-restickify edges rather than
  graph-input-sourced restickifies

Acceptance should stay the same: first prove nonzero eligible baseline
byte-hops, then prove Stage 3B reduction, then run repeated timing.
