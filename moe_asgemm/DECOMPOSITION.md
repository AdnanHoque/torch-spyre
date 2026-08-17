# Generated-kernel decomposition

## Why this is an experimental decomposition

The runtime exposes the entire activation-stationary bundle as one
`ComputeOnDevice` job. It does not expose twelve independently timestamped
SDSCs. A claimed direct per-SDSC timing table would therefore be fabricated.

The retained decomposition uses three measurable controls instead:

1. an expert-count sweep of the exact accepted full graph;
2. full-graph pointwise substitutions that keep all three matmuls and the same
   zero-spill execution contract; and
3. standalone M32 gate/down matmuls, reported as proxies because they include
   an extra job launch and extra HBM traffic.

All measurements below are synchronized five-call block medians on cdx. The
expert-count and pointwise controls passed correctness and retained the full
three-BMM, direct-weight-streaming, zero-spill execution contract. The
standalone leaf controls passed their projection-specific structural gates but
necessarily add a job boundary and HBM activation traffic.

## Expert-count sweep

| Experts | Median per call |
|---:|---:|
| 2 | 0.992 ms |
| 8 | 2.938 ms |
| 32 | 11.118 ms |
| 64 | 21.477 ms |
| 128 | 42.408 ms |

The least-squares fit is:

```text
time(E) = 0.408112 ms + E * 0.328597 ms
R squared = 0.999949
```

This is the most important result. The program is already almost perfectly
linear in expert-body work. Only about one percent of the E128 runtime appears
as fixed bundle/job cost.

## Pointwise substitution controls

The controls change one operation but preserve the full three-matmul graph and
the accepted placement. Slopes use matched E2 and E32 measurements.

| Control | Slope per expert | Delta from full |
|---|---:|---:|
| Full graph | 338.537 us | — |
| No GELU | 337.013 us | GELU adds 1.524 us |
| Hidden add instead of multiply | 337.722 us | multiply-minus-add is 0.815 us |
| Route add instead of multiply | 337.515 us | multiply-minus-add is 1.022 us |

These are substitution deltas, not absolute operation times. They do show that
the visible choice among these LX pointwise operations is a small part of the
expert slope.

## Matmul leaf proxies

Standalone M32 controls measured:

| Projection | Median | Extra HBM bytes versus in-loop BMM |
|---|---:|---:|
| Gate/up shape `512x2816 @ 2816x704` | 0.571 ms | 3,604,480 |
| Down shape `512x704 @ 704x2816` | 0.469 ms | 3,604,480 |

Each standalone call pays a separate job cost and reads/writes the activation
through HBM. The real expert loop pays one job cost for all 128 experts and
keeps activation operands/results in LX. Therefore these are not exact BMM
timestamps.

For a calibrated proxy only, subtract the measured full-program fixed term and
the extra bytes at 150 GB/s:

```text
gate/up proxy      0.139330 ms each
down proxy         0.037079 ms
three-matmul sum   0.315739 ms per expert
measured slope     0.328597 ms per expert
residual proxy     0.012859 ms per expert
```

At E128, fixed cost plus that residual is 2.054 ms, or 4.843 percent of the
42.408 ms clean block baseline.

## Per-SDSC map and calibrated attribution

The accepted bundle contains twelve SDSCs.  SDSCs 2 through 10 execute once
per expert inside the static 128-trip loop; SDSCs 0, 1, and 11 execute once per
logical FFN call.

| SDSC | Role | Frequency | Storage | Timing attribution at E128 |
|---:|---|---:|---|---:|
| 0 | X preheader copy | once | HBM to LX | included in fixed/residual |
| 1 | accumulator fill | once | LX | included in fixed/residual |
| 2 | gate matmul | 128 | HBM weight, LX activation/output | 17.834 ms proxy |
| 3 | tanh GELU | 128 | LX | substitution delta 0.195 ms total |
| 4 | up matmul | 128 | HBM weight, LX activation/output | 17.834 ms proxy |
| 5 | gate-times-up | 128 | LX | multiply-minus-add delta 0.104 ms total |
| 6 | down matmul | 128 | HBM weight, LX activation/output | 4.746 ms proxy |
| 7 | runtime alpha copy | 128 | HBM to LX | included in fixed/residual |
| 8 | post-down alpha multiply | 128 | LX | multiply-minus-add delta 0.131 ms total |
| 9 | unit local contribution identity | 128 | LX | included in fixed/residual |
| 10 | expert accumulator add | 128 | LX | included in fixed/residual |
| 11 | final output drain | once | LX to HBM | included in fixed/residual |

The three adjusted matmul proxies sum to 40.415 ms.  The expert-fit intercept
plus non-matmul residual is 2.054 ms.  Because these are independently fitted
controls, they are not an exact additive partition of the observed 42.408 ms;
the resulting overage is about 0.061 ms.  The defensible rounded attribution
is therefore:

```text
matmul work                         about 40.4 ms, about 95 percent
fixed + sequencing + pointwise     about  2.1 ms, about  5 percent
```

The three pointwise rows report matched substitution deltas, not standalone
operation latency.  No exact time is assigned to SDSCs 0, 1, 7, 9, 10, or 11
because the runtime does not timestamp those calls independently.

## Interpretation

The evidence supports this hypothesis:

- the generated program has already removed the large launch and HBM
  intermediate costs that a native composite kernel would normally target;
- the three matmul primitives and streamed weights dominate the expert slope;
- pointwise sequencing is measurable but small; and
- a native DDL may improve constants, but a large multi-fold gain is not the
  expected outcome at this shape.

This is not physical bus telemetry and not an exact internal-SDSC trace. The
full retained machine-readable analysis is
`moe_asgemm/artifacts/decomposition/analysis.json`.

## Retained instruments

```text
experiments/dasx_component_sweep_probe.py
experiments/dasx_matmul_leaf_probe.py
moe_asgemm/tools/analyze_dasx_decomposition.py
```

The first leaf-only decomposition attempts are excluded because they spilled
through HBM when a single matmul consumer could not share the accepted
ownership map. No timing from those invalid controls is used.
