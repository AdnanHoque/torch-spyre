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

All measurements below are synchronized five-call block medians on cdx. Every
accepted control passed correctness and proved three BMMs, direct expert-weight
streaming, all internal compute in LX, zero HBM-pool allocations, zero HBM
restickify operations, and one final HBM output.

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
