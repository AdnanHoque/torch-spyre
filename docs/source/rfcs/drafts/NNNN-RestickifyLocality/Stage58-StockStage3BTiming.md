# Stage 58: Stock Template Stage 3B Timing

## Summary

Stage 58 reran the high-signal `adds_then_matmul` size `2048` timing guardrail
after Stage 57 proved that stock Deeptools templates retire the minimal
`ReStickifyOpHBM` bundle.

The experiment used:

```sh
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export SENCORES=32
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```

and compared:

```text
baseline: SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=0
          SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=0

Stage 3B: SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1
          SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1
```

Each mode ran in a separate Python process with `warmup=5` and `iters=30`. The
three repeats alternated order:

```text
baseline, Stage 3B, Stage 3B, baseline, baseline, Stage 3B
```

## Result

The byte-hop guardrail stayed exactly stable:

| mode | restickifies | bytes moved | byte-hops |
| --- | ---: | ---: | ---: |
| baseline | 2 | 16,777,216 | 67,108,864 |
| Stage 3B | 2 | 16,777,216 | 0 |

Timing:

| repeat | baseline median ms | Stage 3B median ms | speedup |
| --- | ---: | ---: | ---: |
| 1 | 1.549 | 1.476 | 1.050x |
| 2 | 1.565 | 1.483 | 1.055x |
| 3 | 1.555 | 1.464 | 1.062x |

Median of medians:

```text
baseline: 1.555 ms
Stage 3B: 1.476 ms
delta:    0.080 ms
speedup:  1.054x
```

## Hardware-Bandwidth Sanity

For the eligible in-graph row, baseline telemetry reports:

```text
bytes moved: 8,388,608
byte-hops:   67,108,864
avg hops:    8.0
max hops:    16
```

Using the RIU numbers as simple plausibility bounds:

```text
8,388,608 bytes / 166 GB/s  ~= 0.051 ms
8,388,608 bytes / 333 GB/s  ~= 0.025 ms
67,108,864 byte-hops / 333 GB/s ~= 0.202 ms
67,108,864 byte-hops / 166 GB/s ~= 0.404 ms
```

The observed median delta is about `0.080 ms`. That is directionally plausible:
it is larger than a single 8 MiB transfer at peak RIU bandwidth, but smaller
than treating every modeled byte-hop as serialized on one global bottleneck.

The important caveat is that `byte_hops` is a compiler locality model, not a
hardware counter. The ring has multiple links and two directions, so byte-hop
sums can be served with spatial concurrency. Whole-kernel timing also includes
launch, scheduling, LX/SFP/PT/PE work, and any HBM/LX movement that Stage 3B does
not eliminate.

## Interpretation

This is the strongest current evidence for the narrow Stage 3B claim:

- stock templates retire cleanly;
- restickify count is unchanged;
- bytes moved are unchanged;
- eligible in-graph modeled byte-hops drop from `67,108,864` to `0`;
- repeated runtime improves by about `5.4%` on the high-signal shape.

This still does not prove fabric-level RIU traffic reduction. It proves that the
compiler can make a locality-preserving ownership choice for an eligible
in-graph restickify and that the high-signal microbenchmark becomes faster in a
repeatable way.

## Next Step

Use the same stock-template setup to measure whether this win generalizes beyond
the single square `2048` case:

1. sweep `adds_then_matmul` sizes around the transition point;
2. repeat only cases with nonzero in-graph byte-hops;
3. keep the RIU bandwidth comparison in the summary;
4. separately continue the profiler/counter path for direct RIU/HBM evidence.
