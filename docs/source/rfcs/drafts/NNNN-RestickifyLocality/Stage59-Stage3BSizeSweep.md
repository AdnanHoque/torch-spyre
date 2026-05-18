# Stage 59: Stage 3B Size Sweep

## Summary

Stage 59 swept the proven `adds_then_matmul` pattern across square sizes with
stock Deeptools templates:

```text
512, 1024, 1536, 2048, 3072
```

The goal was to test whether the Stage 3B byte-hop reduction turns into a broad
runtime curve or remains a narrow shape-dependent win.

The answer is shape-dependent: Stage 3B consistently reduces modeled in-graph
byte-hops once the restickify has enough work slices to steer, but only the
`2048` case shows a meaningful repeated runtime improvement in this sweep.

## Telemetry Sweep

The telemetry-only sweep also included `128`, `256`, and `768`:

| size | baseline byte-hops | Stage 3B byte-hops | reduction | baseline avg/max hops | Stage 3B avg/max hops |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 286,720 | 286,720 | 0.0% | 4.375 / 16 | 4.375 / 16 |
| 256 | 1,048,576 | 1,048,576 | 0.0% | 4.000 / 16 | 4.000 / 16 |
| 512 | 1,376,256 | 655,360 | 52.4% | 1.312 / 7 | 0.625 / 3 |
| 768 | 6,930,432 | 4,767,744 | 31.2% | 2.938 / 16 | 2.021 / 9 |
| 1024 | 11,141,120 | 1,048,576 | 90.6% | 2.656 / 15 | 0.250 / 1 |
| 1536 | 37,748,736 | 18,874,368 | 50.0% | 4.000 / 16 | 2.000 / 8 |
| 2048 | 67,108,864 | 0 | 100.0% | 4.000 / 16 | 0.000 / 0 |
| 3072 | 150,994,944 | 75,497,472 | 50.0% | 4.000 / 16 | 2.000 / 8 |

Restickify count stayed `2` for every row in both modes. Bytes moved stayed
unchanged for every row.

The eligible in-graph split choices explain the curve:

| size | producer split | baseline restickify split | Stage 3B restickify split |
| ---: | --- | --- | --- |
| 128 | `d1:32` | `d0:2,d1:2` | `d0:2,d1:2` |
| 256 | `d1:32` | `d0:4,d1:4` | `d0:4,d1:4` |
| 512 | `d1:32` | `d0:8,d1:4` | `d0:4,d1:8` |
| 768 | `d1:32` | `d0:12,d1:2` | `d0:2,d1:12` |
| 1024 | `d1:32` | `d0:16,d1:2` | `d0:2,d1:16` |
| 1536 | `d1:32` | `d0:24` | `d1:24` |
| 2048 | `d1:32` | `d0:32` | `d1:32` |
| 3072 | `d1:32` | `d0:24` | `d1:24` |

The clean zero-hop case appears when the producer and restickify can both use
the same full `d1:32` ownership.

## Timing Sweep

Timing used `warmup=5`, `iters=30`, and three repeats per mode, alternating
baseline and Stage 3B process order.

| size | baseline median ms | Stage 3B median ms | speedup |
| ---: | ---: | ---: | ---: |
| 512 | 0.130 | 0.130 | 1.000x |
| 1024 | 0.352 | 0.349 | 1.006x |
| 1536 | 0.688 | 0.681 | 1.010x |
| 2048 | 1.468 | 1.400 | 1.049x |
| 3072 | 3.222 | 3.213 | 1.003x |

The `2048` result remains the only clearly meaningful runtime win. This sweep's
`2048` median speedup is `1.049x`; Stage 58 measured `1.054x` in the single-size
guardrail.

## Bandwidth Sanity

Comparing observed median deltas against a serialized RIU byte-hop bound:

| size | byte-hop reduction | observed delta ms | byte-hop / 333 GB/s | byte-hop / 166 GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 720,896 | 0.000 | 0.002 | 0.004 |
| 1024 | 10,092,544 | 0.002 | 0.030 | 0.061 |
| 1536 | 18,874,368 | 0.007 | 0.057 | 0.114 |
| 2048 | 67,108,864 | 0.068 | 0.202 | 0.404 |
| 3072 | 75,497,472 | 0.009 | 0.227 | 0.455 |

The runtime deltas are smaller than a naive serialized byte-hop model, especially
outside `2048`. That is expected if the RIU traffic is served with spatial
concurrency, if the kernel is dominated by other work, or if the byte-hop model
is a locality proxy rather than a direct fabric-time model.

## Interpretation

Stage 3B is real but narrow:

- It changes physical ownership for eligible in-graph restickifies.
- It can remove modeled byte-hops entirely when producer and restickify split
  factors exactly match after symbol mapping.
- It does not change restickify placement, count, or total bytes moved.
- Runtime benefit is visible only when the restickify locality cost is on the
  critical path for the generated program.

For an upstream-facing proposal, this argues for keeping Stage 3B default-off or
debug-gated until we have broader workload evidence. The strongest production
argument is not "this speeds up all restickify"; it is "this is a conservative
ownership-continuity improvement with a clear locality certificate and a proven
win on one high-signal synthetic case."

## Next Step

Run the same stock-template telemetry and timing style on a second family where
restickify appears naturally, rather than only `adds_then_matmul`. The best next
candidates are:

1. `matmul_then_add`, to test producer-output restickify before pointwise joins;
2. `transpose_chain`, to test view/layout boundary materialization without
   matmul dominating the runtime;
3. `mlp_gated_projection_join`, to test whether the high-signal edge survives
   inside a fused model-block-like graph.
