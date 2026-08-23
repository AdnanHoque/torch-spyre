# Gemma-4 real-layer schedule comparison

This is a matched device comparison of the routed-expert FFN for Gemma-4
layer 0 at `E=128`, `T=512`, `H=2816`, and `F=704`.

Inputs:

- real layer-0 gate/up/down weights from the Gemma-4 checkpoint
- real captured layer-0 activation and routing values for 64 rows, repeated
  eight times to exercise the exact T=512 geometry
- contiguous token-major routing storage `[T,E,1]`

This is a layer-level schedule comparison. It is not a full-model prefill run:
the 512 rows repeat one captured 64-row payload, and the production router is
outside the compiled region.

## Compared schedules

Optimized:

```text
gate/up:       T8 x reduction-H4
GELU/multiply: T32
down/tail:     T8 x output-H4
```

Common row:

```text
all operations: T32; every other split is 1
```

Each arm was compiled in a separate process with a fresh compiler cache. The
generated source and bundle hashes differ, so this is not compiler-cache reuse.

## Results

| Result | Optimized | Common row |
| --- | ---: | ---: |
| Median of 7 synchronized calls | 38.359 ms | 45.634 ms |
| rel-L2 vs sparse FP32 reference | 0.008368 | 0.008395 |
| cosine vs sparse FP32 reference | 1.000022 | 1.000018 |
| max absolute error | 0.015363 | 0.013831 |

The optimized schedule is `1.1897x` faster, or `15.9%` lower latency. The two
device outputs agree with each other at rel-L2 `0.003497`.

Both programs emit one 128-trip expert loop, three BMMs, zero activation
`hbm_pool` allocations, zero HBM restickifies, and one final drain. The
optimized bundle additionally emits three LX-only shuffles to bridge its faster
matmul ownerships. The common-row bundle emits none.

## Environment boundary

- pod: `adnan-cdx-spyre-dev-pf`
- compiler overlay: `/tmp/gemma4-full-edge-composite-20260822`
- local compiler worktree: `/tmp/gemma4-edge-composite.QCTXjm`
- branch: `ah/gemma4-edge-composite-20260822`
- inherited head: `bf25a477`
- `DXP_LX_FRAC_AVAIL=0.2`
- `TORCH_SPYRE_NATIVE_PACKER=0`

The compiler changes remain uncommitted and unpushed.
