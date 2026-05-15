# Stage 3D Shape-Sweep Results

This note records the first fused model-slice shape sweep for the Restickify
Locality RFC. The goal was to answer whether sequence length, prefill, decode,
or batched decode changes the Stage 3B signal.

The sweep is still synthetic. It is designed to keep several operations inside
one compiled graph so the telemetry can observe in-graph
producer-to-restickify edges. It is not an end-to-end Granite benchmark.

## Probe Extension

The scenario probe now includes fused forward-looking model slices:

- `prefill_projection_join`
- `decode_projection_join`
- `attention_prefill_no_softmax`
- `attention_decode_no_softmax`
- `mamba_chunk_projection_join`
- `moe_two_expert_join`

The projection-join slices use a token-by-hidden shape and intentionally create
the known layout-boundary pattern:

```python
(x + y.t() + z.t()) @ w
```

This is a model-slice proxy for prefill or batched-decode projection work. The
first run used `hidden=512`.

## Projection Join: Prefill And Batched Decode

The projection-join family produced the clearest shape-dependent signal. Prefill
and batched-decode use the same tensor shape in this proxy, so their telemetry
matched when active token counts matched.

| Active tokens | Baseline byte-hops | Stage 3B byte-hops | Reduction | Baseline avg/max hops | Stage 3B avg/max hops |
|---:|---:|---:|---:|---:|---:|
| 128 | 1,048,576 | 1,048,576 | 0.0% | 4.000 / 16 | 4.000 / 16 |
| 512 | 1,376,256 | 655,360 | 52.4% | 1.312 / 7 | 0.625 / 3 |
| 1024 | 5,570,560 | 1,310,720 | 76.5% | 2.656 / 15 | 0.625 / 3 |
| 2048 | 16,777,216 | 2,621,440 | 84.4% | 4.000 / 16 | 0.625 / 3 |
| 4096 | 33,554,432 | 5,242,880 | 84.4% | 4.000 / 16 | 0.625 / 3 |

The key split transition is the same as earlier Stage 3B results:

- baseline often assigns the restickify split to the token dimension, for
  example `d0:32`
- Stage 3B steers toward the producer-corresponding hidden dimension, for
  example `d0:4,d1:8`

In this rectangular `tokens x hidden=512` shape, Stage 3B does not drive
byte-hops to zero. It still caps the observed max hop at `3` for larger token
counts.

## Decode Shape Behavior

The tiny-token decode proxy did not compile cleanly for active token counts
below 128:

| Active tokens | Result |
|---:|---|
| 1 | stick incompatibility for pointwise op |
| 8 | Deeptools scheduler candidate failure |
| 32 | Deeptools scheduler candidate failure |
| 128 | compiled; same telemetry as prefill token count 128 |
| 512 | compiled; same telemetry as prefill token count 512 |

This means the current proxy is useful for batched decode or session-packed
decode, but not yet for single-token decode. Single-token decode needs either a
different synthetic slice or padding/packing behavior that resembles the real
runtime.

## Attention, Mamba, And MoE Proxies

The first attention, Mamba, and MoE proxies did not produce eligible nonzero
in-graph byte-hop telemetry.

| Case | Sizes | Result |
|---|---|---|
| `attention_prefill_no_softmax` | 128, 512 | compiled, 2 restickifies, all graph-input/missing-producer skips |
| `attention_decode_no_softmax` | 128, 512, 2048 KV length | failed with pointwise stick incompatibility |
| `mamba_chunk_projection_join` | 128, 512, 2048 | compiled, 1 restickify, graph-input/missing-producer skip |
| `moe_two_expert_join` | 128, 512, 2048 | compiled, 2 restickifies, graph-input/missing-producer skips |

This is not a final architectural conclusion. It means these first proxies did
not create the specific producer-to-restickify shape that Stage 3B optimizes.
They need better fused slices before they can answer the model-impact question.

## Runtime Smoke

The positive projection-join case was timed with `hidden=512`, `warmup=5`, and
`iters=20`.

| Active tokens | Baseline median ms | Stage 3B median ms | Median speedup | Byte-hop reduction |
|---:|---:|---:|---:|---:|
| 512 | 0.127587 | 0.131332 | 0.971x | 52.4% |
| 2048 | 0.325854 | 0.322931 | 1.009x | 84.4% |
| 4096 | 0.573630 | 0.568277 | 1.009x | 84.4% |

The timing result is modest. The telemetry shows a large locality improvement,
but at `hidden=512` the local-kernel runtime gain is only around 1% at larger
token counts and negative/noise at 512 tokens.

## Interpretation

The shape sweep supports a careful conclusion:

- Sweeping shape matters. The Stage 3B byte-hop reduction improves as active
  tokens grow.
- Prefill and batched decode are the more promising directions because they
  expose enough active tokens to make physical locality visible.
- Single-token decode is not yet measured by this proxy.
- Attention/Mamba/MoE still need better fused proxies; the first simple slices
  mostly produced graph-input-sourced restickifies.
- Even when byte-hops drop by 84.4%, local runtime gain can be small if the
  restickify movement is not a large enough fraction of kernel time.

The next experiment should either increase hidden size, use a projection slice
where restickify is a larger runtime fraction, or construct more faithful fused
attention/Mamba/MoE slices that create eligible in-graph restickifies.
