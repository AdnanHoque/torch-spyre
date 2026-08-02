# Torch-Spyre FP8 M=64/M=512 ceiling pass

## Outcome

This DD2-only pass asks whether the complete dynamic FP8 path can reach the
often-quoted `1.6-2.0x` range at `M=64` and `M=512`. It separates the FP8
matmul, activation conversion, output scaling, core-grid choice, and
cross-projection reuse. Static weight packing remains outside every timed FP8
graph.

The answer is shape- and M-dependent:

- At `M=64`, the best complete standalone sum remains `1.441x` over FP16. Even
  the prepacked-activation BMM-plus-scale controls sum to only `1.531x`; raw
  prepacked BMMs sum to `1.619x`. Frontend conversion cleanup alone therefore
  cannot produce a `1.6-2.0x` complete operation at this M. The remaining work
  is inside the FP8 BMM feed/schedule and a genuinely combined packing path.
- At `M=512`, the best standalone sum reaches `1.683x`. Compiling gate and up
  together so they reuse one activation conversion raises the matched grouped
  sum to `1.726x`. Down is already at its optimistic traffic ceiling; Q/O and
  K/V remain limited by dynamic activation conversion, with narrow-output K/V
  also limited by its raw BMM and scale cost.

These are standalone linear-kernel sums, not Granite end-to-end results.

## Complete dynamic FP8 results

Every FP8 row includes per-row activation-scale derivation, activation
normalization and clipping, FP8 packing, the FP8 matmul, and two FP16 output
scale applications. The K/V row is a two-projection graph with one shared
activation conversion; the other rows are one projection each.

### M=64

| Projection | K | N | FP16 us | FP8 us | FP8 speedup |
|---|---:|---:|---:|---:|---:|
| K/V shared pair | 4096 | 1024 | 124.904 | 93.182 | 1.340x |
| Q or O | 4096 | 4096 | 228.363 | 146.749 | 1.556x |
| gate or up | 4096 | 12800 | 707.643 | 517.675 | 1.367x |
| down | 12800 | 4096 | 700.042 | 449.877 | 1.556x |

One K/V pair, two Q/O projections, two gate/up projections, and one down sum
to `2696.958 us` for FP16 and `1871.907 us` for FP8: `1.441x`.

### M=512

| Projection | K | N | FP16 us | FP8 us | FP8 speedup |
|---|---:|---:|---:|---:|---:|
| K/V shared pair | 4096 | 1024 | 188.880 | 143.105 | 1.320x |
| Q or O | 4096 | 4096 | 318.476 | 216.597 | 1.470x |
| gate or up | 4096 | 12800 | 996.369 | 593.423 | 1.679x |
| down | 12800 | 4096 | 1141.845 | 589.406 | 1.937x |

The Q/O result uses the newly revalidated `M4 x N8` grid. The standalone sum
is `3960.417 us` for FP16 and `2352.553 us` for FP8: `1.683x`.

## What remains outside the matmul

The prepacked-activation control still applies both output scales, but excludes
dynamic scale derivation, normalization, clipping, and QFP8MB packing from the
timed graph.

| M | Projection | Complete FP8 us | Prepacked activation + BMM + scales us | Path difference us |
|---:|---|---:|---:|---:|
| 64 | K/V shared pair | 93.182 | 82.778 | 10.403 |
| 64 | Q or O | 146.749 | 149.020 | -2.271 |
| 64 | gate or up | 517.675 | 491.450 | 26.225 |
| 64 | down | 449.877 | 397.807 | 52.071 |
| 512 | K/V shared pair | 143.105 | 123.002 | 20.104 |
| 512 | Q or O | 216.597 | 177.152 | 39.445 |
| 512 | gate or up | 593.423 | 563.514 | 29.909 |
| 512 | down | 589.406 | 465.879 | 123.527 |

These differences are path-level diagnostics, not additive operator timings.
Changing graph consumers can change output placement and scheduling. In
particular, a standalone raw FP8 BMM writes its FP16 result to HBM, while the
scaled graph keeps BMM -> row scale -> column scale in LX. At M=512 Q/O,
`BMM + scales` is therefore `177.152 us`, faster than the raw-output BMM's
`180.465 us`.

The most useful M=512 reading is that Q/O would be `1.798x` over its FP16
baseline with the activation already packed, but falls to `1.470x` when the
real dynamic conversion is included. K/V reaches only `1.536x` even in its
prepacked BMM-plus-scale control because its narrow N leaves less useful
matmul work over which to amortize fixed packing and scale costs.

At M=64, the prepacked BMM-plus-scale controls sum to `1.531x` over FP16, and
raw BMMs sum to `1.619x`. This measured current-kernel ceiling is much more
restrictive than the optimistic external-byte bound because PT feed,
stationary-weight loading, FP16 output movement, and fixed scheduling work are
not doubled by FMA8.

## M=512 core-grid rescreen

The earlier complete pass forced SenDNN's `M8 x N4` grid for Q/O. Torch-Spyre's
existing FP8 cost model had already learned `M4 x N8` for this shape. A fresh
control-treatment-control bracket confirms that the latter remains better for
the complete scaled path:

| Run | Grid | Kernel us |
|---|---:|---:|
| control A | M8 x N4 | 220.551 |
| treatment | M4 x N8 | 216.597 |
| control B | M8 x N4 | 220.438 |

The treatment is `1.018x` faster than the mean control. It raises Q/O from
about `1.44x` to `1.47x` over FP16; it does not explain the remaining dynamic
conversion cost.

The weak K/V family still prefers `M8 x N4` (`143.652 us` in the 10-repetition
screen versus `148.092 us` for `M4 x N8`). `M2 x N16` is physically illegal
because it cuts a QFP8WT weight group. Gate/up retains `M8 x N4`; down retains
`M4 x N8`. Thus core-grid selection has now been rescreened rather than assumed.

Both corelets were already active for activation normalization, clipping,
QFP8MB, FP8 BMM, and both scale programs in the final M=64 Q/O artifact. The
remaining gap is not explained by losing a corelet. Corelets execute within
one outer core and share that core's LX; they cannot remove an ownership change
between different outer cores.

## Cross-projection activation reuse

Gate and up consume the same MLP activation. The pair benchmark deliberately
spells the two dynamic quantization chains independently and requires the
compiler reuse pass to canonicalize them. Its generated graph contains exactly
one `quantscalepertokenfp8`, one `qfp8mb`, two FP8 BMMs, and four scale passes.

| M | FP16 gate+up us | FP8 gate+up us | Speedup |
|---:|---:|---:|---:|
| 64 | 1384.490 | 1020.046 | 1.357x |
| 512 | 2014.808 | 1141.680 | 1.765x |

At M=512, sharing and joint scheduling reduce FP8 pair latency by `3.96%`
versus twice the standalone gate/up result. Combining this matched pair with
the K/V pair, two separately sourced Q/O projections, and down gives a grouped
linear-kernel sum of `3982.486 us` FP16 versus `2307.387 us` FP8: `1.726x`.
At M=64 the analogous grouped sum is `1.436x`; small-M fixed schedule cost
still dominates.

Q/K/V activation sharing remains unmeasured in this pass. It is the next
model-shaped reuse test, but it cannot by itself remove K/V's narrow-output raw
BMM and scale limit.

## Rejected first-scale epilogue

A safer variant of the DD2 BMM-plus-row-scale epilogue was prototyped. Unlike
the former invalid experiment, it retained the compact non-unit row scale at
its globally valid HBM address instead of rewriting it to a per-core LX
address. Correctness passed with relative L2 error around `4.72%`, and the
fused graph removed one SDSC.

It did not improve the limiting path:

| Shape | Control us | Fused us | Result |
|---|---:|---:|---|
| Q/O M=512 | 220.495 | 221.504 | neutral/slower |
| gate/up M=64 | 517.675 | 511.950 | about 1% faster |
| gate/up M=512 | 593.423 | 594.519 | neutral/slower |

The experimental code was therefore discarded. This is evidence that the
current LX-resident BMM -> scale -> scale handoff is already effective; merely
removing the first scale program is not the large remaining opportunity.

## Upper-limit read and next work

`2x` remains the FMA8 arithmetic opportunity, not the scaled operation's
expected result. The optimistic M=512 stream-only ceilings are approximately
`1.856x` for K/V, `1.817x` for Q/O, `1.805x` for gate/up, and `1.933x` for down.
The measurements imply:

1. **Down M=512 is effectively done as a standalone operator.** Its `1.937x`
   result is within modeling/timing noise of the stream-only ceiling.
2. **Gate/up M=512 is close.** Standalone reaches `1.679x`; real pair reuse
   reaches `1.765x`. More work is lower priority than Q/K/V.
3. **Q/O M=512 still has recoverable conversion cost.** Its already-packed
   scaled path reaches `1.798x`; the complete path is `1.470x`.
4. **K/V needs both reuse and a better narrow-N primitive.** Even its
   prepacked BMM-plus-scale control is only `1.536x`; eliminating conversion
   alone cannot make it a 1.8x operation.
5. **M=64 has not hit a hardware theorem, but it has hit the current kernel
   stack's limit.** Reaching complete `1.6x` requires improving the raw FP8 BMM
   feed/schedule and/or adding a DeepTools template that combines activation
   normalization, clipping, and QFP8MB packing. More LX pinning or a different
   corelet count is insufficient.

The next high-value implementation is therefore a combined activation
normalization-and-QFP8MB conversion primitive (or an upstream FP8-resident
activation contract), followed by Q/K/V sharing and one-layer/E2E validation.
That work may require a new DeepTools DDL/dataflow template; the current
`quantization_single_pad.ddl` admits one conversion operation and cannot fold
the preceding divide/clip chain by itself.

## Provenance

All measurements used:

```text
target:             DD2 / Spyre 1.0 (SENARCH=rcudd1a)
cores/corelets:     32 / 2
torch:              2.11.0+aiu.kineto.1.1.2
Torch-Spyre branch: ah/fp8-lx-relayout-poc
DeepTools source:   a74a581a85315ea8860250b831996a3a65745a67 + relayout patch
DXP LX fraction:    0.2
```

Primary device roots:

```text
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/granite_linear_m64_compact_scale_poc_20260802
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/granite_linear_m512_compact_scale_poc_20260802
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/granite_linear_m512_stage_isolation_20260802
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/granite_fp8_m512_grid_rescreen_20260802
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/granite_fp8_m512_qo_grid_bracket_20260802
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/granite_fp8_m512_qo_4x8_stage_20260802
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/granite_fp8_gate_up_shared_activation_20260802
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/granite_fp8_safe_epilogue_poc_20260802
```

No Sentient 1.5 source, target, or artifact was used. No public issue or pull
request was created.
