# Torch-Spyre DD2 FP8 scaled-matmul focused pass

> **Superseded epilogue result:** later testing with non-unit per-row scales
> found the late first-scale epilogue probe numerically incorrect. It rewrote
> the scale address after work division without redistributing each row-scale
> block to the four N owners that consume it. The historical timings below are
> retained as diagnostic data only; commit `1a110ac` is removed by the follow-up
> Q/O LX/qscale PoC.

## Outcome

This branch moves Torch-Spyre from a raw FP8 reduction experiment to a
functional DD2 `aten._scaled_mm` path and identifies two distinct performance
problems:

1. the original FP16-oriented planner selected poor FP8 core grids; and
2. after fixing that grid, Torch-Spyre still wrote the packed activation and
   raw matmul output to HBM where SenDNN retains them in LX scratchpad.

The first problem is addressed by the FP8-aware planner. For the second, an
environment-gated proof of concept emits FP8 matmul followed by the first scale
multiply in one SDSC. Across 12 Granite prefill cases it improves 10, is flat
within 2.5% on one, and regresses one by 2.8%. The gain ranges from 1.15x to
1.81x where profitable. This proves the backend capability, but the late
code-generation implementation is not production shaped.

All work and measurements in this report target DD2 / Spyre 1.0. No 1p5 stack,
artifact, or target was used.

## Implemented contract

The public `aten._scaled_mm` decomposition now supports:

- 2D E4M3 activation and weight inputs;
- scalar FP32 scales or `[M,1]` activation plus `[1,N]` weight scales;
- optional FP16 or FP32 `[N]` bias;
- E4M3, FP16, or FP32 output;
- scalar FP32 `scale_result` for E4M3 output; and
- `use_fast_accum=True`.

DD2 exposes one native FP8 accumulation schedule, so
`use_fast_accum=False` is rejected rather than silently treated as true. The
raw FP8 reduction is a private primitive; scale, bias, and output conversion
semantics remain explicit and testable.

For row/column scales, the raw FP16 result passes through two specialized
scale-and-offset operations. The emitted backend name is `batchnormfwd`; it is
an affine multiply/add on the chip's elementwise engine, not neural-network
batch normalization in the usual model sense. Bias is folded into the offset
of the second pass.

## What SenDNN and Torch-Spyre execute

Counting compute operations after the activation is packed, the audited
SenDNN path has FP8 matmul, row-scale multiply, and column-scale multiply as
three separate compute stages. Including dynamic activation packing makes
four. DeepTools also emits LX relayout and compact-scale transfer/reshape
stages, so the complete audited program has 10 execute stages plus two
model-load scale preloads.

Separate stages do not imply an HBM roundtrip. The emitted allocation evidence
is:

| Large-tensor edge | SenDNN M512 | Torch-Spyre one-layer baseline |
|---|---|---|
| packed activation to FP8 matmul | LX-only | HBM pool |
| FP8 matmul to first scale | LX-only | HBM pool |
| first scale to second scale | direct LX handoff | direct LX handoff |

Both SenDNN scale stages and all 14 scale stages in the Torch-Spyre one-layer
graph execute as SFP `batchnormfwd`. SenDNN uses LX relayouts when producer and
consumer core ownership differs. Compact scale vectors have their own transfer
stages; that is not an M-by-N intermediate spill. Only the final second-scale
result is written to HBM in the audited SenDNN path.

The current Torch-Spyre one-layer baseline contains seven FP8 linear
subsequences. In every one, allocation records show:

```text
qfp8mb output       -> hbm_pool
FP8 matmul output   -> hbm_pool
first scale output  -> LX
second scale output -> hbm_pool
```

Therefore the concrete transport work is to remove the first two HBM
boundaries. The second scale edge is already LX-resident.

## Work division and corelets

The earlier SenDNN Q/O regression had a separate, source-backed cause. The
exact emitted work divisions for the two audited large-M cases are:

| M | Variant | QFP8 packing | FP8 matmul | Row-scale multiply | Column-scale multiply |
|---:|---|---|---|---|---|
| 512 | stock | `M:32 x N:1` | `M:8 x N:4` | `M:8 x N:4` | **`M:1 x N:1`** |
| 512 | diagnostic improvement | `M:32 x N:1` | `M:8 x N:4` | `M:8 x N:4` | **`M:8 x N:4`** |
| 2048 | stock | `M:32 x N:1` | `M:32 x N:1` | `M:32 x N:1` | **`M:1 x N:1`** |
| 2048 | diagnostic improvement | `M:32 x N:1` | `M:8 x N:4` | `M:8 x N:4` | **`M:8 x N:4`** |

Every 32-way stage uses 32 cores and two corelets per core. The one-way stage
uses one core and two corelets. Thus QFP8 packing is allowed to have different
ownership, while the improved matmul and both scale applications share one
output grid in these artifacts.

DeepTools saw the tiny static weight-scale vector as favorable to preload and
added a hard work-division constraint equivalent to:

```text
product(M, X, Y, N splits) <= 1
```

That reuse heuristic described the small scale input but accidentally
serialized the large M-by-N output operation. The diagnostic
`weipreload=0` switch removed the constraint and allowed the scale stage to
retain the matmul's `M:8 x N:4` 32-core grid. At Q/O M512, scaled-FP8 latency
fell from 414.814 us to 220.793 us (1.879x), changing FP8/FP16 from 0.843x to
1.584x. The global switch is evidence, not a production fix.

The preserved M=2048 treatment artifact also shows why `M:8 x N:4` is not a
universal constant: stock matmul at that M used `M:32 x N:1`, and the global
switch changed it too. Thus the switch has broader planner effects outside the
isolated M=512 diagnosis. A production rule should co-plan each scale stage
with its actual producer, not hard-code 8x4 or disable weight preload globally.

Corelets were therefore relevant only in the general sense that they are
compute lanes inside each core. Both the slow and fast programs already used
two corelets. Two lanes in one core cannot replace distributing the output
over 32 cores; the decisive error was the outer core grid, not losing a
corelet.

Torch-Spyre now handles the narrow scaled-matmul pattern directly: each scale
operation inherits the preceding same-shaped matmul output grid after mapping
its renamed iteration variables and rechecking layout legality. That fixes core
fanout. It does not, by itself, fix HBM transport. The epilogue experiment below
starts after both scale stages already have the 32-core grid and isolates the
remaining composite-scheduling/transport opportunity.

## First-scale epilogue proof of concept

DeepTools' DD2 BMM template can carry one `batchnormfwd` epilogue. The private
Torch-Spyre probe recognizes each eligible FP8-BMM/first-scale pair and emits
one composite SDSC:

```text
PT:  batchmatmulfp8mb -> Tensor4
SFP: batchnormfwd     -> Tensor4
```

The control graph has 10 SDSCs and the treatment has 9. The BMM outer work
division is identical in every control/treatment pair: 32 cores, generally
`M:8 x N:4`, with `M:4 x N:8` for Q/O at M=512. The A/B therefore does not
confound transport with a different matmul planner choice.

Kernel time is the mean of 10 Kineto `cat == "kernel"` iterations after three
warmups. Weight packing and dynamic scale derivation are outside the timed
graph; supplied row and column scale tensors are genuinely applied. All 24
runs passed finite/allclose correctness.

| Projection | M | Baseline us | Epilogue us | Speedup | Latency change |
|---|---:|---:|---:|---:|---:|
| K/V, K4096 N1024 | 512 | 119.420 | 103.655 | 1.152x | -13.20% |
| K/V | 1024 | 223.023 | 184.268 | 1.210x | -17.38% |
| K/V | 2048 | 443.015 | 385.125 | 1.150x | -13.07% |
| Q/O, K4096 N4096 | 512 | 286.258 | 227.207 | 1.260x | -20.63% |
| Q/O | 1024 | 557.875 | 433.330 | 1.287x | -22.32% |
| Q/O | 2048 | 1203.989 | 884.588 | 1.361x | -26.53% |
| gate/up, K4096 N12800 | 512 | 1166.149 | 646.035 | 1.805x | -44.60% |
| gate/up | 1024 | 1903.165 | 1304.185 | 1.459x | -31.47% |
| gate/up | 2048 | 4643.081 | 3597.818 | 1.291x | -22.51% |
| down, K12800 N4096 | 512 | 817.712 | 649.840 | 1.258x | -20.53% |
| down | 1024 | 1264.465 | 1300.244 | 0.973x | +2.83% |
| down | 2048 | 4933.247 | 4809.868 | 1.026x | -2.50% |

Full-precision rows are in
[`first_scale_epilogue_sweep.csv`](first_scale_epilogue_sweep.csv). A repeat
confirmed both the large gate/up M512 gain (1.818x) and down M1024 regression
(0.972x), so production fusion needs a profitability rule.

The treatment also moves compact row-scale and zero preparation ahead of the
composite operation. Its timing therefore measures the entire co-lowering and
schedule change, not an isolated HBM-bandwidth microbenchmark.

Eleven treatment graphs leave the composite output in LX for the second scale.
The late-allocation gate/up M2048 probe leaves it in HBM. This is another reason
to move production fusion before liveness and allocation instead of repairing
an already-allocated OpSpec list.

## Activation scale and packing

For FP16 model activations, `abs` and row-wise `max` only select an existing
FP16 value. Reducing in FP16 and converting the compact `[M,1]` maxima before
FP32 division therefore produces the same scale as converting the full
`[M,K]` activation first, without an M-by-K FP16-to-FP32 conversion.

The standalone component gate was bit-exact to the full-FP32 CPU scale and
measured:

| Shape | Compact scale derivation |
|---|---:|
| M512 K4096 | 120.618 us |
| M512 K12800 | 389.579 us |

The combined dynamic quantize-plus-matmul timing remains diagnostic: its Q/O
M512 relative-L2 error was 4.72% but it failed the strict elementwise gate. The
large-K CPU oracle also overflowed before scale application. Those timings are
not accepted functional results. The packed activation still lands in HBM, so
the compact reduction optimization reduces conversion work but does not solve
the quantize-to-matmul transport boundary.

The FMS-MO integration bridge deliberately retains the DD2-native FP16 qparam
approximation used by the completed one-layer smoke and labels it as such. It
does not claim exact TorchAO dynamic quantization. Static checkpoint weights
are now packed once into QFP8WT before graph capture rather than repacked on
every linear invocation.

## Granite integration status

The non-epilogue one-layer M512 prefill gate completed on the pinned integration
stack. It executed all seven FP8 linears, each with QFP8MB, FP8 BMM, and two
real scale applications. The main compiled-graph Kineto event was 24.586 ms;
the approximately 1.2-second host token time includes CPU fallback and other
host overhead and is not comparable with the archived 8.056 ms SenDNN
production-stack one-layer reference.

An exact-commit, fresh-cache treatment-control-treatment bracket then exercised
the epilogue prototype across all seven FP8 linears in the same one-layer
program:

| Run | Main compiled-graph Kineto time | SDSCs | Result |
|---|---:|---:|---|
| treatment 1 | 23.613 ms | 163 | pass |
| control | 24.208 ms | 170 | pass |
| treatment 2 | 24.172 ms | 163 | pass |

All three emitted the same model output. Each treatment combined all seven
eligible matmul/first-scale pairs and removed exactly seven standalone SDSCs.
Control divided by the mean treatment time is 1.013x. The two treatment runs
span almost the entire apparent improvement, so this is a confirmed structural
win, not yet a stable model-level performance claim.

The 40-layer prefill gate is tracked separately below. Neither a one-layer
token match nor a standalone kernel A/B is an end-to-end Granite speedup claim.

## Remaining production work

1. Move BMM/first-scale fusion before liveness, HBM-pool, and LX planning; add
   a shape-aware profitability decision.
2. Co-schedule QFP8 packing and FP8 BMM so the packed activation remains in LX,
   including any ownership-changing LX relayout.
3. Pass a strict numerical gate for dynamic activation scale derivation and
   packing before enabling the exact compact-FP32 path in FMS-MO integration.
4. Recalibrate/validate the FP8 planner beyond the original Q/O oracle and on
   shapes outside Granite.
5. Resolve the DD2 QFP8CH/double-padding failure at M=1, then validate decode.
6. Complete the 40-layer prefill gate and only then run a controlled FP16/FP8
   Granite end-to-end comparison.

## Evidence and provenance

Implementation commits through the epilogue probe:

```text
291de76  complete DD2 scaled-matmul semantics
29b7ce6  FP8-aware work division and scale-grid inheritance
9ff6191  shared activation-quantization CSE
343fdef  real Granite integration layout support
45242c6  activation-scale and integration benchmarks
1a110ac  private first-scale epilogue probe
```

Validated stack:

```text
torch:       2.11.0+aiu.kineto.1.1.2
DeepTools:   +1401 (ee2f97a)
Flex:        +388 (81385a4)
target:      DD2 / Spyre 1.0
```

Preserved roots:

```text
epilogue sweep:
  /tmp/fp8_bmm_bn_epilogue_granite_matrix_20260731_v2
epilogue repeat:
  /tmp/fp8_bmm_bn_epilogue_granite_matrix_repeat_20260731
one-layer baseline:
  /tmp/torch-spyre-granite-fp8-one-layer-fp16scale-5c35bb3
one-layer epilogue treatment 1:
  /tmp/granite_fp8_epilogue_1a110ac_run1
one-layer epilogue control:
  /tmp/granite_fp8_epilogue_1a110ac_control1
one-layer epilogue treatment 2:
  /tmp/granite_fp8_epilogue_1a110ac_treatment2
activation study:
  /home/adnan/codex-isolated/fp8_activation_overhead_20260731_01
SenDNN direct pair:
  /home/adnan-cdx/codex-isolated/fp8_sendnn_study_20260728_210752/benchmarks/
  direct_pair_m512_k4096_n1024_20260729_012100
```

No public issue or pull request was created.
