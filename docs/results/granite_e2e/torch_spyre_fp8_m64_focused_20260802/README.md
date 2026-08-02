# Torch-Spyre FP8 Granite M=64 focused pass

## Outcome

This pass isolates and reduces the non-matmul overhead in the complete dynamic
FP8 operation at `M=64` for every unique Granite 3 8B TP1 linear shape. The
final path retains per-row activation-scale derivation, activation division,
E4M3 saturation clipping, FP8 packing, the FP8 matmul, and both output-scale
applications. Static weight packing remains outside the timed region.

| Projection | K | N | FP16 us | Initial FP8 us | Focused FP8 us | Initial speedup | Focused speedup | Focused / initial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K/V shared pair | 4096 | 1024 | 124.904 | 113.829 | 93.182 | 1.097x | 1.340x | 1.222x |
| Q/O | 4096 | 4096 | 228.363 | 180.482 | 146.749 | 1.265x | 1.556x | 1.230x |
| gate/up | 4096 | 12800 | 707.643 | 529.459 | 517.675 | 1.337x | 1.367x | 1.023x |
| down | 12800 | 4096 | 700.042 | 483.892 | 449.877 | 1.447x | 1.556x | 1.076x |

K/V is compiled as a pair so its one dynamic activation conversion is reused
by both projections. The other rows are single-projection measurements.
Full-precision values, plus the comparable SenDNN M=64 results, are in
[`m64_results.csv`](m64_results.csv).

Using one K/V pair, two Q/O projections, two gate/up projections, and one down
projection gives this standalone linear-kernel sum:

| Path | Seven-projection sum us | FP8 / FP16 |
|---|---:|---:|
| FP16 | 2696.958 | 1.000x |
| Initial dynamic FP8 | 2017.602 | 1.337x |
| Focused dynamic FP8 | 1871.907 | 1.441x |

This is not a Granite end-to-end measurement. It is a serial sum of standalone
linear kernels and excludes attention, normalization, collectives, launch
interaction, and any additional full-layer sharing.

## What limited the initial path

Phase isolation shows that the FP8 multiply itself is healthy but not 2x:

| Projection | FP16 us | Raw prepacked FP8 us | Raw speedup | Complete initial FP8 us |
|---|---:|---:|---:|---:|
| K/V | 62.922 | 35.963 | 1.750x | 73.938 |
| Q/O | 228.363 | 141.360 | 1.615x | 180.482 |
| gate/up | 707.643 | 451.192 | 1.568x | 529.459 |
| down | 700.042 | 408.579 | 1.713x | 483.892 |

The raw gap from 2x is inside the matmul schedule: PT feed, stationary-weight
loading, FP16 accumulation/output movement, and fixed scheduling work do not
double with `FMA8`. The optimistic external-byte ceilings at M=64 are
1.97-1.99x, but those bounds assume each tensor is streamed once and do not
model internal feed or utilization.

Outside the multiply, activation normalization and packing added about
28-52 microseconds in the phase controls. Applying the two output scales added
about 40 microseconds for gate/up. These phase timings are not strictly
additive because changing the graph can change scheduling; the exact data is
in [`m64_phase_isolation.csv`](m64_phase_isolation.csv).

## Focused changes

### 1. Keep compact scales in their backend FP16 form

The specialized per-row scale reduction already produces the compact FP16
format consumed by the scale programs. The PoC stops widening those M-element
and N-element vectors to FP32 only to narrow them again before execution. This
removes redundant conversion programs while retaining the numerical scale.

### 2. Reuse one activation conversion

The graph reuse pass now recognizes both supported dynamic quantization forms:

```text
activation / row_scale -> optional clamp -> qfp8mb
activation * reciprocal(row_scale) -> clamp -> qfp8mb
```

Identical chains are shared only when their activation source and all scale
parameters match. The final K/V graph emits one per-row scale derivation and
one QFP8 pack feeding two FP8 matmuls; different scale parameters remain
distinct.

### 3. Co-plan activation normalization and packing ownership

The initial generic QFP8 pack planner considered a K split that is physically
illegal for the compound FP8 stick, spent its greedy core budget there, and
only reset K to one afterward. The PoC exposes legal M-row-pair ownership and
also assigns the preceding division and clamp to the same owners as QFP8MB.

The selected M=64 ownership is:

| Projection | divide / clamp / QFP8 pack | FP8 matmul | two scale applications |
|---|---|---|---|
| K/V | M32 | M8 x N4 | M8 x N4 |
| Q/O | M8 | M2 x N16 | M2 x N16 |
| gate/up | M8 | M8 x N4 | M8 x N4 |
| down | M8 | M4 x N8 | M4 x N8 |

With the same eight M owners, the division output is allocated in LX, the
clamp reads and writes LX, and QFP8MB reads that LX value. The packed FP8
activation then uses HBM where the matmul has a different 32-core ownership.
K/V instead benefits from M32 packing and the existing shared LX redistribution
because its two consumers amortize that movement.

Maximum pack fanout is not a general rule. Forcing all 32 pack owners adds an
ownership redistribution before Q/O and down and made those shapes slower.
The measured M8 choice is a balance between conversion parallelism and the
cost of changing ownership before the matmul.

## Corelets

Corelet underuse does not explain the remaining Q/O gap. A final post-DXP Q/O
artifact, generated with `DXP_DEBUG=1 --dump-bundle-module`, shows two
corelets for activation division, clipping, QFP8MB, FP8 matmul, and both scale
applications. The scale reduction uses one corelet. The matmul SMC contains
`PTOP_FMA8`; its 14,208 textual tokens prove emitted code presence, not a
dynamic instruction count.

Corelets can split compute inside one core, and both share that core's LX.
They cannot make an LX value owned by one outer core directly visible to a
different outer core. The performance-sensitive decision here was therefore
outer-core ownership and the resulting HBM/shuffle path, not simply whether
two corelets were enabled.

## SenDNN M=64 reference

The existing SenDNN sweep on the same DD2 target reports:

| Projection | FP16 us | SenDNN scaled FP8 us | Speedup |
|---|---:|---:|---:|
| K/V | 66.112 | 47.308 | 1.397x |
| Q/O | 224.296 | 140.601 | 1.595x |
| gate/up | 677.767 | 511.562 | 1.325x |
| down | 726.205 | 423.579 | 1.714x |

SenDNN uses fixed unit row and column scales in this sweep. Its timed graph
includes QFP8 packing, FP8 matmul, and both scale applications, but excludes
dynamic row-scale derivation and activation normalization. The Torch-Spyre
numbers above include those dynamic steps, so the cross-stack comparison is
directional rather than an exact parity benchmark.

## Remaining headroom

- Q/O reaches 1.556x, 96% of its measured 1.615x raw-matmul speedup. Closing
  more of this gap requires improving the FP8 BMM feed/schedule, not merely
  removing frontend conversion.
- Gate/up remains at 1.367x because its large `M x N` output is traversed by
  two separate `batchnormfwd` scale programs. Expressing both scales as a
  generic pointwise expression still emitted two programs. A real fused
  row-and-column scale operation, or a correct matmul epilogue, is the next
  high-value implementation.
- Down reaches 1.556x; its residual gap is split between packing the long-K
  activation and output scaling.
- K/V reaches 1.340x for the shared pair. Its narrow N makes fixed conversion
  and four output-scale passes large relative to the two small matmuls.

## Validation and provenance

- 121 focused Torch-Spyre unit tests passed on the pod.
- Every selected device case used 30 Kineto repetitions after five warmups.
- Every selected FP8 case passed finite-output, relative-L2 <= 10%, and
  peak-normalized maximum-error <= 10% gates; observed relative L2 is about
  4.72-4.80%.
- The exact clamped Q/O result was repeated with final-stage debug enabled:
  147.466 us over five repetitions versus 146.749 us over the primary 30,
  with correctness passing.

Pinned target and stack:

```text
target:             DD2 / Spyre 1.0 (SENARCH=rcudd1a)
cores/corelets:     32 / 2
torch:              2.11.0+aiu.kineto.1.1.2
DeepTools source:   a74a581a85315ea8860250b831996a3a65745a67 + relayout patch
DXP LX fraction:    0.2
```

Main artifact roots:

```text
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/granite_linear_m64_20260802
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/granite_linear_m64_phase_isolation_20260802
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/granite_linear_m64_compact_scale_poc_20260802
```

The post-DXP Q/O proof is under
`qo/compiler_pack_chain_m8_clamped_post_dxp/manual_dxp_debug/debug` in the
last root. No 1p5 target, source path, or artifact was used. No public issue or
pull request was created.

## Production next steps

1. Replace the private pack-grid oracles with a cost model that prices legal
   M-row-pair fanout, ownership changes, number of consumers, and LX lifetime.
2. Lower the complete scaled-matmul contract without the PoC-only compact-scale
   bypass, including non-unit weight scales, bias, result scaling, and the
   accuracy behavior of fast accumulation.
3. Add a real fused row-and-column output-scale operation or a proven matmul
   epilogue; validate that the large output never makes an HBM round trip
   between scale applications.
4. Share one activation conversion across Q/K/V and one across gate/up in a
   full Granite layer, then remeasure every M regime and run end to end.
