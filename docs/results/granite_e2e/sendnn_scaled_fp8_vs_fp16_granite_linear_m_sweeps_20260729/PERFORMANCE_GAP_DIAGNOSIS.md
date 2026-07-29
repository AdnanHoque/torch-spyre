# SenDNN scaled-FP8 performance-gap diagnosis

## Bottom line

`FMA8` provides a 2x inner-arithmetic opportunity, not a 2x scaled-matmul
guarantee. The measured operation is:

```text
FP16 activation
  -> Qfp8 cast and packing
  -> optional LX relayout
  -> FP8 BatchMatMul
  -> activation-scale recovery
  -> weight-scale recovery
  -> FP16 output
```

The best current points prove the DD2 hardware and stack can get close to the
bandwidth-side expectation: gate/up reaches 92% and 89% of the optimistic
stream-only speedup at `M=1024` and `M=2048`. K/V, Q/O, and down do not. The
gap is therefore primarily shape-dependent fission, work division, and data
movement around the FP8 matmul, not a universal absence of FP8 arithmetic
capability.

## Theory versus measurement

For `M x K @ K x N`, the arithmetic ceiling is 2x. An optimistic streaming
model, with FP8 activation and weight, FP16 output, and FP32 row/channel
scales, gives:

```text
FP16 bytes       = 2 * (M*K + K*N + M*N)
scaled-FP8 bytes = M*K + K*N + 2*M*N + 4*(M + N)
```

This model excludes activation-scale derivation and the extra internal traffic
of starting from FP16 activation, so it is an optimistic bandwidth-side
reference rather than a universal hardware bound.

| Projection | M | Stream-only speedup | Measured speedup | Fraction of stream reference |
|---|---:|---:|---:|---:|
| K/V | 512 | 1.856x | 0.975x | 52.6% |
| K/V | 2048 | 1.749x | 0.648x | 37.1% |
| Q/O | 512 | 1.817x | 0.844x | 46.4% |
| Q/O | 2048 | 1.599x | 0.686x | 42.9% |
| gate/up | 512 | 1.805x | 0.599x | 33.2% |
| gate/up | 1024 | 1.682x | 1.555x | 92.4% |
| gate/up | 2048 | 1.536x | 1.367x | 89.0% |
| down | 512 | 1.933x | 1.198x | 62.0% |
| down | 2048 | 1.824x | 1.106x | 60.6% |

The fixed-scale standalone results do not reproduce one universal 1.5x
SenDNN number. They range from a regression to 1.56x depending on shape and
schedule.

## What corelets explain

Each DD2 core has two compute-side corelets. They provide two PT-side compute
opportunities, but share the core's LX and external data path. DeepTools'
corelet splitter is generic:

- it cannot split the K reduction when that would require cross-corelet
  partial-sum reduction;
- it splits a legal output dimension instead;
- the split must satisfy evenness, half-stick alignment, padding, mask,
  broadcast, and reuse constraints;
- FP8 affects legality indirectly through its different stick/layout contract.

For all four aligned Granite shape families, both the FP16 and FP8 matmuls
used 32 cores and two corelets. There is no missing-FP8-corelet explanation.

A controlled `SENCORELETS=1` diagnostic measured the benefit of restoring the
default second corelet. Each value below is `one-corelet latency /
two-corelet latency`; larger is better.

| Projection | M=512 FP16 | M=512 FP8 pipeline | M=2048 FP16 | M=2048 FP8 pipeline |
|---|---:|---:|---:|---:|
| K/V | 1.530x | 1.279x | 1.795x | 1.285x |
| Q/O | 1.632x | 1.291x | 1.767x | 1.352x |
| gate/up | 1.601x | 1.212x | 1.914x | 1.730x |
| down | 1.728x | 1.445x | 1.916x | 1.505x |

Removing a corelet is not an optimization: every path becomes slower.
However, FP16 usually gets much closer to the ideal 2x compute-side gain. The
complete FP8 kernel gets only 1.21-1.51x on most points because Qfp8, scale
recovery, relayout, FP16 output movement, and shared feed/LX work do not all
scale with duplicated PT arithmetic. Gate/up at `M=2048` reaches 1.73x when
the surrounding plan is healthy, reinforcing that this is a scheduling and
feed issue rather than a fundamental FP8 corelet restriction.

## Directly observed causes

1. **The compiler costs only the inner matmul.** The emitted ideal-cycle model
   halves FP8 BatchMatMul cycles, but assigns zero cycles to Qfp8, relayout,
   and both scale-recovery stages. The optimizer is therefore blind to much of
   the timed operation.

2. **Recovery fanout and ownership can be pathological.** Gate/up at `M=512`
   uses the same 32-core, two-corelet FP8 matmul grid as `M=1024`, but its
   recovery stages run on one core and an LX relayout is inserted. At
   `M=1024`, recovery expands to 32 cores and the relayout disappears. FP8
   latency falls from 1822 us to 1189 us even though M doubles.

3. **Some serial recovery remains at other shapes.** The final K/V recovery
   uses one core at both audited `M=512` and `M=2048` points. Down at `M=512`
   also has a one-core final recovery.

4. **Relayout is a contributor, but not the whole explanation.** Q/O loses its
   input relayout at `M=2048`, yet FP8 throughput remains near 41 TFLOP/s.
   Recovery, output movement, or matmul feed still limits that shape.

5. **Dynamic-working-set serialization hurts large down projection.** At
   `M=2048`, FP8 down is emitted as two serial `M=1024` working sets, repeating
   Qfp8, matmul, and recovery. FP16 remains one working set. Its speedup drops
   from about 1.20x at `M=1024` to 1.11x.

These are operation- and emitted-program-level findings. The retained
artifacts identify `batchmatmulfp8mb` and the exact halved ideal BMM cycles,
but do not contain textual `FMA8` disassembly.

## Ranked remaining hypotheses

1. The largest recoverable loss is the independently planned, serial fission
   pipeline: Qfp8, ownership changes, two full-output recovery passes, and
   synchronization.
2. Shared LX/feed/output bandwidth limits how much the complete pipeline gains
   from the second corelet, especially for K/V and Q/O.
3. FP8 block load, packing, forwarding, and output circulation may limit the
   inner matmul on some shape families. The current fused Kineto event cannot
   apportion this separately.
4. For end-to-end Granite, dynamic per-row scale derivation and duplicated
   Q/K/V and gate/up activation quantization can create additional loss. Those
   costs are deliberately absent from this fixed-scale sweep.

K-tail waste is not a plausible explanation for these shapes: the Granite K
values are aligned to the FP8 stick contract.

## Closure plan

1. Add per-stage timing or hardware-counter attribution for Qfp8, BMM,
   recovery 1, recovery 2, and relayout. A contrived unsupported raw-matmul
   API is not required; instrument the actual fissioned operation.
2. Make the planner cost the full scaled operation: cycles and bytes for
   Qfp8, each recovery, relayout, synchronization, FP16 output, core/corelet
   fanout, and dynamic-working-set count.
3. Use gate/up `M=512` as the first causal target. Independently force
   32-core recovery, preserve the Qfp8-to-BMM in-place handoff, and remove the
   relayout; measure each change against the current repeatable cliff.
4. Co-plan partition ownership across Qfp8, BMM, and recovery so the same
   MB/OUT pieces stay resident in LX. Do not independently optimize only the
   BMM grid.
5. Fuse the two scale-recovery passes into one output epilogue where numerical
   semantics allow it, avoiding repeated FP16 `M x N` circulation.
6. Remove the one-core final recovery for K/V and avoid the two-working-set
   FP8 split for down at `M=2048`.
7. Keep two corelets for these shapes, but make the score precision-aware:
   include FP8 stick legality and the shared feed/output ceiling rather than
   assuming that two corelets double full-operation throughput.
8. After standalone scaled matmul is healthy, reuse one quantized activation
   and its scale across Q/K/V and across gate/up in the Granite graph, then
   run the end-to-end experiment.

An initial engineering target is 80-90% of the optimistic stream-only
speedup, not 2x. The existing gate/up `M=1024/2048` points show that range is
attainable. It corresponds roughly to 1.4-1.55x for K/V, 1.3-1.45x for Q/O,
1.3-1.4x for large gate/up, and 1.45-1.65x for down, subject to the actual
stage attribution.

All measurements and audits used the pinned DD2 stack. No 1p5 result or
artifact is included.
