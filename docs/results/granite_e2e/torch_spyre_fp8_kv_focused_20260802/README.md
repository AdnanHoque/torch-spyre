# Torch-Spyre FP8 Granite K/V focused pass

## Outcome

For the Granite 3 8B TP1 K/V shape (`K=4096`, `N=1024`), the FP8 multiply was
not the main problem. The raw prequantized FP8 matmul is 1.71-1.79x faster than
FP16. The complete standalone operation lost that advantage to dynamic
activation conversion, a mismatched activation work division, repeated
conversion across K and V, and repeated redistribution of the same packed
activation.

This pass compiles K and V together and implements three focused changes:

1. recognize and reuse the specialized dynamic activation-quantization chain;
2. materialize one M32-to-M8xN4 LX view and reuse it for both matmuls; and
3. as an opt-in lifetime PoC, release the dead FP16 normalization buffer after
   `qfp8mb`, allowing the shared FP8 destination to reuse its LX address.

Matched 30-repetition Kineto results are:

| M | FP16 K+V us | Final FP8 K+V us | FP8 / FP16 | Final FP8 TFLOP/s |
|---:|---:|---:|---:|---:|
| 512 | 188.880 | 146.390 | 1.290x | 58.679 |
| 1024 | 392.472 | 263.343 | 1.490x | 65.238 |
| 2048 | 656.650 | 537.534 | 1.222x | 63.921 |

These are complete dynamic FP8 K+V timings, not raw matmul timings. They include
one dynamic per-row activation-scale derivation, activation normalization,
clipping and packing, two FP8 matmuls, and two FP16 scale applications per
output. Static weight packing is outside the timed region.

## Why K/V is far from 2x

`FMA8` creates a real 2x arithmetic opportunity, but K/V is a narrow-output
projection. Converting the `[M,4096]` activation costs the same as it does for a
Q/O projection, while each K/V matmul produces only `N=1024` columns. That
fixed `O(MK)` conversion and packing work is therefore large relative to the
useful `O(MKN)` matmul.

An optimistic external-byte model for a K+V pair assumes the FP16 baseline
streams the activation for both matmuls, while FP8 converts it once and keeps
the packed intermediate on chip:

```text
FP16 pair bytes = 4 * (M*K + K*N + M*N)
FP8 pair bytes  = 2*M*K + 2*K*N + 4*M*N + 4*M + 8*N
```

It excludes the compute and local traffic required to derive scales,
normalize, pack, shuffle, and apply output scales. It therefore predicts only
an optimistic stream bound:

| M | Arithmetic ideal | Optimistic stream bound | Observed | Fraction of stream bound |
|---:|---:|---:|---:|---:|
| 512 | 2.000x | 1.856x | 1.290x | 69.5% |
| 1024 | 2.000x | 1.799x | 1.490x | 82.8% |
| 2048 | 2.000x | 1.749x | 1.222x | 69.8% |

The best regime is M=1024. At M=512, fixed stage and launch work is a larger
fraction of the small matmuls. At M=2048, the packed activation is 8 MiB and
the simultaneous FP16-input, packed-source, and replicated-destination
lifetimes exceed the normal frontend LX budget unless the dead FP16 input is
released at the pack-to-shuffle boundary.

## Phase isolation

The original single-projection phase probe established that the FP8 arithmetic
path is healthy:

| M | FP16 us | Raw FP8 BMM us | Raw speedup | Complete dynamic FP8 us | Complete speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 97.552 | 57.064 | 1.710x | 96.551 | 1.010x |
| 1024 | 187.485 | 108.408 | 1.729x | 182.064 | 1.030x |
| 2048 | 331.758 | 185.622 | 1.787x | 405.687 | 0.818x |

Adding activation packing to the raw matmul costs approximately 28.6, 42.5,
and 176.5 microseconds at M=512, 1024, and 2048. The nonlinear M=2048 increase
is the large-activation storage and transport cliff, not a loss of `FMA8`
throughput.

## What changed

### 1. Reuse dynamic activation conversion

The existing graph pass recognized the generic reduction/divide/clip chain but
did not recognize the specialized DD2 `spyre.quant_scale_per_token_fp8`
producer. K and V therefore reached lowering as distinct sources even though
they quantized the same activation with identical parameters.

The matcher now treats that specialized reduction as a scale-derivation node.
The generated K+V graph contains exactly:

```text
1 quantscalepertokenfp8
1 qfp8mb
2 batchmatmulfp8mb
4 batchnormfwd
```

### 2. Share one redistributed LX view

`qfp8mb` distributes the activation over all 32 cores along M (`M32`). The
fastest K/V matmul division is `M8 x N4`: eight M owners and four N owners. A
cross-core shuffle must combine four M32 source slices and replicate the
result to the four N owners.

Before this pass, K and V each materialized an identical destination and each
emitted its own shuffle. Equivalent relayout plans now share one destination,
one allocation lifetime, and one emitted shuffle. The destination stays live
through both consumers, including when scheduling places them in different
generated kernels.

Same-pod old/new controls show that removing the duplicate shuffle improves the
pair by 6.2% at M=512 and 5.9% at M=1024:

| M | Two-shuffle FP8 us | One-shuffle FP8 us | Improvement |
|---:|---:|---:|---:|
| 512 | 155.377 | 146.282 | 1.062x |
| 1024 | 278.984 | 263.449 | 1.059x |

### 3. Shorten the dead FP16-input lifetime

At M=2048, the relevant per-core live storage is approximately:

```text
FP16 normalized activation input   512 KiB
packed FP8 source                  256 KiB
M8xN4 replicated destination      1024 KiB
```

The conservative allocator kept all three live at the shuffle boundary, about
1.75 MiB/core, so the relayout atomically fell back to HBM under the normal LX
partition. The emitted program is ordered `qfp8mb -> shuffle -> matmul`, so an
opt-in PoC ends the FP16 input lifetime after `qfp8mb`. The destination then
reuses that input's LX address.

At the normal `DXP_LX_FRAC_AVAIL=0.2`, this changes M=2048 from 599.450 to
537.534 microseconds, a 1.115x improvement. A separate capacity oracle using a
5% backend reservation reached 535.702 microseconds, independently confirming
that storage lifetime was the gating variable; the final result does not use
that unsafe reservation.

## Corelets

Corelet underuse is not the explanation for the remaining gap. Final
post-DXP artifacts for M=2048 show:

| Stage | Outer work division | Cores | Corelets |
|---|---|---:|---:|
| activation-scale derivation | M32 | 32 | 1 |
| `qfp8mb` packing | M32 | 32 | 2 |
| each FP8 matmul | M8 x N4 | 32 | 2 |
| each output-scale application | M8 x N4 | 32 | 2 |

For each FP8 matmul, an outer core owns an `M=256, N=256` output tile at
M=2048; DeepTools splits its N work `128/128` across the two corelets. The
important problem was the mismatch between the M32 packing ownership and the
M8xN4 matmul ownership. Corelets share their core's LX and cannot by themselves
move data between different outer cores, so using two corelets does not remove
that shuffle.

## Final emitted-program proof

The M=2048 generated order is:

```text
sdsc_7   qfp8mb
sdsc_8   shuffle
sdsc_9   K batchmatmulfp8mb
sdsc_16  V batchmatmulfp8mb
```

There is one shuffle SDSC. The generated allocation assigns:

```text
qfp8mb FP16 input LX address       262144
qfp8mb FP8 output LX address       0
shuffle destination LX address     262144
K and V activation input address   262144
```

Thus the destination reuses storage only after the pack has consumed its FP16
input, and both matmuls read the same materialized FP8 view. Both K and V pass
the numerical gate with relative L2 error about 4.72% and peak-normalized
maximum error below 4.6%.

## Validation and status

Validated:

- 18 focused relayout and FP8-quantization-reuse unit tests;
- a scratchpad compilation smoke through the pinned patched DeepTools stack;
- 30-repetition Kineto measurements for every reported final point;
- numerical correctness for both K and V at all three M values;
- one scale/pack chain, one shuffle, two BMMs, and four scale applications in
  generated artifacts; and
- final post-DXP corelet counts and splits at M=2048.

One broader scratchpad invocation initially failed because it used the pod's
stock `dxp_standalone`, whose `restickify.ddl` path points to a missing build
tree. The same first test passes through the pinned patched wrapper; that stock
stack failure is not attributed to this change.

The activation reuse and shared-destination changes are general mechanisms.
The early qfp8mb-input release remains an opt-in PoC until the backend ordering
contract is made explicit and stress-tested across more graphs. M=1 decode is
also outside this result because it uses the channel-packed path rather than
the even-M minibatch layout.

Pinned stack:

```text
target:             DD2 / Spyre 1.0 (SENARCH=rcudd1a)
cores/corelets:     32 / 2
torch:              2.11.0+aiu.kineto.1.1.2
DeepTools source:   a74a581a85315ea8860250b831996a3a65745a67 + relayout patch
DXP LX fraction:    0.2 for final results
```

Main artifact roots:

```text
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/kv_focused_phase_isolation_20260802
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/kv_pair_shared_lx_destination_bracket_20260802
/home/adnan/codex-isolated/fp8_lx_relayout_poc_20260801/runs/kv_pair_shared_lx_destination_sweep_20260802
```

The exact-final-code reruns are the `final_head_m512`, `final_head_m1024`, and
`final_head_m2048` subdirectories of the last root.

## Next production steps

1. Replace the private K/V grid and byte thresholds with a cost model that
   prices conversion reuse, destination fan-in/fan-out, LX peak lifetime, and
   the number of consumers.
2. Turn qfp8mb input release into a proven sequential-lifetime rule, or tile M
   so the live set fits without relying on whole-tensor address reuse.
3. Share the same activation conversion and destination across Q, K, and V,
   not only the K/V pair measured here.
4. Reduce or fuse the remaining normalization/packing and four output-scale
   programs; these now dominate the difference from the 1.75-1.86x stream
   opportunity.
5. Validate every Granite projection and then run one-layer and end-to-end
   Granite with matched numerical and timing gates.

No public issue or pull request was created.
