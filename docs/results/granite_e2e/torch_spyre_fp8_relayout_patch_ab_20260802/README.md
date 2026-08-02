# DeepTools direct-LX-shuffle patch A/B

## Outcome

Removing the DeepTools direct-LX-shuffle promotion does not merely make the
optimized FP8 path slower. It makes the Granite Q/O experiment fail during DXP
scheduling. With the patch, the same byte-identical Torch-Spyre input graph
compiles, passes the numerical gate, and runs in `425.42 us`, versus
`652.29 us` for the matched FP16 control (`1.533x` speedup).

| Path | DeepTools arm | Compile | Kernel mean us | TFLOP/s | Numerical gate |
|---|---|---:|---:|---:|---:|
| FP16 control | patched build | pass | 652.287 | 52.676 | pass |
| optimized FP8 | base `a74a581a8` | **fail** | N/A | N/A | not run |
| optimized FP8 | direct-shuffle promotion | pass | 425.424 | 80.766 | pass |

The no-patch failure reproduced both with final-bundle debug dumping enabled
and with normal DXP settings:

```text
DtException: Scheduler failed to find a suitable op mapping for sdsc: 8_shuffle
```

There is therefore no valid no-patch device latency to compare. Reporting a
number for that arm would measure a different graph or fallback, not this FP8
LX path.

## Controlled experiment

The selected linear layer is the Granite Q/O projection:

```text
[M=1024, K=4096] @ [K=4096, N=4096]
```

The optimized FP8 timing includes dynamic per-row activation-scale derivation,
activation normalization/clipping/`qfp8mb` packing, the FP8 matmul, and both
output-scale applications. Static weight packing is performed once in a
separate graph and excluded. The FP8 matmul and both scale applications use the
experimental `M:8 x N:4` work division.

Both arms used the same pod, Torch-Spyre source, inputs, environment, and fresh
Inductor caches, and ran serially. The only executable difference was
`SdscRelayoutInsertion.cpp` inside `libdxp.so`; every other DeepTools object and
library came from the same build. The Torch-Spyre `bundle.mlir` and the relevant
input SDSCs were byte-for-byte identical between arms, including `sdsc_8.json`:

```text
bundle.mlir sha256  16dceb3b84e1341c5bd69593e1d20b05dfb1b8533d34115fb71358f57f2d20f3
sdsc_8.json sha256 1a3e2337ed418ee8881f1cd92b1c234a265f2f0fb84cbefec7882b78d5556c25
```

The clean timing used five warmups and 30 measured launches. Latency is the
mean aggregate Kineto `cat == "kernel"` duration per launch. The profiler-loop
wall values include host dispatch, synchronization, and profiler-step overhead
and are not used for the device speedup.

No 1p5 target, binary, or artifact was used. The target was DD2 / Spyre 1.0
(`SENARCH=rcudd1a`, 32 cores, two corelets, `DXP_LX_FRAC_AVAIL=0.2`).

## What the patch changes

Torch-Spyre deliberately emits an LX ownership-change node between 32-core
M-sharded `qfp8mb` packing and the `M:8 x N:4` FP8 matmul. The node is a data
movement operation: it redistributes the already-packed activation inside LX;
it has no arithmetic to execute.

Without the patch, DeepTools creates an `STCDPOpLx` transfer as a separate
relayout object, rewrites the original shuffle to an identity compute op, and
retains that original compute SDSC. For this graph the scheduler cannot find a
legal mapping for the remaining `8_shuffle` SDSC and aborts.

With the patch, DeepTools recognizes this exact direct LX shuffle as copy-only,
moves the `STCDPOpLx` data operation into the shuffle's original execute slot,
and removes the now-empty compute program. Final debug output for `sdsc_8`
contains `STCDPOpLx`; the unpatched output contains `identity` and never reaches
a runnable bundle.

This establishes that the patch is a functional requirement for the current
Torch-Spyre FP8 LX PoC, not just a performance cleanup.

## Results and provenance

Machine-readable values are in
[`matched_results.csv`](matched_results.csv). The full device artifacts remain
on the shared study volume:

```text
/home/adnan/codex-isolated/fp8_relayout_patch_ab_20260802
```

Important roots:

```text
runs/round1_without_patch              debug failure and pre-DXP graph
runs/confirmation_without_patch_nodebug no-debug failure confirmation
runs/round1_with_patch                 debug pass and final STCDPOpLx evidence
runs/clean_with_patch_matched           clean FP16/FP8 30-repetition timing
```

Pinned source/binary evidence:

```text
Torch branch before this report: ah/fp8-lx-relayout-poc @ 87dda1a
benchmark script sha256:         2df5b869cd475db089ea6718adc7d3669c67ca527e0c31615f8bd0e70f52f7c8
DeepTools base:                  a74a581a85315ea8860250b831996a3a65745a67
DeepTools patch:                 3b5d123a11e43c69177fa9a86172bf4b0fcf54a1
patched libdxp.so sha256:        6f5753e74c72dadc27073aea8f5b75076019363de325c9aeef20e7c9e22a6c3b
unpatched libdxp.so sha256:      f3d306b869bb06303796d1861f803d10ed39037d0c93937c439042d0245a1d14
```

The measured patched binary was built from the pre-commit working version of
the patch. Its only difference from `3b5d123a1` is clang-format line wrapping;
the promotion logic is identical.

