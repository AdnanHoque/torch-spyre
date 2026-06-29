# Dldsc LX Relayout Granite Gap Report

Date: 2026-06-29

## Goal

Reproduce the earlier approximately 1.2x Granite block prefill speedup using
the dldsc LX relayout implementation, and identify the remaining Deeptools /
runtime gaps if current dldsc support cannot reach that target.

## Current Profiled State

Shape: Granite block causal prefill, `B=1`, `S=512`, `E=4096`.

| variant | kernel_ms_per_iter | wall median ms | speedup vs current baseline |
| --- | ---: | ---: | ---: |
| current baseline | 12.4803195 | 19.4289684 | 1.000x |
| dldsc relayout, normal LX (`DXP_LX_FRAC_AVAIL=0.2`) | 11.35398295 | 18.2752609 | 1.099x |
| dldsc relayout, full-LX endpoint | 11.26920045 | 18.2604790 | 1.107x |

The 1.2x target against the current baseline would require
`10.40026625 ms/kernel-iter`.  The best current run is still
`0.8689342 ms` above that target.

Full-LX availability improves only about `0.085 ms` over the normal setting, so
the remaining gap is not primarily LX capacity.

## Current Planned Direct Relayout Edges

Current safe dldsc run plans 9 scatter resident remap edges:

| source | consumer | source bytes | reserved bytes/core group |
| --- | --- | ---: | ---: |
| buf5 | buf6 | 4194304 | 131072 |
| buf9 | buf14 | 4194304 | 131072 |
| buf14 | buf15 | 16777216 | 524288 |
| buf23 | buf24 | 4194304 | 131072 |
| buf24 | buf25 | 4194304 | 131072 |
| buf32 | buf33 | 4194304 | 131072 |
| buf33 | buf34,buf35 | 13107200 | 409600 |
| buf35 | buf36 | 13107200 | 409600 |
| buf36 | buf37 | 4194304 | 131072 |

Total planned source bytes: `68157440`.

These cover the scatter resident-remap class used by the MLP/SwiGLU and some
attention pointwise handoffs.

## Skipped Edge Classification

| skipped edge | classification | current reason |
| --- | --- | --- |
| `buf45 -> buf6` | QKV projection weight/restickified weight input | matmul non-primary input; weight-like, not activation target |
| `buf13 -> buf46` | key-side restickify / layout-changing attention handoff | unsupported dims `{0}->{3}` |
| `buf46 -> buf14` | key-side QK matmul operand | unsupported dim remap / grouped-query transpose path |
| `buf21 -> buf22` | value-side PV matmul operand | non-primary matmul input requiring streaming/broadcast |
| `buf47 -> buf24` | attention output projection weight | weight-like |
| `buf48 -> buf33` | MLP first projection weight | weight-like |
| `buf49 -> buf36` | MLP down projection weight | weight-like |

The non-weight remaining opportunities are attention key/value paths, not MLP
activation relayout.

## Forced `buf21 -> buf22` Evidence

Diagnostic run:

`/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_buf21_force_20260629_114808`

Reduced DXP reproducer:

`/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_gap_report_20260629/buf21_dxp_repro`

The reproducer contains only `16_batchmatmul` plus a one-line bundle MLIR.  Run:

```bash
REPRO=/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_gap_report_20260629/buf21_dxp_repro
export PATH=/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/tools/dxp-master-wrapper:$PATH
export DEEPTOOLS_PATH=/home/adnan-cdx/codex-worktrees/deeptools-master-relayout
export DXP_BACKEND_LX_FRAC_AVAIL=1
dxp_standalone --bundle -d "$REPRO"
```

The planner was temporarily relaxed to force only `buf21 -> buf22`.  DXP failed
before AIU execution with:

```text
Coordinates of transfer transfer_lds0_src:lxlu_dst:sfp and allocateNode
allocate-Tensor1_lx are not consistent.
```

The generated `allocate-Tensor1_lx` for `16_batchmatmul` had producer residency:

```text
core 0  -> {out: 0,  in: 0, x: 0}
core 1  -> {out: 1,  in: 0, x: 0}
...
core 31 -> {out: 31, in: 0, x: 0}
```

The consumer matmul transfer, however, needs temporal access over `in/out/x`.
That means each `mb`-split consumer core needs a loop-scoped stream of value
pieces from the producer `out` shards.  This is not the same contract as
materializing a resident post-relayout tensor before the consumer.

Additional Deeptools instrumentation on the reduced repro showed that the
merged relayout insertion does see the mismatch, but cannot place the
post-relayout form in LX:

```text
sdsc=16_batchmatmul lds=Tensor1 pinned=LX custom=32 sdscCoreMap=32
lx-probe-failed sdsc=16_batchmatmul lds=Tensor1 core=0
out_piece_size=4.1943e+06
insert-hbm sdsc=16_batchmatmul lds=Tensor1
cleared-original-coreIdToWkSlice-hbm sdsc=16_batchmatmul lds=Tensor1
```

So the backend behavior is:

1. detect the incompatible resident tensor distribution;
2. try to reserve one post-relayout LX piece per consumer core;
3. fail the LX probe immediately on core 0 because the piece is about 4 MiB;
4. fall back to HBM relayout;
5. then fail DDC coordinate validation while compiling the HBM relayout input.

The DDC failure after HBM fallback is a backend robustness gap, but it would not
recover the desired performance even if fixed.  The performance path needs an
on-chip loop-scoped transfer, not an HBM fallback for a full resident copy.

## Forced `buf46 -> buf14` Evidence

Diagnostic run:

`/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_buf46_force_20260629_115627`

Even with the non-primary matmul guard relaxed, the Torch planner rejected the
edge:

```text
LX_RELAYOUT_SKIP source=buf46 producer=buf46 consumer=buf14 consumer_op=None
reason=unsupported_dims producer_dims={'3': 32} consumer_dims={'2': 2}
```

This is the key-side QK operand.  It crosses view/restickify/transpose/GQA-style
dimensions, so it is not a same-dimension direct relayout.  Treating it as a
plain coordinate relayout would produce invalid or ambiguous residency.

## Deeptools Support Boundary

Current Deeptools dldsc relayout insertion handles the scatter resident-remap case:

1. An LX input allocation has `coreIdToWkSlice_`.
2. The allocation's tensor split differs from the consumer compute split.
3. Deeptools inserts an `LxRelayout` SuperDSC containing one `STCDPOpLx`
   datadsc.
4. The relayout materializes a post-relayout form in LX before the consumer
   compute row.

This is enough for scatter resident remaps.  It is not enough for attention
K/V matmul operands where the consumer wants loop-scoped streaming/broadcast
from producer shards into the matmul transfer schedule.

The current backend also explicitly validates allocation coordinates against
consumer transfer coordinates in `ddc/ddc_fold.cpp`.  Forced `buf21` fails that
validation, which is the correct fail-closed behavior for the current lowering.

## Required Next Communication Class

To close the remaining Granite block speedup gap on current main, dldsc relayout
needs a second class beyond scatter resident remap:

**Loop-scoped LX broadcast/all-gather/streaming for matmul operands.**

The high-level contract should be:

1. Torch/LX planner classifies a producer-to-consumer edge as a matmul operand
   all-gather/replicate edge, not scatter resident remap.
2. Torch emits producer tensor distribution and consumer transfer/compute
   requirement in dldsc coordinates.
3. Deeptools synthesizes scheduled ring movement into the matmul operand
   transfer loop, instead of materializing a full post-relayout resident tensor.
4. The implementation must be capacity-aware and loop-aware; full replication
   of the value tensor onto every `mb` consumer core is not a viable general
   substitute.

## Conclusion

The current dldsc LX relayout implementation is value-correct and speed-positive
for the scatter resident-remap class, reaching about `1.10x` on current-main
Granite block prefill.

Reaching `1.2x` against the current baseline is unlikely to be achieved by
another LX capacity sweep or by broadening the existing direct-relayout guard.
The remaining non-weight opportunity is attention K/V streaming/broadcast,
which needs a new Deeptools lowering class and a corresponding Torch planner
classification.
