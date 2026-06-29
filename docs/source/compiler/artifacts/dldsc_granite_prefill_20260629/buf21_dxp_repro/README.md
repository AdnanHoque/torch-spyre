# Buf21 Attention Value DXP Reproducer

This directory contains a reduced DXP reproducer for the Granite causal prefill
attention value operand relayout gap.

## Files

- `bundle.mlir`: one `sdscbundle.sdsc_execute` call.
- `buf21_batchmatmul.json`: extracted `16_batchmatmul` SDSC from the forced
  `buf21 -> buf22` diagnostic run.
- `dxp_stdout.log` / `dxp_stderr.log`: captured direct DXP output.
- `source_dir.txt`: source full Granite SDSC directory.

## Run Command

```bash
REPRO=/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/runs/dldsc_gap_report_20260629/buf21_dxp_repro
export PATH=/home/adnan-cdx/codex-worktrees/pr-lx-relayout-dldsc-post2829/tools/dxp-master-wrapper:$PATH
export DEEPTOOLS_PATH=/home/adnan-cdx/codex-worktrees/deeptools-master-relayout
export DXP_BACKEND_LX_FRAC_AVAIL=1
dxp_standalone --bundle -d "$REPRO"
```

## Expected Current Failure

The reduced case fails before AIU execution with:

```text
Coordinates of transfer transfer_lds0_src:lxlu_dst:sfp and allocateNode
allocate-Tensor1_lx are not consistent.
```

## Why This Matters

This is not the same class as the current Deeptools
`test_core_work_div_incompt` case.  That test exercises a resident relayout:

1. tensor is already LX-pinned;
2. tensor distribution differs from consumer compute split;
3. Deeptools inserts one `LxRelayout` SDSC using `STCDPOpLx`;
4. the relayout materializes a post-relayout resident tensor before compute.

In this reproducer, `allocate-Tensor1_lx` is the value operand of the attention
PV matmul.  Its producer residency is sharded by `out`:

```text
core 0  -> {out: 0,  in: 0, x: 0}
core 1  -> {out: 1,  in: 0, x: 0}
...
core 31 -> {out: 31, in: 0, x: 0}
```

The consumer matmul is split by `mb` and its transfer node needs temporal
`in/out/x` access.  Each consumer core therefore needs a loop-scoped stream or
broadcast/all-gather of value pieces from producer shards.  Materializing a
full post-relayout tensor on each consumer core is not the intended general
solution and may exceed LX.

The missing backend feature is a loop-aware matmul operand communication class,
not another direct resident relayout case.
