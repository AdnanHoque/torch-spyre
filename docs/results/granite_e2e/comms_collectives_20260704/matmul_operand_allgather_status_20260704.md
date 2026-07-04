# Matmul Operand All-Gather Status - 2026-07-04

## Goal

Implement the next non-scatter communication class needed by Granite and attention:
an LX-resident matmul RHS operand is produced sharded by output columns, while
the consumer matmul is split by M and each consumer core needs the full RHS.

This is an all-gather/replicate pattern:

```text
producer mul:       cores 0..3 own RHS out chunks 0..3
consumer matmul:    cores 0..3 each compute M chunks and need all RHS chunks
communication:      each producer chunk is copied to every consumer core
```

## Repro

Pod:

```text
adnan-cdx-spyre-dev-pf
```

Root:

```text
/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
```

Stable repro:

```text
runs/min_stable_matmul_operand_broadcast_20260704_100506/stable_matmul_operand_broadcast.py
```

Known run:

```text
runs/min_stable_matmul_operand_broadcast_20260704_100506/generic_stcdp_multi_dstoffset_restored_mul_one_M64K64N512_141326
```

Command shape:

```bash
CASE=generic_stcdp_multi_dstoffset_restored_mul_one_M64K64N512
export DEEPTOOLS_ENABLE_UNSAFE_MATMUL_OPERAND_BROADCAST=1
export DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1
export DEEPTOOLS_STITCH_DATA_ONLY_BEFORE_DLDSC=1
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_SINGLE_IFN
unset DEEPTOOLS_ENABLE_STANDALONE_IFN_DATAOP
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_COMBINED_IFN
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_REPLACE_DL_STEP
unset DEEPTOOLS_ENABLE_IFN_DL_COMBINED_SCHEDULE
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR
bash "$RUN/run_one.sh" "$CASE" mul_one 1 64 64 512
```

Split LX env still matters:

```text
Torch sees:   DXP_LX_FRAC_AVAIL=0
DXP sees:     DXP_BACKEND_LX_FRAC_AVAIL=1 through the wrapper
```

## Current Result

The repro compiles and runs, but is not value-correct:

```text
ALLCLOSE False
MAX_DIFF 15.9609375
MISMATCH 28544 / 32768
```

The error is structured. Chunk 0 is correct, while later RHS output-column
chunks are wrong:

```text
row 0 cols 0:    correct
row 0 cols 128:  repeats chunk 0 values
row 0 cols 256:  often one-hot / stale-looking values
row 0 cols 384:  repeats chunk 0 values
```

## What Is Working

The Torch-side DLDSC classification fires for the right edge:

```text
kind:                  matmul_operand_broadcast
communication_class:   all_gather
communication_pattern: all_gather_replicate
transfer_count:        16
```

The backend emits STCDPOpLx data ops and DCC sees ring traffic.

The experimental stitcher ordering patch works enough to put data-op modules
before the DL module:

```text
module order per core: 1 2 3 4 0
```

Generated DCC IR shows L3LU ring stores before the matmul transfer. The stores
land at distinct destination offsets:

```text
chunk 0 -> LX offset 0
chunk 1 -> LX offset 8192
chunk 2 -> LX offset 16384
chunk 3 -> LX offset 24576
```

## Failed Experiments

### Compact Logical Source Coordinates

Patch idea:

```text
source piece dimToStart = 0
destination piece dimToStart = global chunk start
```

Result:

```text
DtException in dcg/dcg_fe/pcfg_gen/stcdpOp.cpp:440
checkSubPieceCoverage failed
```

Interpretation:

STCDPOpLx expects source/output subpieces to cover the same logical tensor
range. Making the source logically compact breaks STCDP coverage.

### Negative Compensated Source Base

Patch idea:

```text
source piece dimToStart = global chunk start
source physical base = producer_lx_base - logical_offset
```

Result:

```text
Program verification failed
Register initialization out of boundary
```

Interpretation:

Negative LX base addresses are not legal, so we cannot express
"global logical coordinates, compact physical source" by subtracting the
logical offset from the address.

## Current Hypothesis

The remaining blocker is not classification and not basic ring emission.
It is one of:

1. STCDPOpLx lacks a clean way to represent a piece whose logical coordinate
   range is global but whose physical source allocation is compact local.
2. Data-op stores into generic `lx` are not being consumed by the later DL
   matmul read through the per-corelet `lx`/`lxlu` view as intended.
3. The required synchronization between L3LU/LXSU writes and the DL matmul
   transfer is still incomplete even though module order is correct.

## Most Useful DCC Dump

Replay dump:

```text
generic_stcdp_multi_dstoffset_mul_one_M64K64N512_135039/dxp_dcc_dump_dstoffset_regen_140917
```

Useful file:

```text
codegen_dumps/1_batchmatmul/complete_dataflow_ir.mlir
```

Evidence:

```text
early L3LU ring stores:
  c0-l3lu-ringDT-ring-lx-OL-0-1 -> LX offset 8192
  c0-l3lu-ringDT-ring-lx-OL-0-2 -> LX offset 16384
  c0-l3lu-ringDT-ring-lx-OL-0-3 -> LX offset 24576

later DL matmul read:
  transfer_lds1_src:lxlu_dst:ptrow*
```

## Next Patch Direction

Do not move to Granite/attention until this repro passes.

The next useful backend change should be one of:

1. Add an explicit STCDPOpLx notion of logical coordinate range versus physical
   local source offset.
2. Materialize into a backend-owned temporary LX allocation whose logical and
   physical base both start at zero, then bind the consumer matmul operand to
   that allocation.
3. Fix DCC unit binding/synchronization if data-op writes and DL reads are
   proven not to alias.

The smallest next diagnostic is to dump or inspect post-subpiece `DataOpDsc`
state for each STCDPOpLx and confirm the exact source/output subpieces and
their placement addresses before PCFG generation.
