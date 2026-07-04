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

## Update - 2026-07-04 15:10

Additional diagnostics narrowed the all-gather/replicate blocker. This section
supersedes the earlier chunk-pattern description where it differs.

### Confirmed Frontend Contract

Single-output repro:

```text
before_sync_mul_one_M64K64N512_150459
```

The generated SDSC contract is the intended one:

```text
sdsc_0: 0_mul
  compute split: {mb:1, out:4}
  output: allocate-Tensor2_lx on cores 0..3, each core owns one out chunk

sdsc_1: 1_batchmatmul
  compute split: {mb:4, out:1, in:1}
  input Tensor1_lx carries classification:
    kind                  = matmul_operand_broadcast
    communication_class   = all_gather
    communication_pattern = all_gather_replicate
    transfer_count        = 16
```

So Torch is exposing the mismatch. The next issue is backend realization, not
frontend classification.

### Data-op Realization Observed

With:

```bash
DEEPTOOLS_ENABLE_UNSAFE_MATMUL_OPERAND_BROADCAST=1
DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1
DEEPTOOLS_STITCH_DATA_ONLY_BEFORE_DLDSC=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_UNICAST_PAIRS=1
```

DCC emits 16 data modules before the DL module:

```text
module order: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 0
```

Representative data modules store the four RHS chunks into distinct per-core
LX offsets:

```text
chunk 0 -> offset 0
chunk 1 -> offset 8192
chunk 2 -> offset 16384
chunk 3 -> offset 24576
```

The later matmul RHS loader reads `transfer_lds1_src:lxlu_dst:ptrow*` from a
folded view of the same logical RHS tensor. The generated read view is not a
simple flat `{0,8192,16384,24576}` pattern; it uses the matmul KERNEL/PT-feed
layout, for example bases like `0..3584` and `16384..19968` across ptrow/corelet
folds. That means the collective materialization must match the consumer KERNEL
transfer layout exactly; a naive contiguous concat is not enough evidence by
itself.

### Value Result

The single-output repro still fails value correctness:

```text
ALLCLOSE False
MAX_DIFF 15.9609375
MISMATCH 28544 / 32768
```

The failure is structured: every 128-column output chunk repeats chunk 0:

```text
row 0 cols 0:    correct chunk 0
row 0 cols 128:  repeats chunk 0
row 0 cols 256:  repeats chunk 0
row 0 cols 384:  repeats chunk 0
```

This points to either stale source reads or a mismatch between the data-op write
layout and the matmul KERNEL read layout.

### Experiments Since The First Writeup

1. Full destination allocation per consumer core:
   - Allocate full RHS-sized LX space with `checkAndAddDs` per consumer core.
   - Result: compiles/runs, but still repeats chunk 0.

2. Refresh scheduled compute `DataInfo.startAddr_`:
   - After changing the allocation base, refresh `ComputeNode.inputsLdsAndLoopOffsets_` for the RHS LDS.
   - Result: generated matmul reads use updated bases, but output still repeats chunk 0.

3. Unicast-pair mode:
   - Emit one STCDPOpLx per `(source core, destination core)` pair.
   - Result: still repeats chunk 0, so grouped/multicast STCDP is not the only issue.

4. Producer-only control:
   - Run only `rhs = v * aux` with producer split `{N:4}`.
   - Result: RHS returned to host is correct. Producer math and basic work split are valid.

5. Tuple-return control:
   - Return both `rhs` and `x @ rhs` from the graph.
   - Result: both are correct, but SDSCs become HBM-backed and no backend plan is emitted. This is a useful control but not proof that LX all-gather works.

6. Destination coordinate experiments:
   - Setting destination allocation coordinates to the consumer compute split fails DDC consistency.
   - Setting destination allocation coordinates to full replicated `{in:0,out:0}` on each consumer core also fails DDC consistency.
   - DDC reports the matmul RHS transfer has row/ptrow-specific coordinates, so normal KERNEL-to-PT transfers expect an empty allocation `coreIdToWkSlice_` or a very specific matching coordinate model.

7. Data-op `before_sync`:
   - Insert data-op schedule rows with both `before_sync=true` and `after_sync=true`.
   - Result: still repeats chunk 0. The failure is not fixed by a simple data-op local barrier.

8. Kernel-neighbor diagnostic path:
   - `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1` lowers the plan as `lowered_loop_scoped_kernel_neighbor`.
   - Without diagnostic override it hits the known double-buffering/input-neighbor coexistence guard.
   - With `DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1`, it triggered a PCIe bus-fence RAS error on CDX. Treat this path as unsafe for now.

### Current Best Hypothesis

The all-gather/replicate class is not blocked by Torch classification. It is
blocked by backend materialization semantics for a matmul KERNEL operand:

```text
producer LX shard layout      != consumer KERNEL/PT-feed read layout
simple STCDPOpLx concat       != sufficient to satisfy matmul read contract
```

The current in-place mutation of the consumer input LDS is also too hacky for a
production design. The cleaner direction is likely a real staged RHS LDS/buffer
whose lifetime, allocation, coordinates, and KERNEL transfer binding are planned
together. That staged buffer must be laid out exactly as the consumer matmul
loader reads it, not merely as logical contiguous tensor order.

### Device Note

The kernel-neighbor diagnostic bus-fenced CDX. The hot-reset utility takes PCI
BDF, not VFIO group. For CDX:

```text
/dev/vfio/80 -> /sys/kernel/iommu_groups/80/devices/0000:b0:00.0
```

The attempted reset was:

```bash
/opt/ibm/spyre/senlib/bin/aiu_dd2_hot_reset -t chip -d b0:00.0
```

It opened `/dev/vfio/80` but aborted with:

```text
RISCV config not found
```

The Linux reset variant requires root. Do not reuse CDX for hardware runs until
it is restarted or reset cleanly.

### Next Step

Do not move this class to Granite/attention yet. The next useful implementation
step is to introduce a separate staged matmul RHS LDS/buffer in Deeptools for
`matmul_operand_broadcast`, then bind the matmul RHS transfer to that staged LDS
instead of mutating the original producer-shard LDS in place. The staged LDS must
be allocated with the consumer KERNEL transfer layout and populated by the
all-gather in that same layout.
