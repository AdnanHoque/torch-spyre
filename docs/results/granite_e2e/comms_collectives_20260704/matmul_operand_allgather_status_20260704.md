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

## 2026-07-04 update: matmul RHS all-gather is actually all-gather plus KERNEL restickify

Latest focused repro:

- Pod/root: `adnan-cdx-spyre-dev-pf:/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507`
- Repro harness: `runs/min_stable_matmul_operand_broadcast_20260704_100506/run_one.sh`
- Representative failing run: `compute_read_debug_mul_one_M64K64N512_151848`
- Existing DCC dump inspected: `unicast_pairs_mul_one_M64K64N512_144000/dxp_dcc_dump_unicast_144310`
- Shape: `mul_one`, `M=64`, `K=64`, `N=512`
- Producer split: `mul` with `{out:4}` owns RHS activation column chunks on cores 0..3
- Consumer split: `batchmatmul` with `{mb:4,out:1,in:1}` expects a replicated full RHS KERNEL operand per consumer core

Observed correctness:

```text
ALLCLOSE False
MAX_DIFF 15.9609375
MISMATCH 28544 / 32768
CHUNK_ROW 0 COLS 128..136 repeats COLS 0..8
CHUNK_ROW 0 COLS 256..264 repeats COLS 0..8
CHUNK_ROW 0 COLS 384..392 repeats COLS 0..8
```

What is proven now:

1. Torch correctly classifies the edge as `matmul_operand_broadcast` / `all_gather_replicate`.
2. Deeptools diagnostic lowering emits 16 logical producer-to-consumer transfers and schedules those rows before the DL matmul row.
3. The producer-only and tuple-return controls are value-correct, so the producer math and logical input values are not the issue.
4. Refreshing the scheduled `DataInfo.startAddr_` is not sufficient. The scheduler-side debug shows the matmul RHS `Tensor1` sees the rewritten LX base, but the result is still wrong.
5. The failure is therefore not a pure stale-base problem. The missing piece is layout/form conversion: the producer owns an activation-layout RHS shard, while the consumer reads a `KERNEL`-layout matmul operand.

DCC evidence:

- The inserted STCDP/ring rows move producer chunks between LX locations.
- The downstream matmul does not read a flat dense activation slab. Its RHS loader builds PT row-local KERNEL staging, with reads like:

```text
transfer_lds1_src:lxlu_dst:ptrow0
memref<64x64x8xf16>
vector_load ... [0, arg13 + arg12 * 8 + arg11 * 64 + arg8 * 64, arg10 + arg2 * 8]
```

- That means the resident destination must match the consumer KERNEL coordinate/layout contract, not merely contain the logical full `[K,N]` tensor bytes in activation order.

Interpretation:

This edge should be classified as:

```text
activation shard -> replicated matmul KERNEL operand
communication: all-gather / multicast
layout conversion: activation layout -> KERNEL layout
required carrier: staged KERNEL LDS + ReStickifyOpLx-style layout realization, or backend KERNEL staging support
```

The current raw `STCDPOpLx` prototype is useful as a diagnostic, but it is not the production carrier for this class because it moves bytes without satisfying the consumer KERNEL restickify/layout contract.

Recommended next implementation step:

1. Allocate a separate staged RHS `LabeledDsInfo` for the consumer KERNEL operand.
2. Materialize it using an LX-side restickify-aware path (`ReStickifyOpLx` or equivalent KERNEL staging), not only raw STCDP movement.
3. Rebind the consumer matmul input to that staged LDS before `L3DlOpsScheduler::fillLoopOffsetsAndAddresses()` runs.
4. Let `fillLoopOffsetsAndAddresses()` recompute `ComputeNode.inputsLdsAndLoopOffsets_[rhs].startAddr_`, `loopEleOffsets_`, `bufferAddrOffset_`, and corelet views.
5. Keep pure same-layout scatter as PR1. Treat `matmul_operand_broadcast` as a second communication class: all-gather plus layout conversion.

Useful code references from inspection:

- `dsc/dsc2.h`: `dsc2::DataInfo` contains the scheduled truth for `myLdsIdx_`, `startAddr_`, and offsets.
- `dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp`: `fillLoopOffsetsAndAddresses()` clones allocation addresses and derives offsets for scheduled compute nodes.
- `dcc/src/Conversion/DSC2ToDataflowIR/V3/SNComputeLowering.cpp`: lowering consumes `ComputeNode.inputsLdsAndLoopOffsets_`, not only high-level `ComputeOpInfo.inputLabeledDs`.
- `dcg/dcg_fe/pcfg_gen/restickifyOp.cpp` and `dcg/dcg_fe/transfer_compute/transfer_compute.cpp`: existing `ReStickifyOpLx` infrastructure can express LX-side restickification.

This is the current blocker for using the DLDSC path to remove the Granite/attention matmul RHS HBM round trip.

## 2026-07-04 update: backend carrier experiments after KERNEL-layout diagnosis

After the KERNEL-layout diagnosis above, we tested both backend carrier
directions on the same stable repro. These results are intentionally recorded
before moving back to Granite/attention, because they distinguish "movement was
not emitted" from "movement was emitted but the consumer read contract was
wrong."

### Custom KERNEL-neighbor carrier

Representative runs:

```text
runs/min_stable_matmul_operand_broadcast_20260704_100506/kernel_neighbor_forced_dst_base_155628
runs/min_stable_matmul_operand_broadcast_20260704_100506/kernel_neighbor_localcopy_forced_dst_161303
runs/min_stable_matmul_operand_broadcast_20260704_100506/kernel_neighbor_localcopy_forced_dst_161539
```

Forcing the destination base away from the source base changed the output from
mostly zero to nonzero wrong values:

```text
ALLCLOSE False
MAX_DIFF 24.0
MISMATCH 32621 / 32768
```

Interpretation: source/destination aliasing was a real bug in that diagnostic
path, but fixing aliasing was not enough to make the KERNEL operand correct.

We then tried to include same-core transfers as local `LX -> LX` movement rather
than skipping them. That hit two DCC/runtime constraints:

```text
RegisterTypeAssignment.cpp: expected transfers between LX and HBM
ConstructProgIRHelper.cpp: wrong locale for dst operand
```

Interpretation: pushing local `LX -> LX` through the custom L3 ring-transfer
path is the wrong shape. The existing STCDP path already knows how to do
same-core LX copies through LX-side units; the custom KERNEL-neighbor path would
need separate local-copy support before it is a credible production carrier.

### Existing STCDPOpLx materialization carrier

Representative runs:

```text
runs/min_stable_matmul_operand_broadcast_20260704_100506/unsafe_stcdp_materialized_broadcast_162106
runs/min_stable_matmul_operand_broadcast_20260704_100506/unsafe_stcdp_base_162242
runs/min_stable_matmul_operand_broadcast_20260704_100506/unsafe_stcdp_single_162242
runs/min_stable_matmul_operand_broadcast_20260704_100506/unsafe_stcdp_combined_162242
runs/min_stable_matmul_operand_broadcast_20260704_100506/unsafe_stcdp_combined_allow_162500
runs/min_stable_matmul_operand_broadcast_20260704_100506/unsafe_stcdp_single_combined_allow_162500
runs/min_stable_matmul_operand_broadcast_20260704_100506/unsafe_stcdp_single_combined_replace_allow_162500
runs/min_stable_matmul_operand_broadcast_20260704_100506/unsafe_stcdp_standalone_ifn_162739
runs/min_stable_matmul_operand_broadcast_20260704_100506/unsafe_stcdp_standalone_ifn_single_162739
runs/min_stable_matmul_operand_broadcast_20260704_100506/unsafe_stcdp_standalone_ifn_unicast_162739
```

Base STCDP materialization compiles and schedules data-op modules before the DL
matmul module:

```text
module order per core: 1 2 3 4 0
```

But it remains value-wrong:

```text
ALLCLOSE False
MAX_DIFF 15.9609375
MISMATCH 28544 / 32768
```

Single-dataop mode changes the mismatch count but not correctness:

```text
ALLCLOSE False
MAX_DIFF 15.9609375
MISMATCH 20992 / 32768
```

Combined IFN/data-op scheduling variants hit existing DCG guards:

```text
Do not support double buffering and input-neighbor fetch coexisting in the same DSC
data_dldscIdx_inf.first == scheduleStep.datadsc_idx
reqInpFetch ^ reqDLOp
```

Replacing the DL schedule step with a combined IFN/DL probe runs, but is even
more wrong:

```text
ALLCLOSE False
MAX_DIFF 32.0
MISMATCH 32759 / 32768
```

Standalone IFN routing is not usable for this case yet:

```text
inputNeighFetchOp.cpp: !is_any_of(lds.pinnedComponent(), HBM, NO_COMPONENT)
```

Interpretation: STCDPOpLx can realize the ring movement and can handle same-core
pieces better than the custom L3 carrier, but the current prototype writes a
logical activation-layout replica while the consumer matmul reads a KERNEL/PT
feed layout. The remaining issue is therefore not "emit more transfers"; it is
"materialize and bind the staged operand in the exact layout the matmul loader
will read."

### Current backend handoff point

The important code path is:

```text
dxp/SdscRelayoutInsertion.cpp
  attachMatmulOperandBroadcastInputFetch()
    allocates a replicated destination
    emits STCDPOpLx rows
    mutates the existing KERNEL LDS allocation/start address

dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp
  fillLoopOffsetsAndAddresses()
    derives scheduled DataInfo for compute nodes

dcc/src/Conversion/DSC2ToDataflowIR/V3/SNComputeLowering.cpp
  constructComputeInputOperand()
    lowers matmul input reads from inputsLdsAndLoopOffsets_ plus corelet views
```

This explains why simply refreshing `DataInfo.startAddr_` was insufficient:
the address changed, but the consumer read layout/corelet view did not become a
valid staged KERNEL layout.

### Updated conclusion

For this communication class, the clean design is:

```text
Torch/DLDSC:
  classify edge as matmul_operand_broadcast / all_gather_replicate
  provide producer and consumer logical coordinate maps

Deeptools:
  allocate a backend-owned staged RHS/KERNEL LDS
  populate it with all-gather movement in the consumer matmul's KERNEL layout
  bind the consumer matmul RHS transfer to that staged LDS before compute lowering
```

PR1 scatter remains the right first feature for one-to-one same-layout LX
relayout. The Granite/attention RHS edge is a second class: all-gather/multicast
plus KERNEL restickification/staging.

## 2026-07-04 follow-up: fresh STCDP run and sharper diagnosis

Fresh repro on `adnan-cdx-spyre-dev-pf` after rebuilding the current dirty
Deeptools tree:

```text
runs/min_stable_matmul_operand_broadcast_20260704_100506/fresh_unsafe_stcdp_current_164204
```

Result:

```text
ALLCLOSE False
MAX_DIFF 15.9609375
MISMATCH 28544 / 32768
```

With `CODEGEN_DUMP_IRS=1`:

```text
runs/min_stable_matmul_operand_broadcast_20260704_100506/fresh_unsafe_stcdp_dump_164517
```

The value signature is structured, not random. Output columns in later chunks
repeat data from earlier chunks, for example row 0 columns `128:136` repeat row
0 columns `0:8` instead of containing the `0.125...` range.

### What the fresh IR shows

The frontend metadata is correct for the broad class:

```text
producer: mul, work split out=4
consumer: batchmatmul, work split mb=4
classification: matmul_operand_broadcast
communication_pattern: all_gather_replicate
logical_transfer_count: 16
```

The lower IR shows the current backend prototype is too simple. It treats the
producer shard as a flat LX byte stream, but the producer pointwise output is
not a flat KERNEL operand.

Producer `0_mul` stores its LX output through a pointwise activation view:

```text
codegen_dumps/0_mul/stitcher_dataflow_ir_dldsc_.mlir
  #map3 = affine_map<(d0, d1, d2) -> (d2 * 4096 + d1 * 64 + d0)>
  dataflow.get_logical_memory_view ..., memref<64x64x2xf16>
  agen.composite_store ... dbgName="transfer_lds2_src:sfp_dst:lxsu"
```

The store is also corelet-aware:

```text
corelet 0 base: 0
corelet 1 base: 4096
```

The injected STCDPOpLx rows read/write through a flat `memref<64xf16>` stream:

```text
codegen_dumps/1_batchmatmul/stitcher_dataflow_ir_datadsc_2.mlir
  lxlu0 reads base 0 + arg2 * 64
  lxsu0 writes base 8192 + arg2 * 64
```

The consumer matmul then reads RHS through its KERNEL/PT-feed view:

```text
codegen_dumps/1_batchmatmul/stitcher_dataflow_ir_dldsc_.mlir
  #map9 = affine_map<(d0, d1, d2) -> (d2 * 4096 + d1 * 64 + d0)>
  dataflow.get_logical_memory_view ..., memref<64x64x8xf16>
  agen.vector_load ... dbgName="transfer_lds1_src:lxlu_dst:ptrow*"
```

So the bug is not just "wrong destination base." The data movement is copying
producer activation-layout bytes into a consumer KERNEL-layout staging region
without performing the logical relayout/restickify between the two views.

### Corelet diagnostic

Run:

```text
runs/min_stable_matmul_operand_broadcast_20260704_100506/unsafe_stcdp_sencorelets1_165051
```

With `SENCORELETS=1`, correctness still fails:

```text
ALLCLOSE False
MAX_DIFF 0.375
MISMATCH 29632 / 32768
```

The error shape changes, which confirms corelet layout participates in the
failure, but disabling corelet split does not make the flat-copy carrier value
correct. This points to the broader requirement: the carrier must be
layout-aware, not merely corelet-aware.

### Implication for the Granite/attention communication roadmap

This edge should be classified as:

```text
matmul RHS all-gather/multicast + activation-to-KERNEL restickify
```

It is not covered by PR1 scatter, and it is not covered by a naive STCDPOpLx
flat copy. A production implementation needs one of these shapes:

1. A backend-owned staged KERNEL LDS whose population path understands producer
   coordinates, producer layout/corelet split, and consumer KERNEL layout.
2. An explicit `ReStickifyOpLx`/DLDSC relayout op that materializes the operand
   in the consumer layout, followed by the existing matmul consumer.

The clean North Star still holds:

```text
Torch: classify/cost the edge and expose producer/consumer coordinate contracts.
Deeptools: synthesize the layout-aware movement/restickify and bind the staged
           KERNEL operand to the matmul.
```

The next implementation step is not adding more transfer pairs. It is teaching
the backend relayout insertion to materialize a logical producer-to-consumer
layout transform for KERNEL operands.

## 2026-07-04 Contract Update

After the fresh STCDPOpLx rerun, the Torch-side metadata was tightened so this
edge cannot be mistaken for a plain same-layout all-gather.

Focused Torch test:

```text
TORCH_DEVICE_BACKEND_AUTO=0 \
PYTHONPATH=$ROOT/repos/torch-spyre \
/home/adnan-cdx/dt-inductor-mixed/.venv/bin/python \
  -m pytest tests/inductor/test_lx_relayout_dldsc.py -q

16 passed in 0.09s
```

The generated consumer SDSC now contains:

```json
{
  "kind": "matmul_operand_broadcast",
  "communication_pattern": "all_gather_replicate",
  "materialization_pattern": "all_gather_replicate_with_layout_conversion",
  "requires_layout_conversion": true,
  "layout_transform": {
    "kind": "activation_lx_to_matmul_kernel_operand",
    "source": "producer_lx_residency",
    "target": "consumer_matmul_kernel_operand",
    "source_coordinates": "producer_tensor_distribution",
    "target_coordinates": "consumer_compute_distribution",
    "carrier_hint": "lx_all_gather_then_local_restickify"
  },
  "source_lx_tensor": {
    "allocation_name": "allocate-Tensor2_lx",
    "dsType_": "OUTPUT",
    "layoutDimOrder_": ["mb", "out"],
    "stickDimOrder_": ["out"]
  },
  "target_kernel_tensor": {
    "allocation_name": "allocate-Tensor1_lx",
    "dsType_": "KERNEL",
    "layoutDimOrder_": ["in", "out"],
    "stickDimOrder_": ["out"]
  },
  "staged_destination": {
    "component_": "KERNEL",
    "operand_read_index": 1,
    "scope": "matmul_transfer_loop"
  }
}
```

Deeptools was updated to require and preserve those fields in its backend plan
artifact whenever `requires_layout_conversion=true`. This is still not the
physical lowering; it is the contract checkpoint that makes the remaining
backend gap explicit.

Focused Deeptools test:

```text
$ROOT/build-deeptools/util/util_unit_test \
  --gtest_filter=LayoutAllgatherRestickify.matmulOperandBroadcastRecordsLayoutConversionContract

[  PASSED  ] 1 test.
```

DXP rebuild:

```text
cmake --build $ROOT/build-deeptools --target dxp_standalone -j8
```

Fresh metadata-preservation run:

```text
runs/min_stable_matmul_operand_broadcast_20260704_100506/rich_source_target_contract_174821
```

The run intentionally fails closed at DXP because physical lowering is still
blocked, but the emitted backend plan now records the correct class:

```json
{
  "kind": "matmul_operand_broadcast",
  "communication_pattern": "all_gather_replicate",
  "materialization_pattern": "all_gather_replicate_with_layout_conversion",
  "requires_layout_conversion": true,
  "source_lx_tensor": {
    "allocation_name": "allocate-Tensor2_lx",
    "dataFormat_": "SEN169_FP16",
    "dsType_": "OUTPUT",
    "layoutDimOrder_": ["mb", "out"],
    "stickDimOrder_": ["out"],
    "wordLength": 2,
    "startAddressCoreCorelet_": "...",
    "coordinateInfo_": "..."
  },
  "target_kernel_tensor": {
    "allocation_name": "allocate-Tensor1_lx",
    "dataFormat_": "SEN169_FP16",
    "dsType_": "KERNEL",
    "layoutDimOrder_": ["in", "out"],
    "stickDimOrder_": ["out"],
    "wordLength": 2,
    "startAddressCoreCorelet_": "...",
    "coordinateInfo_": "..."
  },
  "stages": [
    "source_operand_shards",
    "grouped_all_gather_replicate",
    "local_layout_conversion",
    "loop_scoped_input_fetch",
    "bind_matmul_kernel_operand"
  ],
  "physical_lowering_status": "blocked"
}
```

### Updated Gap

The consumer SDSC has the target matmul/KERNEL layout:

```text
Tensor1_lx: layoutDimOrder_ = [in, out], stickDimOrder_ = [out]
```

The producer SDSC has the source activation layout:

```text
Tensor2_lx: layoutDimOrder_ = [mb, out], stickDimOrder_ = [out]
```

The backend mutation point now sees the consumer input LDS, producer residency
coordinates, and explicit source and target layout objects, including
`startAddressCoreCorelet_`, `coordinateInfo_`, `wordLength`, and `dataFormat_`.
The remaining implementation is the two-stage physical carrier:

```text
STCDPOpLx same-layout gather/multicast into temporary LX
then ReStickifyOpLx/ReStickifyOpWithPTLx into the matmul KERNEL operand layout
```

Until that exists, enabling the unsafe flat STCDPOpLx path is expected to be
value-wrong on this class.

### Smallest Correct Backend Hook

The next Deeptools implementation should stay in:

```text
dxp/SdscRelayoutInsertion.cpp
```

Specifically, add a helper near the current
`attachMatmulOperandBroadcastInputFetch` experiment:

```text
attachMatmulOperandBroadcastGatherThenRestickify(...)
```

That helper should only accept:

```text
plan.requiresLayoutConversion == true
```

and emit normal mixed-SDSC data ops before the consumer batchmatmul:

```text
1. STCDPOpLx
   producer activation-layout LX shards
   -> backend-owned temporary LX, still in producer/source layout

2. ReStickifyOpLx
   temporary producer-layout LX
   -> final KERNEL-layout LX consumed by the matmul RHS
```

The existing `ReStickifyOpLx` backend is already wired through DCG/PCFG
generation. The construction reference is:

```text
dcg/unit_tests/datadsc_gen.cpp::populateDataDSCwithReStickfyLX
```

Do not continue the scheduler-side LX-neighbor experiment for this path. That
route generated useful diagnostics, but it still left the key correctness issue:
the matmul saw bytes gathered into the wrong layout.
