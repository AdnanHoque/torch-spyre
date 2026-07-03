# Flash Attention `105_batchmatmul` Matmul-Operand Broadcast Experiments

Date: 2026-07-03  
Pod: `adnan-cdx-spyre-dev-pf`  
Torch workspace: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/torch-spyre`  
Deeptools workspace: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/deeptools`  
Test: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/test-spyre-scripts/test_flash.py`

## What This Is Testing

This isolates the attention edge where `sdsc_105.json` / `105_batchmatmul` consumes Tensor1 as a KERNEL operand. The producer-side LX tensor distribution is split over `(out, in)` while the consumer matmul compute is split over `(x, mb)`. That is not PR1 scatter. It is an all-gather/broadcast style contract: every consumer core needs all producer chunks for this operand.

The Torch-side metadata emits:

- `kind = matmul_operand_broadcast`
- `communication_pattern = all_gather_replicate`
- `realization_strategy = loop_scoped_input_fetch`
- `logical_transfer_count = 1024`
- `source_core_count = 32`
- `consumer_core_count = 32`

## Runs

All runs used the split frontend/backend LX knob setup:

- Torch sees `DXP_LX_FRAC_AVAIL=0`
- DXP subprocess sees `DXP_BACKEND_LX_FRAC_AVAIL=1` through the local wrapper

| Variant | CDX run directory | DXP/runtime result | Correctness result | Interpretation |
| --- | --- | --- | --- | --- |
| `phasefix` | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_matmul_operand_only105_fission4_phasefix_20260703_075649` | DXP and runtime complete | `15,589,744 / 16,777,216` mismatched (`92.9%`), max diff `inf` | Phase/cardinality fixes are enough to run, but physical movement is value-wrong. |
| `dstoffset` | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_matmul_operand_only105_fission4_dstoffset_20260703_080553` | DXP and runtime complete | `14,908,697 / 16,777,216` mismatched (`88.9%`), max diff `inf` | Adding destination shard offsets helps slightly, so overwrite/address placement is part of the bug, but not sufficient. |
| `srcdstoffset` | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_matmul_operand_only105_fission4_srcdstoffset_20260703_081110` | DXP and runtime complete | `15,619,793 / 16,777,216` mismatched (`93.1%`), max diff `inf` | Source shards are likely already addressed as per-core local shards; applying source offsets makes the result worse. |
| 4-core one-hot, original tree order | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_execdebug_20260703_090459` | DXP and runtime complete | `262100 / 262144` mismatched, output mostly zero | Inserted relayout SDSCs compile to nonzero program frames, but DXP compiles/runs them after the consumer matmul because `SdscTree::getAllSdscNodes()` returns storage order, not tree order. |
| 4-core one-hot, tree-order traversal fix | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_treeorder_20260703_090751` | DXP and runtime complete | `249432 / 262144` mismatched, output nonzero but wrong | Fixing traversal order moves the relayout rows before the matmul. Movement now executes, but the materialized RHS layout is still wrong. |
| 4-core one-hot, tree order + no destination physical offset | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_no_dstoffset_20260703_090954` | DXP and runtime complete | `262144 / 262144` mismatched, output nonzero but wrong | Removing the extra destination byte offset does not fix semantics. The remaining issue is not just double-applying the destination offset. |
| 4-core one-hot, single destination data-ops | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_single_dst_20260703_091725` | DXP and runtime complete | `262100 / 262144` mismatched, output resembles one-hot input | Splitting each source/destination pair into a separate STCDPOpLx row changes the failure mode but does not fix values. At the default destination base, the staged RHS appears to collide with matmul input staging. |
| 4-core one-hot, single destination data-ops + high destination base | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_single_dst_highbase_20260703_092119` | DXP and runtime complete | `262144 / 262144` mismatched, output repeats values from the last source band | Moving the destination to `1048576` removes the one-hot collision pattern. The remaining wrong output shows source bands are not landing in distinct destination slots. |
| 4-core one-hot, high destination base + destination byte offsets | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_single_dst_highbase_dstoffset_20260703_092445` | DXP and runtime complete | `228312 / 262144` mismatched, output still repeats one value across `out` | Adding physical destination offsets improves the mismatch count, but STCDPOpLx still does not materialize the 2-D RHS rectangle in the layout consumed by matmul. |
| 4-core one-hot, grouped all-gather data-op | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_grouped_fission4_highbase_20260703_093024` | DXP and runtime complete | `228312 / 262144` mismatched, same repeated-value pattern | Describing all source pieces and all destination full-output pieces in one data-op, matching the existing Deeptools all-gather helper shape more closely, still does not produce a value-correct RHS. |
| 4-core one-hot, grouped data-op + forced `out,in` layout | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_grouped_fission4_highbase_outin_20260703_093311` | DXP and runtime complete | `261880 / 262144` mismatched | Forcing the data-op layout order from `[in,out]` to `[out,in]` changes frame size and values but remains wrong. This rules out a trivial two-dimension order swap. |
| Standalone 4-core STCDP broadcast, same destination base | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/dataop_matmul_operand_broadcast4_20260703_100107` | `DataOpStandalone` and `senpcfg` complete | Destination windows collapse/overwrite | Replicating all producer chunks to the same output start address is not enough; later chunks overwrite the same physical destination slot. |
| Standalone 4-core STCDP broadcast, destination chunk offsets | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/dataop_matmul_operand_broadcast4_chunk_offsets_20260703_101002` | `DataOpStandalone` and `senpcfg` complete | `BROADCAST_MAP_OK True`; every destination core receives all four chunks | STCDPOpLx can materialize the RHS broadcast if each source chunk is assigned its own logical destination offset. |
| Integrated 4-core one-hot, high base + fission + destination chunk offsets | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_multiplacement_offsets_fission4_highbase_debug_20260703_102440` | DXP and runtime complete; relayout row executes before matmul | `228312 / 262144` mismatched, max diff `47.0` | The same offset idea is not sufficient once inserted before matmul. Movement is scheduled and the consumer base is rewritten, but the matmul KERNEL operand reader still consumes the staged RHS incorrectly. |
| Integrated 4-core one-hot, explicit replicated wkSlice | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_explicit_replicated_wkslice_20260703_104312` | DXP aborts in DDC | `Unexpected corelet cardinality mismatch for nodes allocate-Tensor1_lx and transfer_lds1_src:lxlu_dst:ptrow0` | Naively representing the staged RHS as "all cores own the same full slice" is not accepted by existing DDC folding. |
| Integrated 4-core one-hot, post-codegen KERNEL state debug | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_postcodegen_20260703_105302` | DXP and runtime complete | `coreStateInitSize=0`, `lxBufferSize=UINT64_MAX`, still `228312 / 262144` mismatched | The previous LBR-state hypothesis is not enough for this generated path; the KERNEL input is not represented through populated `coreStateInit_` here. |
| 4-core one-hot, no-relayout control | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_control_no_relayout_20260703_105538` | DXP and runtime complete | `ALLCLOSE True`, max diff `0.03125` | The base matmul and input data are valid. The value error is introduced by the inserted relayout path, not by the probe itself. |
| Integrated 4-core one-hot, min LDS base instead of last chunk base | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_minbase_core_lxstart_20260703_110300` | DXP and runtime complete | unchanged: `228312 / 262144` mismatched, max diff `47.0` | Keeping `coreIdTolxStartAddress_` at the first/min destination chunk does not fix the integrated matmul read. |
| Integrated 4-core one-hot, DataInfo/unit-view dump | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_datainfo_view_20260703_111341` | DXP and runtime complete | `DataInfo.startAddr_` is `0`; unit view is `[(out,64),(in,16),(out,1)]` with only `in` outer loops | Pre-DDC allocation rewrite does not survive into the live MAC input address/view. |
| Integrated 4-core one-hot, force `DataInfo.startAddr_ = 1048576` after DDC | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_force_datainfo_base_20260703_111728` | DXP and runtime complete | unchanged: `228312 / 262144` mismatched, max diff `47.0` | Rewriting `DataInfo.startAddr_` after DDC is still not enough; the remaining failure is lower than, or in addition to, the DataInfo base field. |
| Integrated 4-core one-hot, preserve producer `coreIdToWkSlice_` | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_preserve_coords_autoload_20260703_113418` | DXP aborts in DDC | `Can not propagate coordinates for coreletSplit dimensionout from allocateNode allocate-Tensor1_lx with custom coreIdToWkSlice` | The original producer shard map cannot simply be preserved for the staged broadcast operand. DDC needs a valid post-relayout allocation/view contract rather than the old producer map plus address rewrites. |
| Integrated 4-core one-hot, rewrite staged allocation to consumer `coreIdToWkSlice_` | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_consumer_coords_20260703_114626` | DXP aborts in DDC | `Unexpected corelet cardinality mismatch for nodes allocate-Tensor1_lx and transfer_lds1_src:lxlu_dst:ptrow0` | Consumer compute coordinates are not a valid drop-in replacement for the existing KERNEL allocation. The staged operand still needs a view that DDC can reconcile with the transfer rows. |
| Integrated 4-core one-hot, clear allocation coordinates entirely | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_clear_coords_20260703_114846` | DXP aborts in DDC | `Coordinates of transfer transfer_lds1_src:lxlu_dst:ptrow1 and allocateNode allocate-Tensor1_lx are not consistent` | Clearing the coordinate object is also insufficient. DDC reconstructs/uses an allocation coordinate that is still inconsistent with the transfer rows, so in-place mutation is not isolated enough. |
| Integrated 4-core one-hot, append staged KERNEL labeled DS | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_staged_lds_20260703_115623` | DXP aborts in scheduler | `Expect only one LabeledDs with DsType KERNEL` | A second KERNEL operand is rejected. The consumer matmul descriptor cannot simply keep the original RHS KERNEL and add a staged KERNEL. |
| Integrated 4-core one-hot, staged KERNEL + retag original as INTERNAL | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_staged_lds_internal_20260703_115938` | DXP aborts in scheduler | `out_reuse_dim.size() == 1` | Appending the staged operand after the output breaks a positional scheduler assumption: matmul still expects output to be the last labeled DS. |
| Integrated 4-core one-hot, insert staged KERNEL before output | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_staged_lds_insert_20260703_120414` | DXP aborts later | `vector::_M_range_check: __n (which is 3) >= this->size() (which is 3)` | Preserving output-last avoids the previous scheduler assertion, but adding a fourth labeled DS still trips code that sizes vectors for the original three matmul operands. |
| Integrated 4-core one-hot, replace RHS memorg with cloned staged allocation | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_staged_lds_inplace_20260703_120833` | DXP aborts in DDC | `External allocate node not connected to memorg: allocate-Tensor1_lx` | Replacing the memorg pointer leaves the original allocation node in the schedule tree but detached from the labeled tensor. |
| Integrated 4-core one-hot, mutate RHS allocation node in place | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_staged_lds_inplace_mutate_20260703_121120` | DXP aborts in DDC | `Coordinates of transfer transfer_lds1_src:lxlu_dst:ptrow1 and allocateNode allocate-Tensor1_lx are not consistent` | Reusing the original allocation node fixes graph wiring, but the old producer coordinate map still conflicts with the consumer matmul transfer view. |
| Integrated 4-core one-hot, mutate in place and explicitly clear stale `coreIdToWkSlice_` | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_staged_lds_inplace_clear_wkslice_20260703_121518` | DXP aborts later | `vector::_M_range_check: __n (which is 3) >= this->size() (which is 3)` | Removing the stale ownership map gets past the coordinate mismatch but still exposes a downstream three-operand vector assumption. |
| Integrated 4-core one-hot, mutate in place with replicated RHS coordinates | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/min_matmul_operand_broadcast_onehot_staged_lds_replicated_coords_20260703_122244` | DXP aborts in DDC | `Coordinates of transfer transfer_lds1_src:lxlu_dst:ptrow1 and allocateNode allocate-Tensor1_lx are not consistent` | A simple replicated `coreIdToWkSlice_` map is also not enough. The post-broadcast RHS view must include the matmul reader's transfer/corelet/loop semantics, not just a repeated per-core slice map. |

## Artifact Files

Each variant directory contains:

- `run_command.sh`: exact run command and environment.
- `105_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`: backend plan artifact with the 1024 logical source/destination core pairs.

The 4-core one-hot subdirectories contain:

- `summary.txt`: concise correctness and DXP execution-order extract.
- `run.log`: full torch-spyre/DXP/runtime log.
- `backend_plans/`: emitted 16-transfer plan plus inserted `STCDPOpLx` relayout SDSC JSONs.

`cdx_deeptools_current_experimental_diff.patch` records the CDX experimental backend diff at the best current diagnostic state. This is not production-clean; it includes investigation hooks and filters.

Additional focused artifacts from the 2026-07-03 offset-address probe:

- `standalone_offset_broadcast/collapsed_same_base_sdsc.json`: standalone negative control where all destination chunks share one physical base.
- `standalone_offset_broadcast/offset_chunks_sdsc.json`: standalone passing STCDPOpLx descriptor with per-chunk destination addresses.
- `standalone_offset_broadcast/offset_chunks_memory_map_check.txt`: byte-level source/destination comparison; all four destination cores match all four source chunks.
- `integrated_offset_highbase/inserted_relayout_sdsc.json`: inserted pre-matmul relayout SDSC generated by the DXP experiment.
- `integrated_offset_highbase/matmul_operand_broadcast_plan.json`: logical 4x4 transfer plan used by the integrated probe.
- `integrated_offset_highbase/run_key_excerpts.txt`: compact proof that the inserted relayout row executes before matmul and the final matmul output is still wrong.
- `deeptools_current_dirty_matmul_broadcast_diff.patch`: full dirty CDX diagnostic diff for provenance only.
- `explicit_replicated_wkslice/summary.txt`: DDC abort when the staged RHS is represented by a replicated `coreIdToWkSlice_`.
- `post_codegen_kernel_state/summary.txt`: proof that the generated KERNEL input state has empty `coreStateInit_` even after DDC/codegen.
- `control_no_relayout/summary.txt`: value-correct control with relayout disabled.
- `minbase_core_lxstart/summary.txt`: negative result for the "LDS base overwritten by last chunk" hypothesis.
- `deeptools_current_dirty_after_minbase_probe.patch`: full dirty CDX diagnostic diff after the min-base probe.
- `datainfo_view/view_summary.txt`: post-DDC/post-codegen `DataInfo` and `UnitView` dump for the KERNEL RHS.
- `force_datainfo_base/summary.txt`: negative result for forcing the live `DataInfo.startAddr_` to the staged RHS base.
- `preserve_producer_coords/summary.txt`: DDC abort proving that preserving the original producer coordinate map is not a valid staged broadcast representation.
- `consumer_coords/summary.txt`: DDC abort proving that replacing producer coordinates with consumer compute coordinates is not a valid staged broadcast representation.
- `clear_coords/summary.txt`: DDC abort proving that clearing the original allocation coordinates in place is not enough to create a fresh staged RHS view.
- `cdx_deeptools_current_experimental_diff_after_clear_coords.patch`: dirty CDX diagnostic diff after the latest coordinate probes.
- `staged_lds_append/`: DXP abort proving a second KERNEL labeled DS is not accepted.
- `staged_lds_internal/`: DXP abort proving that retagging the original KERNEL as INTERNAL while appending the staged KERNEL breaks output-last scheduler assumptions.
- `staged_lds_insert/`: DXP abort proving that inserting the staged KERNEL before the output still hits three-operand vector assumptions.
- `staged_lds_inplace/`: DDC abort proving that replacing the RHS memorg pointer leaves the original schedule-tree allocation detached.
- `staged_lds_inplace_mutate/`: DDC abort proving in-place allocation retargeting still has stale producer-coordinate semantics.
- `staged_lds_inplace_clear_wkslice/`: DXP abort proving explicit stale-map clearing gets farther but still does not define a complete post-relayout RHS view.
- `staged_lds_replicated_coords/`: DDC abort proving a naive replicated `coreIdToWkSlice_` map is not enough for the matmul KERNEL reader.

## Current Read

The matmul-operand broadcast contract is correctly identified at the Torch/DLDSC level, and the backend can now compile and run the isolated edge with fissioned movement SDSCs. The standalone `MatmulOperandBroadcast4` probe proves one important point: STCDPOpLx can express the needed 4-source to 4-destination RHS broadcast when the output pieces use distinct per-chunk destination offsets.

The remaining failure is integration with the consumer matmul KERNEL operand. In the integrated one-hot probe, the inserted relayout SDSC executes before `batchmatmul`, and the consumer base is rewritten to the staged destination region. The final matmul output is still wrong. That means the next bug is not basic ring movement; it is the contract between the materialized RHS layout and the matmul operand reader/staging path.

This is why PR1 scatter is not enough for this attention spill. Scatter is one producer shard to one consumer shard. This edge is all-gather/replicate: every consumer core needs the full producer-sharded operand.

The latest probes also rule out three tempting explanations:

- The problem is not the probe inputs: the no-relayout control is value-correct.
- The problem is not simply an LBR rewrite bug: `coreStateInit_` is empty for this KERNEL input after DDC/codegen.
- The problem is not simply the data-op LDS base being overwritten by the last destination chunk: forcing it to the first/min destination base leaves values unchanged.
- The problem is not solved by a late `DataInfo.startAddr_` rewrite: forcing the MAC RHS input base to the staged high-base region leaves the same value pattern.
- The problem is not solved by preserving the producer coordinate map: DDC rejects that representation because the old producer `coreIdToWkSlice_` does not propagate through the consumer's `out` corelet split.
- The problem is not solved by replacing the producer map with the consumer map: DDC rejects that representation with a corelet cardinality mismatch.
- The problem is not solved by clearing the original allocation coordinates in place: DDC still sees coordinates inconsistent with the transfer rows.

Source inspection points to the active matmul RHS address path:

- `L3DlOpsScheduler::fillDataInfo` copies `AllocateNode::startAddressCoreCorelet_` into `ComputeNode::inputsLdsAndLoopOffsets_[1].startAddr_`.
- `DesignSpaceConfig::finalizeScheduleTree` builds `ComputeNode::coreletViews_[cl].inputsLoopsAndSizes_[1]` from that `DataInfo`.
- DCC lowers the MAC RHS load from `inputsLdsAndLoopOffsets_[1].startAddr_` plus `coreletViews_[cl].inputsLoopsAndSizes_[1]`.

So the next debug target is the post-DDC/post-codegen `DataInfo` and `UnitView` for KERNEL input 1, not `coreStateInit_`.

The post-DDC view dump shows:

```text
DataInfo.startAddr_ = 0
sizesNoGaps_ = [(out,64), (in,16), (out,1)]
outerLoops_ = in only
```

Forcing `DataInfo.startAddr_` to the staged base changes the dumped address to `1048576`, but the output remains unchanged. That suggests either DCC's emitted vector-load path is using an already-derived/cached address form, or the KERNEL unit-view/layout contract still describes the original sharded operand rather than the post-relayout replicated RHS.

Preserving the original producer `coreIdToWkSlice_` makes DDC abort before codegen. This rules out the smallest backend-only fix of "keep the producer coordinates and update addresses." For matmul-operand broadcast, the backend or frontend must construct a real post-relayout logical allocation/view: per destination core, the staged RHS has all source chunks in a layout that the matmul KERNEL reader can consume.

Two follow-on coordinate probes were also negative. Replacing the staged operand map with the consumer compute map trips a corelet cardinality mismatch. Clearing the coordinate object in place trips a transfer/allocation coordinate mismatch. Together, these rule out simple in-place mutation of the existing `Tensor1` allocation. The next viable prototype should either clone the KERNEL labeled DS and redirect the consumer matmul input to that staged DS before DDC, or have Torch emit a named post-relayout operand contract that Deeptools binds to a staged allocation.

The staged-LDS probes narrow that recommendation further. A naive cloned KERNEL labeled DS conflicts with existing Deeptools matmul assumptions: only one KERNEL operand is allowed, output is expected to be last, and some vectors are still sized for the original three labeled tensors. The lowest-friction backend shape appears to be a first-class post-relayout operand contract attached to the existing RHS operand: Deeptools should know that Tensor1's consumer-side KERNEL view is a replicated/all-gathered view materialized at a staged LX address, then build the allocation, coordinates, and matmul reader state coherently in one path.

## Runbook: 4-Core Matmul Operand Probe

Use the 4-core PyTorch/torch-spyre probe as the fast repro before returning to full `test_flash.py`. The shape is a computed RHS operand feeding matmul Tensor1:

```text
rhs = v * scale
out = x @ rhs
producer rhs work_div: N across 4 cores
consumer matmul work_div: M across 4 cores
expected contract: matmul_operand_broadcast / all_gather_replicate
expected logical transfers: 4 producer chunks x 4 consumer replicas = 16
```

Run with the matmul operand lane enabled and layout-allgather disabled:

```bash
SENCORES=4
SPYRE_LX_PLANNER_RELAYOUT=1
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=0
SPYRE_INDUCTOR_LOG=1
SPYRE_INDUCTOR_LOG_LEVEL=DEBUG
DXP_LX_FRAC_AVAIL=0
```

The SDSC evidence to verify is:

```text
kind=matmul_operand_broadcast
communication_pattern=all_gather_replicate
operand_read_index=1
input_labeled_ds=Tensor1
realization_strategy=loop_scoped_input_fetch
source_core_count=4
consumer_core_count=4
logical_transfer_count=16
```

The forced-destination one-hot diagnostic narrowed the failure in two steps. First, forcing every transfer to one chosen destination changes the consumer operand base selected by the generated matmul program, so DXP is seeing and applying the intended destination base. Second, after adding a tree-order traversal fix, inserted relayout SDSCs compile and execute before the consumer matmul instead of after it.

That rules out several earlier suspects: the issue is not pure Torch metadata, not simple destination-base lookup, not lack of generated movement programs, and not only execution ordering. The remaining gap is STCDPOpLx materialization semantics for this RHS all-gather layout. The inserted movement rows are now scheduled before the matmul and produce nonzero output, but the logical RHS chunks land/read as the wrong layout.

The latest probes narrow that further:

- One data-op per source/destination pair is not sufficient.
- A high destination base avoids a likely collision with matmul input staging, but still leaves source bands mapped incorrectly.
- Adding physical destination byte offsets partially helps, but the output still repeats one value across the `out` dimension.
- A grouped all-gather-style data-op with all sources and all destination full-output pieces is still value-wrong.
- Forcing `[out,in]` instead of `[in,out]` is still value-wrong.
- A standalone STCDPOpLx descriptor with explicit destination chunk offsets is value-correct, so the backend has enough low-level machinery for the ring copy itself.
- The integrated pre-matmul version is still value-wrong, so the next investigation should target matmul KERNEL operand address/layout state, not another generic transfer-only tweak.
- Explicitly encoding the staged RHS as a replicated work-slice map currently trips DDC cardinality checks.
- `coreStateInit_`/LBR state is not the active KERNEL address path in this generated probe, so address debugging must move to the actual batchmatmul KERNEL descriptor/setup path.
- A direct post-DDC `DataInfo.startAddr_` override is not sufficient, so the next likely fix needs a proper post-relayout logical allocation/view for the replicated RHS, not just a relocated address.

The next check should not be another full attention run. It should be a Deeptools-level unit or hardware test for the exact STCDPOpLx piece model needed here: a sharded LX KERNEL operand, all-gathered into a full per-consumer-core RHS region, then consumed by a matmul descriptor. That test should verify bytes immediately after movement, before matmul, so we can separate movement materialization from matmul reader behavior.

## Next Backend/Frontend Gap

The next clean implementation should make `matmul_operand_broadcast` a first-class backend-lowered communication class:

1. Preserve the DLDSC coordinate contract from Torch: producer tensor distribution, consumer compute distribution, operand index, communication kind.
2. Backend derives source chunk coordinates and destination slots, not just source/destination core pairs.
3. Backend materializes either a full resident replicated operand or a loop-scoped input fetch that feeds the matmul reader without overwriting chunks.
4. Add a small value-pattern unit test before re-running full flash attention.

## Current Backend Bugs Proven By The 4-Core Probe

1. `SdscTree::getAllSdscNodes()` is documented as pre-order but returned storage insertion order. Inserted relayout SDSCs were linked before the consumer in the tree, but later DXP codegen iterated the storage vector and executed them after the matmul. The experimental tree-order fix changes execution order to `mul -> relayout fissions -> batchmatmul`.
2. After fixing ordering, the STCDPOpLx rows execute but produce the wrong replicated RHS. The inserted SDSC JSON is logically shaped as source shard to all destination cores, but the final matmul sees shifted/constant chunks rather than the one-hot expected RHS rows.
3. Removing destination physical subpiece offsets does not fix the value error. The likely remaining problem is the contract between physical `startAddr`, logical `dimToStart`, and how STCDPOpLx generates multi-destination LX writes for a matmul KERNEL operand.
4. Moving the destination base high changes the wrong values, proving destination allocation/collision matters.
5. Destination physical offsets, grouped source pieces, and forced `out,in` layout each change the failure mode but do not produce correctness. The current backend implementation needs a real matmul-operand all-gather lowering, not another small address tweak.
