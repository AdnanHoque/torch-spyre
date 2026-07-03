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

## Artifact Files

Each variant directory contains:

- `run_command.sh`: exact run command and environment.
- `105_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`: backend plan artifact with the 1024 logical source/destination core pairs.

The 4-core one-hot subdirectories contain:

- `summary.txt`: concise correctness and DXP execution-order extract.
- `run.log`: full torch-spyre/DXP/runtime log.
- `backend_plans/`: emitted 16-transfer plan plus inserted `STCDPOpLx` relayout SDSC JSONs.

`cdx_deeptools_current_experimental_diff.patch` records the CDX experimental backend diff at the best current diagnostic state. This is not production-clean; it includes investigation hooks and filters.

## Current Read

The matmul-operand broadcast contract is correctly identified at the Torch/DLDSC level, and the backend can now compile and run the isolated edge with fissioned movement SDSCs. The remaining failure is semantic: a 32-source to 32-destination operand all-gather cannot be represented as only `(source_core, destination_core)` transfers. The physical lowering also needs the logical destination slot for each producer chunk, and likely needs the matmul operand reader to consume the staged chunks with the same logical coordinates.

This is why PR1 scatter is not enough for this attention spill. Scatter is one producer shard to one consumer shard. This edge is all-gather/replicate: every consumer core needs the full producer-sharded operand.

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
