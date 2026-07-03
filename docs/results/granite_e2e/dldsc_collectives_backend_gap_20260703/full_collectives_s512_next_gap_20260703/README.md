# Full Collectives S512 Next Gap

This artifact records the first full S512 Granite prefill attempt after grouped KERNEL operand broadcast replay was fixed.

## Result

- Torch SHA: `f0b7b5748d0381b6c5bb41cd5639718d986ef425`
- Deeptools SHA: `1446330381d84c6086e5131742011727f6883d5b`
- Run: `/home/adnan/codex-isolated/dldsc_granite_clean_relayout_20260703_163108/runs/granite_relayout_s512_kernel_neighbor_after_20260703_234640`
- Return code: `1`

The run progressed past the previous grouped RHS/KERNEL broadcast blocker, emitted realized backend plans, and then failed on the next communication class.

## Observed Plans

```json
[
  {
    "artifact_kind": "layout_allgather_restickify_backend_plan",
    "sdsc_name": "10_batchmatmul",
    "input_labeled_ds": "Tensor1",
    "realized": true,
    "physical_lowering_status": null,
    "movement_pattern": "",
    "communication_pattern": null,
    "logical_transfer_count": "0",
    "source_core_count": "0",
    "consumer_core_count": "0",
    "group_count": "0"
  },
  {
    "artifact_kind": "matmul_operand_broadcast_backend_plan",
    "sdsc_name": "18_batchmatmul",
    "input_labeled_ds": "Tensor1",
    "realized": true,
    "physical_lowering_status": "lowered_loop_scoped_kernel_neighbor",
    "movement_pattern": null,
    "communication_pattern": "all_gather_replicate",
    "logical_transfer_count": "1024",
    "source_core_count": "32",
    "consumer_core_count": "32",
    "group_count": "1"
  }
]
```

## Failure

The relevant backend error is captured in `logs/error_and_classification_excerpt.txt`:

```text
DtException: [buildFoldFromAllocation] Can not propagate coordinates for coreletSplit dimensionmb from allocateNode allocate-Tensor0_lx with custom coreIdToWkSlice.
```

## Current Read

This is not the same failure as the earlier KERNEL-neighbor broadcast capacity issue. The matmul operand broadcast plan now emits as realized through `lowered_loop_scoped_kernel_neighbor`.

The next gap is `layout_allgather_restickify`: the plan artifact currently has zero movement fields, which means the backend accepted the classification key but did not synthesize a real movement plan for this Granite coordinate shape. DXP then falls into diagnostic/generic relayout glue and DDC cannot propagate the custom LX allocation coordinates.

## Next Implementation Work

1. Fail closed when `layout_allgather_restickify` metadata is present but the synthesized movement plan is invalid, and include the synthesis reason in the artifact.
2. Extend layout-allgather/restickify planning to derive source groups/chunks and consumer groups from the explicit coordinate maps, not just hard-coded dimension names/indexes.
3. Realize that class as a staged KERNEL-neighbor/restickify path, rather than reusing the matmul broadcast IFN helper with an empty plan.
