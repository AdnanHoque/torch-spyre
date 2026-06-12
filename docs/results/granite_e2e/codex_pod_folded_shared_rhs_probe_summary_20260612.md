# Codex Pod Folded Shared-RHS E2E Cost Model Probe

## Summary

This probe validates the e2e planner/lowering fix for Granite projection matmuls that are folded to no-batch 2D matmul form before `_cost_model_matmul_planner` runs. The key production concept is `rhs_loaded_once`: a folded projection may have no explicit batch dims left, but its RHS is still an unbatched/shared weight and should be costed as loaded once. True attention BMMs keep `rhs_loaded_once=False` because their RHS depends on batch/head dimensions.

## Code Branch

- Branch: `AdnanHoque/torch-spyre:cost-model-tuned`
- Commit: `dbf4da7` (`inductor: recognize folded shared-rhs matmuls in cost model`)

## Validation

- Unit test: `tests/inductor/test_work_division_cost_model.py` passed (`5 passed`).
- Granite 12-shape split probe emitted the intended work divisions.
- Important transition: prefill `MLP down` now emits `1_4_8_1` instead of the previous `1_8_4_1` when combined with the jointly fitted per-core PSUM / long-K terms.
- Attention BMMs remain on true-BMM splits: prefill `attn@V = 1_16_2_1`, decode `attn@V = 1_4_2_3`, decode `QK^T = 1_4_3_2`.

## Probe CSV

```csv
op,phase,shape,shared_weight,compact_split,sdsc_split,device_us_median
QK^T,prefill,512x32x512x128,False,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
attn@V,prefill,32x512x128x512,False,1_16_2_1,"{'x': 1, 'mb': 16, 'out': 2, 'in': 1}",
Q/O proj,prefill,1x512x4096x4096,True,1_8_4_1,"{'x': 1, 'mb': 8, 'out': 4, 'in': 1}",
K/V proj,prefill,1x512x1024x4096,True,1_8_4_1,"{'x': 1, 'mb': 8, 'out': 4, 'in': 1}",
MLP up,prefill,1x512x12800x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
MLP down,prefill,1x512x4096x12800,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
QK^T,decode,64x32x576x128,False,1_4_3_2,"{'x': 1, 'mb': 4, 'out': 3, 'in': 2}",
attn@V,decode,32x64x128x576,False,1_4_2_3,"{'x': 1, 'mb': 4, 'out': 2, 'in': 3}",
Q/O proj,decode,1x64x4096x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
K/V proj,decode,1x64x1024x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
MLP up,decode,1x64x12800x4096,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
MLP down,decode,1x64x4096x12800,True,1_4_8_1,"{'x': 1, 'mb': 4, 'out': 8, 'in': 1}",
```

## Read

This resolves the prior planner/lowering mismatch: the offline fitter expected folded no-batch projections to be costed as shared/unbatched RHS, but the real planner only set that flag for multi-row-dim broadcast cases. The fix makes the cost-model concept match Granite e2e lowering.
