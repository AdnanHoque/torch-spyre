# SDSC dump: MLP matmul [1,512,4096] x [4096,12800] — COST-MODEL PICK

Captured 2026-05-28 from cost-model branch (HEAD `7ea54b7`) with
`SPYRE_COST_MODEL_MATMUL_PLANNER=1` actually engaged.

Cost-model planner picked `(m=8, n=4, k=1)`. See
`sdsc_0_batchmatmul.json` `numWkSlicesPerDim_`:
```
{"mb": 8, "out": 4, "in": 1}
```

This is the planner's "updated" pick. Compare against the heuristic's
`(32, 1, 1)` pick in `sdsc_dumps/mlp_M512_K4096_N12800_heuristic/`:
```
{"x": 32, "out": 1, "in": 1}
```

Note the LABEL DIFFERENCE in addition to the split values:
- Heuristic SDSC uses `"x"` (batch-like label) for the M-side split
- Cost-model SDSC uses `"mb"` (M+minibatch label)

This suggests the heuristic is treating the matmul as a batched op with
M as the batch dim, while the cost-model planner is treating it as a
genuine 2D matmul with M as a row dim. May be relevant to whether
corelet splitting is firing.

## Files

Same as the heuristic dump (bundle.mlir, runtime descriptors,
segment_size.json, sdsc_0_batchmatmul.json).
