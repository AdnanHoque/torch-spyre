# SDSC dump: MLP matmul [1,512,4096] x [4096,12800]

Captured 2026-05-28 from cost-model branch (HEAD `7ea54b7`) with
`SPYRE_COST_MODEL_MATMUL_PLANNER=1`.

This is the Granite-style MLP `gate`/`up` projection at prefill bs=1:
torch.nn.functional.linear(x, W.T) where
- x has shape `[1, 512, 4096]` (B=1, M=512, K=4096) fp16
- W has shape `[4096, 12800]` (K=4096, N=12800) fp16

Cost-model planner picked (m=8, n=4, k=1) — see `sdsc_0_batchmatmul.json`
`numWkSlicesPerDim_`. Documented as the "Granite N=12800" known-limit
case in `docs/source/_static/cost_model_findings.md` (empirical winner
is (4,8,1); the planner picks (8,4,1) which is still a net win over the
heuristic).

Files:
- `sdsc_0_batchmatmul.json` — SDSC for the matmul kernel
- `bundle.mlir` — MLIR bundle wrapper
- `execute*`, `loadprogram_*`, `loadmodel_*` — runtime descriptors
- `segment_size.json` — HBM segment plan
