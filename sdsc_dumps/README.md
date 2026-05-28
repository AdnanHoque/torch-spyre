# SDSC dumps for MLP matmul [1,512,4096] x [4096,12800]

Two SDSC bundles for the same matmul:

| Dir | Planner | numWkSlicesPerDim_ | Note |
|---|---|---|---|
| `mlp_M512_K4096_N12800_heuristic/` | OFF (default) | `{"x": 32, "out": 1, "in": 1}` | Heuristic — splits batch-side 32 ways |
| `mlp_M512_K4096_N12800_cost_model/` | ON (`SPYRE_COST_MODEL_MATMUL_PLANNER=1`) | `{"mb": 8, "out": 4, "in": 1}` | Cost-model — m=8 × n=4 = 32 cores |

**Label difference worth noting:** the heuristic SDSC uses `"x"` for the
M-side split (treating M as batch); the cost-model SDSC uses `"mb"`
(treating it as a 2D matmul with M+minibatch). This may affect whether
corelet splitting fires in the backend.

## Source program

See `../source/` for the Python source, env, and Inductor compile-debug
artifacts (FX graph, IR pre/post fusion, output_code).
