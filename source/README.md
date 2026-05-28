# Source program for the SDSC dump

Python source + Inductor compile artifacts that produced
`sdsc_dumps/mlp_M512_K4096_N12800/`.

## Files

- `compile_mlp_matmul.py` — runnable Python source. Reproduces the compile.
- `env.sh` — exact environment used (cost-model planner ON, etc.).
- `inductor_debug/` — Inductor compile-debug artifacts captured with
  `TORCH_COMPILE_DEBUG=1`:
  - `fx_graph_readable.py` — Dynamo FX graph (human-readable)
  - `fx_graph_runnable.py` — runnable FX graph for re-execution
  - `fx_graph_transformed.py` — FX graph after AOT/inductor transforms
  - `ir_pre_fusion.txt` — Inductor IR before fusion
  - `ir_post_fusion.txt` — Inductor IR after fusion (what feeds codegen)
  - `output_code.py` — Inductor's emitted Python wrapper that calls the SDSC

## To reproduce

```bash
source env.sh             # sets PYTHONPATH, SPYRE_COST_MODEL_MATMUL_PLANNER=1, etc.
python compile_mlp_matmul.py
```

SDSC bundle lands in `/tmp/torchinductor_<user>/inductor-spyre/sdsc_fused_linear_*/`.

## Matmul shape

`torch.nn.functional.linear(x, W.T)` with:
- `x` shape `[1, 512, 4096]` fp16
- `W` shape `[4096, 12800]` fp16 (so `W.T` is `[12800, 4096]`)
- Output `[1, 512, 12800]`

Cost-model picks `(m=8, n=4, k=1)`. SDSC's `numWkSlicesPerDim_` should
reflect this.
