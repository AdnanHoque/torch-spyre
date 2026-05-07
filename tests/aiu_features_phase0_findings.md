# AIU hardware-features Phase 0 batch — findings

## TL;DR

Three investigations, three different verdicts:

| project | hardware support | deeptools support | torch_spyre support | solo torch_spyre? |
|---|---|---|---|---|
| **A. 2:4 sparse PT decomposition** | yes (ISA §2.9) | INT8-only (3 templates) | none | **no** — needs fp16 sparse kernel template |
| **B/C. LX_SAMV/LX_SPMV (paged-KV / MoE)** | yes | LX_SAMV used narrowly; LX_SPMV unused | none | **no** — needs significant deeptools work |
| **E. PE fusion audit** | yes (PE rich ISA) | 66 templates exist | fusion.py limits to 6 tensors/bundle | **yes — and a new project surfaces** |

The biggest finding: **torch_spyre's fusion is hard-capped at 6 tensors per bundle** (per a TODO citing issue #827). This is likely the bottleneck preventing deeper fusion patterns like matmul + rms_norm + scale. **Lifting that limit could be the most leveraged solo torch_spyre project on the menu.**

## Investigation 1: 2:4 sparse PT

### What I found

Sparse kernel templates EXIST in deeptools:
```
/home/adnan/dt-inductor/deeptools/dvs/libtemplates/
├── batchmatmul_int8_fwd_sparse.smc        ← INT8 batched matmul, sparse
├── conv2d_int8_fwd_dd1a_sparse.smc        ← INT8 conv2d, sparse  
└── conv2d_sparsekg3_fwd.smc                ← INT8 conv with KG3 dataflow
```

These use sparse encoding via Nnz/Ndp/Nptrow indices (per the kernel
template comments). All INT8.

**No fp16 sparse kernel template.** The "2:4" references in the
codebase turned out to be `l0_load_type` codes (`2: 4B, 3: 2B`),
unrelated to N:M structured sparsity ratios.

torch_spyre has **zero** sparse-related code (only a fallback for
`torch.sparse_*` ops to CPU).

### What this means

- For **INT8+sparse path**: hardware ready, kernels exist. Could
  ship 2:4 sparsity if combined with INT8 quantization. ~6-10 weeks
  if quantization infra is in place; longer if not.
- For **fp16+sparse path** (typical LLM workload): need deeptools-
  side new fp16 sparse kernel template first. **Not solo torch_spyre.**

### Project A status

**Closed as solo torch_spyre.** Could be revisited in partnership
with deeptools for an fp16 sparse kernel + torch_spyre lowering.
Or as INT8+sparse if quantization rollout is on the roadmap.

## Investigation 2: LX_SAMV / LX_SPMV

### What I found

**LX_SAMV (Set Address Mask Vector)**: USED in deeptools, but narrowly.
The use cases are all "mask out the remainder elements at the end of
a stick during reduction":
- `sum_fwd.smc`, `sum_opt_fwd.smc` — sum reductions
- `max_fwd.smc`, `max_opt_fwd.smc` — max reductions
- `exx2_fwd.smc` — exp²
- `csq_int4_chil_fwd.smc`, `csq_int8_chil_fwd.smc` — quantization
- `q_fp8_chil_fwd.smc` — fp8 quantization

That's it. SAMV is used for boundary masking, not for general gather.

**LX_SPMV (Set Permutation Mask Vector)**: ZERO uses across deeptools
and torch_spyre. **Completely unused hardware primitive.**

torch_spyre has no paged-attention, KV-cache, or gather-scatter
infrastructure.

### What this means

- LX_SPMV is a hardware primitive **going to waste**. Strong novelty
  angle for a paper/patent.
- For paged-KV (project B) or MoE routing (project C), you'd need
  to build:
  1. New deeptools-side kernel template that uses LX_SPMV
  2. torch_spyre lowering to that template
  3. KV cache / MoE routing infrastructure
- This is a substantial deeptools-side project, not solo torch_spyre.

### Project B/C status

**Closed as solo torch_spyre.** But uniquely strong patent angle:
"first use of LX_SPMV for paged attention on AIU" or "first use of
LX_SPMV for MoE expert routing on AIU." If you can pair with the
deeptools team, the IP claim is clear.

## Investigation 3: PE fusion (the actually-actionable finding)

### What I found

**torch_spyre has a fusion pass** in `_inductor/fusion.py:38`. It
groups SchedulerNodes into bundles (one SDSC Bundle = one kernel
launch). Hard-capped at **6 tensors per bundle**:

```python
# Until https://github.com/torch-spyre/torch-spyre/issues/827 is completed.
_MAX_BUNDLE_TENSORS = 6
```

Behavior: greedy linear pass through nodes, accumulates tensors,
seals the bundle when adding the next node would exceed 6 unique
tensors.

This explains a lot:
- Why RMS norm at decode is ~3 ms (one LF, one bundle) — its
  decomposition fits in 6 tensors.
- Why MLP projections each take their own kernel launch — adding
  rms_norm + matmul + output would exceed 6 tensors.
- Why we see launch floor on so many op pairs — the tensor cap
  forces them apart.

### Existing kernel templates suggest deeper fusion is possible

Inventory of 66 kernel templates in deeptools/dvs/libtemplates:
- `tanh_rsqrt_reciprocal_exp_sigmoid_fuse_fwd.smc` — 5 unary functions
  fused with optional 4-way PE/SFP interleaving
- `add_mul_sub_fwd.smc` — fused arithmetic
- `biasadd_fwd.smc`, `fusedbatchnorm_fwd.smc`, `relu_reluX_fwd.smc`
- `layernorm_norm_fwd.smc` + `layernorm_scale_fwd.smc` (2 kernels for
  LayerNorm — possible split worth investigating)

So the deeptools side has **fused activation kernels**. The torch_spyre
side bundles ops into SDSC Bundles up to 6 tensors. The 6-tensor cap
is the gating constraint.

### Project E status

**This is the strongest solo torch_spyre project surfaced by the
batch.** Two flavors:

**E1: Lift the 6-tensor fusion cap.** Find issue #827, understand
why the cap is there (might be SDSC Bundle hardware limit, might
be a software TODO). If it's lifted to e.g. 10-12 tensors per bundle,
deeper fusion patterns become possible.

**E2: Audit for fusion misses within the 6-tensor cap.** Even within
the current cap, there might be patterns torch_spyre's greedy pass
isn't finding. Reorder fusion-eligible ops, batch better.

Phase 0 sub-investigation: read issue #827, run a few patterns to
see what gets fused vs not. ~1 week to scope.

## What this changes about the brainstorm

Three of three flagship projects from the new-features brainstorm
(A, B, C) close as solo torch_spyre because they all depend on
deeptools-side novel kernel work.

But: the **PE fusion audit (E)** has surfaced a real solo
opportunity in the form of the 6-tensor fusion cap.

Updated solo torch_spyre project ranking:

1. **Investigate / lift the 6-tensor fusion cap** (newly surfaced).
   1-2 weeks Phase 0 → could enable 5-15% block-level wins via
   better fusion. Strong leverage.
2. **LX residency planner**. Still strongest novel project, untouched
   by these investigations.
3. **Fix SDPA-to-bmm regression**. Quick concrete win.
4. **Cost-model-driven planner heuristic**. Builds on existing work.

## What's worth filing externally

- **fp16 sparse kernel template** as a deeptools feature request,
  paired with the 2:4-sparsity-on-AIU paper/patent angle.
- **LX_SPMV-based PagedAttention** as a strategic deeptools+torch_spyre
  joint project. The hardware primitive is wasted today.
- **`torch.maximum` UnimplementedOp** (from the FA tiling work).
- **`to_dtype IEEE_FP32`** unsupported (also from FA work).
- **Dynamic-offset slicing fails stickify** (also from FA work).

## Next step

Strongest next move: investigate issue #827 (the 6-tensor cap).
Cheap, high-leverage, solo torch_spyre. If the cap is a software-
only constraint, lifting it is a single-PR fix that compounds
with every fusion downstream.

## Files

- This doc — combined Phase 0 batch findings
