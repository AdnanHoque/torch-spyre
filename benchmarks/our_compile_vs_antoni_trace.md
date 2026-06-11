# Our compile vs Antoni's e2e trace — delta & perf-transfer analysis

Compares the kernel inventory our 1-layer Granite compile emits against
[`antoni_trace_golden_granite_ops.md`](antoni_trace_golden_granite_ops.md)
(the ground-truth launch-kernel set from two full Granite-8B e2e Kineto traces).
The question this answers: **does optimizing the matmuls in our compile transfer
to Antoni's e2e run, or does the fusion difference poison it?**

## TL;DR

- **Kernel-name match: 10 / 20.** The 10 mismatches are **not different math** —
  they are the *same transformer roles* with a different **fusion boundary**,
  almost entirely about *where rms_norm gets fused*. The matmul shapes are
  identical (architecture-fixed).
- **Work-division splits are fusion-name-invariant** (proven below): one matmul
  shape gets one split regardless of which fused kernel it lives in. The planner
  keys on `(B,M,N,K)`, not the surrounding pointwise fusion.
- **Therefore shape-level cost-model improvements transfer to Antoni's e2e.**
  Validate on the bare `B×M×N×K` matmul (as in
  [`golden_granite_ops.md`](golden_granite_ops.md)), not on our fused-kernel
  names, and the result holds wherever that matmul is fused.

## The delta (role-aligned)

| transformer role | Antoni kernel | our kernel | nature of delta |
|---|---|---|---|
| Q/K/V proj + input-norm | `linear_mul_rms_norm_sum_unsqueeze_view_0` | `linear_mul_rms_norm_0` | his fuses the norm reduction tail in |
| prefill MLP down + residual | `add_linear_mul_4` | `add_linear_mul_rms_norm_5/6` | ours folds next-layer rms_norm in |
| decode MLP gate/up SiLU | `linear_mul_rms_norm_silu_4` | `add_linear_mul_rms_norm_silu_4` | residual-add + norm placement |
| decode MLP down + residual | `add_linear_mul_silu_6` | `add_linear_mul_rms_norm_6` | norm placement |
| decode MLP residual tail | `add_mul_5` + `add_mean_mul_rsqrt_0` | `add_mul_rms_norm_4` + `rms_norm_6/7` | ours fuses norm in, emits standalone norm |
| prefill attn out + norm | `…_add_clone_expand_linear_mul_rms_norm_unsqueeze_view_3` | `…_clone_expand_linear_unsqueeze_view_3` | rms_norm + mul fused in his |
| decode KV-cache overwrite | `linear_overwrite_slice_transpose_view_1` | `linear_mul_overwrite_slice_sum_transpose_unsqueeze_view_1` | ours fuses extra mul+sum+unsqueeze |
| embedding multiplier | `mul_0` | (fused away) | harness |
| input padding | (none) | `constant_pad_nd_0` | harness (`pad_input_ids`) |

## Root cause — harness/model code, not the device backend

The matmul shapes are architecture-fixed and identical. The epilogue fusion
differs because of:

- our `generation.py` CPU-side modifications (mask build, token scatter) reshape
  the decode graph's pointwise ops → different fusion groups;
- Antoni's Priyanka `generation.py` (different decode-multiple / KV path);
- harness: our `pad_input_ids` emits `constant_pad_nd`; embedding-scale handling
  becomes his standalone `mul_0`.

## Proof: splits are fusion-name-invariant

In our own run, several matmul shapes appear in **multiple differently-named
fused kernels**, and every one gets the **same split**:

| shape (B×M×N×K) | split | # distinct fused kernels |
|---|---|---:|
| 1×512×4096×4096 (Q/O proj) | `mb4,out8,in1` | 3 |
| 1×64×4096×4096 (Q/O proj) | `mb4,out8,in1` | 3 |
| 1×64×1024×4096 (K/V proj) | `mb4,out8,in1` | 3 |
| 1×64×12800×4096 (MLP up) | `mb4,out8,in1` | 4 |
| 1×64×4096×12800 (MLP down) | `mb4,out8,in1` | 2 |
| …all 12 shapes | one split each | CONSISTENT |

Same shape → same split across every fused kernel it appears in. The planner
decides on the matmul iteration space; the fusion name is downstream.

## Risk assessment for cost-model work

- **Projection + MLP matmuls (the compute bulk): low risk.** Splits are
  shape-keyed and shapes are identical in Antoni's run → improvements transfer.
- **Attention kernels: low-medium risk.** A fused transpose/view *could* alter
  the iteration space the planner sees. Our data shows consistent splits even
  there, and the shapes are architecture-fixed, so exposure is small. Derisk by
  validating on the bare shape.
- **Harness kernels (`constant_pad_nd_0`, `mul_0`): irrelevant** to perf transfer.

## Getting a literally trace-representative run

For *trace-name* matching (not required for cost-model validation): align the
harness with Antoni — Priyanka's `generation.py` (drop the CPU-mask hack),
`FakeEmbedding`/`FakeLinear` (removes `mul_0` + `constant_pad_nd`), matching
sdpa/attention config. Expect convergence toward 20/20.
