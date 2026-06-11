# Exhaustive 32-core split sweep — device-best vs cost-model pick

Every full-utilization (32-core) work-division split, device-timed standalone,
for all 12 golden Granite matmul shapes (240 forced-split runs, 233 valid, 7
device hangs skipped). This is the rigorous version of "are we picking
device-best splits" — the prior hand-picked sweep covered only ~40% of the
32-core split space and **0%** of the K-split / batch-split families, which is
where the misses turn out to live.

Data: [`exhaustive_summary.csv`](exhaustive_summary.csv) (per-shape best),
[`exhaustive_full.csv`](exhaustive_full.csv) (all 233 split timings).

## Method

- For each shape, force every `(b,m,n,k)` with `b·m·n·k = 32` (each dim
  dividing its size) via a `_matmul_split_cost` patch, compile standalone,
  median of 7 device runs.
- Device noise floor here is **~10%** (same split re-timed varies ~9%), so only
  gains **>10%** are trustworthy; 5–10% is borderline.

## Result

| shape | role | phase | cost pick | cost ms | device-best | best ms | gain | winner |
|---|---|---|---|---:|---|---:|---:|---|
| 512×4096×4096 | Q/O | prefill | mb4,out8,in1 | 3.243 | same | 3.243 | 0% | **optimal** |
| 512×1024×4096 | K/V | prefill | mb4,out8,in1 | 1.245 | same | 1.245 | 0% | **optimal** |
| 512×4096×12800 | MLP down | prefill | mb4,out8,in1 | 4.899 | same | 4.899 | 0% | **optimal** |
| 512×12800×4096 | MLP up | prefill | mb4,out8,in1 | 12.457 | **mb4,out4,in2** | 10.632 | **14.7%** | **K-split** |
| 32×512×128×512 | attn@V | prefill | mb32 (pure-M) | 4.803 | b8,m2,n2 | 4.521 | 5.9% | batch-split |
| 64×12800×4096 | MLP up | decode | mb4,out8,in1 | 2.007 | same | 2.007 | 0% | **optimal** |
| 64×4096×4096 | Q/O | decode | mb4,out8,in1 | 0.829 | m2,**in16** | 0.745 | 10.1% | K-split |
| 64×4096×12800 | MLP down | decode | mb4,out8,in1 | 1.307 | mb4,out4,**in2** | 1.213 | 7.2% | K-split |
| 64×1024×4096 | K/V | decode | mb4,out8,in1 | 0.300 | m8,**in4** | 0.283 | 5.7% | K-split |
| 32×64×128×576 | attn@V | decode | mb16,out2 | 0.776 | b4,m8 | 0.701 | 9.7% | batch-split |

(The two head-folded prefill QK^T shapes had 90%+ noise spreads and are omitted
from conclusions.)

## Findings

1. **The cost model is structurally blind to K-splits and batch-splits.** It
   emits `in1` / `x1` on every shape. Every above-noise miss is a K-split
   (the cost model's PT/cohort terms favor `in1`) or a batch-split (the
   `b^1.4` penalty forbids splitting the attention head dim). The device
   prefers them on wide-K and small-M shapes.

2. **Prefill Q/O, K/V, MLP-down: cost model is device-optimal.** Validated.

3. **Prefill MLP-up: 14.7% K-split miss — the headline.** Above noise, and
   prefill (where standalone == e2e is proven), so e2e-actionable *if* the
   fused MLP-up kernel can take a K-split.

4. **attn@V: device confirms pure-M > (16,2)** (4.803 < 5.123) — the prefill
   360-vs-408 regression direction, finally seen at the kernel level. The true
   optimum is a batch-split (4.521), which the cost model cannot reach.

## Critical caveat (post-lowering split equality)

Standalone split == e2e split is **proven for prefill (4/4)** but **fails for
decode (3/4)** — standalone the planner takes K-splits the fused e2e pins to
`in1` (span-reduction commits K to the RMSNorm reduction). So:

- **Prefill** rows above are e2e-representative.
- **Decode** rows are the *unconstrained* optimum, **not** e2e-feasible until a
  **fused-micro probe** confirms the decode fusion can take the split.

Every K-split / batch-split win above is gated on the same question: *can the
fused kernel use it, or does span-reduction pin `in1`/`x1`?* That probe is the
next step before any cost-model change.
