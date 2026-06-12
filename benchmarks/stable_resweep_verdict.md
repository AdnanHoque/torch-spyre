# Stable-stack re-sweep — the trustworthy verdict

**This supersedes [`exhaustive_split_sweep.md`](exhaustive_split_sweep.md)**, which
ran on the ~20% noise clc/flex-tuan environment and whose "device-best" gaps were
measurement artifacts. This re-sweep ran on a **stable device** (~5% noise, no
wedges) and every flagged gap was **repeat-confirmed** before counting.

## The stable measurement stack

The clc fresh-card firmware wedge (all freshly-allocated PF cards come up
DMA-wedged; only ba:00.0, allocated weeks ago, works) was sidestepped by
**harvesting** a newer pod's `/opt/ibm/spyre` onto the shared NFS
(`/home/adnan/opt-newer`: senlib +148, deeptools +932, flex +292, comms — all
4-arg ABI, no shim). Built `_C.so` + flex PR #1019 (`fix-profiling-in-streams`,
lost-CB fix) against it; runs on ba:00.0's firmware. Result: same-split variance
dropped **20% → ~5%**, and the probes that wedged on clc now complete clean.

## Method

240 forced 32-core splits × 12 golden shapes, then **multi-pass repeat-confirm**
of every gap >5%. A gap counts only if the two splits' run distributions are
**non-overlapping** across passes (the test the clc single-pass sweep lacked).

## Verdict: the cost model is device-optimal except ONE shape

| shape | role | single-pass flag | repeat-confirm | verdict |
|---|---|---|---|---|
| 512×4096×4096 | Q/O prefill | 0% | — | **optimal** |
| 512×12800×4096 | MLP-up prefill | 0% | — | **optimal** |
| 512×1024×4096 | K/V prefill | 7.1% | cost-pick 7.5% faster | noise |
| 512×4096×12800 | MLP-down prefill | 6.9% | tied, overlap | noise |
| 64×* (all proj/MLP) | decode | 1.7–6.8% | tied/overlap | noise |
| 32×64×128×576 | attn@V decode | 17.7% | −0.8%, overlap | noise |
| **32×512×128×512** | **attn@V prefill** | 10.7% | **+11.0%, clean** | **REAL** |

**The one confirmed cost-model miss: prefill attn@V wants a batch-split** —
`(b4,m4,n2)` = 6.23 ms vs the cost model's pure-M 7.0 ms, **+11%, non-overlapping
over 4 passes**. Splitting the 32 attention heads across cores beats pure-M; the
cost model's `b^1.4` `_BATCH_SPLIT_EXPONENT` penalty structurally forbids it. The
penalty is right for independent-batch GEMMs (tile in time) but wrong for
attention's heads, which *are* the natural parallelism.

## Sharpens the prefill 360-vs-408 story

Standalone, **pure-M ≈ (16,2) are tied** (both 7.0 ms). So Antoni's e2e gap
between them is **not** the raw matmul — it's fusion-context (restickify/layout),
confirming Codex's post-lowering point. The real lever is the batch-split (11%),
which the e2e currently can't reach (it emits `x1`).

## The decisive open question (gates actionability)

Is the e2e attention kernel's `x1` the **cost model's choice** (→ a scoped
`batch_penalty` fix captures the 11%) or **fusion-pinned** (→ needs a lowering
change to allow head-splitting first)? Tested by forcing/allowing a batch-split
in the fused attention compile and checking whether the attn@V bmm flips to `x4`.

## ANSWER: x1 is the cost model's choice, not fusion-pinned — the 11% is capturable

Recompiled the 1-layer prefill with `_BATCH_SPLIT_EXPONENT = 0` (batch penalty
off). The fused attention kernel's attn@V bmm **flipped `x1 → x2`** (`x2,mb8,out2`)
and compiled+ran clean. So the e2e attention kernel **can** take a batch-split;
`x1` was the `b^1.4` penalty's doing, not a lowering constraint.

**Conclusion: the confirmed +11% prefill-attn@V batch-split is capturable by a
scoped cost-model fix** — skip/soften the `_BATCH_SPLIT_EXPONENT` penalty in the
batched-attention regime (batch dim = head count, small N) so the planner splits
heads across cores. Global zeroing is too aggressive (wrongly batch-splits other
shapes); the fix must be gated to that regime. First fully-validated cost-model
lever: survived trustworthy device + repeat-confirm + e2e-feasibility.
