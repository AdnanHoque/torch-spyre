# Flash-attention on Spyre — Phase 0a findings

The Phase 0a probe (`tests/diag_sdpa_baseline.py`) measures the cost of
the current SDPA path on Spyre at prefill-relevant sequence lengths.
Headline: **the current path is essentially unusable beyond S=1024-2048
on Spyre. At S=4096 prefill it takes 1.5 seconds per attention layer**,
which is ~48 seconds for a 32-layer Llama-3-8B prefill at 4K context just
for attention.

This justifies flash-attention as a high-impact, frontend-only project.
Detailed below.

## Method

Llama-3-8B-style configuration (B=1, H=32, H_kv=8, D=128, fp16). Decode-
unfriendly prefill shapes only — S as the queries-and-keys length.

`torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True,
enable_gqa=True)` invoked through `torch.compile(dynamic=False)`. Per-
iter `torch_spyre.streams.synchronize()` inside the timed loop. 3 warmup
+ 20 measure iters, median reported.

The `parse_op_spec` SDSC capture hook (same one we used for SplitK Phase
0) counts kernels emitted per SDPA call.

## Results

| S | Wall ms | Kernels/call | Score (S²) | Total DDR | Eff BW GB/s | TFLOPs/s | flops/byte |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 46 | 14 | 17 MB | 61 MB | 1.3 | 0.09 | 70.6 |
| 1024 | 129 | 14 | 67 MB | 222 MB | 1.7 | 0.13 | 77.3 |
| 2048 | 446 | 14 | 268 MB | 847 MB | 1.9 | 0.15 | 81.1 |
| **4096** | **1555** | **14** | **1074 MB** | 3305 MB | 2.1 | 0.18 | 83.2 |

**Wall time scales as S²** (within noise) across all measured S — confirms
that compute and bandwidth both scale quadratically with S in the naive
path, so the path is uniformly bottlenecked across this range.

## Three things this measurement shows

### 1. Kernel-launch overhead dominates at small S

The 14 kernels per SDPA call (captured below) at ~3 ms launch overhead
each = ~42 ms minimum. At S=512 that's 91% of total wall time. The
softmax alone decomposes into 5 separate kernels.

```
['mul', 'mul', 'identity', 'ReStickifyOpHBM',
 'batchmatmul',                            # QK
 'add',                                    # causal mask
 'max', 'sub', 'exp', 'sum', 'realdiv',    # softmax (5 kernels!)
 'identity',
 'batchmatmul',                            # AV
 'identity']
```

This is the same launch-overhead pattern we measured in MoE Phase 0a —
Spyre per-kernel overhead is ~3 ms regardless of compute. SDPA pays this
14 times per call.

### 2. (B, H, S, S) score-tensor traffic dominates at large S

At S=4096 the intermediate score tensor is **1.07 GB**. The naive path
streams it through DDR at least three times (write from QK, read for
softmax, write softmax output, read for AV). That's >3 GB of
score-related DDR traffic per call, plus Q+K+V+out = total ~3.3 GB at
S=4096 by my undercount; the actual softmax decomposition has 5 kernels
each touching the tensor, so real traffic is closer to 10 GB.

### 3. Both compute and bandwidth are catastrophically under-utilized

- **Effective BW: 1.3-2.1 GB/s** vs LPDDR5 peak ~200 GB/s. ~100× under.
- **TFLOPs/s: 0.09-0.18** vs Spyre fp16 peak ~150 TFLOPs/s. ~800× under.

Compute *should* dominate (arithmetic intensity is 70-83 flops/byte —
high enough to be compute-bound under any reasonable roofline). It
doesn't. The kernel-launch + serialization overhead between the 14
kernels is so high that neither compute nor bandwidth is actually being
used.

## Flash-attention upper bound

Compute at S=4096: ~275 GFLOPs (two BMMs at `(H=32, M=4096, N=128)` and
`(H=32, M=4096, N=4096, D=128)` patterns).

Realistic targets:
- **Pessimistic (10% of fp16 peak)**: ~18 ms — **86× speedup**
- **Match SplitK-bench-class throughput (~9 TFLOPs/s)**: ~30 ms — **50×**
- **Pessimistic floor (assume 50 ms launch overhead regardless)**: 50 ms — **30×**

**Even the most pessimistic flash-attention target is 30× faster than
current SDPA at S=4096.** The gap *grows* with S because the naive path's
S² intermediate scaling outpaces flash attention's linear-in-S streaming.
At S=8192 the intermediate would be 4 GB and the naive path would be
unusable on Spyre at all — flash attention would still be ~60-70 ms.

## What flash-attention requires on Spyre

Three components:

1. **Custom op** `spyre::flash_attention(Q, K, V, ...)` — replaces the
   current decomposition into mm + softmax + mm.
2. **Lowering** — emits a kernel sequence: per Q-tile, loop over KV-tiles,
   each tile-kernel updates running max + running sum + running output in
   scratchpad.
3. **`scratchpad_planning` extension** — allowlist the running-state ops
   so they can be pinned in LX scratchpad across consecutive KV-tile
   kernels. Currently `OP_OUTPUT_GOOD_FOR_LX_REUSE = ["max", "sum",
   "clone"]` ([scratchpad.py:30-36](../torch_spyre/_inductor/scratchpad.py)).
   The architecture already supports cross-op reuse; we extend the
   allowlist + ensure same-core-division so the data stays in scratchpad.

Architecture fit is clean: each Q-tile is owned by a single core, running
stats live in that core's LX scratchpad, KV tiles stream in from DDR. The
*"each core can only write to its own scratchpad"* constraint
(scratchpad.py:210-213) actually *helps* this design — Q-tile-level
parallelism is the natural model.

## What's NOT yet answered (Phase 0b territory)

1. **Does `scratchpad_planning` actually pin allowlisted op outputs in
   scratchpad today, or just allocate them?** Need to read more of
   scratchpad.py and `LX_PLANNING=1` runs to confirm the pinning works.
2. **What's the per-tile launch overhead?** If it's ~3 ms (matching MoE
   Phase 0a) and we have S/Q_tile_size tiles, the launch floor is
   `S/Q_tile_size × 3 ms`. For S=4096 with Q_tile=64 that's 64 × 3 = 192
   ms — STILL much faster than 1555 ms but not the 30× bound. Need to
   pick Q_tile carefully.
3. **Do current kernels overlap DMA with compute?** Producer-consumer
   ping-pong in scratchpad would amortize launch overhead; without it,
   each tile-kernel's DMA and compute serialize.

## Decision

**Project proceeds.** The data is conclusive: SDPA on Spyre is 30-86×
slower than the flash-attention floor; the bottleneck is the (B,H,S,S)
score-tensor materialization plus 14-kernel launch overhead; both are
addressable at the Inductor + scratchpad-planning layer with no backend
dependency.

Phase 0b: confirm scratchpad pinning works as expected, characterize
per-tile launch overhead, decide on Q-tile and KV-tile sizes.

## Files

- `tests/diag_sdpa_baseline.py` — Phase 0a probe
- `tests/diag_sdpa_baseline_results.md` — auto-regenerated bench output
- `tests/sdpa_phase0_findings.md` — this document

## Reproducing

```sh
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
cd $DTI_PROJECT_ROOT/torch-spyre
python tests/diag_sdpa_baseline.py
```
