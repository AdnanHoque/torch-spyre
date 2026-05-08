# M4 falsification probe — findings

Companion to `diag_m4_peak_lx_falsification.py`. Static analysis of
peak per-core LX occupancy on transformer-block ops under two
residency policies.

## TL;DR

**M4 is conditionally viable, regime-dependent.**

| regime | current OP_OUTPUT_GOOD_FOR_LX_REUSE | hypothetical expanded (matmul outputs pinned) |
|---|---|---|
| decode (M ≤ 128) | peak LX = 8-128 KB (0-8% of cap) — **no target** | peak LX = 56-832 KB (3-51%) — **no target** |
| medium (M = 512) | peak LX = 128-512 KB (8-31%) — **no target** | peak LX = 0.9-3.25 MB (55-203%) — **target on largest models** |
| prefill (M = 2048) | peak LX = 0.5-2.0 MB (31-125%) — **borderline on L3-405B** | peak LX = 3.5-13 MB (219-813%) — **target on all models** |

The reordering search space depends on which regime the workload
runs in:

- At **decode**, peak LX is well under the 2 MB cap regardless of
  policy. Activations are too small to fill the scratchpad cross-op.
  M4 has nothing to do.
- At **prefill on L3-405B (current behaviour)**, the softmax output
  alone is 2.00 MB per core — exactly at the hard cap. This single
  case may already be hitting LX eviction. M4 reordering wouldn't
  help (only one tensor is resident; nothing to reorder).
- At **prefill with hypothetical expanded matmul-residency**, peak
  LX exceeds the cap by 2-8× on every model. Reordering becomes
  the critical lever — but only AFTER a separate project expands
  the eligible-op set.

## Per-(model, M) data

Pure-M split (32, 1, 1). Sizes are per-core bytes for tensors that
would be LX-resident under each policy.

| model | M | A peak | B peak | A % | B % |
|---|---:|---:|---:|---:|---:|
| Llama 8B   | 32   | 8 KB    | 56 KB   | 0%   | 3% |
| Llama 8B   | 128  | 32 KB   | 224 KB  | 2%   | 14% |
| Llama 8B   | 512  | 128 KB  | 896 KB  | 8%   | 55% |
| Llama 8B   | 2048 | 512 KB  | 3.50 MB | 31%  | **219%** |
| Llama 70B  | 32   | 16 KB   | 112 KB  | 1%   | 7% |
| Llama 70B  | 128  | 64 KB   | 448 KB  | 4%   | 27% |
| Llama 70B  | 512  | 256 KB  | 1.75 MB | 16%  | **109%** |
| Llama 70B  | 2048 | 1.00 MB | 7.00 MB | 63%  | **438%** |
| Llama 405B | 32   | 32 KB   | 208 KB  | 2%   | 13% |
| Llama 405B | 128  | 128 KB  | 832 KB  | 8%   | 51% |
| Llama 405B | 512  | 512 KB  | 3.25 MB | 31%  | **203%** |
| Llama 405B | 2048 | **2.00 MB** | 13.00 MB | **125%** | **813%** |
| Mixtral    | 32   | 8 KB    | 56 KB   | 0%   | 3% |
| Mixtral    | 128  | 32 KB   | 224 KB  | 2%   | 14% |
| Mixtral    | 512  | 128 KB  | 896 KB  | 8%   | 55% |
| Mixtral    | 2048 | 512 KB  | 3.50 MB | 31%  | **219%** |
| DeepSeek V3 | 32   | 14 KB   | 72 KB   | 1%   | 4% |
| DeepSeek V3 | 128  | 56 KB   | 288 KB  | 3%   | 18% |
| DeepSeek V3 | 512  | 224 KB  | 1.12 MB | 14%  | 70% |
| DeepSeek V3 | 2048 | 896 KB  | 4.50 MB | 55%  | **281%** |

(% is of 1.6 MB usable per-core LX after the 0.2 backend reserve.)

## What's load-bearing in this analysis

**Model A reflects today's torch_spyre.** Only `max`, `sum`, `clone`
output ops get pinned by `ScratchPadAllocator`. In a transformer
block, the only relevant ops are inside softmax (its max/sum
reductions) and explicit clones. The probe approximates this as
"the softmax output is LX-resident" — which actually OVER-estimates
peak (the actual max/sum reductions are M × n_heads scalars, much
smaller).

So the Model A column is a generous upper bound. Real Model A peak
is smaller. The L3-405B M=2048 case at 2 MB is the closest to
binding under any plausible interpretation.

**Model B is what M4 would unlock.** If matmul outputs were also
LX-pinned, peak LX would balloon. The peak op is consistently
`up_proj` (the second of two parallel MLP projections producing
`M × intermediate` outputs).

## Two findings worth flagging independently

### Finding 1 — Current LX residency is essentially never at the cap

At decode and medium M, peak LX is <512 KB on every model. The 2 MB
cap is enormous relative to current cross-op residency. Whatever
LX_PLANNING is currently doing, it's operating with substantial
headroom.

The exception is **L3-405B M=2048 softmax**: the output of
attention-compute alone is 2 MB per core. This single case may
already be triggering LX eviction in production at prefill on the
largest model. Worth a separate quick measurement (e.g., compile
that op with `LX_PLANNING=1` and check the allocator's
`lx_usage_hist`) to see if the allocator is actually placing this
in LX or refusing.

### Finding 2 — M4 is gated on a separate project

M4's reordering only matters when peak LX *under default ordering*
already exceeds the cap. From this probe:

- Under current (Model A) behaviour, that's true on at most 1 cell
  out of 20 (L3-405B M=2048). Even there, only one tensor is
  resident at peak — nothing for reordering to swap.
- Under expanded (Model B) behaviour, peak LX exceeds cap on
  every M=2048 row across every model (4 of 5 models, all M=2048).
  But Model B requires expanding `OP_OUTPUT_GOOD_FOR_LX_REUSE` to
  include matmul outputs — that's a separate project (the LX
  residency planner already committed to in the broader portfolio).

So M4 sits *behind* the residency-expansion project. Doing M4
without that expansion produces no measurable benefit. Doing both
in sequence: first expand the eligible-op set, then layer the
FPT-pebbling reordering pass on top.

## Verdict

**M4 closes for now.** Rationale:

1. **Decode regime never approaches the cap.** The bulk of production
   token-generation traffic is at M ≤ 128. At that regime, M4 has
   no headroom to recover, with or without expanded LX residency.
2. **Prefill regime under current behaviour rarely approaches the
   cap.** Only 1 cell hits the cap (L3-405B M=2048, single tensor),
   and there's no reordering improvement available there.
3. **Prefill regime under expanded behaviour DOES exceed the cap**,
   but expanding LX residency is a separate, larger project. Until
   it ships, M4 has nothing to do.

If the residency-expansion project happens, **revisit M4**. The
expanded-behaviour data here would translate directly into
M4's reordering search space, and pebbling theory suggests
substantial peak reduction is achievable.

## Recommended sequencing

Don't build M4 next. Instead:

1. **Quick probe of the L3-405B M=2048 softmax case** to confirm
   whether LX eviction is actually happening today (or whether the
   allocator is refusing to pin and the tensor is going to HMI).
   This would either confirm or close the "current behaviour
   borderline" finding.
2. **If the residency-expansion project is on the table**, use the
   Model B numbers from this probe as its motivation. They show a
   real binding constraint that reordering could unlock.
3. **Park M4 itself** until step 2 makes it relevant.

## Caveats and limitations

- **Pure-M split assumed.** Under (m, n, k) splits with m < 32,
  per-core tensor sizes increase, which would push peak LX higher.
  Under (1, 16, 2) the M_per stays full but n=16 cores share each
  output cell — peak might double or worse. Worth re-running this
  probe with the splits the planner actually picks.
- **Liveness model is approximate.** Each tensor is assumed consumed
  by the very next op only. Residual connections in transformer
  blocks create longer liveness intervals (input_rmsnorm output
  lives until post_attn_residual). The probe under-estimates peak
  by missing these. Real peak could be ~2× the numbers reported.
- **Static analysis doesn't model fragmentation.** The
  `ScratchPadAllocator` has limited defragmentation logic; effective
  capacity could be lower than the 1.6 MB usable.
- **Softmax handling is approximate.** Probe treats softmax output
  as LX-resident, which OVER-estimates Model A peak. The actual
  pinned tensors under current behaviour are the internal max/sum
  reductions — much smaller.

The caveats mean Model A peak might be 0.5-2× the reported numbers
in either direction. But they don't change the headline conclusion:
peak LX is well under the cap at decode and at most borderline at
prefill, under current behaviour.

## Files

- `tests/diag_m4_peak_lx_falsification.py` — probe
- `tests/diag_m4_peak_lx_falsification_results.txt` — raw output
- This doc

## Branch

`AdnanHoque/m4-peak-lx-falsification` — cut from main, self-contained
(does not depend on the cost-model V4 / emission-aware-lx work).
