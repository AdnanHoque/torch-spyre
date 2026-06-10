# Decode regression investigation: cost-model PR #2407 vs the singleton 2-PR sequence

Device A/B isolating which merge moved Granite matmul time between the
June&nbsp;4 and June&nbsp;9 e2e traces (prefill matmuls 494→348&nbsp;ms,
decode matmuls 259→341&nbsp;ms).

## Trace breakdown (40-layer Granite e2e, decode)

The +82&nbsp;ms decode regression concentrates in four kernels:

| Kernel family | per-call | share |
|---|---|---:|
| `fused_linear_mul_rms_norm_silu_4` | 2.54 → 3.90 ms (+54%) | ~62% |
| small `sdpa_fused_attention_override` | 497 → 900 us (+81%) | ~20% |
| large `sdpa_fused_attention_override` | 3.12 → 3.36 ms (+7%) | ~11% |
| `linear_overwrite_slice_transpose_view_1` (KV cache) | 64 → 170 us (2.6×) | ~5% |

## A/B setup

- FMS (eager_spyre) Granite 8b, 1 layer, random weights; MLP sub-block
  `ff_sub_layer(ff_ln(x))·0.22 + x` — compiles to the same fused
  `linear_mul_rms_norm_silu` kernel that dominates the regression.
- Four stack points, Python-only flips on two fixed `_C.so` builds, single
  AIU, median of 5 wall-clock runs.
- Prefill `[1,512,4096]` / `[4,64,4096]`; decode `[1,1,4096]` / `[4,1,4096]`.

## Results (median ms)

| Stack point | prefill bs1×512 | decode bs1 | prefill bs4×64 | decode bs4 |
|---|---:|---:|---:|---:|
| pre-#2407 (937aa1c) | 19.64 | 8.72 | 13.99 | 8.28 |
| post-#2407 (20978b92) | **14.43** | 8.70 | 13.94 | 7.99 |
| pre-singleton (f520b5e) | 13.46 | 8.74 | 13.59 | 7.99 |
| post-singleton (9035fb8) | 13.08 | 8.72 | 13.57 | 7.97 |

## Conclusions

1. **Cost model #2407: prefill 1.36× faster, decode flat.** Matches the
   1.42× prefill improvement in the e2e traces; no decode harm.
2. **Singleton sequence (#2550 + #2588): inert.** All numbers flat (≤3%).
   Consistent with the mechanism analysis: `mark_direct_unit_bmm_pass` only
   fires on standalone `aten.bmm`, and real Granite fuses every MLP matmul
   into `linear_mul_rms_norm_silu` kernels (6/6 prefill kernels byte-identical
   SuperDSC with marking on/off). The singleton PRs caused neither the decode
   regression nor the prefill speedup.
3. **Remaining prefill gain (14.4 → 13.5 ms) landed between #2407 and #2533**,
   i.e. the LX-scratchpad era.

## Decode-regression suspects (signature: prefill faster, decode slower)

- **#2459** — `LX_PLANNING` default 0→1 (Jun 5).
- **#2533** — added `mul/mean/add/rsqrt` (the RMSNorm/SiLU ops) to LX
  reuse/inplace lists (Jun 8). LX pinning amortizes over 512 prefill rows but
  adds per-call clone/copy at M=1 decode.
- **#2480** — restickify legalized under-filled sticks (dims < 64) and
  layout-committed mutation ops; matches the 2.6× KV-cache
  `overwrite_slice` kernel.
- Coarse tiling (#2497/#2572) is inert for Granite: it only fires inside
  `spyre_hint()` scopes, which Granite never uses.

Cheapest next test: e2e decode with `LX_PLANNING=0`.

## Caveats

- Wall-clock decode (~8.7 ms) is host-dominated; a ms-scale device-side decode
  delta sits below this probe's resolution. Decode confirmation needs
  device-event profiling.
- The fused SDPA SuperDSC needs latest deeptools to chunk; on older SDKs only
  the MLP sub-block compiles (≈62% of the regression).

---

## Update: full-block A/B (latest deeptools) — #2550 cost hunk IS implicated in decode

Rebuilding deeptools at latest master made the fused SDPA SuperDSC compile, so
the full Granite block (attention + KV cache, padded decode_multiple=64) ran
at all four points with per-kernel `ideal_cycles.json` + SuperDSC artifacts.

**The padded decode block is M=64, not M=1.** The earlier MLP-only A/B used
M=1 decode and was flat — that probe missed the regression.

In the e2e graph, the singleton window flips decode-block bmm splits:

| bmm (decode block, M=64) | f520b5e | 9035fb8 |
|---|---|---|
| attention out-proj `2_bmm` | (m4, n8) | **(m32, n1) pure-M** |
| attention `14_bmm` | (m4, n8) | **(m32, n1) pure-M** |

The unit-bmm marking is inert (proven on real fused kernels). The flip comes
from the `target_m_us` cost-model hunk in #2550, which fires on every matmul:
at M=64, `pt_passes = max(1, m_t/PT_ROWS) = 1` regardless of m-split, so the
pure-M candidate's penalty matches the m×n candidate and the tie now breaks
to pure-M — 2 rows/core on the 64-row PT array, weight reloaded everywhere.

**Device confirmation, standalone `[64,4096]·[4096,4096]` matmul:**

| stack | split | median |
|---|---|---:|
| f520b5e (pre) | (m4, n8) | 0.631 ms |
| 9035fb8 (post) | **(m32, n1)** | **1.260 ms (2.0×)** |

This matches the decode regression signature: every padded-decode matmul runs
at M=64, and Antoni's e2e decode matmuls are ~1.3× slower (only some kernels
flip → smaller aggregate factor).

**Fix direction:** in `_matmul_split_cost`, the `target_m_us` term must not
reward higher m-splits when `m_t ≤ PT_ROWS` (M ≤ 64 → m-split beyond 1 adds
zero PT throughput, costs N-parallelism); the prefill speedups don't depend
on that regime.

LX-pass commits (#2459/#2533) remain suspects for the residual diffs (e2e
LX_PLANNING=0 ablation still worthwhile).
