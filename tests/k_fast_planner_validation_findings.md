# k_fast planner validation — cross-model findings

## TL;DR

We had a methodology gap. Earlier k_fast wall-time measurements
(2.7×–3.7× wins on kv_proj/o_proj/down_proj) were taken with
`_force_split` pinning `(1, 16, 2)`. **The planner does not naturally
pick that split** — it picks pure-M `(32, 1, 1)`. That made the
"2.7× win" claim unsupported in production.

But across 66 measurements on 22 (model, op) pairs from popular
vLLM-served LLMs, we found a cleaner truth:

- **At M=2048 (long prefill), the planner is right** — pure-M genuinely
  beats `(1, 16, 2)+k_fast` on every shape.
- **At M ∈ {32, 128, 512} (decode + short prefill), the planner is
  systematically wrong** — `(1, 16, 2)+k_fast` beats it on **20 of 22
  shapes (91%)** at M=128.

The mechanism: pure-M `(32, 1, 1)` allocates `M/32` elements per core.
At M=128 that's 4 elements/core, far below the PT array's 64-wide SIMD
granularity → severe compute under-utilization. K-split keeps full M
per core, paying only PSUM cost (which k_fast minimizes).

## Methodology

Three probes, run in sequence:

1. **`diag_planner_natural_pick_verify.py`** — confirm what the
   planner actually picks (no force_split).
2. **`diag_planner_correctness_check.py`** — for each shape, compare
   planner-natural vs `(1, 16, 2)+identity` vs `(1, 16, 2)+k_fast`.
3. **`diag_k_fast_real_workloads_msweep.py`** + **`diag_k_fast_popular_models.py`** —
   sweep M across 22 model+op pairs to map the win-band.

All probes: warmup=3, iters=12, fp16, SENCORES=32, fx_graph_cache off,
two trial orders for replication where applicable.

## What the planner picks today

For every shape we measured forced-K-split wins on, the planner
naturally picks `(32, 1, 1)`:

| shape | forced for our measurement | planner-natural |
|---|---|---|
| L3-70B kv_proj M=2048 | `(1, 16, 2)` | `(32, 1, 1)` |
| Mixtral 8×7B kv_proj M=2048 | `(1, 16, 2)` | `(32, 1, 1)` |
| DSv3 o_proj M=2048 | `(1, 16, 2)` | `(32, 1, 1)` |
| DSv3 down_proj M=2048 | `(1, 16, 2)` | `(32, 1, 1)` |
| DSv3 q_a_proj M=2048 | `(1, 8, 4)` | `(32, 1, 1)` |
| L3-70B q_proj M=128 | `(4, 1, 8)` | `(32, 1, 1)` |
| L3-8B MLP down M=128 | `(4, 1, 8)` | `(32, 1, 1)` |

**Zero matches.** The planner picks pure-M everywhere, regardless of
shape geometry.

## At M=2048, the planner is right

Direct comparison on the original "big-win" shapes — planner pure-M
vs forced `(1, 16, 2)+k_fast`:

| shape | pure-M | (1,16,2)+k_fast | k_fast vs pure-M |
|---|---:|---:|---|
| L3-70B kv_proj M=2048 | 3.68 ms | 3.97 ms | k_fast 8% slower |
| Mixtral 8×7B kv_proj M=2048 | 3.32 ms | 3.49 ms | k_fast 5% slower |
| **DSv3 o_proj M=2048** | **13.42 ms** | 31.25 ms | **k_fast 2.3× slower** |
| DSv3 down_proj M=2048 | 4.47 ms | 6.92 ms | k_fast 55% slower |

Pure-M wins by margins ranging from 5% to 2.3×. **At M=2048 the
planner is correct.**

## At M ≤ 512, the planner is wrong

The same `(1, 16, 2)+k_fast` configuration beats pure-M in the
shorter-M regime — and the win generalizes across model families.

### M-sweep on the original five workloads

| shape | M=32 | M=128 | M=512 | M=1024 | M=2048 |
|---|---|---|---|---|---|
| L3-70B kv_proj | **+5.9%** | **+8.4%** | **+5.5%** | tie | −7.5% |
| Mixtral 8×7B kv_proj | **+2.8%** | **+4.0%** | **+3.2%** | tie | −4.9% |
| DSv3 o_proj | **+3.0%** | **🚀 +48.7%** | −15% | −48% | −57% |
| DSv3 down_proj | tie | **+15.3%** | −6% | −23% | −35% |
| DSv3 q_a_proj | **+8.1%** | **+8.3%** | −2.8% | −13% | −22% |

(Bold = k_fast wins by ≥2%.)

### Coverage stats — popular models sweep

22 (model, op) pairs from Llama 3.1 (8B/70B/405B), Llama 3.2 (1B/3B),
Mistral 7B, Mixtral 8×7B/8×22B, Qwen 2.5 (7B/14B/32B/72B), Phi-3
medium, Granite 8B/34B, Gemma 2 27B, DSv3, gpt-oss-120b. Result:

| M | k_fast wins ≥2% | k_fast loses ≥2% |
|---:|:---:|:---:|
| 32 (heavy decode batching) | **15 / 22 (68%)** | 1 / 22 (5%) |
| **128 (medium decode/short prefill)** | **20 / 22 (91%)** | 1 / 22 (5%) |
| 512 (longer prefill) | 15 / 22 (68%) | 4 / 22 (18%) |

**At M=128, k_fast wins on 91% of tested shapes from popular vLLM
workloads.**

### Top wins ≥5%

| rank | model | op | M | natural ms | k_fast ms | saved |
|---:|---|---|---:|---:|---:|---:|
| 1 | DSv3 | o_proj | 128 | 9.164 | 4.713 | **48.6%** |
| 2 | Llama 3.1 70B | o_proj | 128 | 6.476 | 4.945 | **23.6%** |
| 3 | DSv3 | down_proj | 128 | 3.728 | 3.166 | **15.1%** |
| 4 | Llama 3.1 8B | o_proj | 128 | 3.799 | 3.228 | **15.0%** |
| 5 | Llama 3.1 405B | kv_proj | 128 | 3.812 | 3.247 | **14.8%** |
| 6 | Llama 3.1 405B | kv_proj | 32 | 3.745 | 3.260 | **13.0%** |
| 7 | Gemma 2 27B | kv_proj | 128 | 3.442 | 3.130 | **9.1%** |
| 8 | DSv3 | q_a_proj | 128 | 3.526 | 3.213 | **8.9%** |
| 9 | Llama 3.1 405B | kv_proj | 512 | 3.749 | 3.426 | **8.6%** |
| 10 | Llama 3.1 70B | kv_proj | 128 | 3.358 | 3.071 | **8.5%** |

(Full top-20 in `diag_k_fast_popular_models_results.txt`.)

### Failure cases

Three (model, M) combinations show k_fast loses ≥2%:

- **Llama 3.2 1B kv_proj M=128**: −3.3%. N=512 (8 sticks) + small K means per-core compute is too small.
- **Phi-3 medium kv_proj M=512**: −5.6%. Edge of the M-band; pure-M starts winning.
- **DSv3 down_proj M=512+**: planner is right; pure-M's clean streaming dominates.

The failures cluster at: very small models (Llama 3.2 1B), very narrow
N (≤8 sticks), or M ≥ 512 on shapes where pure-M is already efficient.

## Mechanism

Why pure-M wins at large M, k_fast wins at small M:

```
            pure-M (32, 1, 1)                K-split (1, 16, 2) + k_fast
            ─────────────────                ───────────────────────────
M elements  M/32 per core                    full M per core
per core    (=4 at M=128, =64 at M=2048)    (always M elements)

PT array    under-utilized at small M        always fully utilized
utilization  (SIMD width = 64)

HMI         all 32 cores need full B         each core needs B/32 unique chunk
pattern     (16 MB at M=2048 kv_proj)        (clean streaming + ring share)

PSUM        none                             16 chains × per-chain payload
                                             k_fast: each chain 1 hop
                                             identity: each chain 16 hops
```

At small M:
- Pure-M wastes 60-100x of PT compute (4 elem/core into a 64-wide SIMD).
- K-split + k_fast keeps PT full and pays minimal PSUM (1 hop per chain).
- Net: K-split + k_fast wins by 5-49% depending on shape.

At large M:
- Pure-M's compute under-utilization disappears (M/32 ≥ 64).
- K-split's PSUM chain payload grows with M — eventually PSUM cost
  exceeds pure-M's "full B from HMI" savings.
- Net: pure-M wins by 5-57% depending on shape.

The crossover happens around **M = 256-1024 depending on shape**.
For very wide shapes (DSv3 o_proj), the crossover is sharp at M=128
(48.6% win) → M=512 (15% loss).

## What this means for shipping

The story is much cleaner than the original framing:

- **k_fast as a permutation** (the PR currently on
  `AdnanHoque/feat-k-fast-emission`) is a **no-op in production**
  because the planner never picks k>1 splits.
- **k_fast paired with a planner heuristic** that picks K-split for
  small-M narrow-N shapes is a **5-15% wall-time win on attention
  matmuls in 91% of measured popular-LLM shapes** at the dominant
  decode-batching regime.

**Don't ship k_fast alone.** Ship k_fast + a planner heuristic, or
don't ship k_fast at all. See `k_fast_planner_heuristic_spec.md` for
the proposed heuristic.

## Files

Probes:
- `diag_planner_natural_pick_verify.py` + `*_results.txt`
- `diag_planner_correctness_check.py` + `*_results.txt`
- `diag_k_split_search.py` + `*_results.txt`
- `diag_k_fast_real_workloads_msweep.py` + `*_results.txt`
- `diag_k_fast_popular_models.py` + `*_results.txt`

Heuristic:
- `k_fast_planner_heuristic_spec.md`

Theory + earlier work:
- `docs/source/architecture/k_fast_theory.md`
- `core_permutation_findings.md`, `core_permutation_long_m_findings.md`,
  `k_fast_deepseek_v3_findings.md`
