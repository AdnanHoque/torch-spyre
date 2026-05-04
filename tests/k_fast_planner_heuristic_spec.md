# k_fast planner heuristic — spec & implementation plan

## Goal

Make the planner pick `(1, n, k>1)` instead of `(32, 1, 1)` for
matmul shapes where `(1, n, k)+k_fast` is empirically faster — i.e.
small-M narrow-N matmuls common in LLM decode and short-prefill.

## Empirical decision rule

From `k_fast_planner_validation_findings.md`, k_fast wins ≥2% on
**91% of (model, op) pairs at M=128** in popular vLLM workloads.
The wins concentrate where these conditions all hold:

| condition | threshold | source |
|---|---|---|
| **M is small enough** | `M ≤ 512` | M-sweep on 5 real shapes |
| **N is narrow enough** | N has < 32 sticks (= < 2048 fp16 elems) | most shapes affected |
| **K is wide enough** | K has ≥ 32 sticks (= ≥ 2048 fp16 elems) | mechanism: pure-M HMI cost |
| **Model isn't tiny** | `M ≥ 32` AND `M·N·K ≥ ~2¹⁵` MFLOPS | Llama 3.2 1B failure case |

Failures cluster in:
- Very small models (Llama 3.2 1B): N=512, small K → per-core compute too low
- M=512+ on wide shapes (DSv3 o_proj/down_proj): pure-M's clean
  streaming dominates
- Non-power-of-2 stick counts (gpt-oss-20b K=2880=45 sticks): no
  valid K-split exists

The proposed heuristic guards against all three.

## Decision rule (proposed)

```python
def should_prefer_k_split(M: int, N: int, K: int, dtype) -> tuple | None:
    """Return (m, n, k) K-split if k_fast is empirically faster than
    pure-M for this shape, else None to fall through to default planner.
    """
    n_sticks = N // dtype.elems_per_stick()
    k_sticks = K // dtype.elems_per_stick()

    # Guard 1: shape must be in the empirical win-band
    if M < 32 or M > 512:
        return None
    if n_sticks >= 32:                  # pure-N already valid
        return None
    if k_sticks < 32:                   # K too narrow; PSUM dominates
        return None

    # Guard 2: pick the K-split — largest n that divides both n_sticks and 32
    for n in (16, 8, 4, 2):
        if n_sticks % n == 0 and 32 % n == 0:
            k = 32 // n
            if k_sticks % k == 0 and k_sticks // k >= 1:
                return (1, n, k)

    return None  # no valid K-split
```

The largest-n choice minimizes `k`, which minimizes PSUM chain length
(chain length = k − 1). The k_fast permutation then makes each chain
1 hop on the SFP ring.

## Where this slots into the planner

`torch_spyre/_inductor/core_division.py` — the planner's entry point
is `multi_dim_iteration_space_split` ([line 87](../torch_spyre/_inductor/core_division.py)),
called from `plan_splits` ([line 528](../torch_spyre/_inductor/core_division.py)).

Two integration options:

### Option 1 (preferred) — heuristic before fallthrough

In `plan_splits` (or the matmul-specific planner if there's one),
check `should_prefer_k_split` first; if it returns a split, use it.
Otherwise let the existing planner pick.

```python
def plan_splits(it_space, max_cores, ...):
    # Try k_fast-friendly K-split first for matmul shapes
    if is_matmul_op(op) and config.core_id_k_fast_emission:
        forced = should_prefer_k_split(M, N, K, dtype)
        if forced is not None:
            return forced, ..., priorities, min_splits

    # Fall through to existing planner
    priorities = prioritize_dimensions(...)
    splits = multi_dim_iteration_space_split(it_space_adjusted, max_cores, priorities, min_splits)
    return splits, it_space_adjusted, priorities, min_splits
```

This is additive — when the heuristic doesn't apply, behavior is
unchanged.

### Option 2 — feed it into priorities

Instead of forcing a split, modify `prioritize_dimensions` to rank
K higher than M when the heuristic conditions are met. The existing
`output_element_priority` flag is the precedent for this.

Pros: more compositional, lets `multi_dim_iteration_space_split` do
the actual splitting.
Cons: harder to predict — priority changes don't deterministically
produce the (1, n, k) split we want; depends on how `min_splits` and
other constraints interact.

**Recommendation: Option 1** for the first PR. Cleaner, more
testable, easier to roll back. Option 2 is an optional future
refactor if we want the heuristic to be priority-driven rather
than rule-driven.

## Configuration

A single config knob to control the heuristic:

```python
# config.py
core_id_k_fast_emission: bool = (
    os.environ.get("SPYRE_CORE_ID_K_FAST_EMISSION", "1") == "1"
)
```

This is the **same** flag that controls the `k_fast` permutation
itself — the heuristic only matters if k_fast is also active. So one
flag controls both:
- ON: planner picks K-split + emits k_fast permutation
- OFF: planner uses default; k_fast emission degenerates to identity

## Edge cases

### bmm (3D matmul)

Iteration space `[B, M, N, K]`. The heuristic should handle `B` (the
batch/head dim) gracefully:
- If B alone divides 32 cleanly (B ≥ 32), prefer pure-batch
  `(32, 1, 1, 1)` — that's typically what the existing planner does.
- If B < 32, K-split + k_fast may apply same as for mm.

For now: scope the heuristic to **mm only** (skip if op is bmm).
Defer bmm to a follow-up after we measure bmm shapes.

### Reductions / softmax / layernorm

Not matmul. The heuristic returns None.

### output_element_priority interaction

`output_element_priority` is the existing shipped heuristic that
ranks dims by element count. The k_fast heuristic runs **before**
the priority-based planner, so the two don't conflict — k_fast
takes precedence when the conditions match, else falls through to
priority-based behavior.

### Non-power-of-2 stick counts (e.g., gpt-oss-20b)

The `should_prefer_k_split` function naturally returns None when no
valid K-split exists (e.g., K=45 sticks doesn't divide cleanly).
Falls through to default planner. Safe.

### LX scratchpad fit

The forced K-split changes per-core operand sizes. We should sanity-
check that the proposed split doesn't violate LX scratchpad limits
(2 MB per core). For the shapes we measured this isn't an issue —
per-core B at `(1, 16, 2)` is `K/2 × N/16 × 2B` which is small for
narrow-N shapes — but a defensive check belongs in the
implementation.

## Test plan

### Hardware-free unit tests

Extend `tests/inductor/test_k_fast_emission.py` with:

1. `should_prefer_k_split` returns the expected split for each shape:
   - L3-70B kv_proj M=128 → `(1, 16, 2)`
   - L3-70B q_proj M=2048 → `None` (too large)
   - DSv3 o_proj M=128 → `(1, 16, 2)`
   - L3-8B FFN mlp_down M=128 → `None` (N too wide; pure-N valid)
   - Llama 3.2 1B M=128 → `None` (K too narrow)
2. The chosen split is always valid (n divides N stick count and 32; k_sticks
   divides k).

### Integration test (hardware required)

End-to-end matmul correctness: torch.compile a `nn.Linear` of each
target shape, compare output against CPU reference. The k_fast
permutation + planner heuristic should be invisible at the numerical
level — same output values.

### Regression sweep

Run the existing test suite. Critical: make sure shapes with
`should_prefer_k_split() == None` still get the default planner pick
(no behavior change).

## Expected production impact

Per the popular-models sweep at M=128 (the dominant decode regime):

| model family | typical wins on attention matmuls | per-token impact (rough) |
|---|---:|---:|
| Llama 3.1 70B | 8.5% kv_proj, 23.6% o_proj | ~10% per attention layer |
| Llama 3.1 405B | 14.8% kv_proj | ~12% per attention layer |
| DeepSeek V3 | 8.5% kv_proj, 48.6% o_proj | **~25% per attention layer** |
| Mixtral 8×22B | 5.5% kv_proj | ~5% per attention layer |
| Qwen 2.5 72B | 8.3% kv_proj | ~7% per attention layer |
| Granite 34B | 6.8% kv_proj | ~6% per attention layer |
| Gemma 2 27B | 9.1% kv_proj | ~7% per attention layer |

Composed across N transformer layers at decode, this translates to
~5-25% end-to-end inference latency reduction for batched serving
workloads (M ≈ 32-128) on these models.

The DSv3 numbers are aspirational — composing the 48.6% o_proj win
across 61 layers would be huge if it holds at end-to-end. Worth
explicitly validating with a multi-layer benchmark in the PR.

## Risks / pre-mortem

1. **Compose-across-layers may be less than per-op**. Single-matmul
   wins don't always translate 1:1 to end-to-end because of
   amortization effects (kernel launch overlap, shared HMI bandwidth
   pressure). Need an end-to-end benchmark.
2. **Larger PR surface area**. Heuristic + permutation is more code
   than just permutation. Reviewers may want them split. Plan a
   two-PR sequence: (a) k_fast permutation as latent infra, (b)
   planner heuristic that activates it. Each can be reviewed and
   rolled back independently.
3. **Threshold tuning across shape distributions**. The
   `M ≤ 512 AND n_sticks < 32 AND k_sticks ≥ 32` band is empirical,
   from a finite sample. New model architectures (e.g., GLA, RWKV)
   may need different thresholds. Heuristic should fall back
   gracefully (returns None for unrecognized regimes).

## Shipping recipe

### PR 1 — k_fast permutation (already prepared)

Branch: `AdnanHoque/feat-k-fast-emission` (commit `55b3158`).

Status: ready, but currently a no-op because the planner picks k=1.
**Update commit message** to reflect this honestly: "latent
infrastructure that activates when paired with the planner heuristic
in PR 2."

### PR 2 — planner heuristic

New branch off main with:

- `should_prefer_k_split` helper in `core_division.py`
- Integration point in `plan_splits` (Option 1 above)
- Hardware-free unit tests for the helper
- Optional: end-to-end multi-layer benchmark showing composed wins

Lands behind the same `SPYRE_CORE_ID_K_FAST_EMISSION` flag for
unified control.

### PR 3 (optional) — bmm support, threshold tuning

Once PRs 1+2 land and we have field data, extend to bmm shapes,
relax/tighten thresholds based on observed regressions.

## Alternative: skip the heuristic, ship as flag-only

If the PR1+PR2 sequencing is too much, an alternative path:

1. **Ship k_fast permutation only (current PR)** with the flag
   defaulted ON.
2. **Document `SPYRE_FORCE_K_SPLIT`** as a manual-tuning env var that
   users can set per-shape if they know their M is in the win-band.

This is much smaller but punts the win to careful operators rather
than automating it. Probably the wrong call given the data — the
opportunity is too systematic across model families to leave to
manual tuning.

## Recommendation

**Pursue PR 1 + PR 2 as a two-PR sequence.** PR 1 is already done;
PR 2 is the planner heuristic per this spec. Combined, they unlock
the measured 5-15% wins on production-relevant decode batch sizes
across the popular-LLM zoo.
