# Flash-attention on Spyre — Phase 1 design

This document specifies the design for a flash-attention implementation
on Spyre. It does not include implementation; the deliverable is the
design rationale, the API surface, the lowering strategy, and the open
questions that implementation will answer.

The design is grounded in the Phase 0 measurements:

- **0a** ([sdpa_phase0_findings.md](sdpa_phase0_findings.md)): naive SDPA
  is 1555 ms at S=4096, with 14 kernels per call and a 1 GB (B,H,S,S)
  intermediate. 30-86× speedup is achievable.
- **0b** ([sdpa_phase0b_findings.md](sdpa_phase0b_findings.md)): per-launch
  overhead is ~3 ms flat across all relevant work sizes. Tile-size choice
  is bounded by scratchpad/span, not launch cost.
- **0c** ([sdpa_phase0c_findings.md](sdpa_phase0c_findings.md)): LX
  scratchpad pinning is invisible at our shape. Flash attention's design
  does not depend on it — the win comes from eliminating the (S,S)
  intermediate, not from per-tile state pinning.

## High-level approach

Replace the current SDPA decomposition (`decompositions.py:494
spyre__sdpa_overrideable`) with a tile-streaming implementation: for each
Q-tile, loop over KV-tiles maintaining online-softmax running statistics
per Q-row. The full (S, S) score tensor is never materialized — only one
(Q_block, K_block) slab at a time exists, fitting in scratchpad.

Conceptually:

```
for q_start in range(0, S_q, Q_block):
    Q_tile = Q[..., q_start:q_start+Q_block, :]
    out_tile = zeros(Q_block, D)            # running output, fp32
    m = full(-inf)                          # running max, per Q-row, fp32
    l = zeros(Q_block)                      # running sum, per Q-row, fp32
    for k_start in range(0, S_k, K_block):
        K_tile = K[..., k_start:k_start+K_block, :]
        V_tile = V[..., k_start:k_start+K_block, :]
        S_block = (Q_tile @ K_tile.T) * scale     # one bmm kernel
        if causal: apply causal mask in-tile
        m_new = max(m, max(S_block, dim=-1))
        scale_old = exp(m - m_new)
        S_norm = exp(S_block - m_new)
        l = l * scale_old + sum(S_norm, dim=-1)
        out_tile = out_tile * scale_old + S_norm @ V_tile
        m = m_new
    out[..., q_start:q_start+Q_block, :] = out_tile / l
```

This is the standard FlashAttention-2 algorithm (Dao 2023), adapted to
Spyre's static-dataflow + per-launch-overhead constraints.

## API design

### Choice A: Custom op `spyre::flash_attention`

```python
@torch.library.custom_op("spyre::flash_attention", mutates_args=())
def flash_attention(
    q: torch.Tensor,         # (B, H, S_q, D)
    k: torch.Tensor,         # (B, H, S_k, D) — already GQA-expanded
    v: torch.Tensor,         # (B, H, S_k, D)
    is_causal: bool = False,
    scale: float = None,
) -> torch.Tensor:           # (B, H, S_q, D)
    ...
```

GQA expansion happens in the calling decomposition (matching the current
`spyre__sdpa_overrideable` flow), so this op sees fully-expanded heads.

### Choice B: Replace the decomposition with a Python implementation

Have `spyre__sdpa_overrideable` itself emit the tile-loop directly,
producing many small ops via `torch.compile` rather than a single custom
op.

**Recommendation: Choice A.** A custom op is a clean contract for the
op-set, easier to test in isolation, and lets us swap implementations
later (e.g., a single-launch fused kernel if multi-dsc support lands at
the backend). The current decomposition then becomes a wrapper:

```python
def spyre__sdpa_overrideable(q, k, v, ...):
    # GQA expansion + scale + clone (existing logic)
    out = spyre.flash_attention(q, k, v_expanded, is_causal, scale)
    # Return tuple matching aten signature
    return (out, logsumexp_dummy, ...)
```

### Output dtype

Spyre's matmul operates at fp16; the running output accumulator should
ideally be fp32 for numerical accuracy. Empirically, the SplitK Phase 0
work showed Spyre's per-core accumulator is fp16, so downcast happens at
the matmul boundary — but flash attention's running state lives between
kernels, where we control the dtype.

**Recommendation: fp32 running state, fp16 final output.** The running
state is small (Q_block × D × 4 bytes for output, Q_block × 4 for max
and sum); fp32 doesn't blow up scratchpad. fp32 output preserves the
"K-split improves fp16 accuracy" finding from SplitK Phase 0 by
shortening per-kernel accumulator chains.

## Tile-size constraints

Per-core scratchpad: 2 MB (with `dxp_lx_frac_avail=0.2` reserved → ~1.6
MB usable). Per-core span: 256 MB. fp16 stick = 64 elements.

Per-core working set during one (Q_block, K_block) inner-loop iteration,
assuming 1 head per core (typical when 32 heads × 32 cores match):

| Item | Size |
|---|---|
| Q-tile (loaded once per Q-block sequence) | `Q_block × D × 2` |
| K-tile (current iter) | `K_block × D × 2` |
| V-tile (current iter) | `K_block × D × 2` |
| Score tile `(Q_block, K_block)` | `Q_block × K_block × 2` |
| Running output (fp32) | `Q_block × D × 4` |
| Running max, sum (fp32) | `2 × Q_block × 4` |

For Llama-3-8B (D=128, H=32, fp16) with Q_block=K_block=512:

- Q-tile: 128 KB
- K-tile + V-tile: 256 KB
- Score tile: 512 KB
- Running output: 256 KB
- Running max + sum: 4 KB
- **Total: ~1.16 MB per core** ← fits in 1.6 MB available

Larger tiles (Q_block=K_block=1024) would exceed scratchpad due to the
score tile growing as `Q_block × K_block`. **Default tile size: 512**.

For larger D (e.g., 256 in some MQA models), Q-tile + K-tile dominate
and Q_block needs to shrink. Tile sizing is dtype/D-dependent and
should be a config knob.

## Lowering strategy

The custom op lowers to a Python-level loop over (q_start, k_start)
pairs, each iteration emitting the compiled-path ops the existing
infrastructure already supports:

| Step | Op(s) | Output shape |
|---|---|---|
| 1. Q-tile load | `slice` | `(B, H, Q_block, D)` |
| 2. K-tile load | `slice` | `(B, H, K_block, D)` |
| 3. V-tile load | `slice` | `(B, H, K_block, D)` |
| 4. Score: `Q @ K^T * scale` | `bmm` | `(B, H, Q_block, K_block)` |
| 5. Causal mask (if applicable) | `add` (with -inf triu) | same |
| 6. Running max update | `amax` + `max` | `(B, H, Q_block, 1)` |
| 7. Score normalization | `sub` + `exp` | `(B, H, Q_block, K_block)` |
| 8. Running sum update | `mul` (scale_old) + `sum` + `add` | `(B, H, Q_block, 1)` |
| 9. Running output update | `mul` (scale_old) + `bmm` + `add` | `(B, H, Q_block, D)` |

That's ~9 kernels per inner-loop iteration when fully decomposed (the
Phase 0c probe showed softmax already decomposes to 5 kernels; the
running-state update adds 2 more bmms + 2 more pointwise ops). For
S=4096 with Q_block=K_block=512: 8 × 8 = 64 iterations × 9 kernels =
**576 kernels per attention layer** at ~3 ms each = **1728 ms**. *Worse
than naive SDPA at 1555 ms.*

This is the implementation reality: **frontend-only flash attention with
512-tile tiling does not beat naive SDPA at S=4096.** The tile-streaming
approach IS bandwidth-superior, but Spyre's per-launch overhead (3 ms
flat) is large enough that the kernel-count explosion outweighs the
bandwidth savings at small tiles.

### Two paths to recover the win

**Path A: bigger tiles + accept fewer iterations.** With Q_block=2048,
K_block=512: the score tile is 2048 × 512 × 2 = 2 MB per head per core.
That exceeds the per-core scratchpad. So we'd need to ALSO split Q-rows
across cores (instead of pure head-split). Spyre's per-core scratchpad
isolation makes this awkward; cores can't easily share a Q-tile.

For S=4096 with Q_block=2048, K_block=2048 and assuming we can fit the
score tile per core (8 MB → spills to DDR temp): 2 × 2 = 4 iters × 9 =
36 kernels × 3 ms = **108 ms → 14× speedup**. Realistic but constrained
by per-core memory.

**Path B: fuse softmax + running-state update into a custom kernel.**
The SDSC schema permits multi-dsc compute kernels (one launch covers
several internal ops); the dxp/dsm code base currently asserts dscs_.size()==1
in ~21 places. **This is the same backend gap we hit in MoE Phase 1** —
the multi-dsc backend support that would also unlock grouped GEMM, MoE
kernel fusion, and many other amortizations. Once landed there, flash
attention drops to ~3 kernels per iteration: bmm + (fused softmax-and-
state-update) + bmm. For S=4096, Q_block=512: 64 × 3 = 192 kernels at 3
ms = 576 ms. **Path B alone gives 2.7× speedup**.

**Path A + B together**: 4 iters × 3 fused kernels = 12 kernels = **36 ms
→ 43× speedup**.

### What this design recommends

**Phase 1 implementation = Path A only (frontend-only).** Build the
tile-streaming flash attention with the smallest practical tiles, accept
that at S=4096 the win is moderate (perhaps 5-15× depending on actual
empirical numbers), and ship the API + lowering as the *foundation* for
the eventual Path B win. **The big speedup target moves to Phase 2,
gated on multi-dsc backend work.**

This is the same conclusion we reached for MoE grouped-GEMM: ship the
op contract now, real perf win comes after the backend work lands.
**Both projects share the same blocker** and would unlock together.

## Numerical correctness

Compare against an fp32 CPU reference:

```python
def cpu_sdpa_fp32(q, k, v, is_causal, scale):
    q32, k32, v32 = q.float(), k.float(), v.float()
    s = q32 @ k32.transpose(-2, -1) * scale
    if is_causal:
        s = s + torch.full_like(s, float("-inf")).triu(1)
    a = torch.softmax(s, dim=-1)
    return (a @ v32).to(q.dtype)
```

Phase 0 SplitK work showed Spyre's K-split actually IMPROVES fp16
accuracy because per-core accumulator chains are shorter. Flash
attention has the same property — per-Q-tile accumulation is bounded
by Q_block (e.g., 512) instead of S (e.g., 4096). Numerical drift
should be comparable to or better than naive SDPA.

Tolerance: `atol=0.05, rtol=0.05` initially, tightening based on
measured drift. The naive SDPA's `test_sdpa_cpu` uses similar bounds.

## Integration

The custom op replaces the current decomposition completely once
validated. Two safety nets during transition:

1. **Config flag**: `config.use_flash_attention: bool = False` — opt-in
   initially, allows users (and CI) to compare paths.
2. **Shape gating**: only fire flash attention for S ≥ some threshold
   (e.g., 1024). Below that, the naive path is fine and the kernel-count
   overhead of flash attention isn't worth it.

```python
def spyre__sdpa_overrideable(q, k, v, ...):
    S = q.size(-2)
    if config.use_flash_attention and S >= config.flash_attn_min_seq:
        out = spyre.flash_attention(q, k_expanded, v_expanded, ...)
    else:
        # existing decomposition
        ...
```

## Open questions for implementation

1. **Q-row splitting across cores within a head.** Pure head-split (1
   head per core) is the natural fit for H=32, num_cores=32. For models
   with H ≠ 32 (e.g., Llama-3-70B has H=64 with TP=2 → H_per_card=32),
   does this extend cleanly? What about H=8 (Llama-3-8B GQA after kv
   expansion fans Q-heads back to 32, so 1-per-core works). What about
   H=64 (no TP — Llama-3-70B unsharded)?

2. **Score-tile staging.** For a (B, H, Q_block, K_block) score tile,
   the existing bmm op produces a (B, H, S_q, S_k) output. Slicing it to
   (Q_block, K_block) tiles via `slice` IS supported. But each kernel
   invocation needs to know its tile bounds. The `coordinate_masking_`
   field in SDSC may handle this; need to verify.

3. **K-tile / V-tile dtype.** They flow through DDR per iteration. fp16
   matches Q. No design choice.

4. **The `philox` and `logsumexp` outputs of `_scaled_dot_product_fused_
   attention_overrideable`.** Used for backward + dropout. Phase 1
   targets inference-only (no dropout, no backward); these can be
   constant-zero stubs.

5. **GQA expansion timing.** Currently happens in
   `spyre__sdpa_overrideable` via `key.unsqueeze(2).expand(...)
   .flatten(1, 2)`. This produces a (B, H, S, D) tensor that's a view
   on the (B, H_kv, S, D) original. Does the flash attention loop need
   the expanded version, or can it use the smaller K via per-head
   indexing? Expanded is simpler for Phase 1; optimize later.

## Test plan

1. **Unit tests** in `tests/inductor/test_flash_attention.py` (new):
   - Numerical correctness vs fp32 CPU reference for various shapes
   - Causal vs non-causal
   - With and without GQA expansion
   - With config flag off (baseline) vs on (flash-attention path)

2. **Integration tests** — extend `test_inductor_ops.py:test_sdpa_cpu`
   to also run with `use_flash_attention=True` and verify output match.

3. **Bench harness** — extend `tests/diag_sdpa_baseline.py` to compare:
   - Naive SDPA (current path)
   - Flash attention with Q_block=K_block ∈ {256, 512, 1024}
   - Wall time, kernel count, peak DDR traffic (theoretical)

4. **Long-context regression** — at S=8192 (where naive OOMs because
   intermediate is 4 GB), flash attention should run cleanly. Adding
   this test prevents future regressions.

## Phased implementation plan

| Sub-phase | Scope | Estimated time |
|---|---|---:|
| 1.0 Custom op definition + numeric reference | Define `spyre::flash_attention`, hook lowering, eager fallback to CPU SDPA for correctness baseline | 3-5 days |
| 1.1 Tile-streaming Python loop | Implement the FlashAttention-2 algorithm via existing Spyre ops in the lowering | 1 week |
| 1.2 Numerical validation | Match CPU reference within `atol=0.05`. Bench at S ∈ {512, 1024, 2048, 4096} | 3-5 days |
| 1.3 Tile-size tuning | Sweep Q_block, K_block. Pick best per-shape default | 3 days |
| 1.4 Integration + config gate | Wire into `spyre__sdpa_overrideable`. Config flag + shape gate. | 2-3 days |
| 1.5 Long-context regression test | Add S=8192 test that fails on naive but passes on flash | 1 day |

**Total Phase 1: ~3-4 weeks** for a frontend-only implementation that
delivers a moderate (5-15×) win at S=4096 prefill and unblocks long-
context inference.

**Phase 2 (gated on backend multi-dsc work)**: fuse softmax + running-
state update into a single multi-dsc kernel. Drops kernel count per
iter from 9 to 3. Recovers the 30-86× speedup target. Cross-team work
shared with MoE grouped-GEMM project.

## Decision

**Phase 1 design as specified above. Implementation can begin.**

The frontend-only path delivers a real moderate win, ships the API
contract that downstream serving code (vLLM-Spyre, etc.) can build on,
and unblocks long-context inference (where naive SDPA is unusable due
to the 1+ GB intermediate). Phase 2's full-speedup version is gated on
the same backend multi-dsc work that MoE grouped-GEMM also needs —
both projects benefit from advocating for it together.

## Files

- `tests/sdpa_phase1_design.md` — this document
- Phase 0 findings: `sdpa_phase0_findings.md`, `sdpa_phase0b_findings.md`,
  `sdpa_phase0c_findings.md`
- Phase 0 probes: `diag_sdpa_baseline.py`, `diag_launch_overhead.py`,
  `diag_lx_planning.py`
