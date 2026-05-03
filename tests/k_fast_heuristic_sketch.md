# `k_fast` permutation as a default — planner heuristic sketch

## Statement

Make `core_id_permutation = "k_fast"` the default (currently `"identity"`).
Equivalently, change the SDSC core_id-to-slice emitter so that
K-collaborators (cores that share `(m_slice, n_slice)` and differ only
in `k_slice`) end up at consecutive physical core IDs.

## Why this is the right shape of heuristic

Across every K-split (k > 1) measurement we've taken on this branch,
k_fast wins. Across every k = 1 measurement, k_fast equals identity.
This is the algebra:

```
perm[c] = (c mod k) * (m * n) + (c // k)
```

When `k = 1`, `c mod 1 = 0` and `c // 1 = c`, so `perm[c] = c` (identity).
When `k > 1`, K-cluster `j` (logicals `{j, j + m·n, j + 2·m·n, ..., j + (k-1)·m·n}`)
maps to physical positions `{j·k, j·k + 1, ..., j·k + k − 1}`. PSUM
chains traverse `k − 1` ring hops instead of `(k − 1) · m · n`, an
`m · n` × reduction in chain length.

## Empirical evidence (this branch)

Direct verification of the linear ring-distance model
([`diag_core_pairwise_distance.py`](diag_core_pairwise_distance.py)):

| K-pair distance d | wall ms |
|---:|---:|
| 1 | 3.97 |
| 2 | 4.00 |
| 4 | 5.01 |
| 8 | 6.98 |
| 16 | 10.89 |

Linear fit `wall ≈ 3.22 + 0.476·d`, RMSE 0.16 ms across 5 points.
Each unit of K-pair physical distance adds ~0.48 ms wall time.
core_id is monotonically adjacent on the physical ring (otherwise
this fit would have plateaus or noise).

`k_fast` validation
([`diag_core_k_fast_validate.py`](diag_core_k_fast_validate.py)):

| shape | split | k_fast vs identity | crashes? |
|---|---|---:|---|
| kv_proj M=2048 | (1, 16, 2) | **2.727×** | no |
| q_proj K-split | (4, 1, 8) | 1.033× | no |
| L3-8B MLP down K | (4, 1, 8) | 1.008× | no |
| Mixed | (2, 4, 4) | 1.074× | no |
| pure-N | (1, 32, 1) | 0.997× (≈identity) | no |

## Risk analysis

### What could go wrong if we ship k_fast as default?

1. **Crash on shapes we haven't tested.** The narrow probe found that
   `reversed`, `antipodal`, and `random_42` reproducibly crash dxp on
   K-split shapes with the error
   `"Workslice information for coreId=N was not found for node
   transfer_lds4_src:lxlu_dst:pe"`. The structural property the
   runtime requires is unknown.
   - Mitigation: k_fast doesn't fall into any of the patterns that
     crash. It *packs* K-collaborators contiguously rather than
     scattering them. All the crashes we've seen are from scattering.
   - But: this isn't proof. A shape we haven't tested could still
     crash.

2. **Marginal regression on (1, 32, 1).** Pure-N showed 0.997× — a
   0.3% slowdown that's within noise but consistently negative across
   trials. If real, it's negligible vs. the K-split wins.

3. **Output-reorder shape (16, 2, 1) loses the block_cyclic
   advantage.** Our broad sweep found block_cyclic gives 2.1% on
   L3-70B MLP down with (16, 2, 1). k_fast is identity there. We'd
   leave that 2.1% on the table.
   - Mitigation: keep `core_id_permutation` env var; users who pin
     (16, 2, 1) can opt into block_cyclic.

4. **L3-70B q_proj/o_proj long-M lose the block_cyclic advantage.**
   Same — we'd leave 2-3% on the table for these specific shapes.
   - Mitigation: same as above.

### What's the worst case?

Per the data: 0.3% loss on pure-N (within noise), 2-3% loss on the
narrow set of shapes where block_cyclic was a local pocket. All
recoverable via the existing env var.

### What's the best case?

Per the data: 2.7× speedup on kv_proj-style narrow-N matmuls
(1, 16, 2). 7.4% on mixed splits (which the planner doesn't pick
today but might in future). 1-4% on K-split mixed (m, 1, k).

The asymmetry is overwhelming.

## Two-stage shipping plan

### Stage 1 — make k_fast the default behind a config flag

```python
# config.py change:
core_id_permutation: str = os.environ.get("CORE_ID_PERMUTATION", "k_fast")
```

Old behavior recoverable via `CORE_ID_PERMUTATION=identity`.

Any user-visible regression can be diagnosed by setting the env var.
Easy revert via single-line config change.

### Stage 2 — soak in CI / benchmark suite

Run the full inductor benchmark suite with k_fast on. Catch any
regression on shapes we haven't tested. If clean → make it the
unconditional default.

### Stage 3 — deeper integration

Either:
- (a) Move the perm into `_get_core_to_slice_mapping` itself (so it's
  not a runtime config knob, just the emitter's behavior).
- (b) Generalize to non-matmul ops (the current k_fast assumes the
  last dim of `iteration_space` is K, which is matmul-specific).

Both are cleanups, not new functionality.

## What this doesn't address

1. **Bidirectional ring** — both CW and CCW data rings exist; only
   one is used per chain today. K-fast halves the chain hops; using
   both directions could halve them again.
2. **Output writeback patterns** — the L3-70B q_proj long-M finding
   suggests there's *some* additional lever from output-side
   permutations. Not understood; not addressed by k_fast.
3. **The dxp validator constraint** — we still don't know what
   structural property makes some perms crash. Worth flagging to
   deeptools team.

## Open questions

1. **Does k_fast hold up on multi-AIU splits?** This branch only
   tested single-chip. With QGI cross-chip traffic the optimal
   permutation might differ.
2. **Should we generalize to other ops?** Reductions (softmax,
   layernorm) also have a "reduction dim" — packing reducers
   contiguously might help there too.
3. **Is there a smarter heuristic that picks per-shape (k_fast vs
   block_cyclic vs identity)?** k_fast leaves 2-3% on the table for
   a few specific shapes. A planner-aware heuristic could capture
   those, but the maintenance cost vs ~3% gain is questionable.

## Recommendation

**Ship Stage 1.** The change is one line of config, fully reversible
via env var, with measured 1-170% wins on K-split shapes and ≤0.3%
loss on pure-N. Stage 2 and 3 follow if Stage 1 holds clean.

The kv_proj 2.7× alone justifies the change — kv_proj is in every
transformer's attention layer and runs once per token in decode.
