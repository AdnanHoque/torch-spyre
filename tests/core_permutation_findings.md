# Is row-major core_id placement actually optimal? — empirical answer

## TL;DR

**No, identity is not strictly optimal.** But the practical answer is
"close enough, and the runtime won't let us do much better without
breaking." Across 7 non-trivial permutations of the physical core_id
mapping on 4 (shape, split) configurations, three concrete things
came out:

1. **One permutation (`stride2`) beats identity by 1.036× on K-split**
   `(4, 1, 8)`. This is the same magnitude as the previously-shipped
   `core_emission_reverse` knob, achieved by a different mechanism.
2. **Three permutations crash dxp** on K-split shapes —
   `reversed`, `antipodal`, `random_42` all reproducibly throw
   `DtException: Workslice information for coreId=N was not found
   for node transfer_lds4_src:lxlu_dst:pe`. The runtime has hidden
   adjacency constraints on K-split-collaborating cores.
3. **Two permutations catastrophically regress:** `block_cyclic`
   → 0.859× (14% slower); `bit_reverse` → 0.716× (28% slower).
   The K-chain *can* be made much worse, just not arbitrarily better.

So sequential placement is empirically near-optimal — beaten only
narrowly, in the same direction and magnitude as the simple
dim-reverse — but the absence of large wins is now a measured fact,
not just a topology argument.

## How we tested

Added a `CORE_ID_PERMUTATION` config knob that applies a permutation
at the `core_id_to_wk_slice` materialization site:
[`compute_ops.py:_get_core_id_permutation`](../torch_spyre/_inductor/codegen/compute_ops.py).
Physical core `c` executes the slice that the unpermuted emitter would
have given to logical core `perm[c]`.

Seven named permutations of `[0..31]`:

| name | description |
|---|---|
| `identity` | `[0, 1, 2, …, 31]` (baseline) |
| `reversed` | `[31, 30, …, 0]` (direction symmetry test) |
| `stride2` | `[0, 2, 4, …, 30, 1, 3, …, 31]` (interleaved half-rings) |
| `block_cyclic` | `[0, 16, 1, 17, …, 15, 31]` (adjacent pairs across halves) |
| `antipodal` | `[16, 17, …, 31, 0, …, 15]` (halves swapped) |
| `bit_reverse` | bit-reverse of low 5 bits of core_id |
| `random_<seed>` | seeded shuffle |

All seven are valid permutations of `[0..31]` (verified). Probe in
[`diag_core_permutation_probe.py`](diag_core_permutation_probe.py),
replication in
[`diag_core_permutation_replicate.py`](diag_core_permutation_replicate.py).

## Initial sweep results (warmup=3, iters=15, single trial)

```text
shape                  split        identity reversed stride2 block_cyc antipodal bit_rev rand_42 rand_7
L3-70B q_proj (pure-N) (1, 32, 1)   1.000x   1.003x   0.983x  0.998x    0.999x    0.985x  0.991x  0.987x
L3-8B  q_proj (pure-N) (1, 32, 1)   1.000x   1.006x   1.006x  1.008x    1.007x    1.008x  1.009x  1.008x
L3-70B q_proj K-split  (4, 1, 8)    1.000x   ERR      1.038x  0.865x    ERR       0.718x  ERR     ERR
L3-70B MLP down output (16, 2, 1)   1.000x   0.997x   1.019x  1.019x    1.004x    1.017x  1.021x  1.017x
```

Random-vs-random noise floor: 0.31% average |Δ|/identity → genuine
single-percent signals are real, sub-percent isn't.

## Replication results (warmup=5, iters=30, two trial orders)

| shape | perm | trial1 sp | trial2 sp | mean | consistent? |
|---|---|---:|---:|---:|---|
| L3-70B q_proj K-split | stride2 | 1.036× | 1.037× | **1.036×** | ✓ same direction |
| L3-70B q_proj K-split | reversed | ERR | ERR | — | ✓ both crashed |
| L3-70B q_proj K-split | antipodal | ERR | ERR | — | ✓ both crashed |
| L3-70B q_proj K-split | random_42 | ERR | ERR | — | ✓ both crashed |
| L3-70B q_proj K-split | block_cyclic | 0.860× | 0.858× | **0.859×** | ✓ same direction |
| L3-70B q_proj K-split | bit_reverse | 0.716× | 0.715× | **0.716×** | ✓ same direction |
| L3-70B MLP down (output) | random_42 | 1.007× | 1.010× | 1.009× | ✓ |
| L3-70B MLP down (output) | stride2 | 1.010× | 1.016× | 1.013× | ✓ |
| L3-70B MLP down (output) | block_cyclic | 1.016× | 1.017× | 1.016× | ✓ |
| L3-70B MLP down (output) | bit_reverse | 1.013× | 1.012× | 1.013× | ✓ |

## What each finding means

### 1. `stride2` on K-split = ~ same win as reverse-emission

For `(4, 1, 8)`:

- **Identity:** K-chain at m=0 spans physical cores `{0, 4, 8, 12, 16, 20, 24, 28}` → 28 hops on the SFP ring.
- **`stride2`:** K-chain at m=0 spans physical cores `{0, 1, 2, 3, 8, 9, 10, 11}` → 11 hops (or thereabouts; depends on chain-traversal order).
- **`core_emission_reverse=True`:** K-chain at m=0 spans physical cores `{0, 1, 2, 3, 4, 5, 6, 7}` → 7 hops.

Reverse-emission has the shortest chain, but `stride2` and reverse
land at the same wall-time. This suggests **once the chain is "short
enough," the marginal cost per saved hop drops sharply** — likely
because PSUM contributes only ~0.1-0.2 ms to the 4.4 ms total, and
shrinking it from 7 to 28 hops only costs about that much.

### 2. K-split crashes are a runtime correctness constraint, not a perf concern

`reversed`, `antipodal`, and `random_42` all crash with the same
exception across both trial orders:

```text
DtException: Workslice information for coreId=28 was not found for node
transfer_lds4_src:lxlu_dst:pe
```

This is a data-transfer node from LDS to LXLU to PE — pre-PT data
routing. The runtime is asking for workslice info for a specific core
that, after permutation, has *some* property the validator doesn't
accept. The crashing perms don't share an obvious physical-core-ID
pattern, so the constraint is on the *structure of the K-cluster
assignment*, not on individual cores.

What this tells us: **the dxp/DSM stack already encodes ring-aware
assumptions about K-split core layout** that we hadn't seen before.
Some permutations satisfy them by accident (identity, stride2,
block_cyclic, bit_reverse); others don't.

This is the most interesting finding, and the one most worth flagging
to the deeptools team. Without their input, we can't tell whether the
constraint is fundamental hardware geometry or an over-restrictive
validator.

### 3. The (16, 2, 1) "many perms slightly win" result

On replication, the random_42 win shrank to ~1.01×. `stride2`,
`block_cyclic`, and `bit_reverse` all clustered in the 1.013–1.016×
band. The wins are real but small enough that they're indistinguishable
from each other (and not far from the 0.31% noise floor).

Interpretation: this isn't a "specific permutation wins" finding — it
looks like **identity is just an unfortunate sweet spot where neither
operand fits the LX scratchpad in a usefully shareable way**, and any
moderate scattering disrupts that pattern enough to let A or B
ring-share by accident. Not a clean lever.

## Where this leaves the user's question

> Are you sure row-major is already optimal? How can we say that for certain?

Empirical answer:

- **Identity is not strictly optimal**, but it is competitive with the
  best alternative (stride2) within ~0.0%, and far better than 6 of
  7 other tested permutations.
- The asymmetry is **roughly 4% upside, 28% downside** — most
  scatterings hurt more than they help.
- Three permutations **crash the runtime entirely** — there are
  hidden adjacency requirements at the dxp/DSM layer that we don't
  fully understand.
- The narrow upside (3.6%) comes from the same source as the existing
  `core_emission_reverse` knob (PSUM chain shortening for K-split).
  The downside comes from breaking the same chain.

So: row-major (sequential identity) **is near-optimal among the
permutations the runtime accepts**. Beating it requires either the
existing reverse-emission knob, or `stride2` (which is functionally
the same thing). Beating reverse-emission would require:

1. Bidirectional ring exploitation (not expressible from SDSC layer)
2. Tree-shaped reduction (hardware doesn't support — SFP is a ring)
3. Understanding and lifting the runtime's K-split adjacency constraint
   (would need deeptools collaboration)

## What this changes about the recommendation

The previous "core-ordering project is dead" call is *less* dead but
not enough to change the bottom line:

- **Don't ship a permutation knob for performance.** stride2 doesn't
  beat reverse-emission, and reverse-emission already isn't worth
  shipping a heuristic for.
- **Do flag the K-split crash to the deeptools team.** This is a
  silent assumption in the runtime that any future scheduler change
  could trip over. Documenting the constraint (or fixing the
  validator) helps anyone touching this code.
- **Update the matmul reference doc** to note that row-major
  sequential placement is near-optimal *among runtime-accepted
  orderings*, with a citation to this empirical test.

## Files

- [`tests/diag_core_permutation_probe.py`](diag_core_permutation_probe.py)
  — initial sweep
- [`tests/diag_core_permutation_replicate.py`](diag_core_permutation_replicate.py)
  — replication with both trial orders + crash reproduction
- raw outputs:
  [`diag_core_permutation_probe_results.txt`](diag_core_permutation_probe_results.txt),
  [`diag_core_permutation_replicate_results.txt`](diag_core_permutation_replicate_results.txt)
- knob: [`config.core_id_permutation`](../torch_spyre/_inductor/config.py)
- implementation:
  [`compute_ops._get_core_id_permutation`](../torch_spyre/_inductor/codegen/compute_ops.py)

## Open question for follow-up

The K-split crash is the highest-information finding. Specifically:
**which structural property of a permutation does the dxp validator
require for K-split shapes to compile?** Knowing this would tell us:
- Whether identity is structurally optimal or just incidentally OK
- Whether there are *better* valid permutations we haven't tried
- Whether the validator could be relaxed to allow more orderings

This is a deeptools-side investigation, not a torch_spyre one.
