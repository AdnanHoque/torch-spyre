# Restickify verdict categories

How to classify a restickify op for the ring-aware optimization project.

## The categories

| Verdict | Trigger | Can the ring help? | Why |
|---|---|---|---|
| **HBM-LOAD** | Restickify input is a **graph input** (weight, kv-cache pointer, etc.). In an SDSC bundle, the restickify SDSC appears *before* any compute op. | **No** | Graph inputs live in HBM by definition. The *read* half must come from HBM. (A related project could save the *write* half: write the restickified output to LX instead of HBM.) |
| **FUNDAMENTAL** | Restickify sits between two on-chip ops, **and** producer partition ≠ consumer partition (`prod_part ≠ cons_part` by host stride). | **Yes — fully** | The cross-core stick movement is structurally required. No `work_distribution` split can give the restickify a single partition matching both sides. The ring (`STCDPOpLx`) is the on-chip shuffle that replaces today's HBM round trip. |
| **INCIDENTAL** | Restickify sits between two on-chip ops, **and** producer partition == consumer partition. | **Not needed** | The simpler fix: give the restickify the same split as its neighbors. `core_div_mismatch` doesn't fire, the buffer stays in LX, and a per-core `ReStickifyOpLx` (or a layout absorbed into the producer/consumer) suffices. Pure `work_distribution` change, no ring. |

## Decision rule

```
if producer is None (graph input):
    verdict = HBM-LOAD
elif prod_part == cons_part:
    verdict = INCIDENTAL
else:
    verdict = FUNDAMENTAL
```

`prod_part` / `cons_part` are the producer's and consumer's `op_it_space_splits[0]`
(output-side splits) — keyed by **host stride** of the buffer. They're directly
comparable across ops because a buffer's host strides are layout-invariant.

## Why the rule works (one paragraph derivation)

A restickify lowers to a `Pointwise` whose `inner_fn = loader(index)` — read
index == write index. The relayout lives only in the device `SpyreTensorLayout`,
and the restickify's input and output buffers share the same host layout
(`FixedTiledLayout` preserves host `size`/`stride`). Therefore a restickify
induces **one** host-stride partition on both its input and its output. Both
edges are alignable iff `rs_part == prod_part == cons_part`. That's achievable
iff `prod_part == cons_part` — pick `rs_part` equal to both. If `prod_part ≠
cons_part`, no choice of `rs_part` makes both edges aligned: one side will
always be cross-core. That cross-core requirement is what only an on-chip
shuffle (the ring / `STCDPOpLx`) can satisfy without going through HBM.

## What this looks like in real probes

From `diag_restickify_lx_trace.py` at `sencores=32, lx_planning=True,
allow_all_ops=True`:

| case | verdict | producer → consumer partitions |
|---|---|---|
| `linear_x_Wt_decode` (`x @ W.t()`) | HBM-LOAD | graph input → matmul `{s4096:x32}` |
| `transposed_computed_intermediate` (`(a+b).t() + c`) | HBM-LOAD | graph input → pointwise `{s1:x32}` |
| `chained_matmul_transposed` (`(a@b).t() + c`) | HBM-LOAD | graph input → pointwise `{s1:x32}` |
| `matmul_then_transposed_add` (`(a@b) + c.t()`) | **FUNDAMENTAL** | matmul `{s128:x32}` → pointwise `{s1:x32}` |
| `matmul_transposed_matmul` (`(a@b).t() @ c`) | **FUNDAMENTAL** | matmul `{s256:x32}` → matmul `{s1:x32}` |

The two FUNDAMENTAL cases share the canonical signature: **matmul partitions by
the generated/N dim (large stride), transposed consumer wants the unit-stride
dim, no split bridges them.**

## Position-based proxy (for cache-bundle analysis)

When reading SDSC bundles directly (rather than running the trace probe), a
position-based heuristic is a defensible proxy because torch_spyre emits each
fused kernel as a sequence of SDSC files in topological order:

- Restickify SDSC appears **before** any compute op in the bundle → **HBM-LOAD**
  (weight prep — restickify is preparing a graph-input weight for the compute
  op that follows).
- Restickify SDSC appears **after** a compute op → **FUNDAMENTAL or INCIDENTAL**
  (post-compute relayout). To distinguish further you need the
  `op_it_space_splits` comparison.

In practice at multi-core under `LX_PLANNING=1 + allow_all_ops`, the post-compute
case skews FUNDAMENTAL rather than INCIDENTAL — the trace data above showed 0
INCIDENTAL cases — so position-based "post-compute restickify ≈ FUNDAMENTAL" is
a reasonable approximation for cache analysis.

## Implications for the project

- **The ring (`STCDPOpLx`) only addresses FUNDAMENTAL restickifies.** It cannot
  help HBM-LOAD (input is HBM by definition) or INCIDENTAL (split alignment
  already suffices).
- **HBM-LOAD is the largest share** of restickify HBM bytes in current bundles
  (weight prep dominates). A separate but mechanism-adjacent optimization —
  writing the restickified weight to LX instead of HBM — could save the *write*
  half of HBM-LOAD traffic. Same deeptools `datadscs_` machinery applies.
- **INCIDENTAL has not been observed** in the patterns we've probed at multi-core
  — the matmul→transpose-consumer signature dominates and is fundamentally
  cross-core.
