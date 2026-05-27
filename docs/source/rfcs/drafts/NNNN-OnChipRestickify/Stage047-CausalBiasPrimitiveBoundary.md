# Stage 047: Causal Bias Primitive Boundary

Date: 2026-05-27

## Purpose

Stage046 made the final causal readiness check gate-shaped:

```text
tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --cases b1h2d64_block64_causal \
  --forbid-fallbacks
```

That command still fails today because the square causal prefill path builds its
triangular additive bias with `aten.triu.default`, which falls back to CPU.  This
stage records the compiler boundary for the replacement so the next attempt does
not repeat the Stage044 graph-level mask rewrites.

## Conclusion

The viable replacement is a score-layout-anchored backend primitive, not another
Python decomposition of the triangular mask.

The primitive should operate on the already materialized score block:

```text
scores: [batch, heads, query_row, key_block_col]
key_start: Python int for the current KV block
```

and produce a same-shape, same-layout additive bias:

```text
bias[..., q, k_block] = -inf if key_start + k_block > q else 0
```

The output must stay anchored to the score tensor's `FixedTiledLayout`, because
the next operation is `scores + bias` before the score transpose.

## Why The Upstream-Style Lowering Is Not Enough

The natural Inductor lowering shape would be:

```python
def inner_fn(index):
    q = ops.index_expr(index[-2], torch.int64)
    k = ops.index_expr(index[-1] + key_start, torch.int64)
    mask = ops.gt(k, q)
    return ops.where(
        mask,
        ops.constant(float("-inf"), dtype),
        ops.constant(0.0, dtype),
    )
```

That is the right expression model for a generic Inductor backend, but it is not
currently accepted by Spyre codegen:

- `SpyreOpFuncs` has comparison and `where` names, but no `index_expr` handler.
- `SpyreKernel.store` only accepts tensor accesses as arguments to a
  `PointwiseOp`; nested pointwise values and immediate constants are rejected.
- Scalar constants are handled today by converting them to `spyre.constant`
  tensors or op metadata, not by embedding them in nested RValue trees.

So a custom op that lowers directly to nested `where(gt(index_expr(...)))` would
fail before it reached DeepTools.

## Backend Primitive Contract

A workable first implementation should be explicit:

```text
spyre::causal_score_bias_like(scores: Tensor, key_start: int) -> Tensor
```

Expected registration:

- custom op in `torch_spyre/_inductor/customops.py`;
- fake implementation returning `scores.new_empty(scores.size())`;
- CPU kernel only if the helper is used from CPU-direct tests;
- Spyre lowering in `torch_spyre/_inductor/lowering.py`;
- `SpyreOpFuncs` entry emitting a single backend op with `key_start` in
  `op_info["constants"]`;
- backend/SDSC support for using output coordinates and `key_start` to choose
  `0` or `-inf`.

The lowering should read `scores` as a real tensor input even if the score value
is not mathematically needed.  That input is the layout anchor that lets the
single-argument pointwise output inherit the score layout.

Sketch:

```python
def lower_causal_score_bias_like(scores, key_start: int):
    fn = lowering.ops_wrapper(torch.ops.spyre.causal_score_bias_like.__name__)
    loader = scores.make_loader()

    def inner_fn(index):
        return fn(loader(index), key_start)

    return Pointwise.create(
        device=scores.get_device(),
        dtype=scores.get_dtype(),
        inner_fn=inner_fn,
        ranges=scores.get_size(),
        origin_node=scores.get_origin_node(),
        traceback=scores.get_traceback(),
    )
```

The backend op must not depend on a materialized `[L, block]` constant buffer.
It must derive the triangular predicate from coordinates inside the score-shaped
operation.

## Dead Ends To Avoid

Do not replace the current `triu` fallback with another graph-level mask built
from:

- `arange` query/key comparisons plus `masked_fill`;
- `zeros`/`full`/`cat`/`stack` block-local bias tensors;
- Python-constructed constant bias tensors;
- score-shaped `full_like` masks plus slice assignment;
- row-prefix or diagonal recomposition through `cat` slices.

Those paths were already rejected in Stage044 by stick-layout incompatibility,
`Unexpected stick expression 63`, `FixedLayout` constant buffers, or AOT
functionalization failures.

## Validation Target

After the backend primitive exists, the minimal proof is:

```sh
"$PYTHON" tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --cases b1h2d64_block64_causal \
  --forbid-fallbacks \
  --case-output-dir /tmp/sdpa-stage047-causal-bias-json \
  --cache-prefix /tmp/sdpa-stage047-causal-bias \
  --timeout-s 700 \
  --output-json /tmp/sdpa-stage047-causal-bias.json
```

Expected result:

```text
PROMOTION_GATE_PASSED gate=onchip_layout_xform cases=1 rows=2
```

The failure signatures to eliminate are:

```text
FallbackWarning
aten.triu.default
falling back to cpu
```

## Local Validation

```text
tests/_inductor/test_onchip_sdpa_sweep_logic.py          6/6 passed
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py 9/9 passed
py_compile(decompositions.py) passed
git diff --check passed
```

## Next

- add backend/SDSC support for `causal_score_bias_like`;
- wire the custom op and Spyre lowering to `_flash_attention_prefill`; and
- rerun the Stage046 fallback-forbidden causal promotion gate until it exits 0.
