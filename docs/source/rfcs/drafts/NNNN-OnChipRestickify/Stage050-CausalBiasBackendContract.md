# Stage050: Causal Bias Backend Contract

Stage049 showed that reusing an existing DeepTools `opFuncName` is not enough:
the probe can compile `identity`, `maskbyindex`, `where3`, and `greaterthan`,
but none of them produce the causal `0/-inf` bias pattern. This stage records
the narrower backend contract for a real implementation of
`spyre::causal_score_bias_like(scores, key_start)`.

## Current frontend contract

Torch-Spyre now emits a score-layout-anchored custom op:

```python
torch.ops.spyre.causal_score_bias_like(scores, key_start)
```

The Spyre lowering emits one pointwise SDSC:

```text
opFuncName="causal_score_bias_like"
inputLabeledDs=["Tensor0-idx0"]
outputLabeledDs=["Tensor1-idx1"]
constantInfo_: name_="keyStart"
```

The input tensor is a layout anchor only. The output must have the same shape,
layout, dtype, and sharding as `scores`. For each output element:

```text
output[..., q, k_block] = -inf  if key_start + k_block > q
                           0    otherwise
```

For the current causal prefill gate shape, the generated SDSC score layout is:

```text
layoutDimOrder_=["x", "mb", "out"]
stickDimOrder_=["out"]
N_: mb_=2, x_=4, out_=64   # probe shape B=1, H=2, Q=4, Kblock=64
```

The inferred query dimension is `x`, and the inferred key-block dimension is
the stick dimension `out`.

## Why existing DDL templates are insufficient

DeepTools has pointwise templates for unary and broadcast operations:

- `unary_parallel.ddl` maps ops like `identity`, `relu`, `exp`, `sqrt`.
- `broadcast_ops.ddl` maps ops like `add`, `where3`, `greaterthan`.
- `topk.ddl` maps `maskbyindex`.

Those templates operate on input tensor values plus scalar or fixed constants.
They do not expose output coordinates as SFP operands. DDL has loop conditions,
but `value_expr` is limited to `first`, `last`, or numeric literals; it cannot
express `key_start + k > q`, compare two loop dimensions, or construct a
per-lane key index for the `out` stick.

This is why the backend-reuse probe can compile known `opFuncName`s while still
returning the wrong tensor.

## Backend implementation requirement

A correct DeepTools implementation must add more than enum/string recognition.
It needs an implementation of `causal_score_bias_like` that can:

1. Parse `keyStart` from `constantInfo_`.
2. Identify the score output query dimension and key-block stick dimension.
3. Use absolute output coordinates, including per-core work-slice offsets and
   per-lane stick positions, not only local loop counters.
4. Write the same dtype as the score tensor, with exact `0` and `-inf` values.
5. Preserve the output layout and sharding so the following `scores + bias`
   pointwise add can remain device-native.

The existing `IdxToMask`/`causalMask` path is a host data-conversion feature.
It is not currently reachable from this score-shaped SuperDSC pointwise custom
op without a new Torch-Spyre/DeepTools metadata bridge.

## Implementation paths

The agent audit narrowed this to two real paths.

### New backend opFunc

This keeps the current frontend contract unchanged. The minimum parser surface
in DeepTools is:

- add `OpFuncs::CAUSAL_SCORE_BIAS_LIKE` to `dsc/dscdefn.h`;
- add `{OpFuncs::CAUSAL_SCORE_BIAS_LIKE, "causal_score_bias_like"}` to
  `dsc/dscdefn.cpp`;
- add the `{1, 1}` arity entry in `dsc/designSpaceConfig.cpp`;
- include the op in the pointwise scheduler class if it schedules like a unary
  SFP operation.

That parser patch only moves the failure. The semantic lowering still needs to
consume `keyStart`, recover logical `q` and `k` coordinates from the score
layout, account for `coreIdToWkSlice_`, and emit dtype-correct `0/-inf`.

### IdxToMask plus where3

DeepTools already has a causal `IdxToMask` data-convert path. It produces a
`1/0` FP16 mask, not an additive `0/-inf` bias. For query length `Q` and key
block start `key_start`, the useful setting is:

```text
length-one INT64 input = [Q]
idxToMaskValidElementOffset = -key_start
causalMask = true
```

The DeepTools causal valid-count formula is:

```text
numValidElems = input + offset + q - (Q - 1)
              = q - key_start + 1
```

That matches the number of allowed keys in the block. A following
`where3(mask, 0, -inf)` can turn the predicate into the additive bias.

This path avoids inventing a new pointwise opFunc, but it requires Torch-Spyre
to emit a new internal data-convert representation with the DCI/NodeProperty
metadata:

```text
isIdxToMaskSdc = true
idxToMaskDimIdx = <key stick dimension>
idxToMaskValidElementOffset = -key_start
causalMask = true
invertedMask = false
reversedMask = false
```

Torch-Spyre's current `SDSCSpec` and pointwise `computeOp_` generation have no
surface for that metadata, so this is a new codegen path, not a remap of the
existing pointwise custom op.

There is also an imported-SuperDSC parser gap on the DeepTools side: current
`datadscs_` JSON accepts copy/restickify-style data ops, but not
`op.name = "IdxToMask"`. The schema extension Torch-Spyre would need to emit is
shaped like:

```json
"op": {
  "name": "IdxToMask",
  "idxToMaskDimIdx": 2,
  "idxToMaskValidElementOffset": -2,
  "invertedMask": 0,
  "reversedMask": 0,
  "causalMask": 1
}
```

The DSM/host DCI path also constrains causal masks to a single causal plane:
the mask dimension must be the stick dimension, exactly one non-mask causal
dimension may be greater than one, and all other dimensions must be size one.
For the current probe layout:

```text
score layout sizes: x=4, mb=2, out=64
mask layout sizes:  x=4, mb=1, out=64
DCI output_shape_:  [64, 4, 1, 1]
```

The following `where3` must therefore broadcast the mask predicate across the
score tensor's `mb` dimension.

## Probe update

`tools/causal_score_bias_backend_probe.py` now records a
`causal_score_bias_contract` block for generated causal-bias SDSCs, even when
DeepTools aborts before execution. This captures:

- opFunc, input/output counts, and constants;
- inner-DSC `N_`/`numCoresUsed_` plus SuperDSC `numWkSlicesPerDim_`
  and `coreIdToWkSlice_`;
- output `layoutDimOrder_`, `stickDimOrder_`, and stick size;
- inferred query/key dimensions for the current score layout.

It also records a `causal_idx_to_mask_candidate` block. That block is a
descriptor, not runtime support: it reports the IdxToMask DCI metadata,
the collapsed causal-plane mask layout, the `where3` composition, and the
current `datadscs_` parser blocker.

For parser bring-up, the probe can now materialize the next Torch-Spyre emission
plan without enabling it in runtime bundle generation:

```sh
"$PYTHON" tools/causal_score_bias_backend_probe.py \
  --cache-dir /tmp/stage050-plan-probe \
  --candidate-plan-json /tmp/stage050-causal-idx-to-mask-plan.json
```

The plan contains the proposed mixed-SuperDSC schedule, an `IdxToMask` data-op
fragment with per-core causal-plane mask pieces, and the `where3` compute
fragment. It is intentionally marked `runtime_status = "not_emitted"` until the
DeepTools data-op parser accepts `IdxToMask` and Torch-Spyre has real tensor
sources for the `where3` true/false inputs.

For a sharper imported-SuperDSC parser check, the probe can also write a
single-SDSC executable mini-bundle:

```sh
"$PYTHON" tools/causal_score_bias_backend_probe.py \
  --cache-dir /tmp/stage050-parser-probe \
  --candidate-parser-probe-dir /tmp/stage050-parser-probe-bundle \
  --run-candidate-parser-probe
```

That artifact rewrites the generated compute op to `identity`, leaves
`opFuncsUsed_` empty, and places `IdxToMask` only in `datadscs_[].op.name`.
Its expected near-term failure is therefore the DeepTools data-op parser
boundary, not the original `causal_score_bias_like` opFunc import.

The same non-executed plan can be emitted by normal bundle generation with:

```sh
SPYRE_CAUSAL_IDX_TO_MASK_PLAN_ARTIFACT=1
```

This writes `causal_idx_to_mask_plan_<idx>.json` next to the generated SDSCs but
does not add that file to `bundle.mlir`.

This keeps the backend contract executable and visible in the same probe that
will later prove semantic correctness with `matches_expected=true`.

## Acceptance ladder

Backend bring-up is not complete until all three checks pass:

```sh
"$PYTHON" tools/causal_score_bias_backend_probe.py \
  --batch 1 --heads 2 --query-len 4 --key-len 64 --key-start 2 \
  --opfunc causal_score_bias_like \
  --cache-dir /tmp/probe-causal-score-bias-real
```

Expected final result:

```text
RESULT_JSON.status="ok"
RESULT_JSON.matches_expected=true
```

Then causal SDPA must avoid the old `aten.triu.default` fallback:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128 \
  --variants onchip_master_layout_xform \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --is-causal --forbid-fallbacks \
  --warmup 1 --iters 1 --timeout-s 480 \
  --cache-prefix /tmp/sdpa-causal-native-smoke \
  --output-json /tmp/sdpa-causal-native-smoke.json
```

Finally, the causal promotion gate must pass:

```sh
"$PYTHON" tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --cases b1h2d64_block64_causal \
  --forbid-fallbacks \
  --case-output-dir /tmp/sdpa-causal-forbid-gate-json \
  --cache-prefix /tmp/sdpa-causal-forbid-gate \
  --timeout-s 700 \
  --output-json /tmp/sdpa-causal-forbid-gate.json
```

Expected final result:

```text
PROMOTION_GATE_PASSED gate=onchip_layout_xform cases=1 rows=2
```

Only after the probe returns the causal `0/-inf` pattern should
`_flash_attention_prefill` replace the Stage047 `triu` fallback with:

```python
scores = scores + torch.ops.spyre.causal_score_bias_like(scores, start)
```
