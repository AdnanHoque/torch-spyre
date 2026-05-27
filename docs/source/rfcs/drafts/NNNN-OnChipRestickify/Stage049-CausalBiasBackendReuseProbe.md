# Stage 049: Causal Bias Backend Reuse Probe

Date: 2026-05-27

## Purpose

Stage048 added the Torch-Spyre scaffold for:

```text
spyre::causal_score_bias_like(scores, key_start)
```

and proved that the generated SuperDSC reaches DeepTools with:

```text
opFuncName="causal_score_bias_like"
constantInfo_: keyStart
```

This stage checks whether the remaining backend work can be avoided by mapping
the scaffold to an existing DeepTools opFunc.

## Probe Tool

Added:

```text
tools/causal_score_bias_backend_probe.py
```

The tool compiles and runs a tiny function:

```python
def fn(scores):
    return torch.ops.spyre.causal_score_bias_like(scores, key_start)
```

It can optionally remap the emitted backend opFunc:

```sh
"$PYTHON" tools/causal_score_bias_backend_probe.py --opfunc identity
"$PYTHON" tools/causal_score_bias_backend_probe.py --opfunc maskbyindex
"$PYTHON" tools/causal_score_bias_backend_probe.py --opfunc where3
"$PYTHON" tools/causal_score_bias_backend_probe.py --opfunc greaterthan
```

Each run prints `RESULT_JSON` with:

- compile/run status;
- generated SDSC paths;
- compute op names;
- constants carried through the SDSC; and
- a small output summary and `matches_expected` when the run succeeds.

## DeepTools Inventory

The patched DeepTools source recognizes these relevant opFuncs:

```text
identity
maskbyindex
where3
greaterthan
lesserthan
```

It also has a higher-level `IdxToMask` data-conversion path with causal-mask
metadata:

```text
isIdxToMaskSdc
idxToMaskDimIdx
idxToMaskValidElementOffset
causalMask
```

That path produces 0/1 masks from int64 indices during host data conversion. It
is not a SuperDSC pointwise compute op and is not directly reachable from the
current score-shaped `causal_score_bias_like` lowering.

## Manual OpFunc Reuse Probe

Starting from the generated causal-bias SDSC, manually replacing only
`opFuncName` with each existing candidate compiled successfully:

```text
identity     rc=0
maskbyindex  rc=0
where3       rc=0
greaterthan  rc=0
```

Runtime semantics do not match the required causal additive bias:

```text
identity:
  output is the score input unchanged

maskbyindex:
  output is a constant-looking -0.001300811767578125 pattern

where3:
  output is the same constant-looking pattern as maskbyindex

greaterthan:
  output is the same constant-looking pattern as maskbyindex
```

None of those outputs is the required:

```text
bias[..., q, k_block] = -inf if key_start + k_block > q else 0
```

With the committed probe tool, those successful remap runs report:

```text
matches_expected: false
```

## Pod Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
```

Commands:

```sh
"$PYTHON" tools/causal_score_bias_backend_probe.py \
  --opfunc causal_score_bias_like \
  --cache-dir /tmp/stage049-probe-causal_score_bias_like

"$PYTHON" tools/causal_score_bias_backend_probe.py \
  --opfunc identity \
  --cache-dir /tmp/stage049-probe-identity

"$PYTHON" tools/causal_score_bias_backend_probe.py \
  --opfunc maskbyindex \
  --cache-dir /tmp/stage049-probe-maskbyindex

"$PYTHON" tools/causal_score_bias_backend_probe.py \
  --opfunc where3 \
  --cache-dir /tmp/stage049-probe-where3

"$PYTHON" tools/causal_score_bias_backend_probe.py \
  --opfunc greaterthan \
  --cache-dir /tmp/stage049-probe-greaterthan
```

Results:

```text
causal_score_bias_like status=failed
  sdsc opfuncs ["causal_score_bias_like"]
  constants ["keyStart"]
  error contains "Unrecognized opFunc: causal_score_bias_like"

identity status=ok matches_expected=false
maskbyindex status=ok matches_expected=false
where3 status=ok matches_expected=false
greaterthan status=ok matches_expected=false
```

## Interpretation

The one-input score-layout scaffold cannot be made correct by renaming the
backend opFunc to an existing primitive.

There are two plausible implementation paths left:

- add a real DeepTools/SDSC opFunc for `causal_score_bias_like`; or
- introduce a different Torch-Spyre path that can emit/use the existing
  higher-level `IdxToMask` data-conversion machinery, then transform the 0/1
  mask into additive score bias without CPU fallbacks.

The first path is narrower and matches the Stage047 contract. The second path
would require a larger representation change because current SuperDSC pointwise
emission has no hook for `isIdxToMaskSdc`/`causalMask` DCI metadata.

## Next

- keep `causal_score_bias_like` as the frontend contract;
- prototype DeepTools opFunc support or produce a patch for that layer; and
- only wire `_flash_attention_prefill` once the probe returns the causal
  `0/-inf` pattern for the real `causal_score_bias_like` opFunc.

## Local Validation

```text
py_compile(tools/causal_score_bias_backend_probe.py) passed
git diff --check passed
```
