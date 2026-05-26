# Stage 017: Flash Deferred Score Scaling

Date: 2026-05-26

## Purpose

Stage 014 found that real generated flash-prefill `batchmatmul` inputs were
blocked by pre-matmul Q/K scaling:

```text
2_mul -> 5_batchmatmul input0
3_mul -> 5_batchmatmul input1
```

Those edges are poor Tier 1 targets because the producer and consumer physical
stick layouts differ.  Mathematically, SDPA scaling can be applied as either:

```text
(Q * sqrt(scale)) @ (K * sqrt(scale)).T
```

or:

```text
(Q @ K.T) * scale
```

Stage 017 moves the flash-prefill path to the second form.  This keeps the
ordinary non-flash SDPA lowering on the existing sqrt-scaled Q/K path, but lets
flash-prefill avoid materialized Q/K scalar producers before the score matmul.

## Implementation

Code change:

- `torch_spyre/_inductor/decompositions.py`
  - `_flash_attention_prefill` now accepts `score_scale`.
  - Flash score tiles now compute `torch.matmul(query, key_block_t)` and then
    multiply the score tensor by `score_scale`.
  - `spyre__sdpa_overrideable` computes `score_scale` once.
  - The non-flash fallback still applies `sqrt(score_scale)` to Q and K before
    the existing dense SDPA sequence.

This is a lowering-only change.  No new uncertified on-chip edge is enabled.

## Validation

Local:

```text
py_compile(torch_spyre/_inductor/decompositions.py)  passed
git diff --check                                     passed
```

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
commit=1bd0fdb
```

Focused mixed-flash command:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-flash-defer-scale-1779821036
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

Result:

```text
1 passed, 6 deselected in 12.34s
```

Direct flash-prefill command:

```sh
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-flash-prefill-suite-1779821230
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_prefill" -q -s
```

Result:

```text
1 passed, 6 deselected in 13.48s
```

Pointwise-handoff command:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=1
export SPYRE_ONCHIP_HANDOFF_MIN_BYTES=0
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-flash-defer-scale-pointwise-1779821171
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

Result:

```text
1 passed, 6 deselected in 14.41s
```

## Generated Graph Shape

Before this change, the first flash-prefill bundle started with scalar Q/K
producers before the score matmul:

```text
0_identity
1_identity
2_mul
3_mul
4_ReStickifyOpHBM
5_batchmatmul
...
```

After this change, the same bundle starts:

```text
0_identity
1_identity
2_ReStickifyOpHBM
3_batchmatmul
4_mul
5_ReStickifyOpHBM
...
```

The old pre-score `mul -> batchmatmul` blockers are gone.  The scale multiply is
now on the score tensor:

```text
3_batchmatmul -> 4_mul
```

## Pointwise Realization Result

With `SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=1`, the pass still realizes the
two later same-layout pointwise handoffs in the second flash bundle:

```text
bundle: sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_y24kuhw3
mixed:  sdsc_11_mul.json
mixed:  sdsc_17_add.json
```

DXP debug evidence:

```text
debug/sdsc_11_mul/senprog.txt:
  HBM=0
  L3_LDU=0
  L3_STU=0
  LX_LDSTU=64

debug/sdsc_17_add/senprog.txt:
  HBM=0
  L3_LDU=0
  L3_STU=0
  LX_LDSTU=64
```

The new score-scale edge is not realized yet.  It is now a better edge than the
old pre-matmul Q/K scaling, but it is still `batchmatmul -> mul`, and the
generated scalar-mul consumer presents the transposed pointwise layout:

```text
3_batchmatmul OUTPUT:
  layout=[mb, x, out]
  stick=[out]

4_mul input/output pointwise layout:
  layout=[out, mb, x]
  stick=[x]
```

That remains a layout boundary, so the Tier 1 pass correctly fails closed.

## Interpretation

This stage removes a real lowering artifact that made the Stage 014 blocker
harder than necessary.  Flash-prefill now exposes the SDPA score scale at the
place where a production implementation wants it: immediately after the score
matmul and before the online-softmax reductions.

This is still not full on-chip flash attention.  The next production-shaped
step is to make the score-scale edge legal, either by lowering the scalar score
multiply in the score matmul's physical layout or by introducing a certified
same-tile PT-to-SFP/LX bridge for the `batchmatmul -> mul` boundary.
