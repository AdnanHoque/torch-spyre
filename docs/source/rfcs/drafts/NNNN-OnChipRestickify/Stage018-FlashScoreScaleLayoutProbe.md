# Stage 018: Flash Score-Scale Layout Probe

Date: 2026-05-26

## Purpose

Stage 017 moved SDPA scaling from pre-matmul Q/K producers to the score tensor:

```text
scores = Q @ K.T
scores = scores * scale
scores = scores.transpose(-1, -2).contiguous()
```

That removed the old pre-score `mul -> batchmatmul` blockers, but the new
score-scale edge still did not become a Tier 1 same-layout handoff.  Stage 018
tests whether simple lowering barriers can make the score `mul` materialize in
the `batchmatmul` output layout.

## Experiments

Three variants were tested on the pod.

### Variant A: Scale after transpose

```text
scores = Q @ K.T
scores = scores.transpose(-1, -2).contiguous()
scores = scores * scale
```

This is mathematically equivalent, but generated a worse score path:

```text
3_batchmatmul
4_ReStickifyOpHBM
5_mul
```

The score-scale `mul` did not become mixed.  Its `senprog.txt` showed:

```text
sdsc_5_mul:
  HBM=0
  L3_LDU=0
  L3_STU=0
  LX_LDSTU=128
```

The Stage 017 shape had the corresponding score `mul` at `LX_LDSTU=64`, so this
variant was not kept.

### Variant B: `contiguous()` after score scale

```text
scores = Q @ K.T
scores = (scores * scale).contiguous()
scores = scores.transpose(-1, -2).contiguous()
```

This did not change the generated score layout.  The score `mul` remained:

```text
4_mul:
  layout=[out, mb, x]
  stick=[x]
```

### Variant C: explicit clone after score scale

```text
scores = Q @ K.T
scores = (scores * scale).clone(memory_format=torch.contiguous_format)
scores = scores.transpose(-1, -2).contiguous()
```

`propagate_layouts.py` treats explicit `aten.clone` as a row-major materializing
operation, so this was the stronger barrier attempt.  The generated graph still
matched the Stage 017 shape:

```text
0_identity
1_identity
2_ReStickifyOpHBM
3_batchmatmul
4_mul
5_ReStickifyOpHBM
...
```

The clone did not survive as a separate useful SDSC barrier for this path, but
it is value-correct and does not regress the generated score path relative to
Stage 017.

## Final Code Shape

The branch currently keeps Variant C:

```python
scores = torch.matmul(query, key_block_t)
if score_scale != 1.0:
    scores = (scores * score_scale).clone(
        memory_format=torch.contiguous_format
    )
scores = scores.transpose(-1, -2).contiguous()
```

## Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
commit=8811ad9
```

Static:

```text
py_compile(torch_spyre/_inductor/decompositions.py)  passed
```

Mixed flash, no handoff realization:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-flash-score-scale-clone-1779821771
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

Result:

```text
1 passed, 6 deselected in 13.70s
```

Mixed flash, pointwise handoff realization:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=1
export SPYRE_ONCHIP_HANDOFF_MIN_BYTES=0
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-flash-score-scale-clone-pointwise-1779821944
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

Result:

```text
1 passed, 6 deselected in 14.15s
```

Direct flash-prefill:

```sh
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-flash-score-scale-clone-prefill-1779821992
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_prefill" -q -s
```

Result:

```text
1 passed, 6 deselected in 12.86s
```

## Handoff Evidence

The final pointwise run still realizes the same two later pointwise handoffs:

```text
bundle: sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_dwo524gg
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

The score-scale `mul` remains unmixed:

```text
sdsc_4_mul / sdsc_5_mul:
  mixed=False
  opFuncsUsed_=[]
  datadscs_=0
```

## Diagnosis

The score-scale edge is now cleaner than the original pre-Q/K scaling graph, but
it is not a same-shard Tier 1 pointwise edge.

Generated score producer:

```text
3_batchmatmul:
  layout=[mb, x, out]
  stick=[out]
  shard={x: 1, mb: 32, out: 1, in: 1}
```

Generated score-scale consumer:

```text
4_mul:
  layout=[out, mb, x]
  stick=[x]
  shard={mb: 1, x: 1, out: 32}
```

So the next blocker is twofold:

```text
layout/stick changes: [mb,x,out]/out -> [out,mb,x]/x
core ownership changes: split mb -> split out
```

The second part is important.  This is not merely a local same-core pointwise
handoff; it is a score-tile redistribution from query-row-owned cores to
key-block-owned cores.  The existing core-to-core primitive has the ingredients
for same-stick cross-split movement, but the flash score edge still needs either:

- a lowering/layout-propagation change that keeps the score `mul` in the
  `batchmatmul` score layout/stick, then a same-layout `mb -> out` STCDP bridge;
  or
- a certified PT/SFP-LX bridge that handles the current layout/stick change.

Until one of those is implemented, the pass should keep this edge fail-closed.
