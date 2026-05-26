# Stage 019: Flash Score-Scale Handoff Gate

Date: 2026-05-26

## Purpose

Stage 018 showed that the flash-prefill score-scale edge still failed Tier 1
matching because the generated scalar `mul` appeared to change both
layout/stick and core ownership relative to the score `batchmatmul`.

Stage 019 tested the narrower hypothesis that this was partly a descriptor-label
problem: the score tensor is logically four-dimensional, but the leading batch
dimension is size 1 and the scalar pointwise SuperDSC is emitted as a 3D op.  The
generic pointwise axis labels therefore made the score `mul` look like it used a
different physical layout even when its logical shape and strides matched the
`batchmatmul` output.

## Implementation

The compiler now preserves the score-scale axis identity for the flash score
`mul` case:

```text
batchmatmul score output: layout=[mb,x,out] stick=[out] split=mb
score-scale mul input:   layout=[mb,x,out] stick=[out] split=mb
```

This is implemented in `codegen/superdsc.py` by recognizing scalar score-scale
pointwise ops and using `["x", "out", "mb"]` as the pre-reversal 3D label order,
which emits the physical `["mb", "x", "out"]` order expected by the score
producer.

The on-chip realizer also gained a flash-specific detector for:

```text
batchmatmul -> scalar mul
same layout, same stick, same shard
single future consumer
```

Unit coverage proves that this edge can be lowered into a mixed SDSC with one
`STCDPOpLx` data-op when explicitly requested.

## Device Result

The diagnostic path was then run on the pod:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=1
export SPYRE_ONCHIP_HANDOFF_MIN_BYTES=0
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-flash-score-scale-handoff-1779822591
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

This emitted mixed score-scale SDSCs:

```text
bundle 0:
  sdsc_4_mul mixed=True opFuncs=['STCDPOpLx'] datadscs=1

bundle 1:
  sdsc_5_mul mixed=True opFuncs=['STCDPOpLx'] datadscs=1
  sdsc_11_mul mixed=True opFuncs=['STCDPOpLx'] datadscs=1
  sdsc_17_add mixed=True opFuncs=['STCDPOpLx'] datadscs=1
```

The generated score-scale data-op used the intended same-stick frame:

```text
0_STCDPOpLx_dataop:
  layoutDimOrder_ ['mb_', 'x_', 'out_']
  stickDimOrder_  ['out_']
  dimToLayoutSize_ {'mb_': 128, 'x_': 2, 'out_': 64}
  hbm 0
  lx  2097152
```

DXP compiled and executed the bundle, but device value correctness failed:

```text
75.7% elements mismatched
max abs diff 6.8359375
```

This is an important negative result.  The descriptor-level match is now
possible, but direct PT `batchmatmul` score output -> SFP scalar `mul` LX value
flow is not certified yet.  The remaining bug is either in the PT/SFP LX
contract for this edge or in the exact STCDP geometry/addressing needed for
PT-produced score tiles.

## Safety Gate

The score-scale realization is now separated from the working flash pointwise
handoff flag:

```text
SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=1
SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF=1
```

`SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF` defaults to off.  With only
`SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=1`, the compiler still realizes the
later SFP pointwise chain handoffs, but leaves the unsafe score-scale
`batchmatmul -> mul` edge HBM-backed.

This keeps the useful detector and synthetic mixed-SDSC proof in tree while
failing closed for production-shaped device execution.

## Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
```

Unit checks:

```text
tests/_inductor/test_onchip_realize_logic.py      24 passed in 0.33s
tests/_inductor/test_onchip_flash_pipeline_logic.py 9 passed in 0.06s
tests/_inductor/test_onchip_streaming_logic.py     9 passed in 0.17s
tests/_inductor/test_onchip_handoff_logic.py       3 passed in 0.04s
py_compile(superdsc.py, onchip_realize.py, decompositions.py, config.py, bundle.py) passed
```

Focused device run with the score-scale gate unset:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=1
unset SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF
export SPYRE_ONCHIP_HANDOFF_MIN_BYTES=0
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-flash-score-scale-default-off-1779822870
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

Result:

```text
1 passed, 6 deselected in 14.03s
```

Generated SDSC evidence from the passing run:

```text
sdsc_4_mul  mixed=False opFuncs=[] datadscs=0
sdsc_5_mul  mixed=False opFuncs=[] datadscs=0
sdsc_11_mul mixed=True  opFuncs=['STCDPOpLx'] datadscs=1
sdsc_17_add mixed=True  opFuncs=['STCDPOpLx'] datadscs=1
```

`sdsc_11_mul` and `sdsc_17_add` retain the proven SFP pointwise on-chip
handoffs:

```text
dataop=0_STCDPOpLx_dataop
layout=['mb_', 'x_', 'out_']
stick=['x_']
hbm=0
lx=2097152
```

## Conclusion

Stage 019 turns the score-scale handoff into a controlled diagnostic rather than
a production path.  The compiler can now express the desired mixed SDSC, but the
device result says the PT-produced score tile cannot yet be trusted as an SFP
consumer input through the current STCDP bridge.

The next useful production step is to keep expanding proven SFP-to-SFP
pointwise-chain handoffs inside flash prefill while separately building a
PT-output LX contract test for `batchmatmul -> scalar mul`.
