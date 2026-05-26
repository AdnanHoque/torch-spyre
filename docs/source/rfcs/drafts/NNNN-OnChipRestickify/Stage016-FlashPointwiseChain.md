# Stage 016: Flash Pointwise Chain Realization

Date: 2026-05-26

## Purpose

Stage 015 proved one real flash-prefill pointwise value-flow handoff:

```text
2_add -> 10_mul
```

After that first edge is realized, the generated flash graph exposes another
legal Tier 1 pointwise edge:

```text
10_mul -> 16_add
```

Stage 016 changes the flash-specific pointwise handoff gate to realize every
legal same-layout pointwise edge in the bundle, rather than stopping after the
first one.  The broader generic `SPYRE_ONCHIP_HANDOFF_REALIZE` path still keeps
its one-edge behavior.

## Implementation

Code changes:

- `torch_spyre/_inductor/onchip_realize.py`
  - Added `detect_pointwise_handoff`, which performs the full legality check
    before returning an edge.
  - Added `realize_pointwise_handoff`.
  - Added `realize_flash_attention_pointwise_handoffs`, which repeatedly applies
    legal pointwise handoffs until no further eligible edge remains.
  - Kept layout-changing, multi-split, fanout, and non-pointwise edges
    fail-closed.
- `torch_spyre/_inductor/codegen/bundle.py`
  - `SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=1` now calls the flash chain
    realizer.
  - `SPYRE_ONCHIP_HANDOFF_REALIZE=1` still calls the original one-edge realizer.
- `tests/_inductor/test_onchip_realize_logic.py`
  - Added a synthetic three-op pointwise chain test proving two handoffs are
    realized in order.

## Edge Classification

Using the Stage 015 cache, the remaining pointwise candidates were classified.
Most edges are intentionally rejected for one of these reasons:

- producer or consumer is not pointwise (`batchmatmul`, `ReStickifyOpHBM`,
  `maxnonstick`, `sumnonstick`);
- score softmax edges use multi-split `{mb: 2, out: 2}`;
- producer/consumer shards differ;
- physical layouts differ.

After `2_add -> 10_mul` is realized, this edge is eligible:

```text
10_mul.2 -> 16_add.0
layout=[mb_, x_, out_]
stick=x_
split=out_
slice=262144
```

## Validation

Local:

```text
tests/_inductor/test_onchip_realize_logic.py         22/22 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   9/9 passed
tests/_inductor/test_onchip_streaming_logic.py        9/9 passed
tests/_inductor/test_onchip_handoff_logic.py          3/3 passed
py_compile(onchip_realize.py, bundle.py)              passed
git diff --check                                      passed
```

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
commit=80c0289
```

Pod standalone tests:

```text
tests/_inductor/test_onchip_realize_logic.py         22/22 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   9/9 passed
tests/_inductor/test_onchip_streaming_logic.py        9/9 passed
tests/_inductor/test_onchip_handoff_logic.py          3/3 passed
```

Device command:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=1
export SPYRE_ONCHIP_HANDOFF_MIN_BYTES=0
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-flash-pointwise-chain-1779820739
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

Result:

```text
1 passed, 6 deselected in 12.40s
```

Realized mixed SDSCs:

```text
bundle: sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_r373_x5m
mixed:  sdsc_10_mul.json
mixed:  sdsc_16_add.json
```

Descriptor evidence:

```text
sdsc_2_add Tensor2 OUTPUT:
  hbmSize_=0
  lxSize_=2147483647

sdsc_10_mul Tensor0 input:
  hbmSize_=0
  lxSize_=2147483647

sdsc_10_mul Tensor2 OUTPUT:
  hbmSize_=0
  lxSize_=2147483647

sdsc_16_add Tensor0 input:
  hbmSize_=0
  lxSize_=2147483647
```

Both mixed consumers use the real descriptor geometry:

```text
op=STCDPOpLx
layoutDimOrder_=[mb_, x_, out_]
stickDimOrder_=[x_]
dimToLayoutSize_={mb_: 2, x_: 128, out_: 64}
```

DXP debug evidence:

```text
debug/sdsc_10_mul/senprog.txt:
  HBM=0
  L3_LDU=0
  L3_STU=0
  LX_LDSTU=64

debug/sdsc_16_add/senprog.txt:
  HBM=0
  L3_LDU=0
  L3_STU=0
  LX_LDSTU=64
```

As in Stage 015, these are local same-owner pointwise handoffs, so zero L3 is
expected.  The important proof is value correctness plus HBM-free mixed data-op
execution on real generated flash-prefill SDSCs.

## Interpretation

The flash-prefill path now has a real two-edge on-chip value-flow chain:

```text
2_add output LX -> STCDPOpLx -> 10_mul input LX
10_mul output LX -> STCDPOpLx -> 16_add input LX
```

This still is not full on-chip flash attention.  The layout-changing
`batchmatmul` inputs from Stage 014 remain fail-closed.  The next meaningful
production step is still to make the Q/K/V `batchmatmul` tile feeds legal for
Tier 1 or to port a certified PT-LX layout bridge into this branch.
