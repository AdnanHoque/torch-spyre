# Stage 020: Flash Score-Scale PT Endpoint

Date: 2026-05-26

## Purpose

Stage 019 proved that the compiler can express the flash score-scale edge as a
same-layout mixed SDSC:

```text
batchmatmul score output -> STCDPOpLx -> scalar mul input
```

but the first device run was value-incorrect.  The failed path used the generic
SFP producer flip contract:

```text
producer LX base: 16384
consumer LX base: 278528
producer coreStateInit_: injected
producer numCoreletsUsed_DSC2_: unchanged
```

The older first-principles PT-LX stages had a different successful PT endpoint
contract: allocator-shaped non-overlapping bases starting at `0`, explicit
producer corelet count, and no producer-side `coreStateInit_` injection.  Stage
020 applies that PT-shaped endpoint contract to the flash score-scale edge.

## Code Change

`apply_lx_flip` now accepts two optional controls:

```text
core_state_init: whether to inject coreStateInit_ on the flipped LDS
num_corelets: optional numCoreletsUsed_/numCoreletsUsed_DSC2_ override
```

The normal SFP-to-SFP pointwise handoffs keep the old behavior.  The flash
score-scale handoff now calls the shared realizer with:

```text
region0 = 0
producer_core_state_init = False
producer_num_corelets = 1
```

The consumer scalar `mul` still receives `coreStateInit_`, because it is the
consumer-side LX input endpoint for the mixed SDSC.

## Endpoint Evidence

Generated focused SDPA run:

```text
TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-flash-score-scale-test-method-1779823370
```

The producer `batchmatmul` score output is now:

```text
sdsc_3_batchmatmul Tensor2 OUTPUT
numCoreletsUsed_ = 1
numCoreletsUsed_DSC2_ = 1
hbmSize_ = 0
lxSize_ = 2147483647
coreStateInit_ = absent
allocate Tensor2 = lx @ 0
```

The score-scale mixed `mul` uses the matching bridge:

```text
sdsc_4_mul:
  mixed=True
  opFuncs=['STCDPOpLx']
  dataIN  layout=['mb_', 'x_', 'out_'] stick=['out_'] startAddr=[0]
  dataOUT layout=['mb_', 'x_', 'out_'] stick=['out_'] startAddr=[262144]
```

The second flash block has the same shape as `sdsc_5_mul`.

## Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
```

Unit and static checks:

```text
tests/_inductor/test_onchip_realize_logic.py
tests/_inductor/test_onchip_flash_pipeline_logic.py
tests/_inductor/test_onchip_streaming_logic.py
tests/_inductor/test_onchip_handoff_logic.py

45 passed in 0.18s

py_compile(superdsc.py, onchip_realize.py, decompositions.py, config.py,
           bundle.py, test_building_blocks.py) passed
```

Focused SDPA score-scale handoff test:

```sh
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-flash-score-scale-test-method-1779823370
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "score_scale_handoff" -q -s
```

Result:

```text
1 passed, 7 deselected in 12.44s
```

Manual larger-shape smoke:

```text
B=1, H=2, L=256, D=64
block_size=64
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=True
SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=True
SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF=True
```

Result:

```text
ok shape (1, 2, 256, 64)
```

DXP debug counts from both the focused test and the `L=256` smoke show the
score-scale `mul` SDSCs executing HBM-free mixed data-ops:

```text
sdsc_4_mul:
  HBM=0
  L3_LDU=0
  L3_STU=0
  LX_LDSTU=128

sdsc_5_mul:
  HBM=0
  L3_LDU=0
  L3_STU=0
  LX_LDSTU=128
```

The later proven SFP pointwise handoffs remain present:

```text
sdsc_11_mul:
  HBM=0
  LX_LDSTU=64

sdsc_17_add:
  HBM=0
  LX_LDSTU=64
```

One default-off rerun was attempted after the larger smoke, but the process
stalled inside the runtime H2D artifact load path and was terminated after more
than two minutes.  That run is recorded as inconclusive rather than validation
evidence.  The same default-off path had passed earlier in Stage 019 before the
PT endpoint change.

## Interpretation

The Stage 019 failure was not because the score-scale descriptor labels were
wrong.  It was because the generic SFP producer flip is not a valid PT producer
endpoint contract.

With the PT-shaped endpoint contract, the score-scale handoff is value-correct
for the focused SDPA test and for a `L=256` smoke case.  This moves the on-chip
attention variant from "later SFP pointwise chain only" to:

```text
score batchmatmul output
  -> on-chip STCDPOpLx score-scale handoff
  -> on-chip SFP pointwise chain handoffs
```

The path remains gated by `SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF=1` until a
broader shape/performance sweep is complete.

## Next Step

Run a structured SDPA sweep across typical prefill sizes and collect:

- value correctness;
- mixed SDSC count;
- `senprog.txt` HBM/LX/L3 signatures;
- latency versus vanilla Torch-Spyre SDPA.

If the sweep holds, promote the score-scale handoff from diagnostic gate to the
default production on-chip SDPA variant flag.
