# Stage 022: Flash Block Size Sweep and Fail-Closed Score-Scale Gate

Date: 2026-05-26

## Purpose

Stage 021 showed that the current on-chip SDPA prototype can improve the
flash-prefill HBM path at longer sequence length, but it still loses badly to
stock Spyre SDPA.  The next obvious tuning knob is the flash prefill block size:
larger blocks reduce the number of flash chunks and mixed SDSCs, but they also
change the score tensor geometry used by the PT batchmatmul -> SFP score-scale
handoff.

This stage answers two questions:

1. How does block size affect the current flash HBM and on-chip variants?
2. Which score-scale geometries are actually value-correct today?

## Implementation

Added a harness variant:

```text
pointwise_only:
  SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
  SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=1
  SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF=0
  SPYRE_ONCHIP_HANDOFF_MIN_BYTES=0
```

This isolates the later SFP pointwise-chain handoffs from the earlier PT
batchmatmul score output -> scalar SFP mul score-scale handoff.

Added a conservative realizer cap:

```text
FLASH_SCORE_SCALE_MAX_STICK_ELEMS = 128
```

`detect_flash_score_scale_handoff` now rejects score-scale candidates whose
consumer stick iteration size is wider than 128 elements.  The wider geometry
falls back to the existing HBM-backed score-scale path while the later SFP
pointwise handoffs can still be realized on chip.

## Results

All rows use:

```text
pod: adnan-cdx-spyre-dev-pf
shape: B1 H2 L1024 D64
warmup: 1
iters: 3
reference: PyTorch CPU
```

| Block size | Variant | Median ms | Mean ms | Max error | Mixed SDSCs | Status |
|---:|---|---:|---:|---:|---:|---|
| 64 | flash_hbm | 1.826894 | 4.169785 | 0.00134277 | 0 | ok |
| 64 | onchip | 1.470972 | 1.473971 | 0.00134277 | 45 | ok |
| 128 | flash_hbm | 1.428585 | 1.445275 | 0.00134277 | 0 | ok |
| 128 | onchip | 1.312600 | 1.321387 | 0.00134277 | 21 | ok |
| 256 | flash_hbm | 1.243677 | 1.234565 | 0.00134277 | 0 | ok |
| 256 | onchip, before cap | n/a | n/a | 3.35546875 | n/a | failed |
| 256 | pointwise_only | 1.365727 | 1.374226 | 0.00183105 | 6 | ok |
| 256 | onchip, after cap | 1.253195 | 1.253288 | 0.00134277 | 6 | ok |
| 512 | flash_hbm | 1.161441 | 1.169349 | 0.00134277 | 0 | ok |
| 512 | onchip, after cap | 1.150027 | 1.155373 | 0.00134277 | 2 | ok |

Median speedups against the flash HBM variant:

| Block size | On-chip state | Speedup |
|---:|---|---:|
| 64 | score-scale + pointwise | 1.242x |
| 128 | score-scale + pointwise | 1.088x |
| 256 | fail-closed score-scale, pointwise only | 0.992x |
| 512 | fail-closed score-scale, pointwise only | 1.010x |

The Stage 021 stock Spyre SDPA result for this same shape was:

```text
vanilla median: 0.550508 ms
```

So even the best current on-chip flash-prefill row remains much slower than the
stock SDPA path:

```text
best on-chip flash row: 1.150027 ms
relative to vanilla:   0.479x
```

## Failure Analysis

The uncapped BS256 on-chip run produced value corruption:

```text
Mismatched elements: 48266 / 131072 (36.8%)
Greatest absolute difference: 3.35546875 at index (0, 1, 940, 2)
```

The same BS256 shape with `pointwise_only` passed, and the patched BS256
`onchip` run also passed with the same six mixed pointwise SDSCs.  That isolates
the correctness problem to the score-scale PT->SFP handoff for the 256-wide
score block, not to the later pointwise-chain handoffs.

The patched BS256 mixed SDSCs were:

```text
sdsc_11_mul
sdsc_17_add
sdsc_26_mul
sdsc_32_add
sdsc_41_mul
sdsc_47_add
```

The previously unsafe score-scale SDSCs (`sdsc_4_mul`, `sdsc_5_mul`,
`sdsc_20_mul`, and similar score-block scale ops) no longer receive
`STCDPOpLx` data descriptors at BS256 or BS512.

## Interpretation

Block-size tuning reduces flash-prefill overhead.  Moving from BS64 to BS512
cuts the flash HBM median from `1.826894 ms` to `1.161441 ms`.  It also reduces
the number of mixed on-chip SDSCs from 45 to 2.

The score-scale handoff is currently certified only through 128-wide score
blocks.  Wider score blocks must fail closed until we understand and certify the
PT endpoint geometry.  The fail-closed behavior preserves value correctness and
still allows the later SFP pointwise-chain handoffs to use LX.

This changes the near-term production stance:

```text
score-scale handoff:
  enabled only for <=128-wide score blocks

larger flash blocks:
  keep score-scale on HBM
  keep eligible SFP pointwise handoffs on chip
```

The broader performance conclusion is unchanged from Stage 021: the mixed SDSC
primitive is working and useful, but the current flash-prefill decomposition is
not yet competitive with stock Spyre SDPA.  The next optimization needs to
attack decomposition/SDSC overhead or implement a more fused flash structure,
not simply add more same-layout handoffs.

## Validation

Focused unit/static validation:

```text
python3 -m pytest tests/_inductor/test_onchip_realize_logic.py -q
25 passed in 0.23s

python3 -m py_compile \
  tools/onchip_sdpa_sweep.py \
  torch_spyre/_inductor/onchip_realize.py \
  torch_spyre/_inductor/config.py
```

Device command examples:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 1024 \
  --variants flash_hbm,onchip \
  --batch 1 --heads 2 --dim 64 \
  --block-size 512 \
  --warmup 1 --iters 3 \
  --timeout-s 300 \
  --cache-prefix /tmp/sdpa-stage022-bs512-patched \
  --output-json /tmp/sdpa-stage022-bs512-patched.json
```

Key artifact locations:

```text
/tmp/sdpa-stage022-bs64.json
/tmp/sdpa-stage022-bs128.json
/tmp/sdpa-stage022-bs256-paired.json
/tmp/sdpa-stage022-bs512-patched.json
/tmp/sdpa-stage022-bs256-pointwiseonly-1779825046
```

## Next Step

Use BS512 as the current flash-prefill baseline when measuring graph/SDSC
overhead, but keep score-scale handoff certification at BS128.  The next
production-shaped design should explore a more fused or overlapped flash
pipeline so the mixed SDSC mechanism removes movement without multiplying the
number of standalone SDSC launches.
