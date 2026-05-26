# Stage 021: On-Chip SDPA Sweep Harness

Date: 2026-05-26

## Purpose

Stage 020 made the flash score-scale handoff value-correct by using a PT-shaped
producer LX endpoint.  Stage 021 adds a repeatable benchmark/sweep harness and
uses it to compare three SDPA variants:

```text
vanilla   : stock Torch-Spyre SDPA
flash_hbm : flash-prefill decomposition, HBM-backed handoffs
onchip    : flash-prefill + score-scale handoff + SFP pointwise handoffs
```

The goal is to distinguish two questions:

1. Does the on-chip handoff improve the current flash-prefill variant?
2. Does the current flash-prefill on-chip variant beat stock Spyre SDPA?

The answer is currently: yes for the first question at longer sequence length,
no for the second question on the measured shapes.

## Harness

Added:

```text
tools/onchip_sdpa_sweep.py
```

The parent process launches one child process per shape/variant.  This keeps
Torch-Spyre config and runtime state isolated across variants.  Each child:

- sets a fresh `TORCHINDUCTOR_CACHE_DIR`;
- compiles `torch.nn.functional.scaled_dot_product_attention`;
- checks value correctness against PyTorch CPU;
- warms up the compiled Spyre function;
- times repeated compiled calls with `torch.spyre.synchronize()` before stopping
  the timer;
- scans generated SDSC JSON for mixed `datadscs_`/`opFuncsUsed_`;
- scans `debug/**/senprog.txt` for HBM/L3/LX token counts.

Variant flags:

```text
vanilla:
  SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=0
  SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=0
  SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF=0

flash_hbm:
  SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
  SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=0
  SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF=0

onchip:
  SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
  SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF=1
  SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF=1
  SPYRE_ONCHIP_HANDOFF_MIN_BYTES=0
```

Example command:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128,256 \
  --variants vanilla,flash_hbm,onchip \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --warmup 2 --iters 5 \
  --timeout-s 120 \
  --cache-prefix /tmp/sdpa-stage021 \
  --output-json /tmp/sdpa-stage021-sweep.json
```

## Results

All rows below were value-correct against the PyTorch CPU reference.

| Shape | Variant | Median ms | Mean ms | Max error | Mixed SDSCs |
|---|---|---:|---:|---:|---:|
| B1 H2 L128 D64 | vanilla | 0.125904 | 0.126915 | 0.00537109 | 0 |
| B1 H2 L128 D64 | flash_hbm | 0.259530 | 0.259028 | 0.00341797 | 0 |
| B1 H2 L128 D64 | onchip | 0.286888 | 0.287437 | 0.00341797 | 4 |
| B1 H2 L256 D64 | vanilla | 0.149595 | 0.149314 | 0.00268555 | 0 |
| B1 H2 L256 D64 | flash_hbm | 0.356656 | 0.358930 | 0.00292969 | 0 |
| B1 H2 L256 D64 | onchip | 0.367679 | 0.365850 | 0.00292969 | 10 |
| B1 H2 L512 D64 | vanilla | 0.223035 | 0.227002 | 0.00231934 | 0 |
| B1 H2 L512 D64 | flash_hbm | 0.749094 | 0.732551 | 0.00244141 | 0 |
| B1 H2 L512 D64 | onchip | 0.747921 | 0.744850 | 0.00244141 | 22 |
| B1 H2 L1024 D64 | vanilla | 0.550508 | 0.561067 | 0.00292969 | 0 |
| B1 H2 L1024 D64 | flash_hbm | 1.638273 | 1.645297 | 0.00134277 | 0 |
| B1 H2 L1024 D64 | onchip | 1.484521 | 1.484400 | 0.00134277 | 45 |
| B1 H8 L256 D128 | vanilla | 0.320449 | 0.324962 | 0.00341797 | 0 |
| B1 H8 L256 D128 | flash_hbm | 0.765428 | 0.760833 | 0.00323486 | 0 |
| B1 H8 L256 D128 | onchip | 1.142887 | 1.546234 | 0.00323486 | 10 |

Median speedups:

| Shape | On-chip vs flash_hbm | On-chip vs vanilla |
|---|---:|---:|
| B1 H2 L128 D64 | 0.905x | 0.439x |
| B1 H2 L256 D64 | 0.970x | 0.407x |
| B1 H2 L512 D64 | 1.002x | 0.298x |
| B1 H2 L1024 D64 | 1.104x | 0.371x |
| B1 H8 L256 D128 | 0.670x | 0.280x |

## Descriptor Evidence

The on-chip rows emitted mixed SDSCs.  Representative `B1 H2 L1024 D64` cache:

```text
/tmp/sdpa-stage021-L1024-onchip-B1-H2-L1024-D64-502107-185673
```

Summary:

```text
mixed SDSCs: 45
score-scale mul examples: sdsc_4_mul, sdsc_5_mul
later pointwise examples: sdsc_11_mul, sdsc_17_add
```

Representative `senprog.txt` counts:

```text
sdsc_4_mul:
  HBM=0
  L3_LDU=0
  L3_STU=0
  LX_LDSTU=96

sdsc_5_mul:
  HBM=0
  L3_LDU=0
  L3_STU=0
  LX_LDSTU=96

sdsc_11_mul:
  HBM=0
  L3_LDU=0
  L3_STU=0
  LX_LDSTU=64

sdsc_17_add:
  HBM=0
  L3_LDU=0
  L3_STU=0
  LX_LDSTU=64
```

For the small `B1 H2 L128 D64` and `B1 H2 L256 D64` rows, the score-scale
`mul` SDSCs showed `LX_LDSTU=128` and `HBM=0`.  The `B1 H8 L256 D128` row also
emitted mixed SDSCs, but was slower than both baselines.

## Interpretation

The current on-chip SDPA variant is real and value-correct for the measured
shapes.  It now keeps these pieces on chip:

```text
score batchmatmul output -> score-scale scalar mul
selected SFP pointwise edges in the online-softmax/update chain
```

The implementation removes HBM-backed handoffs inside the flash-prefill graph,
and the `L=1024` result shows that this can amortize:

```text
flash_hbm median: 1.638273 ms
onchip median:    1.484521 ms
speedup:          1.104x
```

But the flash-prefill decomposition itself is currently much slower than stock
Spyre SDPA on the measured shapes.  The stock `vanilla` path remains faster:

```text
B1 H2 L1024 D64 vanilla median: 0.550508 ms
B1 H2 L1024 D64 onchip median:  1.484521 ms
```

So the next production task is not simply "turn on on-chip handoffs by default."
The next task is to reduce the flash-prefill decomposition overhead or target a
shape where its algorithmic structure is actually required.  Until then, the
on-chip variant should stay opt-in and be described as a value-correct
on-chip-flash prototype, not a production replacement for stock SDPA.

## Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
```

Static:

```text
python3 -m py_compile tools/onchip_sdpa_sweep.py
git diff --check
```

Pod command groups:

```text
/tmp/sdpa-stage021-sweep.json
/tmp/sdpa-stage021-sweep-L512.json
/tmp/sdpa-stage021-sweep-typical-L256.json
/tmp/sdpa-stage021-sweep-L1024.json
/tmp/sdpa-stage021-sweep-L1024-vanilla.json
```

All rows in those JSON files have `status="ok"`.

## Next Step

Use the harness to drive optimization work:

1. Add a graph census mode that explains where flash-prefill spends SDSCs and
   why stock SDPA is faster.
2. Try coarser flash blocks and fewer mixed handoffs to reduce launch/schedule
   overhead.
3. Re-run the sweep for larger production prefill shapes only after the flash
   decomposition overhead is reduced.
