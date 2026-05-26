# Stage 029: Natural Score Layout Probe

Date: 2026-05-26

## Purpose

Stage 028 showed that the current production-candidate on-chip SDPA gate is
value-correct, but still slower than stock Spyre SDPA because the flash-prefill
decomposition has high graph and launch overhead.  This stage checks two
lower-risk compiler policy ideas before returning to deeper mixed-SDSC work:

1. Does an even larger block size beat the Stage 028 block-512 default?
2. Can the flash-prefill decomposition avoid the score transpose/restickify path
   by keeping scores in their natural `[B, H, Q, K-block]` layout?

The answer is no for both as production changes today.  Block 1024 is
value-correct but does not beat block 512 for the long target.  Natural score
layout fails in the current layout optimizer before DXP.

## Block-1024 Sweep

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
```

Command:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 512,1024 \
  --variants flash_hbm,onchip_master \
  --batch 1 --heads 2 --dim 64 --block-size 1024 \
  --warmup 2 --iters 5 \
  --timeout-s 300 \
  --cache-prefix /tmp/sdpa-stage029-bs1024 \
  --output-json /tmp/sdpa-stage029-bs1024.json
```

Results:

| L | Variant | Block | Median ms | Max error | Mixed SDSCs |
|---:|---|---:|---:|---:|---:|
| 512 | flash_hbm | 1024 | 0.481447 | 0.00195312 | 0 |
| 512 | onchip_master | 1024 | 0.497742 | 0.00195312 | 0 |
| 1024 | flash_hbm | 1024 | 1.187956 | 0.00134277 | 0 |
| 1024 | onchip_master | 1024 | 1.186160 | 0.00134277 | 0 |

Interpretation:

- Block 1024 is value-correct.
- It emits no mixed SDSCs for these shapes, so the master path degenerates to
  the flash-HBM path.
- It does not beat the Stage 028 block-512 `L=1024` on-chip result:

```text
block 512 onchip_master L=1024:  1.160441 ms, mixed=2
block 1024 onchip_master L=1024: 1.186160 ms, mixed=0
```

So the Stage 028 default remains block 512.

## Natural Score Layout Candidate

The attempted compiler change was:

```text
scores: [B, H, Q, K-block]
block_max = amax(scores, dim=-1)
exp_scores = exp(scores - next_max.unsqueeze(-1))
denominator += exp_scores.sum(dim=-1)
output += bmm(exp_scores, value_block)
```

This is mathematically equivalent to the current decomposition and would avoid
the final `exp_scores.transpose(-1, -2)` before the PV `bmm`.  The first attempt
also let the master gate enable:

```text
SPYRE_FLASH_ATTENTION_NATURAL_SCORE_LAYOUT
```

That commit was pushed as a probe, then reverted after validation failed:

```text
e1bfee5 Use natural score layout for on-chip SDPA
6dd29b6 Normalize natural score reductions
df647be Revert "Normalize natural score reductions"
2a7a4f8 Revert "Use natural score layout for on-chip SDPA"
```

## Failure

Device-side building-block validation failed during Inductor layout selection,
before DXP:

```sh
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "natural_score_layout" -q -s
```

The important error was:

```text
NotImplementedError: buf9 (Pointwise): no mechanism to resolve stick incompatibility

Inputs:
  buf3: size=[1, 2, 128]
    STL 0: device_size=[2, 2, 1, 64]
           d_coords=[d0, floor(d1/64), 0, Mod(d1, 64)]

  buf8: size=[1, 2, 128]
    STL 0: device_size=[2, 128, 1, 1, 64]
           d_coords=[d0, d1, 0, 0, 0]

Output:
  size=[1, 2, 128]
    STL 0: device_size=[2, 2, 1, 64]
           d_coords=[d0, floor(d1/64), 0, Mod(d1, 64)]

Problem:
  buf8 STL 0 --> Out STL 0:
  No mechanism to gather elements from multiple sticks into single stick
```

Adding `contiguous()` to the reduced vectors did not change the layout chosen
for the failing edge.  The failure is at:

```text
next_max = maximum(max_running, block_max)
```

where reducing over the natural last dimension produces a vector layout that
cannot currently be combined with the running max layout.

## Current-Branch Validation After Revert

After reverting the probe, the current branch returned to the proven Stage 028
path:

```text
HEAD after revert: 2a7a4f8
```

Validation:

```text
tests/_inductor/test_config_logic.py  4/4 passed
py_compile(config.py, decompositions.py, test_config_logic.py, test_building_blocks.py) passed
git diff --check passed
```

Current master smoke:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 512 \
  --variants onchip_master \
  --batch 1 --heads 2 --dim 64 --block-size 0 \
  --warmup 1 --iters 2 \
  --timeout-s 180 \
  --cache-prefix /tmp/sdpa-stage029-post-revert-smoke \
  --output-json /tmp/sdpa-stage029-post-revert-smoke.json
```

Result:

```text
status=ok
variant=onchip_master
shape=B1 H2 L512 D64
block_size=512
block_size_env=""
median=0.497564 ms
max_err=0.00195312
mixed=0
```

## Interpretation

Natural score layout is not a valid production path with the current layout
solver.  It needs either:

```text
1. a restickify/layout rule that can normalize the natural reduction vector
   into the running max/denominator layout, or

2. a different fused lowering where max/sum/update stay in one operator and do
   not expose the incompatible intermediate vector as a standalone pointwise
   input.
```

This reinforces the Stage 028 conclusion: the next useful implementation should
not add more standalone pointwise edges.  It should either change the layout
solver contract for this reduction pattern or introduce a fused flash update
lowering that avoids materializing the problematic intermediate layout.
