# Stage 030: Singleton-Stick Gather Contract Probe

Date: 2026-05-26

## Purpose

Stage 029 showed that a natural flash score layout fails in the layout optimizer
at this shape:

```text
maximum(max_running, block_max)
```

The reduced `block_max` vector carries the surviving sequence coordinate as a
non-stick device coordinate while its stick coordinate is constant zero.  The
consumer wants the same sequence coordinate packed into the stick.  That is the
same structural pattern as:

```python
a.sum(1) + b
```

This stage tested whether we could unlock the natural score layout with a narrow
Inductor restickify rule for:

```text
singleton-stick vector -> packed-stick vector
```

The answer is no with the current Foundation/DXP contract.  The Inductor rule
can emit a plausible `ReStickifyOpHBM`, but DXP rejects the descriptor.

## Candidate Implementation

Prototype commit:

```text
0e928a8 Support singleton-stick reduction restickify
```

Changes:

- `torch_spyre/_inductor/pass_utils.py`
  - allowed `compute_restickify_target_layout(...)` to build a target layout
    when the input stick coordinate is exactly zero and the requested target
    stick variable appears in exactly one non-stick device coordinate.
- `torch_spyre/_inductor/decompositions.py`
  - re-enabled the natural score layout candidate:

```text
scores: [B, H, Q, K-block]
block_max = amax(scores, dim=-1)
exp_scores = exp(scores - next_max.unsqueeze(-1))
denominator += exp_scores.sum(dim=-1)
output += bmm(exp_scores, value_block)
```

- `torch_spyre/_inductor/config.py`
  - added `SPYRE_FLASH_ATTENTION_NATURAL_SCORE_LAYOUT`, enabled by the on-chip
    SDPA master gate during the probe.
- tests:
  - changed the old unsupported `a.sum(1) + b` case into a positive regression.
  - added a natural-score SDPA building-block test.

## Failure

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
```

Stable failure command:

```sh
export TORCHINDUCTOR_CACHE_DIR=/tmp/stage030-singleton-stick-gather-dxp
export DXP_DEBUG=1
"$PYTHON" -m pytest tests/inductor/test_restickify.py \
  -k "singleton_stick_reduction_pointwise" -q -s
```

Result:

```text
1 failed, 97 deselected
RC=1
CACHE=/tmp/stage030-singleton-stick-gather-dxp
```

DXP abort:

```text
DtException: Could not find any suitable dimension mapping
file /home/adnan-cdx/dt-inductor-mixed/deeptools-onchip-foundation-clean/ddc/ddl/ddl_conversion.cpp line 2493
```

The generated bundle reached DXP and contained the intended inserted restickify:

```text
/tmp/stage030-singleton-stick-gather-dxp/inductor-spyre/sdsc_fused_add_sum_0_i4awe4_k/
  bundle.mlir
  sdsc_0_sum.json
  sdsc_1_ReStickifyOpHBM.json
  sdsc_2_add.json
```

The `ReStickifyOpHBM` descriptor was structurally simple:

```text
op: ReStickifyOpHBM
N_: out_=128
labeledDs_:
  Tensor0 dsType_=OUTPUT memOrg={hbm,lx}
  Tensor1 dsType_=OUTPUT memOrg={hbm,lx}
```

but Foundation's DDL conversion could not map its dimensions.

## Revert

The candidate was reverted:

```text
59b1086 Revert "Support singleton-stick reduction restickify"
```

That removed the natural-score master gate and restored the explicit unsupported
state for singleton-stick gather pointwise.

## Current-Branch Validation

After the revert:

```text
HEAD: 59b1086
```

Static/config validation:

```text
tests/_inductor/test_config_logic.py  4/4 passed
py_compile(pass_utils.py, config.py, decompositions.py, test_config_logic.py,
           test_restickify.py, test_building_blocks.py) passed
git diff --check passed
```

Current on-chip SDPA master smoke:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 512 \
  --variants onchip_master \
  --batch 1 --heads 2 --dim 64 --block-size 0 \
  --warmup 1 --iters 2 \
  --timeout-s 180 \
  --cache-prefix /tmp/sdpa-stage030-post-revert-smoke \
  --output-json /tmp/sdpa-stage030-post-revert-smoke.json
```

Result:

```text
status=ok
block_size=512
block_size_env=""
mixed=0
max_abs_error=0.001953125
median=0.538825 ms
```

## Interpretation

The Stage 029 natural-score failure is not just an Inductor candidate-layout
omission.  If Inductor emits the needed singleton-stick gather restickify, DXP
rejects the descriptor.  That makes this a Foundation/DXP contract gap today.

Near-term implication:

```text
Do not enable natural score layout under SPYRE_FLASH_ATTENTION_ONCHIP_SDPA.
```

The production-safe master gate remains the Stage 028 serial on-chip handoff
path with block size 512.

## Next Step

There are two viable next implementation paths:

```text
1. Foundation path:
   Add/certify DXP support for singleton-stick gather restickify:
     [d0, d1, 0, ..., stick=0] -> [d0, floor(d1/64), ..., stick=d1%64]

2. Compiler path:
   Avoid exposing this intermediate as a standalone tensor by introducing a
   fused flash update lowering where max/sum/update stay inside one operator.
```

Given current DXP behavior, the compiler-only natural-layout route should stay
fail-closed until one of those two paths exists.
