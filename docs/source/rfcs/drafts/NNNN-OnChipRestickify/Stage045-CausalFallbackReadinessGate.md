# Stage 045: Causal Fallback Readiness Gate

Date: 2026-05-27

## Purpose

Stage044 proved the full twenty-row on-chip layout-transform gate, including the
two causal rows, but causal square prefill still constructs its triangular bias
through `aten.triu.default`, which falls back to CPU.  This stage adds an
executable readiness check to the sweep harness:

```text
tools/onchip_sdpa_sweep.py --forbid-fallbacks
```

The normal promotion gate remains unchanged.  The new flag is for targeted
readiness runs that should fail until the causal mask path is device-native.

## Implementation

Updated:

```text
tools/onchip_sdpa_sweep.py
tests/_inductor/test_onchip_sdpa_sweep_logic.py
```

When `--forbid-fallbacks` is passed to the parent, it is forwarded to each child
process.  In the child, `torch_spyre.ops.fallbacks.FallbackWarning` is treated as
an error:

```text
warnings.simplefilter("error", FallbackWarning)
```

Successful result JSON now records:

```text
fallbacks_forbidden: true|false
```

Failure and timeout rows also record both:

```text
is_causal
fallbacks_forbidden
```

This keeps readiness failures self-describing instead of relying on the cache key
suffix.

## Pod Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
```

Logic test:

```sh
"$PYTHON" tests/_inductor/test_onchip_sdpa_sweep_logic.py
```

Result:

```text
tests/_inductor/test_onchip_sdpa_sweep_logic.py 6/6 passed
```

Noncausal readiness command:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128 \
  --variants onchip_master_layout_xform \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --forbid-fallbacks \
  --warmup 1 --iters 1 --timeout-s 480 \
  --cache-prefix /tmp/sdpa-stage045-forbid-fallback-noncausal \
  --output-json /tmp/sdpa-stage045-forbid-fallback-noncausal.json
```

Result:

```text
L=128 onchip_master_layout_xform status=ok median=0.270030ms mean=0.270030ms max_err=0.00341797 mixed=9
```

Causal readiness command:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128 \
  --variants onchip_master_layout_xform \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --is-causal --forbid-fallbacks \
  --warmup 1 --iters 1 --timeout-s 480 \
  --cache-prefix /tmp/sdpa-stage045-forbid-fallback-causal-v2 \
  --output-json /tmp/sdpa-stage045-forbid-fallback-causal-v2.json
```

Expected current result:

```text
L=128 onchip_master_layout_xform status=failed rc=1
STAGE045_CAUSAL_FORBID_RC=1
status failed
is_causal True
fallbacks_forbidden True
returncode 1
FallbackWarning present
aten.triu.default present
falling back to cpu present
```

## Interpretation

`--forbid-fallbacks` is now the causal-mask readiness switch.  It is narrow
enough that the noncausal certified path still passes, and strict enough that
the current causal fallback fails with a clear `aten.triu.default` signature.

The promotion gate should continue to run without this flag until a device-native
causal mask primitive exists.  The success criterion for that backend/compiler
work is simple:

```text
the causal readiness command above exits 0 and records status=ok
```

## Local Validation

```text
tests/_inductor/test_onchip_sdpa_sweep_logic.py 6/6 passed
py_compile(onchip_sdpa_sweep.py, test_onchip_sdpa_sweep_logic.py) passed
git diff --check passed
```

## Next

- implement a backend-supported tiled causal mask or equivalent score-bias path;
- rerun the causal readiness command with `--forbid-fallbacks`; and
- once it passes, decide whether causal square prefill can move from opt-in gate
  coverage toward broader/default enablement.
