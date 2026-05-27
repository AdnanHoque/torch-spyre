# Stage 046: Promotion-Gate Fallback Readiness

Date: 2026-05-27

## Purpose

Stage045 added `--forbid-fallbacks` to the SDPA sweep harness.  This stage wires
that readiness switch through the promotion-gate wrapper so the same gate tool
can express the final causal-mask requirement:

```text
tools/onchip_sdpa_promotion_gate.py --forbid-fallbacks
```

The default promotion gate remains unchanged.  The new flag is explicit and is
intended for readiness checks before claiming the causal path is device-native.

## Implementation

Updated:

```text
tools/onchip_sdpa_promotion_gate.py
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py
```

`sweep_command` now accepts `forbid_fallbacks` and appends:

```text
--forbid-fallbacks
```

to the child sweep command when requested.  Gate row validation also verifies
that successful rows from a fallback-forbidden run record:

```text
fallbacks_forbidden == True
```

This prevents a stale or non-readiness sweep JSON from satisfying a
fallback-forbidden promotion-gate run.

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
"$PYTHON" tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py
```

Result:

```text
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py 9/9 passed
```

Noncausal gate-level readiness command:

```sh
"$PYTHON" tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --cases b2h2d64_block64 \
  --forbid-fallbacks \
  --case-output-dir /tmp/sdpa-stage046-gate-forbid-noncausal-json \
  --cache-prefix /tmp/sdpa-stage046-gate-forbid-noncausal \
  --timeout-s 700 \
  --output-json /tmp/sdpa-stage046-gate-forbid-noncausal.json
```

Result:

```text
L=128 onchip_master_layout_xform status=ok median=0.319256ms mean=0.319256ms max_err=0.00585938 mixed=7
L=256 onchip_master_layout_xform status=ok median=0.575215ms mean=0.575215ms max_err=0.00317383 mixed=15
PROMOTION_GATE_PASSED gate=onchip_layout_xform cases=1 rows=2
rows 2
statuses ['ok']
fallbacks_forbidden [True]
mixed [7, 15]
```

Causal gate-level readiness command:

```sh
"$PYTHON" tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --cases b1h2d64_block64_causal \
  --forbid-fallbacks \
  --case-output-dir /tmp/sdpa-stage046-gate-forbid-causal-json \
  --cache-prefix /tmp/sdpa-stage046-gate-forbid-causal \
  --timeout-s 700 \
  --output-json /tmp/sdpa-stage046-gate-forbid-causal.json
```

Expected current result:

```text
L=128 onchip_master_layout_xform status=failed rc=1
L=256 onchip_master_layout_xform status=failed rc=1
PROMOTION_GATE_FAILED
STAGE046_GATE_CAUSAL_FORBID_RC=1
rows 2
statuses ['failed', 'failed']
fallbacks_forbidden [True, True]
is_causal [True, True]
FallbackWarning present
aten.triu.default present
falling back to cpu present
```

## Interpretation

The promotion gate can now be used in two modes:

- normal promotion mode, without `--forbid-fallbacks`, which proves value
  correctness and layout-transform sidecar coverage; and
- fallback-readiness mode, with `--forbid-fallbacks`, which proves the selected
  cases do not use Torch-Spyre CPU fallback kernels.

The noncausal readiness case passes today.  The causal readiness case correctly
fails today because square causal prefill still uses `aten.triu.default` to build
the triangular score bias.

The backend/compiler success criterion is now gate-shaped:

```text
tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --cases b1h2d64_block64_causal \
  --forbid-fallbacks
```

must exit 0.

## Local Validation

```text
tests/_inductor/test_onchip_sdpa_sweep_logic.py          6/6 passed
tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py 9/9 passed
py_compile(onchip_sdpa_promotion_gate.py, test_onchip_sdpa_promotion_gate_logic.py) passed
git diff --check passed
```

## Next

- keep normal promotion gate coverage green while causal mask work continues;
- implement a tiled/device-native causal mask or score-bias path; and
- rerun the causal fallback-readiness promotion-gate command until it passes.
