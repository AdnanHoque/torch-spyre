# Stage 064: HBM-KV Promotion Gate

Date: 2026-05-27

## Purpose

Stage063 proved the production-aligned HBM-KV-safe layout-transform lane on the
neutral K/V shape that exposed the raw LX input1 boundary:

```text
B=1, H=8, L=256, D=64, block=64
```

Stage064 moves that row into the repeatable promotion machinery.  The goal is
to make the gate protect the implementation direction that is now credible for
AIU warp-specialized attention: keep K/V under Foundation's HBM-backed input1
contract while promoting the safe query-side layout-transform work into mixed
sidecars.

## Gate Changes

The promotion gate default variant is now:

```text
onchip_hbm_kv_layout_xform
```

That variant is intentionally equivalent to the passing master layout-transform
path except that it names the invariant and explicitly disables K/V-repack
probe gates.

The `onchip_layout_xform` gate now includes a dedicated HBM-KV coverage case:

```text
name = b1h8d64_block64_hbmkv
batch = 1
heads = 8
dim = 64
block_size = 64
lengths = (256,)
min_mixed_by_length = {256: 19}
layout_xform_lengths = (256,)
```

The full gate grows from twenty rows to twenty-one rows.  Existing H2/H4,
causal, block128, D128, and long-shape rows remain unchanged.

## Device Result

Targeted gate run:

```text
tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --cases b1h8d64_block64_hbmkv \
  --python /home/adnan-cdx/dt-inductor-mixed/.venv/bin/python \
  --timeout-s 300 \
  --cache-prefix /tmp/sdpa-stage064-hbmkv-gate \
  --case-output-dir /tmp/sdpa-stage064-hbmkv-gate-json \
  --output-json /tmp/sdpa-stage064-hbmkv-gate.json
```

Result:

```text
L=256 onchip_hbm_kv_layout_xform status=ok
median = 0.553002 ms
max_abs_error = 0.00439453
mixed_sdscs = 19
cache = /tmp/sdpa-stage064-hbmkv-gate-b1h8d64_block64_hbmkv-onchip_hbm_kv_layout_xform-B1-H8-L256-D64-C0-680695-415100

PROMOTION_GATE_PASSED gate=onchip_layout_xform cases=1 rows=1
```

This is stronger than the Stage063 one-off sweep because the result is now
validated by the same gate logic used for promotion: shape, block size, causal
flag, max error, minimum mixed sidecar count, and required layout-transform
consumer sidecar.

## Interpretation

The gate now has a small but explicit guardrail against regressing back into a
misleading "K/V is on-chip" claim.  The protected path is:

```text
Q/input0 layout-transform sidecar: on-chip mixed path
K/V input1: Foundation HBM-backed batchmatmul path
```

This is not the final warp-specialized attention variant yet.  It is the
repeatable baseline for building one: future stages can layer score/pointwise
handoffs or other AIU-appropriate overlap around this row while the gate keeps
the K/V boundary honest.

## Verification

Local:

```text
python3 -m py_compile tools/onchip_sdpa_promotion_gate.py tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py tools/onchip_sdpa_sweep.py tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py  # 9/9 pass
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py           # 55/55 pass
```

Pod:

```text
python3 -m py_compile tools/onchip_sdpa_promotion_gate.py tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py
python3 tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py  # 9/9 pass
```
