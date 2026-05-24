# Stage 003: Production SDPA Score Handoff Device Proof

Date: 2026-05-24

## Purpose

This stage records the first production-shaped stock SDPA score-matrix handoff
that runs value-correct on device from the compiler path.  The target is the
existing Inductor SDPA lowering: `QK^T` `batchmatmul` writes the score matrix,
then softmax `max` and `sub` consume it.

## Method

The realization remains default-off:

```sh
SPYRE_ONCHIP_HANDOFF_REALIZE=1
SPYRE_ONCHIP_ATTENTION_SCORE_HANDOFF=1
SPYRE_ONCHIP_HANDOFF_MIN_BYTES=1048576
```

The implementation bridges the full score fanout.  Once the `QK^T` producer
output is LX-only, both `max` and `sub` must be LX-fed.  Each consumer becomes a
mixed SuperDSC with a two-`STCDPOpLx` roundtrip.  This preserves the same-stick
contract and gives real L3 traffic without enabling the uncertified PT-LX
restickify path.

Two runtime gotchas were fixed or avoided:

- For same-stick/same-split sub-stick chunks, the data-op frame must be padded to
  the proven 2048-frame shape.  Tight 512-frame `PieceInfo` compiled but hung on
  device for the add control.
- Use the clean foundation DXP binary:
  `$DTI_PROJECT_ROOT/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone`.
  `$DTI_PROJECT_ROOT/build/deeptools/dxp/dxp_standalone` accepts mixed SDSCs but
  emitted a smaller broken `init.txt` and hung at the D2H barrier.

## Evidence

Logic tests:

```text
tests/_inductor/test_onchip_handoff_logic.py
tests/_inductor/test_onchip_realize_logic.py
tests/_inductor/test_onchip_streaming_logic.py

21 passed in 0.16s
```

Add control:

```text
CACHE=/tmp/onchip-add-cleandxp-cache-471573
ADD_ONCHIP_CLEANDXP_OK max_err 0.0048828125
mixed [('sdsc_2_add.json', ['STCDPOpLx'], 1, {'mb_': 2048, 'out_': 2048}, {'mb_': 2048, 'out_': 64})]
```

SDPA smoke:

```text
CACHE=/tmp/sdpa-onchip-cleandxp-smoke-cache-471754
SDPA_ONCHIP_CLEANDXP_SMOKE_OK max_err 0.0001220703125
mixed:
  sdsc_4_max.json opFuncsUsed_=['STCDPOpLx', 'STCDPOpLx'] datadscs=2
  sdsc_5_sub.json opFuncsUsed_=['STCDPOpLx', 'STCDPOpLx'] datadscs=2
```

Representative benchmark, shape `(B=1, H=8, L=1024, D=128)`, 5 warmup / 10
timed iterations:

```text
baseline HBM:
  cache=/tmp/sdpa-bench-hbm-1024x8-153940
  max_err=0.0000651479
  median_ms=2.6005 mean_ms=2.6029 min_ms=2.5881 max_ms=2.6189

on-chip score handoff:
  cache=/tmp/sdpa-bench-onchip-1024x8-153954
  max_err=0.0000651479
  median_ms=1.8484 mean_ms=1.8454 min_ms=1.8331 max_ms=1.8550
  mixed=[sdsc_4_max.json, sdsc_5_sub.json]
```

Descriptor evidence for the on-chip benchmark:

```text
sdsc_3_batchmatmul.json OUTPUT: hbmSize_=0, hbmStartAddress_=-1, lxSize_=2147483647
sdsc_4_max.json: opFuncsUsed_=['STCDPOpLx', 'STCDPOpLx'], datadscs_=2
sdsc_5_sub.json: opFuncsUsed_=['STCDPOpLx', 'STCDPOpLx'], datadscs_=2
```

DXP debug senprog evidence:

```text
code_dir=/tmp/sdpa-bench-onchip-1024x8-153954/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_2jz953wg

debug/sdsc_4_max/senprog.txt: HBM=0 L3_LDU=64 L3_STU=64
debug/sdsc_5_sub/senprog.txt: HBM=0 L3_LDU=64 L3_STU=64
```

Negative control:

```text
Moved loadprogram_to_device/.../init.txt, reran with runner redirect to the
on-chip code_dir, and got:

RuntimeError: Failed to open file: .../loadprogram_to_device/.../init.txt

The file was restored after the check.
```

## Conclusion

The compiler path now emits and runs a value-correct, HBM-free, same-stick
core-to-core score handoff for stock SDPA.  The first production completion gate
is met for the score-matrix edge.  The implementation remains scoped to Tier 1:
same-stick `STCDPOpLx` only, with layout-changing/PT-LX paths still fail-closed.
