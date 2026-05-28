# Stage 065: HBM-KV Gate Invariants

Date: 2026-05-27

## Purpose

Stage064 moved the HBM-KV-safe row into the promotion gate.  Stage065 tightens
what that gate proves.  The device row must now show the composite shape we want
for the AIU warp-specialized analogue:

```text
Q/input0 layout-transform sidecar is present
flash pointwise handoff sidecars are present
K/V-repack sidecars are absent
K/V input1 therefore remains on Foundation's HBM-backed batchmatmul path
```

This matters because a plain mixed-sidecar count can hide two different bugs:
the pointwise part of the composed path could silently disappear, or an
experimental K/V-repack probe could leak into the HBM-KV-safe variant.

## Gate Changes

`tools/onchip_sdpa_promotion_gate.py` now validates two additional invariants
for each row by default:

```text
require_pointwise_handoff = true
forbid_kv_repack = true
```

The pointwise check looks for a non-`mixed_flash_*` SDSC whose name is a flash
pointwise op such as `_add` or `_mul` and whose data movement includes
`STCDPOpLx`.

The K/V check rejects any mixed SDSC whose name, file path, or
`flashAttentionPipeline_.source` contains a K/V-repack marker.

Two escape hatches remain available for lower-level probes:

```text
--no-require-pointwise
--allow-kv-repack
```

The default promotion path keeps both invariants enabled.

## Real Row Evidence

The Stage064 H8/L256 device row was revalidated with the stricter gate:

```text
tools/onchip_sdpa_promotion_gate.py \
  --gate onchip_layout_xform \
  --cases b1h8d64_block64_hbmkv \
  --reuse-existing \
  --case-output-dir /tmp/sdpa-stage064-hbmkv-gate-json \
  --output-json /tmp/sdpa-stage065-hbmkv-invariants.json
```

Result:

```text
PROMOTION_GATE_PASSED gate=onchip_layout_xform cases=1 rows=1
```

The row contains nineteen mixed SDSCs:

```text
pointwise handoffs:
  4_mul
  11_mul
  17_add
  20_mul
  26_mul
  32_add
  35_mul
  41_mul
  47_add
  5_mul

layout-transform consumer:
  mixed_flash_layout_xform_pair_tile_2_consumer

flash pipeline proof sidecars:
  mixed_flash_pipeline_tile_0..6 in the second flash bundle
  mixed_flash_pipeline_tile_0 in the first flash bundle
```

No mixed SDSC name, file path, or flash-pipeline source contains a K/V-repack
marker.  This is the intended composite path: pointwise and query-side
layout-transform handoffs are on chip, while K/V input1 stays HBM-backed.

## Interpretation

This still does not finish the warp-specialized attention objective, but it
turns the current best AIU-shaped variant into a more honest production
candidate.  The promotion gate now fails if the implementation regresses toward
either of the two misleading states:

- "HBM-KV-safe" row accidentally executes K/V-repack sidecars.
- "Composite on-chip" row silently drops the pointwise handoffs and only keeps
  the layout-transform sidecar.

The next implementation step can now work on additional overlap shape, such as
runtime-safe row overlap for the layout-transform sidecar, with this baseline
guarding the K/V boundary.

## Verification

Local:

```text
python3 -m py_compile tools/onchip_sdpa_promotion_gate.py tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py
python3 tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py  # 11/11 pass
python3 tools/onchip_sdpa_promotion_gate.py --gate onchip_layout_xform --cases b1h8d64_block64_hbmkv --dry-run
```

Pod:

```text
python3 -m py_compile tools/onchip_sdpa_promotion_gate.py tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py
python3 tests/_inductor/test_onchip_sdpa_promotion_gate_logic.py  # 11/11 pass
python3 tools/onchip_sdpa_promotion_gate.py --gate onchip_layout_xform --cases b1h8d64_block64_hbmkv --reuse-existing --case-output-dir /tmp/sdpa-stage064-hbmkv-gate-json --output-json /tmp/sdpa-stage065-hbmkv-invariants.json
```
