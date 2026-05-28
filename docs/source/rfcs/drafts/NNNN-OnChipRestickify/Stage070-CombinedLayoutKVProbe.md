# Stage 070: Combined Layout + HBM-Staged K/V Probe

Date: 2026-05-27

## Purpose

Stage063 proved the HBM-KV layout-transform path.  Stage069 proved the
HBM-staged K/V consumer path.  The next question was whether those two
individually-correct sidecars can coexist in one attention bundle.

Stage070 lets non-conflicting mixed sidecar replacements compose instead of
making K/V pair selection suppress the serial layout-transform pair.

## Implementation

Bundle generation now still builds the serial layout-transform pair when a K/V
pair was requested, then adds it only if its replacements and omissions do not
conflict with earlier sidecars.  Pointwise handoff placement now uses the
highest active `pointwise_lx_region0` among active mixed sidecars.

New sweep probe:

```text
onchip_hbm_kv_layout_xform_kv_hbm_staged_probe
```

The probe enables:

```text
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1
SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM=1
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_TILE=-2
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_IFN_TRANSFER=0
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PAIR_HBM_STAGED=1
```

## Device Result

Run:

```text
tools/onchip_sdpa_sweep.py \
  --variants onchip_hbm_kv_layout_xform_kv_hbm_staged_probe \
  --batch 1 --heads 8 --lengths 256 --dim 64 --block-size 64 \
  --warmup 1 --iters 3 --timeout-s 300 --dxp-debug \
  --cache-prefix /tmp/sdpa-stage070-combined-hbmkv \
  --output-json /tmp/sdpa-stage070-combined-hbmkv.json
```

The bundle emitted both intended sidecars:

```text
sdsc_mixed_flash_layout_xform_pair_tile_2_predecessor
sdsc_mixed_flash_layout_xform_pair_tile_2_consumer
sdsc_mixed_flash_kv_repack_broadcast_pair_3_input1_consumer
```

But the combined execution failed value checking:

```text
status = failed
mismatches = 7366 / 131072
max_abs_error = 0.6376953125
```

Control run after the same bundle change:

```text
variant = onchip_hbm_kv_layout_xform
status = ok
median = 0.573635 ms
max_abs_error = 0.00439453
mixed_sdscs = 19
```

## Interpretation

This narrows the next stack problem.  The failure is not that either sidecar is
individually illegal:

- the layout-transform HBM-KV path still passes;
- the HBM-staged K/V sidecar still has the correct DXP HBM-to-LX staging
  contract from Stage069;
- the combined bundle emits non-conflicting replacements by SDSC name.

The failure is therefore a cross-sidecar composability issue.  The next probe
should compare value flow around the tile2 layout-transform output and tile3
K/V-staged consumer, then decide whether the fix belongs in Torch-side sidecar
ordering/dependency metadata or in a lower-stack transfer/allocation contract.

## Verification

Local:

```text
python3 -m py_compile \
  torch_spyre/_inductor/codegen/bundle.py \
  tools/onchip_sdpa_sweep.py \
  tests/_inductor/test_onchip_realize_logic.py \
  tests/_inductor/test_onchip_sdpa_sweep_logic.py

manual no-arg test harness:
  ran 212 tests
```

Pod:

```text
/home/adnan-cdx/dt-inductor-mixed/.venv/bin/python -m py_compile \
  torch_spyre/_inductor/codegen/bundle.py \
  tools/onchip_sdpa_sweep.py
```
