# Stage 071: Rebuild K/V Sidecar After Pointwise Handoff

Date: 2026-05-27

## Purpose

Stage070 proved that the serial layout-transform sidecar and HBM-staged K/V
sidecar could be emitted in the same bundle, but the on-chip variant failed
values when pointwise handoff was also enabled.

A control run with the same block-64 graph and both sidecars, but without the
on-chip pointwise umbrella, passed.  That localized the failure to stale sidecar
construction: the K/V consumer sidecar was built before
`realize_flash_attention_pointwise_handoffs(...)` mutated the source SDSCs.

## Implementation

Bundle generation now rebuilds an active K/V repack broadcast pair after flash
pointwise handoff realization succeeds.

This keeps the Stage069 HBM-staged input1 contract, while letting the K/V
consumer inherit any post-pointwise descriptor/schedule changes for its other
inputs.

## Device Result

Run:

```text
tools/onchip_sdpa_sweep.py \
  --variants onchip_hbm_kv_layout_xform_kv_hbm_staged_probe \
  --batch 1 --heads 8 --lengths 256 --dim 64 --block-size 64 \
  --warmup 1 --iters 3 --timeout-s 300 --dxp-debug \
  --cache-prefix /tmp/sdpa-stage071-rebuild-kv-after-pointwise \
  --output-json /tmp/sdpa-stage071-rebuild-kv-after-pointwise.json
```

Result:

```text
status = ok
median = 0.552882 ms
max_abs_error = 0.00439453
mixed_sdscs = 20
```

The successful bundle contains both sidecars:

```text
mixed_flash_layout_xform_pair_tile_2_consumer -> replaces 15_batchmatmul
mixed_flash_kv_repack_broadcast_pair_3_input1_consumer -> replaces 19_batchmatmul
```

## Control

The Stage070 combined on-chip run before this rebuild failed:

```text
status = failed
mismatches = 7366 / 131072
max_abs_error = 0.6376953125
```

The same combined sidecars without the on-chip pointwise umbrella passed:

```text
variant = combined_layout_kv_hbm_staged_no_pointwise_manual_b64
status = ok
median = 0.611004 ms
max_abs_error = 0.003662109375
```

## Descriptor Evidence

The failing pre-rebuild K/V consumer put both input staging allocations under
the same stale input loop:

```text
allocate_lds0_lx prev = loop_ds0_ds1_in
allocate_lds1_lx prev = loop_ds0_ds1_in
```

After rebuilding the K/V sidecar from the post-pointwise SDSCs, DXP attaches the
allocations to the updated loop dimensions:

```text
allocate_lds0_lx prev = loop_ds0_ds1_mb
allocate_lds1_lx prev = loop_ds0_ds1_x
```

The input1 K/V contract remains HBM-staged:

```text
Tensor1 memOrg = hbm + lx + ptxrf
transfer_lds1_src:hbm_dst:lx
```

## Interpretation

This is the first passing device result with all three pieces active in one
attention bundle:

- query-side layout-transform pair;
- HBM-staged K/V consumer pair;
- flash pointwise handoff.

It is still a serial/probe composition rather than true warp-specialized overlap,
but it removes the stale-descriptor failure and gives the next overlap work a
correct combined baseline.

## Verification

Local:

```text
python3 -m py_compile \
  torch_spyre/_inductor/codegen/bundle.py \
  tests/_inductor/test_onchip_realize_logic.py

manual no-arg test harness:
  ran 213 tests
```

Pod:

```text
/home/adnan-cdx/dt-inductor-mixed/.venv/bin/python -m py_compile \
  torch_spyre/_inductor/codegen/bundle.py \
  tools/onchip_sdpa_sweep.py
```
