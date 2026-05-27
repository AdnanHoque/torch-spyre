# Stage 057: K/V Repack Broadcast Plan Artifact

Date: 2026-05-27

## Purpose

Stage056 named the real blocker for the block64 prefill graph: the future
K/V operand is produced by a low-core `ReStickifyOpHBM`, but the future
`batchmatmul` consumes it from a 32-core schedule.  The consumer is split over
query rows (`mb_`), while the K/V operand layout does not contain `mb_`.

Stage057 adds the next descriptor-only probe for that boundary.  It does not
execute or replace any SDSC.  It writes a plan artifact that shows the physical
shape a future executable primitive must support:

```text
low-core ReStickifyOpHBM output pieces
  -> broadcast/repack into every future batchmatmul consumer core
```

## Change

A new default-off gate was added:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PLAN_ARTIFACT=1
```

Bundle generation scans generated flash attention tiles for non-input0 K/V
operands and calls:

```text
build_flash_attention_kv_repack_broadcast_plan_artifact(...)
```

When the Stage056 boundary is present, the bundle directory now gets a sidecar
diagnostic file such as:

```text
sdsc_flash_kv_repack_broadcast_plan_1_input1.json
```

The file is intentionally not added to `bundle.mlir`, `sidecar_replacements`,
`sidecar_omissions`, or `bundle_attrs_by_file`.  This keeps the artifact
non-executed while still making it visible to the sweep cache summarizer.

The sweep harness exposes:

```text
layout_xform_hoist_kv_repack_plan_auto
```

That variant enables the mixed pipeline, the layout-transform hoist scanner, and
the K/V repack plan emission gate.

## Descriptor Shape

For the synthetic real-like `input1` boundary, the plan records:

- source SDSC: `1_ReStickifyOpHBM`;
- consumer SDSC: `2_batchmatmul`;
- producer cores: 2;
- consumer cores: 32;
- producer split `mb_` mapped to operand `x_`;
- consumer split `mb_` absent from the K/V operand;
- source PieceInfo count: 2;
- destination PieceInfo count: 64.

The destination pieces duplicate each source logical slice once per consumer
core and include explicit broadcast metadata:

```text
broadcastSourcePieceKey_
broadcastConsumerCore_
```

LX source and destination bases are allocated from the same non-overlapping
region calculation and recorded in metadata:

```text
kv_repack_source_lx_base
kv_repack_consumer_lx_base
slice_bytes
```

The artifact also records `kvRepackBroadcastPlan_.runtime_status =
not_executed` and blockers for the unproven executable shape.  The duplicated
PieceInfo broadcast model is a plan contract, not a claim that DXP already
accepts this as an executable SDSC.

## Local Validation

```text
python3 -m py_compile torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/onchip_realize.py \
  tools/onchip_sdpa_sweep.py \
  tests/_inductor/test_config_logic.py \
  tests/_inductor/test_onchip_realize_logic.py \
  tests/_inductor/test_onchip_sdpa_sweep_logic.py \
  tests/_inductor/test_onchip_flash_pipeline_logic.py
python3 tests/_inductor/test_config_logic.py
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_onchip_realize_logic.py
python3 tests/_inductor/test_onchip_flash_pipeline_logic.py
git diff --check
```

Results:

```text
test_config_logic.py: 13/13 passed
test_onchip_sdpa_sweep_logic.py: 16/16 passed
test_onchip_realize_logic.py: 62/62 passed
test_onchip_flash_pipeline_logic.py: 11/11 passed
git diff --check: clean
```

Pod validation in:

```text
/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
```

also passed:

```text
test_config_logic.py: 13/13 passed
test_onchip_sdpa_sweep_logic.py: 16/16 passed
test_onchip_realize_logic.py: 62/62 passed
test_onchip_flash_pipeline_logic.py: 11/11 passed
```

## Device Probe

A direct source-tree block64 L128 probe used explicit `torch_spyre._autoload()`
and:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE=-2
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_HOIST_TILE=-2
SPYRE_FLASH_ATTENTION_KV_REPACK_BROADCAST_PLAN_ARTIFACT=1
SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE=64
```

Cache:

```text
/tmp/sdpa-stage057-kv-repack-plan-direct-layout_xform_hoist_kv_repack_plan_auto-B1-H2-L128-D64-C0
```

The run compiled and executed, but values remain wrong on the underlying raw
path:

```text
max_abs_error=17280.0
```

The cache emitted:

```text
inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_1_yliu2go_/sdsc_flash_kv_repack_broadcast_plan_1_input1.json
```

The plan matches the real Stage056 boundary:

```text
kv_repack_source_sdsc=3_ReStickifyOpHBM
kv_repack_consumer_sdsc=4_batchmatmul
kv_repack_input_idx=1
kv_repack_producer_cores=2
kv_repack_consumer_cores=32
kv_repack_producer_split=mb_
kv_repack_mapped_split=x_
kv_repack_consumer_split=mb_
slice_bytes=262144
kv_repack_source_lx_base=16384
kv_repack_consumer_lx_base=278528
kv_repack_source_piece_count=2
kv_repack_destination_piece_count=64
kv_repack_broadcast_executable=False
```

The corresponding `bundle.mlir` does not reference:

```text
sdsc_flash_kv_repack_broadcast_plan_1_input1.json
```

## Current Status

This stage still does not complete the warp-specialized prefill attention path.
It advances the path from a named blocker to an emitted, test-covered physical
plan artifact.  Executing the repack remains a later promotion step after the
descriptor contract is accepted and device value correctness is proven.
