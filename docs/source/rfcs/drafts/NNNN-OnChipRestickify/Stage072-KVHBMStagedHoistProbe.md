# Stage 072: K/V HBM-Staged Hoist Probe

Date: 2026-05-27

## Purpose

Stage071 gave a correct combined baseline for query-side layout transform,
HBM-staged K/V input1 consume, and flash pointwise handoff.  Stage072 starts
pulling that serial baseline toward warp-specialized scheduling by decoupling a
future low-core K/V producer from its later 32-core attention consumer.

## Implementation

Added a default-off gate:

```text
SPYRE_FLASH_ATTENTION_KV_REPACK_HBM_STAGED_HOIST_TILE
```

`-1` disables it.  `-2` scans for the first safe hoist candidate.

The passing implementation is deliberately conservative:

- clone the future low-core `ReStickifyOpHBM` producer;
- insert that clone before the current attention tile;
- omit the original later producer from the bundle;
- keep the later K/V consumer in the Stage069 HBM-staged form.

This proves bundle-level producer hoisting without asking DXP to run an 8-core
producer compute DSC and a 32-core attention compute DSC inside the same mixed
SuperDSC root.

## Failed Shape

The first attempt embedded both compute DSCs in one mixed sidecar:

```text
dsc0 = 8-core ReStickifyOpHBM
dsc1 = 32-core batchmatmul
```

DXP rejected that shape:

```text
DtException: Different cardinality between json and caller
```

That suggests the next true-overlap step needs either cardinality-normalized
fold metadata or a dataop-style producer representation, not a raw mixed
compute-DSC list.

## Device Result

Run:

```text
tools/onchip_sdpa_sweep.py \
  --variants onchip_hbm_kv_layout_xform_kv_hbm_staged_hoist_probe \
  --batch 1 --heads 8 --lengths 256 --dim 64 --block-size 64 \
  --warmup 1 --iters 3 --timeout-s 300 --dxp-debug \
  --cache-prefix /tmp/sdpa-stage072-kv-hbm-staged-hoist-v2 \
  --output-json /tmp/sdpa-stage072-kv-hbm-staged-hoist-v2.json
```

Result:

```text
status = ok
median = 0.553183 ms
max_abs_error = 0.00439453
mixed_sdscs = 20
```

## Bundle Evidence

The main flash graph executes the hoisted future producer before tile 0:

```text
sdsc_mixed_flash_kv_repack_hbm_staged_hoist_0_future_producer.json
sdsc_0_batchmatmul.json
...
sdsc_mixed_flash_kv_repack_hbm_staged_hoist_0_future_kv_3_input1_consumer.json
```

The original future producer is omitted from `bundle.mlir`:

```text
sdsc_18_ReStickifyOpHBM.json  # not executed
```

Hoist metadata:

```text
role = future_producer
numCoresUsed = 8
inserted_before = 0_batchmatmul
future_tile = 3
future_input_idx = 1
omitted_future_predecessor = 18_ReStickifyOpHBM
```

The later K/V consumer remains HBM-staged:

```text
role = consumer
numCoresUsed = 32
kv_repack_hbm_source = true
kv_repack_hbm_staged = true
kv_repack_source_sdsc = 18_ReStickifyOpHBM
replaces_sdsc = 19_batchmatmul
```

## Interpretation

This is not a speedup yet; median runtime is essentially tied with Stage071.
The value is architectural: a future K/V producer can be legally moved ahead of
the current tile and paired with a later HBM-staged consumer while preserving
attention correctness.

The next overlap target is to make that hoisted producer run concurrently with
current-tile work rather than merely earlier in the bundle.

## Verification

Local:

```text
python3 -m py_compile \
  torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/onchip_realize.py \
  tools/onchip_sdpa_sweep.py \
  tests/_inductor/test_config_logic.py \
  tests/_inductor/test_onchip_realize_logic.py \
  tests/_inductor/test_onchip_sdpa_sweep_logic.py

manual no-arg test harness:
  ran 217 tests
```

Pod:

```text
/home/adnan-cdx/dt-inductor-mixed/.venv/bin/python -m py_compile \
  torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/onchip_realize.py \
  tools/onchip_sdpa_sweep.py
```
