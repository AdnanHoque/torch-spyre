# Stage 230-231: Producer-Mixed Row-Stripe PT-LX Handoff

## Summary

This stage moved the cross-bundle PT-LX restickify prototype through the next two production-shaped blockers:

1. DXP rejected a standalone data-op bridge because `datadscs_` were present without a DLDSC schedule.
2. After packaging the bridge into the producer SDSC, the 2048 case failed DCC with `Require larger IBUFF` because the bridge emitted 3072 per-tile data ops.

The current prototype now packages the bridge as a producer-plus-data-op mixed SDSC and coalesces the simple 2048 ownership pattern into row stripes.

## Changes

- Replaced the standalone cross-bundle bridge SDSC with a mixed producer SDSC:
  - The producer `dscs_` remain present.
  - The in-graph `ReStickifyOpHBM` SDSC is omitted.
  - The bridge `datadscs_` are attached to the producer SDSC.
  - `coreIdToDscSchedule` runs producer DL first, then bridge data ops.
- Kept the next bundle's consumer input patched to the LX endpoint.
- Added a row-stripe direct-output lowering for simple one-owner tiles:
  - Applies when every logical tile has exactly one producer owner and one destination owner.
  - Gathers a whole destination row stripe into the bridge core.
  - Runs `ReStickifyOpWithPTLx` once per stripe.
  - Writes directly to the consumer LX endpoint, avoiding the extra local scatter data op.

## Compile-Only Results

Command shape:

```sh
SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS=7 \
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1 \
SPYRE_RESTICKIFY_PTLX_CROSS_BUNDLE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1 \
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0 \
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=1048576 \
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1 \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 512 --size 1024 --size 2048 \
  --skip-correctness --skip-kernel-launch --copy-kernel-code \
  --output-dir /tmp/stage231-row-stripe-sweep \
  --fail-on-error
```

Result:

```text
ok    size=512   case=adds_then_matmul  restickifies=2 bytes=1048576  byte_hops=0
ok    size=1024  case=adds_then_matmul  restickifies=2 bytes=4194304  byte_hops=0
ok    size=2048  case=adds_then_matmul  restickifies=2 bytes=16777216 byte_hops=0
Completed 3 rows with 0 errors
```

Generated bridge shapes:

| Size | Bridge form | Data ops | Notes |
|---:|---|---:|---|
| 512 | per-tile gather/restickify/scatter | 192 | Fragmented ownership, no coalescing |
| 1024 | per-tile gather/restickify/scatter | 768 | Fragmented ownership, no coalescing |
| 2048 | row-stripe gather/restickify direct-output | 64 | 1024 logical tiles coalesced into 32 stripes |

The 2048 mixed SDSC now contains:

```text
sdsc_3_CrossBundleProducerStreamingReStickifyOpWithPTLx.json
  dscs_: 1
  datadscs_: 64
  STCDPOpLx: 32
  ReStickifyOpWithPTLx: 32
  streamingPTLXFull_.coalescing: row-stripe-direct-output
```

The consumer bundle contains:

```text
sdsc_0_batchmatmul.json
  Tensor0 allocation component: lx
```

The remaining graph-input restickify is intentionally unchanged:

```text
sdsc_0_ReStickifyOpHBM.json
```

## Validation

Pod tests:

```sh
python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_tile_ownership_probe.py \
  -q
```

Result:

```text
35 passed
```

Mapping-alignment regression:

```sh
python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q
```

Result:

```text
26 passed
```

## Interpretation

This does not prove hardware runtime correctness yet, because these runs used `--skip-kernel-launch`. It does prove the compiler can now generate a DXP-accepted, no-HBM, cross-bundle producer-to-consumer PT-LX value-flow artifact for the high-signal 2048 case.

The old blockers moved as follows:

| Old blocker | Status |
|---|---|
| Standalone data-op bridge rejected by DXP | Fixed by producer-mixed SDSC packaging |
| 2048 value-flow assertion missing local tiles | Fixed by materializing all logical tiles |
| 2048 DCC IBUFF overflow from 3072 data ops | Fixed by row-stripe coalescing to 64 data ops |

## Next Step

Run careful hardware validation:

1. Good-known tiny stock Torch-Spyre smoke.
2. `adds_then_matmul` with PT-LX enabled at 512.
3. If 512 is value-correct and device health is stable, try 1024.
4. Then try the row-stripe 2048 case.

Acceptance for the next stage is value correctness without `ReStickifyOpHBM` for the in-graph edge and without a stream hardware error.
