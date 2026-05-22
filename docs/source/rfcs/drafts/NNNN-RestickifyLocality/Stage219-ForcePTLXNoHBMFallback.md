# Stage 219: Forced PT-LX Without HBM Fallback

## Summary

Tested the question: can we simply force PT-LX, remove the HBM restickify path,
and time the result?

Answer: only for sizes that the current full-bridge PT-LX contract can actually
represent. In this square benchmark family, forced codegen can emit a no-HBM
mixed PT-LX artifact for `2048`, and it can superficially emit one for `3072`,
but `3072` fails during Deeptools compilation. Smaller sizes and larger sizes
still cannot use the current full bridge.

Artifacts:

```text
artifacts/stage219_force_ptlx/codegen
artifacts/stage219_force_ptlx/timing
```

## Forced Codegen Results

The force run used:

```text
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1
```

| Size | Forced PT-LX Codegen | Reason |
|---:|---|---|
| 512 | no | `ptlx-piece-smaller-than-stick:producer-input:mb:split=32:max=8` |
| 1024 | no | `ptlx-piece-smaller-than-stick:producer-input:mb:split=32:max=16` |
| 1536 | no | `ptlx-piece-smaller-than-stick:producer-input:mb:split=32:max=24` |
| 2048 | yes | emits `1_MixedReStickifyOpWithPTLxConsumer` |
| 3072 | codegen yes, Deeptools compile no | `Different cardinality between json and caller` |
| 4096 | no | endpoint ranges overlap / no full-bridge LX room with chosen bases |
| 8192 | no | endpoint ranges overlap / full bridge does not fit |
| 16384 | no | endpoint ranges overlap / full bridge does not fit |

For `512/1024/1536`, the producer split creates pieces smaller than a
64-element stick, so the current single full-tensor `ReStickifyOpWithPTLx` bridge
cannot legally express the movement. These are exactly the shapes that need the
Stage216 streaming tiled gather path.

For `4096+`, the full bridge wants producer endpoint, consumer endpoint, and
intermediate storage in per-core LX at the same time. The 2 MB/core LX budget is
too tight for the full-tensor bridge; these are the shapes that need tiled
workspace reuse.

## Forced Timing

Only `2048` produced a runnable forced PT-LX timing:

| Size | Forced PT-LX median | p10 | p90 | Result |
|---:|---:|---:|---:|---|
| 2048 | 1.017 ms | 1.009 ms | 1.027 ms | ok |
| 3072 | n/a | n/a | n/a | Deeptools compile failure |

The `2048` forced PT-LX timing matches the normal PT-LX result from Stage217
(`~1.014 ms`). That is expected: normal PT-LX was already patching `2048`; force
mode only bypassed allocator-backed endpoint requirements.

## 2048 No-HBM Evidence

For `2048`, codegen produced:

```text
sdsc_1_MixedReStickifyOpWithPTLxConsumer.json
```

The restickify boundary contains:

```text
ReStickifyOpWithPTLx
STCDPOpLx
```

and no `sdsc_1_ReStickifyOpHBM.json` for that boundary. The audit row reported:

- `status=patched`
- `replacement_sdsc=1_MixedReStickifyOpWithPTLxConsumer`
- producer endpoint: LX `0..262144`
- consumer endpoint: LX `262144..524288`
- intermediate: LX `524288..786432`
- `value_flow_contract.valid=true`

This is codegen/runtime evidence that `2048` is using the PT-LX path. It is not
yet hardware-counter proof of RIU traffic; that still needs fabric counters.

## Conclusion

We should not delete the HBM path globally. Today it is still the required
fallback for unsupported restickifies. The production-safe direction is:

1. keep the stock HBM restickify path as fallback,
2. use PT-LX when the compiler can prove and emit a valid LX contract,
3. add streaming tiled PT-LX lowering so the unsupported sizes become expressible
   without requiring full-tensor LX residency.
