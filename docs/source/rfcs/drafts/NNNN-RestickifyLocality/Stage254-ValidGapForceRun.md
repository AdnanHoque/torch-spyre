# Stage 254: Forced Valid-Gap PT-LX Run

## Summary

Stage 253 showed that the valid-gap consumer-shaped PT-LX descriptor satisfies
compiler-side contracts but is not semantically certified. This stage added a
validation-only force gate so we could run that candidate as the real
producer/restickify/consumer bridge:

```sh
SPYRE_RESTICKIFY_PTLX_FORCE_VALIDGAP_CONSUMER_TILE_E2E=1
```

This flag is intentionally not a production eligibility rule. It only answers
whether the valid-gap descriptor is value-correct on hardware.

## Result

The forced `adds_then_matmul` size `512` run compiled and launched, but failed
correctness:

```text
Mismatched elements: 204455 / 262144 (78.0%)
Greatest absolute difference: 1.8662109375 at index (15, 289)
```

The audit proves that the candidate really replaced the target HBM restickify:

| Field | Value |
|---|---:|
| Replacement SDSC | `3_CrossBundleProducerStreamingReStickifyOpWithPTLx` |
| Coalescing | `validgap-consumer-64x64-tiles` |
| Tiles | 64 |
| Data ops | 128 |
| Candidate HBM placements | 0 |
| Gather count | 64 |
| Valid-gap PT-LX tile count | 64 |
| Endpoint contract | pass |
| Consumer descriptor contract | pass |
| Live-element preservation | pass |
| Forced semantic gate | pass |
| Runtime values | fail |

Artifacts:

- `artifacts/stage254_validgap_force_512/`

## Interpretation

This closes one tempting path: using `validGap_` on a synthetic source stick
dimension is not enough to make Deeptools perform the desired logical
`mb/out -> mb/in` value transform.

The failure is still useful:

- The mixed bridge can compile and launch without a stream hardware error.
- The producer and consumer LX endpoint plumbing is now strong enough to reach
  a value comparison.
- The remaining problem is the value transform inside the PT-LX tile, not the
  high-level bundle splice.

## Next Direction

The next path should stop relying on a synthetic valid-gap axis as a semantic
alias. We need one of:

1. a Deeptools-native PT-LX descriptor whose input/output coordinates encode the
   real value mapping directly;
2. a small explicit data-op bridge that materializes the consumer view from the
   producer tile without inventing a live-but-sparse stick axis;
3. a lower-level producer output planning change that makes the producer emit
   the consumer-readable stick layout directly, avoiding this bridge for the
   eligible edge.

For the current objective, option 1 is still the most production-shaped if we
can find the exact descriptor contract. Option 2 is the most controllable
prototype path if option 1 keeps producing wrong values.
